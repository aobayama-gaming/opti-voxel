#pragma once

#include "godot_cpp/classes/rendering_device.hpp"
#include "godot_cpp/variant/dictionary.hpp"
#include "godot_cpp/variant/packed_byte_array.hpp"
#include "godot_cpp/variant/rid.hpp"
#include "godot_cpp/variant/string.hpp"
#include "godot_cpp/variant/vector3.hpp"

#include "chunk_math.hpp"
#include "exact_noise.hpp"
#include "material_library.hpp"
#include "page_format.hpp"

#include <godot_cpp/classes/worker_thread_pool.hpp>

#include <array>
#include <functional>
#include <vector>

namespace godot {
class RenderingDevice;
}

using namespace godot;

namespace Cascade {
struct Spec;
}

// The instance table of an SDF (archetype instances): the fixed-size DefRecord array and the per-batch
// list of instance indices. The SDF hands out the data; each mesher owns the buffers on its own
// RenderingDevice (the legacy mesher runs on a local one), splices glsl_buffers() at the given bindings
// and creates/frees two storage buffers. glsl_set_chunk() is spliced into main() to fill the ChunkInput
// global g_sdf_chunk. The generated arch_fold (archetype_table.hpp) reads sdf_records / sdf_list.
class SDFGpuTable {
public:
    virtual ~SDFGpuTable() = default;

    // The DefRecord struct and the two buffer declarations at the given bindings.
    virtual String glsl_buffers(int p_record_binding, int p_list_binding) const = 0;
    // The statement assigning g_sdf_chunk from the current chunk (the mesher's main()).
    virtual String glsl_set_chunk() const = 0;

    // The snapshot version: an edit bumps it (the chunks it re-meshes require it).
    virtual uint32_t version() const = 0;
    virtual uint32_t instance_count() const = 0;
    // How many records the GPU buffer is sized for: the table's (runtime edits only append; more is refused), then the
    // streamed records' slots.
    virtual uint32_t instance_capacity() const { return instance_count(); }
    // What a mesher's records buffer holds: the snapshot version it was sent and the streamed placements
    // (InstanceStream::placements) it has seen.
    struct UploadCursor {
        uint32_t snap_version = UINT32_MAX; // UINT32_MAX: nothing sent yet
        uint64_t slot_seq = 0;
        bool operator==(const UploadCursor &p_o) const { return snap_version == p_o.snap_version && slot_seq == p_o.slot_seq; }
        bool operator!=(const UploadCursor &p_o) const { return !(*this == p_o); }
    };
    // The cursor of the records now: a mesher whose own differs calls record_bytes().
    virtual UploadCursor upload_cursor() const { return UploadCursor{ version(), 0 }; }
    // The encoded DefRecords changed since p_have, as spans for upload_records(): uint32 span count, then per span its
    // byte offset in the buffer, its byte size and the bytes; r_now is the cursor they bring it to (one snapshot,
    // whatever edits land meanwhile). An edit re-sends the pages it changed, a placement its key's slots.
    virtual PackedByteArray record_bytes(const UploadCursor &p_have, UploadCursor &r_now) const = 0;
    // Writes record_bytes()' spans into the records buffer (on the device that owns it).
    static void upload_records(RenderingDevice *p_rd, const RID &p_buffer, const PackedByteArray &p_spans) {
        int64_t at = 4;
        for (uint32_t s = 0; s < (p_spans.size() >= 4 ? p_spans.decode_u32(0) : 0u); s++) {
            const uint32_t offset = p_spans.decode_u32(at);
            const uint32_t size = p_spans.decode_u32(at + 4);
            p_rd->buffer_update(p_buffer, offset, size, p_spans.slice(at + 8, at + 8 + size));
            at += 8 + size;
        }
    }
    // Bytes per record in the GPU buffer.
    virtual uint32_t record_size() const = 0;

    // Main thread: patch each ChunkInput of the batch (id.w = this chunk's list offset in uint32 words,
    // anchor_hi.w = its instance count) and append the list words (record id | palette index << LIST_ID_BITS) to
    // r_list. r_chunk_ids already holds the batch's ChunkInput array (GPUFrame::encode_chunk_input). The lists never
    // exceed p_capacity_words (the mesher's list buffer): returns how many leading chunks went in whole, and the
    // caller dispatches only those and queues the rest again (a LOD-6 forest chunk lists ~9k records). Only a first
    // chunk over the whole buffer is cut (ERR, counted: a wrong surface there). r_palettes (if given) gets each kept
    // chunk's palette (chunk_palette.hpp): the global materials its corners' palette indices name. r_streamed (if given)
    // gets each kept chunk's words of streamed records (the rest are the scene's and the edits').
    virtual uint32_t build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list,
            uint32_t p_capacity_words, std::vector<std::vector<uint32_t>> *r_palettes = nullptr, std::vector<uint32_t> *r_streamed = nullptr) const = 0;
    // The entries every chunk palette starts with (ChunkPalette::build): {GROUND}, or with a climate GROUND then each
    // biome's material, a ground sample's palette index being 1 + its biome (SDFCascade::ground_material). Longer than
    // one: every chunk's samples carry a material (the cascade writes it, both meshers read it), and the palette pool's
    // ground block holds these entries.
    virtual const std::vector<uint32_t> &base_palette() const = 0;

    // The average list a batch's chunks may carry: the meshers size their list buffer for batch size * this.
    static constexpr uint32_t MAX_CHUNK_LIST = 4096;
    // A list word: the record id in its low LIST_ID_BITS, the record's chunk palette index (chunk_palette.hpp) above.
    // So a table holds at most 2^LIST_ID_BITS (8M) instances, removed ones included.
    static constexpr uint32_t LIST_ID_BITS = 32 - PageFormat::PALETTE_BITS;
    static constexpr uint32_t LIST_ID_MASK = (1u << LIST_ID_BITS) - 1;
    static uint32_t list_word(uint32_t p_record, uint32_t p_palette_index) { return p_record | p_palette_index << LIST_ID_BITS; }
    uint32_t max_chunk_list() const { return MAX<uint32_t>(MIN(instance_capacity(), MAX_CHUNK_LIST), 1u); }
    // List statistics (main thread): lengths per LOD, build time, truncations.
    virtual Dictionary get_stats() const { return Dictionary(); }
};

class SDFBase {
public:
    virtual ~SDFBase() = default;

    // Called once before any thread uses the SDF (VoxelTerrain::_ready). Chunks exist at LODs
    // [0, p_lod_count).
    virtual void configure(const ChunkParam &p_param, int p_lod_count) {
        (void)p_lod_count;
        param = p_param;
    }

    // CPU evaluation at an absolute position (integer frame), for a chunk of LOD p_lod: signed
    // distance in metres, negative inside. The LOD only sets the unit of the internal gradients (per
    // voxel of that LOD), the distance is the same at every LOD.
    virtual float evaluate(const WorldPos &p_pos, int p_lod) const = 0;
    // evaluate for the camera's per-frame queries (clip planes, the player's altitude and dig ray): main thread, never
    // computes streamed instances there (those not resident yet are left out until they land).
    virtual float evaluate_view(const WorldPos &p_pos, int p_lod) const { return evaluate(p_pos, p_lod); }

    // The value the discoverer's cull reads at a corner, leaving out what cull_exempt() covers anyway (SDFCascade:
    // the bare cascade, the GPU discoverer's point items bit for bit).
    virtual float evaluate_cull(const WorldPos &p_pos, int p_lod) const = 0;

    // evaluate_cull's field at a LOD0 voxel as LOD p_lod sees it, with its gradient, both in LOD0 voxels (the cell sites,
    // Archetype::FieldFn: LOD 0 is the field, a coarser LOD its nearest lattice sample extended along its gradient).
    // Thread-safe.
    virtual ExactNoise::Float4 ground_field(const VoxelPos &p_voxel, int p_lod) const = 0;
    // ground_field's coarse reading in steps (the GPU's field points read the middle one, SDFCascade::ground_unit): the
    // lattice point a LOD-p_lod reading is taken at (the nearest point of Λ_lod, ties up), then that reading p_unit (unit
    // gradient, distance in voxels of the LOD) extended along its gradient to p_voxel, the distance in LOD0 voxels.
    static VoxelPos ground_lattice(const VoxelPos &p_voxel, int p_lod) {
        const int64_t half = p_lod > 0 ? int64_t(1) << (p_lod - 1) : 0;
        return VoxelPos((p_voxel.x + half) >> p_lod << p_lod, (p_voxel.y + half) >> p_lod << p_lod, (p_voxel.z + half) >> p_lod << p_lod);
    }
    static ExactNoise::Float4 extend_field(ExactNoise::Float4 p_unit, const VoxelPos &p_at, const VoxelPos &p_voxel, int p_lod) {
        const VoxelPos d = p_voxel - p_at;
        p_unit.w = p_unit.w * ExactNoise::exact_pow2(p_lod) + p_unit.x * float(d.x) + p_unit.y * float(d.y) + p_unit.z * float(d.z);
        return p_unit;
    }

    // The unit direction upright things (trees) grow along at a LOD0 voxel: the one source of "up" for placement.
    // +y on the flat world; a planet overrides it.
    virtual UpAxis up_at(const VoxelPos & /*p_voxel*/) const { return UP_Y; }

    // evaluate_cull() at p_count LOD0 voxels (frac 0), the same bits: the discoverer's corners, one batch per
    // wave, over the WorkerThreadPool in blocks of BATCH_BLOCK points (evaluate_cull is thread-safe).
    void evaluate_batch(const VoxelPos *p_points, int p_count, int p_lod, float *r_out) const {
        BatchJob job{ this, p_points, p_count, p_lod, r_out };
        const int blocks = (p_count + BATCH_BLOCK - 1) / BATCH_BLOCK;
        if (blocks <= 1) {
            if (blocks == 1) {
                _batch_block(&job, 0);
            }
            return;
        }
        WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
        const WorkerThreadPool::GroupID id = pool->add_native_group_task(&SDFBase::_batch_block, &job, blocks, -1, true, "SDFBase::evaluate_batch");
        pool->wait_for_group_task_completion(id);
    }

    // GPU code spliced into the cascade passes (SDFCascade: the cascade's register, metres and skirt
    // functions). The shader provides the chunk_param
    // uniform, the frame functions of gpu_frame.hpp and the exact noise of exact_noise.hpp.
    virtual String get_glsl_code() const = 0;
    // The GLSL the cascade level passes' fold variant splices after the table buffers (cascade_glsl.hpp), defining
    // `vec4 cascade_sample_sdf(vec4 c0, WorldPos p, int lod, WorldPos group_lo, WorldPos group_hi, uint chunk, out uint
    // material, out float skirt_extra)` (material: the winning record's palette index, 0 = ground; skirt_extra: metres
    // added to the sample's skirt depth), called by every invocation of a p_group_size workgroup together (it has
    // barriers), [group_lo, group_hi] holding the workgroup's points.
    virtual String get_sample_fold_glsl(uint32_t p_group_size) const = 0;
    // With a per-sample ground material (base_palette longer than one): the GLSL the cascade level passes' fields variant
    // splices after its channel code, defining `uint cascade_ground_material(Ch ch, WorldPos p)`, the palette index of a
    // sample no record colours (ch: its channels on Λ_L, the fields carried; p: its absolute point). Empty otherwise.
    virtual String get_ground_material_glsl() const { return String(); }
    // Whether a sample's ground is more than the cascade's ground channels (a climate's height terms or terraces): then
    // the cascade level passes carry the fields at every level, point items too, and splice get_ground_glsl after their
    // channel code, defining `vec4 climate_ground(Ch ch)`, the ground channels (value, gradient) of a sample.
    virtual bool has_ground_terms() const { return false; }
    virtual String get_ground_glsl() const { return String(); }
    // The dual-number helpers the climate's generated functions share, spliced once before them all (Climate's
    // splices_helpers: a shaped modulation or ground terms); empty when each brings its own.
    virtual String get_climate_helpers_glsl() const { return String(); }

    // The instance table's GPU buffers (SDFTable). The meshers splice and bind it, and the cascade passes fold it
    // into their samples. The const form only reads (the shader text, the sizes).
    virtual SDFGpuTable *get_gpu_table() = 0;
    const SDFGpuTable *get_gpu_table() const { return const_cast<SDFBase *>(this)->get_gpu_table(); }
    // The materials the table's records name (TableRecord::material): the page shader's material_tex.
    virtual const Material::Library &get_materials() const = 0;

    // The discoverer's no-cull rule: a node whose box intersects any instance's influence box (grown by
    // the cull band, 2 chunk sizes at the node's LOD) must never be culled, whatever the corners say.
    // This makes the cull independent of the slope bound (steep, rotated or subtracted shapes). p_past_band_m: how far
    // the smallest |evaluate_cull| at the node's corners is past the cull band (metres, > 0, else it isn't asked).
    virtual bool cull_exempt(const ChunkId &p_node, int p_lod, float p_past_band_m) const = 0;

    // Mesher gating (SDFTable's streamed sources): a chunk is meshed once every record it lists is in the GPU table, so
    // its surface never depends on what had streamed in when.
    enum class Admit {
        READY, // mesh it now
        PARKED, // kept: take_ready() hands it back once its records are there
        LATER, // no room for its records now: queue it again and stop popping this frame
    };
    // Main thread, for each chunk a mesher pops (p_owner: the mesher; p_tag comes back with the chunk; p_rank: lower is
    // served first). READY: r_words gets a bound of the words its instance list adds (the streamed records its keys hold;
    // 0 without them).
    virtual Admit admit(const ChunkId & /*p_chunk*/, uint64_t /*p_owner*/, uint64_t /*p_tag*/, uint64_t /*p_rank*/, uint64_t &r_words) {
        r_words = 0;
        return Admit::READY;
    }
    // Main thread: p_owner's tags are p_tag or later from now on. Its chunks parked under an older one with p_rank(chunk)
    // >= 0 take p_tag and that rank (p_kept(chunk); take_ready() hands them back), the others (-1) are forgotten and the
    // records only they asked for dropped if not started.
    virtual void set_tag(uint64_t /*p_owner*/, uint64_t /*p_tag*/, const std::function<int64_t(const ChunkId &)> & /*p_rank*/,
            const std::function<void(const ChunkId &)> & /*p_kept*/) {}
    // Main thread: p_owner's parked chunks now ready (or to ask again), with their tags, in the order they were parked.
    virtual void take_ready(uint64_t /*p_owner*/, std::vector<std::pair<ChunkId, uint64_t>> &r_out) { r_out.clear(); }
    // Main thread: p_owner's chunks parked (a mesher holding some isn't idle).
    virtual size_t parked_count(uint64_t /*p_owner*/) const { return 0; }
    // Main thread, once per frame before the meshers: what the streamed sources computed since, the index base an edit
    // rebuilt.
    virtual void poll() {}

    // Runtime instance edits. rotation (9 floats, row major) and params (8 floats) may be null (identity / zero).
    // p_op is a Combine. Returns the new instance id (>= 0) or -1. When any edit succeeds, last_edit_box() holds
    // the region it changed (old box of a remove/move + new box of an add/move).
    virtual int add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k) = 0;
    virtual bool remove_instance(int p_id) = 0;
    virtual bool move_instance(int p_id, const VoxelPos &p_anchor, const float *p_rotation) = 0;
    // Whether a chunk's samples can change from the last edit (its box grown by the locality margin at
    // its LOD meets the edit region): the chunks the terrain re-meshes.
    virtual bool edit_affects_chunk(const ChunkId &p_chunk) const = 0;
    // The region the last edit changed.
    const VoxelBox &last_edit_box() const { return edit_box; }

    // The Hermite cascade's bands (never null): the meshers compute the samples with the cascade passes
    // (cascade_glsl.hpp), and evaluate() only takes lattice points (frac 0, voxel a multiple of 2^lod).
    virtual const Cascade::Spec *get_cascade() const = 0;

    // The smallest skirt depth (voxels of the chunk's LOD) this terrain needs: how far apart the surfaces of a
    // LOD L and a LOD L+1 chunk can be (a band dropped between them), so a seam shows no crack. VoxelTerrain raises
    // skirt_depth to it before configure().
    virtual float min_skirt_depth(const ChunkParam &p_param, int p_lod_count) const = 0;

    // Depth (metres) of the skirt of a LOD-p_lod chunk: how far past its bounds the SDF is pushed towards air
    // (SDFCascade sizes it from its octave amplitudes).
    virtual float skirt_depth_m(int p_lod) const = 0;

protected:
    ChunkParam param;
    VoxelBox edit_box;

private:
    static constexpr int BATCH_BLOCK = 64;
    struct BatchJob {
        const SDFBase *sdf;
        const VoxelPos *points;
        int count;
        int lod;
        float *out;
    };

    static void _batch_block(void *p_job, uint32_t p_block) {
        const BatchJob &job = *static_cast<const BatchJob *>(p_job);
        const int begin = int(p_block) * BATCH_BLOCK;
        WorldPos pos;
        for (int i = begin; i < MIN(job.count, begin + BATCH_BLOCK); i++) {
            pos.voxel = job.points[i];
            job.out[i] = job.sdf->evaluate_cull(pos, job.lod);
        }
    }
};
