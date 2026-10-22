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

    // The snapshot version the GPU should mirror (record uploads are skipped while it is unchanged).
    virtual uint32_t version() const = 0;
    virtual uint32_t instance_count() const = 0;
    // How many instances the GPU buffers are sized for (runtime edits only append; more is refused).
    virtual uint32_t instance_capacity() const { return instance_count(); }
    // The encoded DefRecords of the current snapshot changed since p_have_version (UINT32_MAX: none sent yet,
    // all of them), as spans for upload_records(): uint32 span count, then per span its byte offset in the
    // buffer, its byte size and the bytes. r_version is the version they are: one snapshot for both, whatever
    // edits land meanwhile. An edit re-sends the pages it changed, not the table.
    virtual PackedByteArray record_bytes(uint32_t p_have_version, uint32_t &r_version) const = 0;
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
    // chunk's palette (chunk_palette.hpp): the global materials its corners' palette indices name.
    virtual uint32_t build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list,
            uint32_t p_capacity_words, std::vector<std::vector<uint32_t>> *r_palettes = nullptr) const = 0;
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

    // The value the discoverer's cull reads at a corner, leaving out what cull_exempt() covers anyway (SDFCascade:
    // the bare cascade, the GPU discoverer's point items bit for bit).
    virtual float evaluate_cull(const WorldPos &p_pos, int p_lod) const = 0;

    // evaluate_cull's field at a LOD0 voxel as LOD p_lod sees it, with its gradient, both in LOD0 voxels (the cell sites,
    // Archetype::FieldFn: LOD 0 is the field, a coarser LOD its nearest lattice sample extended along its gradient).
    // Thread-safe.
    virtual ExactNoise::Float4 ground_field(const VoxelPos &p_voxel, int p_lod) const = 0;

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
    // `vec4 cascade_sample_sdf(vec4 c0, WorldPos p, int lod, uint chunk, out uint material, out float skirt_extra)`
    // (material: the winning record's palette index, 0 = ground; skirt_extra: metres added to the sample's skirt depth).
    virtual String get_sample_fold_glsl() const = 0;
    // With a per-sample ground material (base_palette longer than one): the GLSL the cascade level passes' fields variant
    // splices after its channel code, defining `uint cascade_ground_material(Ch ch, WorldPos p)`, the palette index of a
    // sample no record colours (ch: its channels on Λ_L, the fields carried; p: its absolute point). Empty otherwise.
    virtual String get_ground_material_glsl() const { return String(); }

    // The instance table's GPU buffers (SDFTable). The meshers splice and bind it, and the cascade passes fold it
    // into their samples. The const form only reads (the shader text, the sizes).
    virtual SDFGpuTable *get_gpu_table() = 0;
    const SDFGpuTable *get_gpu_table() const { return const_cast<SDFBase *>(this)->get_gpu_table(); }
    // The materials the table's records name (TableRecord::material): the page shader's material_tex.
    virtual const Material::Library &get_materials() const = 0;

    // The discoverer's no-cull rule: a node whose box intersects any instance's influence box (grown by
    // the cull band, 2 chunk sizes at the node's LOD) must never be culled, whatever the corners say.
    // This makes the cull independent of the slope bound (steep, rotated or subtracted shapes).
    virtual bool cull_exempt(const ChunkId &p_node, int p_lod) const = 0;

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
