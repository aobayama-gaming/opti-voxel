#pragma once

#include "godot_cpp/variant/dictionary.hpp"
#include "godot_cpp/variant/packed_byte_array.hpp"
#include "godot_cpp/variant/rid.hpp"
#include "godot_cpp/variant/string.hpp"
#include "godot_cpp/variant/vector3.hpp"

#include "chunk_math.hpp"

#include <godot_cpp/classes/worker_thread_pool.hpp>

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
// global g_sdf_chunk. The generated terrain_sdf (archetype_table.hpp) reads sdf_records / sdf_list.
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
    // The encoded DefRecords of the current snapshot changed since p_have_version (UINT32_MAX: none sent
    // yet, all of them): records [r_first, r_first + bytes / record_size()), to write at r_first *
    // record_size(). r_version is the version those bytes are: one snapshot for both, whatever edits land
    // meanwhile. An edit re-sends the pages it changed, not the table.
    virtual PackedByteArray record_bytes(uint32_t p_have_version, uint32_t &r_version, uint32_t &r_first) const = 0;
    // Bytes per record in the GPU buffer.
    virtual uint32_t record_size() const = 0;

    // Main thread: patch each ChunkInput of the batch (id.w = this chunk's list offset in uint32 words,
    // anchor_hi.w = its instance count) and append the indices (uint32) to r_list. r_chunk_ids already
    // holds the batch's ChunkInput array (GPUFrame::encode_chunk_input). A list never exceeds
    // max_chunk_list(), so a batch's lists fit in batch size * max_chunk_list() words.
    virtual void build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list) const = 0;

    // The longest list one chunk carries: the meshers size their list buffer for batch size * this. The band
    // rule keeps lists to the instances big enough for the chunk's LOD; a longer list is truncated (ERR, counted).
    static constexpr uint32_t MAX_CHUNK_LIST = 4096;
    uint32_t max_chunk_list() const { return MAX<uint32_t>(MIN(instance_capacity(), MAX_CHUNK_LIST), 1u); }
    // List statistics (main thread): lengths per LOD, build time, truncations.
    virtual Dictionary get_stats() const { return Dictionary(); }
};

// An axis-aligned box in LOD0 voxels (inclusive), the region an instance edit changed.
struct VoxelBox {
    VoxelPos lo;
    VoxelPos hi;
    bool valid = false;

    bool intersects(const VoxelBox &p_o) const {
        return valid && p_o.valid && lo.x <= p_o.hi.x && p_o.lo.x <= hi.x && lo.y <= p_o.hi.y && p_o.lo.y <= hi.y && lo.z <= p_o.hi.z &&
                p_o.lo.z <= hi.z;
    }
};

class SDFBase {
public:
    // Every terrain's datum, metres of absolute y: a flat world, so far origins only show terrain at y = 0.
    static constexpr double DATUM_M = 300.0;

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

    // evaluate() at p_count LOD0 voxels (frac 0), the same bits: the discoverer's corners, one batch per
    // wave. The default loops over evaluate(); a terrain may spread it over threads.
    virtual void evaluate_batch(const VoxelPos *p_points, int p_count, int p_lod, float *r_out) const {
        WorldPos pos;
        for (int i = 0; i < p_count; i++) {
            pos.voxel = p_points[i];
            r_out[i] = evaluate(pos, p_lod);
        }
    }

    // GPU code: must define "vec4 terrain_sdf(WorldPos p, int lod)" (xyz gradient, w distance). The
    // shader provides the chunk_param uniform, the frame functions of gpu_frame.hpp and the exact
    // noise of exact_noise.hpp.
    virtual String get_glsl_code() const = 0;

    // The instance table's GPU buffers, if this terrain owns one (archetype instances). The meshers
    // splice and bind it; a terrain without a table (Dummy, cascade) leaves it null.
    virtual SDFGpuTable *get_gpu_table() { return nullptr; }

    // The discoverer's no-cull rule: a node whose box intersects any instance's influence box (grown by
    // the cull band, 2 chunk sizes at the node's LOD) must never be culled, whatever the corners say.
    // This makes the cull independent of the slope bound (steep, rotated or subtracted shapes). False for
    // a terrain without instances.
    virtual bool cull_exempt(const ChunkId &p_node, int p_lod) const {
        (void)p_node;
        (void)p_lod;
        return false;
    }

    // Runtime instance edits (archetype instances). A terrain without an instance table rejects them.
    // rotation (9 floats, row major) and params (8 floats) may be null (identity / zero). p_op is a
    // Combine. Returns the new instance id (>= 0) or -1. When any edit succeeds, last_edit_box() holds
    // the region it changed (old box of a remove/move + new box of an add/move).
    virtual int add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k) {
        (void)p_type;
        (void)p_anchor;
        (void)p_rotation;
        (void)p_params;
        (void)p_op;
        (void)p_k;
        return -1;
    }
    virtual bool remove_instance(int p_id) {
        (void)p_id;
        return false;
    }
    virtual bool move_instance(int p_id, const VoxelPos &p_anchor, const float *p_rotation) {
        (void)p_id;
        (void)p_anchor;
        (void)p_rotation;
        return false;
    }
    // Whether a chunk's samples can change from the last edit (its box grown by the locality margin at
    // its LOD meets the edit region): the chunks the terrain re-meshes.
    virtual bool edit_affects_chunk(const ChunkId &p_chunk) const {
        (void)p_chunk;
        return false;
    }
    // The region the last edit changed.
    const VoxelBox &last_edit_box() const { return edit_box; }

    // The Hermite cascade's bands, if this terrain is one (SDFCascade): the meshers then compute the
    // samples with the cascade passes (cascade_glsl.hpp) instead of calling terrain_sdf, and evaluate()
    // only takes lattice points (frac 0, voxel a multiple of 2^lod).
    virtual const Cascade::Spec *get_cascade() const { return nullptr; }

    // The smallest skirt depth (voxels of the chunk's LOD) this terrain needs: how far apart the surfaces of a
    // LOD L and a LOD L+1 chunk can be (a band dropped between them), so a seam shows no crack. VoxelTerrain raises
    // skirt_depth to it before configure().
    virtual float min_skirt_depth(const ChunkParam &, int) const { return 0.0f; }

    // Depth (metres) of the skirt of a LOD-p_lod chunk: how far past its bounds the SDF is pushed towards
    // air (gpu_glsl.hpp skirt_sdf). By default skirt_depth voxels of the LOD; SDFCascade sizes it from its
    // octave amplitudes.
    virtual float skirt_depth_m(int p_lod) const {
        return param.skirt_depth * param.voxel_size * float(int64_t(1) << p_lod);
    }

protected:
    ChunkParam param;
    VoxelBox edit_box;

    // evaluate_batch() over the WorkerThreadPool, in blocks of BATCH_BLOCK points (the same bits): the override
    // of a terrain whose evaluate() is thread-safe.
    static constexpr int BATCH_BLOCK = 64;
    void evaluate_batch_parallel(const VoxelPos *p_points, int p_count, int p_lod, float *r_out) const {
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

private:
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
        job.sdf->SDFBase::evaluate_batch(job.points + begin, MIN(job.count - begin, BATCH_BLOCK), job.lod, job.out + begin);
    }
};
