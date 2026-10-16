#pragma once

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/hash_map.hpp>

#include "chunk_math.hpp"
#include "chunk_memory.hpp"
#include "debug_log.hpp"
#include "lod_structures.hpp"
#include "quad_page_pool.hpp"
#include "sdf_base.hpp"

#include <vector>

using namespace godot;

// GPU-direct mesher: SDF -> surface-nets quads entirely on the main RenderingDevice, written
// straight into QuadPagePool MultiMesh pages. The CPU never sees the geometry: per batch it only
// receives (asynchronously, a few frames later) a small directory of where each chunk landed,
// which is bookkeeping for page recycling and chunk destruction, not needed to draw.
//
// Batch lifecycle (one batch per frame at most):
//   main:   pop <= batch_size chunks (Morton order), acquire pages_per_batch pages
//   render: clear counters, 5 compute passes, copy staging -> pages, copy fill -> indirect counts,
//           async readback of directory + status
//   render: (frames later) readback callback -> completed queue
//   main:   get_results(): chunks become loaded (ChunkOutput with page/offset/count), unused
//           pages go back to the pool, overflowed chunks are re-queued
// Destroying a chunk clears its quad range (zero transforms = degenerate, never rasterised) and
// releases the page once nothing alive is left in it.
class GPUDirectMesher : public Object, public ChunkReleaser {

public:
    struct Settings {
        uint32_t batch_size = 128;
        uint32_t pages_per_batch = 16;
        uint32_t page_capacity = 32768;
        uint32_t initial_pages = 64;
    };

    void initialize(SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers, RID p_scenario, const Settings &p_settings);
    // Frees everything and deletes this object (on the render thread, after pending work).
    // Loaded chunks must have been released before.
    void destroy();

    void send_chunk(LODHashSet p_chunks);
    void process_queue();
    bool get_results(VoxelState &output);
    bool is_idle();

    void release_chunk(const ChunkOutput &p_chunk) override;

    void set_debug_view(int p_view);
    Dictionary get_stats();
    QuadPagePool *get_pool() const { return pool; }

private:
    struct InFlightBatch {
        std::vector<Vector3i> chunks;
        std::vector<int> pages;
        uint64_t dispatch_frame = 0;
    };

    struct CompletedBatch {
        uint32_t batch_id;
        PackedByteArray directory;
        PackedByteArray status;
    };

    struct PageState {
        uint32_t live_quads = 0;
        int live_chunks = 0;
    };

    SDFBase *sdf = nullptr;
    ChunkParam chunk_param;
    Settings settings;
    RID scenario;

    QuadPagePool *pool = nullptr;

    std::vector<LODHashSet *> chunk_queue;
    LODHashSet working_generation;

    uint32_t next_batch_id = 0;
    HashMap<uint32_t, InFlightBatch> in_flight;
    std::vector<PageState> page_state; // indexed by pool page

    // Kills collected during the frame, flushed to the render thread once.
    PackedInt32Array pending_kill_pages;
    Array pending_kill_multimeshes;
    PackedInt64Array pending_kill_ranges; // offset, count pairs (quads)

    Ref<Mutex> mutex; // guards `completed` and the render-thread timing stats
    std::vector<CompletedBatch> completed;

    uint64_t frame = 0;

    // ---- render thread only ----
    RenderingDevice *rd = nullptr;
    RID buffer_chunk_dir;
    RID buffer_input_chunk;
    RID buffer_sdf;
    RID buffer_edges;
    RID buffer_vertices;
    RID buffer_chunk_param;
    RID staging_texture;
    RID buffer_status;
    RID shaders[5];
    RID pipelines[5];
    RID uniform_set;
    uint32_t status_size_bytes = 0;

    void _rt_initialize();
    RID _rt_compile(const char *p_pass);
    void _rt_dispatch(uint32_t p_batch_id, const PackedByteArray &p_chunk_ids, uint32_t p_count,
            const Array &p_page_multimeshes, const PackedInt32Array &p_pages);
    void _rt_on_directory(const PackedByteArray &p_data, uint32_t p_batch_id);
    void _rt_on_status(const PackedByteArray &p_data, uint32_t p_batch_id);
    void _rt_kill(const PackedInt32Array &p_pages, const Array &p_multimeshes, const PackedInt64Array &p_ranges);
    void _rt_read_timestamps();
    void _rt_destroy();

    // completion assembly (render thread writes, main thread consumes; under mutex)
    HashMap<uint32_t, CompletedBatch> partial;

    bool _pop_queue(Vector3i &r_chunk);
    void _release_page_if_dead(int p_page);
    AABB _chunk_aabb(const Vector3i &p_chunk) const;

    struct Stats {
        uint64_t batches = 0;
        uint64_t chunks_meshed = 0;
        uint64_t quads_emitted = 0;
        uint64_t overflow_chunks = 0;
        uint64_t missing_neighbor = 0;
        uint64_t nan_vertices = 0;
        uint64_t kills = 0;
        uint64_t pages_returned_unused = 0;
        TimingStat dispatch_cpu;    // main thread: process_queue when it dispatches
        TimingStat collect_cpu;     // main thread: get_results when it consumes a batch
        TimingStat rt_record_cpu;   // render thread: recording one batch
        uint64_t readback_frames_total = 0; // frames between dispatch and directory arrival
        uint64_t readback_frames_max = 0;
        TimingStat gpu_compute;     // GPU time of passes 1-5 (from timestamps)
        TimingStat gpu_copy;        // GPU time of the page copies
        TimingStat gpu_pass[5];
        uint64_t live_quads = 0;
    } stats;
};
