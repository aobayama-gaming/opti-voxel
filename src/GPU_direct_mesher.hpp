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
//   main:   pop <= batch_size chunks, acquire pages_per_batch pages. The queue is kept in Morton
//           order, so a batch is a compact region and so is every page: pages are frustum culled
//           as a whole, with the union of their chunks' AABBs
//   render: clear counters, 5 compute passes, copy staging -> pages, copy fill -> indirect counts,
//           async readback of directory + status
//   render: (frames later) readback callback -> completed queue
//   main:   get_results(): chunks become loaded (ChunkOutput with page/offset/count), unused
//           pages go back to the pool, overflowed chunks are re-queued
// Destroying a chunk clears its quad range (zero transforms = degenerate, never rasterised),
// shrinks the page AABB to the chunks still alive, and releases the page once nothing is left.
//
// Compaction (main thread, once per frame, after chunk destruction): pages less than half full of
// live quads are re-packed GPU to GPU into fresh pages, survivors in Morton order, a few pages per
// frame. The CPU knows every chunk's range, so it rewrites the ChunkOutputs in place: no readback.
// Spare free pages are then freed, so VRAM follows the pages in use.
class GPUDirectMesher : public Object, public ChunkReleaser {
public:
    struct Settings {
        uint32_t batch_size = 128;
        uint32_t pages_per_batch = 16;
        uint32_t page_capacity = 32768;
        uint32_t initial_pages = 64;
        bool compaction = true;
        uint32_t compact_pages_per_frame = 8; // source pages re-packed per frame at most
        // A re-packed page is never wider than this many of its smallest chunks, so merging
        // sparse pages from distant places, or distant LOD shells, cannot defeat frustum culling.
        float compact_max_extent = 8.0f;
        uint32_t spare_pages = 64; // free pages kept allocated (4 in-flight batches' worth)
    };

    void initialize(SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers, RID p_scenario, const Settings &p_settings);
    // Frees everything and deletes this object (on the render thread, after pending work).
    // Loaded chunks must have been released before.
    void destroy();

    void send_chunk(const LODHashSet &p_chunks);
    void process_queue();
    bool get_results(VoxelState &output);
    bool is_idle();
    // Re-packs sparse pages and frees spare ones. Rewrites page/offset of the moved chunks in
    // r_state.loaded_chunk, so call it after the frame's chunk destruction (to_destroy_chunk holds
    // copies) and not while something holds page ranges (validator, auditor).
    void compact(VoxelState &r_state);

    void release_chunk(const ChunkOutput &p_chunk) override;

    // Discovery audit: quad counts of chunks that are not loaded. Passes sdf, edges and vertices only,
    // nothing is written to pages, nothing becomes loaded. Batches run only when the mesh queue is empty.
    void audit_count_chunks(const std::vector<ChunkId> &p_chunks);
    // Counts that came back since the last call (consumed by get_results()).
    void take_audit_counts(std::vector<std::pair<ChunkId, uint32_t>> &r_counts);
    bool is_audit_idle() const { return audit_queue.empty() && audit_in_flight.is_empty(); }

    void set_debug_view(int p_view);
    Dictionary get_stats();
    QuadPagePool *get_pool() const { return pool; }

private:
    struct InFlightBatch {
        std::vector<ChunkId> chunks;
        std::vector<int> pages;
        uint64_t dispatch_frame = 0;
    };

    struct CompletedBatch {
        uint32_t batch_id;
        PackedByteArray directory;
        PackedByteArray status;
    };

    struct PageChunk {
        ChunkId id;
        uint32_t offset = 0; // quads, a multiple of QUAD_GROUP
        uint32_t count = 0;
    };

    struct PageState {
        uint32_t live_quads = 0;
        uint32_t live_groups = 0; // groups covered by live chunks
        uint32_t groups = 0; // groups drawn (the page's indirect instance count)
        std::vector<PageChunk> chunks; // live chunks with geometry in this page
        AABB bounds;
        bool bounds_dirty = false; // a chunk died: recompute `bounds` at the next process_queue
        bool settled = false; // compaction found nothing to gain; cleared when a chunk dies
    };

    SDFBase *sdf = nullptr;
    ChunkParam chunk_param;
    Settings settings;
    RID scenario;

    QuadPagePool *pool = nullptr;

    // Per LOD, sorted by descending Morton key: pop_back() walks space along the Morton curve.
    std::vector<std::vector<ChunkId>> chunk_queue;
    LODHashSet working_generation;

    uint32_t next_batch_id = 0;
    HashMap<uint32_t, InFlightBatch> in_flight;
    std::vector<PageState> page_state; // indexed by pool page
    std::vector<int> dirty_bounds_pages;

    // Kills collected during the frame, flushed to the render thread once.
    PackedInt32Array pending_kill_pages;
    PackedInt64Array pending_kill_ranges; // offset, count pairs (quads)

    Ref<Mutex> mutex; // guards `completed`, `audit_completed` and the render-thread timing stats
    std::vector<CompletedBatch> completed;

    std::vector<ChunkId> audit_queue;
    HashMap<uint32_t, std::vector<ChunkId>> audit_in_flight;
    std::vector<CompletedBatch> audit_completed; // directory only
    std::vector<std::pair<ChunkId, uint32_t>> audit_counts;

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
            const PackedInt32Array &p_pages);
    void _rt_dispatch_count(const PackedByteArray &p_chunk_ids, uint32_t p_count, uint32_t p_batch_id);
    void _rt_on_directory(const PackedByteArray &p_data, uint32_t p_batch_id);
    void _rt_on_audit_directory(const PackedByteArray &p_data, uint32_t p_batch_id);
    void _rt_on_status(const PackedByteArray &p_data, uint32_t p_batch_id);
    void _rt_kill(const PackedInt32Array &p_pages, const PackedInt64Array &p_ranges);
    void _rt_compact(const PackedInt32Array &p_copies, const PackedInt32Array &p_group_counts);
    void _rt_read_timestamps();
    void _rt_destroy();

    // completion assembly (render thread writes, main thread consumes; under mutex)
    HashMap<uint32_t, CompletedBatch> partial;

    bool _pop_queue(ChunkId &r_chunk);
    void _dispatch_audit_batch();
    void _flush_kills();
    bool _compact_pages(const std::vector<int> &p_pages, VoxelState &r_state);
    void _release_page_if_dead(int p_page);
    void _update_dirty_bounds();
    AABB _chunk_aabb(const ChunkId &p_chunk) const;

    struct Stats {
        uint64_t batches = 0;
        uint64_t chunks_meshed = 0;
        uint64_t quads_emitted = 0;
        uint64_t overflow_chunks = 0;
        uint64_t overflow_requeued = 0; // chunks sent back to the queue because their batch overflowed
        uint64_t audit_batches = 0;
        uint64_t missing_neighbor = 0;
        uint64_t nan_vertices = 0;
        uint64_t kills = 0;
        uint64_t pages_returned_unused = 0;
        uint64_t compact_rounds = 0;
        uint64_t compact_skipped = 0; // windows with nothing to gain (pages marked settled)
        uint64_t pages_compacted = 0; // source pages re-packed
        uint64_t pages_compacted_into = 0;
        uint64_t chunks_moved = 0;
        uint64_t compact_copies = 0; // texture copies (runs of contiguous chunks are merged)
        TimingStat compact_cpu; // main thread: compact when it re-packs
        TimingStat dispatch_cpu; // main thread: process_queue when it dispatches
        TimingStat collect_cpu; // main thread: get_results when it consumes a batch
        TimingStat rt_record_cpu; // render thread: recording one batch
        uint64_t readback_frames_total = 0; // frames between dispatch and directory arrival
        uint64_t readback_frames_max = 0;
        TimingStat gpu_compute; // GPU time of passes 1-5 (from timestamps)
        TimingStat gpu_copy; // GPU time of the page copies
        TimingStat gpu_pass[5];
        uint64_t live_quads = 0;
    } stats;
};
