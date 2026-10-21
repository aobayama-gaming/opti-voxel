#pragma once

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/hash_map.hpp>
#include <godot_cpp/templates/hash_set.hpp>

#include "chunk_math.hpp"
#include "chunk_memory.hpp"
#include "debug_log.hpp"
#include "gpu_cost_model.hpp"
#include "lod_structures.hpp"
#include "quad_page_pool.hpp"
#include "sdf_base.hpp"
#include <godot_cpp/classes/image_texture.hpp>

#include <vector>

using namespace godot;

// GPU-direct mesher: SDF -> surface-nets quads entirely on the main RenderingDevice, written
// straight into QuadPagePool MultiMesh pages. The CPU never sees the geometry: per batch it only
// receives (asynchronously, a few frames later) a small directory of where each chunk landed,
// which is bookkeeping for page recycling and chunk destruction, not needed to draw.
//
// Batch lifecycle (up to max_batches_per_frame per frame, max_in_flight waiting for their directory):
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
class CascadePasses;

class GPUDirectMesher : public Object, public ChunkReleaser {
public:
    // The mesher's own compute passes: edges, vertices, reserve, emit (the samples are the cascade passes).
    static constexpr int COMPUTE_PASSES = 4;
    // The mesher set's bindings past the table's (8, 9): sdf_material, the palette texture, the batch's palettes.
    static constexpr int MATERIAL_BINDING = 10;
    static constexpr int PALETTE_IMAGE_BINDING = 11;
    static constexpr int PALETTE_BATCH_BINDING = 12;
    // A batch captures up to 10 GPU timestamps; Godot keeps 256 per frame by default
    // (debug/settings/profiler/max_timestamp_query_elements), shared with the renderer and the discoverer.
    static constexpr uint32_t MAX_BATCHES_PER_FRAME = 16;

    struct Settings {
        uint32_t batch_size = 128;
        uint32_t pages_per_batch = 16;
        // Pacing: batches dispatched per frame at most, and batches (mesh + audit) waiting for their
        // directory at most. A directory comes back 2 frames after its dispatch, so sustaining N batches
        // per frame needs 3N in flight (0 = that); each one holds pages_per_batch pages until then. More
        // per frame loads faster but lengthens the loading frames (~2-3 ms of GPU per batch).
        uint32_t max_batches_per_frame = 4;
        uint32_t max_in_flight = 0;
        // GPU time the mesh batches of one frame may take, estimated by GPUCostModel from the measured
        // batches; a batch is cut where the budget runs out (at least one chunk per frame, at most
        // max_batches_per_frame batches). 0: no budget, whole batches.
        float gpu_budget_ms = 0.0f;
        uint32_t page_capacity = 32768;
        uint32_t initial_pages = 64;
        bool compaction = true;
        uint32_t compact_pages_per_frame = 8; // source pages re-packed per frame at most
        // A re-packed page is never wider than this many of its smallest chunks, so merging
        // sparse pages from distant places, or distant LOD shells, cannot defeat frustum culling.
        float compact_max_extent = 8.0f;
        // Free pages kept allocated (0 = max_in_flight batches' worth: max_in_flight * pages_per_batch),
        // so the next batches don't recreate the pages the previous ones returned.
        uint32_t spare_pages = 0;
        // Entries of the palette texture (PalettePool, 4 B each; a power of two): the loaded chunks' palettes.
        uint32_t palette_capacity = 1u << 21;
    };

    void initialize(SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers, RID p_scenario, const Settings &p_settings);
    // Frees everything and deletes this object (on the render thread, after pending work).
    // Loaded chunks must have been released before.
    void destroy();

    // The chunks to mesh (replacing the queue), loaded in the order of ChunkMath::load_queue for p_view.
    void send_chunk(const LODHashSet &p_chunks, const ScanView &p_view);
    // Re-mesh already-loaded chunks whose surface an instance edit changed (finest LOD first).
    void remesh_chunks(const std::vector<ChunkId> &p_chunks);
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
    // Dual contouring of Hermite data (DC-lite) in the vertex pass: sharp UNION / SUBTRACT creases
    // are placed on the crease instead of the Surface Nets mass point (gpu_glsl.hpp). Set it before
    // initialize(): the render thread compiles the vertex shader in _rt_initialize.
    void set_dual_contouring(bool p_enable) { dual_contouring = p_enable; }
    Dictionary get_stats();
    QuadPagePool *get_pool() const { return pool; }

private:
    struct InFlightBatch {
        std::vector<ChunkId> chunks;
        std::vector<PalettePool::Block> palettes; // per chunk, placed at dispatch
        std::vector<int> pages;
        uint64_t dispatch_frame = 0;
        uint32_t table_version = 0; // the instance table version the batch was recomputed from
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
    // Dual contouring of Hermite data (DC-lite) in the vertex pass (gpu_glsl.hpp). Set before
    // initialize(): the worker compiles the vertex shader there.
    bool dual_contouring = false;

    QuadPagePool *pool = nullptr;
    PalettePool palettes; // blocks: main thread; texture: render thread
    Ref<ImageTexture> material_texture; // the pages' material_tex (a material param holds its RID only)

    // One queue per ChunkMath::load_queue (in the view cone first, then by LOD), each sorted by descending
    // Morton key: pop_back() walks space along the Morton curve, so a batch's chunks stay close together.
    std::vector<std::vector<ChunkId>> chunk_queue;
    ScanView load_view; // the view of the last send_chunk, for the chunks queued again
    int lod_count = 0;
    // Instance edits: their chunks, dispatched before chunk_queue, finest LOD first. send_chunk() does
    // not touch this. required_version is the table version a chunk must be re-meshed from (staleness).
    std::vector<std::vector<ChunkId>> remesh_queue;
    HashMap<ChunkId, uint32_t, Vector3i64Hasher> required_version;
    HashSet<ChunkId, Vector3i64Hasher> remesh_queued; // the chunks in remesh_queue (each at most once)
    LODHashSet working_generation;

    uint32_t next_batch_id = 0;
    HashMap<uint32_t, InFlightBatch> in_flight;
    std::vector<PageState> page_state; // indexed by pool page
    std::vector<int> dirty_bounds_pages;

    // Kills collected during the frame, flushed to the render thread once.
    PackedInt32Array pending_kill_pages;
    PackedInt64Array pending_kill_ranges; // offset, count pairs (quads)

    Ref<Mutex> mutex; // guards `completed`, `audit_completed`, `measured_batches` and the render-thread timing stats
    std::vector<CompletedBatch> completed;

    std::vector<ChunkId> audit_queue;
    HashMap<uint32_t, std::vector<ChunkId>> audit_in_flight;
    std::vector<CompletedBatch> audit_completed; // directory only
    std::vector<std::pair<ChunkId, uint32_t>> audit_counts;

    uint64_t frame = 0;

    // GPU budget (Settings::gpu_budget_ms). The render thread hands each batch's measured GPU time
    // (us, timestamps begin..copy) to `measured_batches` (under mutex); the main thread matches it with
    // the batch's chunks per LOD and corrects the model.
    GPUCostModel cost_model;
    HashMap<uint32_t, std::vector<uint32_t>> batch_lod_counts;
    std::vector<std::pair<uint32_t, uint64_t>> measured_batches;
    void _update_cost_model();

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
    RID buffer_material; // per sample: its palette index (cascade level 0 -> vertex pass)
    RID buffer_palette_batch; // the batch's PalettePool::place_batch payload
    // The instance table's buffers (archetype instances): records + the current batch's list.
    RID buffer_table_records;
    RID buffer_table_list;
    uint64_t table_list_capacity = 0; // bytes
    uint64_t sent_table_version = SDFGpuTable::NOTHING_SENT; // main thread: the upload version last bound to a dispatch
    RID shaders[COMPUTE_PASSES];
    RID pipelines[COMPUTE_PASSES];
    RID uniform_set;
    uint32_t status_size_bytes = 0;
    // The cascade passes: the samples (stat / timestamp pass 0).
    CascadePasses *cascade = nullptr;
    uint64_t cascade_scratch_bytes = 0; // under `mutex` (get_stats reads it on the main thread)
    uint64_t timestamps_frame_read = UINT64_MAX; // RD frame whose captured timestamps were consumed

    void _rt_initialize();
    RID _rt_compile(const char *p_pass, const char *p_name);
    void _queue_remesh(const ChunkId &p_chunk);
    void _rt_upload_table(const PackedByteArray &p_list, const PackedByteArray &p_records);
    // Main thread: the batch's instance lists (patched into r_ids), the chunks' palettes (if r_palettes) and, if
    // the table changed since the last dispatch, the records it changed (SDFGpuTable::record_bytes spans). Returns
    // the table version they are. The lists fill the list buffer at most: r_chunks keeps the leading chunks that
    // fit, r_rest gets the others (r_ids shrunk to match) for the caller to queue again.
    uint32_t _table_payload(std::vector<ChunkId> &r_chunks, std::vector<ChunkId> &r_rest, PackedByteArray &r_ids, PackedByteArray &r_list,
            PackedByteArray &r_records, std::vector<std::vector<uint32_t>> *r_palettes = nullptr);
    // Main thread: a chunk popped for a batch it couldn't go in (overflow, or its list didn't fit) back in its queue:
    // remesh_queue (p_remesh) or chunk_queue.
    void _requeue(const ChunkId &p_chunk, bool p_remesh);
    void _rt_dispatch(uint32_t p_batch_id, const PackedByteArray &p_chunk_ids, uint32_t p_count, const PackedInt32Array &p_pages, int p_min_lod,
            const PackedByteArray &p_list, const PackedByteArray &p_records, const PackedByteArray &p_palette_batch);
    void _rt_dispatch_count(const PackedByteArray &p_chunk_ids, uint32_t p_count, uint32_t p_batch_id, int p_min_lod,
            const PackedByteArray &p_list, const PackedByteArray &p_records);
    // Pass 0: the cascade's levels (timestamps "<prefix>_c<group>" after each level group
    // when p_prefix is not empty), folding the instances when the batch's list p_list holds any.
    void _rt_record_samples(uint32_t p_count, int p_min_lod, const String &p_prefix, const PackedByteArray &p_list);
    void _rt_on_directory(const PackedByteArray &p_data, uint32_t p_batch_id);
    void _rt_on_audit_directory(const PackedByteArray &p_data, uint32_t p_batch_id);
    void _rt_on_status(const PackedByteArray &p_data, uint32_t p_batch_id);
    void _rt_kill(const PackedInt32Array &p_pages, const PackedInt64Array &p_ranges);
    void _rt_compact(const PackedInt32Array &p_copies, const PackedInt32Array &p_group_counts);
    void _rt_read_timestamps();
    void _rt_destroy();
    // Async readback callbacks are not skipped when their object is freed (CLAUDE.md, Gotchas): _rt_destroy
    // frees the GPU resources, then the object deletes itself once the last readback has landed.
    void _rt_readback_done();
    int rt_readbacks_in_flight = 0;
    bool rt_destroying = false;

    // completion assembly (render thread writes, main thread consumes; under mutex)
    HashMap<uint32_t, CompletedBatch> partial;

    // r_remesh (if given): whether the chunk came from remesh_queue.
    bool _pop_queue(ChunkId &r_chunk, bool *r_remesh = nullptr);
    // One batch each; false when there was nothing to dispatch. r_budget_us: the frame's GPU budget left
    // (estimated), reduced by the batch's estimate; the batch stops where it runs out.
    bool _dispatch_mesh_batch(double &r_budget_us);
    bool _dispatch_audit_batch();
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
        uint64_t list_requeued = 0; // chunks sent back before dispatch: their instance list didn't fit the batch's buffer
        uint64_t residency_waits = 0; // pops of a chunk whose streamed records weren't resident yet (put back)
        uint64_t audit_batches = 0;
        uint64_t dispatch_frames = 0; // frames that dispatched at least one batch
        uint64_t in_flight_max = 0; // mesh + audit batches waiting for their directory
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
        uint64_t remesh_chunks = 0; // chunks queued by remesh_chunks()
        uint64_t remesh_pops = 0; // remesh chunks dispatched
        uint64_t remesh_stale_dropped = 0; // results from a table version older than the chunk's edit
        uint64_t remesh_queue_max = 0;
        // Cascade terrain: GPU time of the level groups j >= 3, j = 2, j = 1, j = 0 (pass 0 is their sum).
        TimingStat gpu_cascade[4];
        uint64_t live_quads = 0;
        TimingStat budget_frame_estimate; // estimated GPU time of the mesh batches dispatched in a frame
        double model_ratio_sum = 0.0; // measured / estimated GPU time of the batches matched
        uint64_t model_ratio_count = 0;
    } stats;
};
