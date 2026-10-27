#pragma once

#include "chunk_math.hpp"
#include "debug_log.hpp"
#include "local_points_gpu.hpp"
#include "lod_structures.hpp"
#include "ov/executor.hpp"
#include "sdf_base.hpp"

#include "ov/containers.hpp"
#include "ov/math.hpp"
#include "ov/value.hpp"

#include <mutex>

#include <memory>
#include <vector>

class ChunkDiscoverer : public ov::WorkerLoop {
public:
    // How a node of the LOD tree is dropped as holding no surface.
    enum CullMode {
        CULL_NONE = 0, // keep everything (reference set of the discovery audit)
        CULL_LEGACY = 1, // |SDF| at the 8 corners > 2 chunk sizes
    };

    // One wave of a scan: the nodes of one LOD.
    struct WaveStats {
        int lod = 0;
        int64_t nodes = 0;
        int64_t evaluated = 0; // nodes whose corners were evaluated (no cached verdict, or verify)
        int64_t corners = 0; // their corners, 8 per node
        int64_t unique_corners = 0; // after dedup: the SDF evaluations
        uint64_t eval_usec = 0; // the corner batch (SDF evaluations only)
        bool gpu = false; // evaluated by LocalPointsGPU
        int batches = 0; // GPU submissions
        double gpu_ms = 0.0; // GPU timestamps around the cascade levels (-1: not available)
        int64_t readback_bytes = 0;
    };

    // Work done by one scan.
    struct ScanCounters {
        int64_t nodes_visited = 0;
        int64_t sdf_calls = 0; // corner evaluations after dedup
        int64_t corners = 0; // corners of the evaluated nodes, before dedup
        int64_t culled = 0;
        int64_t out_of_view = 0; // scanned chunks outside the view cone
        uint64_t cull_nsec = 0; // time spent in the cull test (corners, dedup, SDF, verdicts), in ns
        int64_t cache_hits = 0;
        int64_t verify_mismatches = 0;
        std::vector<WaveStats> waves;
    };

    // Per-node verdict of the cull test. It only depends on the node id and the SDF, which does not
    // change while the discoverer runs, so a cached verdict is exact. Worker thread only.
    struct CullCache {
        struct Entry {
            bool outside = false;
            uint32_t last_scan = 0;
        };
        ov::HashMap<ChunkId, Entry, Vector3i64Hasher> entries;
        uint32_t scan_index = 0;
        bool verify = false; // recompute the verdict on every hit and count mismatches
    };

private:
    // The scans run on a worker of the core's executor (start / stop): enter(), a step() per wake (a camera chunk
    // crossing or an edit), exit().
    std::unique_ptr<ov::Worker> worker;
    std::mutex mutex;

    // Worker only: the last scan's view and set (kept up to date by applying each delta, so the published set leaves
    // without a copy).
    bool has_scanned = false;
    ScanView last_scanned_view;
    LODHashSet previous_scan;

    // non thread safe variable
    ChunkParam chunk_param = ChunkParam();
    std::vector<float> lod_distances;
    double cone_half_angle = ov::PI; // the view cone of ScanView: every node is in it by default
    double back_scale = 0.0; // ScanView::back_scale
    double memory_distance = 0.0; // chunk-id units (set_view_memory); 0: no memory
    CullMode cull_mode = CULL_LEGACY;
    bool use_cache = true;
    CullCache cache;
    bool want_gpu = false; // discoverer_mode 1: corners on LocalPointsGPU
    static constexpr uint32_t GPU_POINTS = 65536; // corners per GPU batch (a wave above it is split)
    bool compare = false; // --discovery_compare: diff every scan against a fresh CPU scan
    // Owned by the worker: created in enter(), freed in exit(), on its thread.
    LocalPointsGPU *gpu = nullptr;

    // actively mutexed
    ScanView view; // the camera's latest view that asked for a scan
    // Region(s) an instance edit changed (LOD0 voxels): the worker drops the cull verdicts of the nodes
    // near them and rescans even if the camera chunk didn't change.
    std::vector<VoxelBox> pending_invalidations;
    bool rescan_requested = false;
    uint64_t camera_change_usec = 0; // when set_camera last changed the view
    std::unique_ptr<LODHashSet> scanned_chunks; // published scan, handed over without a copy
    std::unique_ptr<ScanDelta> scanned_delta; // last taken scan -> published scan
    ScanView scanned_view; // the view the published scan was made for
    bool scanned_chunks_dirty = false;
    uint64_t published_scans = 0; // the published scan's number: its trace flow to the main thread
    struct Stats {
        TimingStat scan;
        TimingStat cull; // part of `scan` spent in the cull test
        TimingStat publish; // handing the set over (under the mutex), merging deltas included
        TimingStat delta; // previous scan -> this one (ScanDelta::between, then apply)
        int64_t deltas_merged = 0; // scans the main thread skipped: their delta merged into the next one
        int64_t last_delta_entered = 0;
        int64_t last_delta_left = 0;
        TimingStat latency; // view change (set_camera) -> scan published
        int64_t scans_skipped_same_chunk = 0;
        int64_t last_scan_size = 0;
        int64_t last_out_of_view = 0;
        int64_t last_memories = 0;
        int64_t memories_added = 0;
        int64_t memories_replaced = 0;
        int64_t delta_entered_total = 0; // chunks that entered / left the scanned set, over every scan
        int64_t delta_left_total = 0;
        int64_t last_nodes_visited = 0;
        int64_t last_sdf_calls = 0;
        int64_t last_culled = 0;
        int64_t nodes_visited_total = 0;
        int64_t sdf_calls_total = 0;
        int64_t last_cache_hits = 0;
        int64_t cache_hits_total = 0;
        int64_t cache_size = 0;
        int64_t cache_evicted_total = 0;
        int64_t verify_mismatches_total = 0;
        ov::Value first_scan = ov::Value::object(); // the first scan's counters and waves (a cold cache)
        ov::Value last_scan = ov::Value::object();
        bool gpu_active = false;
        double gpu_scratch_mb = 0.0;
        uint64_t gpu_init_usec = 0; // local device + point shader, before the first scan
        int64_t compare_scans = 0;
        int64_t compare_differences = 0;
        TimingStat compare_cpu; // the fresh CPU scans of --discovery_compare (in latency, not in scan or publish)
    } stats;

    SDFBase *sdf;

    // The cull verdicts of one wave (all nodes of LOD p_lod): cached ones from p_cache, the others from
    // their corners, deduplicated and evaluated in one batch. r_outside[i]: node i holds no surface.
    void _cull_wave(const std::vector<ChunkId> &p_wave, int p_lod, CullMode p_mode, std::vector<uint8_t> &r_outside, ScanCounters &r_counters,
            CullCache *p_cache, LocalPointsGPU *p_gpu) const;
    // Fresh verdicts of nodes p_nodes (indices into p_wave, all of LOD p_lod).
    void _corner_verdicts(const std::vector<ChunkId> &p_wave, const std::vector<uint32_t> &p_nodes, int p_lod, std::vector<uint8_t> &r_outside,
            WaveStats &r_wave, LocalPointsGPU *p_gpu) const;
    // --discovery_compare: the CPU scan with a fresh cache, diffed with p_scanned; ERROR on a difference.
    void _compare_with_cpu(const ScanView &p_view, const LODHashSet &p_scanned);
    // A node is split when its bounding sphere comes closer to the camera than its LOD's distance, the
    // distance being grown away from the view cone (ChunkMath::lod_distance).
    bool _should_split(const ScanView &p_view, const ChunkId &p_node, int p_lod) const;
    // Drops the cached verdicts no recent scan used; the number dropped.
    int64_t _evict_cache(int64_t p_nodes_visited);
    // Forgets the cull verdicts of the nodes whose box grown by the cull band meets one of p_regions (an edit's).
    void _drop_stale_verdicts(const std::vector<VoxelBox> &p_regions);
    // Hands the scan over to the main thread (get_scanned_chunks) and records its stats. p_ready_usec: when
    // the scan (and its compare) finished.
    void _publish(std::unique_ptr<LODHashSet> p_scan, std::unique_ptr<ScanDelta> p_delta, const ScanView &p_view, const ScanCounters &p_counters, int64_t p_evicted, uint64_t p_scan_usec,
            uint64_t p_ready_usec);
    static ov::Value _scan_dict(const ScanCounters &p_counters, uint64_t p_scan_usec);

public:
    void start();
    void stop();

    // A turn of the view axis past this angle from the views remembered at the camera chunk remembers a new
    // one, which asks for a rescan (when the LOD rule depends on the direction).
    static constexpr double RESCAN_TURN_DEG = 8.0;
    // The angle between unit axes p_a and p_b is over RESCAN_TURN_DEG.
    static bool turned_past(const Float3 &p_a, const Float3 &p_b);
    // lod_view_memory_m < 0: the memory distance is where this LOD starts.
    static constexpr int VIEW_MEMORY_AUTO_LOD = 2;

    // The view of a camera at render position p_pos looking along p_forward, without memories.
    ScanView make_view(const Float3 &p_pos, const Float3 &p_forward) const;
    // p_new needs another scan than p_scanned: another camera chunk or other memories.
    static bool view_changed(const ScanView &p_scanned, const ScanView &p_new);
    enum Remembered {
        REMEMBERED_KNOWN, // a memory at the camera chunk is within RESCAN_TURN_DEG of forward
        REMEMBERED_ADDED,
        REMEMBERED_REPLACED, // added over the most faded one (full)
    };
    // r_view (camera chunk, forward and the previous view's memories) remembers its own view, with the fades
    // of p_memory_distance (chunk-id units) from its camera chunk, sorted by fade. Up to MAX_VIEW_MEMORIES;
    // with p_memory_distance 0, only one: the view replaced on every new camera chunk or turn. No back_scale
    // or no forward: no memories. Static for the unit tests.
    static Remembered remember_view(ScanView &r_view, double p_memory_distance);
    // Remembers the camera's view (remember_view) and asks for a scan when the view changed (view_changed).
    void set_camera(const Float3 &p_pos, const Float3 &p_forward);
    // The view of the latest set_camera, memories included.
    ScanView get_view();
    // How far the camera walks from where it had a view before that view starts to fade (render metres;
    // < 0: where LOD VIEW_MEMORY_AUTO_LOD starts, 0: no memory). After setup_lod, before start().
    void set_view_memory(double p_metres);
    // An instance edit changed the SDF in p_box: the next scan drops the cull verdicts of the nodes
    // near it and runs even if the camera chunk didn't change.
    void invalidate(const VoxelBox &p_box);
    // Takes the latest scan if there is a new one; r_view is the view it was made for and r_delta the
    // change from the scan taken before (from the empty set for the first one).
    bool get_scanned_chunks(std::unique_ptr<LODHashSet> &r_scanned_chunks, std::unique_ptr<ScanDelta> &r_delta, ScanView &r_view);

    // The full scan around `p_view`'s camera chunk. Const and thread safe: the discovery audit calls it from
    // the main thread to rebuild the published scan without culling.
    // `p_cache` (optional) remembers each node's verdict across scans. `p_gpu` (optional, the worker
    // thread's own) evaluates the corners on the GPU instead of SDFBase::evaluate_batch, the same bits.
    // Breadth first, one wave per LOD from the top-LOD grid down: the wave's verdicts (_cull_wave), then
    // its surviving nodes are kept or split into the next wave by distance. The scanned set doesn't
    // depend on the visit order, so it is the one the old recursive walk gave (unit-tested).
    void scan(const ScanView &p_view, CullMode p_mode, LODHashSet &r_chunks, ScanCounters &r_counters,
            CullCache *p_cache = nullptr, LocalPointsGPU *p_gpu = nullptr) const;
    int get_max_lod() const { return int(lod_distances.size()) - 2; }
    const std::vector<float> &get_lod_distances() const { return lod_distances; }
    // The scan grid (after setup_lod): top-LOD chunk centres step `r_step` ids, from -r_count to
    // +r_count steps around the camera's top-LOD ancestor.
    void get_scan_grid(int64_t &r_step, int64_t &r_count) const;

    ov::Value get_stats();

    // Must be called before start().
    void set_config(
            ChunkParam &p_chunk_param,
            SDFBase *p_sdf);
    void set_cull_mode(CullMode p_mode) { cull_mode = p_mode; }
    void set_cache(bool p_enable, bool p_verify) {
        use_cache = p_enable;
        cache.verify = p_verify;
    }
    // Corners on the GPU (LocalPointsGPU; CPU with one warning if it can't be created) and the debug diff
    // against a fresh CPU scan.
    void set_gpu(bool p_enable) { want_gpu = p_enable; }
    void set_compare(bool p_enable) { compare = p_enable; }

    // ov::WorkerLoop: the GPU discoverer's device (enter, exit) and one scan per wake (step).
    void enter() override;
    bool step() override;
    void exit() override;

    // The LOD distances of `target_pixel_error`. Outside a view cone of half-angle `cone_half_angle_rad`
    // they shrink towards those of `target_pixel_error_back` (ignored unless coarser than the front error).
    // `target_pixel_errors[L]` (> 0) replaces the error of LOD L, which sizes its band: a finer error widens it, more
    // of its chunks load; the back ratio applies on top. LODs past the list keep `target_pixel_error`.
    void setup_lod(
            int lod_levels = 4,
            float target_pixel_error = 10.0f,
            float viewport_height = 1080.0f,
            float fov_rad = 75.0f * (3.14159265f / 180.0f),
            float target_pixel_error_back = 0.0f,
            float cone_half_angle_rad = float(ov::PI),
            const std::vector<float> &target_pixel_errors = {});
};
