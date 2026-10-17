#pragma once

#include "chunk_math.hpp"
#include "debug_log.hpp"
#include "lod_structures.hpp"
#include "sdf_base.hpp"
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include <godot_cpp/classes/thread.hpp>

#include <memory>
#include <vector>

using namespace godot;

class ChunkDiscoverer : public Object {
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
    };

    // Work done by one scan.
    struct ScanCounters {
        int64_t nodes_visited = 0;
        int64_t sdf_calls = 0; // corner evaluations after dedup
        int64_t corners = 0; // corners of the evaluated nodes, before dedup
        int64_t culled = 0;
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
        HashMap<ChunkId, Entry, Vector3i64Hasher> entries;
        uint32_t scan_index = 0;
        bool verify = false; // recompute the verdict on every hit and count mismatches
    };

private:
    Ref<Thread> worker;

    Ref<Semaphore> semaphore;
    Ref<Mutex> mutex;
    bool exit_thread = false;

    // non thread safe variable
    ChunkParam chunk_param = ChunkParam();
    PackedFloat32Array lod_distances;
    CullMode cull_mode = CULL_LEGACY;
    bool use_cache = true;
    CullCache cache;

    // actively mutexed
    ChunkId camera_chunk;
    uint64_t camera_change_usec = 0;
    std::unique_ptr<LODHashSet> scanned_chunks; // published scan, handed over without a copy
    ChunkId scanned_camera_chunk; // camera chunk the published scan was made for
    bool scanned_chunks_dirty = false;
    struct Stats {
        TimingStat scan;
        TimingStat cull; // part of `scan` spent in the cull test
        TimingStat publish; // handing the set over (under the mutex)
        TimingStat latency; // camera chunk change -> scan published
        int64_t scans_skipped_same_chunk = 0;
        int64_t last_scan_size = 0;
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
        Dictionary first_scan; // the first scan's counters and waves (a cold cache)
        Dictionary last_scan;
    } stats;

    SDFBase *sdf;

    // The cull verdicts of one wave (all nodes of LOD p_lod): cached ones from p_cache, the others from
    // their corners, deduplicated and evaluated in one batch. r_outside[i]: node i holds no surface.
    void _cull_wave(const std::vector<ChunkId> &p_wave, int p_lod, CullMode p_mode, std::vector<uint8_t> &r_outside, ScanCounters &r_counters,
            CullCache *p_cache) const;
    // Fresh verdicts of nodes p_nodes (indices into p_wave, all of LOD p_lod).
    void _corner_verdicts(const std::vector<ChunkId> &p_wave, const std::vector<uint32_t> &p_nodes, int p_lod, std::vector<uint8_t> &r_outside,
            WaveStats &r_wave) const;
    // A node is split when its bounding sphere comes closer to the camera than its LOD's distance.
    bool _should_split(const ChunkId &p_camera_chunk, const ChunkId &p_node, int p_lod) const;
    static Dictionary _scan_dict(const ScanCounters &p_counters, uint64_t p_scan_usec);

public:
    void start();
    void stop();

    void set_camera_chunk(Vector3 camera_pos);
    // Takes the latest scan if there is a new one; r_camera_chunk is the camera chunk it was made for.
    bool get_scanned_chunks(std::unique_ptr<LODHashSet> &r_scanned_chunks, ChunkId &r_camera_chunk);

    // The full scan around `p_camera_chunk`. Const and thread safe: the discovery audit calls it from
    // the main thread to rebuild the published scan without culling.
    // `p_cache` (optional) remembers each node's verdict across scans.
    // Breadth first, one wave per LOD from the top-LOD grid down: the wave's verdicts (_cull_wave), then
    // its surviving nodes are kept or split into the next wave by distance. The scanned set doesn't
    // depend on the visit order, so it is the one the old recursive walk gave (unit-tested).
    void scan(const ChunkId &p_camera_chunk, CullMode p_mode, LODHashSet &r_chunks, ScanCounters &r_counters,
            CullCache *p_cache = nullptr) const;
    int get_max_lod() const { return lod_distances.size() - 2; }
    const PackedFloat32Array &get_lod_distances() const { return lod_distances; }
    // The scan grid (after setup_lod): top-LOD chunk centres step `r_step` ids, from -r_count to
    // +r_count steps around the camera's top-LOD ancestor.
    void get_scan_grid(int64_t &r_step, int64_t &r_count) const;

    Dictionary get_stats();

    // Must be called before start().
    void set_config(
            ChunkParam &p_chunk_param,
            SDFBase *p_sdf);
    void set_cull_mode(CullMode p_mode) { cull_mode = p_mode; }
    void set_cache(bool p_enable, bool p_verify) {
        use_cache = p_enable;
        cache.verify = p_verify;
    }

    void discovery_loop();

    void setup_lod(
            int lod_levels = 4,
            float target_pixel_error = 10.0f,
            float viewport_height = 1080.0f,
            float fov_rad = 75.0f * (3.14159265f / 180.0f));
};
