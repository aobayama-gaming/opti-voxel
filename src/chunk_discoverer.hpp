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

using namespace godot;

class ChunkDiscoverer : public Object {
public:
    // How a node of the LOD tree is dropped as holding no surface.
    enum CullMode {
        CULL_NONE = 0, // keep everything (reference set of the discovery audit)
        CULL_LEGACY = 1, // |SDF| at the 8 corners > 2 chunk sizes
    };

    // Work done by one scan.
    struct ScanCounters {
        int64_t nodes_visited = 0;
        int64_t sdf_calls = 0;
        int64_t culled = 0;
        uint64_t cull_nsec = 0; // time spent in the cull test (SDF evaluations), summed in ns (a node can take < 1 us)
        int64_t cache_hits = 0;
        int64_t verify_mismatches = 0;
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
    } stats;

    SDFBase *sdf;

    bool _chunk_is_outside_sdf(const ChunkId &chunk_pos, CullMode p_mode, ScanCounters &r_counters) const;
    bool _timed_cull(const ChunkId &chunk_pos, CullMode p_mode, ScanCounters &r_counters) const;
    void _recursive_chunk_scan(const ChunkId &camera_chunk, const ChunkId &parent_chunk, CullMode p_mode,
            LODHashSet &scanned_chunks, ScanCounters &r_counters, CullCache *p_cache) const;

public:
    void start();
    void stop();

    void set_camera_chunk(Vector3 camera_pos);
    // Takes the latest scan if there is a new one; r_camera_chunk is the camera chunk it was made for.
    bool get_scanned_chunks(std::unique_ptr<LODHashSet> &r_scanned_chunks, ChunkId &r_camera_chunk);

    // The full scan around `p_camera_chunk`. Const and thread safe: the discovery audit calls it from
    // the main thread to rebuild the published scan without culling.
    // `p_cache` (optional) remembers each node's verdict across scans.
    void scan(const ChunkId &p_camera_chunk, CullMode p_mode, LODHashSet &r_chunks, ScanCounters &r_counters,
            CullCache *p_cache = nullptr) const;
    int get_max_lod() const { return lod_distances.size() - 2; }
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
