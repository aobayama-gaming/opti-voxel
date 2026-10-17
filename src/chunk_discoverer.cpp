#include "chunk_discoverer.hpp"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <chrono>
#include <vector>

using namespace godot;

bool ChunkDiscoverer::_chunk_is_outside_sdf(const ChunkId &chunk_pos, CullMode p_mode, ScanCounters &r_counters) const {
    if (p_mode == CULL_NONE) {
        return false;
    }

    const float chunk_size = ChunkMath::world_chunk_size(chunk_pos, chunk_param);
    const float chunk_outer_radius = chunk_size * 2; //0.8660254f // 2 is extra margin
    const VoxelPos chunk_center = ChunkMath::chunk_center_voxel(chunk_pos, chunk_param);
    // Half the chunk size in voxels: the corners are at the centre +- this.
    const int64_t half_voxels = ChunkMath::get_lod_size(chunk_pos) * chunk_param.half_chunk_voxels();
    const int lod = ChunkMath::get_lod(chunk_pos);

    for (int corner = 0; corner < 8; ++corner) {
        WorldPos corner_pos;
        corner_pos.voxel = chunk_center + VoxelPos((corner & 1) ? half_voxels : -half_voxels, (corner & 2) ? half_voxels : -half_voxels, (corner & 4) ? half_voxels : -half_voxels);

        const float corner_distance = std::abs(sdf->evaluate(corner_pos, lod));
        r_counters.sdf_calls++;
        if (corner_distance <= chunk_outer_radius) {
            return false;
        }
    }

    return true;
}

bool ChunkDiscoverer::_timed_cull(const ChunkId &chunk_pos, CullMode p_mode, ScanCounters &r_counters) const {
    const auto t0 = std::chrono::steady_clock::now();
    const bool outside = _chunk_is_outside_sdf(chunk_pos, p_mode, r_counters);
    r_counters.cull_usec += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    return outside;
}

void ChunkDiscoverer::start() {
    worker.instantiate();
    mutex.instantiate();
    semaphore.instantiate();

    worker->start(callable_mp(this, &ChunkDiscoverer::discovery_loop), Thread::PRIORITY_NORMAL);
}

void ChunkDiscoverer::stop() {
    mutex->lock();
    exit_thread = true;
    mutex->unlock();

    semaphore->post();

    if (worker.is_valid()) {
        worker->wait_to_finish();
    }

    worker.unref();
}

void ChunkDiscoverer::discovery_loop() {
    bool has_scanned = false;
    ChunkId last_scanned_camera_chunk;

    while (true) {
        semaphore->wait();
        // One post per camera chunk crossing: posts that piled up during the last scan all ask for
        // the same thing, a scan around the latest camera chunk.
        while (semaphore->try_wait()) {
        }

        mutex->lock();
        bool should_exit = exit_thread;

        ChunkId actual_camera_chunk = camera_chunk;
        // Add other variable import.
        mutex->unlock();

        if (should_exit) {
            break;
        }

        if (has_scanned && actual_camera_chunk == last_scanned_camera_chunk) {
            mutex->lock();
            stats.scans_skipped_same_chunk++;
            mutex->unlock();
            continue;
        }
        has_scanned = true;
        last_scanned_camera_chunk = actual_camera_chunk;

        const uint64_t t_scan_start = now_usec();
        std::unique_ptr<LODHashSet> new_scanned_chunk = std::make_unique<LODHashSet>();
        ScanCounters counters;
        cache.scan_index++;
        scan(actual_camera_chunk, cull_mode, *new_scanned_chunk, counters, use_cache ? &cache : nullptr);
        // Forget the verdicts of nodes no scan has visited for a while, once the cache is well
        // past the size of one scan (flying keeps adding new nodes).
        int64_t evicted = 0;
        if (use_cache && cache.entries.size() > uint32_t(4 * counters.nodes_visited)) {
            std::vector<ChunkId> stale;
            for (const KeyValue<ChunkId, CullCache::Entry> &kv : cache.entries) {
                if (kv.value.last_scan + 4 < cache.scan_index) {
                    stale.push_back(kv.key);
                }
            }
            for (const ChunkId &id : stale) {
                cache.entries.erase(id);
            }
            evicted = (int64_t)stale.size();
        }
        if (counters.verify_mismatches > 0) {
            ERR_PRINT(String("ChunkDiscoverer: ") + String::num_int64(counters.verify_mismatches) +
                    " cached cull verdicts differ from a fresh evaluation.");
        }
        const uint64_t t_scan_end = now_usec();

        mutex->lock();
        const int64_t size = new_scanned_chunk->size();
        scanned_chunks = std::move(new_scanned_chunk);
        scanned_camera_chunk = actual_camera_chunk;
        scanned_chunks_dirty = true;
        const uint64_t t_published = now_usec();
        stats.scan.add(t_scan_end - t_scan_start);
        stats.cull.add(counters.cull_usec);
        stats.publish.add(t_published - t_scan_end);
        if (camera_change_usec > 0) {
            stats.latency.add(t_published - camera_change_usec);
        }
        stats.last_scan_size = size;
        stats.last_nodes_visited = counters.nodes_visited;
        stats.last_sdf_calls = counters.sdf_calls;
        stats.last_culled = counters.culled;
        stats.nodes_visited_total += counters.nodes_visited;
        stats.sdf_calls_total += counters.sdf_calls;
        stats.last_cache_hits = counters.cache_hits;
        stats.cache_hits_total += counters.cache_hits;
        stats.cache_size = cache.entries.size();
        stats.cache_evicted_total += evicted;
        stats.verify_mismatches_total += counters.verify_mismatches;
        mutex->unlock();
    }
}

void ChunkDiscoverer::scan(const ChunkId &p_camera_chunk, CullMode p_mode, LODHashSet &r_chunks, ScanCounters &r_counters,
        CullCache *p_cache) const {
    const int max_lod = get_max_lod();
    int64_t max_lod_radius;
    int64_t max_lod_number;
    get_scan_grid(max_lod_radius, max_lod_number);

    const ChunkId ancestor_chunk = ChunkMath::get_parent_from_child_until(p_camera_chunk, max_lod);

    const ChunkId search_space = ChunkId(max_lod_number, max_lod_number, max_lod_number) * max_lod_radius;

    const ChunkId start_scan = ancestor_chunk - search_space;

    const ChunkId end_scan = ancestor_chunk + search_space;

    for (int64_t x = start_scan.x; x <= end_scan.x; x += max_lod_radius) {
        for (int64_t y = start_scan.y; y <= end_scan.y; y += max_lod_radius) {
            for (int64_t z = start_scan.z; z <= end_scan.z; z += max_lod_radius) {
                _recursive_chunk_scan(p_camera_chunk, ChunkId(x, y, z), p_mode, r_chunks, r_counters, p_cache);
            }
        }
    }
}

void ChunkDiscoverer::get_scan_grid(int64_t &r_step, int64_t &r_count) const {
    const int lod_count = lod_distances.size() - 1;

    const int max_lod = lod_count - 1; // The maximum LOD level to consider for loading. This is determined by the size of the lod_distances array, which should be set up so that each LOD level corresponds to a distance threshold.

    r_step = int64_t(2) << (max_lod); // The radius of the largest LOD. This is the maximum distance at which chunks will be loaded, so it determines the size of the area to scan.

    r_count = MIN(int64_t(std::ceil(lod_distances[lod_count - 1] / double(r_step))) + 2, int64_t(max_lod));
}

void ChunkDiscoverer::set_config(
        ChunkParam &p_chunk_param,
        SDFBase *p_sdf) {
    sdf = p_sdf;
    chunk_param = p_chunk_param;
}

bool ChunkDiscoverer::get_scanned_chunks(std::unique_ptr<LODHashSet> &r_scanned_chunks, ChunkId &r_camera_chunk) {
    mutex->lock();

    bool dirty = scanned_chunks_dirty;

    if (dirty) {
        r_scanned_chunks = std::move(scanned_chunks);
        r_camera_chunk = scanned_camera_chunk;
        scanned_chunks_dirty = false;
    }

    mutex->unlock();

    return dirty;
}

void ChunkDiscoverer::setup_lod(
        int lod_levels,
        float target_pixel_error,
        float viewport_height,
        float fov_rad) {
    const float focal_length = viewport_height / (2.0f * std::tan(fov_rad * 0.5f));

    lod_distances.resize(lod_levels + 1);

    float previous_distance = 0.0f;
    for (int i = 0; i < lod_levels + 1; ++i) {
        // Geometric error follows voxel size per LOD (voxel doubles each level).
        const float lod_voxel_size_world = chunk_param.voxel_size * static_cast<float>(int64_t(1) << i);
        const float geometric_error_world = lod_voxel_size_world * 0.5f;

        // Compute threshold in world units, then convert to chunk-coordinate units.
        // One chunk coordinate step equals CHUNK_DIMENSION / 2 in world units.
        float dist_world = (geometric_error_world * focal_length) / target_pixel_error;
        float dist_chunk = dist_world / (chunk_param.chunk_world_dimension() * 0.5f);

        // Keep thresholds strictly increasing and avoid near-zero first level.
        if (i == 0 && dist_chunk < 1.0f) {
            dist_chunk = 1.0f;
        }
        if (dist_chunk <= previous_distance) {
            dist_chunk = previous_distance + 1.0f;
        }

        lod_distances.set(i, dist_chunk);
        previous_distance = dist_chunk;
    }
}

void ChunkDiscoverer::_recursive_chunk_scan(const ChunkId &camera_chunk, const ChunkId &parent_chunk, CullMode p_mode,
        LODHashSet &scanned_chunks, ScanCounters &r_counters, CullCache *p_cache) const {
    const int current_lod = ChunkMath::get_lod(parent_chunk);
    r_counters.nodes_visited++;

    bool outside;
    CullCache::Entry *cached = p_cache ? p_cache->entries.getptr(parent_chunk) : nullptr;
    if (cached) {
        r_counters.cache_hits++;
        cached->last_scan = p_cache->scan_index;
        outside = cached->outside;
        if (p_cache->verify && _timed_cull(parent_chunk, p_mode, r_counters) != outside) {
            r_counters.verify_mismatches++;
        }
    } else {
        outside = _timed_cull(parent_chunk, p_mode, r_counters);
        if (p_cache) {
            CullCache::Entry e;
            e.outside = outside;
            e.last_scan = p_cache->scan_index;
            p_cache->entries.insert(parent_chunk, e);
        }
    }
    if (outside) {
        r_counters.culled++;
        return;
    }

    if (current_lod == 0) {
        scanned_chunks.insert(parent_chunk);
        return;
    }

    const double distance_to_player = camera_chunk.distance_to(parent_chunk);

    // Work fully in chunk-coordinate units.
    // Half-extent of current LOD chunk in chunk-coord units is 2^lod.
    const float half_extent = static_cast<float>(int64_t(1) << current_lod);
    // Bounding sphere radius of the chunk cube in chunk-coord units.
    const float chunk_outer_radius = std::sqrt(3.0f) * half_extent;

    // lod_distances[L] is the distance from which LOD L is within the pixel error: split when closer.
    if (distance_to_player - chunk_outer_radius < lod_distances[current_lod]) {
        const int child_lod = current_lod - 1;
        const int64_t child_size = int64_t(1) << child_lod;

        for (int i = 0; i < 8; ++i) {
            ChunkId child_offset(
                    (i & 1) ? child_size : -child_size,
                    (i & 2) ? child_size : -child_size,
                    (i & 4) ? child_size : -child_size);

            _recursive_chunk_scan(
                    camera_chunk,
                    parent_chunk + child_offset,
                    p_mode,
                    scanned_chunks,
                    r_counters,
                    p_cache);
        }
    } else {
        scanned_chunks.insert(parent_chunk);
    }
}

void ChunkDiscoverer::set_camera_chunk(Vector3 camera_pos) {
    /*
    Function to set the camera chunk in a thread-safe manner.
    */

    mutex->lock();
    ChunkId old_camera_chunk = camera_chunk;
    mutex->unlock();

    ChunkId new_camera_chunk = ChunkMath::chunk_at_render(camera_pos, chunk_param);

    if (new_camera_chunk != old_camera_chunk) {
        mutex->lock();
        camera_chunk = new_camera_chunk;
        camera_change_usec = now_usec();
        mutex->unlock();

        semaphore->post();

        VLOG(2, "new chunk:", WorldCoord::to_string(new_camera_chunk));
    }
}
Dictionary ChunkDiscoverer::get_stats() {
    Dictionary d;
    mutex->lock();
    d["scan"] = stats.scan.to_dict();
    d["cull"] = stats.cull.to_dict();
    d["publish"] = stats.publish.to_dict();
    d["latency"] = stats.latency.to_dict();
    d["cull_mode"] = (int)cull_mode;
    d["scans_skipped_same_chunk"] = stats.scans_skipped_same_chunk;
    d["last_scan_size"] = stats.last_scan_size;
    d["last_nodes_visited"] = stats.last_nodes_visited;
    d["last_sdf_calls"] = stats.last_sdf_calls;
    d["last_culled"] = stats.last_culled;
    d["nodes_visited_total"] = stats.nodes_visited_total;
    d["sdf_calls_total"] = stats.sdf_calls_total;
    d["cache"] = use_cache;
    d["verify"] = cache.verify;
    d["last_cache_hits"] = stats.last_cache_hits;
    d["cache_hits_total"] = stats.cache_hits_total;
    d["cache_size"] = stats.cache_size;
    d["cache_evicted_total"] = stats.cache_evicted_total;
    d["verify_mismatches_total"] = stats.verify_mismatches_total;
    // Share of visited nodes whose verdict came from the cache, and the cost of one corner SDF evaluation.
    d["cache_hit_rate"] = stats.nodes_visited_total ? double(stats.cache_hits_total) / double(stats.nodes_visited_total) : 0.0;
    d["usec_per_sdf_call"] = stats.sdf_calls_total ? double(stats.cull.total) / double(stats.sdf_calls_total) : 0.0;
    mutex->unlock();
    return d;
}
