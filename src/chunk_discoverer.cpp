#include "chunk_discoverer.hpp"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <chrono>
#include <vector>

using namespace godot;

void ChunkDiscoverer::_corner_verdicts(const std::vector<ChunkId> &p_wave, const std::vector<uint32_t> &p_nodes, int p_lod,
        std::vector<uint8_t> &r_outside, WaveStats &r_wave, DiscovererGPU *p_gpu) const {
    // The 8 corners of each node (centre +- half its size, points of Λ_lod), shared by up to 8 nodes:
    // deduplicated on their voxel, then evaluated in one batch.
    const int64_t half_voxels = (int64_t(1) << p_lod) * chunk_param.half_chunk_voxels();
    HashMap<VoxelPos, uint32_t, Vector3i64Hasher> index;
    index.reserve(uint32_t(p_nodes.size() * 2)); // ~1.3 unique corners per node (dedup ratio ~6)
    std::vector<VoxelPos> points;
    std::vector<uint32_t> corner_point(p_nodes.size() * 8);
    for (size_t n = 0; n < p_nodes.size(); n++) {
        const VoxelPos center = ChunkMath::chunk_center_voxel(p_wave[p_nodes[n]], chunk_param);
        for (int corner = 0; corner < 8; ++corner) {
            const VoxelPos p = center + VoxelPos((corner & 1) ? half_voxels : -half_voxels, (corner & 2) ? half_voxels : -half_voxels, (corner & 4) ? half_voxels : -half_voxels);
            const uint32_t *known = index.getptr(p);
            if (known) {
                corner_point[n * 8 + corner] = *known;
            } else {
                corner_point[n * 8 + corner] = uint32_t(points.size());
                index.insert(p, uint32_t(points.size()));
                points.push_back(p);
            }
        }
    }
    std::vector<float> values(points.size());
    const uint64_t t_eval = now_usec();
    DiscovererGPU::BatchStats gpu_stats;
    r_wave.gpu = p_gpu && p_gpu->evaluate(points.data(), int(points.size()), p_lod, values.data(), gpu_stats);
    if (r_wave.gpu) {
        r_wave.batches = gpu_stats.batches;
        r_wave.gpu_ms = gpu_stats.gpu_ms;
        r_wave.readback_bytes = gpu_stats.readback_bytes;
    } else {
        sdf->evaluate_batch(points.data(), int(points.size()), p_lod, values.data());
    }
    r_wave.eval_usec = now_usec() - t_eval;
    r_wave.evaluated = int64_t(p_nodes.size());
    r_wave.corners = int64_t(p_nodes.size()) * 8;
    r_wave.unique_corners = int64_t(points.size());

    // Outside iff |SDF| > 2 chunk sizes at all 8 corners: the old per-node test, which stopped at the first
    // near corner, so the same verdict.
    const float chunk_outer_radius = ChunkMath::world_chunk_size(p_wave[p_nodes[0]], chunk_param) * 2;
    r_outside.assign(p_nodes.size(), 1);
    for (size_t n = 0; n < p_nodes.size(); n++) {
        for (int corner = 0; corner < 8; ++corner) {
            if (std::abs(values[corner_point[n * 8 + corner]]) <= chunk_outer_radius) {
                r_outside[n] = 0;
                break;
            }
        }
    }
}

void ChunkDiscoverer::_cull_wave(const std::vector<ChunkId> &p_wave, int p_lod, CullMode p_mode, std::vector<uint8_t> &r_outside,
        ScanCounters &r_counters, CullCache *p_cache, DiscovererGPU *p_gpu) const {
    WaveStats wave;
    wave.lod = p_lod;
    wave.nodes = int64_t(p_wave.size());
    r_outside.assign(p_wave.size(), 0);

    // Cached verdicts first; the other nodes (and every hit with verify) get a fresh one.
    std::vector<uint32_t> fresh;
    std::vector<uint8_t> cached(p_wave.size(), 0);
    for (uint32_t i = 0; i < uint32_t(p_wave.size()); i++) {
        CullCache::Entry *entry = p_cache ? p_cache->entries.getptr(p_wave[i]) : nullptr;
        if (entry) {
            r_counters.cache_hits++;
            entry->last_scan = p_cache->scan_index;
            cached[i] = 1;
            r_outside[i] = entry->outside;
            if (!p_cache->verify) {
                continue;
            }
        }
        fresh.push_back(i);
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> verdict(fresh.size(), 0);
    if (p_mode != CULL_NONE && !fresh.empty()) {
        _corner_verdicts(p_wave, fresh, p_lod, verdict, wave, p_gpu);
    }
    r_counters.cull_nsec += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    r_counters.sdf_calls += wave.unique_corners;
    r_counters.corners += wave.corners;

    for (size_t f = 0; f < fresh.size(); f++) {
        const uint32_t i = fresh[f];
        if (cached[i]) {
            r_counters.verify_mismatches += verdict[f] != r_outside[i] ? 1 : 0;
            continue;
        }
        r_outside[i] = verdict[f];
        if (p_cache) {
            CullCache::Entry e;
            e.outside = verdict[f];
            e.last_scan = p_cache->scan_index;
            p_cache->entries.insert(p_wave[i], e);
        }
    }
    r_counters.waves.push_back(wave);
}

bool ChunkDiscoverer::_should_split(const ChunkId &p_camera_chunk, const ChunkId &p_node, int p_lod) const {
    const double distance_to_player = p_camera_chunk.distance_to(p_node);

    // Work fully in chunk-coordinate units.
    // Half-extent of current LOD chunk in chunk-coord units is 2^lod.
    const float half_extent = static_cast<float>(int64_t(1) << p_lod);
    // Bounding sphere radius of the chunk cube in chunk-coord units.
    const float chunk_outer_radius = std::sqrt(3.0f) * half_extent;

    // lod_distances[L] is the distance from which LOD L is within the pixel error: split when closer.
    return distance_to_player - chunk_outer_radius < lod_distances[p_lod];
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

    // The GPU discoverer's local device lives on this thread only. Creating it (device + the point
    // shader's compile) delays the first scan: gpu_init_ms.
    const uint64_t t_init = now_usec();
    if (want_gpu) {
        gpu = memnew(DiscovererGPU);
        if (!gpu->create(sdf, chunk_param)) {
            WARN_PRINT("ChunkDiscoverer: the GPU discoverer needs a cascade terrain and a local RenderingDevice: the corners are evaluated on the CPU.");
            memdelete(gpu);
            gpu = nullptr;
        }
    }
    mutex->lock();
    stats.gpu_active = gpu != nullptr;
    stats.gpu_scratch_mb = gpu ? double(gpu->scratch_bytes()) / 1048576.0 : 0.0;
    stats.gpu_init_usec = want_gpu ? now_usec() - t_init : 0;
    mutex->unlock();

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
        scan(actual_camera_chunk, cull_mode, *new_scanned_chunk, counters, use_cache ? &cache : nullptr, gpu);
        const int64_t evicted = _evict_cache(counters.nodes_visited);
        if (counters.verify_mismatches > 0) {
            ERR_PRINT(String("ChunkDiscoverer: ") + String::num_int64(counters.verify_mismatches) +
                    " cached cull verdicts differ from a fresh evaluation.");
        }
        const uint64_t t_scan_end = now_usec();
        if (compare) {
            _compare_with_cpu(actual_camera_chunk, *new_scanned_chunk);
        }
        const uint64_t t_compared = now_usec();

        _publish(std::move(new_scanned_chunk), actual_camera_chunk, counters, evicted, t_scan_end - t_scan_start, t_compared);
    }

    if (gpu) {
        gpu->free();
        memdelete(gpu);
        gpu = nullptr;
    }
}

int64_t ChunkDiscoverer::_evict_cache(int64_t p_nodes_visited) {
    // Forget the verdicts of nodes no scan has visited for a while, once the cache is well
    // past the size of one scan (flying keeps adding new nodes).
    if (!use_cache || cache.entries.size() <= uint32_t(4 * p_nodes_visited)) {
        return 0;
    }
    std::vector<ChunkId> stale;
    for (const KeyValue<ChunkId, CullCache::Entry> &kv : cache.entries) {
        if (kv.value.last_scan + 4 < cache.scan_index) {
            stale.push_back(kv.key);
        }
    }
    for (const ChunkId &id : stale) {
        cache.entries.erase(id);
    }
    return (int64_t)stale.size();
}

void ChunkDiscoverer::_publish(std::unique_ptr<LODHashSet> p_scan, const ChunkId &p_camera_chunk, const ScanCounters &p_counters, int64_t p_evicted,
        uint64_t p_scan_usec, uint64_t p_ready_usec) {
    mutex->lock();
    const int64_t size = p_scan->size();
    scanned_chunks = std::move(p_scan);
    scanned_camera_chunk = p_camera_chunk;
    scanned_chunks_dirty = true;
    const uint64_t t_published = now_usec();
    stats.scan.add(p_scan_usec);
    stats.cull.add(p_counters.cull_nsec / 1000);
    stats.publish.add(t_published - p_ready_usec);
    if (camera_change_usec > 0) {
        stats.latency.add(t_published - camera_change_usec);
    }
    stats.last_scan_size = size;
    stats.last_nodes_visited = p_counters.nodes_visited;
    stats.last_sdf_calls = p_counters.sdf_calls;
    stats.last_culled = p_counters.culled;
    stats.nodes_visited_total += p_counters.nodes_visited;
    stats.sdf_calls_total += p_counters.sdf_calls;
    stats.last_cache_hits = p_counters.cache_hits;
    stats.cache_hits_total += p_counters.cache_hits;
    stats.cache_size = cache.entries.size();
    stats.cache_evicted_total += p_evicted;
    stats.verify_mismatches_total += p_counters.verify_mismatches;
    stats.last_scan = _scan_dict(p_counters, p_scan_usec);
    if (stats.first_scan.is_empty()) {
        stats.first_scan = stats.last_scan;
    }
    mutex->unlock();
}

void ChunkDiscoverer::scan(const ChunkId &p_camera_chunk, CullMode p_mode, LODHashSet &r_chunks, ScanCounters &r_counters,
        CullCache *p_cache, DiscovererGPU *p_gpu) const {
    const int max_lod = get_max_lod();
    int64_t max_lod_radius;
    int64_t max_lod_number;
    get_scan_grid(max_lod_radius, max_lod_number);

    const ChunkId ancestor_chunk = ChunkMath::get_parent_from_child_until(p_camera_chunk, max_lod);

    const ChunkId search_space = ChunkId(max_lod_number, max_lod_number, max_lod_number) * max_lod_radius;

    const ChunkId start_scan = ancestor_chunk - search_space;

    const ChunkId end_scan = ancestor_chunk + search_space;

    std::vector<ChunkId> wave;
    for (int64_t x = start_scan.x; x <= end_scan.x; x += max_lod_radius) {
        for (int64_t y = start_scan.y; y <= end_scan.y; y += max_lod_radius) {
            for (int64_t z = start_scan.z; z <= end_scan.z; z += max_lod_radius) {
                wave.push_back(ChunkId(x, y, z));
            }
        }
    }

    std::vector<ChunkId> next;
    std::vector<uint8_t> outside;
    for (int lod = max_lod; !wave.empty(); lod--) {
        _cull_wave(wave, lod, p_mode, outside, r_counters, p_cache, p_gpu);
        next.clear();
        for (size_t i = 0; i < wave.size(); i++) {
            const ChunkId &node = wave[i];
            r_counters.nodes_visited++;
            if (outside[i]) {
                r_counters.culled++;
                continue;
            }
            if (lod == 0 || !_should_split(p_camera_chunk, node, lod)) {
                r_chunks.insert(node);
                continue;
            }
            const int64_t child_size = int64_t(1) << (lod - 1);
            for (int c = 0; c < 8; ++c) {
                next.push_back(node + ChunkId((c & 1) ? child_size : -child_size, (c & 2) ? child_size : -child_size, (c & 4) ? child_size : -child_size));
            }
        }
        wave.swap(next);
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
void ChunkDiscoverer::_compare_with_cpu(const ChunkId &p_camera_chunk, const LODHashSet &p_scanned) {
    const uint64_t t0 = now_usec();
    LODHashSet cpu_set;
    ScanCounters cpu_counters;
    scan(p_camera_chunk, cull_mode, cpu_set, cpu_counters);
    int64_t only_here = 0;
    int64_t only_cpu = 0;
    for (const ChunkId &id : p_scanned) {
        only_here += cpu_set.has(id) ? 0 : 1;
    }
    for (const ChunkId &id : cpu_set) {
        only_cpu += p_scanned.has(id) ? 0 : 1;
    }
    if (only_here + only_cpu > 0) {
        ERR_PRINT(vformat("ChunkDiscoverer: --discovery_compare: the %s scan around %s differs from the CPU scan: %d chunks only in it, %d only in the CPU's.",
                gpu ? "GPU" : "CPU", WorldCoord::to_string(p_camera_chunk), only_here, only_cpu));
    }
    mutex->lock();
    stats.compare_scans++;
    stats.compare_differences += only_here + only_cpu;
    stats.compare_cpu.add(now_usec() - t0);
    mutex->unlock();
}

Dictionary ChunkDiscoverer::_scan_dict(const ScanCounters &p_counters, uint64_t p_scan_usec) {
    Dictionary d;
    const double cull_ms = double(p_counters.cull_nsec) / 1e6;
    d["scan_ms"] = double(p_scan_usec) / 1000.0;
    d["cull_ms"] = cull_ms;
    d["walk_ms"] = double(p_scan_usec) / 1000.0 - cull_ms; // everything else: the tree walk, cache lookups and eviction
    d["nodes_visited"] = p_counters.nodes_visited;
    d["culled"] = p_counters.culled;
    d["cache_hits"] = p_counters.cache_hits;
    d["corners"] = p_counters.corners;
    d["sdf_calls"] = p_counters.sdf_calls;
    d["dedup_ratio"] = p_counters.sdf_calls ? double(p_counters.corners) / double(p_counters.sdf_calls) : 0.0;
    Array waves;
    for (const WaveStats &w : p_counters.waves) {
        Dictionary wd;
        wd["lod"] = w.lod;
        wd["nodes"] = w.nodes;
        wd["evaluated"] = w.evaluated;
        wd["corners"] = w.corners;
        wd["unique_corners"] = w.unique_corners;
        wd["eval_ms"] = double(w.eval_usec) / 1000.0;
        if (w.gpu) {
            wd["gpu_ms"] = w.gpu_ms;
            wd["batches"] = w.batches;
            wd["readback_bytes"] = w.readback_bytes;
        }
        waves.push_back(wd);
    }
    d["waves"] = waves;
    return d;
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
    d["mode"] = stats.gpu_active ? "gpu" : "cpu";
    d["gpu_scratch_mb"] = stats.gpu_scratch_mb;
    d["gpu_init_ms"] = double(stats.gpu_init_usec) / 1000.0;
    d["compare"] = compare;
    d["compare_scans"] = stats.compare_scans;
    d["compare_differences"] = stats.compare_differences;
    d["compare_cpu"] = stats.compare_cpu.to_dict();
    d["first_scan"] = stats.first_scan.duplicate(true);
    d["last_scan"] = stats.last_scan.duplicate(true);
    mutex->unlock();
    return d;
}
