#include "chunk_discoverer.hpp"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
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
        // No-cull rule (archetype instances): a node box near an instance's box is never culled, so a
        // steep, rotated or subtracted shape can't be dropped by the slope-bounded corner test. Asked only
        // when the corners would cull it.
        if (r_outside[n] && sdf->cull_exempt(p_wave[p_nodes[n]], p_lod)) {
            r_outside[n] = 0;
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

bool ChunkDiscoverer::_should_split(const ScanView &p_view, const ChunkId &p_node, int p_lod) const {
    // lod_distances[L] is the distance from which LOD L is within the pixel error: split when closer.
    return ChunkMath::lod_split(p_node, p_lod, p_view, lod_distances[p_lod]);
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
    ScanView last_scanned_view;
    // The previous scan, kept up to date by applying each delta (the published set leaves without a copy).
    LODHashSet previous_scan;

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

        const ScanView actual_view = view;
        const bool force_rescan = rescan_requested;
        rescan_requested = false;
        std::vector<VoxelBox> invalidations = std::move(pending_invalidations);
        pending_invalidations.clear();
        mutex->unlock();

        if (should_exit) {
            break;
        }

        // An instance edit: drop the cull verdicts of the nodes near the changed region before the scan
        // (the no-cull rule reads the current SDF, but the cache may hold a verdict from before).
        if (!invalidations.empty()) {
            std::vector<ChunkId> stale;
            for (const KeyValue<ChunkId, CullCache::Entry> &kv : cache.entries) {
                const int lod = ChunkMath::get_lod(kv.key);
                const int64_t span = int64_t(chunk_param.chunk_voxel_dimension()) << lod;
                const VoxelPos min = ChunkMath::chunk_min_voxel(kv.key, chunk_param);
                VoxelBox node;
                node.valid = true;
                const int64_t grow = 2 * span; // the cull band
                for (int a = 0; a < 3; a++) {
                    node.lo[a] = min[a] - grow;
                    node.hi[a] = min[a] + span + grow;
                }
                for (const VoxelBox &region : invalidations) {
                    if (node.intersects(region)) {
                        stale.push_back(kv.key);
                        break;
                    }
                }
            }
            for (const ChunkId &id : stale) {
                cache.entries.erase(id);
            }
        }

        // set_camera only posts for a view that changed: an equal one is a post that piled up. Without
        // back_scale the forward doesn't change the set: a new one alone is no reason to scan.
        if (has_scanned && !view_changed(last_scanned_view, actual_view) && !force_rescan) {
            mutex->lock();
            stats.scans_skipped_same_chunk++;
            mutex->unlock();
            continue;
        }
        has_scanned = true;
        last_scanned_view = actual_view;

        const uint64_t t_scan_start = now_usec();
        std::unique_ptr<LODHashSet> new_scanned_chunk = std::make_unique<LODHashSet>();
        ScanCounters counters;
        cache.scan_index++;
        scan(actual_view, cull_mode, *new_scanned_chunk, counters, use_cache ? &cache : nullptr, gpu);
        const int64_t evicted = _evict_cache(counters.nodes_visited);
        if (counters.verify_mismatches > 0) {
            ERR_PRINT(String("ChunkDiscoverer: ") + String::num_int64(counters.verify_mismatches) +
                    " cached cull verdicts differ from a fresh evaluation.");
        }
        const uint64_t t_scan_end = now_usec();
        if (compare) {
            _compare_with_cpu(actual_view, *new_scanned_chunk);
        }
        const uint64_t t_compared = now_usec();

        std::unique_ptr<ScanDelta> delta = std::make_unique<ScanDelta>();
        ScanDelta::between(previous_scan, *new_scanned_chunk, *delta);
        delta->apply(previous_scan);
        const uint64_t t_delta = now_usec();
        mutex->lock();
        stats.delta.add(t_delta - t_compared);
        stats.last_delta_entered = delta->entered.size();
        stats.last_delta_left = delta->left.size();
        stats.delta_entered_total += delta->entered.size();
        stats.delta_left_total += delta->left.size();
        mutex->unlock();

        _publish(std::move(new_scanned_chunk), std::move(delta), actual_view, counters, evicted, t_scan_end - t_scan_start, t_delta);
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

void ChunkDiscoverer::_publish(std::unique_ptr<LODHashSet> p_scan, std::unique_ptr<ScanDelta> p_delta, const ScanView &p_view,
        const ScanCounters &p_counters, int64_t p_evicted, uint64_t p_scan_usec, uint64_t p_ready_usec) {
    mutex->lock();
    const int64_t size = p_scan->size();
    scanned_chunks = std::move(p_scan);
    // The main thread hasn't taken the previous scan: its delta still leads from the last scan taken.
    if (scanned_chunks_dirty) {
        scanned_delta->merge(*p_delta);
        stats.deltas_merged++;
    } else {
        scanned_delta = std::move(p_delta);
    }
    scanned_view = p_view;
    scanned_chunks_dirty = true;
    const uint64_t t_published = now_usec();
    stats.scan.add(p_scan_usec);
    stats.cull.add(p_counters.cull_nsec / 1000);
    stats.publish.add(t_published - p_ready_usec);
    if (camera_change_usec > 0) {
        stats.latency.add(t_published - camera_change_usec);
    }
    stats.last_scan_size = size;
    stats.last_out_of_view = p_counters.out_of_view;
    stats.last_memories = p_view.memory_count;
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

void ChunkDiscoverer::scan(const ScanView &p_view, CullMode p_mode, LODHashSet &r_chunks, ScanCounters &r_counters,
        CullCache *p_cache, DiscovererGPU *p_gpu) const {
    const int max_lod = get_max_lod();
    int64_t max_lod_radius;
    int64_t max_lod_number;
    get_scan_grid(max_lod_radius, max_lod_number);

    const ChunkId ancestor_chunk = ChunkMath::get_parent_from_child_until(p_view.camera_chunk, max_lod);

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
            if (lod == 0 || !_should_split(p_view, node, lod)) {
                r_chunks.insert(node);
                r_counters.out_of_view += ChunkMath::view_cone_gap(node, lod, p_view) > 0.0 ? 1 : 0;
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

bool ChunkDiscoverer::get_scanned_chunks(std::unique_ptr<LODHashSet> &r_scanned_chunks, std::unique_ptr<ScanDelta> &r_delta, ScanView &r_view) {
    mutex->lock();

    bool dirty = scanned_chunks_dirty;

    if (dirty) {
        r_scanned_chunks = std::move(scanned_chunks);
        r_delta = std::move(scanned_delta);
        r_view = scanned_view;
        scanned_chunks_dirty = false;
    }

    mutex->unlock();

    return dirty;
}

void ChunkDiscoverer::setup_lod(
        int lod_levels,
        float target_pixel_error,
        float viewport_height,
        float fov_rad,
        float target_pixel_error_back,
        float cone_half_angle_rad) {
    cone_half_angle = cone_half_angle_rad;
    // The LOD distances scale as 1 / error: far from the cone, (1 + back_scale) times the distance meets
    // the back error's thresholds.
    back_scale = target_pixel_error_back > target_pixel_error ? double(target_pixel_error_back) / double(target_pixel_error) - 1.0 : 0.0;
    if (back_scale > 3.0) {
        // Ratio 4 keeps face neighbours within one LOD at 720p from error 1 or 2; 8 at error 2 gave jumps of 2.
        WARN_PRINT(vformat("ChunkDiscoverer: lod_pixel_error_back is %.1fx lod_pixel_error: past 4x, face neighbours can be 2 LODs apart (holes). Check with --discovery_audit=1.",
                back_scale + 1.0));
    }
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

void ChunkDiscoverer::invalidate(const VoxelBox &p_box) {
    if (!p_box.valid) {
        return;
    }
    mutex->lock();
    pending_invalidations.push_back(p_box);
    rescan_requested = true;
    mutex->unlock();
    semaphore->post();
}

ScanView ChunkDiscoverer::get_view() {
    mutex->lock();
    const ScanView v = view;
    mutex->unlock();
    return v;
}

ScanView ChunkDiscoverer::make_view(const Vector3 &p_pos, const Vector3 &p_forward) const {
    ScanView v;
    v.camera_chunk = ChunkMath::chunk_at_render(p_pos, chunk_param);
    v.forward = p_forward.length_squared() > 0.0f ? p_forward.normalized() : Vector3();
    v.set_half_angle(cone_half_angle);
    v.back_scale = back_scale;
    return v;
}

bool ChunkDiscoverer::turned_past(const Vector3 &p_a, const Vector3 &p_b) {
    static const double cos_turn = std::cos(Math::deg_to_rad(RESCAN_TURN_DEG));
    return p_a.dot(p_b) < cos_turn;
}

bool ChunkDiscoverer::view_changed(const ScanView &p_scanned, const ScanView &p_new) {
    if (p_new.camera_chunk != p_scanned.camera_chunk || p_new.memory_count != p_scanned.memory_count) {
        return true;
    }
    // Memories are copied from view to view: an unchanged one is bit for bit the same.
    for (int i = 0; i < p_new.memory_count; i++) {
        const ScanView::ViewMemory &a = p_new.memories[i];
        const ScanView::ViewMemory &b = p_scanned.memories[i];
        if (a.apex != b.apex || a.dir != b.dir) {
            return true;
        }
    }
    return false;
}

ChunkDiscoverer::Remembered ChunkDiscoverer::remember_view(ScanView &r_view, double p_memory_distance) {
    if (r_view.back_scale == 0.0 || r_view.forward == Vector3()) {
        r_view.memory_count = 0; // the set doesn't depend on the direction
        return REMEMBERED_KNOWN;
    }
    // Without memory distance, only the camera chunk's own view is kept.
    const int capacity = p_memory_distance > 0.0 ? ScanView::MAX_VIEW_MEMORIES : 1;
    const uint32_t now = ++r_view.memory_clock;
    for (int i = 0; i < r_view.memory_count; i++) {
        ScanView::ViewMemory &m = r_view.memories[i];
        m.fade = MAX(0.0, r_view.camera_chunk.distance_to(m.apex) - p_memory_distance);
    }
    Remembered result = REMEMBERED_ADDED;
    int slot = -1;
    for (int i = 0; i < r_view.memory_count && slot < 0; i++) {
        if (r_view.memories[i].apex == r_view.camera_chunk && !turned_past(r_view.memories[i].dir, r_view.forward)) {
            slot = i;
            result = REMEMBERED_KNOWN;
        }
    }
    if (result == REMEMBERED_KNOWN) {
        r_view.memories[slot].used = now;
    } else {
        slot = r_view.memory_count;
        if (slot >= capacity) {
            // Over the most faded one, the least recently used among equals (all fade 0 within the distance).
            slot = 0;
            for (int i = 1; i < r_view.memory_count; i++) {
                const ScanView::ViewMemory &m = r_view.memories[i];
                const ScanView::ViewMemory &worst = r_view.memories[slot];
                slot = m.fade > worst.fade || (m.fade == worst.fade && m.used < worst.used) ? i : slot;
            }
            result = REMEMBERED_REPLACED;
        } else {
            r_view.memory_count++;
        }
        r_view.memories[slot] = { r_view.camera_chunk, r_view.forward, 0.0, now };
    }
    // Sorted by fade, which a new camera chunk changes (lod_split stops at the first fade too large).
    // Insertion sort: stable, no allocation, and the list is short and nearly sorted.
    for (int i = 1; i < r_view.memory_count; i++) {
        const ScanView::ViewMemory m = r_view.memories[i];
        int j = i;
        for (; j > 0 && r_view.memories[j - 1].fade > m.fade; j--) {
            r_view.memories[j] = r_view.memories[j - 1];
        }
        r_view.memories[j] = m;
    }
    return result;
}

void ChunkDiscoverer::set_camera(const Vector3 &p_pos, const Vector3 &p_forward) {
    ScanView new_view = make_view(p_pos, p_forward);
    mutex->lock();
    new_view.copy_memories_from(view);
    const Remembered remembered = remember_view(new_view, memory_distance);
    stats.memories_added += remembered != REMEMBERED_KNOWN ? 1 : 0;
    stats.memories_replaced += remembered == REMEMBERED_REPLACED ? 1 : 0;
    const bool changed = view_changed(view, new_view);
    if (changed) {
        view = new_view;
        camera_change_usec = now_usec();
        mutex->unlock();
        semaphore->post();
        VLOG(2, "new view:", WorldCoord::to_string(new_view.camera_chunk), " ", new_view.forward, " memories ", new_view.memory_count);
        return;
    }
    // Same chunk and memories, in the same order: keep the use times for eviction.
    view.copy_memories_from(new_view);
    if (turned_past(new_view.forward, view.forward)) {
        // A turn that doesn't change the scanned set (back_scale 0, or towards a remembered view): no
        // rescan, but the next scan (camera chunk, instance edit) orders the meshers' queues by the current
        // direction.
        view.forward = new_view.forward;
    }
    mutex->unlock();
}

void ChunkDiscoverer::set_view_memory(double p_metres) {
    ERR_FAIL_COND_MSG(lod_distances.size() < 2, "ChunkDiscoverer::set_view_memory: call setup_lod first.");
    if (p_metres < 0.0) {
        memory_distance = lod_distances[MIN(VIEW_MEMORY_AUTO_LOD, get_max_lod())];
    } else {
        // One chunk-id unit is half a LOD0 chunk.
        memory_distance = p_metres / (chunk_param.chunk_world_dimension() * 0.5);
    }
}
void ChunkDiscoverer::_compare_with_cpu(const ScanView &p_view, const LODHashSet &p_scanned) {
    const uint64_t t0 = now_usec();
    LODHashSet cpu_set;
    ScanCounters cpu_counters;
    scan(p_view, cull_mode, cpu_set, cpu_counters);
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
                gpu ? "GPU" : "CPU", WorldCoord::to_string(p_view.camera_chunk), only_here, only_cpu));
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
    d["out_of_view"] = p_counters.out_of_view;
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
    d["delta"] = stats.delta.to_dict();
    d["deltas_merged"] = stats.deltas_merged;
    d["last_delta_entered"] = stats.last_delta_entered;
    d["last_delta_left"] = stats.last_delta_left;
    d["latency"] = stats.latency.to_dict();
    d["cull_mode"] = (int)cull_mode;
    d["scans_skipped_same_chunk"] = stats.scans_skipped_same_chunk;
    d["last_scan_size"] = stats.last_scan_size;
    d["last_out_of_view"] = stats.last_out_of_view;
    d["last_memories"] = stats.last_memories;
    d["memories_added"] = stats.memories_added;
    d["memories_replaced"] = stats.memories_replaced;
    d["memory_distance"] = memory_distance;
    d["delta_entered_total"] = stats.delta_entered_total;
    d["delta_left_total"] = stats.delta_left_total;
    d["back_scale"] = back_scale;
    d["cone_half_angle_deg"] = Math::rad_to_deg(cone_half_angle);
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
