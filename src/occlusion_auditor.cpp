#include "occlusion_auditor.hpp"

#include "debug_log.hpp"
#include "instance_table.hpp"

#include <godot_cpp/classes/worker_thread_pool.hpp>
#include <godot_cpp/variant/array.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>

namespace {

const char *band_of(int p_lod) {
    return p_lod <= 3 ? "near" : (p_lod <= 6 ? "mid" : "far");
}

// A ray stops after this many steps (counted apart from misses).
constexpr int MAX_STEPS = 200000;
// The sphere tracing's slope bound: twice the steepest occluder bound the terrains give (the cascade's 2).
constexpr double SPHERE_TRACE_LIPSCHITZ = 4.0;

enum RayResult {
    RAY_BLOCKED,
    RAY_BLOCKED_BY_HIDDEN,
    RAY_MISS,
    RAY_OUT_OF_STEPS,
};

struct Ray {
    VoxelPos eye_base;
    double eye[3]; // eye offset from eye_base (LOD0 voxels)
    double dir[3]; // unit
    double t_enter; // where the ray enters the target's box (LOD0 voxels from the eye)
    ChunkId target;
};

struct Job {
    const SDFBase *sdf;
    const ChunkParam *param;
    const LODHashSet *scan;
    const LODHashSet *hidden;
    const std::vector<Ray> *rays;
    std::vector<uint8_t> *results;
    std::atomic<int64_t> steps{ 0 };
    double lipschitz; // the sphere tracing's bound (per metre, metres)
};

float eval(const SDFBase &p_sdf, const VoxelPos &p_voxel, int p_lod) {
    WorldPos p;
    p.voxel = p_voxel;
    return p_sdf.evaluate(p, p_lod);
}

RayResult march(const Job &p_job, const Ray &p_ray, int64_t &r_steps) {
    const ChunkParam &param = *p_job.param;
    const double voxel = param.voxel_size;
    const int64_t chunk_voxels = 2 * param.half_chunk_voxels();
    // The cell of the last lookup: its LOD, minimum corner, solidity and smallest corner value.
    int cell_lod = -2;
    VoxelPos cell_min;
    bool cell_solid = false;
    double cell_min_m = 0.0;
    double t = 0.0;
    for (int64_t step = 0; step < MAX_STEPS; step++) {
        r_steps++;
        if (t >= p_ray.t_enter) {
            return RAY_MISS;
        }
        VoxelPos v;
        for (int a = 0; a < 3; a++) {
            v[a] = p_ray.eye_base[a] + int64_t(std::floor(p_ray.eye[a] + t * p_ray.dir[a]));
        }
        const ChunkId lod0(2 * WorldCoord::floor_div(v.x, chunk_voxels) + 1, 2 * WorldCoord::floor_div(v.y, chunk_voxels) + 1,
                2 * WorldCoord::floor_div(v.z, chunk_voxels) + 1);
        const int lod = scanned_lod_at(*p_job.scan, lod0, p_job.scan->max_lod);
        double advance_m;
        if (lod < 0) {
            // Culled: deep air or deep solid (reaching it crossed a surface of a scanned chunk).
            const double w = eval(*p_job.sdf, v, 0);
            if (!(w > 0.0)) {
                return RAY_BLOCKED;
            }
            advance_m = w / p_job.lipschitz;
        } else {
            const ChunkId leaf = ChunkMath::get_parent_from_child_until(lod0, lod);
            if (leaf != p_ray.target && p_job.hidden->has(leaf)) {
                return RAY_BLOCKED_BY_HIDDEN;
            }
            const int64_t cell = int64_t(1) << lod;
            const VoxelPos base(WorldCoord::floor_div(v.x, cell) * cell, WorldCoord::floor_div(v.y, cell) * cell, WorldCoord::floor_div(v.z, cell) * cell);
            if (lod != cell_lod || base != cell_min) {
                cell_lod = lod;
                cell_min = base;
                cell_solid = true;
                cell_min_m = INFINITY;
                for (int c = 0; c < 8; c++) {
                    const double w = eval(*p_job.sdf, base + VoxelPos((c & 1) ? cell : 0, (c & 2) ? cell : 0, (c & 4) ? cell : 0), lod);
                    cell_solid = cell_solid && !(w > 0.0);
                    cell_min_m = MIN(cell_min_m, w);
                }
            }
            if (cell_solid) {
                return RAY_BLOCKED;
            }
            // The point is within a cell diagonal of the nearest corner.
            advance_m = MAX(cell_min_m / p_job.lipschitz - std::sqrt(3.0) * double(cell) * voxel, 0.5 * double(cell) * voxel);
        }
        t += MAX(advance_m / voxel, 0.5);
    }
    return RAY_OUT_OF_STEPS;
}

void march_one(void *p_job, uint32_t p_index) {
    Job &job = *static_cast<Job *>(p_job);
    int64_t steps = 0;
    (*job.results)[p_index] = uint8_t(march(job, (*job.rays)[p_index], steps));
    job.steps += steps;
}

// Where the ray from the origin along p_dir (unit) enters [p_lo, p_hi] (relative to the eye): the slab test.
double box_entry(const double p_dir[3], const double p_lo[3], const double p_hi[3]) {
    double enter = 0.0;
    for (int a = 0; a < 3; a++) {
        if (std::abs(p_dir[a]) < 1e-12) {
            continue;
        }
        const double t0 = p_lo[a] / p_dir[a];
        const double t1 = p_hi[a] / p_dir[a];
        enter = MAX(enter, MIN(t0, t1));
    }
    return enter;
}

// The rays from eye p_eye (offset from p_eye_base, LOD0 voxels) to the centre of p_target's vertex-grid box and its
// corners pulled a tenth of the way in.
void add_rays(const ChunkId &p_target, const ChunkParam &p_param, const VoxelPos &p_eye_base, const Vector3 &p_eye, std::vector<Ray> &r_rays) {
    VoxelPos lo, hi;
    Archetype::chunk_sample_box(p_target, p_param, lo, hi);
    double blo[3], bhi[3];
    for (int a = 0; a < 3; a++) {
        blo[a] = double(lo[a] - p_eye_base[a]) - double(p_eye[a]);
        bhi[a] = double(hi[a] - p_eye_base[a]) - double(p_eye[a]);
    }
    for (int k = 0; k < 9; k++) {
        double q[3];
        for (int a = 0; a < 3; a++) {
            const double mid = 0.5 * (blo[a] + bhi[a]);
            q[a] = k == 8 ? mid : mid + 0.9 * (((k >> a) & 1) ? bhi[a] - mid : blo[a] - mid);
        }
        const double len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        if (len < 1e-9) {
            continue;
        }
        Ray r;
        r.eye_base = p_eye_base;
        r.target = p_target;
        for (int a = 0; a < 3; a++) {
            r.eye[a] = double(p_eye[a]);
            r.dir[a] = q[a] / len;
        }
        r.t_enter = box_entry(r.dir, blo, bhi);
        r_rays.push_back(r);
    }
}

// Marches p_rays over the WorkerThreadPool: one RayResult each.
std::vector<uint8_t> march_all(const SDFBase &p_sdf, const ChunkParam &p_param, const LODHashSet &p_scan, const LODHashSet &p_hidden,
        const std::vector<Ray> &p_rays, int64_t &r_steps) {
    std::vector<uint8_t> results(p_rays.size(), RAY_MISS);
    Job job;
    job.sdf = &p_sdf;
    job.param = &p_param;
    job.scan = &p_scan;
    job.hidden = &p_hidden;
    job.rays = &p_rays;
    job.results = &results;
    job.lipschitz = SPHERE_TRACE_LIPSCHITZ;
    if (!p_rays.empty()) {
        WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
        const WorkerThreadPool::GroupID id = pool->add_native_group_task(&march_one, &job, int(p_rays.size()), -1, true, "OcclusionAuditor");
        pool->wait_for_group_task_completion(id);
    }
    r_steps = job.steps.load();
    return results;
}

} // namespace

Dictionary OcclusionAuditor::ray_census(const SDFBase &p_sdf, const ChunkParam &p_param, const LODHashSet &p_scan, const VoxelPos &p_eye_base,
        const Vector3 &p_eye, int p_min_lod, int p_per_lod) {
    const uint64_t t0 = now_usec();
    // Up to p_per_lod chunks per LOD, every k-th by distance.
    std::vector<std::vector<std::pair<double, ChunkId>>> by_lod(ChunkMath::MAX_LOD_NUMBERS);
    for (const ChunkId &id : p_scan) {
        const int lod = ChunkMath::get_lod(id);
        if (lod >= p_min_lod) {
            by_lod[lod].push_back({ ChunkMath::chunk_center_voxel(id, p_param).distance_to(p_eye_base), id });
        }
    }
    std::vector<Ray> rays;
    for (std::vector<std::pair<double, ChunkId>> &chunks : by_lod) {
        std::sort(chunks.begin(), chunks.end(), [](const std::pair<double, ChunkId> &a, const std::pair<double, ChunkId> &b) { return a.first < b.first; });
        const size_t stride = MAX<size_t>(1, chunks.size() / MAX(p_per_lod, 1));
        for (size_t i = 0; i < chunks.size(); i += stride) {
            add_rays(chunks[i].second, p_param, p_eye_base, p_eye, rays);
        }
    }
    int64_t steps = 0;
    const LODHashSet none;
    const std::vector<uint8_t> results = march_all(p_sdf, p_param, p_scan, none, rays, steps);

    // Per LOD: chunks, chunks with every ray blocked, rays, rays blocked.
    HashMap<ChunkId, bool, Vector3i64Hasher> all_blocked;
    std::vector<int64_t> rays_lod(ChunkMath::MAX_LOD_NUMBERS, 0), blocked_lod(ChunkMath::MAX_LOD_NUMBERS, 0);
    int64_t out_of_steps = 0;
    for (size_t i = 0; i < rays.size(); i++) {
        const int lod = ChunkMath::get_lod(rays[i].target);
        const bool blocked = results[i] == RAY_BLOCKED;
        out_of_steps += results[i] == RAY_OUT_OF_STEPS ? 1 : 0;
        rays_lod[lod]++;
        blocked_lod[lod] += blocked ? 1 : 0;
        bool *b = all_blocked.getptr(rays[i].target);
        if (!b) {
            all_blocked.insert(rays[i].target, blocked);
        } else {
            *b = *b && blocked;
        }
    }
    std::vector<int64_t> chunks_lod(ChunkMath::MAX_LOD_NUMBERS, 0), hidden_lod(ChunkMath::MAX_LOD_NUMBERS, 0);
    for (const KeyValue<ChunkId, bool> &kv : all_blocked) {
        const int lod = ChunkMath::get_lod(kv.key);
        chunks_lod[lod]++;
        hidden_lod[lod] += kv.value ? 1 : 0;
    }
    Array lods;
    for (int lod = 0; lod < ChunkMath::MAX_LOD_NUMBERS; lod++) {
        if (chunks_lod[lod] == 0) {
            continue;
        }
        Dictionary l;
        l["lod"] = lod;
        l["chunks"] = chunks_lod[lod];
        l["chunks_all_blocked"] = hidden_lod[lod];
        l["rays"] = rays_lod[lod];
        l["rays_blocked"] = blocked_lod[lod];
        l["ray_share"] = double(blocked_lod[lod]) / double(rays_lod[lod]);
        l["chunk_share"] = double(hidden_lod[lod]) / double(chunks_lod[lod]);
        lods.push_back(l);
    }
    Dictionary d;
    d["lods"] = lods;
    d["rays"] = int64_t(rays.size());
    d["out_of_steps"] = out_of_steps;
    d["steps_avg"] = rays.empty() ? 0.0 : double(steps) / double(rays.size());
    d["ms"] = double(now_usec() - t0) / 1000.0;
    return d;
}

Dictionary OcclusionAuditor::hidden_stats(const LODHashSet &p_hidden, const ChunkHashMap &p_loaded) {
    Dictionary bands;
    int64_t loaded_quads = 0;
    int64_t hidden_quads = 0;
    int64_t hidden_loaded = 0;
    for (const char *band : { "near", "mid", "far" }) {
        Dictionary b;
        b["loaded_chunks"] = 0;
        b["loaded_quads"] = 0;
        b["hidden_chunks"] = 0;
        b["hidden_quads"] = 0;
        bands[band] = b;
    }
    for (const KeyValue<ChunkId, ChunkOutput> &kv : p_loaded) {
        Dictionary b = bands[band_of(ChunkMath::get_lod(kv.key))];
        const int64_t quads = kv.value.triangle_count / 2;
        b["loaded_chunks"] = int64_t(b["loaded_chunks"]) + 1;
        b["loaded_quads"] = int64_t(b["loaded_quads"]) + quads;
        loaded_quads += quads;
        if (p_hidden.has(kv.key)) {
            b["hidden_chunks"] = int64_t(b["hidden_chunks"]) + 1;
            b["hidden_quads"] = int64_t(b["hidden_quads"]) + quads;
            hidden_quads += quads;
            hidden_loaded++;
        }
    }
    Dictionary d;
    d["hidden"] = int64_t(p_hidden.size());
    d["hidden_loaded"] = hidden_loaded;
    d["loaded_quads"] = loaded_quads;
    d["hidden_quads"] = hidden_quads;
    d["hidden_quad_share"] = loaded_quads > 0 ? double(hidden_quads) / double(loaded_quads) : 0.0;
    d["bands"] = bands;
    return d;
}

Dictionary OcclusionAuditor::audit(const SDFBase &p_sdf, const ChunkParam &p_param, const LODHashSet &p_scan, const LODHashSet &p_hidden,
        const ChunkHashMap &p_loaded, const VoxelPos &p_eye_base, const std::vector<Vector3> &p_eyes, int p_count) {
    const uint64_t t0 = now_usec();
    // The hidden chunks by distance, those with quads first when they are loaded; then every k-th, so the sample
    // spans the LODs.
    std::vector<std::pair<double, ChunkId>> candidates;
    for (const ChunkId &id : p_hidden) {
        const ChunkOutput *out = p_loaded.getptr(id);
        const bool empty = out && out->triangle_count == 0;
        const double d = ChunkMath::chunk_center_voxel(id, p_param).distance_to(p_eye_base);
        candidates.push_back({ d + (empty ? 1e30 : 0.0), id });
    }
    // By distance, then by id: the same sample whatever the set's order.
    std::sort(candidates.begin(), candidates.end(), [](const std::pair<double, ChunkId> &a, const std::pair<double, ChunkId> &b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        for (int k = 0; k < 3; k++) {
            if (a.second[k] != b.second[k]) {
                return a.second[k] < b.second[k];
            }
        }
        return false;
    });
    std::vector<ChunkId> targets;
    const size_t stride = MAX<size_t>(1, candidates.size() / MAX(p_count, 1));
    for (size_t i = 0; i < candidates.size() && int(targets.size()) < p_count; i += stride) {
        targets.push_back(candidates[i].second);
    }

    std::vector<Ray> rays;
    for (const ChunkId &target : targets) {
        for (const Vector3 &eye : p_eyes) {
            add_rays(target, p_param, p_eye_base, eye, rays);
        }
    }
    int64_t steps = 0;
    const std::vector<uint8_t> results = march_all(p_sdf, p_param, p_scan, p_hidden, rays, steps);

    int64_t counts[4] = {};
    HashSet<ChunkId, Vector3i64Hasher> miss_chunks;
    Array examples;
    for (size_t i = 0; i < rays.size(); i++) {
        counts[results[i]]++;
        if (results[i] == RAY_MISS && !miss_chunks.has(rays[i].target)) {
            miss_chunks.insert(rays[i].target);
            if (examples.size() < 16) {
                Dictionary e;
                e["chunk"] = WorldCoord::to_string(rays[i].target);
                e["lod"] = ChunkMath::get_lod(rays[i].target);
                e["distance_m"] = rays[i].t_enter * p_param.voxel_size;
                examples.push_back(e);
            }
        }
    }
    Dictionary d;
    d["hidden"] = int64_t(p_hidden.size());
    d["checked_chunks"] = int64_t(targets.size());
    d["eyes"] = int64_t(p_eyes.size());
    d["rays"] = int64_t(rays.size());
    d["blocked"] = counts[RAY_BLOCKED];
    d["blocked_by_hidden"] = counts[RAY_BLOCKED_BY_HIDDEN];
    d["misses"] = counts[RAY_MISS];
    d["out_of_steps"] = counts[RAY_OUT_OF_STEPS];
    d["miss_chunks"] = int64_t(miss_chunks.size());
    d["miss_examples"] = examples;
    d["steps_avg"] = rays.empty() ? 0.0 : double(steps) / double(rays.size());
    d["ms"] = double(now_usec() - t0) / 1000.0;
    d["ok"] = counts[RAY_MISS] == 0 && counts[RAY_OUT_OF_STEPS] == 0;
    return d;
}
