#include "test_common.hpp"

#include "chunk_discoverer.hpp"
#include "sdf_cascade.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <memory>

namespace {

// The discoverer's walk before the wave walk (recursive, depth first, one SDF call per corner), kept as the
// reference the wave walk must reproduce.
struct ReferenceWalk {
    const ChunkParam &param;
    const SDFBase &sdf;
    const std::vector<float> &lod_distances;
    int64_t nodes_visited = 0;
    int64_t culled = 0;

    bool outside(const ChunkId &p_chunk) const {
        const float chunk_outer_radius = ChunkMath::world_chunk_size(p_chunk, param) * 2;
        const VoxelPos center = ChunkMath::chunk_center_voxel(p_chunk, param);
        const int64_t half_voxels = ChunkMath::get_lod_size(p_chunk) * param.half_chunk_voxels();
        const int lod = ChunkMath::get_lod(p_chunk);
        float nearest = FLT_MAX;
        for (int corner = 0; corner < 8; ++corner) {
            WorldPos corner_pos;
            corner_pos.voxel = center + VoxelPos((corner & 1) ? half_voxels : -half_voxels, (corner & 2) ? half_voxels : -half_voxels, (corner & 4) ? half_voxels : -half_voxels);
            nearest = MIN(nearest, std::abs(sdf.evaluate_cull(corner_pos, lod)));
        }
        // The no-cull rule near instances (SDFBase::cull_exempt), asked when the corners would cull.
        return nearest > chunk_outer_radius && !sdf.cull_exempt(p_chunk, lod, nearest - chunk_outer_radius);
    }

    void walk(const ChunkId &p_camera, const ChunkId &p_node, LODHashSet &r_set) {
        const int lod = ChunkMath::get_lod(p_node);
        nodes_visited++;
        if (outside(p_node)) {
            culled++;
            return;
        }
        if (lod == 0) {
            r_set.insert(p_node);
            return;
        }
        const double distance = p_camera.distance_to(p_node);
        const float radius = std::sqrt(3.0f) * static_cast<float>(int64_t(1) << lod);
        if (distance - radius < lod_distances[lod]) {
            const int64_t child = int64_t(1) << (lod - 1);
            for (int i = 0; i < 8; ++i) {
                walk(p_camera, p_node + ChunkId((i & 1) ? child : -child, (i & 2) ? child : -child, (i & 4) ? child : -child), r_set);
            }
        } else {
            r_set.insert(p_node);
        }
    }

    void scan(const ChunkDiscoverer &p_discoverer, const ChunkId &p_camera, LODHashSet &r_set) {
        int64_t step;
        int64_t count;
        p_discoverer.get_scan_grid(step, count);
        const ChunkId ancestor = ChunkMath::get_parent_from_child_until(p_camera, p_discoverer.get_max_lod());
        const ChunkId span = ChunkId(count, count, count) * step;
        for (int64_t x = ancestor.x - span.x; x <= ancestor.x + span.x; x += step) {
            for (int64_t y = ancestor.y - span.y; y <= ancestor.y + span.y; y += step) {
                for (int64_t z = ancestor.z - span.z; z <= ancestor.z + span.z; z += step) {
                    walk(p_camera, ChunkId(x, y, z), r_set);
                }
            }
        }
    }
};

int64_t set_difference(const LODHashSet &p_a, const LODHashSet &p_b) {
    int64_t n = 0;
    for (const ChunkId &id : p_a) {
        n += p_b.has(id) ? 0 : 1;
    }
    return n;
}

// Both walks around three camera poses (near the dummy's ground, the 15- and 25-LOD cascade grounds at
// ~3 and ~6 km), same sets and the same node counts; the wave walk again with a cache (cold, then warm).
void compare_walks(SDFBase &p_sdf, int p_lods) {
    ChunkParam param;
    p_sdf.configure(param, p_lods);
    ChunkDiscoverer *discoverer = memnew(ChunkDiscoverer);
    discoverer->set_config(param, &p_sdf);
    // The bench's 1280x720 view at lod_error 4 (a coarser tree than the default 2, to keep the test short).
    discoverer->setup_lod(p_lods, 4.0f, 720.0f, 75.0f * (3.14159265f / 180.0f));
    const godot::Vector3 poses[] = { godot::Vector3(-52.76f, 330.0f, -95.64f), godot::Vector3(3000.0f, 3200.0f, 2000.0f), godot::Vector3(-800.0f, 6500.0f, 400.0f) };
    for (const godot::Vector3 &pose : poses) {
        CAPTURE(pose);
        const ChunkId camera = ChunkMath::chunk_at_render(ov_f3(pose), param);
        ReferenceWalk reference{ param, p_sdf, discoverer->get_lod_distances() };
        LODHashSet want;
        reference.scan(*discoverer, camera, want);

        LODHashSet got;
        ChunkDiscoverer::ScanCounters counters;
        ScanView view;
        view.camera_chunk = camera;
        discoverer->scan(view, ChunkDiscoverer::CULL_LEGACY, got, counters);
        CHECK(want.size() > 0);
        CHECK(got.size() == want.size());
        CHECK(set_difference(got, want) == 0);
        CHECK(set_difference(want, got) == 0);
        CHECK(counters.nodes_visited == reference.nodes_visited);
        CHECK(counters.culled == reference.culled);
        // One wave per LOD from the top, down to the last LOD a node was split to.
        CHECK(!counters.waves.empty());
        CHECK(counters.waves.size() <= size_t(discoverer->get_max_lod() + 1));
        for (size_t w = 0; w < counters.waves.size(); w++) {
            CHECK(counters.waves[w].lod == discoverer->get_max_lod() - int(w));
        }
        CHECK(counters.sdf_calls < counters.corners); // shared corners are evaluated once

        ChunkDiscoverer::CullCache cache;
        cache.verify = true;
        for (int pass = 0; pass < 2; pass++) {
            cache.scan_index++;
            LODHashSet cached;
            ChunkDiscoverer::ScanCounters c;
            discoverer->scan(view, ChunkDiscoverer::CULL_LEGACY, cached, c, &cache);
            CHECK(set_difference(cached, want) == 0);
            CHECK(cached.size() == want.size());
            CHECK(c.verify_mismatches == 0);
            CHECK(c.cache_hits == (pass == 0 ? 0 : c.nodes_visited));
        }
    }
    memdelete(discoverer);
}

} // namespace

TEST_CASE("ChunkDiscoverer: the wave walk gives the recursive walk's set (SDFCascade with instances: cull_exempt, 15 and 25 LODs)") {
    for (const int lods : { 15, 25 }) {
        CAPTURE(lods);
        SDFCascade sdf;
        sdf.set_scatter(400, 200.0);
        compare_walks(sdf, lods);
    }
}

TEST_CASE("ChunkDiscoverer: the wave walk gives the recursive walk's set (SDFCascade, 15 and 25 LODs)") {
    for (const int lods : { 15, 25 }) {
        CAPTURE(lods);
        SDFCascade sdf;
        compare_walks(sdf, lods);
    }
}

// lod_pixel_error_back: the chunks in the view cone are those of the front error alone, there are fewer
// outside it, and face neighbours stay within one LOD (the effective distance changes by at most
// 1 + back_scale per unit, ChunkMath::lod_distance).
TEST_CASE("ChunkDiscoverer: view-dependent LOD keeps the cone's chunks and one LOD across faces") {
    ChunkParam param;
    SDFCascade sdf;
    const int lods = 15;
    sdf.configure(param, lods);
    ChunkDiscoverer *discoverer = memnew(ChunkDiscoverer);
    discoverer->set_config(param, &sdf);
    // The bench's 1280x720 view, error 1 in the view cone and 4 behind (the main_scene case).
    const float fov = godot::Math::deg_to_rad(75.0f);
    discoverer->setup_lod(lods, 1.0f, 720.0f, fov, 4.0f, ChunkMath::view_cone_half_angle(fov, 16.0f / 9.0f, godot::Math::deg_to_rad(20.0f)));
    const godot::Vector3 pose(3000.0f, 3200.0f, 2000.0f); // near the 15-LOD cascade ground
    size_t split_x_size = 0; // forward +x: the set of the split, and of the front error (reused below)
    size_t front_x_size = 0;
    for (const godot::Vector3 &forward : { godot::Vector3(1, 0, 0), godot::Vector3(0.3f, -0.8f, 0.5f), godot::Vector3(-0.2f, 0.3f, -1.0f) }) {
        CAPTURE(forward);
        const ScanView split = discoverer->make_view(ov_f3(pose), ov_f3(forward));
        CHECK(split.back_scale == doctest::Approx(3.0));
        ScanView front = split;
        front.back_scale = 0.0;
        LODHashSet got;
        LODHashSet want;
        ChunkDiscoverer::ScanCounters split_counters;
        ChunkDiscoverer::ScanCounters front_counters;
        discoverer->scan(split, ChunkDiscoverer::CULL_LEGACY, got, split_counters);
        discoverer->scan(front, ChunkDiscoverer::CULL_LEGACY, want, front_counters);

        auto in_cone = [&split](const LODHashSet &p_set) {
            LODHashSet r;
            for (const ChunkId &id : p_set) {
                if (ChunkMath::view_cone_gap(id, ChunkMath::get_lod(id), split) == 0.0) {
                    r.insert(id);
                }
            }
            return r;
        };
        const LODHashSet got_in = in_cone(got);
        const LODHashSet want_in = in_cone(want);
        CHECK(got_in.size() > 0);
        CHECK(got_in.size() == want_in.size());
        CHECK(set_difference(got_in, want_in) == 0);
        CHECK(split_counters.out_of_view > 0);
        CHECK(got.size() < want.size());
        if (forward == godot::Vector3(1, 0, 0)) {
            split_x_size = got.size();
            front_x_size = want.size();
        }
        MESSAGE("chunks: ", int64_t(want.size()), " at error 1, ", int64_t(got.size()), " with 4 behind (", split_counters.out_of_view, " out of view)");

        std::vector<int64_t> faces;
        face_lod_jumps(got, discoverer->get_max_lod(), faces);
        CHECK(faces.size() <= 2); // jumps of 0 and 1 only
    }

    // Remembered views (a camera that looked around, then walked): the LOD rule's distance is still
    // 1-Lipschitz (each term is a distance to a cone plus a constant), so face neighbours stay within one
    // LOD, and the set lies between the current view's alone and the front error's.
    {
        ScanView looked = discoverer->make_view(ov_f3(pose), Float3{ 1, 0, 0 });
        const double memory_distance = 30.0;
        const ChunkId spot = looked.camera_chunk - ChunkId(64, 0, 0); // 64 units back along x
        for (int i = 0; i < 9; i++) {
            const godot::Vector3 f = godot::Vector3(1, 0, 0).rotated(godot::Vector3(0, 1, 0), godot::Math::deg_to_rad(40.0f * i));
            looked.forward = ov_f3(f);
            ScanView there = looked;
            there.camera_chunk = spot;
            ChunkDiscoverer::remember_view(there, memory_distance);
            looked.copy_memories_from(there);
        }
        looked.forward = Float3{ 1.0f, 0.0f, 0.0f };
        CHECK(ChunkDiscoverer::remember_view(looked, memory_distance) == ChunkDiscoverer::REMEMBERED_ADDED);
        CHECK(looked.memory_count == 10);
        CHECK(looked.memories[0].fade == 0.0);
        CHECK(looked.memories[9].fade == doctest::Approx(64.0 - 30.0));
        LODHashSet got;
        ChunkDiscoverer::ScanCounters c;
        discoverer->scan(looked, ChunkDiscoverer::CULL_LEGACY, got, c);
        CHECK(split_x_size < got.size());
        CHECK(got.size() < front_x_size);
        std::vector<int64_t> faces;
        face_lod_jumps(got, discoverer->get_max_lod(), faces);
        CHECK(faces.size() <= 2);
    }

    // Without memory, a turn rescans past RESCAN_TURN_DEG, and only when the LOD rule depends on the direction.
    auto turned = [&](const ScanView &p_from, float p_degrees, double p_back_scale) {
        ScanView v = discoverer->make_view(ov_f3(pose), ov_f3(godot::Vector3(1, 0, 0).rotated(godot::Vector3(0, 1, 0), godot::Math::deg_to_rad(p_degrees))));
        v.back_scale = p_back_scale;
        v.copy_memories_from(p_from);
        ChunkDiscoverer::remember_view(v, 0.0);
        return v;
    };
    const ScanView a = turned(ScanView(), 0.0f, 3.0);
    CHECK(a.memory_count == 1);
    CHECK(!ChunkDiscoverer::view_changed(a, turned(a, 5.0f, 3.0)));
    CHECK(ChunkDiscoverer::view_changed(a, turned(a, 12.0f, 3.0)));
    const ScanView off = turned(ScanView(), 0.0f, 0.0);
    CHECK(off.memory_count == 0);
    CHECK(!ChunkDiscoverer::view_changed(off, turned(off, 12.0f, 0.0)));
    memdelete(discoverer);
}

// remember_view: a camera turning in place remembers a view per RESCAN_TURN_DEG during its first turn,
// then its scanned set stops changing; walking on, the views it had fade with the distance.
TEST_CASE("ChunkDiscoverer: remembered views stop the rescans of a camera turning in place") {
    auto turn = [](ScanView &r_view, double p_memory_distance, int p_degrees, int &r_changes) {
        r_changes = 0;
        const int start = int(std::lround(godot::Math::rad_to_deg(std::atan2(-r_view.forward.z, r_view.forward.x))));
        for (int deg = start + 1; deg <= start + p_degrees; deg++) {
            ScanView next = r_view;
            const godot::Vector3 dir = godot::Vector3(1, 0, 0).rotated(godot::Vector3(0, 1, 0), godot::Math::deg_to_rad(float(deg)));
            next.forward = ov_f3(dir);
            CHECK(ChunkDiscoverer::remember_view(next, p_memory_distance) != ChunkDiscoverer::REMEMBERED_REPLACED);
            if (ChunkDiscoverer::view_changed(r_view, next)) {
                r_changes++;
                r_view = next;
            }
        }
    };
    ScanView view;
    view.camera_chunk = ChunkId(1, 1, 1);
    view.forward = Float3{ 1.0f, 0.0f, 0.0f };
    view.back_scale = 3.0;
    const double memory_distance = 30.0;
    CHECK(ChunkDiscoverer::remember_view(view, memory_distance) == ChunkDiscoverer::REMEMBERED_ADDED);
    int first = 0;
    int second = 0;
    turn(view, memory_distance, 360, first);
    turn(view, memory_distance, 360, second);
    CHECK(first >= 30); // one view every RESCAN_TURN_DEG + 1 degree
    CHECK(second == 0);
    CHECK(view.memory_count == first + 1);

    // Without memory every RESCAN_TURN_DEG of every turn rescans: one view, replaced.
    ScanView off = view;
    off.memory_count = 0;
    CHECK(ChunkDiscoverer::remember_view(off, 0.0) == ChunkDiscoverer::REMEMBERED_ADDED);
    int changes = 0;
    for (int deg = 1; deg <= 360; deg++) {
        ScanView next = off;
        const godot::Vector3 dir = godot::Vector3(1, 0, 0).rotated(godot::Vector3(0, 1, 0), godot::Math::deg_to_rad(float(deg)));
        next.forward = ov_f3(dir);
        ChunkDiscoverer::remember_view(next, 0.0);
        CHECK(next.memory_count == 1);
        if (ChunkDiscoverer::view_changed(off, next)) {
            changes++;
            off = next;
        }
    }
    CHECK(changes >= 30);

    // Walking along x: within memory_distance nothing fades; past it, the turn's views fade with the
    // distance, the camera chunk's own view first.
    for (int64_t x = 3; x <= 101; x += 2) {
        ScanView next = view;
        next.camera_chunk = ChunkId(x, 1, 1);
        CHECK(ChunkDiscoverer::remember_view(next, memory_distance) != ChunkDiscoverer::REMEMBERED_KNOWN);
        CHECK(ChunkDiscoverer::view_changed(view, next));
        view = next;
    }
    bool own = false;
    for (int i = 0; i < view.memory_count; i++) {
        own = own || (view.memories[i].apex == view.camera_chunk && view.memories[i].fade == 0.0);
    }
    CHECK(own);
    CHECK(view.memories[view.memory_count - 1].apex == ChunkId(1, 1, 1));
    CHECK(view.memories[view.memory_count - 1].fade == doctest::Approx(100.0 - 30.0));
    for (int i = 1; i < view.memory_count; i++) {
        CHECK(view.memories[i - 1].fade <= view.memories[i].fade);
    }
    // Full: a new view evicts the most faded one (a view of the turn, the farthest).
    CHECK(view.memory_count == ScanView::MAX_VIEW_MEMORIES);
    ScanView next = view;
    next.forward = Float3{ 0.0f, 0.0f, 1.0f };
    CHECK(ChunkDiscoverer::remember_view(next, memory_distance) == ChunkDiscoverer::REMEMBERED_REPLACED);
    CHECK(next.memory_count == ScanView::MAX_VIEW_MEMORIES);
}

// The memories stay sorted by fade when the camera comes back to a chunk it already had a view in (a
// known view, nothing added): lod_split stops at the first fade too large.
TEST_CASE("ChunkDiscoverer: remember_view re-sorts the fades of a known view") {
    ScanView view;
    view.back_scale = 3.0;
    view.forward = Float3{ 1.0f, 0.0f, 0.0f };
    const double memory_distance = 10.0;
    for (int64_t x : { int64_t(1), int64_t(201) }) { // out to x = 201, a view at each end
        view.camera_chunk = ChunkId(x, 1, 1);
        CHECK(ChunkDiscoverer::remember_view(view, memory_distance) == ChunkDiscoverer::REMEMBERED_ADDED);
    }
    CHECK(view.memories[0].apex == ChunkId(201, 1, 1));
    view.camera_chunk = ChunkId(1, 1, 1); // and back
    CHECK(ChunkDiscoverer::remember_view(view, memory_distance) == ChunkDiscoverer::REMEMBERED_KNOWN);
    CHECK(view.memories[0].apex == ChunkId(1, 1, 1));
    CHECK(view.memories[0].fade == 0.0);
    CHECK(view.memories[1].fade == doctest::Approx(200.0 - memory_distance));
}

// Full, with every fade 0 (looking around in place): the least recently used view goes, not the newest.
TEST_CASE("ChunkDiscoverer: a full memory replaces the least recently used view") {
    std::vector<godot::Vector3> dirs;
    for (int yaw = 0; yaw < 360; yaw += 9) {
        dirs.push_back(godot::Vector3(1, 0, 0).rotated(godot::Vector3(0, 1, 0), godot::Math::deg_to_rad(float(yaw))));
    }
    for (int yaw = 0; int(dirs.size()) < ScanView::MAX_VIEW_MEMORIES + 2; yaw += 13) { // 9.2 degrees apart at 45 degrees up
        dirs.push_back(godot::Vector3(1, 1, 0).normalized().rotated(godot::Vector3(0, 1, 0), godot::Math::deg_to_rad(float(yaw))));
    }
    ScanView view;
    view.back_scale = 3.0;
    view.camera_chunk = ChunkId(1, 1, 1);
    auto look = [&view](const godot::Vector3 &p_dir) {
        view.forward = ov_f3(p_dir);
        return ChunkDiscoverer::remember_view(view, 30.0);
    };
    for (int i = 0; i < ScanView::MAX_VIEW_MEMORIES; i++) {
        CHECK(look(dirs[i]) == ChunkDiscoverer::REMEMBERED_ADDED);
    }
    CHECK(look(dirs[0]) == ChunkDiscoverer::REMEMBERED_KNOWN); // dirs[1] is now the least recently used
    auto has = [&view](const godot::Vector3 &p_dir) {
        for (int i = 0; i < view.memory_count; i++) {
            if (view.memories[i].dir == ov_f3(p_dir)) {
                return true;
            }
        }
        return false;
    };
    CHECK(look(dirs[ScanView::MAX_VIEW_MEMORIES]) == ChunkDiscoverer::REMEMBERED_REPLACED);
    CHECK(has(dirs[0]));
    CHECK(!has(dirs[1]));
    CHECK(look(dirs[ScanView::MAX_VIEW_MEMORIES + 1]) == ChunkDiscoverer::REMEMBERED_REPLACED);
    CHECK(has(dirs[ScanView::MAX_VIEW_MEMORIES])); // the newest stays
    CHECK(!has(dirs[2]));
}
