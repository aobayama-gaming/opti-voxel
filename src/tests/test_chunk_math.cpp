#include "test_common.hpp"

#include "chunk_math.hpp"
#include "godot_conv.hpp"
#include "subcube.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

bool same_bits(const WorldPos &p_a, const WorldPos &p_b) {
    return p_a.voxel == p_b.voxel && std::memcmp(&p_a.frac, &p_b.frac, sizeof(Float3)) == 0;
}

// An origin as VoxelTerrain::_ready builds it: a multiple of the root chunk span (2^lod_numbers ids).
ChunkId snapped_origin(int64_t p_x, int64_t p_y, int64_t p_z) {
    const int64_t root = int64_t(1) << 15;
    return ChunkId(WorldCoord::floor_div(p_x, root) * root, WorldCoord::floor_div(p_y, root) * root,
            WorldCoord::floor_div(p_z, root) * root);
}

// Deterministic coordinates in [-1000, 1000] (an LCG: the same sequence on every platform).
struct Lcg {
    uint64_t state;
    int64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return int64_t(state >> 40) % 2001 - 1000;
    }
};

} // namespace

TEST_CASE("ChunkMath: parent of a coordinate") {
    int lod = -1;
    CHECK(ChunkMath::get_parent_from_child(int64_t(5), lod) == 6); // 101 -> 110
    CHECK(lod == 1);
    CHECK(ChunkMath::get_parent_from_child(int64_t(7), lod) == 6);
    CHECK(lod == 1);
    CHECK(ChunkMath::get_parent_from_child(int64_t(6), lod) == 4); // 110 -> 100
    CHECK(lod == 2);
    CHECK(ChunkMath::get_parent_from_child(int64_t(-1), lod) == -2);
    CHECK(lod == 1);
    CHECK(ChunkMath::get_parent_from_child(int64_t(-3), lod) == -2);
    CHECK(ChunkMath::get_parent_from_child(int64_t(-2), lod) == -4);
    CHECK(lod == 2);
    CHECK(ChunkMath::get_parent_from_child(int64_t(0)) == 0);
    // Past the top of the int64 octree the coordinate is its own parent.
    CHECK(ChunkMath::get_parent_from_child(int64_t(1) << 62) == int64_t(1) << 62);
}

TEST_CASE("ChunkMath: the 8 children of a chunk share its parent") {
    const ChunkId parent(4, -4, 12); // LOD 2: children at +-2
    CHECK(ChunkMath::get_lod(parent) == 2);
    for (int i = 0; i < 8; i++) {
        const ChunkId child = parent + ChunkId((i & 1) ? 2 : -2, (i & 2) ? 2 : -2, (i & 4) ? 2 : -2);
        CAPTURE(child);
        CHECK(ChunkMath::get_lod(child) == 1);
        int lod = -1;
        CHECK(ChunkMath::get_parent_from_child(child, lod) == parent);
        CHECK(lod == 2);
    }
}

TEST_CASE("ChunkMath: LOD and size come from the lowest set bit") {
    CHECK(ChunkMath::get_lod(ChunkId(1, 3, -5)) == 0);
    CHECK(ChunkMath::get_lod(ChunkId(2, 6, -2)) == 1);
    CHECK(ChunkMath::get_lod(ChunkId(-8, 8, 24)) == 3);
    CHECK(ChunkMath::get_lod_size(ChunkId(-8, 8, 24)) == 8);
    CHECK(ChunkMath::get_lod_size(ChunkId(1, 1, 1)) == 1);
}

TEST_CASE("ChunkMath: a batch's LOD range") {
    CHECK(ChunkMath::lod_range({}) == ov::ivec2(0, 0));
    CHECK(ChunkMath::lod_range({ ChunkId(-8, 8, 24) }) == ov::ivec2(3, 3));
    CHECK(ChunkMath::lod_range({ ChunkId(2, 6, -2), ChunkId(1, 3, -5), ChunkId(-8, 8, 24), ChunkId(4, 4, 4) }) == ov::ivec2(0, 3));
}

TEST_CASE("ChunkMath: ancestor at a target LOD") {
    CHECK(ChunkMath::get_parent_from_child_until(ChunkId(1, 1, 1), 3) == ChunkId(8, 8, 8));
    CHECK(ChunkMath::get_parent_from_child_until(ChunkId(-1, 1, -3), 2) == ChunkId(-4, 4, -4));
}

TEST_CASE("ChunkMath: subcube_bits are the ancestor's sub-cubes a descendant overlaps") {
    const uint32_t axis = Subcube::AXIS;
    auto bit = [](uint32_t x, uint32_t y, uint32_t z) { return uint64_t(1) << Subcube::bit(x, y, z); };
    // LOD 3 ancestor at (8, 8, 8): ids (0, 16) per axis, sub-cubes of 4 ids.
    const ChunkId a(8, 8, 8);
    CHECK(ChunkMath::subcube_bits(a, ChunkId(1, 1, 1)) == bit(0, 0, 0)); // LOD 0, the low corner
    CHECK(ChunkMath::subcube_bits(a, ChunkId(15, 1, 13)) == bit(axis - 1, 0, axis - 1));
    CHECK(ChunkMath::subcube_bits(a, ChunkId(6, 10, 2)) == bit(1, 2, 0)); // LOD 1: one sub-cube
    // A child (LOD 2) covers the 2x2x2 sub-cubes of its octant.
    uint64_t octant = 0;
    for (uint32_t i = 0; i < 8; i++) {
        octant |= bit(2 + (i & 1), i >> 1 & 1, 2 + (i >> 2));
    }
    CHECK(ChunkMath::subcube_bits(a, ChunkId(12, 4, 12)) == octant);
    // Below a LOD-1 ancestor, a LOD0 child is 2x2x2 sub-cubes too.
    CHECK(WorldCoord::popcount64(ChunkMath::subcube_bits(ChunkId(2, 2, 2), ChunkId(1, 3, 1))) == 8);
    // Negative ids and a deep descendant: a LOD-10 ancestor and a LOD-5 chunk inside its (-x, +y, -z) corner.
    const ChunkId big(-3 * 1024, 1024, -1024);
    const ChunkId deep = big + ChunkId(-1024 + 32, 1024 - 32, -1024 + 32);
    CHECK(ChunkMath::get_lod(deep) == 5);
    CHECK(ChunkMath::subcube_bits(big, deep) == bit(0, axis - 1, 0));
    // The 8 children of a chunk together cover every sub-cube once.
    uint64_t all = 0;
    for (int i = 0; i < 8; i++) {
        const uint64_t b = ChunkMath::subcube_bits(a, a + ChunkId((i & 1) ? 4 : -4, (i & 2) ? 4 : -4, (i & 4) ? 4 : -4));
        CHECK((all & b) == 0);
        all |= b;
    }
    CHECK(all == ~uint64_t(0));
}

TEST_CASE("ChunkMath: chunk corners and centres in LOD0 voxels") {
    const ChunkParam param; // 32 samples, skirt 2: 28 voxels per LOD0 chunk, 14 per id unit
    CHECK(param.half_chunk_voxels() == 14);
    CHECK(ChunkMath::chunk_center_voxel(ChunkId(1, 1, 1), param) == VoxelPos(14, 14, 14));
    CHECK(ChunkMath::chunk_min_voxel(ChunkId(1, 1, 1), param) == VoxelPos(0, 0, 0));
    CHECK(ChunkMath::chunk_min_voxel(ChunkId(3, -1, 1), param) == VoxelPos(28, -28, 0));
    CHECK(ChunkMath::chunk_center_voxel(ChunkId(-1, -1, -1), param) == VoxelPos(-14, -14, -14));
    // A LOD1 chunk covers its two LOD0 children on each axis.
    CHECK(ChunkMath::chunk_min_voxel(ChunkId(2, 2, 2), param) == VoxelPos(0, 0, 0));
    CHECK(ChunkMath::chunk_center_voxel(ChunkId(2, 2, 2), param) == VoxelPos(28, 28, 28));
}

TEST_CASE("ChunkMath: chunk_at_voxel is the chunk whose span holds the voxel") {
    const ChunkParam param;
    const int64_t far = int64_t(1) << 60;
    for (int lod = 0; lod <= 30; lod += 3) {
        const int64_t span = int64_t(param.chunk_voxel_dimension()) << lod;
        for (const int64_t base : { int64_t(0), int64_t(-1), far - 77, -far + 5, (int64_t(1) << 32) - 3 }) {
            for (const int64_t d : { int64_t(0), int64_t(1), span - 1, span, -span }) {
                const VoxelPos v(base + d, -(base + d), base / 2 - d);
                const ChunkId id = ChunkMath::chunk_at_voxel(v, lod, param);
                CAPTURE(lod);
                CAPTURE(v);
                CHECK(ChunkMath::get_lod(id) == lod);
                const VoxelPos lo = ChunkMath::chunk_min_voxel(id, param);
                for (int a = 0; a < 3; a++) {
                    CHECK(lo[a] <= v[a]);
                    CHECK(v[a] < lo[a] + span);
                }
            }
        }
    }
}

TEST_CASE("ChunkMath: chunk_at_render finds the chunk whose centre was rendered") {
    ChunkParam param;
    const ChunkId origins[] = { ChunkId(), snapped_origin(int64_t(1) << 40, -(int64_t(1) << 40), 123456789) };
    for (const ChunkId &origin : origins) {
        param.origin_chunk = origin;
        for (const ChunkId offset : { ChunkId(1, 1, 1), ChunkId(-1, 3, -7), ChunkId(201, -999, 55) }) {
            const ChunkId chunk = origin + offset;
            CAPTURE(chunk);
            CHECK(ChunkMath::chunk_at_render(ChunkMath::chunk_render_center(chunk, param), param) == chunk);
        }
    }
}

TEST_CASE("ChunkMath: grid_to_render_lo32 matches grid_to_render across a low-word wrap") {
    // What the page shader does from a header (low words only) must land where the CPU puts it.
    ChunkParam param;
    param.origin_chunk = snapped_origin((int64_t(1) << 32) - (int64_t(1) << 16), 0, -(int64_t(1) << 35));
    const Float3 grid{ 5.5f, 2.0f, 30.0f };
    for (const ChunkId offset : { ChunkId(1, 1, 1), ChunkId(131073, -5, 7), ChunkId(-131071, 9, -1), ChunkId(131080, 8, -8) }) {
        const ChunkId chunk = param.origin_chunk + offset;
        CAPTURE(chunk);
        CHECK(ChunkMath::grid_to_render_lo32(WorldCoord::lo32_i(chunk), grid, param) ==
                ChunkMath::grid_to_render(chunk, grid, param));
    }
}

TEST_CASE("ChunkMath: sample_pos gives the same WorldPos from every chunk reaching a voxel") {
    const ChunkParam param;
    const WorldPos a = ChunkMath::sample_pos(ChunkId(1, 1, 1), Float3{ 30.0f, 2.0f, 2.0f }, param);
    const WorldPos b = ChunkMath::sample_pos(ChunkId(3, 1, 1), Float3{ 2.0f, 2.0f, 2.0f }, param); // neighbour
    const WorldPos c = ChunkMath::sample_pos(ChunkId(2, 2, 2), Float3{ 16.0f, 2.0f, 2.0f }, param); // parent
    CHECK(a.voxel == VoxelPos(28, 0, 0));
    CHECK(same_bits(a, b));
    CHECK(same_bits(a, c));

    const WorldPos half = ChunkMath::sample_pos(ChunkId(1, 1, 1), Float3{ 2.5f, 2.0f, 2.0f }, param);
    CHECK(half.voxel == VoxelPos(0, 0, 0));
    CHECK(half.frac.x == 0.5f);
}

TEST_CASE("ChunkMath: view_cone_gap is the distance from a chunk's sphere to the view cone") {
    ScanView view;
    view.camera_chunk = ChunkId(1, 1, 1);
    view.forward = Float3{ 0.0f, 0.0f, -1.0f };
    view.set_half_angle(WorldCoord::PI / 4.0);
    const double r0 = ChunkMath::chunk_bounding_radius(0);
    CHECK(ChunkMath::view_cone_gap(ChunkId(1, 1, -99), 0, view) == 0.0); // on the axis
    CHECK(ChunkMath::view_cone_gap(ChunkId(1, 1, 101), 0, view) == doctest::Approx(100.0 - r0)); // behind: the apex
    CHECK(ChunkMath::view_cone_gap(ChunkId(101, 1, 1), 0, view) == doctest::Approx(100.0 * std::sqrt(0.5) - r0)); // 45 degrees out
    CHECK(ChunkMath::view_cone_gap(ChunkId(3, 1, 1), 0, view) == 0.0); // 45 degrees out, but the sphere reaches the cone
    const ChunkId far_behind(8, 8, 1000);
    CHECK(ChunkMath::view_cone_gap(far_behind, 3, view) ==
            doctest::Approx(view.camera_chunk.distance_to(far_behind) - ChunkMath::chunk_bounding_radius(3)));

    ScanView none = view;
    none.forward = Float3{};
    CHECK(ChunkMath::view_cone_gap(ChunkId(1, 1, 101), 0, none) == 0.0); // no direction: all in view

    // Past 90 degrees (a wide margin): the axis is inside, only the back cone is out.
    ScanView wide = view;
    wide.set_half_angle(godot::Math::deg_to_rad(120.0));
    CHECK(ChunkMath::view_cone_gap(ChunkId(1, 1, -99), 0, wide) == 0.0);
    CHECK(ChunkMath::view_cone_gap(ChunkId(101, 1, 1), 0, wide) == 0.0);
    CHECK(ChunkMath::view_cone_gap(ChunkId(1, 1, 101), 0, wide) == doctest::Approx(100.0 * std::sin(godot::Math::deg_to_rad(60.0)) - r0));
    // The default half-angle (pi): every chunk is in the cone.
    ScanView all = view;
    all.cos_half_angle = ScanView().cos_half_angle;
    all.sin_half_angle = ScanView().sin_half_angle;
    for (const ChunkId &c : { ChunkId(1, 1, -99), ChunkId(101, 1, 1), ChunkId(1, 1, 101) }) {
        CHECK(ChunkMath::view_cone_gap(c, 0, all) == 0.0);
    }

    // 1-Lipschitz in the centre: the LOD rule's balance relies on it.
    const godot::Vector3 f3 = godot::Vector3(0.3f, -0.5f, 0.8f).normalized();
    view.forward = ov_f3(f3);
    Lcg rng{ 12345 };
    auto next = [&rng]() { return rng.next(); };
    for (int i = 0; i < 2000; i++) {
        const ChunkId a(next() * 2 + 1, next() * 2 + 1, next() * 2 + 1);
        const ChunkId b = a + ChunkId(next() % 16 * 2, next() % 16 * 2, next() % 16 * 2);
        CAPTURE(a);
        CAPTURE(b);
        CHECK(std::abs(ChunkMath::view_cone_gap(a, 0, view) - ChunkMath::view_cone_gap(b, 0, view)) <= a.distance_to(b) + 1e-9);
    }
}

TEST_CASE("ChunkMath: lod_cone_gap is the least faded gap to the remembered cones") {
    ScanView view;
    view.camera_chunk = ChunkId(1, 1, 1);
    view.forward = Float3{ 0.0f, 0.0f, -1.0f };
    view.set_half_angle(godot::Math::deg_to_rad(70.0));
    CHECK(ChunkMath::lod_cone_gap(ChunkId(1, 1, 101), 0, view) == ChunkMath::view_cone_gap(ChunkId(1, 1, 101), 0, view)); // no memory
    // Views from the camera chunk and from spots it walked away from (sorted by fade, as remember_view does).
    const ScanView::ViewMemory memories[] = {
        { ChunkId(1, 1, 1), Float3{ 0.0f, 0.0f, -1.0f }, 0.0 },
        { ChunkId(1, 1, 1), Float3{ 1.0f, 0.0f, 0.0f }, 0.0 },
        { ChunkId(-39, 3, 41), Float3{ 0.3f, -0.8f, 0.5f }.normalized(), 10.0 },
        { ChunkId(201, -7, 1), Float3{ -0.6f, 0.1f, 0.7f }.normalized(), 150.0 },
    };
    for (const ScanView::ViewMemory &m : memories) {
        view.memories[view.memory_count++] = m;
    }
    Lcg rng{ 777 };
    auto next = [&rng]() { return rng.next(); };
    view.back_scale = 3.0;
    for (int i = 0; i < 2000; i++) {
        const ChunkId a(next() * 2 + 1, next() * 2 + 1, next() * 2 + 1);
        double want = INFINITY;
        for (const ScanView::ViewMemory &m : memories) {
            ScanView one = view;
            one.camera_chunk = m.apex;
            one.forward = m.dir;
            want = std::min(want, m.fade + ChunkMath::view_cone_gap(a, 0, one));
        }
        CAPTURE(a);
        CHECK(ChunkMath::lod_cone_gap(a, 0, view) == want);
        // Still 1-Lipschitz: the LOD rule's one-LOD balance relies on it.
        const ChunkId b = a + ChunkId(next() % 16 * 2, next() % 16 * 2, next() % 16 * 2);
        CHECK(std::abs(ChunkMath::lod_cone_gap(a, 0, view) - ChunkMath::lod_cone_gap(b, 0, view)) <= a.distance_to(b) + 1e-9);
        // lod_split's early exits give the comparison of the full minimum, bit for bit.
        for (int lod = 0; lod < 4; lod++) {
            const double threshold = double(next() + 1000) * 3.0;
            CHECK(ChunkMath::lod_split(a, lod, view, threshold) == (ChunkMath::lod_distance(a, lod, view) < threshold));
        }
    }
}

TEST_CASE("ChunkMath: view_cone_half_angle is the diagonal half-FOV plus the margin") {
    // A 90 degree square view: the diagonal is at atan(sqrt(2)) off the axis.
    CHECK(ChunkMath::view_cone_half_angle(float(Math_PI / 2.0), 1.0f, 0.0f) == doctest::Approx(std::atan(std::sqrt(2.0))));
    CHECK(ChunkMath::view_cone_half_angle(float(Math_PI / 2.0), 1.0f, 0.25f) == doctest::Approx(std::atan(std::sqrt(2.0)) + 0.25));
}

TEST_CASE("ChunkMath: lod_distance grows away from the view cone, up to the back error") {
    ScanView view;
    view.camera_chunk = ChunkId(1, 1, 1);
    view.forward = Float3{ 0.0f, 0.0f, -1.0f };
    view.set_half_angle(WorldCoord::PI / 4.0);
    const ChunkId ahead(4, 4, -500);
    const ChunkId behind(4, 4, 500);
    const float r2 = ChunkMath::chunk_bounding_radius(2);
    // Off (back_scale 0): the plain distance, bit for bit what the discoverer compared before.
    CHECK(ChunkMath::lod_distance(behind, 2, view) == view.camera_chunk.distance_to(behind) - r2);
    view.back_scale = 3.0; // error 1 in the cone, 4 behind
    CHECK(ChunkMath::lod_distance(ahead, 2, view) == view.camera_chunk.distance_to(ahead) - r2);
    CHECK(ChunkMath::lod_distance(behind, 2, view) == doctest::Approx(4.0 * (view.camera_chunk.distance_to(behind) - r2)));
}

TEST_CASE("ChunkMath: far_ids rounds the far plane up to a power of two of chunk-id units") {
    ChunkParam param; // 0.5 m voxels: a chunk-id unit is 14 voxels, 7 m
    param.voxel_size = 0.5f;
    CHECK(ChunkMath::far_ids(0.0, param) == 0);
    CHECK(ChunkMath::far_ids(-5.0, param) == 0);
    CHECK(ChunkMath::far_ids(INFINITY, param) == 0);
    CHECK(ChunkMath::far_ids(NAN, param) == 0);
    CHECK(ChunkMath::far_ids(1.0, param) == 1);
    CHECK(ChunkMath::far_ids(7.0, param) == 1);
    CHECK(ChunkMath::far_ids(7.5, param) == 2);
    CHECK(ChunkMath::far_ids(12000.0, param) == 2048); // 1715 units
    CHECK(ChunkMath::far_ids(14336.0, param) == 2048); // exactly 2048 units
    CHECK(ChunkMath::far_ids(1e30, param) == 0); // past 2^62 units: no limit
}

TEST_CASE("ChunkMath: beyond_far holds for nodes wholly past the far distance") {
    ScanView view;
    view.camera_chunk = ChunkId(-37, 101, 5);
    const ChunkId cam = view.camera_chunk;
    CHECK(!ChunkMath::beyond_far(ChunkId(cam.x + (int64_t(1) << 40), cam.y, cam.z), 0, view)); // no far plane
    view.far_ids = 256;
    // A LOD0 node d units along x: its grown box starts 2 units before its centre, the camera may be 1 unit past
    // its own, so the gap is d - 3.
    CHECK(!ChunkMath::beyond_far(ChunkId(cam.x + 258, cam.y, cam.z), 0, view)); // gap 255
    CHECK(ChunkMath::beyond_far(ChunkId(cam.x + 260, cam.y, cam.z), 0, view)); // gap 257
    CHECK(ChunkMath::beyond_far(ChunkId(cam.x + 200, cam.y + 200, cam.z), 0, view)); // diagonal: 197 * sqrt(2) = 279 units
    CHECK(!ChunkMath::beyond_far(ChunkId(cam.x + 180, cam.y + 180, cam.z), 0, view)); // 177 * sqrt(2) = 250 units
    Lcg lcg{ 7 };
    for (int i = 0; i < 2000; i++) {
        const int lod = 1 + int((lcg.next() + 1000) % 12);
        const int64_t unit = int64_t(1) << lod;
        auto axis = [&](int64_t p_c) { return (WorldCoord::floor_div(p_c + lcg.next() * 4, 2 * unit) * 2 + 1) * unit; };
        const ChunkId node(axis(cam.x), axis(cam.y), axis(cam.z));
        CAPTURE(node);
        CAPTURE(lod);
        // The brute force: the nearest point of the node's grown box (twice its span) to the camera's LOD0 box.
        double sq = 0.0;
        for (int a = 0; a < 3; a++) {
            const double lo = double(node[a] - 2 * unit) - double(cam[a] + 1);
            const double hi = double(cam[a] - 1) - double(node[a] + 2 * unit);
            const double gap = std::max({ lo, hi, 0.0 });
            sq += gap * gap;
        }
        CHECK(ChunkMath::beyond_far(node, lod, view) == (std::sqrt(sq) > double(view.far_ids)));
    }
}

TEST_CASE("ChunkMath: load_queue puts the view cone first, then the finest LOD") {
    ScanView view;
    view.camera_chunk = ChunkId(1, 1, 1);
    view.forward = Float3{ 1.0f, 0.0f, 0.0f };
    view.set_half_angle(godot::Math::deg_to_rad(60.0));
    const int lods = 15;
    CHECK(ChunkMath::load_queue(ChunkId(41, 1, 1), lods, view) == 0);
    CHECK(ChunkMath::load_queue(ChunkId(1000, 8, 8), lods, view) == 3);
    CHECK(ChunkMath::load_queue(ChunkId(-39, 1, 1), lods, view) == lods);
    CHECK(ChunkMath::load_queue(ChunkId(-1000, 8, 8), lods, view) == lods + 3);
    CHECK(ChunkMath::load_queue(ChunkId(-39, 1, 1), lods, ScanView()) == 0); // no view: LOD order
}
