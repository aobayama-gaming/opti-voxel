#include "test_common.hpp"

#include "chunk_math.hpp"

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

TEST_CASE("ChunkMath: ancestor at a target LOD") {
    CHECK(ChunkMath::get_parent_from_child_until(ChunkId(1, 1, 1), 3) == ChunkId(8, 8, 8));
    CHECK(ChunkMath::get_parent_from_child_until(ChunkId(-1, 1, -3), 2) == ChunkId(-4, 4, -4));
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
    const Vector3 grid(5.5, 2.0, 30.0);
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
