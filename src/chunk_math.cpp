
#include "chunk_math.hpp"

#include <cmath>

using WorldCoord::ctz64;

int64_t ChunkMath::get_parent_from_child(int64_t coord, int &lod) {
    // Get the center coordinate of the parent chunk: clear the lowest set bit, set the next one.
    // For 5 (101) the LSB is at index 0, the parent is 6 (110); for 6 it is at index 1, the parent is 4.
    if (coord == 0) {
        return 0; // If coord is 0, its parent is 0.
    }
    const int lsb_index = ctz64(uint64_t(coord));
    if (lsb_index >= 62) {
        return coord; // past the top of the int64 octree
    }

    lod = lsb_index + 1; // +1 because we output the lod of the parent

    const uint64_t cleared_coord = uint64_t(coord) & ~(uint64_t(1) << lsb_index);
    return int64_t(cleared_coord | (uint64_t(1) << (lsb_index + 1)));
}

int64_t ChunkMath::get_parent_from_child(int64_t coord) {
    int dummy_lod;
    return get_parent_from_child(coord, dummy_lod);
}

ChunkId ChunkMath::get_parent_from_child(const ChunkId &coord, int &lod) {
    return ChunkId(
            get_parent_from_child(coord.x, lod),
            get_parent_from_child(coord.y),
            get_parent_from_child(coord.z));
}

ChunkId ChunkMath::get_parent_from_child(const ChunkId &coord) {
    return ChunkId(
            get_parent_from_child(coord.x),
            get_parent_from_child(coord.y),
            get_parent_from_child(coord.z));
}

int ChunkMath::get_lod(const ChunkId &coord) {
    // The LOD of a chunk is the index of the lowest set bit of its coordinates (x is enough).
    if (coord.x == 0) {
        return 0;
    }
    return ctz64(uint64_t(coord.x));
}

int64_t ChunkMath::get_lod_size(const ChunkId &coord) {
    return int64_t(1) << get_lod(coord);
}

ChunkId ChunkMath::get_parent_from_child_until(const ChunkId &coord, int target_lod) {
    // Get the parent chunk at the target LOD by repeatedly getting the parent.

    ChunkId parent = coord;
    int current_lod = -1;

    while (current_lod < target_lod && parent.x != 0) {
        parent = get_parent_from_child(parent, current_lod);
    }

    return parent;
}

float ChunkMath::world_chunk_size(const ChunkId &chunk_coord, const ChunkParam &chunk_param) {
    // The world size of a chunk is the LOD0 chunk size scaled by its LOD.
    return float(get_lod_size(chunk_coord)) * chunk_param.chunk_world_dimension();
}

VoxelPos ChunkMath::chunk_min_voxel(const ChunkId &chunk_coord, const ChunkParam &chunk_param) {
    const int64_t lod_size = get_lod_size(chunk_coord);
    return (chunk_coord - ChunkId(lod_size, lod_size, lod_size)) * chunk_param.half_chunk_voxels();
}

VoxelPos ChunkMath::chunk_center_voxel(const ChunkId &chunk_coord, const ChunkParam &chunk_param) {
    return chunk_coord * chunk_param.half_chunk_voxels();
}

namespace {

// Render position of a vertex-grid point, from the chunk's minimum corner in voxels relative to the
// origin (exact integers, small). Double precision: the result is converted to Godot's real_t last.
Vector3 _grid_point_render(const Vector3i64 &p_min_rel_voxel, int64_t p_lod_size, const Vector3 &p_grid, const ChunkParam &p_param) {
    const double ls = double(p_lod_size);
    const double vs = double(p_param.voxel_size);
    const double skirt = double(p_param.skirt_size);
    return Vector3(
            real_t((double(p_min_rel_voxel.x) + (double(p_grid.x) - skirt) * ls) * vs),
            real_t((double(p_min_rel_voxel.y) + (double(p_grid.y) - skirt) * ls) * vs),
            real_t((double(p_min_rel_voxel.z) + (double(p_grid.z) - skirt) * ls) * vs));
}

} // namespace

Vector3 ChunkMath::chunk_render_center(const ChunkId &chunk_coord, const ChunkParam &chunk_param) {
    const Vector3i64 rel = chunk_center_voxel(chunk_coord, chunk_param) - chunk_param.origin_voxel();
    const double vs = double(chunk_param.voxel_size);
    return Vector3(real_t(double(rel.x) * vs), real_t(double(rel.y) * vs), real_t(double(rel.z) * vs));
}

Vector3 ChunkMath::grid_to_render(const ChunkId &chunk_coord, const Vector3 &grid, const ChunkParam &chunk_param) {
    const Vector3i64 min_rel = chunk_min_voxel(chunk_coord, chunk_param) - chunk_param.origin_voxel();
    return _grid_point_render(min_rel, get_lod_size(chunk_coord), grid, chunk_param);
}

Vector3 ChunkMath::grid_to_render_lo32(const Vector3i &chunk_lo, const Vector3 &grid, const ChunkParam &chunk_param) {
    using namespace WorldCoord;
    // Wrapping difference of the low words: exact while the chunk is within 2^31 ids of the origin.
    const Vector3i64 rel(
            as_i32(lo32(chunk_lo.x) - lo32(chunk_param.origin_chunk.x)),
            as_i32(lo32(chunk_lo.y) - lo32(chunk_param.origin_chunk.y)),
            as_i32(lo32(chunk_lo.z) - lo32(chunk_param.origin_chunk.z)));
    // The origin is a multiple of the root span, so the low word keeps the LOD bit.
    const int64_t lod_size = chunk_lo.x == 0 ? 1 : int64_t(1) << ctz64(uint64_t(lo32(chunk_lo.x)));
    const Vector3i64 min_rel = (rel - Vector3i64(lod_size, lod_size, lod_size)) * chunk_param.half_chunk_voxels();
    return _grid_point_render(min_rel, lod_size, grid, chunk_param);
}

ChunkId ChunkMath::chunk_at_render(const Vector3 &render_pos, const ChunkParam &chunk_param) {
    // Cells of one LOD0 chunk, relative to the origin; LOD0 chunk centers are odd: ..., -3, -1, 1, 3, ...
    const double dim = double(chunk_param.chunk_world_dimension());
    auto cell = [dim](real_t p_v) {
        const double c = std::floor(double(p_v) / dim);
        return int64_t(CLAMP(c, -1e15, 1e15));
    };
    return chunk_param.origin_chunk + ChunkId(cell(render_pos.x) * 2 + 1, cell(render_pos.y) * 2 + 1, cell(render_pos.z) * 2 + 1);
}

// The float operations below mirror gpu_frame.hpp one by one, in float32, as separate statements
// (built with -ffp-contract=off, see SConstruct; the GPU side uses `precise`).

WorldPos ChunkMath::sample_pos(const ChunkId &chunk_coord, const Float3 &grid, const ChunkParam &chunk_param) {
    const float lod_size = float(get_lod_size(chunk_coord));
    const float skirt = float(chunk_param.skirt_size);
    const VoxelPos anchor = chunk_min_voxel(chunk_coord, chunk_param);
    auto axis = [lod_size, skirt](float p_grid, int64_t p_anchor, float &r_frac) {
        const float d = p_grid - skirt;
        const float off = d * lod_size;
        const float fl = std::floor(off);
        r_frac = off - fl;
        return p_anchor + int64_t(fl);
    };
    WorldPos pos;
    pos.voxel.x = axis(grid.x, anchor.x, pos.frac.x);
    pos.voxel.y = axis(grid.y, anchor.y, pos.frac.y);
    pos.voxel.z = axis(grid.z, anchor.z, pos.frac.z);
    return pos;
}
