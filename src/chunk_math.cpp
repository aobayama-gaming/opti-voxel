
#include "chunk_math.hpp"

#include "ov/on_error.hpp"
#include "subcube.hpp"

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

ov::ivec2 ChunkMath::lod_range(const std::vector<ChunkId> &p_chunks) {
    if (p_chunks.empty()) {
        return ov::ivec2();
    }
    ov::ivec2 r(get_lod(p_chunks[0]), get_lod(p_chunks[0]));
    for (const ChunkId &c : p_chunks) {
        const int lod = get_lod(c);
        r = ov::ivec2(std::min(r.x, lod), std::max(r.y, lod));
    }
    return r;
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

uint64_t ChunkMath::subcube_bits(const ChunkId &p_ancestor, const ChunkId &p_descendant) {
    const int la = get_lod(p_ancestor);
    const int ld = get_lod(p_descendant);
    OV_ERR_FAIL_COND_V_MSG(ld >= la || int(Subcube::AXIS_BITS) > la + 1, 0, "subcube_bits: not a descendant at least AXIS_BITS finer.");
    OV_ERR_FAIL_COND_V_MSG(get_parent_from_child_until(p_descendant, la) != p_ancestor, 0, "subcube_bits: not an ancestor.");
    // Units from the ancestor's low corner, in [0, 2^(la+1)); a sub-cube spans 2^shift of them.
    const int shift = la + 1 - int(Subcube::AXIS_BITS);
    const int64_t half = get_lod_size(p_descendant);
    const int64_t a_half = get_lod_size(p_ancestor);
    const ChunkId local = p_descendant - p_ancestor + ChunkId(a_half, a_half, a_half);
    uint32_t lo[3], hi[3];
    for (int a = 0; a < 3; a++) {
        const int64_t v = local[a];
        lo[a] = uint32_t((v - half) >> shift);
        hi[a] = uint32_t((v + half - 1) >> shift);
    }
    uint64_t bits = 0;
    for (uint32_t z = lo[2]; z <= hi[2]; z++) {
        for (uint32_t y = lo[1]; y <= hi[1]; y++) {
            for (uint32_t x = lo[0]; x <= hi[0]; x++) {
                bits |= uint64_t(1) << Subcube::bit(x, y, z);
            }
        }
    }
    return bits;
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

ChunkId ChunkMath::chunk_at_voxel(const VoxelPos &p_voxel, int p_lod, const ChunkParam &chunk_param) {
    const int64_t span = int64_t(chunk_param.chunk_voxel_dimension()) << p_lod;
    const int64_t unit = int64_t(1) << p_lod;
    auto axis = [&](int64_t p_v) { return (WorldCoord::floor_div(p_v, span) * 2 + 1) * unit; };
    return ChunkId(axis(p_voxel.x), axis(p_voxel.y), axis(p_voxel.z));
}

VoxelPos ChunkMath::chunk_center_voxel(const ChunkId &chunk_coord, const ChunkParam &chunk_param) {
    return chunk_coord * chunk_param.half_chunk_voxels();
}

namespace {

// Render position of a vertex-grid point, from the chunk's minimum corner in voxels relative to the
// origin (exact integers, small). Double precision: the result is converted to float32 last.
Float3 _grid_point_render(const Vector3i64 &p_min_rel_voxel, int64_t p_lod_size, const Float3 &p_grid, const ChunkParam &p_param) {
    const double ls = double(p_lod_size);
    const double vs = double(p_param.voxel_size);
    const double skirt = double(p_param.skirt_size);
    return Float3{
        float((double(p_min_rel_voxel.x) + (double(p_grid.x) - skirt) * ls) * vs),
        float((double(p_min_rel_voxel.y) + (double(p_grid.y) - skirt) * ls) * vs),
        float((double(p_min_rel_voxel.z) + (double(p_grid.z) - skirt) * ls) * vs),
    };
}

} // namespace

Float3 ChunkMath::chunk_render_center(const ChunkId &chunk_coord, const ChunkParam &chunk_param) {
    return voxel_to_render(chunk_center_voxel(chunk_coord, chunk_param), chunk_param);
}

Float3 ChunkMath::voxel_to_render(const VoxelPos &p_voxel, const ChunkParam &chunk_param) {
    const Vector3i64 rel = p_voxel - chunk_param.origin_voxel();
    const double vs = double(chunk_param.voxel_size);
    return Float3{ float(double(rel.x) * vs), float(double(rel.y) * vs), float(double(rel.z) * vs) };
}

Float3 ChunkMath::grid_to_render(const ChunkId &chunk_coord, const Float3 &grid, const ChunkParam &chunk_param) {
    const Vector3i64 min_rel = chunk_min_voxel(chunk_coord, chunk_param) - chunk_param.origin_voxel();
    return _grid_point_render(min_rel, get_lod_size(chunk_coord), grid, chunk_param);
}

Float3 ChunkMath::grid_to_render_lo32(const ov::ivec3 &chunk_lo, const Float3 &grid, const ChunkParam &chunk_param) {
    using namespace WorldCoord;
    // Wrapping difference of the low words: exact while the chunk is within 2^31 ids of the origin.
    const Vector3i64 rel(
            as_i32(lo32(chunk_lo.x) - lo32(chunk_param.origin_chunk.x)),
            as_i32(lo32(chunk_lo.y) - lo32(chunk_param.origin_chunk.y)),
            as_i32(lo32(chunk_lo.z) - lo32(chunk_param.origin_chunk.z)));
    // The LOD from the id's own low word (absolute, whatever the origin).
    const int64_t lod_size = chunk_lo.x == 0 ? 1 : int64_t(1) << ctz64(uint64_t(lo32(chunk_lo.x)));
    const Vector3i64 min_rel = (rel - Vector3i64(lod_size, lod_size, lod_size)) * chunk_param.half_chunk_voxels();
    return _grid_point_render(min_rel, lod_size, grid, chunk_param);
}

ChunkId ChunkMath::chunk_at_render(const Float3 &render_pos, const ChunkParam &chunk_param) {
    // Cells of one LOD0 chunk, relative to the origin; LOD0 chunk centers are odd: ..., -3, -1, 1, 3, ...
    const double dim = double(chunk_param.chunk_world_dimension());
    auto cell = [dim](float p_v) {
        const double c = std::floor(double(p_v) / dim);
        return int64_t(std::clamp(c, -1e15, 1e15));
    };
    return chunk_param.origin_chunk + ChunkId(cell(render_pos.x) * 2 + 1, cell(render_pos.y) * 2 + 1, cell(render_pos.z) * 2 + 1);
}

bool ChunkMath::is_render_origin(const ChunkId &p_chunk) {
    return ((p_chunk.x | p_chunk.y | p_chunk.z) & 1) == 0;
}

ChunkId ChunkMath::recentered_origin(const Float3 &p_render_pos, const ChunkParam &chunk_param) {
    return chunk_at_render(p_render_pos, chunk_param) - ChunkId(1, 1, 1);
}

Float3 ChunkMath::origin_shift_m(const ChunkId &p_from, const ChunkId &p_to, const ChunkParam &chunk_param) {
    const ChunkId d = p_to - p_from;
    const double m = chunk_param.id_unit_m();
    return Float3{ float(double(d.x) * m), float(double(d.y) * m), float(double(d.z) * m) };
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

float ChunkMath::chunk_bounding_radius(int p_lod) {
    return std::sqrt(3.0f) * static_cast<float>(int64_t(1) << p_lod);
}

namespace {
// Gap from the bounding sphere of a LOD-p_lod chunk, at p_v from the apex, to the cone around unit axis p_f.
double cone_gap(double vx, double vy, double vz, const Float3 &p_f, int p_lod, const ScanView &p_view) {
    const double fx = p_f.x;
    const double fy = p_f.y;
    const double fz = p_f.z;
    const double along = vx * fx + vy * fy + vz * fz;
    // Distance to the axis from |v x f|, accurate near the axis.
    const double cx = vy * fz - vz * fy;
    const double cy = vz * fx - vx * fz;
    const double cz = vx * fy - vy * fx;
    const double perp = std::sqrt(cx * cx + cy * cy + cz * cz);
    // At an angle a off the axis and distance d, past_sin / past_cos are d sin(a - h) and d cos(a - h) for a
    // cone of half-angle h. Inside (a <= h) the gap is 0; outside, d sin(a - h), or d once a - h passes 90
    // degrees (the apex is then the nearest point). a - h is checked through its sine first: with h above 90
    // degrees a point near the axis also has cos(a - h) < 0.
    const double past_sin = perp * p_view.cos_half_angle - along * p_view.sin_half_angle;
    if (past_sin <= 0.0) {
        return 0.0;
    }
    const double past_cos = along * p_view.cos_half_angle + perp * p_view.sin_half_angle;
    const double gap = (past_cos > 0.0 ? past_sin : std::sqrt(vx * vx + vy * vy + vz * vz)) - ChunkMath::chunk_bounding_radius(p_lod);
    return gap > 0.0 ? gap : 0.0;
}
} //namespace

double ChunkMath::view_cone_gap(const ChunkId &p_chunk, int p_lod, const ScanView &p_view) {
    if (p_view.forward == Float3{}) {
        return 0.0;
    }
    return cone_gap(double(p_chunk.x - p_view.camera_chunk.x), double(p_chunk.y - p_view.camera_chunk.y), double(p_chunk.z - p_view.camera_chunk.z),
            p_view.forward, p_lod, p_view);
}

namespace {
// One term of lod_cone_gap: the memory's fade plus the gap to its cone.
double memory_gap(const ChunkId &p_chunk, int p_lod, const ScanView &p_view, const ScanView::ViewMemory &p_memory) {
    const ChunkId &apex = p_memory.apex;
    return p_memory.fade +
            cone_gap(double(p_chunk.x - apex.x), double(p_chunk.y - apex.y), double(p_chunk.z - apex.z), p_memory.dir, p_lod, p_view);
}

// lod_distance's plain part: from the camera chunk to the chunk's bounding sphere.
double sphere_distance(const ChunkId &p_chunk, int p_lod, const ScanView &p_view) {
    return p_view.camera_chunk.distance_to(p_chunk) - ChunkMath::chunk_bounding_radius(p_lod);
}
} //namespace

double ChunkMath::lod_cone_gap(const ChunkId &p_chunk, int p_lod, const ScanView &p_view) {
    if (p_view.memory_count == 0) {
        return view_cone_gap(p_chunk, p_lod, p_view);
    }
    double best = INFINITY;
    // Sorted by fade: once a fade alone reaches the best gap, no later memory can beat it.
    for (int i = 0; i < p_view.memory_count && p_view.memories[i].fade < best; i++) {
        best = std::min(best, memory_gap(p_chunk, p_lod, p_view, p_view.memories[i]));
    }
    return best;
}

double ChunkMath::lod_distance(const ChunkId &p_chunk, int p_lod, const ScanView &p_view) {
    const double near = sphere_distance(p_chunk, p_lod, p_view);
    if (p_view.back_scale == 0.0) {
        return near;
    }
    return near + p_view.back_scale * lod_cone_gap(p_chunk, p_lod, p_view);
}

bool ChunkMath::lod_split(const ChunkId &p_chunk, int p_lod, const ScanView &p_view, double p_threshold) {
    const double near = sphere_distance(p_chunk, p_lod, p_view);
    if (near >= p_threshold) {
        return false; // lod_distance is never under the plain distance
    }
    if (p_view.back_scale == 0.0 || p_view.memory_count == 0) {
        return lod_distance(p_chunk, p_lod, p_view) < p_threshold;
    }
    // near + back_scale * x only grows with x, also in floating point: the minimum is under the threshold
    // iff one term is, in any order. First the camera chunk's own view nearest in angle (found by dot products
    // alone; the gap grows with the angle): inside any of its cones, that one exact gap decides.
    const double vx = double(p_chunk.x - p_view.camera_chunk.x);
    const double vy = double(p_chunk.y - p_view.camera_chunk.y);
    const double vz = double(p_chunk.z - p_view.camera_chunk.z);
    int nearest = -1;
    double nearest_dot = -INFINITY;
    for (int i = 0; i < p_view.memory_count; i++) {
        const ScanView::ViewMemory &m = p_view.memories[i];
        const double dot = vx * m.dir.x + vy * m.dir.y + vz * m.dir.z;
        if (m.apex == p_view.camera_chunk && dot > nearest_dot) {
            nearest_dot = dot;
            nearest = i;
        }
    }
    if (nearest >= 0 && near + p_view.back_scale * memory_gap(p_chunk, p_lod, p_view, p_view.memories[nearest]) < p_threshold) {
        return true;
    }
    // Then every memory, by fade: once a fade alone fails, the later (more faded) memories fail too.
    for (int i = 0; i < p_view.memory_count && near + p_view.back_scale * p_view.memories[i].fade < p_threshold; i++) {
        if (near + p_view.back_scale * memory_gap(p_chunk, p_lod, p_view, p_view.memories[i]) < p_threshold) {
            return true;
        }
    }
    return false;
}

int64_t ChunkMath::far_ids(double p_far_m, const ChunkParam &p_param) {
    const double unit_m = p_param.id_unit_m();
    if (!(p_far_m > 0.0) || !std::isfinite(p_far_m) || !(unit_m > 0.0)) {
        return 0;
    }
    const double ids = std::ceil(p_far_m / unit_m);
    int k = std::ilogb(ids); // 2^k <= ids < 2^(k + 1)
    k += std::ldexp(1.0, k) < ids ? 1 : 0;
    return k > 62 ? 0 : int64_t(1) << k;
}

bool ChunkMath::beyond_far(const ChunkId &p_chunk, int p_lod, const ScanView &p_view) {
    if (p_view.far_ids <= 0) {
        return false;
    }
    // Per axis, the gap between the chunk's box grown by its half extent 2^lod (its mesh reaches past its bounds
    // by skirt_size + skirt_depth cells, a few of the 2 * half_chunk_voxels across it) and the camera's LOD0 chunk
    // (half extent 1).
    const int64_t reach = (int64_t(2) << p_lod) + 1;
    double sq = 0.0;
    for (int a = 0; a < 3; a++) {
        const int64_t d = p_chunk[a] - p_view.camera_chunk[a];
        const double gap = double(d < 0 ? -d : d) - double(reach);
        sq += gap > 0.0 ? gap * gap : 0.0;
    }
    const double far = double(p_view.far_ids);
    return sq > far * far;
}

float ChunkMath::view_cone_half_angle(float p_fov_rad, float p_aspect, float p_margin_rad) {
    return std::atan(std::tan(p_fov_rad * 0.5f) * std::sqrt(1.0f + p_aspect * p_aspect)) + p_margin_rad;
}

int ChunkMath::load_queue(const ChunkId &p_chunk, int p_lod_count, const ScanView &p_view) {
    const int lod = get_lod(p_chunk);
    return (view_cone_gap(p_chunk, lod, p_view) > 0.0 ? p_lod_count : 0) + lod;
}
