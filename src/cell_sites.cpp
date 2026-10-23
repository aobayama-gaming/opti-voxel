// Procedural sites on the ground (cell_sites.hpp).
#include "cell_sites.hpp"

#include "parallel_for.hpp"

#include <algorithm>
#include <cmath>

namespace Archetype {

int64_t forest_cell_side(int p_count, double p_area_m, double p_voxel) {
    if (p_count <= 0 || !(p_area_m > 0.0)) {
        return 0;
    }
    return std::max<int64_t>(1, std::llround(p_area_m / std::sqrt(double(p_count)) / p_voxel));
}

int site_lod(int64_t p_side) {
    int lod = 0;
    while (lod < SITE_LOD_MAX && (int64_t(3) << (lod + 1)) <= p_side) {
        lod++;
    }
    return lod;
}

namespace {

// The coarse field at the centre of node p_node (in units of its side p_n) of a walk over cells of p_side.
ExactNoise::Float4 centre_field(const VoxelPos &p_node, int64_t p_n, int64_t p_side, const FieldFn &p_field) {
    return p_field(p_node * p_n + VoxelPos(p_n / 2, p_n / 2, p_n / 2), site_lod(p_side));
}

// Whether the plane of p_f (the field at a node's centre, extended along its gradient) crosses the node of side p_n,
// give or take SITE_SLACK cells of p_side and SITE_BEND of the node (the ground bends away from the plane).
bool plane_meets(const ExactNoise::Float4 &p_f, int64_t p_n, int64_t p_side) {
    const double reach = 0.5 * double(p_n) * (std::fabs(double(p_f.x)) + std::fabs(double(p_f.y)) + std::fabs(double(p_f.z)));
    return std::fabs(double(p_f.w)) <= reach + SITE_SLACK * double(p_side) + SITE_BEND * double(p_n);
}

// A Newton step on whole voxels from r_q for the field p_f read at p_at (extended along its gradient to r_q); false when
// it would go over two cell sides (or the gradient is flat).
bool newton_step(const ExactNoise::Float4 &p_f, const VoxelPos &p_at, int64_t p_side, VoxelPos &r_q) {
    const double g2 = double(p_f.x) * p_f.x + double(p_f.y) * p_f.y + double(p_f.z) * p_f.z;
    const VoxelPos d = r_q - p_at;
    const double s = -(double(p_f.w) + double(p_f.x) * double(d.x) + double(p_f.y) * double(d.y) + double(p_f.z) * double(d.z)) / g2;
    if (!(g2 > 0.0) || !(std::fabs(s) * std::sqrt(g2) <= 2.0 * double(p_side))) {
        return false;
    }
    r_q += VoxelPos(std::llround(s * p_f.x), std::llround(s * p_f.y), std::llround(s * p_f.z));
    return true;
}

// cell_site with the coarse field at the cell's centre given (the walk read it); without p_snap the site stays on the
// coarse plane.
bool site_of(const VoxelPos &p_cell, int64_t p_side, uint32_t p_salt, const ExactNoise::Float4 &p_centre, const FieldFn &p_field, bool p_snap,
        Site &r_site) {
    const uint32_t key = ExactNoise::hash_cell(p_cell, p_salt);
    const ExactNoise::UVec3 h = ExactNoise::hash3(key, 0x51E5u, 0u);
    const ExactNoise::UVec3 g = ExactNoise::hash3(key, 0x51E5u, 1u);
    const uint32_t u[3] = { h.x, h.y, g.x };
    const VoxelBox cell{ p_cell * p_side, p_cell * p_side + VoxelPos(p_side - 1, p_side - 1, p_side - 1), true };
    // The middle part spares the axis closest to the ground's normal.
    const float n[3] = { std::fabs(p_centre.x), std::fabs(p_centre.y), std::fabs(p_centre.z) };
    const int normal_axis = n[1] >= n[0] && n[1] >= n[2] ? 1 : (n[0] >= n[2] ? 0 : 2);
    const int64_t margin = std::llround(SITE_INNER * double(p_side));
    VoxelBox middle = cell;
    for (int a = 0; a < 3; a++) {
        if (a != normal_axis) {
            middle.lo[a] += margin;
            middle.hi[a] -= margin;
        }
    }
    VoxelPos q;
    for (int a = 0; a < 3; a++) {
        q[a] = cell.lo[a] + p_side / 2 + std::llround((ExactNoise::unit_of(u[a]) - 0.5) * 0.7 * double(p_side));
    }
    // Whether the site exists: the hashed voxel projected on the coarse plane at the cell's centre lands in the middle part.
    if (!newton_step(p_centre, cell.lo + VoxelPos(p_side / 2, p_side / 2, p_side / 2), p_side, q) || !middle.contains(q)) {
        return false;
    }
    // Snapped onto the LOD-0 ground, within the cell.
    if (p_snap) {
        ExactNoise::Float4 f = p_field(q, 0);
        for (int i = 0; i < SNAP_STEPS && !(std::fabs(f.w) <= 1.0f); i++) {
            if (!newton_step(f, q, p_side, q)) {
                return false;
            }
            f = p_field(q, 0);
        }
        if (!(std::fabs(f.w) <= 1.0f) || !cell.contains(q)) {
            return false;
        }
    }
    r_site.base = q;
    r_site.seed = h.z;
    r_site.variant = h.z >> 16;
    return true;
}

const VoxelPos CORNERS[8] = { VoxelPos(0, 0, 0), VoxelPos(1, 0, 0), VoxelPos(0, 1, 0), VoxelPos(1, 1, 0), VoxelPos(0, 0, 1), VoxelPos(1, 0, 1),
    VoxelPos(0, 1, 1), VoxelPos(1, 1, 1) };

// The nodes of side p_n (voxels) meeting p_box, in node units.
std::vector<VoxelPos> nodes_meeting(const VoxelBox &p_box, int64_t p_n) {
    VoxelPos a, b;
    for (int i = 0; i < 3; i++) {
        a[i] = WorldCoord::floor_div(p_box.lo[i], p_n);
        b[i] = WorldCoord::floor_div(p_box.hi[i], p_n);
    }
    std::vector<VoxelPos> nodes;
    for (int64_t z = a.z; z <= b.z; z++) {
        for (int64_t y = a.y; y <= b.y; y++) {
            for (int64_t x = a.x; x <= b.x; x++) {
                nodes.emplace_back(x, y, z);
            }
        }
    }
    return nodes;
}

// A cell the walk kept, with the coarse field at its centre.
struct WalkCell {
    VoxelPos cell;
    ExactNoise::Float4 centre;
};

// The nodes of p_side << p_stop_level voxels meeting p_box that the surface may cross, in cell order: an absolute octree
// walk from the coarsest level whose node holds the box (at least p_min_level), keeping a node while the coarse plane at
// its centre crosses it (plane_meets). Every node meeting the box when the walk would start at p_stop_level or below (no
// reading: their centres' fields are left unset).
std::vector<WalkCell> surface_cells(const VoxelBox &p_box, int64_t p_side, const FieldFn &p_field, int p_min_level, int p_stop_level = 0) {
    int level = std::max(walk_level(p_box, p_side), std::min(p_min_level, 62));
    if (p_stop_level > 0 && level <= p_stop_level) {
        std::vector<WalkCell> all;
        for (const VoxelPos &n : nodes_meeting(p_box, p_side << p_stop_level)) {
            all.push_back(WalkCell{ n, ExactNoise::Float4() });
        }
        return all; // nodes_meeting's order is z y x: cell order
    }
    std::vector<VoxelPos> nodes = nodes_meeting(p_box, p_side << level);
    std::vector<WalkCell> kept;
    std::vector<ExactNoise::Float4> fields;
    for (;; level--) {
        const int64_t n = p_side << level;
        fields.resize(nodes.size());
        parallel_for(uint32_t(nodes.size()), "surface_cells", [&](uint32_t p_i) { fields[p_i] = centre_field(nodes[p_i], n, p_side, p_field); });
        kept.clear();
        for (size_t i = 0; i < nodes.size(); i++) {
            if (plane_meets(fields[i], n, p_side)) {
                kept.push_back(WalkCell{ nodes[i], fields[i] });
            }
        }
        if (level == p_stop_level) {
            break;
        }
        nodes.clear();
        const int64_t m = n / 2;
        for (const WalkCell &k : kept) {
            for (const VoxelPos &c : CORNERS) {
                const VoxelPos child = k.cell * 2 + c;
                if (VoxelBox{ child * m, child * m + VoxelPos(m - 1, m - 1, m - 1), true }.intersects(p_box)) {
                    nodes.push_back(child);
                }
            }
        }
    }
    std::sort(kept.begin(), kept.end(), [](const WalkCell &p_a, const WalkCell &p_b) { return cell_order(p_a.cell, p_b.cell); });
    return kept;
}

} // namespace

int walk_level(const VoxelBox &p_box, int64_t p_side) {
    int64_t extent = 1;
    for (int a = 0; a < 3; a++) {
        extent = std::max(extent, p_box.hi[a] - p_box.lo[a] + 1);
    }
    int level = 0; // p_side << level >= extent, without overflowing
    while (level < 62 && ((extent - 1) >> level) >= p_side) {
        level++;
    }
    return level;
}

void surface_nodes(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, const FieldFn &p_field, int p_level, int p_min_level,
        std::vector<VoxelPos> &r_out) {
    if (p_side <= 0 || p_level < 0 || p_level > 62) {
        return;
    }
    for (const WalkCell &c : surface_cells(VoxelBox{ p_lo, p_hi, true }, p_side, p_field, p_min_level, p_level)) {
        r_out.push_back(c.cell);
    }
}

bool cell_site(const VoxelPos &p_cell, int64_t p_side, uint32_t p_salt, const FieldFn &p_field, Site &r_site) {
    return site_of(p_cell, p_side, p_salt, centre_field(p_cell, p_side, p_side, p_field), p_field, true, r_site);
}

void surface_sites(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, uint32_t p_salt, const FieldFn &p_field, std::vector<CellSite> &r_out,
        bool p_snap, int p_min_level) {
    if (p_side <= 0) {
        return;
    }
    const VoxelBox box{ p_lo, p_hi, true };
    const std::vector<WalkCell> cells = surface_cells(box, p_side, p_field, p_min_level);
    std::vector<Site> found(cells.size());
    std::vector<uint8_t> ok(cells.size(), 0);
    parallel_for(uint32_t(cells.size()), "surface_sites",
            [&](uint32_t p_i) { ok[p_i] = site_of(cells[p_i].cell, p_side, p_salt, cells[p_i].centre, p_field, p_snap, found[p_i]); });
    for (size_t i = 0; i < cells.size(); i++) {
        if (ok[i] && box.contains(found[i].base)) {
            r_out.push_back(CellSite{ cells[i].cell, found[i] });
        }
    }
}

} // namespace Archetype
