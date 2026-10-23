// Procedural sites on the ground (cell_sites.hpp).
#include "cell_sites.hpp"

#include "ov/containers.hpp"
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

// The centre of node p_node (in units of its side p_n): where the walk reads the coarse field.
VoxelPos node_centre(const VoxelPos &p_node, int64_t p_n) {
    return p_node * p_n + VoxelPos(p_n / 2, p_n / 2, p_n / 2);
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

// The site of p_cell before the snap, from the coarse field at the cell's centre (the walk read it): the hashed voxel
// projected on that plane, which must land in the cell's middle part (it exists), with the cell's seed and variant.
bool site_candidate(const VoxelPos &p_cell, int64_t p_side, uint32_t p_salt, const ExactNoise::Float4 &p_centre, Site &r_site) {
    const uint32_t key = ExactNoise::hash_cell(p_cell, p_salt);
    const ExactNoise::UVec3 h = ExactNoise::hash3(key, 0x51E5u, 0u);
    const ExactNoise::UVec3 g = ExactNoise::hash3(key, 0x51E5u, 1u);
    const uint32_t u[3] = { h.x, h.y, g.x };
    const VoxelBox cell = VoxelBox::cube(p_cell, p_side);
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
    if (!newton_step(p_centre, cell.lo + VoxelPos(p_side / 2, p_side / 2, p_side / 2), p_side, q) || !middle.contains(q)) {
        return false;
    }
    r_site.base = q;
    r_site.seed = h.z;
    r_site.variant = h.z >> 16;
    return true;
}

// A reading: a voxel as a LOD sees it.
struct Reading {
    VoxelPos voxel;
    int lod = 0;
    bool operator==(const Reading &p_o) const { return voxel == p_o.voxel && lod == p_o.lod; }
};
struct ReadingHasher {
    static uint32_t hash(const Reading &p_r) { return ov::murmur3_one_32(uint32_t(p_r.lod), Vector3i64Hasher::hash(p_r.voxel)); }
};

// One round's readings, read in one call of the field. ask() reads a (voxel, LOD) once however many walks ask for it
// (several walks' shared ancestors); push() doesn't look (one walk alone, the snap's bases: nothing repeats).
class Readings {
public:
    uint32_t ask(const VoxelPos &p_voxel, int p_lod) {
        asked++;
        if (const uint32_t *known = index.getptr(Reading{ p_voxel, p_lod })) {
            return *known;
        }
        index.insert(Reading{ p_voxel, p_lod }, uint32_t(points.size()));
        return push_point(p_voxel, p_lod);
    }
    uint32_t push(const VoxelPos &p_voxel, int p_lod) {
        asked++;
        return push_point(p_voxel, p_lod);
    }
    // Reads what was asked (no call for nothing), then the next round starts empty (capacity kept).
    void read(const FieldBatchFn &p_field, WalkStats *r_stats) {
        values.resize(points.size());
        if (!points.empty()) {
            p_field(points.data(), lods.data(), int(points.size()), values.data());
        }
        if (r_stats) {
            r_stats->rounds += points.empty() ? 0 : 1;
            r_stats->readings += asked;
            r_stats->unique += int64_t(points.size());
        }
        index.clear();
        points.clear();
        lods.clear();
        asked = 0;
    }
    const ExactNoise::Float4 &value(uint32_t p_i) const { return values[p_i]; }

private:
    ov::HashMap<Reading, uint32_t, ReadingHasher> index;
    std::vector<VoxelPos> points;
    std::vector<int> lods;
    std::vector<ExactNoise::Float4> values;
    int64_t asked = 0;

    uint32_t push_point(const VoxelPos &p_voxel, int p_lod) {
        points.push_back(p_voxel);
        lods.push_back(p_lod);
        return uint32_t(points.size() - 1);
    }
};

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

// One box's walk of the nodes of side << stop meeting it that the surface may cross: an absolute octree walk from the
// coarsest level whose node holds the box (at least the start level asked), keeping a node while the coarse plane at
// its centre crosses it (plane_meets). Once done, kept holds its cells in cell order. Walked from the stop level or
// below, every node meeting the box, with no reading (their centres' fields left unset).
struct CellWalk {
    VoxelBox box;
    int64_t side = 0;
    int level = 0;
    int stop = 0;
    std::vector<VoxelPos> nodes; // the level's
    std::vector<uint32_t> asked; // their readings in the round
    std::vector<WalkCell> kept;
    bool done = false;

    CellWalk(const VoxelBox &p_box, int64_t p_side, int p_min_level, int p_stop_level) : box(p_box), side(p_side), stop(p_stop_level) {
        if (p_side <= 0) {
            done = true;
            return;
        }
        level = std::max(walk_level(p_box, p_side), std::min(p_min_level, 62));
        if (p_stop_level > 0 && level <= p_stop_level) {
            for (const VoxelPos &n : nodes_meeting(p_box, p_side << p_stop_level)) {
                kept.push_back(WalkCell{ n, ExactNoise::Float4() }); // nodes_meeting's order is z y x: cell order
            }
            done = true;
            return;
        }
        nodes = nodes_meeting(p_box, p_side << level);
    }

    // p_shared: other walks share the round (their readings may coincide with ours: an ancestor of neighbouring tiles).
    void ask(Readings &r_readings, bool p_shared) {
        const int64_t n = side << level;
        asked.resize(nodes.size());
        for (size_t i = 0; i < nodes.size(); i++) {
            const VoxelPos centre = node_centre(nodes[i], n);
            asked[i] = p_shared ? r_readings.ask(centre, site_lod(side)) : r_readings.push(centre, site_lod(side));
        }
    }

    // The round's readings in: keep the level's nodes the plane crosses, then their children meeting the box.
    void step(const Readings &p_readings) {
        const int64_t n = side << level;
        kept.clear();
        for (size_t i = 0; i < nodes.size(); i++) {
            const ExactNoise::Float4 &f = p_readings.value(asked[i]);
            if (plane_meets(f, n, side)) {
                kept.push_back(WalkCell{ nodes[i], f });
            }
        }
        if (level == stop) {
            std::sort(kept.begin(), kept.end(), [](const WalkCell &p_a, const WalkCell &p_b) { return cell_order(p_a.cell, p_b.cell); });
            done = true;
            return;
        }
        nodes.clear();
        const int64_t m = n / 2;
        for (const WalkCell &k : kept) {
            for (const VoxelPos &c : CORNERS) {
                const VoxelPos child = k.cell * 2 + c;
                if (VoxelBox::cube(child, m).intersects(box)) {
                    nodes.push_back(child);
                }
            }
        }
        level--;
    }
};

// Every walk to its stop level, a round per level for all of them.
void walk_cells(std::vector<CellWalk> &r_walks, const FieldBatchFn &p_field, WalkStats *r_stats) {
    Readings readings;
    for (;;) {
        bool any = false;
        for (CellWalk &w : r_walks) {
            if (!w.done) {
                w.ask(readings, r_walks.size() > 1);
                any = true;
            }
        }
        if (!any) {
            return;
        }
        readings.read(p_field, r_stats);
        for (CellWalk &w : r_walks) {
            if (!w.done) {
                w.step(readings);
            }
        }
    }
}

// A site of a walk's kept cell before its snap.
struct Candidate {
    VoxelPos cell;
    size_t walk = 0;
    Site site;
    bool alive = true;
};

// The candidates p_snapped (indices) moved onto the LOD-0 ground, a round per step for all of them: read at its base,
// SNAP_STEPS Newton steps on whole voxels while it is more than a voxel away, each read again; a candidate stays alive if
// it ends within a voxel of the ground and inside its cell of p_sides[walk] (a surface point has one owner).
void snap_candidates(std::vector<Candidate> &r_candidates, std::vector<uint32_t> p_snapped, const int64_t *p_sides, const FieldBatchFn &p_field,
        WalkStats *r_stats) {
    Readings readings;
    std::vector<ExactNoise::Float4> f(r_candidates.size());
    std::vector<uint32_t> asked(r_candidates.size());
    auto read = [&](const std::vector<uint32_t> &p_which) {
        for (const uint32_t c : p_which) {
            asked[c] = readings.push(r_candidates[c].site.base, 0);
        }
        readings.read(p_field, r_stats);
        for (const uint32_t c : p_which) {
            f[c] = readings.value(asked[c]);
        }
    };
    read(p_snapped);
    // pending: the candidates still more than a voxel off the ground, stepped then read again.
    std::vector<uint32_t> pending;
    for (int step = 0; step < SNAP_STEPS; step++) {
        pending.clear();
        for (const uint32_t c : p_snapped) {
            Candidate &k = r_candidates[c];
            if (k.alive && !(std::fabs(f[c].w) <= 1.0f)) {
                k.alive = newton_step(f[c], k.site.base, p_sides[k.walk], k.site.base);
                if (k.alive) {
                    pending.push_back(c);
                }
            }
        }
        if (pending.empty()) {
            break;
        }
        read(pending);
    }
    for (const uint32_t c : p_snapped) {
        Candidate &k = r_candidates[c];
        k.alive = k.alive && std::fabs(f[c].w) <= 1.0f && VoxelBox::cube(k.cell, p_sides[k.walk]).contains(k.site.base);
    }
}

} // namespace

FieldBatchFn batch_of(const FieldFn &p_field) {
    return [p_field](const VoxelPos *p_points, const int *p_lods, int p_count, ExactNoise::Float4 *r_out) {
        parallel_for(uint32_t(p_count), "field_batch", [&](uint32_t p_i) { r_out[p_i] = p_field(p_points[p_i], p_lods[p_i]); });
    };
}

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

void surface_nodes(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, const FieldBatchFn &p_field, int p_level, int p_min_level,
        std::vector<VoxelPos> &r_out) {
    if (p_side <= 0 || p_level < 0 || p_level > 62) {
        return;
    }
    std::vector<CellWalk> walk{ CellWalk(VoxelBox{ p_lo, p_hi, true }, p_side, p_min_level, p_level) };
    walk_cells(walk, p_field, nullptr);
    for (const WalkCell &c : walk[0].kept) {
        r_out.push_back(c.cell);
    }
}

bool cell_site(const VoxelPos &p_cell, int64_t p_side, uint32_t p_salt, const FieldFn &p_field, Site &r_site) {
    std::vector<Candidate> c(1);
    c[0].cell = p_cell;
    if (!site_candidate(p_cell, p_side, p_salt, p_field(node_centre(p_cell, p_side), site_lod(p_side)), c[0].site)) {
        return false;
    }
    snap_candidates(c, { 0 }, &p_side, batch_of(p_field), nullptr);
    r_site = c[0].site;
    return c[0].alive;
}

void surface_sites_many(const SiteWalk *p_walks, size_t p_count, const FieldBatchFn &p_field, WalkStats *r_stats) {
    std::vector<CellWalk> walks;
    std::vector<int64_t> sides(p_count);
    walks.reserve(p_count);
    for (size_t w = 0; w < p_count; w++) {
        walks.emplace_back(p_walks[w].box, p_walks[w].side, p_walks[w].min_level, 0);
        sides[w] = p_walks[w].side;
    }
    walk_cells(walks, p_field, r_stats);
    std::vector<Candidate> candidates;
    std::vector<uint32_t> snapped;
    for (size_t w = 0; w < p_count; w++) {
        for (const WalkCell &k : walks[w].kept) {
            Candidate c;
            c.cell = k.cell;
            c.walk = w;
            if (site_candidate(k.cell, sides[w], p_walks[w].salt, k.centre, c.site)) {
                if (p_walks[w].snap) {
                    snapped.push_back(uint32_t(candidates.size()));
                }
                candidates.push_back(c);
            }
        }
    }
    snap_candidates(candidates, std::move(snapped), sides.data(), p_field, r_stats);
    for (const Candidate &c : candidates) {
        const SiteWalk &w = p_walks[c.walk];
        if (c.alive && w.box.contains(c.site.base)) {
            w.out->push_back(CellSite{ c.cell, c.site });
        }
    }
}

void surface_sites(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, uint32_t p_salt, const FieldFn &p_field, std::vector<CellSite> &r_out,
        bool p_snap, int p_min_level) {
    const SiteWalk walk{ VoxelBox{ p_lo, p_hi, true }, p_side, p_salt, p_snap, p_min_level, &r_out };
    surface_sites_many(&walk, 1, batch_of(p_field));
}

} // namespace Archetype
