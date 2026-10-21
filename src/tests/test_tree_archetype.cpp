#include "test_common.hpp"

#include "sdf_cascade.hpp"
#include "tree_archetype.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <numeric>

using namespace Archetype;
using ExactNoise::float_bits;

namespace {

// The types trees use, registered as SDFTable::build_table does.
struct TreeLibrary {
    DefLibrary lib;
    TreeTypes types;
    explicit TreeLibrary(double p_voxel) { types = add_tree_types(lib, p_voxel); }
};

std::vector<TableRecord> grow(const TreeLibrary &p_lib, const TreeSpecies &p_species, uint32_t p_seed, double p_voxel,
        const VoxelPos &p_base = VoxelPos(), std::vector<TreeStem> *r_stems = nullptr) {
    std::vector<TableRecord> out;
    grow_tree(p_species, p_base, UP_Y, p_seed, p_voxel, p_lib.types, out, r_stems);
    return out;
}

// Every field grow_tree sets, the anchor shifted by p_offset.
bool same_part(const TableRecord &p_a, const TableRecord &p_b, const VoxelPos &p_offset) {
    return p_a.def == p_b.def && p_a.op == p_b.op && float_bits(p_a.k) == float_bits(p_b.k) && float_bits(p_a.inv_k) == float_bits(p_b.inv_k) &&
            p_a.anchor + p_offset == p_b.anchor && std::memcmp(p_a.rotation, p_b.rotation, sizeof(p_a.rotation)) == 0 &&
            std::memcmp(p_a.params, p_b.params, sizeof(p_a.params)) == 0 && p_a.salt == p_b.salt;
}

struct Axis {
    double a[3];
    double b[3];
    double r;
};

// A branch record's axis (world, voxels) and its larger radius.
Axis branch_axis(const TableRecord &p_rec) {
    Axis ax;
    const double h = p_rec.params[0];
    for (int j = 0; j < 3; j++) {
        ax.a[j] = double(p_rec.anchor[j]);
        ax.b[j] = ax.a[j] + h * double(p_rec.rotation[3 + j]); // R^T (0, h, 0): the rotation's middle row
    }
    ax.r = MAX(double(p_rec.params[1]), double(p_rec.params[1]) + double(p_rec.params[2]) * h);
    return ax;
}

double distance_to_axis(const double *p_p, const Axis &p_ax) {
    double ab[3], ap[3];
    double ab2 = 0.0, t = 0.0;
    for (int j = 0; j < 3; j++) {
        ab[j] = p_ax.b[j] - p_ax.a[j];
        ap[j] = p_p[j] - p_ax.a[j];
        ab2 += ab[j] * ab[j];
        t += ap[j] * ab[j];
    }
    t = ab2 > 0.0 ? CLAMP(t / ab2, 0.0, 1.0) : 0.0;
    double d2 = 0.0;
    for (int j = 0; j < 3; j++) {
        const double e = ap[j] - t * ab[j];
        d2 += e * e;
    }
    return std::sqrt(d2);
}

} // namespace

TEST_CASE("Tree: grow_tree is deterministic, its base only translates it") {
    for (const double voxel : { 0.5, 0.05 }) {
        const TreeLibrary tl(voxel);
        for (size_t sp = 0; sp < tree_species().size(); sp++) {
            for (uint32_t seed = 1; seed <= 8; seed++) {
                const std::vector<TableRecord> a = grow(tl, tree_species()[sp], seed, voxel);
                const std::vector<TableRecord> b = grow(tl, tree_species()[sp], seed, voxel);
                const VoxelPos offset((int64_t(1) << 40) + 3, -7, int64_t(1) << 33);
                const std::vector<TableRecord> c = grow(tl, tree_species()[sp], seed, voxel, offset);
                CHECK(a.size() == b.size());
                CHECK(a.size() == c.size());
                if (a.size() != b.size() || a.size() != c.size()) {
                    continue;
                }
                for (size_t i = 0; i < a.size(); i++) {
                    CHECK(same_part(a[i], b[i], VoxelPos()));
                    CHECK(same_part(a[i], c[i], offset));
                }
                const std::vector<TableRecord> other = grow(tl, tree_species()[sp], seed + 100, voxel);
                CHECK((other.size() != a.size() || !same_part(other[0], a[0], VoxelPos()) || !same_part(other.back(), a.back(), VoxelPos())));
            }
        }
    }
}

TEST_CASE("Tree: every part is a valid, bounded, banded record, within the budget") {
    for (const double voxel : { 0.5, 0.05 }) {
        CAPTURE(voxel);
        const TreeLibrary tl(voxel);
        int64_t branches = 0, crowns = 0, trees = 0, parts = 0;
        int max_lod_seen = 0;
        for (size_t sp = 0; sp < tree_species().size(); sp++) {
            const TreeSpecies &species = tree_species()[sp];
            for (uint32_t seed = 1; seed <= 20; seed++) {
                std::vector<TreeStem> stems;
                std::vector<TableRecord> records = grow(tl, species, seed, voxel, VoxelPos(-1000, 6000, -3000), &stems);
                trees++;
                parts += int64_t(records.size());
                CHECK(int(records.size()) <= species.budget);
                for (const TreeStem &stem : stems) {
                    if (stem.order > 0) {
                        CHECK(stem.r0 >= 0.5); // thinner never meshes, even at LOD 0
                    }
                }
                for (TableRecord &rec : records) {
                    CHECK(SDFTable::edit_error(rec.rotation, rec.params, int(rec.op), rec.k).is_empty());
                    CHECK(rec.op == Combine::SMOOTH_UNION);
                    if (rec.def == tl.types.branch) {
                        branches++;
                        CHECK(rec.params[0] >= 1.0f);
                        CHECK(rec.params[1] > 0.0f);
                        CHECK(rec.params[1] + rec.params[2] * rec.params[0] > 0.0f);
                    } else if (rec.def == tl.types.cone) {
                        crowns++;
                        CHECK(rec.params[0] > 0.0f);
                        CHECK(rec.params[1] > 0.0f);
                        CHECK(std::fabs(double(rec.params[2]) * double(rec.params[2]) + double(rec.params[3]) * double(rec.params[3]) - 1.0) < 1e-6);
                        CHECK(float_bits(rec.params[4]) == float_bits(tl.types.crown_amp));
                    } else {
                        CHECK(rec.def == tl.types.crown);
                        crowns++;
                        CHECK(rec.params[0] > 0.0f);
                        CHECK(float_bits(rec.params[1]) == float_bits(tl.types.crown_amp));
                    }
                    finish_record(tl.lib[rec.def], rec, 3.34f, true);
                    CHECK(rec.max_lod != TableRecord::NEVER_DROPPED); // every part leaves the lists at some LOD
                    max_lod_seen = MAX(max_lod_seen, int(rec.max_lod));
                }
            }
        }
        MESSAGE(voxel, " m voxels: ", trees, " trees, ", parts, " records (", branches, " branch segments, ", crowns, " crown spheres and cones), last listed at LOD ",
                max_lod_seen - 1);
        CHECK(branches > trees);
        CHECK(crowns >= trees);
    }
}

TEST_CASE("Tree: branch segments join on whole voxels and every one hangs on an earlier part or the hull") {
    const double voxel = 0.05;
    const TreeLibrary tl(voxel);
    int64_t joined = 0, in_hull = 0;
    for (size_t sp = 0; sp < tree_species().size(); sp++) {
        for (uint32_t seed = 1; seed <= 12; seed++) {
            const std::vector<TableRecord> records = grow(tl, tree_species()[sp], seed, voxel, VoxelPos(500, -20, 77));
            std::vector<Axis> earlier;
            bool first = true;
            for (const TableRecord &rec : records) {
                if (rec.def != tl.types.branch) {
                    continue;
                }
                const Axis ax = branch_axis(rec);
                for (int j = 0; j < 3; j++) {
                    CHECK(std::fabs(ax.b[j] - std::round(ax.b[j])) < 1e-3); // the next segment's anchor
                }
                if (!first) {
                    // Its base is on a branch grown before it (a fork point or the previous joint, snapped: within
                    // sqrt(3) / 2 voxel of the axis), or inside a hull sphere that hides the branch it forks off.
                    bool attached = false;
                    for (const Axis &e : earlier) {
                        attached = attached || distance_to_axis(ax.a, e) <= e.r + 1.0;
                    }
                    bool hidden = false;
                    for (const TableRecord &c : records) {
                        if (c.def == tl.types.crown) {
                            double d2 = 0.0;
                            for (int j = 0; j < 3; j++) {
                                d2 += (ax.a[j] - double(c.anchor[j])) * (ax.a[j] - double(c.anchor[j]));
                            }
                            hidden = hidden || std::sqrt(d2) <= double(c.params[0]);
                        }
                    }
                    CHECK((attached || hidden));
                    joined += attached ? 1 : 0;
                    in_hull += attached ? 0 : 1;
                }
                first = false;
                earlier.push_back(ax);
            }
        }
    }
    MESSAGE("segments on an earlier branch: ", joined, ", based inside the hull: ", in_hull);
    CHECK(joined > 100);
}

TEST_CASE("Tree: the pipe model holds and stems grow thickest first") {
    for (const double voxel : { 0.05, 0.02 }) {
        const TreeLibrary tl(voxel);
        for (size_t sp = 0; sp < tree_species().size(); sp++) {
            for (uint32_t seed = 1; seed <= 10; seed++) {
                std::vector<TreeStem> stems;
                grow(tl, tree_species()[sp], seed, voxel, VoxelPos(), &stems);
                CHECK(stems.size() > (tree_species()[sp].limbs > 0 ? 1u : 0u)); // a conifer is its trunk
                double previous = INFINITY;
                for (const TreeStem &stem : stems) {
                    double sum = stem.r_end * stem.r_end;
                    for (const double r : stem.children) {
                        sum += r * r;
                        CHECK(r <= stem.r0);
                    }
                    CHECK(sum <= stem.r0 * stem.r0 * (1.0 + 1e-9));
                    CHECK(stem.r0 <= previous);
                    previous = stem.r0;
                }
            }
        }
    }
}

TEST_CASE("Tree: a smaller budget cuts the thinnest stems, the rest is unchanged") {
    const double voxel = 0.05;
    const TreeLibrary tl(voxel);
    for (size_t sp = 0; sp < tree_species().size(); sp++) {
        TreeSpecies small = tree_species()[sp];
        if (small.limbs == 0) {
            continue; // a conifer: its trunk and tiers, nothing to cut
        }
        TreeSpecies large = small;
        small.budget = 30;
        large.budget = 400;
        for (uint32_t seed = 1; seed <= 6; seed++) {
            const std::vector<TableRecord> a = grow(tl, small, seed, voxel);
            const std::vector<TableRecord> b = grow(tl, large, seed, voxel);
            const size_t hull = size_t(small.hull_spheres);
            CHECK(a.size() == size_t(small.budget));
            CHECK(b.size() > a.size());
            if (a.size() != size_t(small.budget) || b.size() <= a.size()) {
                continue;
            }
            // The grown part (thickest stems first) is a prefix; the hull closes both.
            for (size_t i = 0; i + hull < a.size(); i++) {
                CHECK(same_part(a[i], b[i], VoxelPos()));
            }
            for (size_t i = 0; i < hull; i++) {
                CHECK(same_part(a[a.size() - hull + i], b[b.size() - hull + i], VoxelPos()));
            }
        }
    }
}

// The sites of every cell meeting [p_lo, p_hi], cell by cell: what surface_sites' walk must find.
std::vector<TreeSite> brute_sites(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, const FieldFn &p_field) {
    std::vector<TreeSite> out;
    for (int64_t z = WorldCoord::floor_div(p_lo.z, p_side); z <= WorldCoord::floor_div(p_hi.z, p_side); z++) {
        for (int64_t y = WorldCoord::floor_div(p_lo.y, p_side); y <= WorldCoord::floor_div(p_hi.y, p_side); y++) {
            for (int64_t x = WorldCoord::floor_div(p_lo.x, p_side); x <= WorldCoord::floor_div(p_hi.x, p_side); x++) {
                TreeSite site;
                if (!cell_site(VoxelPos(x, y, z), p_side, p_field, site)) {
                    continue;
                }
                bool inside = true;
                for (int a = 0; a < 3; a++) {
                    inside = inside && site.base[a] >= p_lo[a] && site.base[a] <= p_hi[a];
                }
                if (inside) {
                    out.push_back(site);
                }
            }
        }
    }
    return out;
}

bool same_sites(const std::vector<CellSite> &p_a, const std::vector<TreeSite> &p_b) {
    if (p_a.size() != p_b.size()) {
        return false;
    }
    for (size_t i = 0; i < p_a.size(); i++) {
        const TreeSite &s = p_a[i].site;
        if (!(s.base == p_b[i].base) || s.seed != p_b[i].seed || s.species != p_b[i].species) {
            return false;
        }
    }
    return true;
}

// surface_sites' trees without their cells.
std::vector<TreeSite> forest_trees(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, const FieldFn &p_field) {
    std::vector<CellSite> cells;
    surface_sites(p_lo, p_hi, p_side, p_field, cells);
    std::vector<TreeSite> out;
    for (const CellSite &cs : cells) {
        out.push_back(cs.site);
    }
    return out;
}

// The smallest distance between two of p_sites (voxels).
double closest_pair(const std::vector<TreeSite> &p_sites) {
    double best = 1e30;
    for (size_t i = 0; i < p_sites.size(); i++) {
        for (size_t j = i + 1; j < p_sites.size(); j++) {
            const VoxelPos d = p_sites[i].base - p_sites[j].base;
            best = MIN(best, std::sqrt(double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z));
        }
    }
    return best;
}

// A plane through p_origin with unit normal (p_nx, p_ny, p_nz), as a field in LOD0 voxels.
FieldFn plane_field(const VoxelPos &p_origin, double p_nx, double p_ny, double p_nz) {
    return [=](const VoxelPos &p_v, int) {
        const double w = double(p_v.x - p_origin.x) * p_nx + double(p_v.y - p_origin.y) * p_ny + double(p_v.z - p_origin.z) * p_nz;
        return ExactNoise::Float4{ float(p_nx), float(p_ny), float(p_nz), float(w) };
    };
}

TEST_CASE("Tree sites: on flat ground, one tree per column of cells, on the surface") {
    const int64_t side = 100;
    const VoxelPos origin(-3000, 6017, 1200);
    const FieldFn flat = plane_field(origin, 0.0, 1.0, 0.0);
    const VoxelPos lo = origin - VoxelPos(2000, 2000, 2000), hi = origin + VoxelPos(2000, 2000, 2000);
    const std::vector<TreeSite> sites = forest_trees(lo, hi, side, flat);
    const int64_t columns = (4000 / side) * (4000 / side);
    MESSAGE("flat: ", sites.size(), " trees for ", columns, " columns of cells");
    CHECK(int64_t(sites.size()) >= columns * 9 / 10);
    CHECK(int64_t(sites.size()) <= columns + 4 * (4000 / side) + 1);
    std::vector<VoxelPos> cells;
    for (const TreeSite &s : sites) {
        CHECK(std::abs(s.base.y - origin.y) <= 1);
        CHECK(s.species >= 0);
        CHECK(s.species < int(tree_species().size()));
        cells.push_back(VoxelPos(WorldCoord::floor_div(s.base.x, side), 0, WorldCoord::floor_div(s.base.z, side)));
    }
    std::sort(cells.begin(), cells.end(), [](const VoxelPos &p_a, const VoxelPos &p_b) { return p_a.x != p_b.x ? p_a.x < p_b.x : p_a.z < p_b.z; });
    CHECK(std::adjacent_find(cells.begin(), cells.end()) == cells.end()); // one per column
}

TEST_CASE("Tree sites: the octree walk finds every cell's tree (brute force), on hills and a 60 degree slope") {
    const int64_t side = 80;
    const VoxelPos origin(int64_t(1) << 40, -517, -(int64_t(1) << 33));
    const double tau = 6.283185307179586;
    const FieldFn hills = [origin, tau](const VoxelPos &p_v, int) {
        const double amp = 60.0, wave = 1500.0;
        const double x = double(p_v.x - origin.x);
        const double w = double(p_v.y - origin.y) - amp * std::sin(tau * x / wave);
        return ExactNoise::Float4{ float(-amp * tau / wave * std::cos(tau * x / wave)), 1.0f, 0.0f, float(w) };
    };
    const FieldFn slope = plane_field(origin, std::sin(1.0471975511965976), std::cos(1.0471975511965976), 0.0);
    const VoxelPos lo = origin - VoxelPos(1200, 1300, 1200), hi = origin + VoxelPos(1200, 1300, 1200);
    for (const FieldFn *f : { &hills, &slope }) {
        std::vector<CellSite> walked;
        surface_sites(lo, hi, side, *f, walked);
        const std::vector<TreeSite> brute = brute_sites(lo, hi, side, *f);
        MESSAGE("walk ", walked.size(), " trees, brute force ", brute.size());
        CHECK(walked.size() > 100);
        CHECK(same_sites(walked, brute));
        for (const CellSite &s : walked) {
            CHECK(std::fabs((*f)(s.site.base, 0).w) <= 1.0f);
        }
        // The middle-part rule keeps cells side by side along the ground 2 SITE_INNER cells apart; across the face of
        // the normal's axis two sites can meet, rarely (their jitters along the ground are independent).
        int64_t close = 0;
        for (size_t i = 0; i < brute.size(); i++) {
            for (size_t j = 0; j < brute.size(); j++) {
                const VoxelPos d = brute[i].base - brute[j].base;
                if (i != j && std::sqrt(double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z) < 2.0 * SITE_INNER * double(side)) {
                    close++;
                    break;
                }
            }
        }
        MESSAGE("closest pair ", closest_pair(brute) / double(side), " cells; ", close, " of ", brute.size(), " trees closer than ",
                2.0 * SITE_INNER, " cells to another");
        CHECK(close * 100 <= 3 * int64_t(brute.size())); // 2% on the 60 degree slope
    }
}

// The walk keeps a node by its centre's coarse distance; the cascade's field is a first-order height-field estimate,
// and the coarse one leaves the fine octaves out. On the real ground at 5 cm the walk must still find every cell's tree
// (SITE_SLACK), at a few evaluations per tree.
TEST_CASE("Tree sites: on the cascade ground, the octree walk finds every cell's tree (brute force)") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.configure(param, 18);
    const int64_t side = 100;
    std::atomic<int64_t> coarse{ 0 }, fine{ 0 };
    const FieldFn ground = [&](const VoxelPos &p_v, int p_lod) {
        (p_lod > 0 ? coarse : fine)++;
        return sdf.ground_field(p_v, p_lod);
    };
    int64_t total = 0;
    for (const int64_t x : { int64_t(-1000), int64_t(250000), int64_t(-3700000) }) {
        const int64_t z = x / 3 + 40000;
        const VoxelPos centre(x, sdf.ground_y(x, z), z);
        const VoxelPos reach(1000, 4000, 1000);
        coarse = 0;
        fine = 0;
        std::vector<CellSite> walked;
        surface_sites(centre - reach, centre + reach, side, ground, walked);
        const int64_t walk_coarse = coarse, walk_fine = fine;
        const std::vector<TreeSite> brute = brute_sites(centre - reach, centre + reach, side, ground);
        MESSAGE("cascade at x ", x, ": walk ", walked.size(), " trees, brute force ", brute.size(), "; per tree ",
                double(walk_coarse) / double(MAX<size_t>(walked.size(), 1)), " coarse + ", double(walk_fine) / double(MAX<size_t>(walked.size(), 1)),
                " fine evaluations");
        CHECK(walked.size() > 100);
        CHECK(same_sites(walked, brute));
        CHECK(walk_coarse + walk_fine <= 12 * int64_t(walked.size())); // ~70 before the plane test
        total += int64_t(walked.size());
        // Unsnapped (the far proxies): no LOD-0 evaluation, every snapped tree's cell kept, the bases on the coarse plane.
        coarse = 0;
        fine = 0;
        std::vector<CellSite> coarse_sites;
        surface_sites(centre - reach, centre + reach, side, ground, coarse_sites, false);
        CHECK(fine == 0);
        size_t j = 0, matched = 0;
        double off = 0.0;
        for (const CellSite &s : walked) {
            while (j < coarse_sites.size() && cell_order(coarse_sites[j].cell, s.cell)) { // both in cell order
                j++;
            }
            if (j < coarse_sites.size() && coarse_sites[j].cell == s.cell) {
                matched++;
                off = MAX(off, std::fabs(double(sdf.ground_field(coarse_sites[j].site.base, 0).w)));
            }
        }
        MESSAGE("unsnapped: ", coarse_sites.size(), " sites, ", matched, " of the snapped trees' cells; bases within ", off, " voxels of the ground");
        CHECK(matched == walked.size());
        CHECK(coarse_sites.size() * 100 <= walked.size() * 105);
    }
    CHECK(total > 300);
    // An empty tile (a node of 8 cells far above the ground): one evaluation.
    const int64_t tile = 8 * side;
    const VoxelPos air(0, WorldCoord::floor_div(sdf.ground_y(0, 0), tile) + 20, 0);
    coarse = 0;
    fine = 0;
    std::vector<CellSite> none;
    surface_sites(air * tile, air * tile + VoxelPos(tile - 1, tile - 1, tile - 1), side, ground, none);
    CHECK(none.empty());
    CHECK(coarse + fine == 1);
}

TEST_CASE("Tree sites: a cell holds the same tree whatever the query box") {
    const int64_t side = 100;
    const VoxelPos origin(500, 300, -700);
    const FieldFn slope = plane_field(origin, 0.6, 0.8, 0.0);
    const std::vector<TreeSite> a = forest_trees(origin - VoxelPos(1500, 1500, 1500), origin + VoxelPos(1500, 1500, 1500), side, slope);
    const std::vector<TreeSite> b = forest_trees(origin - VoxelPos(300, 900, 2500), origin + VoxelPos(2600, 1100, 400), side, slope);
    int64_t shared = 0;
    for (const TreeSite &s : a) {
        for (const TreeSite &t : b) {
            if (s.base == t.base) {
                CHECK(s.seed == t.seed);
                CHECK(s.species == t.species);
                shared++;
            }
        }
    }
    MESSAGE("query boxes: ", a.size(), " and ", b.size(), " trees, ", shared, " in both");
    CHECK(shared > 50);
}

uint64_t fnv(uint64_t p_h, const void *p_data, size_t p_size) {
    const unsigned char *b = static_cast<const unsigned char *>(p_data);
    for (size_t i = 0; i < p_size; i++) {
        p_h = (p_h ^ b[i]) * 0x100000001B3ull;
    }
    return p_h;
}

// Every field grow_tree sets along +y, every species at two voxel sizes, several seeds and bases. Pinned when trees took
// an up axis (feature/tree-frame); that change kept dev's records: the tests growing trees at fixed bases printed the
// same numbers before and after it.
TEST_CASE("Tree: grown along +y, the records are pinned") {
    uint64_t h = 0xCBF29CE484222325ull;
    int64_t records = 0;
    for (const double voxel : { 0.05, 0.5 }) {
        const TreeLibrary tl(voxel);
        for (int sp = 0; sp < int(tree_species().size()); sp++) {
            for (uint32_t seed = 0; seed < 6; seed++) {
                const VoxelPos base(int64_t(seed) * 977 - 3000, int64_t(seed) * -31 + 17, int64_t(1) << (20 + seed));
                for (const TableRecord &r : grow(tl, tree_species()[sp], 0x5EEDu * (seed + 1) + uint32_t(sp), voxel, base)) {
                    h = fnv(h, &r.def, sizeof(r.def));
                    h = fnv(h, &r.op, sizeof(r.op));
                    h = fnv(h, &r.k, sizeof(r.k));
                    h = fnv(h, &r.inv_k, sizeof(r.inv_k));
                    const int64_t a[3] = { r.anchor.x, r.anchor.y, r.anchor.z };
                    h = fnv(h, a, sizeof(a));
                    h = fnv(h, r.rotation, sizeof(r.rotation));
                    h = fnv(h, r.params, sizeof(r.params));
                    h = fnv(h, &r.salt, sizeof(r.salt));
                    records++;
                }
            }
        }
    }
    MESSAGE("tree pin: ", records, " records, hash ", h);
    CHECK(records == 896);
    CHECK(h == 7225423438707016970ull);
}

TEST_CASE("Tree: grown along a turned up, the tree lies along it") {
    const double voxel = 0.05;
    const TreeLibrary tl(voxel);
    const VoxelPos base(123, -456, 789);
    int cones = 0;
    for (int sp = 0; sp < int(tree_species().size()); sp++) {
        std::vector<TableRecord> turned;
        grow_tree(tree_species()[sp], base, UpAxis{ 1.0, 0.0, 0.0 }, 0xABCu + uint32_t(sp), voxel, tl.types, turned);
        int64_t reach_x = 0, reach_y = 0;
        for (const TableRecord &r : turned) {
            reach_x = MAX(reach_x, r.anchor.x - base.x);
            reach_y = MAX(reach_y, std::abs(r.anchor.y - base.y));
            if (int(r.def) == tl.types.cone) {
                cones++;
                CHECK(r.rotation[3] == 1.0f); // the cone's axis (the middle row) is up
                CHECK(r.rotation[4] == 0.0f);
                CHECK(r.rotation[5] == 0.0f);
            }
        }
        CHECK(reach_x > 2 * reach_y); // the tree grows along +x
    }
    CHECK(cones > 0);
}

TEST_CASE("SDFCascade forest: the streamed trees fold into evaluate(), trunks are solid") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.set_forest(30, 40.0);
    sdf.set_moment_blobs(false); // the trees' records alone
    param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 18));
    sdf.configure(param, 18);
    const auto snap = sdf.get_snapshot();
    const ForestField *field = sdf.get_forest();
    CHECK(field != nullptr);
    if (!field) {
        return;
    }
    CHECK(snap->records.size() == 0); // nothing placed at configure: the forest is streamed
    for (int i = 0; i < sdf.get_library().size(); i++) {
        CHECK(sdf.get_library()[i].program.validate(18).is_empty());
    }
    // Every tree of the zone: a group of a LOD-0 tree tile, its first record its first trunk segment.
    const VoxelBox zone = field->setup().generator.zone;
    std::vector<FieldKey> tiles;
    field->keys(zone.lo, zone.hi, 0, tiles);
    // Inside each tree's first trunk segment, a quarter of the way up its axis (the LOD0 voxel: the cascade's lattice):
    // solid at LOD 0, and evaluate() is the fold of every key's records there, in key order (what the meshers list).
    int64_t trees = 0, solid = 0;
    std::vector<FieldKey> keys;
    std::vector<uint32_t> hit, ids;
    std::vector<TableRecord> records;
    for (const FieldKey &tile : tiles) {
        FieldContent c;
        field->content(tile, c);
        for (const FieldContent::Group &g : c.groups) {
            const TableRecord &rec = c.records[g.first];
            CHECK(rec.def == 3); // the branch type, after sphere, capsule and rock
            const double h = rec.params[0];
            WorldPos p;
            for (int j = 0; j < 3; j++) {
                p.voxel[j] = int64_t(std::floor(double(rec.anchor[j]) + 0.25 * h * double(rec.rotation[3 + j])));
            }
            records.clear();
            field->keys(p.voxel, p.voxel, 0, keys);
            for (const FieldKey &k : keys) {
                FieldContent kc;
                field->content(k, kc);
                hit.clear();
                kc.meeting(p.voxel, p.voxel, 0, hit);
                for (const uint32_t i : hit) {
                    records.push_back(kc.records[i]);
                }
            }
            ids.resize(records.size());
            std::iota(ids.begin(), ids.end(), 0u);
            const float value = sdf.evaluate(p, 0);
            float ch[Cascade::CHANNELS];
            sdf.channels_at(p.voxel, 0, ch);
            const Float4 r = sdf.sample_sdf(*snap, RecordView(records.data()), ids.data(), uint32_t(ids.size()), ch, p, 0);
            CHECK(float_bits(value) == float_bits(r.w));
            CHECK(std::isfinite(r.x));
            CHECK(std::isfinite(r.y));
            CHECK(std::isfinite(r.z));
            solid += value < 0.0f ? 1 : 0;
            trees++;
        }
    }
    MESSAGE("forest: ", trees, " trees in ", tiles.size(), " LOD-0 tree tiles, ", solid, " trunks solid at LOD 0");
    CHECK(trees > 20);
    CHECK(solid == trees);
}

// What the band rule leaves of a forest at each LOD, weighted by volume (record_moments): the share of the trees'
// volume drawn whole (no erosion) and drawn at all (listed, eroded by less than its half thickness), under the default
// rule (whole while resolved, eroding just before the drop) and under the rule it replaced (one cell per LOD from LOD 1,
// dropped when that reaches the depth). The new rule keeps at least as much at every LOD.
TEST_CASE("Tree band: the resolution rule keeps more of the trees at every LOD than eroding from LOD 1") {
    const double voxel = 0.05;
    const TreeLibrary tl(voxel);
    constexpr int LODS = 9;
    double total = 0.0, whole_old[LODS] = {}, drawn_old[LODS] = {}, whole_new[LODS] = {}, drawn_new[LODS] = {};
    for (size_t sp = 0; sp < tree_species().size(); sp++) {
        for (uint32_t seed = 1; seed <= 10; seed++) {
            std::vector<TableRecord> records = grow(tl, tree_species()[sp], seed, voxel);
            for (TableRecord &rec : records) {
                const TableDef &def = tl.lib[rec.def];
                finish_record(def, rec, 1.0f, true);
                const Moments m = record_moments(def, rec);
                const double half = m.half_thickness(), depth = band_depth(def, rec);
                total += m.m0;
                for (int lod = 0; lod < LODS; lod++) {
                    const double old_erosion = double(ERODE_CELLS) * (std::ldexp(1.0, lod) - 1.0);
                    const bool old_listed = double(ERODE_CELLS) * (std::ldexp(1.0, lod) - 1.0) < depth;
                    whole_old[lod] += old_listed && old_erosion == 0.0 ? m.m0 : 0.0;
                    drawn_old[lod] += old_listed && old_erosion < half ? m.m0 : 0.0;
                    const double erosion = band_erosion(rec.erode, rec.erode_from, lod);
                    whole_new[lod] += rec.listed_at(lod) && erosion == 0.0 ? m.m0 : 0.0;
                    drawn_new[lod] += rec.listed_at(lod) && erosion < half ? m.m0 : 0.0;
                }
            }
        }
    }
    for (int lod = 0; lod < LODS; lod++) {
        MESSAGE("LOD ", lod, " (", voxel * std::ldexp(1.0, lod), " m cells): whole ", 100.0 * whole_old[lod] / total, "% -> ",
                100.0 * whole_new[lod] / total, "%, drawn ", 100.0 * drawn_old[lod] / total, "% -> ", 100.0 * drawn_new[lod] / total, "%");
        CHECK(whole_new[lod] >= whole_old[lod]);
        CHECK(drawn_new[lod] >= drawn_old[lod]);
    }
    CHECK(whole_new[3] > whole_old[3]);
}
