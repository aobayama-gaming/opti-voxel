#include "test_common.hpp"

#include "sdf_cascade.hpp"
#include "tree_archetype.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace Archetype;
using ExactNoise::float_bits;

namespace {

// The types trees use, registered as SDFTable::build_table does.
struct TreeLibrary {
    DefLibrary lib;
    TreeTypes types;
    explicit TreeLibrary(double p_voxel) {
        const int branch = lib.add(make_branch_type());
        const int crown = lib.add(make_rock_type(crown_k(p_voxel), CROWN_SEED));
        const int cone = lib.add(make_cone_type(crown_k(p_voxel), CROWN_SEED));
        types = crown_types(branch, crown, lib[crown], cone);
    }
};

std::vector<TableRecord> grow(const TreeLibrary &p_lib, const TreeSpecies &p_species, uint32_t p_seed, double p_voxel,
        const VoxelPos &p_base = VoxelPos(), std::vector<TreeStem> *r_stems = nullptr) {
    std::vector<TableRecord> out;
    grow_tree(p_species, p_base, p_seed, p_voxel, p_lib.types, out, r_stems);
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

TEST_CASE("Tree: forest sites fill the square, apart, the same tree per absolute cell") {
    const double voxel = 0.05;
    auto ground = [](int64_t p_x, int64_t p_z) { return (p_x * 3 + p_z) % 17; };
    std::vector<TreeSite> a, b, again;
    forest_sites(400, 100.0, -50.0, -150.0, voxel, ground, a);
    forest_sites(400, 100.0, -50.0, -150.0, voxel, ground, again);
    forest_sites(400, 100.0, 0.0, -150.0, voxel, ground, b); // half a square east
    MESSAGE("forest: ", a.size(), " sites for 400 asked");
    CHECK(a.size() >= 320);
    CHECK(a.size() <= 480);
    CHECK(a.size() == again.size());
    if (a.size() != again.size()) {
        return;
    }
    const int64_t cell = std::llround(100.0 / 20.0 / voxel);
    int64_t shared = 0;
    for (size_t i = 0; i < a.size(); i++) {
        CHECK(a[i].base == again[i].base);
        CHECK(a[i].seed == again[i].seed);
        CHECK(a[i].base.x >= std::llround(-100.0 / voxel));
        CHECK(a[i].base.x <= std::llround(0.0 / voxel));
        CHECK(a[i].base.z >= std::llround(-200.0 / voxel));
        CHECK(a[i].base.z <= std::llround(-100.0 / voxel));
        CHECK(a[i].base.y == ground(a[i].base.x, a[i].base.z));
        CHECK(a[i].species >= 0);
        CHECK(a[i].species < int(tree_species().size()));
        for (size_t j = i + 1; j < a.size(); j++) {
            const double dx = double(a[i].base.x - a[j].base.x), dz = double(a[i].base.z - a[j].base.z);
            CHECK(std::sqrt(dx * dx + dz * dz) >= 0.3 * double(cell) - 2.0);
        }
        for (const TreeSite &s : b) {
            if (s.base == a[i].base) {
                CHECK(s.seed == a[i].seed);
                CHECK(s.species == a[i].species);
                shared++;
            }
        }
    }
    CHECK(shared > int64_t(a.size()) / 3);
}

TEST_CASE("SDFCascade forest: the trees are table records, trunks are solid, evaluate() stays the fold") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.set_forest(30, 40.0);
    param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 18));
    sdf.configure(param, 18);
    const std::vector<TreeSpan> &trees = sdf.get_trees();
    const auto snap = sdf.get_snapshot();
    CHECK(trees.size() > 20);
    uint32_t next = 0;
    for (const TreeSpan &t : trees) {
        CHECK(t.first == next);
        CHECK(t.count > 0);
        next += t.count;
    }
    CHECK(snap->records.size() == next);
    for (int i = 0; i < sdf.get_library().size(); i++) {
        CHECK(sdf.get_library()[i].program.validate(18).is_empty());
    }
    // Inside each tree's first trunk segment, a quarter of the way up its axis (the LOD0 voxel: the cascade's lattice):
    // solid at LOD 0, and the point's candidates hold the segment.
    std::vector<uint32_t> list;
    int64_t solid = 0;
    for (const TreeSpan &t : trees) {
        const TableRecord &rec = snap->records[t.first];
        CHECK(rec.def == 3); // the branch type, after sphere, capsule and rock
        const double h = rec.params[0];
        WorldPos p;
        for (int j = 0; j < 3; j++) {
            p.voxel[j] = int64_t(std::floor(double(rec.anchor[j]) + 0.25 * h * double(rec.rotation[3 + j])));
        }
        snap->point_candidates(p.voxel, 0, list);
        CHECK(std::find(list.begin(), list.end(), t.first) != list.end());
        const float value = sdf.evaluate(p, 0);
        float ch[Cascade::CHANNELS];
        sdf.channels_at(p.voxel, 0, ch);
        const Float4 r = sdf.sample_sdf(*snap, list.data(), uint32_t(list.size()), ch, p, 0);
        CHECK(float_bits(value) == float_bits(r.w));
        CHECK(std::isfinite(r.x));
        CHECK(std::isfinite(r.y));
        CHECK(std::isfinite(r.z));
        solid += value < 0.0f ? 1 : 0;
    }
    MESSAGE("forest: ", trees.size(), " trees, ", snap->records.size(), " records, ", solid, " trunks solid at LOD 0");
    CHECK(solid == int64_t(trees.size()));
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
