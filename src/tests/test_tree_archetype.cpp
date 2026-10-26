#include "test_common.hpp"

#include "test_stream_common.hpp"

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
    explicit TreeLibrary(double p_voxel) { types = add_tree_types(lib, p_voxel); }
};

std::vector<TableRecord> grow(const TreeLibrary &p_lib, const TreeSpecies &p_species, uint32_t p_seed, double p_voxel,
        const VoxelPos &p_base = VoxelPos(), std::vector<TreeStem> *r_stems = nullptr) {
    std::vector<TableRecord> out;
    grow_tree(p_species, p_base, UP_Y, p_seed, p_voxel, p_lib.types, out, r_stems);
    return out;
}

// A unit up turned from +y by a hashed angle (up to ~1.2 rad) about a hashed azimuth.
UpAxis turned_up(uint32_t p_a, uint32_t p_b) {
    const double angle = 1.2 * ExactNoise::unit_of(p_a), azimuth = Math_TAU * ExactNoise::unit_of(p_b);
    return UpAxis{ std::sin(angle) * std::cos(azimuth), std::cos(angle), std::sin(angle) * std::sin(azimuth) };
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
                    CHECK(SDFTable::edit_error(rec.rotation, rec.params, int(rec.op), rec.k).empty());
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

TEST_CASE("SDFCascade forest: the trees are table records, trunks are solid, evaluate() stays the fold") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.set_forest(30, 40.0);
    sdf.set_stream_sources(false); // the trees' spans
    sdf.set_moment_blobs(false); // the trees' records alone
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
        CHECK(sdf.get_library()[i].program.validate(18).empty());
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

// The last max_lod of p_make(x) finished, x from p_lo while <= p_hi by steps of 10%, checking it never decreases.
template <typename F>
int check_monotone(const TreeLibrary &p_tl, const RecordFinish &p_finish, double p_lo, double p_hi, F p_make) {
    int last = 0;
    for (double x = p_lo; x <= p_hi; x *= 1.1) {
        TableRecord rec = p_make(x);
        finish_table_record(p_tl.lib, p_finish, rec);
        CHECK(int(rec.max_lod) >= last);
        last = int(rec.max_lod);
    }
    return last;
}

// Every record of a tree is a branch no thicker than the trunk's base or a crown no larger than the largest clump or hull
// sphere: tree_bounds names them, and max_lod grows with a part's size, so the bounds' max_lod bounds every part's.
TEST_CASE("Tree band: max_lod is non-decreasing in a branch's and a crown's size, and the species' bounds cover every part") {
    for (const double voxel : { 0.5, 0.05 }) {
        CAPTURE(voxel);
        const TreeLibrary tl(voxel);
        const RecordFinish finish{ 3.34f, true, ErodeRule() };
        // Radii over three decades, and lengths over two at a fixed radius.
        auto branch = [&](double p_h, double p_r, double p_k) {
            TableRecord rec;
            rec.def = uint8_t(tl.types.branch);
            rec.params[0] = float(p_h);
            rec.params[1] = float(p_r);
            rec.set_k(float(p_k));
            return rec;
        };
        CHECK(check_monotone(tl, finish, 0.5, 500.0, [&](double p_r) { return branch(20.0 * p_r, p_r, 2.0 * p_r); }) > 3);
        CHECK(check_monotone(tl, finish, 0.5, 500.0, [&](double p_r) {
            TableRecord rec;
            rec.def = uint8_t(tl.types.crown);
            rec.params[0] = float(p_r);
            rec.params[1] = tl.types.crown_amp;
            rec.set_k(float(0.5 * p_r));
            return rec;
        }) > 3);
        check_monotone(tl, finish, 4.0, 400.0, [&](double p_h) { return branch(p_h, 4.0, 8.0); });
        // The bounds: every part of 200 trees per species, any up, has a max_lod at most its bound's.
        for (size_t sp = 0; sp < tree_species().size(); sp++) {
            CAPTURE(sp);
            const TreeSpecies &species = tree_species()[sp];
            TreeBounds b = tree_bounds(species, voxel, tl.types);
            CHECK(b.has_clump == (species.limbs > 0));
            finish_table_record(tl.lib, finish, b.trunk);
            finish_table_record(tl.lib, finish, b.clump);
            finish_table_record(tl.lib, finish, b.hull);
            const int branch_lod = int(b.trunk.max_lod);
            const int crown_lod = MAX(int(b.hull.max_lod), b.has_clump ? int(b.clump.max_lod) : 0);
            int over = 0;
            for (uint32_t seed = 1; seed <= 200; seed++) {
                const ExactNoise::UVec3 h = ExactNoise::hash3(seed, 0x0Du, 0x11Fu);
                const UpAxis up = ExactNoise::unit_of(h.x) < 0.5 ? UP_Y : turned_up(h.y, h.z);
                std::vector<TableRecord> records;
                grow_tree(species, VoxelPos(), up, seed, voxel, tl.types, records);
                for (TableRecord &rec : records) {
                    finish_table_record(tl.lib, finish, rec);
                    over += int(rec.max_lod) > (rec.def == tl.types.branch ? branch_lod : crown_lod);
                }
            }
            CHECK(over == 0);
        }
    }
}

// From a species' hull_only_lod on, the hull-only growth is exactly the full growth's listed records, bit for bit; at 5 cm,
// below it some tree still lists a branch or a clump (the bound isn't loose by a whole LOD).
TEST_CASE("Tree: hull-only growth is the full growth's records listed from hull_only_lod on") {
    for (const double voxel : { 0.5, 0.05 }) {
        CAPTURE(voxel);
        const TreeLibrary tl(voxel);
        const RecordFinish finish{ 3.34f, true, ErodeRule() };
        constexpr int LODS = 15;
        for (size_t sp = 0; sp < tree_species().size(); sp++) {
            CAPTURE(sp);
            const TreeSpecies &species = tree_species()[sp];
            const int hull_lod = tree_hull_only_lod(species, voxel, tl.lib, finish, tl.types);
            CHECK(hull_lod > 0);
            CHECK(hull_lod < LODS);
            int64_t trees = 0, same = 0, non_hull_below = 0, hull_records = 0, listed_at_hull_lod = 0;
            for (uint32_t seed = 1; seed <= 334; seed++) {
                const ExactNoise::UVec3 h = ExactNoise::hash3(seed, 0x4011u, uint32_t(sp));
                const UpAxis up = seed % 3 == 0 ? UP_Y : turned_up(h.x, h.y);
                std::vector<TableRecord> full, hull;
                std::vector<TreeStem> stems;
                grow_tree(species, VoxelPos(7, -3, 11), up, seed, voxel, tl.types, full);
                grow_tree(species, VoxelPos(7, -3, 11), up, seed, voxel, tl.types, hull, &stems, true);
                CHECK(stems.size() == 1); // the trunk only
                for (TableRecord &rec : full) {
                    finish_table_record(tl.lib, finish, rec);
                }
                for (TableRecord &rec : hull) {
                    finish_table_record(tl.lib, finish, rec);
                }
                hull_records += int64_t(hull.size());
                // Every hull record is a full record, in order (the hull is emitted after the loop in both modes).
                size_t at = 0;
                for (const TableRecord &rec : hull) {
                    while (at < full.size() && !same_part(full[at], rec, VoxelPos())) {
                        at++;
                    }
                    CHECK(at < full.size());
                    at++;
                }
                trees++;
                const auto listed = [](const std::vector<TableRecord> &p_records, int p_lod) {
                    std::vector<const TableRecord *> out;
                    for (const TableRecord &rec : p_records) {
                        if (rec.listed_at(p_lod)) {
                            out.push_back(&rec);
                        }
                    }
                    return out;
                };
                bool identical = true;
                for (int lod = hull_lod; lod < LODS; lod++) {
                    const std::vector<const TableRecord *> a = listed(full, lod), b = listed(hull, lod);
                    identical = identical && a.size() == b.size();
                    for (size_t i = 0; identical && i < a.size(); i++) {
                        identical = StreamTest::same_bits(*a[i], *b[i]);
                    }
                    listed_at_hull_lod += lod == hull_lod ? int64_t(a.size()) : 0;
                }
                same += identical;
                for (const TableRecord &rec : full) {
                    non_hull_below += rec.listed_at(hull_lod - 1) && std::none_of(hull.begin(), hull.end(), [&](const TableRecord &p_h) { return same_part(p_h, rec, VoxelPos()); });
                }
            }
            MESSAGE(voxel, " m, species ", sp, ": hull-only from LOD ", hull_lod, ", ", trees, " trees, ", same, " identical from there on, ", hull_records, " hull records, ",
                    listed_at_hull_lod, " listed at that LOD, ", non_hull_below, " non-hull records listed one LOD below");
            CHECK(same == trees);
            // Tight at the forest's 5 cm (trees are meant for it); at 0.5 m the padded bound of the broadleaf and the slender
            // tree is a LOD late (they keep no branch below it either: one LOD of full growth more, never a record less).
            if (voxel < 0.1) {
                CHECK(non_hull_below > 0);
            }
        }
    }
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
