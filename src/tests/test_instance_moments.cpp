#include "test_common.hpp"

#include "archetype_table.hpp"
#include "instance_moments.hpp"
#include "tree_archetype.hpp"

#include <string>

using namespace Archetype;

namespace {

// Stratified sampling of the record's world box (finish_record sets it), one hashed point per cell of a p_grid^3
// grid (sharp faces don't alias with it), the CPU program at LOD 0 (solid where !(w > 0)), summed by hand: an
// integration independent of record_moments (world axes, not the local box; no Moments arithmetic).
Moments grid_moments(const TableDef &p_def, const TableRecord &p_rec, int p_grid) {
    const DefRecord dr = p_rec.def_record();
    std::vector<Float4> regs;
    double size[3];
    for (int a = 0; a < 3; a++) {
        size[a] = double(p_rec.box_hi[a] - p_rec.box_lo[a]);
    }
    const double cell = size[0] * size[1] * size[2] / (double(p_grid) * p_grid * p_grid);
    double v = 0.0, x[3] = {}, xx[3][3] = {};
    for (int i = 0; i < p_grid * p_grid * p_grid; i++) {
        const int g[3] = { i % p_grid, i / p_grid % p_grid, i / (p_grid * p_grid) };
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), 0x5A3Du, 0x9E1u });
        const double jitter[3] = { ExactNoise::unit_of(h.x), ExactNoise::unit_of(h.y), ExactNoise::unit_of(h.z) };
        double w[3];
        for (int a = 0; a < 3; a++) {
            w[a] = double(p_rec.box_lo[a]) + (g[a] + jitter[a]) * size[a] / p_grid;
        }
        run_registers(p_def.program, WorldCoord::offset_pos(p_rec.anchor, w), 0, regs, &dr);
        if (regs[p_def.program.output].w > 0.0f) {
            continue;
        }
        v += cell;
        for (int a = 0; a < 3; a++) {
            x[a] += cell * w[a];
            for (int b = 0; b < 3; b++) {
                xx[a][b] += cell * w[a] * w[b];
            }
        }
    }
    Moments m;
    m.m0 = v;
    for (int a = 0; a < 3; a++) {
        m.c[a] = x[a] / v;
    }
    const int pairs[6][2] = { { 0, 0 }, { 1, 1 }, { 2, 2 }, { 0, 1 }, { 0, 2 }, { 1, 2 } };
    for (int i = 0; i < 6; i++) {
        const int a = pairs[i][0], b = pairs[i][1];
        m.s[i] = xx[a][b] - v * m.c[a] * m.c[b];
    }
    return m;
}

// The largest difference between two moments: volume (relative), centroid (relative to p_extent) and covariance
// (relative to the larger eigenvalue of a's).
struct Diff {
    double m0 = 0.0;
    double c = 0.0;
    double cov = 0.0;
};

Diff diff(const Moments &p_a, const Moments &p_b, double p_extent) {
    Diff d;
    d.m0 = std::abs(p_a.m0 - p_b.m0) / p_a.m0;
    for (int a = 0; a < 3; a++) {
        d.c = MAX(d.c, std::abs(p_a.c[a] - p_b.c[a]) / p_extent);
    }
    double lambda[3];
    p_a.principal(lambda);
    for (int i = 0; i < 6; i++) {
        d.cov = MAX(d.cov, std::abs(p_a.s[i] / p_a.m0 - p_b.s[i] / p_b.m0) / lambda[2]);
    }
    return d;
}

struct Case {
    std::string name;
    TableDefPtr def;
    std::array<float, 8> params;
    double tolerance; // on each Diff component
};

std::vector<Case> analytic_cases() {
    // Cone and island: height 20 / 18 over radius 12 / 14 (cos = h / slant, sin = r / slant); the island's dome: drop
    // 8, radius sqrt(14^2 + 8^2). Their rims are smoothed over half a bump wavelength (2^(k-1) = 2 voxels at k = 2),
    // which the formulas leave out: ~4% on the spread.
    const double cs = 20.0 / std::sqrt(400.0 + 144.0), sn = 12.0 / std::sqrt(400.0 + 144.0);
    const double ics = 18.0 / std::sqrt(324.0 + 196.0), isn = 14.0 / std::sqrt(324.0 + 196.0);
    return {
        { "sphere", make_sphere_type(), { 12.0f }, 0.02 },
        { "box", make_box_type(), { 10.0f, 6.0f, 14.0f }, 0.02 },
        { "capsule", make_capsule_type(), { 30.0f, 8.0f }, 0.02 },
        { "branch narrowing", make_branch_type(), { 40.0f, 8.0f, -0.1f }, 0.02 },
        { "branch widening", make_branch_type(), { 40.0f, 4.0f, 0.1f }, 0.02 },
        { "branch to a tip", make_branch_type(), { 40.0f, 6.0f, -0.25f }, 0.03 }, // radius 0 at y = 24, nothing above
        { "cylinder", make_cylinder_type(), { 12.0f, 7.0f }, 0.02 },
        { "rock, no bumps", make_rock_type(3, 0xB0u), { 14.0f, 0.0f }, 0.02 },
        { "rock, bumps", make_rock_type(3, 0xB0u), { 14.0f, 1.0f }, 0.06 },
        { "cone", make_cone_type(2, 0xC0u), { 12.0f, 20.0f, float(cs), float(sn), 0.0f }, 0.05 },
        { "basin", make_basin_type(2, 0xBAu), { 6.0f, 14.0f, 0.0f }, 0.02 },
        { "island", make_island_type(2, 0x15u), { 14.0f, 18.0f, float(ics), float(isn), float(std::sqrt(196.0 + 64.0)), 8.0f, 0.0f }, 0.05 },
        { "stairs", make_stairs_type(), { 6.0f, 2.0f, 3.0f }, 0.02 },
    };
}

} // namespace

TEST_CASE("Moments: each library type's moments match the integral of its program, turned or not") {
    std::vector<std::array<float, 9>> rots;
    rotations(rots);
    for (const Case &c : analytic_cases()) {
        CHECK(bool(c.def->moments));
        for (const size_t r : { size_t(0), rots.size() - 1 }) {
            TableRecord rec;
            rec.anchor = VoxelPos(-(int64_t(1) << 40), 300, 12345);
            std::copy(rots[r].begin(), rots[r].end(), rec.rotation);
            std::copy(c.params.begin(), c.params.end(), rec.params);
            finish_record(*c.def, rec, 1.0f, false);
            const Moments want = grid_moments(*c.def, rec, 36);
            const Moments got = record_moments(*c.def, rec);
            double extent = 0.0;
            for (int a = 0; a < 3; a++) {
                extent = MAX(extent, double(rec.box_hi[a] - rec.box_lo[a]));
            }
            const Diff d = diff(want, got, extent);
            CAPTURE(c.name);
            CAPTURE(r);
            CAPTURE(d.m0);
            CAPTURE(d.c);
            CAPTURE(d.cov);
            CHECK(d.m0 < c.tolerance);
            CHECK(d.c < c.tolerance);
            CHECK(d.cov < c.tolerance);
        }
    }
}

TEST_CASE("Moments: a type without a formula is integrated on its local box (gallery shapes)") {
    std::vector<std::array<float, 9>> rots;
    rotations(rots);
    const std::pair<std::string, TableDefPtr> types[] = { { "die", make_die_type() }, { "holed_cube", make_swiss_cube_type(1) },
        { "tunnel_cube", make_tunnel_cube_type() } };
    const std::array<float, 8> params[] = { { 10.0f }, { 10.0f, 10.0f, 10.0f, 0.0f, 12.0f }, { 10.0f, 3.0f } };
    for (int t = 0; t < 3; t++) {
        const TableDef &def = *types[t].second;
        CHECK(!def.moments);
        TableRecord rec;
        rec.anchor = VoxelPos(77, -5, int64_t(1) << 50);
        std::copy(rots.back().begin(), rots.back().end(), rec.rotation);
        std::copy(params[t].begin(), params[t].end(), rec.params);
        finish_record(def, rec, 1.0f, false);
        const Moments want = grid_moments(def, rec, 40);
        const Moments got = record_moments(def, rec);
        const Diff d = diff(want, got, 20.0);
        CAPTURE(types[t].first);
        CHECK(d.m0 < 0.08); // the fallback's grid is coarse (24^3): a few percent on these sharp shapes
        CHECK(d.c < 0.03);
        CHECK(d.cov < 0.08);
        // The same bits from any anchor: the points are the anchor plus the same local offsets.
        TableRecord far = rec;
        far.anchor = VoxelPos(int64_t(1) << 60, -(int64_t(1) << 59), 3);
        const Moments again = record_moments(def, far);
        CHECK(again.m0 == got.m0);
        CHECK(again.c[1] == got.c[1]);
        CHECK(again.s[4] == got.s[4]);
    }
}

TEST_CASE("Moments: the parallel axis theorem, rotation and the principal axes") {
    // Two 4 x 6 x 10 boxes side by side along x are one 8 x 6 x 10 box.
    Moments two = box_moments(2.0, 3.0, 5.0);
    const double right[3] = { 4.0, 0.0, 0.0 };
    two.add(box_moments(2.0, 3.0, 5.0), right);
    const Moments one = box_moments(4.0, 3.0, 5.0);
    CHECK(two.m0 == doctest::Approx(one.m0));
    CHECK(two.c[0] == doctest::Approx(2.0));
    for (int i = 0; i < 6; i++) {
        CHECK(two.s[i] == doctest::Approx(one.s[i]).epsilon(1e-12));
    }
    // Raw point sums centre to the same moments as adding the points one by one.
    MomentSums sums;
    Moments added;
    for (int i = 0; i < 50; i++) {
        const double at[3] = { std::sin(i * 1.3) * 7.0, std::cos(i * 0.7) * 3.0 + 100.0, double(i % 7) };
        sums.add(0.5 + 0.01 * i, at);
        Moments point;
        point.m0 = 0.5 + 0.01 * i;
        added.add(point, at);
    }
    const Moments centred = sums.moments();
    CHECK(centred.m0 == doctest::Approx(added.m0));
    for (int i = 0; i < 6; i++) {
        CHECK(centred.s[i] == doctest::Approx(added.s[i]).epsilon(1e-9));
    }
    // A turned box keeps its eigenvalues (h^2 / 3, ascending), and its axes are orthonormal.
    std::vector<std::array<float, 9>> rots;
    rotations(rots);
    const std::array<float, 9> &r = rots.back();
    const Moments turned = box_moments(1.0, 3.0, 2.0).rotated(r.data());
    double lambda[3], axes[9];
    turned.principal(lambda, axes);
    CHECK(lambda[0] == doctest::Approx(1.0 / 3.0).epsilon(1e-6));
    CHECK(lambda[1] == doctest::Approx(4.0 / 3.0).epsilon(1e-6));
    CHECK(lambda[2] == doctest::Approx(9.0 / 3.0).epsilon(1e-6));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const double dot = axes[3 * i] * axes[3 * j] + axes[3 * i + 1] * axes[3 * j + 1] + axes[3 * i + 2] * axes[3 * j + 2];
            CHECK(dot == doctest::Approx(i == j ? 1.0 : 0.0).scale(1.0).epsilon(1e-6));
        }
    }
    // The smallest eigenvector of the turned box is its local x, seen in world axes: row 0 of R (R^T e_x).
    const double along = std::abs(axes[0] * r[0] + axes[1] * r[1] + axes[2] * r[2]);
    CHECK(along == doctest::Approx(1.0).epsilon(1e-5));
    // Half thickness: a sphere's radius, 1.118 times a long cylinder's, 1.29 times a box's half side.
    CHECK(sphere_moments(7.0).half_thickness() == doctest::Approx(7.0));
    CHECK(cylinder_moments(500.0, 2.0).half_thickness() == doctest::Approx(2.0 * std::sqrt(1.25)));
    CHECK(box_moments(3.0, 50.0, 50.0).half_thickness() == doctest::Approx(3.0 * std::sqrt(5.0 / 3.0)));
}

TEST_CASE("Moments: a tree's moments per material are the sum of its parts, leaves above bark") {
    DefLibrary lib;
    const int branch = lib.add(make_branch_type(), Material::BARK);
    const int crown = lib.add(make_rock_type(crown_k(0.05), CROWN_SEED), Material::LEAVES);
    const int cone = lib.add(make_cone_type(crown_k(0.05), CROWN_SEED), Material::LEAVES);
    const TreeTypes types = crown_types(branch, crown, lib[crown], cone);
    for (int species = 0; species < int(tree_species().size()); species++) {
        CAPTURE(species);
        const VoxelPos base(1000, 200, -3000);
        std::vector<TableRecord> parts;
        grow_tree(tree_species()[species], base, UP_Y, 0x7EE5u + uint32_t(species), 0.05, types, parts);
        CHECK(!parts.empty());
        const MaterialMoments mm = span_moments(lib, parts.data(), uint32_t(parts.size()), base);
        const Moments *bark = mm.find(Material::BARK);
        const Moments *leaves = mm.find(Material::LEAVES);
        CHECK(bark);
        CHECK(leaves);
        if (!bark || !leaves) {
            continue;
        }
        double bark_sum = 0.0, leaves_sum = 0.0;
        for (const TableRecord &rec : parts) {
            const double m = record_moments(lib[rec.def], rec).m0;
            (lib.material(rec.def) == Material::BARK ? bark_sum : leaves_sum) += m;
        }
        CHECK(bark->m0 == doctest::Approx(bark_sum));
        CHECK(leaves->m0 == doctest::Approx(leaves_sum));
        // The crown is heavier than the wood, sits above it and spreads wider.
        CHECK(leaves->m0 > bark->m0);
        CHECK(leaves->c[1] > bark->c[1]);
        CHECK(leaves->half_thickness() > bark->half_thickness());
        MESSAGE("species ", species, ": ", parts.size(), " parts, bark ", bark->m0 * 1.25e-4, " m3 (half thickness ", bark->half_thickness() * 0.05,
                " m), leaves ", leaves->m0 * 1.25e-4, " m3 (", leaves->half_thickness() * 0.05, " m)");
    }
}
