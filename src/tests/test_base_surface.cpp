#include "test_common.hpp"

#include "base_surface.hpp"

#include <cmath>
#include <map>
#include <string>

using ExactNoise::Float4;
using ExactNoise::float_bits;

namespace {

// Earth's radius at 5 cm voxels, the design note's case.
constexpr double EARTH_R = 6371000.0 / 0.05;

const Vector3i64 CENTRES[] = { Vector3i64(0, 0, 0), Vector3i64(int64_t(1) << 40, -(int64_t(1) << 40) + 12345, 777),
    Vector3i64(-(int64_t(1) << 60) + 99, (int64_t(1) << 60) - 4321, -(int64_t(1) << 59)) };

// A hashed unit direction (double).
void direction(uint32_t p_i, uint32_t p_salt, double r_dir[3]) {
    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ p_i, p_salt, 0xBA5Eu });
    const double u = double(h.x) / 4294967296.0 * 2.0 - 1.0;
    const double phi = double(h.y) / 4294967296.0 * 2.0 * Math_PI;
    const double s = std::sqrt(1.0 - u * u);
    r_dir[0] = s * std::cos(phi);
    r_dir[1] = u;
    r_dir[2] = s * std::sin(phi);
}

// A lattice point of Λ_L near centre + dir * p_dist (LOD0 voxels).
VoxelPos lattice_point(const Vector3i64 &p_centre, const double p_dir[3], double p_dist, int p_lod) {
    VoxelPos p;
    for (int a = 0; a < 3; a++) {
        const int64_t off = int64_t(std::llround(p_dir[a] * p_dist));
        p[a] = WorldCoord::floor_div(p_centre[a] + off, int64_t(1) << p_lod) * (int64_t(1) << p_lod);
    }
    return p;
}

// The exact distance |p - c| - R in long double (the int64 difference is exact), in voxels of LOD p_lod.
long double sphere_exact(const VoxelPos &p_p, const Vector3i64 &p_c, double p_r, int p_lod) {
    long double s = 0.0L;
    for (int a = 0; a < 3; a++) {
        const long double d = (long double)(p_p[a] - p_c[a]);
        s += d * d;
    }
    return std::ldexp(std::sqrt(s) - (long double)p_r, -p_lod);
}

// A torus around the y axis through p_c: major radius p_big, minor p_small. Its exact distance (long double) and
// gradient, the Hessian by central differences of the gradient (its third derivatives are ~ 1 / p_small^2).
struct Torus {
    Vector3i64 c;
    long double big;
    long double small;

    long double distance(const long double p[3]) const {
        const long double rho = std::sqrt(p[0] * p[0] + p[2] * p[2]);
        const long double q = rho - big;
        return std::sqrt(q * q + p[1] * p[1]) - small;
    }
    void gradient(const long double p[3], long double r_g[3]) const {
        const long double rho = std::sqrt(p[0] * p[0] + p[2] * p[2]);
        const long double q = rho - big;
        const long double len = std::sqrt(q * q + p[1] * p[1]);
        r_g[0] = q / len * p[0] / rho;
        r_g[1] = p[1] / len;
        r_g[2] = q / len * p[2] / rho;
    }
    // Region of the cell holding LOD0 point p_v at LOD p_lod, through make_region (the double conversion).
    BaseSurface::Region region(const Vector3i64 &p_cell, int p_lod) const {
        const VoxelPos rc = BaseSurface::region_centre(p_cell, p_lod);
        const long double p[3] = { (long double)(rc.x - c.x), (long double)(rc.y - c.y), (long double)(rc.z - c.z) };
        long double g[3];
        gradient(p, g);
        long double hm[3][3];
        const long double step = 64.0L;
        for (int a = 0; a < 3; a++) {
            long double pp[3] = { p[0], p[1], p[2] };
            long double pm[3] = { p[0], p[1], p[2] };
            pp[a] += step;
            pm[a] -= step;
            long double gp[3], gm[3];
            gradient(pp, gp);
            gradient(pm, gm);
            for (int b = 0; b < 3; b++) {
                hm[a][b] = (gp[b] - gm[b]) / (2.0L * step);
            }
        }
        const double dg[3] = { double(g[0]), double(g[1]), double(g[2]) };
        const double dh[6] = { double(hm[0][0]), double(hm[1][1]), double(hm[2][2]), double((hm[0][1] + hm[1][0]) / 2.0L),
            double((hm[0][2] + hm[2][0]) / 2.0L), double((hm[1][2] + hm[2][1]) / 2.0L) };
        return BaseSurface::make_region(double(distance(p)), dg, dh, p_lod);
    }
};

} // namespace

// The flat base is the datum difference, saturated, with up (-0, 1, -0) (the default terrain's bits: the SDFCascade
// pin).
TEST_CASE("BaseSurface: the flat base is the saturated datum difference, up +y") {
    const BaseSurface b = BaseSurface::flat(600);
    CHECK(b.kind() == BaseSurface::Kind::FLAT);
    CHECK(b.datum_y() == 600);
    for (const int lod : { 0, 3, 17 }) {
        for (const int64_t y : { int64_t(600), int64_t(0), int64_t(-123456), int64_t(1) << 50, -(int64_t(1) << 50) }) {
            const Float4 a = b.at(VoxelPos(int64_t(1) << 59, y, -999), lod);
            CHECK(float_bits(a.x) == 0x80000000u);
            CHECK(a.y == 1.0f);
            CHECK(float_bits(a.z) == 0x80000000u);
            CHECK(a.w == float(WorldCoord::sat_diff(y, 600, BaseSurface::ALT_LIMIT)) * ExactNoise::exact_pow2(-lod));
        }
    }
    const UpAxis up = b.up_dir(VoxelPos(1, 2, 3));
    CHECK(up[1] == 1.0);
    CHECK(!b.glsl().contains("DATUM_LO"));
}

// Check 1 of the design note: the curvature form within 1% of a cell of the exact distance at LODs 0-16, the radius
// form from the switch LOD, near 0, +-2^40 and +-2^60, points within 4 cells of an Earth-sized sphere. Measured
// (worst, cells): curvature 7e-6 at LODs 0-11, 0.0068 at 16, 0.025 at 17 (the design note's sampling gave 0.004
// there: its points sat closer to the region centres); radius 0.0039 at 12, falling by ~2 per LOD.
TEST_CASE("BaseSurface: an Earth-sized sphere, both forms within 1% of a cell of the exact distance") {
    double worst_curv[17] = {};
    double worst_radius[25] = {};
    double worst_up = 0.0;
    for (const Vector3i64 &raw : CENTRES) {
        const BaseSurface b = BaseSurface::sphere(raw, EARTH_R);
        const Vector3i64 c = b.centre();
        for (int lod = 0; lod < 25; lod++) {
            const double cell = std::ldexp(1.0, lod);
            for (uint32_t i = 0; i < 200; i++) {
                double dir[3];
                direction(i, uint32_t(lod), dir);
                const double alt = (double(i % 17) / 2.0 - 4.0) * cell;
                const VoxelPos p = lattice_point(c, dir, EARTH_R + alt, lod);
                const long double exact = sphere_exact(p, c, EARTH_R, lod);
                if (lod < 17) {
                    const Float4 f = BaseSurface::curvature_form(p, lod, b.region(BaseSurface::region_cell(p, lod), lod));
                    worst_curv[lod] = MAX(worst_curv[lod], double(std::fabs((long double)f.w - exact)));
                    // The up direction: the exact normal at p, to first order.
                    long double n[3], len = 0.0L;
                    for (int a = 0; a < 3; a++) {
                        n[a] = (long double)(p[a] - c[a]);
                        len += n[a] * n[a];
                    }
                    len = std::sqrt(len);
                    const double du = double(std::fabs((long double)f.x - n[0] / len) + std::fabs((long double)f.y - n[1] / len) +
                            std::fabs((long double)f.z - n[2] / len));
                    if (lod < BaseSurface::SWITCH_LOD) {
                        worst_up = MAX(worst_up, du); // at() reads the curvature form there only
                    }
                }
                if (lod >= BaseSurface::SWITCH_LOD || lod == 0) {
                    const Float4 f = b.radius_form(p, lod);
                    worst_radius[lod] = MAX(worst_radius[lod], double(std::fabs((long double)f.w - exact)));
                }
                // at() picks the form by the switch LOD.
                const Float4 a = b.at(p, lod);
                const Float4 want = lod < BaseSurface::SWITCH_LOD ? BaseSurface::curvature_form(p, lod, b.region(BaseSurface::region_cell(p, lod), lod))
                                                                  : b.radius_form(p, lod);
                CHECK(float_bits(a.w) == float_bits(want.w));
            }
        }
    }
    std::string table;
    for (int lod = 0; lod < 17; lod++) {
        CAPTURE(lod);
        CHECK(worst_curv[lod] < 0.01);
        table += " L" + std::to_string(lod) + " " + std::to_string(worst_curv[lod]);
    }
    MESSAGE("curvature form, worst error in cells:", table);
    table.clear();
    for (int lod = BaseSurface::SWITCH_LOD; lod < 25; lod++) {
        CAPTURE(lod);
        CHECK(worst_radius[lod] < 0.01);
        table += " L" + std::to_string(lod) + " " + std::to_string(worst_radius[lod]);
    }
    MESSAGE("radius form, worst error in cells:", table);
    // Why the curvature form at fine LODs: the radius form is several voxels off at LOD 0.
    CHECK(worst_radius[0] > 1.0);
    MESSAGE("up direction below the switch, worst error (sum of components): ", worst_up);
    CHECK(worst_up < 1e-5);
}

// The two forms meet at the switch LOD within 1% of a cell, so the switch moves no surface by more than that.
TEST_CASE("BaseSurface: the curvature and radius forms agree at the switch LOD") {
    const int lod = BaseSurface::SWITCH_LOD;
    double worst = 0.0;
    for (const Vector3i64 &raw : CENTRES) {
        const BaseSurface b = BaseSurface::sphere(raw, EARTH_R);
        const Vector3i64 c = b.centre();
        for (uint32_t i = 0; i < 400; i++) {
            double dir[3];
            direction(i, 0x5717u, dir);
            const VoxelPos p = lattice_point(c, dir, EARTH_R + (double(i % 9) - 4.0) * std::ldexp(1.0, lod), lod);
            const Float4 curv = BaseSurface::curvature_form(p, lod, b.region(BaseSurface::region_cell(p, lod), lod));
            const Float4 rad = b.radius_form(p, lod);
            worst = MAX(worst, double(std::fabs(curv.w - rad.w)));
        }
    }
    MESSAGE("forms' gap at LOD ", lod, ": ", worst, " cells");
    CHECK(worst < 0.01);
}

// Any smooth base through make_region: a torus (major 2^26, minor 2^24 voxels), below the switch LOD.
TEST_CASE("BaseSurface: a torus through make_region, the curvature form within 1% of a cell") {
    for (const Vector3i64 &c : CENTRES) {
        const Torus t{ c, std::ldexp(1.0L, 26), std::ldexp(1.0L, 24) };
        for (int lod = 0; lod < BaseSurface::SWITCH_LOD; lod++) {
            double worst = 0.0;
            for (uint32_t i = 0; i < 200; i++) {
                // A point near the tube: around the ring at angle a, on the tube at angle b.
                const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(lod), 0x7085u });
                const long double a = (long double)h.x / 4294967296.0L * 2.0L * Math_PI;
                const long double bb = (long double)h.y / 4294967296.0L * 2.0L * Math_PI;
                const long double r = t.small + (long double)(int(i % 9) - 4) * std::ldexp(1.0L, lod);
                const long double rho = t.big + r * std::cos(bb);
                VoxelPos p;
                const long double off[3] = { rho * std::cos(a), r * std::sin(bb), rho * std::sin(a) };
                for (int ax = 0; ax < 3; ax++) {
                    p[ax] = WorldCoord::floor_div(c[ax] + int64_t(std::llround(double(off[ax]))), int64_t(1) << lod) * (int64_t(1) << lod);
                }
                const long double q[3] = { (long double)(p.x - c.x), (long double)(p.y - c.y), (long double)(p.z - c.z) };
                const long double exact = std::ldexp(t.distance(q), -lod);
                const Float4 f = BaseSurface::curvature_form(p, lod, t.region(BaseSurface::region_cell(p, lod), lod));
                worst = MAX(worst, double(std::fabs((long double)f.w - exact)));
            }
            CAPTURE(lod);
            CHECK(worst < 0.01);
        }
    }
}

// Check 2 of the design note: a sample shared by two chunks of one LOD gets the same bits from each, because its
// region comes from the point and the LOD. Each chunk uploads the regions its box touches (1 to 8); the shared face
// points are evaluated through each chunk's own table.
TEST_CASE("BaseSurface: two chunks of one LOD give a shared sample the same bits") {
    const BaseSurface b = BaseSurface::sphere(CENTRES[1], EARTH_R);
    const Vector3i64 c = b.centre();
    int shared = 0;
    for (const int lod : { 0, 4, 11 }) {
        const int64_t cell = int64_t(1) << lod;
        const int64_t span = 28 * cell; // a chunk: 28 cells
        double dir[3];
        direction(7, uint32_t(lod), dir);
        const VoxelPos surf = lattice_point(c, dir, EARTH_R, lod);
        const VoxelPos lo_a(WorldCoord::floor_div(surf.x, span) * span, WorldCoord::floor_div(surf.y, span) * span, WorldCoord::floor_div(surf.z, span) * span);
        const VoxelPos lo_b = lo_a + VoxelPos(span, 0, 0); // the +x neighbour
        auto table_of = [&](const VoxelPos &p_lo) {
            std::map<std::array<int64_t, 3>, BaseSurface::Region> t;
            for (int64_t i = 0; i <= 28; i++) {
                for (int64_t j = 0; j <= 28; j++) {
                    for (int64_t k = 0; k <= 28; k++) {
                        const Vector3i64 rc = BaseSurface::region_cell(p_lo + VoxelPos(i, j, k) * cell, lod);
                        t[{ rc.x, rc.y, rc.z }] = b.region(rc, lod);
                    }
                }
            }
            return t;
        };
        const auto ta = table_of(lo_a);
        const auto tb = table_of(lo_b);
        CHECK(ta.size() <= 8);
        CHECK(tb.size() <= 8);
        for (int64_t j = 0; j <= 28; j++) {
            for (int64_t k = 0; k <= 28; k++) {
                const VoxelPos p = lo_b + VoxelPos(0, j, k) * cell; // the shared face
                const Vector3i64 rc = BaseSurface::region_cell(p, lod);
                const Float4 fa = BaseSurface::curvature_form(p, lod, ta.at({ rc.x, rc.y, rc.z }));
                const Float4 fb = BaseSurface::curvature_form(p, lod, tb.at({ rc.x, rc.y, rc.z }));
                CHECK(std::memcmp(&fa, &fb, sizeof(Float4)) == 0);
                shared++;
            }
        }
    }
    CHECK(shared == 3 * 29 * 29);
}

// sphere() refuses what its radius form can't hold to 1% of a cell, and returns the default (flat) base.
TEST_CASE("BaseSurface: a sphere the radius form can't hold is refused") {
    CHECK(BaseSurface::sphere(VoxelPos(), EARTH_R).kind() == BaseSurface::Kind::SPHERE);
    CHECK(BaseSurface::sphere(VoxelPos(), EARTH_R, 10).kind() == BaseSurface::Kind::SPHERE); // 124,000 cells
    CHECK(BaseSurface::sphere(VoxelPos(), EARTH_R, 9).kind() == BaseSurface::Kind::FLAT); // 249,000 cells
    CHECK(BaseSurface::sphere(VoxelPos(), -1.0).kind() == BaseSurface::Kind::FLAT);
    CHECK(BaseSurface::sphere(VoxelPos(), EARTH_R, BaseSurface::MAX_SWITCH_LOD + 1).kind() == BaseSurface::Kind::FLAT);
    // The centre is snapped down to the 2^31 grid.
    const int64_t g = int64_t(1) << BaseSurface::CENTRE_SHIFT;
    CHECK(BaseSurface::sphere(VoxelPos(g + 5, -1, 3 * g), EARTH_R).centre() == VoxelPos(g, -g, 3 * g));
}

// The reports' frame: the reference tangent plane and the altitude above the base.
TEST_CASE("BaseSurface: the tangent plane and the altitude") {
    const BaseSurface flat = BaseSurface::flat(6000);
    CHECK(flat.tangent_point(-7, 11) == VoxelPos(-7, 6000, 11));
    CHECK(flat.altitude(VoxelPos(5, 6123, -9)) == 123.0);
    const BaseSurface s = BaseSurface::sphere(CENTRES[1], EARTH_R);
    const VoxelPos pole = s.tangent_point(0, 0);
    CHECK(std::fabs(s.altitude(pole)) <= 0.5);
    const UpAxis up = s.up_dir(pole);
    CHECK(up[1] == doctest::Approx(1.0).epsilon(1e-12));
    // 10 km along the plane: it rises above the sphere by sqrt(R^2 + u^2) - R.
    const int64_t u = 200000;
    const double rise = std::sqrt(EARTH_R * EARTH_R + double(u) * double(u)) - EARTH_R;
    CHECK(s.altitude(s.tangent_point(u, 0)) == doctest::Approx(rise).epsilon(1e-6));
}

// A stretched octave's point on the flat base: datum + (y - datum) / 2^j, exact, negative y and far points included, a
// fraction that rounds to 1 carried; the gradient's y over 2^j.
TEST_CASE("BaseSurface: the flat stretch is datum + (y - datum) / 2^j, exactly") {
    const int64_t datum = 6000; // 300 m at 5 cm: not a multiple of 2^j
    const BaseSurface b = BaseSurface::flat(datum);
    int checked = 0;
    for (const int j : { 1, 3, 8, 20 }) {
        for (const int64_t y : { int64_t(0), int64_t(5999), int64_t(6000), int64_t(-1), int64_t(-123457), (int64_t(1) << 40) + 3, -(int64_t(1) << 60) + 7 }) {
            for (const float frac : { 0.0f, 0.25f, 1.0f - 1.0f / 16777216.0f }) {
                WorldPos p;
                p.voxel = VoxelPos(11, y, -13);
                p.frac = Float3{ 0.5f, frac, 0.0f };
                const WorldPos s = b.stretched(p, j, ExactNoise::Float4{}, 0);
                CHECK(s.voxel.x == 11);
                CHECK(s.voxel.z == -13);
                CHECK(s.frac.x == 0.5f);
                CHECK(s.frac.y >= 0.0f);
                CHECK(s.frac.y < 1.0f);
                // (y - datum + frac) / 2^j, exact in long double, against the stretched point's offset from the datum.
                const long double want = std::ldexp((long double)(y - datum) + (long double)frac, -j);
                const long double got = (long double)(s.voxel.y - datum) + (long double)s.frac.y;
                CHECK(std::fabs(double(got - want)) <= std::ldexp(1.0, -24));
                checked++;
            }
        }
    }
    CHECK(checked == 4 * 7 * 3);
    // The carry: remainder 2^j - 1 plus a fraction just under 1 rounds to 1.
    WorldPos p;
    p.voxel = VoxelPos(0, datum + 255, 0);
    p.frac = Float3{ 0.0f, 1.0f - 1.0f / 16777216.0f, 0.0f };
    const WorldPos s = b.stretched(p, 8, ExactNoise::Float4{}, 0);
    CHECK(s.voxel.y == datum + 1);
    CHECK(s.frac.y == 0.0f);
    const ExactNoise::Float4 g = b.stretch_gradient(ExactNoise::Float4{ 0.3f, -0.7f, 0.1f, 0.5f }, 3, ExactNoise::Float4{});
    CHECK(g.x == 0.3f);
    CHECK(g.y == -0.7f * 0.125f);
    CHECK(g.z == 0.1f);
    CHECK(g.w == 0.5f);
}

// On a sphere: the point moves by (1 - 2^-j) d n (d, n: the base sample), the gradient's up component scales by 2^-j.
TEST_CASE("BaseSurface: the sphere stretch moves the point along the base's up") {
    const BaseSurface b = BaseSurface::sphere(CENTRES[2], EARTH_R);
    const Vector3i64 c = b.centre();
    double worst_move = 0.0, worst_grad = 0.0;
    for (uint32_t i = 0; i < 300; i++) {
        double dir[3];
        direction(i, 0x57E7u, dir);
        const int m = int(i % 6);
        const double alt = (double(i % 23) - 11.0) * 40.0; // up to 440 voxels off the surface
        WorldPos p;
        p.voxel = lattice_point(c, dir, EARTH_R + alt, m);
        p.frac = Float3{ 0.125f, 0.5f, 0.875f };
        const ExactNoise::Float4 base = b.stretch_base(p.voxel, m);
        for (const int j : { 1, 3, 8 }) {
            const WorldPos s = b.stretched(p, j, base, m);
            const double k = (1.0 - std::ldexp(1.0, -j)) * double(base.w) * std::ldexp(1.0, m);
            const double n[3] = { double(base.x), double(base.y), double(base.z) };
            const double frac_p[3] = { 0.125, 0.5, 0.875 };
            const double frac_s[3] = { double(s.frac.x), double(s.frac.y), double(s.frac.z) };
            for (int a = 0; a < 3; a++) {
                const double moved = double(s.voxel[a] - p.voxel[a]) + frac_s[a] - frac_p[a];
                worst_move = MAX(worst_move, std::fabs(moved + k * n[a]));
                CHECK(frac_s[a] >= 0.0);
                CHECK(frac_s[a] < 1.0);
            }
            const ExactNoise::Float4 g{ 0.3f, -0.7f, 0.1f, 0.5f };
            const ExactNoise::Float4 h = b.stretch_gradient(g, j, base);
            const double gn = 0.3 * n[0] - 0.7 * n[1] + 0.1 * n[2];
            const double hn = double(h.x) * n[0] + double(h.y) * n[1] + double(h.z) * n[2];
            worst_grad = MAX(worst_grad, std::fabs(hn - std::ldexp(gn, -j)));
            CHECK(h.w == 0.5f);
        }
    }
    MESSAGE("sphere stretch: worst move error ", worst_move, " voxels, worst up-gradient error ", worst_grad);
    CHECK(worst_move < 1e-3);
    CHECK(worst_grad < 1e-5);
}
