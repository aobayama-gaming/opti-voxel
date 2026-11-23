#include "test_common.hpp"

#include "base_surface.hpp"
#include "sea.hpp"

#include <cmath>

namespace {

constexpr double VS = 0.05;
constexpr double EARTH_R = 6371000.0 / VS;
constexpr double SMALL_R = 2000.0 / VS; // a 2 km planet

// base_level_span_k's span in double: the roots of |p + t dir - c| = R + level (SPHERE), the level plane (FLAT); p and dir
// render space (metres from the render origin p_origin).
bool reference_span(const BaseSurface &p_base, const VoxelPos &p_origin, double p_level_m, const double p_p[3], const double p_dir[3], double r_t[2]) {
    if (p_base.kind() == BaseSurface::Kind::FLAT) {
        const double below = double(p_base.datum_y() - p_origin.y) * VS + p_level_m - p_p[1];
        const double level = below / p_dir[1];
        r_t[0] = p_dir[1] < 0.0 ? level : -1e30;
        r_t[1] = p_dir[1] > 0.0 ? level : 1e30;
        return p_dir[1] != 0.0 || below > 0.0;
    }
    const VoxelPos &c = p_base.centre();
    const double d[3] = { p_p[0] - double(c.x - p_origin.x) * VS, p_p[1] - double(c.y - p_origin.y) * VS, p_p[2] - double(c.z - p_origin.z) * VS };
    const double r = p_base.get_radius() * VS + p_level_m;
    const double b = d[0] * p_dir[0] + d[1] * p_dir[1] + d[2] * p_dir[2];
    const double disc = b * b - (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - r * r);
    const double s = std::sqrt(std::max(disc, 0.0));
    r_t[0] = -b - s;
    r_t[1] = -b + s;
    return disc >= 0.0;
}

// The float span (Render::level_span_k, the GLSL's formula) against the double one for camera p_p looking along p_dir: the
// same answer, each finite root within 1 mm plus 1e-4 of its length. Returns the float span in r_t.
void check_span(const BaseSurface &p_base, const VoxelPos &p_origin, double p_level_m, const double p_p[3], const double p_dir[3], float r_t[2]) {
    const BaseSurface::Render render = p_base.render(p_origin, VS);
    const float k = float(p_base.level_k(p_origin, VS, p_level_m));
    const double len = std::sqrt(p_dir[0] * p_dir[0] + p_dir[1] * p_dir[1] + p_dir[2] * p_dir[2]);
    const double dir[3] = { p_dir[0] / len, p_dir[1] / len, p_dir[2] / len };
    double want[2];
    const bool want_hit = reference_span(p_base, p_origin, p_level_m, p_p, dir, want);
    const bool hit = render.level_span_k(Float3{ float(p_p[0]), float(p_p[1]), float(p_p[2]) }, Float3{ float(dir[0]), float(dir[1]), float(dir[2]) }, k, r_t);
    CHECK(hit == want_hit);
    for (int i = 0; hit && want_hit && i < 2; i++) {
        if (std::fabs(want[i]) < 1e29) {
            CHECK(std::fabs(double(r_t[i]) - want[i]) <= 1e-3 + 1e-4 * std::fabs(want[i]));
        }
    }
}

// The render origin on the sphere's +y pole (at p_alt_m above its base), a few metres off the axis.
VoxelPos pole_origin(const BaseSurface &p_base, double p_alt_m) {
    return p_base.centre() + VoxelPos(37, std::llround(p_base.get_radius() + p_alt_m / VS), -91);
}

} // namespace

// The sea's level along a ray (base_level_span_k with k from double) at Earth's radius and on a 2 km planet: the camera
// above it (looking down, slanted, up), under it (its exit above, its entry behind), and grazing the horizon (just under:
// the far hit; just over: none).
TEST_CASE("BaseSurface: base_level_span_k matches a double reference on a sphere") {
    for (const double radius : { EARTH_R, SMALL_R }) {
        CAPTURE(radius);
        const BaseSurface s = BaseSurface::sphere(VoxelPos(int64_t(1) << 40, -(int64_t(1) << 33), 12345), radius);
        for (const double level : { 0.0, 12.5, -40.0 }) {
            CAPTURE(level);
            const VoxelPos origin = pole_origin(s, level + 3.0);
            float t[2];
            // Above: 100 m over the level.
            const double above[3] = { 13.5, 97.0, -7.25 };
            const double down[3] = { 0.0, -1.0, 0.0 };
            check_span(s, origin, level, above, down, t);
            CHECK(t[0] == doctest::Approx(100.0).epsilon(2e-3)); // the camera off the axis: the 2 km planet curves away
            const double slant[3] = { 0.6, -1.0, 0.3 };
            check_span(s, origin, level, above, slant, t);
            const double up[3] = { 0.1, 1.0, 0.0 };
            check_span(s, origin, level, above, up, t);
            CHECK(t[1] < 0.0); // behind
            // Under: 5 m below the level; the exit 5 m up, the far root behind.
            const double under[3] = { -2.0, -8.0, 4.0 };
            check_span(s, origin, level, under, up, t);
            CHECK(t[0] < 0.0);
            CHECK(t[1] == doctest::Approx(5.0 * std::sqrt(1.01)).epsilon(1e-2));
            check_span(s, origin, level, under, down, t);
            CHECK(t[0] < 0.0);
            // Grazing: the horizon of a camera 100 m up dips by acos(r / (r + 100)) below its local level; 20% of it more
            // hits the level far out, 20% less misses.
            const double r = radius * VS + level;
            const double dip = std::acos(r / (r + 100.0));
            for (const double f : { 1.2, 0.8 }) {
                const double graze[3] = { std::cos(dip * f), -std::sin(dip * f), 0.0 };
                const double camera[3] = { 0.0, 97.0, 0.0 };
                check_span(s, origin, level, camera, graze, t);
            }
        }
    }
}

TEST_CASE("BaseSurface: base_level_span_k on the flat base is the level plane") {
    const BaseSurface flat = BaseSurface::flat(6000);
    const VoxelPos origin(int64_t(1) << 50, 6000 + 2000, -(int64_t(1) << 45)); // 100 m over the datum
    const double level = 2.0;
    CHECK(flat.level_k(origin, VS, level) == doctest::Approx(-98.0));
    float t[2];
    const double camera[3] = { 3.0, 4.0, -1.0 }; // 102 m over the level
    const double down45[3] = { 1.0, -1.0, 0.0 };
    check_span(flat, origin, level, camera, down45, t);
    CHECK(t[0] == doctest::Approx(102.0 * std::sqrt(2.0)).epsilon(1e-5));
    CHECK(t[1] > 1e29f);
    const double up[3] = { 0.0, 1.0, 0.0 };
    check_span(flat, origin, level, camera, up, t); // above it looking up: never under it
    CHECK(t[1] < 0.0);
    const double under[3] = { 0.0, -110.0, 0.0 };
    check_span(flat, origin, level, under, up, t);
    CHECK(t[0] < -1e29f);
    CHECK(t[1] == doctest::Approx(12.0).epsilon(1e-5)); // 10 m under the datum, 12 under the level
}

// level_k at level 0 is the render K (Render::k_m2), and the torus has none.
TEST_CASE("BaseSurface: level_k at level 0 is the base's K") {
    const BaseSurface s = BaseSurface::sphere(VoxelPos(), EARTH_R);
    const VoxelPos origin = pole_origin(s, 250.0);
    CHECK(float(s.level_k(origin, VS, 0.0)) == s.render(origin, VS).k_m2);
    CHECK(std::isnan(BaseSurface::torus(VoxelPos(), 120000.0, 40000.0, 1).level_k(origin, VS, 0.0)));
}

// The player's sea depth (VoxelTerrain.get_sea_depth_at): the height above the level, its sign under water, on both bases.
TEST_CASE("Sea: height_at is the base altitude minus the level, negative under water") {
    const BaseSurface flat = BaseSurface::flat(6000);
    WorldPos p;
    p.voxel = VoxelPos(17, 6000 + 100, -3); // 5 m over the datum
    p.frac = Float3{ 0.5f, 0.5f, 0.0f };
    CHECK(Sea::height_at(flat, p, VS, 2.0) == doctest::Approx(5.025 - 2.0));
    CHECK(Sea::height_at(flat, p, VS, 8.0) == doctest::Approx(5.025 - 8.0));
    CHECK(Sea::height_at(flat, p, VS, 8.0) < 0.0);
    for (const double radius : { EARTH_R, SMALL_R }) {
        const BaseSurface s = BaseSurface::sphere(VoxelPos(), radius);
        WorldPos q;
        q.voxel = s.centre() + VoxelPos(0, std::llround(radius) - 60, 0); // 3 m under the base at the pole
        CHECK(Sea::height_at(s, q, VS, 0.0) == doctest::Approx(-3.0).epsilon(1e-6));
        CHECK(Sea::height_at(s, q, VS, -10.0) == doctest::Approx(7.0).epsilon(1e-6));
    }
    // Render: off without a sea or on a torus; its k the base's level_k.
    WorldFile::Sea sea;
    CHECK(Sea::render(sea, flat, VoxelPos(), VS).on == 0);
    sea.on = true;
    sea.level_m = 2.0;
    const Sea::Render r = Sea::render(sea, flat, VoxelPos(0, 7000, 0), VS);
    CHECK(r.on == 1);
    CHECK(r.k == float(flat.level_k(VoxelPos(0, 7000, 0), VS, 2.0)));
    CHECK(Sea::render(sea, BaseSurface::torus(VoxelPos(), 120000.0, 40000.0, 1), VoxelPos(), VS).on == 0);
}

// The shaders' text has every constant substituted (no $NAME left) and reads the uniforms Render declares.
TEST_CASE("Sea: the GLSL is substituted") {
    for (const ov::str &text : { Sea::glsl_shore(), Sea::glsl_water() }) {
        CHECK(!ov::contains(text, "$"));
    }
    CHECK(ov::contains(Sea::glsl_uniforms(), "uniform float sea_k;"));
    CHECK(ov::contains(Sea::glsl_uniforms(), "uniform int sea_on;"));
    CHECK(ov::contains(Sea::glsl_water(), "base_level_span_k(camera, dir, sea_k, t)"));
}
