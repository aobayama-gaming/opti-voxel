#include "test_common.hpp"

#include "occlusion_map.hpp"

#include <cmath>
#include <vector>

namespace {

using Vec3 = OcclusionMap::Vec3;

struct Ball {
    Vec3 c;
    double r;
};

// Deterministic doubles in [0, 1) (an LCG: the same sequence on every platform).
struct Lcg {
    uint64_t state;
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(state >> 11) * (1.0 / 9007199254740992.0);
    }
    double range(double p_lo, double p_hi) { return p_lo + (p_hi - p_lo) * next(); }
};

// The ray from p_eye towards p_target enters one of p_balls before reaching p_target.
bool ray_blocked(const Vec3 &p_eye, const Vec3 &p_target, const std::vector<Ball> &p_balls) {
    const Vec3 d{ p_target.x - p_eye.x, p_target.y - p_eye.y, p_target.z - p_eye.z };
    const double len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    for (const Ball &b : p_balls) {
        const Vec3 oc{ b.c.x - p_eye.x, b.c.y - p_eye.y, b.c.z - p_eye.z };
        const double t = (oc.x * d.x + oc.y * d.y + oc.z * d.z) / len; // closest approach along the ray
        const double h2 = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - t * t;
        if (h2 >= b.r * b.r) {
            continue;
        }
        const double enter = t - std::sqrt(b.r * b.r - h2);
        if (enter < len && t + std::sqrt(b.r * b.r - h2) > 0.0) {
            return true;
        }
    }
    return false;
}

Vec3 point_in(const Vec3 &p_lo, const Vec3 &p_hi, Lcg &r_rng) {
    return { r_rng.range(p_lo.x, p_hi.x), r_rng.range(p_lo.y, p_hi.y), r_rng.range(p_lo.z, p_hi.z) };
}

// Random balls around the eye (none holding it), each shrunk by p_margin in the map.
void random_scene(Lcg &r_rng, OcclusionMap &r_map, std::vector<Ball> &r_balls, double p_margin) {
    r_map.clear();
    r_balls.clear();
    for (int i = 0; i < 40; i++) {
        Ball b{ { r_rng.range(-120, 120), r_rng.range(-120, 120), r_rng.range(-120, 120) }, r_rng.range(4, 40) };
        if (std::sqrt(b.c.x * b.c.x + b.c.y * b.c.y + b.c.z * b.c.z) <= b.r + p_margin) {
            continue;
        }
        r_balls.push_back(b);
        r_map.add_ball(b.c, b.r - p_margin);
    }
}

} // namespace

// A ball in front of the eye hides what is behind it and nothing else.
TEST_CASE("OcclusionMap: a ball hides the box behind it") {
    OcclusionMap map;
    map.add_ball({ 50, 0, 0 }, 20);
    CHECK(map.box_hidden({ 100, -2, -2 }, { 104, 2, 2 })); // straight behind
    CHECK_FALSE(map.box_hidden({ 20, -2, -2 }, { 24, 2, 2 })); // in front of it
    CHECK_FALSE(map.box_hidden({ 100, 60, -2 }, { 104, 64, 2 })); // beside its shadow
    CHECK_FALSE(map.box_hidden({ -104, -2, -2 }, { -100, 2, 2 })); // behind the eye
    CHECK_FALSE(map.box_hidden({ -1, -1, -1 }, { 1, 1, 1 })); // around the eye
    CHECK(map.depth_at({ 1, 0, 0 }) == doctest::Approx(50.0));
    CHECK(std::isinf(map.depth_at({ 0, 1, 0 })));
    CHECK(map.get_stats().balls_written == 1);
}

// A ball holding the eye, or too small for a pixel, writes nothing.
TEST_CASE("OcclusionMap: skipped balls") {
    OcclusionMap map;
    map.add_ball({ 1, 0, 0 }, 2);
    map.add_ball({ 1000, 0, 0 }, 0.5);
    map.add_ball({ 0, 50, 0 }, -1);
    CHECK(map.get_stats().balls == 3);
    CHECK(map.get_stats().pixels_written == 0);
    CHECK_FALSE(map.box_hidden({ 100, -2, -2 }, { 104, 2, 2 }));
}

// A ball across a cube face's edge and corner writes on every face it covers.
TEST_CASE("OcclusionMap: balls across faces") {
    OcclusionMap map;
    map.add_ball({ 30, 30, 30 }, 20); // on the corner of three faces
    CHECK(map.depth_at({ 1, 1, 1 }) == doctest::Approx(std::sqrt(2700.0)));
    CHECK(map.depth_at({ 1, 0.9, 1.1 }) < 100.0);
    CHECK(map.depth_at({ 0.9, 1.1, 1 }) < 100.0);
    CHECK(map.box_hidden({ 80, 80, 80 }, { 82, 82, 82 }));
}

// Soundness: every ray from the eye to a box called hidden enters a ball first (rays to random points of the
// box, against the exact ray / sphere intersection). Enough boxes are hidden for the check to mean something.
TEST_CASE("OcclusionMap: hidden boxes are hidden from every ray") {
    Lcg rng{ 0x0CC1u };
    OcclusionMap map;
    std::vector<Ball> balls;
    int hidden = 0;
    int unsound = 0;
    for (int scene = 0; scene < 30; scene++) {
        random_scene(rng, map, balls, 0.0);
        for (int b = 0; b < 300; b++) {
            const Vec3 c{ rng.range(-250, 250), rng.range(-250, 250), rng.range(-250, 250) };
            const double h = rng.range(0.5, 12);
            const Vec3 lo{ c.x - h, c.y - h, c.z - h };
            const Vec3 hi{ c.x + h, c.y + h, c.z + h };
            if (!map.box_hidden(lo, hi)) {
                continue;
            }
            hidden++;
            for (int r = 0; r < 64; r++) {
                unsound += ray_blocked({ 0, 0, 0 }, point_in(lo, hi, rng), balls) ? 0 : 1;
            }
        }
    }
    CAPTURE(hidden);
    CHECK(hidden > 500);
    CHECK(unsound == 0);
}

// The margin: balls shrunk by R in the map, a box hidden from the eye is hidden from any eye within R of it.
TEST_CASE("OcclusionMap: shrunk balls hide from every eye within the margin") {
    Lcg rng{ 0x3A26u };
    OcclusionMap map;
    std::vector<Ball> balls;
    const double margin = 3.0;
    int hidden = 0;
    int unsound = 0;
    for (int scene = 0; scene < 30; scene++) {
        random_scene(rng, map, balls, margin);
        for (int b = 0; b < 300; b++) {
            const Vec3 c{ rng.range(-250, 250), rng.range(-250, 250), rng.range(-250, 250) };
            const double h = rng.range(0.5, 12);
            const Vec3 lo{ c.x - h, c.y - h, c.z - h };
            const Vec3 hi{ c.x + h, c.y + h, c.z + h };
            if (!map.box_hidden(lo, hi)) {
                continue;
            }
            hidden++;
            for (int r = 0; r < 32; r++) {
                // An eye anywhere in the ball of radius `margin` around the map's eye.
                Vec3 e;
                do {
                    e = { rng.range(-margin, margin), rng.range(-margin, margin), rng.range(-margin, margin) };
                } while (e.x * e.x + e.y * e.y + e.z * e.z > margin * margin);
                unsound += ray_blocked(e, point_in(lo, hi, rng), balls) ? 0 : 1;
            }
        }
    }
    CAPTURE(hidden);
    CHECK(hidden > 300);
    CHECK(unsound == 0);
}
