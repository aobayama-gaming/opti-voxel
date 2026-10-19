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

namespace {

struct Box {
    Vec3 lo, hi;
};

// The segment from p_eye to p_target meets one of p_boxes (the slab test).
bool segment_blocked(const Vec3 &p_eye, const Vec3 &p_target, const std::vector<Box> &p_boxes) {
    const Vec3 d{ p_target.x - p_eye.x, p_target.y - p_eye.y, p_target.z - p_eye.z };
    for (const Box &b : p_boxes) {
        double t0 = 0.0, t1 = 1.0;
        for (int a = 0; a < 3 && t0 <= t1; a++) {
            if (std::abs(d[a]) < 1e-12) {
                if (p_eye[a] < b.lo[a] || p_eye[a] > b.hi[a]) {
                    t1 = -1.0;
                }
                continue;
            }
            const double s0 = (b.lo[a] - p_eye[a]) / d[a];
            const double s1 = (b.hi[a] - p_eye[a]) / d[a];
            t0 = std::max(t0, std::min(s0, s1));
            t1 = std::min(t1, std::max(s0, s1));
        }
        if (t0 <= t1) {
            return true;
        }
    }
    return false;
}

// Random boxes around the eye (none within p_margin of it), each shrunk by p_margin in the map; flat ones too
// (walls, trunks).
void random_box_scene(Lcg &r_rng, OcclusionMap &r_map, std::vector<Box> &r_boxes, double p_margin) {
    r_map.clear();
    r_boxes.clear();
    for (int i = 0; i < 40; i++) {
        const Vec3 c{ r_rng.range(-120, 120), r_rng.range(-120, 120), r_rng.range(-120, 120) };
        const Vec3 h{ r_rng.range(2, 40), r_rng.range(2, 40), r_rng.range(2, 40) };
        const Box b{ { c.x - h.x, c.y - h.y, c.z - h.z }, { c.x + h.x, c.y + h.y, c.z + h.z } };
        bool near = true;
        for (int a = 0; a < 3; a++) {
            near = near && b.lo[a] - p_margin <= 0.0 && b.hi[a] + p_margin >= 0.0;
        }
        if (near) {
            continue;
        }
        r_boxes.push_back(b);
        r_map.add_box({ b.lo.x + p_margin, b.lo.y + p_margin, b.lo.z + p_margin }, { b.hi.x - p_margin, b.hi.y - p_margin, b.hi.z - p_margin });
    }
}

// Hidden boxes over random box scenes, every ray from an eye within p_margin checked: (hidden, unsound).
std::pair<int, int> box_soundness(uint64_t p_seed, double p_margin) {
    Lcg rng{ p_seed };
    OcclusionMap map;
    std::vector<Box> boxes;
    int hidden = 0;
    int unsound = 0;
    for (int scene = 0; scene < 30; scene++) {
        random_box_scene(rng, map, boxes, p_margin);
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
                Vec3 e;
                do {
                    e = { rng.range(-p_margin, p_margin), rng.range(-p_margin, p_margin), rng.range(-p_margin, p_margin) };
                } while (e.x * e.x + e.y * e.y + e.z * e.z > p_margin * p_margin);
                unsound += segment_blocked(e, point_in(lo, hi, rng), boxes) ? 0 : 1;
            }
        }
    }
    return { hidden, unsound };
}

} // namespace

// A box in front of the eye hides what is behind it and nothing else; a box holding the eye writes nothing.
TEST_CASE("OcclusionMap: a box hides the box behind it") {
    OcclusionMap map;
    CHECK(map.add_box({ 40, -20, -20 }, { 60, 20, 20 }));
    CHECK(map.box_hidden({ 100, -2, -2 }, { 104, 2, 2 }));
    CHECK_FALSE(map.box_hidden({ 20, -2, -2 }, { 24, 2, 2 }));
    CHECK_FALSE(map.box_hidden({ 100, 60, -2 }, { 104, 64, 2 }));
    CHECK_FALSE(map.box_hidden({ -104, -2, -2 }, { -100, 2, 2 }));
    CHECK(map.depth_at({ 1, 0, 0 }) == doctest::Approx(std::sqrt(60.0 * 60.0 + 2 * 20.0 * 20.0)).epsilon(1e-6));
    CHECK(std::isinf(map.depth_at({ 0, 1, 0 })));
    CHECK_FALSE(map.add_box({ -1, -1, -1 }, { 1, 1, 1 }));
    // A thin trunk-like box beside the eye, across faces, still writes.
    CHECK(map.add_box({ 3, -1, -50 }, { 4, 1, 50 }));
    CHECK(map.get_stats().boxes == 3);
    CHECK(map.get_stats().boxes_written == 2);
}

// Soundness: every ray from the eye to a box called hidden meets an occluder box first.
TEST_CASE("OcclusionMap: boxes: hidden boxes are hidden from every ray") {
    const std::pair<int, int> r = box_soundness(0xB0C5u, 0.0);
    CAPTURE(r.first);
    CHECK(r.first > 500);
    CHECK(r.second == 0);
}

// The margin: boxes shrunk by R in the map, a box hidden from the eye is hidden from any eye within R of it.
TEST_CASE("OcclusionMap: boxes: shrunk boxes hide from every eye within the margin") {
    const std::pair<int, int> r = box_soundness(0xB0C6u, 3.0);
    CAPTURE(r.first);
    CHECK(r.first > 300);
    CHECK(r.second == 0);
}
