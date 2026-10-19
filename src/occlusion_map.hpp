#pragma once

#include <cstdint>
#include <vector>

// A conservative from-point occlusion map: a cube map around an eye holding, per pixel, a distance by which
// every ray of that pixel has entered a solid ball. The discoverer fills it with the balls its SDF corners prove
// solid (OcclusionBound) and asks it whether a chunk's box is hidden: every pixel a ray to the box can pass
// through is blocked closer than the box. Positions are relative to the eye, in any one unit.
//
// Both tests are conservative: a ball only writes the pixels whose whole cone lies inside it, and a box looks at
// every pixel whose cone meets its bounding sphere's. The pixel range on a face comes from a regular spherical
// polygon around the cap (its inradius is the cap's radius): its edges are great circles, straight lines on a
// cube face, so the bounding box of its vertices holds the cap. Pure (no Godot), doctested.
class OcclusionMap {
public:
    struct Vec3 {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double operator[](int p_axis) const { return p_axis == 0 ? x : (p_axis == 1 ? y : z); }
    };

    // Pixels per face side: ~0.7 degrees at the face centre.
    static constexpr int RES = 128;
    // Tiles of TILE x TILE pixels keep their farthest depth: a ball behind a whole tile skips it, and so does a
    // box behind it. Fed front to back, most balls only touch tiles.
    static constexpr int TILE = 8;
    static constexpr int TILES = RES / TILE;

    OcclusionMap();

    // Every pixel unblocked.
    void clear();
    // A solid ball, centre relative to the eye. Skipped when it holds the eye or covers no whole pixel.
    void add_ball(const Vec3 &p_centre, double p_radius);
    // A solid box [p_lo, p_hi], relative to the eye. A pixel is written when its 4 corner rays hit the box: the
    // directions that hit a convex box form a convex cone and a pixel's cone is the hull of its corner rays, so
    // every ray of the pixel hits it, before the box's farthest corner (the depth written). Skipped when it holds
    // the eye. Returns whether a pixel was written.
    bool add_box(const Vec3 &p_lo, const Vec3 &p_hi);
    // Every ray from the eye to the box [p_lo, p_hi] (relative to the eye) is blocked before reaching it. An eye
    // inside or touching the box's bounding sphere: false.
    bool box_hidden(const Vec3 &p_lo, const Vec3 &p_hi) const;

    // The blocking distance of the pixel holding direction p_dir (INFINITY: unblocked). Tests.
    double depth_at(const Vec3 &p_dir) const;

    struct Stats {
        int64_t balls = 0; // add_ball calls
        int64_t balls_written = 0; // balls that wrote at least one pixel
        int64_t boxes = 0; // add_box calls
        int64_t boxes_written = 0; // boxes that wrote at least one pixel
        int64_t pixels_written = 0;
    };
    const Stats &get_stats() const { return stats; }

private:
    // Per pixel of a face (the same on every face): the unit direction in the face's frame (axis, u, v) and the
    // cone's half-angle (centre to farthest corner) as cos / sin.
    struct Pixel {
        double axis, u, v;
        double cos_half, sin_half;
    };
    std::vector<Pixel> pixels; // RES * RES
    double min_half = 0.0; // the smallest and largest pixel half-angles
    double max_half = 0.0;
    std::vector<float> depth; // 6 * RES * RES, INFINITY = unblocked
    std::vector<float> tile_max; // 6 * TILES * TILES: the farthest depth of the tile's pixels
    Stats stats;

    // The pixel rectangle [r_i0, r_i1] x [r_j0, r_j1] of face p_face holding every direction within angle
    // p_radius of p_axis (unit); false if none.
    static bool cap_rect(const Vec3 &p_axis, double p_radius, int p_face, int &r_i0, int &r_i1, int &r_j0, int &r_j1);
    // p_dir in the frame of face p_face (axis, u, v), the face's axis component positive inside it.
    static Vec3 to_face(const Vec3 &p_dir, int p_face);
    // The bounding sphere of [p_lo, p_hi] seen from the eye: its unit axis and angular radius (WIDE_CAP or more
    // when it holds the eye).
    static void box_cap(const Vec3 &p_lo, const Vec3 &p_hi, Vec3 &r_axis, double &r_radius);
    // Recomputes a tile's farthest depth.
    void update_tile(int p_face, int p_ti, int p_tj);
};
