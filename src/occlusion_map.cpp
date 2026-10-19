#include "occlusion_map.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr double PI = 3.14159265358979323846;
// Vertices of the spherical polygon around a cap.
constexpr int CAP_VERTICES = 8;
// A cap this wide covers every face.
constexpr double WIDE_CAP = 1.5;

double length(const OcclusionMap::Vec3 &p_v) {
    return std::sqrt(p_v.x * p_v.x + p_v.y * p_v.y + p_v.z * p_v.z);
}

OcclusionMap::Vec3 scaled(const OcclusionMap::Vec3 &p_v, double p_s) {
    return { p_v.x * p_s, p_v.y * p_s, p_v.z * p_s };
}

// The u (or v) coordinate of a face pixel's centre, edge or corner: p_i in [0, RES].
double face_coord(double p_i) {
    return p_i / OcclusionMap::RES * 2.0 - 1.0;
}

int pixel_floor(double p_coord) {
    return int(std::floor((p_coord + 1.0) * 0.5 * OcclusionMap::RES));
}

} // namespace

OcclusionMap::OcclusionMap() {
    pixels.resize(RES * RES);
    min_half = PI;
    for (int j = 0; j < RES; j++) {
        for (int i = 0; i < RES; i++) {
            const double u = face_coord(i + 0.5);
            const double v = face_coord(j + 0.5);
            const double n = std::sqrt(1.0 + u * u + v * v);
            Pixel &p = pixels[j * RES + i];
            p.axis = 1.0 / n;
            p.u = u / n;
            p.v = v / n;
            // The corners are the farthest points of the pixel from its centre direction.
            double half = 0.0;
            for (int c = 0; c < 4; c++) {
                const double cu = face_coord(i + (c & 1));
                const double cv = face_coord(j + (c >> 1));
                const double cn = std::sqrt(1.0 + cu * cu + cv * cv);
                const double d = (1.0 + u * cu + v * cv) / (n * cn);
                half = std::max(half, std::acos(std::min(1.0, d)));
            }
            p.cos_half = std::cos(half);
            p.sin_half = std::sin(half);
            min_half = std::min(min_half, half);
            max_half = std::max(max_half, half);
        }
    }
    clear();
}

void OcclusionMap::clear() {
    depth.assign(6 * RES * RES, INFINITY);
    tile_max.assign(6 * TILES * TILES, INFINITY);
    stats = Stats();
}

OcclusionMap::Vec3 OcclusionMap::to_face(const Vec3 &p_dir, int p_face) {
    const int a = p_face >> 1;
    const double s = (p_face & 1) ? -1.0 : 1.0;
    return { s * p_dir[a], p_dir[(a + 1) % 3], p_dir[(a + 2) % 3] };
}

bool OcclusionMap::cap_rect(const Vec3 &p_axis, double p_radius, int p_face, int &r_i0, int &r_i1, int &r_j0, int &r_j1) {
    if (p_radius >= WIDE_CAP) {
        r_i0 = 0;
        r_j0 = 0;
        r_i1 = RES - 1;
        r_j1 = RES - 1;
        return true;
    }
    // The regular polygon whose inradius is p_radius: tan(inradius) = tan(circumradius) cos(pi / n).
    const double circum = std::atan(std::tan(p_radius) / std::cos(PI / CAP_VERTICES));
    const Vec3 axis = to_face(p_axis, p_face);
    // Two unit vectors orthogonal to the axis.
    const Vec3 helper = std::abs(axis.x) < 0.9 ? Vec3{ 1.0, 0.0, 0.0 } : Vec3{ 0.0, 1.0, 0.0 };
    Vec3 e1{ helper.y * axis.z - helper.z * axis.y, helper.z * axis.x - helper.x * axis.z, helper.x * axis.y - helper.y * axis.x };
    e1 = scaled(e1, 1.0 / length(e1));
    const Vec3 e2{ axis.y * e1.z - axis.z * e1.y, axis.z * e1.x - axis.x * e1.z, axis.x * e1.y - axis.y * e1.x };
    const double c = std::cos(circum);
    const double s = std::sin(circum);
    double u0 = INFINITY, u1 = -INFINITY, v0 = INFINITY, v1 = -INFINITY;
    int in_front = 0;
    for (int k = 0; k < CAP_VERTICES; k++) {
        const double t = 2.0 * PI * k / CAP_VERTICES;
        const double ct = std::cos(t) * s;
        const double st = std::sin(t) * s;
        const Vec3 d{ axis.x * c + e1.x * ct + e2.x * st, axis.y * c + e1.y * ct + e2.y * st, axis.z * c + e1.z * ct + e2.z * st };
        if (d.x <= 1e-9) {
            continue;
        }
        in_front++;
        u0 = std::min(u0, d.y / d.x);
        u1 = std::max(u1, d.y / d.x);
        v0 = std::min(v0, d.z / d.x);
        v1 = std::max(v1, d.z / d.x);
    }
    if (in_front == 0) {
        return false; // a convex polygon behind the face's plane
    }
    if (in_front < CAP_VERTICES) {
        u0 = -1.0; // it crosses the plane: the whole face
        u1 = 1.0;
        v0 = -1.0;
        v1 = 1.0;
    }
    if (u0 > 1.0 || u1 < -1.0 || v0 > 1.0 || v1 < -1.0) {
        return false;
    }
    r_i0 = std::max(pixel_floor(u0) - 1, 0);
    r_i1 = std::min(pixel_floor(u1) + 1, RES - 1);
    r_j0 = std::max(pixel_floor(v0) - 1, 0);
    r_j1 = std::min(pixel_floor(v1) + 1, RES - 1);
    return true;
}

void OcclusionMap::add_ball(const Vec3 &p_centre, double p_radius) {
    stats.balls++;
    const double d = length(p_centre);
    if (!(p_radius > 0.0) || d <= p_radius) {
        return;
    }
    const double alpha = std::asin(p_radius / d);
    if (alpha <= min_half) {
        return; // no pixel cone fits in it
    }
    const Vec3 axis = scaled(p_centre, 1.0 / d);
    const double cos_a = std::cos(alpha);
    const double sin_a = std::sin(alpha);
    const float ball_depth = float(d);
    bool written = false;
    for (int face = 0; face < 6; face++) {
        int i0, i1, j0, j1;
        if (!cap_rect(axis, alpha, face, i0, i1, j0, j1)) {
            continue;
        }
        const Vec3 fa = to_face(axis, face);
        float *face_depth = depth.data() + face * RES * RES;
        float *face_tiles = tile_max.data() + face * TILES * TILES;
        for (int tj = j0 / TILE; tj <= j1 / TILE; tj++) {
            for (int ti = i0 / TILE; ti <= i1 / TILE; ti++) {
                float &tile = face_tiles[tj * TILES + ti];
                if (tile <= ball_depth) {
                    continue; // every pixel of the tile is already blocked nearer
                }
                bool tile_written = false;
                for (int j = std::max(j0, tj * TILE); j <= std::min(j1, tj * TILE + TILE - 1); j++) {
                    for (int i = std::max(i0, ti * TILE); i <= std::min(i1, ti * TILE + TILE - 1); i++) {
                        const Pixel &p = pixels[j * RES + i];
                        // The pixel's cone inside the ball's: angle(axis, p) + half <= alpha, i.e. cos(angle) >= cos(alpha - half).
                        if (p.sin_half >= sin_a) {
                            continue;
                        }
                        const double cos_angle = fa.x * p.axis + fa.y * p.u + fa.z * p.v;
                        if (cos_angle >= cos_a * p.cos_half + sin_a * p.sin_half && ball_depth < face_depth[j * RES + i]) {
                            face_depth[j * RES + i] = ball_depth;
                            stats.pixels_written++;
                            tile_written = true;
                        }
                    }
                }
                if (tile_written) {
                    written = true;
                    float farthest = 0.0f;
                    for (int j = tj * TILE; j < tj * TILE + TILE; j++) {
                        for (int i = ti * TILE; i < ti * TILE + TILE; i++) {
                            farthest = std::max(farthest, face_depth[j * RES + i]);
                        }
                    }
                    tile = farthest;
                }
            }
        }
    }
    stats.balls_written += written ? 1 : 0;
}

bool OcclusionMap::box_hidden(const Vec3 &p_lo, const Vec3 &p_hi) const {
    const Vec3 centre{ (p_lo.x + p_hi.x) * 0.5, (p_lo.y + p_hi.y) * 0.5, (p_lo.z + p_hi.z) * 0.5 };
    const double radius = length(Vec3{ p_hi.x - p_lo.x, p_hi.y - p_lo.y, p_hi.z - p_lo.z }) * 0.5;
    const double d = length(centre);
    if (d <= radius * (1.0 + 1e-9)) {
        return false;
    }
    // Every point of the box is at least this far.
    const Vec3 gap{ std::max({ p_lo.x, -p_hi.x, 0.0 }), std::max({ p_lo.y, -p_hi.y, 0.0 }), std::max({ p_lo.z, -p_hi.z, 0.0 }) };
    const float nearest = float(length(gap));
    const double gamma = std::asin(radius / d);
    const Vec3 axis = scaled(centre, 1.0 / d);
    const double cos_g = std::cos(gamma);
    const double sin_g = std::sin(gamma);
    bool any = false;
    for (int face = 0; face < 6; face++) {
        int i0, i1, j0, j1;
        if (!cap_rect(axis, gamma + max_half, face, i0, i1, j0, j1)) {
            continue;
        }
        const Vec3 fa = to_face(axis, face);
        const float *face_depth = depth.data() + face * RES * RES;
        const float *face_tiles = tile_max.data() + face * TILES * TILES;
        for (int tj = j0 / TILE; tj <= j1 / TILE; tj++) {
            for (int ti = i0 / TILE; ti <= i1 / TILE; ti++) {
                if (face_tiles[tj * TILES + ti] < nearest) {
                    any = true; // every pixel of the tile is blocked before the box
                    continue;
                }
                for (int j = std::max(j0, tj * TILE); j <= std::min(j1, tj * TILE + TILE - 1); j++) {
                    for (int i = std::max(i0, ti * TILE); i <= std::min(i1, ti * TILE + TILE - 1); i++) {
                        const Pixel &p = pixels[j * RES + i];
                        // The pixel's cone meets the box's: angle(axis, p) - half <= gamma, i.e. cos(angle) >= cos(gamma +
                        // half) (gamma < pi / 2 and half is under a degree, so gamma + half < pi).
                        const double cos_angle = fa.x * p.axis + fa.y * p.u + fa.z * p.v;
                        if (cos_angle < cos_g * p.cos_half - sin_g * p.sin_half) {
                            continue;
                        }
                        any = true;
                        if (!(face_depth[j * RES + i] < nearest)) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return any;
}

double OcclusionMap::depth_at(const Vec3 &p_dir) const {
    int a = 0;
    for (int k = 1; k < 3; k++) {
        a = std::abs(p_dir[k]) > std::abs(p_dir[a]) ? k : a;
    }
    const int face = 2 * a + (p_dir[a] < 0.0 ? 1 : 0);
    const Vec3 f = to_face(p_dir, face);
    const int i = std::clamp(pixel_floor(f.y / f.x), 0, RES - 1);
    const int j = std::clamp(pixel_floor(f.z / f.x), 0, RES - 1);
    return depth[face * RES * RES + j * RES + i];
}
