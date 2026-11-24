#include "weather_grid.hpp"

#include "exact_noise.hpp"

#include "ov/math.hpp"
#include "ov/on_error.hpp"

#include <cmath>

namespace {

constexpr double QUARTER_TURN = 1.5707963267948966; // pi / 2: a face's angle

double dot_axis(const double p_d[3], const int p_axis[3]) {
    return p_d[0] * p_axis[0] + p_d[1] * p_axis[1] + p_d[2] * p_axis[2];
}

// The face of a direction: its largest |component|, x before y before z on a tie (GLSL weather_uvw alike).
int face_of(const double p_d[3]) {
    const double ax = std::fabs(p_d[0]);
    const double ay = std::fabs(p_d[1]);
    const double az = std::fabs(p_d[2]);
    if (ax >= ay && ax >= az) {
        return p_d[0] >= 0.0 ? 0 : 1;
    }
    if (ay >= az) {
        return p_d[1] >= 0.0 ? 2 : 3;
    }
    return p_d[2] >= 0.0 ? 4 : 5;
}

ov::str glsl_axis(const int p_a[3]) {
    return ov::format("vec3(%d.0, %d.0, %d.0)", p_a[0], p_a[1], p_a[2]);
}

constexpr const char GLSL_LOOKUP[] = R"glsl(
// The weather grid's padded texel coordinates of a direction (WeatherGrid::glsl_lookup): x, y over [0, 1], z the face.
vec3 weather_uvw(vec3 dir, float n) {
    vec3 a = abs(dir);
    int face = a.x >= a.y && a.x >= a.z ? (dir.x >= 0.0 ? 0 : 1) : (a.y >= a.z ? (dir.y >= 0.0 ? 2 : 3) : (dir.z >= 0.0 ? 4 : 5));
    vec3 m;
    vec3 u;
    vec3 v;
$FACES
    float s = dot(dir, m);
    vec2 ang = vec2(atan(dot(dir, u) / s), atan(dot(dir, v) / s));
    vec2 cell = (ang / $QUARTER + 0.5) * n;
    return vec3((cell + float($HALO)) / (n + float(2 * $HALO)), float(face));
}
)glsl";

} // namespace

// U x V = MAJOR on every face.
const WeatherGrid::Face WeatherGrid::FACE[FACES] = {
    { { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 } },
    { { -1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } },
    { { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, -1 } },
    { { 0, -1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },
    { { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } },
    { { 0, 0, -1 }, { -1, 0, 0 }, { 0, 1, 0 } },
};

WeatherGrid::WeatherGrid(int p_n) {
    if (p_n < MIN_N || p_n > MAX_N) {
        OV_ERR_PRINT(ov::format("WeatherGrid: %d cells a side is outside [%d, %d].", p_n, MIN_N, MAX_N));
    }
    n = ov::clamp(p_n, MIN_N, MAX_N);
    make_halo();
}

WeatherGrid::Point WeatherGrid::locate(const double p_dir[3]) const {
    Point p;
    p.face = face_of(p_dir);
    const Face &f = FACE[p.face];
    const double s = dot_axis(p_dir, f.major);
    p.x = (std::atan(dot_axis(p_dir, f.u) / s) / QUARTER_TURN + 0.5) * n;
    p.y = (std::atan(dot_axis(p_dir, f.v) / s) / QUARTER_TURN + 0.5) * n;
    return p;
}

void WeatherGrid::direction(int p_face, double p_x, double p_y, double r_dir[3]) const {
    const Face &f = FACE[p_face];
    const double tu = std::tan((p_x / n - 0.5) * QUARTER_TURN);
    const double tv = std::tan((p_y / n - 0.5) * QUARTER_TURN);
    double len2 = 0.0;
    for (int k = 0; k < 3; k++) {
        r_dir[k] = f.major[k] + tu * f.u[k] + tv * f.v[k];
        len2 += r_dir[k] * r_dir[k];
    }
    const double inv = 1.0 / std::sqrt(len2);
    for (int k = 0; k < 3; k++) {
        r_dir[k] *= inv;
    }
}

void WeatherGrid::make_halo() {
    halo.clear();
    std::vector<HaloRead> corners;
    for (int face = 0; face < FACES; face++) {
        for (int j = -HALO; j < n + HALO; j++) {
            for (int i = -HALO; i < n + HALO; i++) {
                const bool out_i = i < 0 || i >= n;
                const bool out_j = j < 0 || j >= n;
                if (!out_i && !out_j) {
                    continue;
                }
                HaloRead h;
                h.dst = uint32_t(index(face, i, j));
                if (out_i && out_j) {
                    // A cube corner, where three faces meet: no face holds the cell's centre. Linear in the face's own
                    // chart from the two edge halo cells beside it and the interior corner cell (after them: second
                    // order like the edges; the nearest cells of a third face made the error first order).
                    const int ii = ov::clamp(i, 0, n - 1);
                    const int jj = ov::clamp(j, 0, n - 1);
                    h.src[0] = uint32_t(index(face, i, jj));
                    h.src[1] = uint32_t(index(face, ii, j));
                    h.src[2] = uint32_t(index(face, ii, jj));
                    h.src[3] = h.src[2];
                    h.w[0] = 1.0f;
                    h.w[1] = 1.0f;
                    h.w[2] = -1.0f;
                    h.w[3] = 0.0f;
                    corners.push_back(h);
                    continue;
                }
                double d[3];
                cell_direction(face, i, j, d);
                const Point p = locate(d);
                // The neighbour's interior centres round the point, clamped inside it.
                const double px = p.x - 0.5;
                const double py = p.y - 0.5;
                const int i0 = ov::clamp(int(std::floor(px)), 0, n - 2);
                const int j0 = ov::clamp(int(std::floor(py)), 0, n - 2);
                const double fx = ov::clamp(px - i0, 0.0, 1.0);
                const double fy = ov::clamp(py - j0, 0.0, 1.0);
                h.src[0] = uint32_t(index(p.face, i0, j0));
                h.src[1] = uint32_t(index(p.face, i0 + 1, j0));
                h.src[2] = uint32_t(index(p.face, i0, j0 + 1));
                h.src[3] = uint32_t(index(p.face, i0 + 1, j0 + 1));
                h.w[0] = float((1.0 - fx) * (1.0 - fy));
                h.w[1] = float(fx * (1.0 - fy));
                h.w[2] = float((1.0 - fx) * fy);
                h.w[3] = float(fx * fy);
                halo.push_back(h);
            }
        }
    }
    halo.insert(halo.end(), corners.begin(), corners.end());
}

void WeatherGrid::fill_halo(float *p_field, int p_comps) const {
    for (const HaloRead &h : halo) {
        for (int c = 0; c < p_comps; c++) {
            float v = 0.0f;
            for (int k = 0; k < 4; k++) {
                v += h.w[k] * p_field[size_t(h.src[k]) * size_t(p_comps) + size_t(c)];
            }
            p_field[size_t(h.dst) * size_t(p_comps) + size_t(c)] = v;
        }
    }
}

void WeatherGrid::sample(const float *p_field, int p_comps, const double p_dir[3], float *r_value) const {
    const Point p = locate(p_dir);
    // The centres round the point: x in [0, N] puts i0 in [-1, N - 1], both columns inside the padded face.
    const double px = ov::clamp(p.x - 0.5, -0.5, n - 0.5);
    const double py = ov::clamp(p.y - 0.5, -0.5, n - 0.5);
    const int i0 = int(std::floor(px));
    const int j0 = int(std::floor(py));
    const float fx = float(px - i0);
    const float fy = float(py - j0);
    const size_t s00 = index(p.face, i0, j0) * size_t(p_comps);
    const size_t s10 = index(p.face, i0 + 1, j0) * size_t(p_comps);
    const size_t s01 = index(p.face, i0, j0 + 1) * size_t(p_comps);
    const size_t s11 = index(p.face, i0 + 1, j0 + 1) * size_t(p_comps);
    for (int c = 0; c < p_comps; c++) {
        const float a = p_field[s00 + size_t(c)] + fx * (p_field[s10 + size_t(c)] - p_field[s00 + size_t(c)]);
        const float b = p_field[s01 + size_t(c)] + fx * (p_field[s11 + size_t(c)] - p_field[s01 + size_t(c)]);
        r_value[c] = a + fy * (b - a);
    }
}

VoxelPos WeatherGrid::cell_voxel(const BaseSurface &p_base, const double p_dir[3], double p_alt) {
    OV_ERR_FAIL_COND_V_MSG(p_base.kind() != BaseSurface::Kind::SPHERE, p_base.centre(), "WeatherGrid: cell_voxel needs a sphere.");
    const double r = p_base.get_radius() + p_alt;
    return p_base.centre() + VoxelPos(std::llround(p_dir[0] * r), std::llround(p_dir[1] * r), std::llround(p_dir[2] * r));
}

ov::str WeatherGrid::glsl_lookup() {
    // One branch per face, the last a plain else (every path sets m, u and v).
    ov::str faces;
    for (int f = 0; f < FACES; f++) {
        const ov::str head = f == 0 ? ov::format("    if (face == %d) {", f) : (f < FACES - 1 ? ov::format(" else if (face == %d) {", f) : ov::str(" else {"));
        faces += ov::format("%s\n        m = %s;\n        u = %s;\n        v = %s;\n    }", head, glsl_axis(FACE[f].major), glsl_axis(FACE[f].u), glsl_axis(FACE[f].v));
    }
    return ov::replace_all(ov::str(GLSL_LOOKUP),
            { { "$FACES", faces }, { "$QUARTER", ExactNoise::glsl_float(float(QUARTER_TURN)) }, { "$HALO", ov::itos(HALO) } });
}
