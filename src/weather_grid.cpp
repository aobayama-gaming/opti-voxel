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
// The weather atlas's coordinates of the column under render-space p (CubeSphereGrid::glsl_lookup): its direction's face
// and angles, the face's block of rows of the atlas, the coordinates within its padded cells.
vec2 weather_uv(vec3 p) {
    vec3 dir = base_up(p);
    vec3 a = abs(dir);
    int face = a.x >= a.y && a.x >= a.z ? (dir.x >= 0.0 ? 0 : 1) : (a.y >= a.z ? (dir.y >= 0.0 ? 2 : 3) : (dir.z >= 0.0 ? 4 : 5));
    vec3 m;
    vec3 u;
    vec3 v;
$FACES
    float s = dot(dir, m);
    vec2 ang = vec2(atan(dot(dir, u) / s), atan(dot(dir, v) / s));
    vec2 cell = clamp((ang / $QUARTER + 0.5) * $N, 0.0, $N) + float($HALO);
    return vec2(cell.x / $STRIDE, (float(face) * $STRIDE + cell.y) / ($STRIDE * float($FACE_COUNT)));
}
)glsl";

} // namespace

// U x V = MAJOR on every face.
const CubeSphereGrid::Face CubeSphereGrid::FACE[FACES] = {
    { { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 } },
    { { -1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } },
    { { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, -1 } },
    { { 0, -1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },
    { { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } },
    { { 0, 0, -1 }, { -1, 0, 0 }, { 0, 1, 0 } },
};

std::unique_ptr<WeatherGrid> WeatherGrid::for_base(const BaseSurface &p_base, double p_voxel_m, double p_cell_m, ov::str *r_error) {
    auto refuse = [&](const ov::str &p_why) {
        if (r_error) {
            *r_error = p_why;
        }
        return std::unique_ptr<WeatherGrid>();
    };
    if (p_base.kind() != BaseSurface::Kind::SPHERE) {
        return refuse("the weather's grid covers a sphere only (a torus's is to come; a flat world has none).");
    }
    const double radius_m = p_base.get_radius() * p_voxel_m;
    const double n = CubeSphereGrid::side_for(radius_m, cell_width(radius_m, p_cell_m));
    if (!(n >= MIN_N && n <= MAX_N)) {
        return refuse(ov::format("cells of %g m make %g a cube face's side over a %g m sphere, outside [%d, %d].", p_cell_m, n, radius_m, MIN_N, MAX_N));
    }
    return std::make_unique<CubeSphereGrid>(int(n), radius_m);
}

double WeatherGrid::cell_width(double p_radius_m, double p_cell_m) {
    return p_cell_m > 0.0 ? p_cell_m : QUARTER_TURN * p_radius_m / DEFAULT_SIDE;
}

WeatherGrid::WeatherGrid(int p_faces, int p_nx, int p_ny, double p_cell_m) :
        face_count(p_faces), nx(p_nx), ny(p_ny), cell_width_m(p_cell_m) {}

void WeatherGrid::make_halo() {
    halo.clear();
    std::vector<HaloRead> corners;
    for (int face = 0; face < face_count; face++) {
        for (int j = -HALO; j < ny + HALO; j++) {
            for (int i = -HALO; i < nx + HALO; i++) {
                const bool out_i = i < 0 || i >= nx;
                const bool out_j = j < 0 || j >= ny;
                if (!out_i && !out_j) {
                    continue;
                }
                HaloRead h;
                h.dst = uint32_t(index(face, i, j));
                if (out_i && out_j) {
                    // A corner, where no face may hold the cell's centre (a cube's, where three faces meet). Linear in
                    // the face's own chart from the two edge halo cells beside it and the interior corner cell (after
                    // them: second order like the edges; the nearest cells of a third face made the error first order).
                    const int ii = ov::clamp(i, 0, nx - 1);
                    const int jj = ov::clamp(j, 0, ny - 1);
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
                double pos[3], up[3];
                cell_surface(face, i, j, pos, up);
                const Point p = locate(pos);
                // The interior centres round the point, clamped inside its face.
                const double px = p.x - 0.5;
                const double py = p.y - 0.5;
                const int i0 = ov::clamp(int(std::floor(px)), 0, nx - 2);
                const int j0 = ov::clamp(int(std::floor(py)), 0, ny - 2);
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

void WeatherGrid::sample(const float *p_field, int p_stride, int p_comps, const double p_pos[3], float *r_value) const {
    const Point p = locate(p_pos);
    // The centres round the point: x in [0, width] puts i0 in [-1, width - 1], both columns inside the padded face.
    const double px = ov::clamp(p.x - 0.5, -0.5, nx - 0.5);
    const double py = ov::clamp(p.y - 0.5, -0.5, ny - 0.5);
    const int i0 = int(std::floor(px));
    const int j0 = int(std::floor(py));
    const float fx = float(px - i0);
    const float fy = float(py - j0);
    const size_t s00 = index(p.face, i0, j0) * size_t(p_stride);
    const size_t s10 = index(p.face, i0 + 1, j0) * size_t(p_stride);
    const size_t s01 = index(p.face, i0, j0 + 1) * size_t(p_stride);
    const size_t s11 = index(p.face, i0 + 1, j0 + 1) * size_t(p_stride);
    for (int c = 0; c < p_comps; c++) {
        const float a = p_field[s00 + size_t(c)] + fx * (p_field[s10 + size_t(c)] - p_field[s00 + size_t(c)]);
        const float b = p_field[s01 + size_t(c)] + fx * (p_field[s11 + size_t(c)] - p_field[s01 + size_t(c)]);
        r_value[c] = a + fy * (b - a);
    }
}

namespace {

int clamp_n(int p_n) {
    return ov::clamp(p_n, WeatherGrid::MIN_N, WeatherGrid::MAX_N);
}

} // namespace

CubeSphereGrid::CubeSphereGrid(int p_n, double p_radius_m) :
        WeatherGrid(FACES, clamp_n(p_n), clamp_n(p_n), QUARTER_TURN * p_radius_m / clamp_n(p_n)), radius(p_radius_m) {
    if (p_n < MIN_N || p_n > MAX_N) {
        OV_ERR_PRINT(ov::format("CubeSphereGrid: %d cells a side is outside [%d, %d].", p_n, MIN_N, MAX_N));
    }
    make_halo();
}

double CubeSphereGrid::side_for(double p_radius_m, double p_cell_m) {
    return std::round(QUARTER_TURN * p_radius_m / p_cell_m);
}

WeatherGrid::Point CubeSphereGrid::locate(const double p_pos[3]) const {
    Point p;
    p.face = face_of(p_pos);
    const Face &f = FACE[p.face];
    const double s = dot_axis(p_pos, f.major);
    const int n = size();
    p.x = (std::atan(dot_axis(p_pos, f.u) / s) / QUARTER_TURN + 0.5) * n;
    p.y = (std::atan(dot_axis(p_pos, f.v) / s) / QUARTER_TURN + 0.5) * n;
    return p;
}

void CubeSphereGrid::direction(int p_face, double p_x, double p_y, double r_dir[3]) const {
    const Face &f = FACE[p_face];
    const int n = size();
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

void CubeSphereGrid::surface(int p_face, double p_x, double p_y, double r_pos[3], double r_up[3]) const {
    direction(p_face, p_x, p_y, r_up);
    for (int k = 0; k < 3; k++) {
        r_pos[k] = r_up[k] * radius;
    }
}

ov::str CubeSphereGrid::glsl_lookup() const {
    // One branch per face, the last a plain else (every path sets m, u and v).
    ov::str faces;
    for (int f = 0; f < FACES; f++) {
        const ov::str head = f == 0 ? ov::format("    if (face == %d) {", f) : (f < FACES - 1 ? ov::format(" else if (face == %d) {", f) : ov::str(" else {"));
        faces += ov::format("%s\n        m = %s;\n        u = %s;\n        v = %s;\n    }", head, glsl_axis(FACE[f].major), glsl_axis(FACE[f].u), glsl_axis(FACE[f].v));
    }
    return ov::replace_all(ov::str(GLSL_LOOKUP),
            { { "$FACES", faces }, { "$QUARTER", ExactNoise::glsl_float(float(QUARTER_TURN)) }, { "$HALO", ov::itos(HALO) },
                    { "$N", ov::format("%d.0", size()) }, { "$STRIDE", ov::format("%d.0", stride_x()) }, { "$FACE_COUNT", ov::itos(FACES) } });
}
