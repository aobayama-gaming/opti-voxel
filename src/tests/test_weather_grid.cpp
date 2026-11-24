#include "test_common.hpp"

#include "base_surface.hpp"
#include "weather_grid.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Deterministic unit directions spread over the sphere (an LCG, not std::rand: the same set on every platform).
struct Directions {
    uint64_t state = 0x9E3779B97F4A7C15ull;
    double next01() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return double(state >> 11) * (1.0 / 9007199254740992.0);
    }
    void next(double r_d[3]) {
        const double z = 2.0 * next01() - 1.0;
        const double a = 6.283185307179586 * next01();
        const double s = std::sqrt(1.0 - z * z);
        r_d[0] = s * std::cos(a);
        r_d[1] = z;
        r_d[2] = s * std::sin(a);
    }
};

double distance(const double p_a[3], const double p_b[3]) {
    return std::sqrt((p_a[0] - p_b[0]) * (p_a[0] - p_b[0]) + (p_a[1] - p_b[1]) * (p_a[1] - p_b[1]) + (p_a[2] - p_b[2]) * (p_a[2] - p_b[2]));
}

// A smooth field of two components over the sphere (no symmetry the cube's faces share).
void smooth(const double p_d[3], float r_v[2]) {
    r_v[0] = float(0.3 + p_d[0] + 2.0 * p_d[1] - 0.5 * p_d[2] + p_d[0] * p_d[2]);
    r_v[1] = float(std::sin(3.0 * p_d[0]) * std::cos(2.0 * p_d[1]) + p_d[2] * p_d[2]);
}

// The field at every interior cell centre, the halo filled.
std::vector<float> smooth_field(const CubeSphereGrid &p_grid) {
    std::vector<float> field(p_grid.padded_cells() * 2, 0.0f);
    for (int f = 0; f < CubeSphereGrid::FACES; f++) {
        for (int j = 0; j < p_grid.size(); j++) {
            for (int i = 0; i < p_grid.size(); i++) {
                double d[3];
                p_grid.cell_direction(f, i, j, d);
                smooth(d, &field[p_grid.index(f, i, j) * 2]);
            }
        }
    }
    p_grid.fill_halo(field.data(), 2);
    return field;
}

// The worst error of a smooth field read through the halo on a grid of p_n: random points, every face's edges (just
// inside each face, both sides of each seam) and the cube's 8 corners.
double worst_read_error(int p_n) {
    const CubeSphereGrid grid(p_n, 1.0);
    const std::vector<float> field = smooth_field(grid);
    double worst = 0.0;
    auto check_at = [&](const double p_d[3]) {
        float want[2];
        float got[2];
        smooth(p_d, want);
        grid.sample(field.data(), 2, p_d, got);
        worst = std::max(worst, std::max(std::fabs(double(got[0] - want[0])), std::fabs(double(got[1] - want[1]))));
    };
    Directions dirs;
    for (int k = 0; k < 4000; k++) {
        double d[3];
        dirs.next(d);
        check_at(d);
    }
    const double n = double(p_n);
    for (int f = 0; f < CubeSphereGrid::FACES; f++) {
        for (int s = 0; s <= 256; s++) {
            const double t = n * s / 256.0;
            const double edge[4][2] = { { 1e-7, t }, { n - 1e-7, t }, { t, 1e-7 }, { t, n - 1e-7 } };
            for (const auto &e : edge) {
                double d[3];
                grid.direction(f, e[0], e[1], d);
                check_at(d);
            }
        }
    }
    for (int c = 0; c < 8; c++) {
        const double k = 1.0 / std::sqrt(3.0);
        const double d[3] = { c & 1 ? k : -k, c & 2 ? k : -k, c & 4 ? k : -k };
        check_at(d);
    }
    return worst;
}

} // namespace

TEST_CASE("WeatherGrid: the padded layout and the faces' axes") {
    const CubeSphereGrid grid(16, 1.0);
    CHECK(grid.size() == 16);
    CHECK(grid.stride_x() == 16 + 2 * WeatherGrid::HALO);
    CHECK(grid.stride_y() == grid.stride_x());
    CHECK(grid.cells() == size_t(6 * 16 * 16));
    CHECK(grid.index(0, -WeatherGrid::HALO, -WeatherGrid::HALO) == 0);
    CHECK(grid.index(CubeSphereGrid::FACES - 1, 16 + WeatherGrid::HALO - 1, 16 + WeatherGrid::HALO - 1) == grid.padded_cells() - 1);
    CHECK(grid.index(2, 1, 0) == grid.index(2, 0, 0) + 1); // x fastest
    for (const CubeSphereGrid::Face &f : CubeSphereGrid::FACE) {
        // U x V = MAJOR.
        const int c[3] = { f.u[1] * f.v[2] - f.u[2] * f.v[1], f.u[2] * f.v[0] - f.u[0] * f.v[2], f.u[0] * f.v[1] - f.u[1] * f.v[0] };
        CHECK(c[0] == f.major[0]);
        CHECK(c[1] == f.major[1]);
        CHECK(c[2] == f.major[2]);
    }
}

TEST_CASE("WeatherGrid: a direction and its cell coordinates round-trip") {
    const CubeSphereGrid grid(32, 1.0);
    Directions dirs;
    for (int k = 0; k < 2000; k++) {
        double d[3];
        dirs.next(d);
        const WeatherGrid::Point p = grid.locate(d);
        CHECK(p.x >= 0.0);
        CHECK(p.x <= 32.0);
        CHECK(p.y >= 0.0);
        CHECK(p.y <= 32.0);
        double back[3];
        grid.direction(p.face, p.x, p.y, back);
        CHECK(distance(d, back) < 1e-12);
    }
    // Every cell centre locates to its own cell.
    for (int f = 0; f < CubeSphereGrid::FACES; f++) {
        for (int j = 0; j < 32; j++) {
            for (int i = 0; i < 32; i++) {
                double d[3];
                grid.cell_direction(f, i, j, d);
                const WeatherGrid::Point p = grid.locate(d);
                CHECK(p.face == f);
                CHECK(std::fabs(p.x - (i + 0.5)) < 1e-9);
                CHECK(std::fabs(p.y - (j + 0.5)) < 1e-9);
            }
        }
    }
}

TEST_CASE("WeatherGrid: equal angles keep the cells' areas within a factor 1.5") {
    const CubeSphereGrid grid(32, 1.0);
    double lo = 1e30;
    double hi = 0.0;
    for (int j = 0; j < 32; j++) {
        for (int i = 0; i < 32; i++) {
            double a[3], b[3], c[3];
            grid.direction(0, i, j, a);
            grid.direction(0, i + 1, j, b);
            grid.direction(0, i, j + 1, c);
            const double e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
            const double e2[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
            const double x[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
            const double area = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            lo = std::min(lo, area);
            hi = std::max(hi, area);
        }
    }
    CHECK(hi / lo < 1.5);
}

// The halo makes a bilinear read cross a seam like any cell: a smooth field read anywhere, on the faces' edges and at
// the cube's corners included, is the field within the bilinear error, second order (h^2 of the curvature, h = pi / 2 /
// N): doubling N divides it by ~4 (a corner read from a third face's nearest cells was first order: 0.015 at N = 32).
TEST_CASE("WeatherGrid: a smooth field reads smoothly across the seams and corners, to second order") {
    const double at32 = worst_read_error(32);
    const double at64 = worst_read_error(64);
    CHECK(at32 < 0.01);
    CHECK(at64 < 0.35 * at32);
}

// A sphere's grid from the planet: its cells about the width asked for, each surface point on the sphere under its up;
// another base has none.
TEST_CASE("WeatherGrid::for_base: a sphere's cube of about cell_m, its surface points on the sphere") {
    const double vs = 0.05;
    const BaseSurface base = BaseSurface::planet(2000.0, vs);
    CHECK_OR_RETURN(base.kind() == BaseSurface::Kind::SPHERE);
    ov::str why;
    const std::unique_ptr<WeatherGrid> grid = WeatherGrid::for_base(base, vs, 45.0, &why);
    CHECK_OR_RETURN(grid != nullptr);
    CHECK(grid->faces() == 6);
    CHECK(grid->width() == 70); // a quarter of 2 pi 2 km over 45 m
    CHECK(grid->cell_m() == doctest::Approx(45.0).epsilon(0.01));
    for (int f = 0; f < grid->faces(); f++) {
        for (int j = 0; j < grid->height(); j += 9) {
            for (int i = 0; i < grid->width(); i += 7) {
                double pos[3], up[3];
                grid->cell_surface(f, i, j, pos, up);
                const double r = std::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
                CHECK(r == doctest::Approx(2000.0));
                CHECK(distance(up, std::vector<double>{ pos[0] / r, pos[1] / r, pos[2] / r }.data()) < 1e-12);
                const WeatherGrid::Point p = grid->locate(pos);
                CHECK(p.face == f);
                CHECK(std::fabs(p.x - (i + 0.5)) < 1e-9);
            }
        }
    }
    CHECK(WeatherGrid::for_base(base, vs, 1.0, &why) == nullptr); // 3142 cells a side
    CHECK(ov::contains(why, "outside"));
    CHECK(WeatherGrid::for_base(BaseSurface(), vs, 45.0, &why) == nullptr);
    CHECK(ov::contains(why, "sphere only"));
}

TEST_CASE("WeatherGrid: the GLSL lookup is substituted from the faces' table") {
    const ov::str code = CubeSphereGrid(16, 1.0).glsl_lookup();
    CHECK(code.find('$') == ov::str::npos);
    CHECK(ov::contains(code, "vec2 weather_uv(vec3 p)"));
    CHECK(ov::contains(code, "* 16.0, 0.0, 16.0)"));
    for (int f = 0; f < CubeSphereGrid::FACES - 1; f++) {
        CHECK(ov::contains(code, ov::format("if (face == %d)", f)));
    }
    CHECK(ov::contains(code, "} else {"));
    CHECK(ov::contains(code, "u = vec3(0.0, 0.0, -1.0);"));
}
