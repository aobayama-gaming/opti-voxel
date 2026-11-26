#include "test_common.hpp"

#include "ov/clock.hpp"
#include "weather_fluid.hpp"

#include <cmath>
#include <cstring>
#include <vector>

// The weather's 3D fluid (WeatherFluid, docs/weather.md "Fluid"): air at rest stays at rest, the projection takes the
// divergence out, a warm bubble rises (the dry thermal of Robert 1993 / Bryan & Fritsch 2002, at the game's scale),
// the same bits run to run; and its cost at ocean_test's size.

namespace {

constexpr int THETA = 0; // the scalar: the potential temperature (K)

WeatherFluid::Params params15() {
    WeatherFluid::Params p;
    p.levels = 15;
    p.top_m = 600.0;
    p.scalars = 1;
    return p;
}

// theta(z) at every cell from a profile per level; the wind 0.
void set_profile(WeatherFluid &p_fluid, const std::vector<double> &p_theta) {
    const WeatherGrid &g = p_fluid.grid();
    std::vector<float> &s = p_fluid.state();
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                float *here = &s[g.index(f, i, j) * size_t(p_fluid.stride())];
                for (int k = 0; k < p_fluid.levels(); k++) {
                    here[p_fluid.scalar_offset(k, THETA)] = float(p_theta[size_t(k)]);
                }
            }
        }
    }
    p_fluid.fill_halo();
}

// One dry step of p_dt (in substeps of at most max_dt).
void dry_step(WeatherFluid &p_fluid, const std::vector<double> &p_ref, double p_dt) {
    std::vector<float> b(p_fluid.cell_count());
    double left = p_dt;
    while (left > 0.0) {
        const double dt = std::min(left, p_fluid.max_dt());
        p_fluid.advect(dt);
        p_fluid.dry_buoyancy(THETA, p_ref, b.data());
        p_fluid.force(dt, b.data());
        p_fluid.project(dt);
        left -= dt;
    }
}

// The largest |w| and the largest w over the interior.
void vertical_wind(const WeatherFluid &p_fluid, double &r_abs, double &r_up) {
    const WeatherGrid &g = p_fluid.grid();
    r_abs = 0.0;
    r_up = -1e30;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                double up[3];
                p_fluid.column_up(g.interior(f, i, j), up);
                const float *here = &p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride())];
                for (int k = 0; k < p_fluid.levels(); k++) {
                    const float *u = here + p_fluid.wind_offset(k);
                    const double w = u[0] * up[0] + u[1] * up[1] + u[2] * up[2];
                    r_abs = std::max(r_abs, std::fabs(w));
                    r_up = std::max(r_up, w);
                }
            }
        }
    }
}

// A warm bubble: theta + p_dtheta cos^2(pi r / 2R) within p_radius_m of the point p_height_m over face 0's middle.
void add_bubble(WeatherFluid &p_fluid, double p_dtheta, double p_radius_m, double p_height_m) {
    const WeatherGrid &g = p_fluid.grid();
    double c[3], cu[3];
    const size_t mid = g.interior(0, g.width() / 2, g.height() / 2);
    p_fluid.column_pos(mid, c);
    p_fluid.column_up(mid, cu);
    for (int a = 0; a < 3; a++) {
        c[a] += cu[a] * p_height_m;
    }
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                double pos[3], up[3];
                p_fluid.column_pos(g.interior(f, i, j), pos);
                p_fluid.column_up(g.interior(f, i, j), up);
                float *here = &p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride())];
                for (int k = 0; k < p_fluid.levels(); k++) {
                    double r2 = 0.0;
                    for (int a = 0; a < 3; a++) {
                        const double d = pos[a] + up[a] * p_fluid.level_m(k) - c[a];
                        r2 += d * d;
                    }
                    const double r = std::sqrt(r2);
                    if (r < p_radius_m) {
                        const double cw = std::cos(0.5 * 3.14159265358979 * r / p_radius_m);
                        here[p_fluid.scalar_offset(k, THETA)] += float(p_dtheta * cw * cw);
                    }
                }
            }
        }
    }
    p_fluid.fill_halo();
}

// The height of the warm anomaly's centroid (theta over p_base, weighted by the excess).
double warm_centroid_m(const WeatherFluid &p_fluid, double p_base) {
    const WeatherGrid &g = p_fluid.grid();
    double sum = 0.0;
    double weighted = 0.0;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const float *here = &p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride())];
                for (int k = 0; k < p_fluid.levels(); k++) {
                    const double e = std::max(0.0, here[p_fluid.scalar_offset(k, THETA)] - p_base);
                    sum += e;
                    weighted += e * p_fluid.level_m(k);
                }
            }
        }
    }
    return sum > 0.0 ? weighted / sum : 0.0;
}

} // namespace

TEST_CASE("WeatherFluid: stratified air at rest stays at rest") {
    const auto grid = std::make_shared<CubeSphereGrid>(16, 1000.0);
    WeatherFluid fluid(grid, params15());
    std::vector<double> ref;
    for (int k = 0; k < fluid.levels(); k++) {
        ref.push_back(300.0 + 0.005 * fluid.level_m(k)); // 5 K/km: stable
    }
    set_profile(fluid, ref);
    for (int t = 0; t < 10; t++) {
        dry_step(fluid, ref, 2.0);
    }
    double w_abs = 0.0, w_up = 0.0;
    vertical_wind(fluid, w_abs, w_up);
    CHECK(w_abs == 0.0);
    CHECK(fluid.max_divergence() == 0.0);
}

TEST_CASE("WeatherFluid: the projection takes the divergence out") {
    const auto grid = std::make_shared<CubeSphereGrid>(16, 1000.0);
    WeatherFluid fluid(grid, params15());
    set_profile(fluid, std::vector<double>(size_t(fluid.levels()), 300.0));
    // A converging, rising wind: horizontal inflow towards face 0's middle and updraft there, smooth.
    const WeatherGrid &g = fluid.grid();
    double c[3];
    fluid.column_pos(g.interior(0, 8, 8), c);
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                double pos[3], up[3];
                fluid.column_pos(g.interior(f, i, j), pos);
                fluid.column_up(g.interior(f, i, j), up);
                double d[3];
                double r2 = 0.0;
                for (int a = 0; a < 3; a++) {
                    d[a] = c[a] - pos[a];
                    r2 += d[a] * d[a];
                }
                const double gauss = std::exp(-r2 / (300.0 * 300.0));
                float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                for (int k = 0; k < fluid.levels(); k++) {
                    const double shape = std::sin(3.14159265358979 * fluid.level_m(k) / fluid.top_m());
                    for (int a = 0; a < 3; a++) {
                        here[fluid.wind_offset(k) + a] = float(gauss * (0.01 * d[a] * shape + 2.0 * up[a] * shape));
                    }
                }
            }
        }
    }
    fluid.vertical_from_centres(); // the w set at the centres
    const double before = fluid.max_divergence();
    for (int t = 0; t < 3; t++) {
        fluid.project(1.0); // warm-started: each projection takes out more
    }
    const double after = fluid.max_divergence();
    MESSAGE("max divergence ", before, " -> ", after, " 1/s");
    CHECK(before > 1e-3);
    CHECK(after < 0.1 * before);
}

TEST_CASE("WeatherFluid: a warm bubble rises at the thermal's speed, the air round it sinking as much") {
    const auto grid = std::make_shared<CubeSphereGrid>(32, 1000.0); // cells ~49 m
    WeatherFluid fluid(grid, params15());
    const std::vector<double> ref(size_t(fluid.levels()), 300.0); // neutral
    set_profile(fluid, ref);
    add_bubble(fluid, 2.0, 150.0, 170.0);
    const double start = warm_centroid_m(fluid, 300.0);
    for (int t = 0; t < 12; t++) { // 120 s
        dry_step(fluid, ref, 10.0);
    }
    const double risen = warm_centroid_m(fluid, 300.0) - start;
    double w_abs = 0.0, w_up = 0.0;
    vertical_wind(fluid, w_abs, w_up);
    // A thermal of radius R and buoyancy g' rises at about 0.5 to 1 sqrt(g' R) (Scorer 1957): ~1.4 to 2.8 m/s here.
    const double scale = std::sqrt(WeatherFluid::GRAVITY * 2.0 / WeatherFluid::THETA_0 * 150.0);
    MESSAGE("risen ", risen, " m, w up ", w_up, " m/s (sqrt(g' R) ", scale, "), max divergence ", fluid.max_divergence());
    CHECK(risen > 60.0);
    CHECK(w_up > 0.3 * scale);
    CHECK(w_up < 3.0 * scale);
    // The air round it comes down as much as the bubble's goes up: between a rigid ground and top, every level's net
    // vertical flux is nothing (within the approximate projection's few percent).
    const WeatherGrid &g = fluid.grid();
    for (int k = 0; k < fluid.levels(); k++) {
        double net = 0.0;
        double total = 0.0;
        for (int f = 0; f < g.faces(); f++) {
            for (int j = 0; j < g.height(); j++) {
                for (int i = 0; i < g.width(); i++) {
                    double up[3];
                    fluid.column_up(g.interior(f, i, j), up);
                    const float *u = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride()) + size_t(fluid.wind_offset(k))];
                    const double w = u[0] * up[0] + u[1] * up[1] + u[2] * up[2];
                    const double area = fluid.column_area_m2(g.interior(f, i, j));
                    net += w * area;
                    total += std::fabs(w) * area;
                }
            }
        }
        CHECK(std::fabs(net) < 0.05 * total);
    }
}

// A 5 m/s rotation about the pole axis, the wind at every cell (halo included) the field's at its surface point.
// Semi-Lagrangian in planet axes, a wind taken from upstream points along the upstream tangent plane: one step without
// the parallel transport (WeatherFluid::transport) left it turned by the distance travelled, 0.125 m/s at the faces'
// edges in one 5 s step, and the error grew to 6.5 m/s in 10 minutes.
TEST_CASE("WeatherFluid: the advection turns the wind with the sphere") {
    const auto grid = std::make_shared<CubeSphereGrid>(32, 1000.0);
    WeatherFluid::Params p;
    p.levels = 2;
    p.top_m = 80.0;
    p.scalars = 1;
    WeatherFluid fluid(grid, p);
    const WeatherGrid &g = fluid.grid();
    auto rotation = [](const double p_pos[3], double r_wind[3]) {
        const double l = std::sqrt(p_pos[0] * p_pos[0] + p_pos[1] * p_pos[1] + p_pos[2] * p_pos[2]);
        r_wind[0] = 5.0 * p_pos[2] / l;
        r_wind[1] = 0.0;
        r_wind[2] = -5.0 * p_pos[0] / l;
    };
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                double pos[3], up[3], u[3];
                g.cell_surface(f, i, j, pos, up);
                rotation(pos, u);
                for (int k = 0; k < fluid.levels(); k++) {
                    for (int a = 0; a < 3; a++) {
                        here[fluid.wind_offset(k) + a] = float(u[a]);
                    }
                }
            }
        }
    }
    fluid.fill_halo();
    fluid.advect(5.0);
    double worst = 0.0;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                double pos[3], up[3], u[3];
                g.cell_surface(f, i, j, pos, up);
                rotation(pos, u);
                worst = std::max(worst, std::sqrt(std::pow(here[0] - u[0], 2) + std::pow(here[1] - u[1], 2) + std::pow(here[2] - u[2], 2)));
            }
        }
    }
    MESSAGE("one step of a 5 m/s rotation: worst error ", worst, " m/s");
    CHECK(worst < 0.08);
}
TEST_CASE("WeatherFluid: the same bits run to run") {
    const auto grid = std::make_shared<CubeSphereGrid>(12, 1000.0);
    std::vector<float> state[2];
    for (int run = 0; run < 2; run++) {
        WeatherFluid fluid(grid, params15());
        const std::vector<double> ref(size_t(fluid.levels()), 300.0);
        set_profile(fluid, ref);
        add_bubble(fluid, 2.0, 200.0, 170.0);
        for (int t = 0; t < 5; t++) {
            dry_step(fluid, ref, 5.0);
        }
        state[run] = fluid.state();
    }
    CHECK(std::memcmp(state[0].data(), state[1].data(), state[0].size() * sizeof(float)) == 0);
}

TEST_CASE("WeatherFluid: a step's cost at ocean_test's size (6 x 70^2 columns, 15 levels)") {
    const auto grid = std::make_shared<CubeSphereGrid>(70, 2000.0);
    WeatherFluid fluid(grid, params15());
    const std::vector<double> ref(size_t(fluid.levels()), 300.0);
    set_profile(fluid, ref);
    add_bubble(fluid, 2.0, 150.0, 170.0);
    std::vector<float> b(fluid.cell_count());
    const uint64_t t0 = ov::now_usec();
    const int steps = 3;
    for (int t = 0; t < steps; t++) {
        fluid.advect(1.0);
        fluid.dry_buoyancy(THETA, ref, b.data());
        fluid.force(1.0, b.data());
        fluid.project(1.0);
    }
    const double ms = double(ov::now_usec() - t0) * 1e-3 / steps;
    MESSAGE("WeatherFluid step: ", ms, " ms for ", fluid.cell_count(), " cells");
    CHECK(ms > 0.0);
}

// The flux-form advection: what leaves a cell enters its neighbour (the total kept to the floats' rounding), and the
// limiter keeps a blob within its start's bounds (no new extreme: rain never negative); on even levels and stretched ones
// (WeatherLevels), the wind turning over through them.
TEST_CASE("WeatherFluid: the advection conserves a scalar and keeps it within its bounds") {
    for (const double stretch : { 1.0, 1.25 }) {
        CAPTURE(stretch);
        const auto grid = std::make_shared<CubeSphereGrid>(24, 1000.0);
        WeatherFluid::Params p;
        p.levels = 6;
        p.top_m = 300.0;
        p.stretch = stretch;
        p.scalars = 1;
        WeatherFluid fluid(grid, p);
        const WeatherGrid &g = fluid.grid();
        auto total = [&](double &r_lo, double &r_hi) {
            double sum = 0.0;
            r_lo = 1e30;
            r_hi = -1e30;
            for (size_t in = 0; in < g.cells(); in++) {
                const int f = int(in / (size_t(g.width()) * size_t(g.height())));
                const int j = int((in / size_t(g.width())) % size_t(g.height()));
                const int i = int(in % size_t(g.width()));
                const float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                for (int k = 0; k < fluid.levels(); k++) {
                    const double v = here[fluid.scalar_offset(k, 0)];
                    sum += v * fluid.column_area_m2(in) * fluid.thick_m(k);
                    r_lo = std::min(r_lo, v);
                    r_hi = std::max(r_hi, v);
                }
            }
            return sum;
        };
        for (int f = 0; f < g.faces(); f++) {
            for (int j = 0; j < g.height(); j++) {
                for (int i = 0; i < g.width(); i++) {
                    float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                    double pos[3], up[3];
                    g.cell_surface(f, i, j, pos, up);
                    const double l = std::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
                    const double u[3] = { 5.0 * pos[2] / l, 0.0, -5.0 * pos[0] / l };
                    // A vertical stir (the projection makes the whole of it divergence-free): the blob crosses the levels.
                    const double stir = 2.0 * std::sin(6.0 * pos[0] / l) * std::cos(5.0 * pos[1] / l);
                    for (int k = 0; k < fluid.levels(); k++) {
                        for (int a = 0; a < 3; a++) {
                            here[fluid.wind_offset(k) + a] = float(u[a] + stir * up[a]);
                        }
                        here[fluid.scalar_offset(k, 0)] = (f == 0 && i > 8 && i < 14 && j > 8 && j < 14 && k == 2) ? 1.0f : 0.0f;
                    }
                }
            }
        }
        fluid.vertical_from_centres();
        // The wind's discrete divergence out of the sides' speeds, the solve converged (a divergent wind makes extremes of
        // its own in flux form: one projection of the stir left them at -1e-4).
        for (int it = 0; it < 40; it++) {
            fluid.project(1.0);
        }
        MESSAGE("stretch ", stretch, ": max divergence ", fluid.max_divergence(), " 1/s");
        double lo0, hi0;
        const double before = total(lo0, hi0);
        for (int s = 0; s < 100; s++) {
            const double dt = fluid.max_dt();
            fluid.advect(dt);
            fluid.project(dt); // as a step does: the wind carries itself into a divergent one otherwise
        }
        double lo, hi;
        const double after = total(lo, hi);
        MESSAGE("stretch ", stretch, ": a blob after 100 steps of a 5 m/s rotation and a stir: total ", after / before, " of its start, range [", lo, ", ", hi, "]");
        CHECK(std::fabs(after / before - 1.0) < 1e-5);
        // The projection's leftover divergence (1e-7 1/s) makes extremes of its own in flux form: -3e-6 on even levels,
        // -8e-6 stretched under the stir.
        CHECK(lo >= -2e-5);
        CHECK(hi <= 1.0 + 1e-6);
    }
}

// The levels' heights (WeatherLevels): each stretch times the one under it, adding up to the top, the closed-form
// index the inverse of the heights; alike when stretch is 1.
TEST_CASE("WeatherLevels: geometric levels, their heights and their index") {
    const WeatherLevels even{ 15, 600.0, 1.0 };
    CHECK(even.thick_m(0) == doctest::Approx(40.0));
    CHECK(even.thick_m(14) == doctest::Approx(40.0));
    CHECK(even.index_at(300.0) == doctest::Approx(7.5));
    const WeatherLevels l = WeatherLevels::for_top(4000.0, 45.0, 1.12, WeatherFluid::MAX_LEVELS);
    MESSAGE("4 km over a 45 m cell at 1.12: ", l.count, " levels, the lowest ", l.thick_m(0), " m, the top's ", l.thick_m(l.count - 1), " m");
    CHECK(l.face_m(0) == 0.0);
    CHECK(l.face_m(l.count) == doctest::Approx(4000.0));
    CHECK(l.thick_m(0) == doctest::Approx(45.0).epsilon(0.15));
    for (int k = 1; k < l.count; k++) {
        CHECK(l.thick_m(k) == doctest::Approx(l.thick_m(k - 1) * 1.12));
    }
    for (int m = 0; m <= l.count; m++) {
        CHECK(l.index_at(l.face_m(m)) == doctest::Approx(double(m)).epsilon(1e-9));
    }
    CHECK(WeatherLevels::for_top(600.0, 45.0, 1.0, 64).count == 13); // as before the stretch: top over the cell
}

namespace {

// A random wind over stable air run p_minutes (p_n cells a face side, Coriolis p_omega), nothing forcing it: the
// largest ratio of the mean kinetic energy at a minute to its lowest before.
double random_wind_regrowth(int p_n, double p_omega, int p_minutes) {
    const auto grid = std::make_shared<CubeSphereGrid>(p_n, 1000.0);
    WeatherFluid::Params params = params15();
    params.omega_rad_s = p_omega;
    WeatherFluid fluid(grid, params);
    const WeatherGrid &g = fluid.grid();
    std::vector<double> ref;
    for (int k = 0; k < fluid.levels(); k++) {
        ref.push_back(300.0 + 0.005 * fluid.level_m(k));
    }
    set_profile(fluid, ref);
    uint32_t key = 1;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                double pos[3], up[3];
                g.cell_surface(f, i, j, pos, up);
                float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                for (int k = 0; k < fluid.levels(); k++) {
                    double u[3];
                    for (int a = 0; a < 3; a++) {
                        key = key * 1664525u + 1013904223u;
                        u[a] = 5.0 * (double(key >> 8) / double(1u << 24) * 2.0 - 1.0);
                    }
                    const double n = u[0] * up[0] + u[1] * up[1] + u[2] * up[2];
                    for (int a = 0; a < 3; a++) {
                        here[fluid.wind_offset(k) + a] = float(u[a] - n * up[a]);
                    }
                }
            }
        }
    }
    fluid.vertical_from_centres();
    fluid.project(1.0);
    double energy = 0.0; // the mean of |u|^2 / 2 over the cells, strongest()'s
    auto strongest = [&](int &r_face, int &r_i, int &r_j, int &r_k) {
        double m = 0.0;
        energy = 0.0;
        for (int f = 0; f < g.faces(); f++) {
            for (int j = 0; j < g.height(); j++) {
                for (int i = 0; i < g.width(); i++) {
                    const float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                    for (int k = 0; k < fluid.levels(); k++) {
                        const float *u = here + fluid.wind_offset(k);
                        const double s = std::sqrt(double(u[0]) * u[0] + double(u[1]) * u[1] + double(u[2]) * u[2]);
                        energy += 0.5 * s * s / double(fluid.cell_count());
                        if (s > m) {
                            m = s;
                            r_face = f;
                            r_i = i;
                            r_j = j;
                            r_k = k;
                        }
                    }
                }
            }
        }
        return m;
    };
    int f = 0, i = 0, j = 0, k = 0;
    const double start = strongest(f, i, j, k);
    MESSAGE(ov::format("start: strongest wind %.2f m/s, mean energy %.4f m^2/s^2", start, energy));
    double lowest = energy, regrowth = 1.0;
    for (int minute = 1; minute <= p_minutes; minute++) {
        dry_step(fluid, ref, 60.0);
        const double m = strongest(f, i, j, k);
        regrowth = std::max(regrowth, energy / lowest);
        lowest = std::min(lowest, energy);
        MESSAGE(ov::format("minute %d: strongest wind %.2f m/s at face %d (%d, %d) level %d, mean energy %.4f m^2/s^2", minute, m, f, i, j, k, energy));
    }
    MESSAGE("largest regrowth of the energy: x", regrowth);
    return regrowth;
}

} // namespace

// A random wind over stable air with the game's rotation (a 5-minute day), nothing forcing it: the dynamics may only take
// energy out. The game's weather blew up from a cube corner (2026-11-26: 17 -> 74 m/s in 2 game minutes): the faces'
// geometry was the halo's points' (the face's own grid past its edge, not the cell across the seam) and the centres
// lost a halo-read pressure gradient; with rotation the seams' error turned into wind the projection can't remove, the
// energy x95 in 20 minutes. With the true neighbours' geometry a corner still regrows it x2.9 once, then it holds.
TEST_CASE("WeatherFluid: a random wind over stable air never regains its energy") {
    CHECK(random_wind_regrowth(70, 2.0 * 3.14159265358979 / 300.0, 20) < 4.0);
}
