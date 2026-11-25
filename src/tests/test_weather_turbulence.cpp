#include "test_common.hpp"

#include "weather_forcing.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// The weather's small swirls (docs/weather.md "Turbulence"): the vorticity confinement keeps a vortex the grid would
// smear, and leaves still air still; the kicks start swirls in still air, the same ones every run.

namespace {

WeatherFluid::Params dry(double p_confinement) {
    WeatherFluid::Params p;
    p.levels = 8;
    p.top_m = 400.0;
    p.scalars = 1;
    p.omega_rad_s = 0.0;
    p.confinement = p_confinement;
    return p;
}

// A Gaussian vortex about face 0's centre column: its speed peaks at p_peak_mps p_r0_m out, through every level.
void add_vortex(WeatherFluid &p_fluid, double p_peak_mps, double p_r0_m) {
    const WeatherGrid &g = p_fluid.grid();
    double centre[3];
    p_fluid.column_pos(g.interior(0, g.width() / 2, g.height() / 2), centre);
    for (int j = 0; j < g.height(); j++) {
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(0, i, j);
            double pos[3], up[3];
            p_fluid.column_pos(in, pos);
            p_fluid.column_up(in, up);
            double d[3] = { pos[0] - centre[0], pos[1] - centre[1], pos[2] - centre[2] };
            const double along = d[0] * up[0] + d[1] * up[1] + d[2] * up[2];
            for (int a = 0; a < 3; a++) {
                d[a] -= along * up[a];
            }
            const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / p_r0_m;
            const double s = p_peak_mps * std::exp(0.5 * (1.0 - r * r)) / p_r0_m; // speed r exp(...) over the distance
            const double turn[3] = { up[1] * d[2] - up[2] * d[1], up[2] * d[0] - up[0] * d[2], up[0] * d[1] - up[1] * d[0] };
            float *here = &p_fluid.state()[g.index(0, i, j) * size_t(p_fluid.stride())];
            for (int k = 0; k < p_fluid.levels(); k++) {
                for (int a = 0; a < 3; a++) {
                    here[p_fluid.wind_offset(k) + a] += float(s * turn[a]);
                }
            }
        }
    }
    p_fluid.fill_halo();
}

double max_speed(const WeatherFluid &p_fluid) {
    const WeatherGrid &g = p_fluid.grid();
    double most = 0.0;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const float *here = &p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride())];
                for (int k = 0; k < p_fluid.levels(); k++) {
                    const float *u = &here[p_fluid.wind_offset(k)];
                    most = std::max(most, std::sqrt(double(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])));
                }
            }
        }
    }
    return most;
}

void dry_run(WeatherFluid &p_fluid, double p_seconds) {
    std::vector<float> b(p_fluid.cell_count(), 0.0f);
    for (double t = 0.0; t < p_seconds; t += 2.0) {
        p_fluid.advect(2.0);
        p_fluid.force(2.0, b.data());
        p_fluid.project(2.0);
    }
}

std::unique_ptr<WeatherForcing> calm_drive(WeatherFluid &p_fluid, double p_kick_mps) {
    WeatherForcing::Params dp;
    dp.prevailing_mps = 0.0;
    dp.kick_mps = p_kick_mps;
    dp.seed = 7;
    std::vector<Climate::BiomeWeather> climate(p_fluid.grid().cells());
    for (Climate::BiomeWeather &c : climate) {
        c.storms = 0.0f;
    }
    return std::make_unique<WeatherForcing>(p_fluid, dp, climate);
}

} // namespace

TEST_CASE("WeatherFluid: the vorticity confinement leaves still air still") {
    const auto grid = std::make_shared<CubeSphereGrid>(16, 1000.0);
    WeatherFluid fluid(grid, dry(1.0));
    dry_run(fluid, 20.0);
    CHECK(max_speed(fluid) == 0.0);
}

TEST_CASE("WeatherFluid: the vorticity confinement keeps a vortex the grid smears") {
    const auto grid = std::make_shared<CubeSphereGrid>(32, 1000.0); // cells ~49 m
    double kept[2];
    for (int n = 0; n < 2; n++) {
        WeatherFluid fluid(grid, dry(n == 0 ? 0.0 : 0.05));
        add_vortex(fluid, 4.0, 100.0); // 2 cells: the size the semi-Lagrangian smears
        dry_run(fluid, 300.0);
        kept[n] = max_speed(fluid) / 4.0;
    }
    MESSAGE("a 2-cell vortex after 5 minutes: ", kept[0], " of its peak, ", kept[1], " with confinement 0.05 (0.15 grew it 2.4x, 0.3 6x)");
    CHECK(kept[1] > kept[0] * 2.0);
    CHECK(kept[1] < 1.2); // and does not run away
}

TEST_CASE("WeatherForcing: the kicks stir still air, the same every run") {
    const auto grid = std::make_shared<CubeSphereGrid>(16, 1000.0);
    std::vector<float> state[2];
    for (int n = 0; n < 2; n++) {
        WeatherFluid fluid(grid, dry(0.0));
        auto drive = calm_drive(fluid, 1.0);
        for (int s = 0; s < 60; s++) { // a minute: each point kicked once on average
            drive->apply(fluid, 1.0, uint64_t(s));
        }
        state[n] = fluid.state();
        const double speed = max_speed(fluid);
        MESSAGE("strongest wind after a minute of kicks: ", speed, " m/s");
        CHECK(speed > 0.5);
        CHECK(speed < 10.0); // overlapping kicks add up
    }
    CHECK(state[0] == state[1]);
    WeatherFluid fluid(grid, dry(0.0));
    auto drive = calm_drive(fluid, 0.0);
    for (int s = 0; s < 60; s++) {
        drive->apply(fluid, 1.0, uint64_t(s));
    }
    CHECK(max_speed(fluid) == 0.0);
}
