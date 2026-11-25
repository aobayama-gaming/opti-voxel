#include "test_common.hpp"

#include "weather_forcing.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// The weather's forcing (WeatherForcing, docs/weather.md "Forcing"): the sounding, the grounds under the sun, the
// prevailing wind, the storms' seeds; then the whole of it over a warm sea gives cumulus.

namespace {

using F = WeatherForcing;
using M = WeatherMoist;

struct Setup {
    std::shared_ptr<CubeSphereGrid> grid;
    std::unique_ptr<WeatherFluid> fluid;
    std::unique_ptr<F> drive;
};

Setup make(int p_n, float p_ground_k, float p_storms, const F::Params &p_params = F::Params()) {
    Setup s;
    s.grid = std::make_shared<CubeSphereGrid>(p_n, 1000.0);
    WeatherFluid::Params fp;
    fp.levels = 15;
    fp.top_m = 600.0;
    fp.scalars = M::SCALARS;
    fp.omega_rad_s = 2.0 * 3.14159265358979323846 / p_params.day_s;
    s.fluid = std::make_unique<WeatherFluid>(s.grid, fp);
    std::vector<Climate::BiomeWeather> climate(s.grid->cells());
    for (Climate::BiomeWeather &c : climate) {
        c.ground_k = p_ground_k;
        c.storms = p_storms;
    }
    s.drive = std::make_unique<F>(*s.fluid, p_params, climate);
    return s;
}

} // namespace

TEST_CASE("WeatherForcing: the sounding's mixed layer saturates at the cloud base, the inversion caps it") {
    Setup s = make(16, 300.5f, 0.0f);
    const M::Sounding snd = s.drive->sounding(*s.fluid);
    const double theta0 = 300.5 - F::AIR_BELOW_GROUND_K;
    CHECK(snd.theta[0] == doctest::Approx(theta0));
    // A ground parcel lifted dry to the base (0.53 of 300 m) is saturated there, under it not.
    const double base = F::CLOUD_BASE * F::Params().inversion_m;
    for (const double z : { base - 40.0, base, base + 40.0 }) {
        const double exner = 1.0 - M::GRAVITY / (M::CP * theta0) * z;
        const double qs = M::saturation(theta0 * exner, M::P0_MB * std::pow(exner, 1.0 / M::KAPPA_EXNER));
        CHECK((snd.qv[0] >= qs * (1.0 - 1e-9)) == (z >= base));
    }
    // Over the inversion: warmer by its jump, dry.
    const int top = s.fluid->levels() - 1;
    CHECK(snd.theta[size_t(top)] > theta0 + F::INVERSION_K);
    CHECK(snd.qv[size_t(top)] == doctest::Approx(F::ABOVE_HUMIDITY * snd.qv[0]));
}

TEST_CASE("WeatherForcing: the ground warms by day, cools by night, its day's mean its ground_k") {
    Setup s = make(8, 299.0f, 0.0f);
    std::vector<Climate::BiomeWeather> climate(s.grid->cells());
    for (Climate::BiomeWeather &c : climate) {
        c.diurnal_k = 10.0f;
    }
    F drive(*s.fluid, F::Params(), climate);
    // The column under the sun at noon, then the sun round its equator: the mean of a day.
    std::vector<M::Ground> ground;
    double up[3];
    s.fluid->column_up(0, up);
    drive.grounds(Float3{ float(up[0]), float(up[1]), float(up[2]) }, ground);
    CHECK(ground[0].temp_k == doctest::Approx(299.0 + 10.0 * (1.0 - 1.0 / 3.14159265358979323846)).epsilon(1e-5));
    drive.grounds(Float3{ float(-up[0]), float(-up[1]), float(-up[2]) }, ground);
    CHECK(ground[0].temp_k == doctest::Approx(299.0 - 10.0 / 3.14159265358979323846).epsilon(1e-5));
    // A day of the sun circling the column's own great circle: the mean is ground_k.
    double side[3] = { -up[1], up[0], 0.0 };
    const double l = std::sqrt(side[0] * side[0] + side[1] * side[1]);
    double mean = 0.0;
    const int hours = 3600;
    for (int h = 0; h < hours; h++) {
        const double a = 2.0 * 3.14159265358979323846 * (h + 0.5) / hours;
        const Float3 sun{ float(up[0] * std::cos(a) + side[0] / l * std::sin(a)), float(up[1] * std::cos(a) + side[1] / l * std::sin(a)), float(up[2] * std::cos(a)) };
        drive.grounds(sun, ground);
        mean += ground[0].temp_k / hours;
    }
    CHECK(mean == doctest::Approx(299.0).epsilon(1e-4));
}

TEST_CASE("WeatherForcing: the prevailing wind blows from the east in the trades, from the west at 45 degrees") {
    const double s = std::sin(3.14159265358979323846 / 12.0); // 15 degrees north, on x = 0
    const double trades[3] = { 0.0, s, std::sqrt(1.0 - s * s) };
    double w[3];
    F::prevailing(trades, 5.0, w);
    // East at z = 1 on the equator is pole x up = +x: the trades blow towards -x.
    CHECK(w[0] == doctest::Approx(-5.0).epsilon(1e-6));
    const double h = std::sqrt(0.5);
    const double westerlies[3] = { 0.0, h * std::cos(0.0), h };
    F::prevailing(westerlies, 5.0, w);
    CHECK(w[0] > 0.0);
}

TEST_CASE("WeatherForcing: storms seed at their rate, the same ones from the same seed") {
    // A planet of storms 60 starts one a minute: 10 minutes of 5 s steps, ~10 storms.
    Setup a = make(16, 300.0f, 60.0f);
    Setup b = make(16, 300.0f, 60.0f);
    for (uint64_t step = 0; step < 120; step++) {
        a.drive->apply(*a.fluid, 5.0, step);
        b.drive->apply(*b.fluid, 5.0, step);
    }
    MESSAGE("storms in 10 minutes at 60 an hour: ", a.drive->storms());
    CHECK(a.drive->storms() >= 3);
    CHECK(a.drive->storms() <= 20);
    CHECK(a.drive->storms() == b.drive->storms());
    CHECK(a.fluid->state() == b.fluid->state());
}

// The forcing's whole start over a sea 1.5 K warmer than its air (a planet of trade-wind sea, its large-scale wind and
// Coriolis on), sampled each minute from 15 to 40 (the convection comes in bursts while the layer adjusts: one instant
// says little): cumulus on average, broken, their base above the ground, nothing over the inversion.
TEST_CASE("WeatherForcing: a trade-wind sea makes broken cumulus") {
    Setup s = make(32, 300.5f, 0.0f);
    const M::Sounding snd = s.drive->sounding(*s.fluid);
    WeatherMoist moist(*s.fluid, snd, M::Params());
    s.drive->start(*s.fluid, snd);
    std::vector<M::Ground> ground;
    s.drive->grounds(Float3{ 0.0f, 1.0f, 0.0f }, ground);
    std::vector<float> b(s.fluid->cell_count());
    const WeatherGrid &g = s.fluid->grid();
    const int levels = s.fluid->levels();
    std::vector<double> level(size_t(levels), 0.0);
    double columns = 0.0, peak = 0.0;
    int samples = 0;
    double t = 0.0;
    uint64_t step = 0;
    for (int minute = 1; minute <= 40; minute++) {
        while (t < 60.0 * minute) {
            const double dt = std::min(5.0, s.fluid->max_dt());
            s.fluid->advect(dt);
            s.drive->apply(*s.fluid, dt, step++);
            moist.apply(*s.fluid, ground, dt, b.data());
            s.fluid->force(dt, b.data());
            s.fluid->project(dt);
            t += dt;
        }
        if (minute < 15) {
            continue;
        }
        size_t cloudy = 0;
        for (int f = 0; f < g.faces(); f++) {
            for (int j = 0; j < g.height(); j++) {
                for (int i = 0; i < g.width(); i++) {
                    const float *here = &s.fluid->state()[g.index(f, i, j) * size_t(s.fluid->stride())];
                    bool any = false;
                    for (int k = 0; k < levels; k++) {
                        const bool c = here[s.fluid->scalar_offset(k, M::QC)] > 1e-5f;
                        level[size_t(k)] += c ? 1.0 : 0.0;
                        any = any || c;
                    }
                    cloudy += any;
                }
            }
        }
        const double share = double(cloudy) / double(g.cells());
        columns += share;
        peak = std::max(peak, share);
        samples++;
    }
    columns /= samples;
    int base = -1;
    double most = 0.0;
    for (int k = 0; k < levels; k++) {
        level[size_t(k)] /= double(samples) * double(g.cells());
        most = std::max(most, level[size_t(k)]);
        if (base < 0 && level[size_t(k)] > 0.0) {
            base = k;
        }
        MESSAGE("level ", k, " (", s.fluid->level_m(k), " m): cloud ", level[size_t(k)]);
    }
    MESSAGE("minutes 15-40: cloudy columns ", columns, " on average, ", peak, " at most");
    CHECK(columns > 0.003); // ~0.8%: BOMEX's ~10% under the twisted rotation (f up to 0.04 /s stiffens the thermals)
    CHECK(peak > 0.02);
    CHECK(peak < 0.6);
    CHECK(level[0] < 0.5 * most); // the base above the ground
    CHECK(level[size_t(levels - 1)] == 0.0);
}
