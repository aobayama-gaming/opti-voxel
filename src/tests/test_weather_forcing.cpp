#include "test_common.hpp"

#include "weather_forcing.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// The weather's forcing (WeatherForcing, docs/weather.md "Forcing"): the sounding, the grounds under the sun, the
// storms' seeds; then the whole of it over a warm sea gives cumulus.

namespace {

using F = WeatherForcing;
using M = WeatherMoist;

struct Setup {
    std::shared_ptr<CubeSphereGrid> grid;
    std::unique_ptr<WeatherFluid> fluid;
    std::unique_ptr<F> drive;
};

Setup make(int p_n, float p_ground_k, float p_storms, const F::Params &p_params = F::Params(), const WeatherLevels &p_levels = WeatherLevels()) {
    Setup s;
    s.grid = std::make_shared<CubeSphereGrid>(p_n, 1000.0);
    WeatherFluid::Params fp;
    fp.levels = p_levels.count;
    fp.top_m = p_levels.top_m;
    fp.stretch = p_levels.stretch;
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

// A deep atmosphere (3 km, stretched levels): a saturated parcel of the mixed layer stays colder than the air over the
// inversion up to the top (the moist adiabat it follows, kept warmer): no thermal crosses it into a deck under the lid.
// The old free atmosphere (3 K/km) let it through from 1.25 km up, +5 K at the top: the whole planet's sky filled in 2
// game minutes (2026-11-27).
TEST_CASE("WeatherForcing: over the inversion, a deep sounding holds a lifted parcel back") {
    Setup s = make(8, 299.5f, 0.0f, F::Params(), WeatherLevels::for_top(3000.0, 45.0, 1.12, WeatherFluid::MAX_LEVELS));
    const M::Sounding snd = s.drive->sounding(*s.fluid);
    const double theta0 = snd.theta[0];
    const double base = F::CLOUD_BASE * F::Params().inversion_m;
    const double over = F::Params().inversion_m + F::INVERSION_DEPTH_M;
    int checked = 0;
    double least = 1e30;
    for (int k = 0; k < s.fluid->levels(); k++) {
        const double z = s.fluid->level_m(k);
        if (z <= over) {
            continue;
        }
        const double parcel = M::moist_adiabat(theta0, base, z, theta0);
        const double qs = M::saturation(parcel * snd.exner[size_t(k)], M::P0_MB * std::pow(snd.exner[size_t(k)], 1.0 / M::KAPPA_EXNER));
        const double margin = snd.theta[size_t(k)] * (1.0 + 0.61 * snd.qv[size_t(k)]) - parcel * (1.0 + 0.61 * qs);
        least = std::min(least, margin);
        CHECK(snd.qv[size_t(k)] < 0.9 * qs); // under saturation
        checked++;
    }
    MESSAGE(checked, " levels over the inversion, the parcel at least ", least, " K colder (virtual)");
    CHECK(checked >= 10);
    CHECK(least > 1.0);
}

TEST_CASE("WeatherForcing: the sun turns west, once a weather day") {
    const double omega = 2.0 * 3.14159265358979323846 / F::Params().day_s;
    // Over z = 1 on the equator east is +x: a quarter day on, the noon sun sets in the west.
    const Float3 noon{ 0.0f, 0.0f, 1.0f };
    const Float3 dusk = F::turn_sun(noon, -omega * F::Params().day_s / 4.0);
    CHECK(dusk.x == doctest::Approx(-1.0).epsilon(1e-6));
    CHECK(std::fabs(dusk.z) < 1e-6f);
    const Float3 day = F::turn_sun(Float3{ 0.3f, 0.4f, 0.866f }, -omega * F::Params().day_s);
    CHECK(day.x == doctest::Approx(0.3).epsilon(1e-5));
    CHECK(day.y == doctest::Approx(0.4).epsilon(1e-6));
    CHECK(day.z == doctest::Approx(0.866).epsilon(1e-5));
}

namespace {

// A slab of p_inertia on the equator at equinox, clear, over 12 weather days of 1 s steps: its last day's mean, swing
// (max - min) and the lag of its warmest after noon (a share of the day).
struct Day {
    double mean = 0.0, swing = 0.0, lag = 0.0;
};

Day slab_day(double p_inertia) {
    constexpr double PI = 3.14159265358979323846;
    const double day_s = F::Params().day_s;
    const double absorbed = F::SUN_W_M2 * (1.0 - F::GROUND_ALBEDO);
    double t_k = 299.0;
    Day d;
    double lo = 1e9, hi = -1e9;
    const int per_day = int(day_s);
    for (int s = 0; s < 12 * per_day; s++) {
        const double h = 2.0 * PI * (s + 0.5) / day_s; // noon at s = 0
        t_k = F::warm_slab(t_k, 299.0, p_inertia, absorbed * (std::max(std::cos(h), 0.0) - 1.0 / PI), day_s, 1.0);
        if (s >= 11 * per_day) {
            d.mean += t_k / per_day;
            lo = std::min(lo, t_k);
            if (t_k > hi) {
                hi = t_k;
                d.lag = double(s - 11 * per_day) / per_day;
            }
        }
    }
    d.swing = hi - lo;
    return d;
}

} // namespace

TEST_CASE("WeatherForcing::warm_slab: land warms into the afternoon, a sea barely moves, the day's mean kept") {
    const Day soil = slab_day(1500.0), sand = slab_day(300.0), sea = slab_day(1e5);
    MESSAGE(std::string(ov::format("swing: soil %.2f K (warmest %.3f day after noon), sand %.2f K, sea %.3f K", soil.swing, soil.lag, sand.swing, sea.swing).c_str()));
    CHECK(soil.mean == doctest::Approx(299.0).epsilon(1e-4));
    CHECK(sea.mean == doctest::Approx(299.0).epsilon(1e-4));
    CHECK(soil.swing > 2.0);
    CHECK(soil.swing < 8.0);
    CHECK(sand.swing == doctest::Approx(soil.swing * 5.0).epsilon(0.05)); // the swing goes as 1 / inertia
    CHECK(sea.swing < 0.1);
    // The force-restore slab lags the sun's warming by about an eighth of a day: the afternoon.
    CHECK(soil.lag > 0.05);
    CHECK(soil.lag < 0.2);
}

TEST_CASE("WeatherForcing::warm: the ground under the noon sun warms, less under a cloud's shade") {
    Setup s = make(8, 299.0f, 0.0f);
    std::vector<Climate::BiomeWeather> climate(s.grid->cells());
    for (Climate::BiomeWeather &c : climate) {
        c.thermal_inertia = 1500.0f;
    }
    F drive(*s.fluid, F::Params(), climate);
    const M::Sounding snd = drive.sounding(*s.fluid);
    // A cumulus of 0.3 g/kg over 120 m: ~0.04 kg/m^2 of water, an albedo of ~0.3.
    CHECK(F::cloud_albedo(0.04) > 0.2);
    CHECK(F::cloud_albedo(0.04) < 0.4);
    CHECK(F::cloud_albedo(0.0) == 0.0);
    double up[3];
    s.fluid->column_up(0, up);
    const Float3 noon{ float(up[0]), float(up[1]), float(up[2]) };
    auto noon_k = [&]() {
        std::vector<M::Ground> ground;
        drive.grounds(ground);
        for (int t = 0; t < 30; t++) {
            drive.warm(noon, 1.0, *s.fluid, snd.rho, nullptr, ground);
        }
        return double(ground[0].temp_k);
    };
    const double clear = noon_k();
    const WeatherGrid &g = *s.grid;
    float *here = &s.fluid->state()[g.index(0, 0, 0) * size_t(s.fluid->stride())];
    CHECK(g.interior(0, 0, 0) == 0);
    for (int k = 4; k < 7; k++) {
        here[s.fluid->scalar_offset(k, M::QC)] = 3e-4f;
    }
    const double shaded = noon_k();
    CHECK(clear > 299.5);
    CHECK(shaded < clear - 0.1);
    CHECK(shaded > 299.0);
}

// The sun sets each ground's baseline: over ground_k on the equator, under it at the poles, a sea's as a land's; the day
// swings it round it (each column's day's mean was held at ground_k before 2026-11-26: nothing to make a wind).
TEST_CASE("WeatherForcing::warm: the sun's baseline, a warm equator and a cold pole, the day swinging round it") {
    Setup s = make(8, 299.0f, 0.0f);
    std::vector<Climate::BiomeWeather> climate(s.grid->cells());
    for (size_t c = 0; c < climate.size(); c++) {
        climate[c].ground_k = 299.0f;
        climate[c].thermal_inertia = c % 2 == 0 ? 1500.0f : 1e5f; // land and sea side by side
    }
    F drive(*s.fluid, F::Params(), climate);
    const M::Sounding snd = drive.sounding(*s.fluid);
    // The baselines average to ground_k over the sphere (area-weighted).
    double area = 0.0, mean = 0.0;
    for (size_t c = 0; c < s.grid->cells(); c++) {
        area += s.fluid->column_area_m2(c);
        mean += s.fluid->column_area_m2(c) * drive.baseline_k(c);
    }
    CHECK(mean / area == doctest::Approx(299.0).epsilon(2e-4));
    size_t equator = 0, pole = 0;
    double low = 2.0, high = -1.0;
    for (size_t c = 0; c < s.grid->cells(); c++) {
        double up[3];
        s.fluid->column_up(c, up);
        if (std::fabs(up[1]) < low) {
            low = std::fabs(up[1]);
            equator = c;
        }
        if (std::fabs(up[1]) > high) {
            high = std::fabs(up[1]);
            pole = c;
        }
    }
    std::vector<M::Ground> ground;
    drive.grounds(ground);
    const double day_s = F::Params().day_s;
    const Float3 noon{ 0.0f, 0.0f, 1.0f }; // the equinox
    double equator_mean = 0.0, pole_mean = 0.0;
    const int per_day = int(day_s);
    for (int t = 0; t < 6 * per_day; t++) {
        drive.warm(F::turn_sun(noon, 2.0 * 3.14159265358979323846 * (t + 0.5) / day_s), 1.0, *s.fluid, snd.rho, nullptr, ground);
        if (t >= 5 * per_day) {
            equator_mean += double(ground[equator].temp_k) / per_day;
            pole_mean += double(ground[pole].temp_k) / per_day;
        }
    }
    MESSAGE("day's mean: equator ", equator_mean, " K (baseline ", drive.baseline_k(equator), "), pole ", pole_mean, " K (baseline ",
            drive.baseline_k(pole), "), ground_k 299");
    CHECK(drive.baseline_k(equator) > 301.0);
    CHECK(drive.baseline_k(pole) < 296.0);
    CHECK(equator_mean == doctest::Approx(drive.baseline_k(equator)).epsilon(3e-4));
    CHECK(pole_mean == doctest::Approx(drive.baseline_k(pole)).epsilon(3e-4));
    // A sea next to the equator's land: the same baseline (a long mean does not depend on the heat it holds).
    const size_t sea = equator % 2 == 0 ? equator + 1 : equator - 1;
    CHECK(std::fabs(drive.baseline_k(sea) - drive.baseline_k(equator)) < 0.5);
    CHECK(ground[sea].temp_k == doctest::Approx(drive.baseline_k(sea)).epsilon(1e-4));
}

// The air relaxes towards its equilibrium (Held & Suarez's radiation, the exchange over the lid for the vapour): the
// start's state; a displacement falls by e over relax_days weather days, the equilibrium stays put.
TEST_CASE("WeatherForcing::relax: the air relaxes to its column's equilibrium over relax_days") {
    Setup s = make(8, 299.0f, 0.0f);
    const M::Sounding snd = s.drive->sounding(*s.fluid);
    s.drive->start(*s.fluid, snd);
    const WeatherGrid &g = *s.grid;
    const int st = s.fluid->stride();
    const int k = 3;
    const size_t at = g.index(0, 2, 5) * size_t(st) + size_t(s.fluid->scalar_offset(k, M::THETA));
    const size_t still = g.index(1, 4, 4) * size_t(st) + size_t(s.fluid->scalar_offset(k, M::THETA));
    const double eq = s.fluid->state()[at];
    const double eq_still = s.fluid->state()[still];
    const size_t at_qv = g.index(0, 2, 5) * size_t(st) + size_t(s.fluid->scalar_offset(k, M::QV));
    const double eq_qv = s.fluid->state()[at_qv];
    s.fluid->state()[at] = float(eq + 1.0);
    s.fluid->state()[at_qv] = float(eq_qv + 1e-3);
    const double span = F::Params().relax_days * F::Params().day_s;
    for (int t = 0; t < 1000; t++) {
        s.drive->relax(*s.fluid, snd, span / 1000.0);
    }
    const double left = s.fluid->state()[at] - eq;
    MESSAGE("1 K over its equilibrium, relax_days on: ", left, " K left");
    CHECK(left == doctest::Approx(std::exp(-1.0)).epsilon(0.01));
    CHECK(s.fluid->state()[still] == doctest::Approx(eq_still).epsilon(1e-6));
    CHECK((s.fluid->state()[at_qv] - eq_qv) / 1e-3 == doctest::Approx(std::exp(-1.0)).epsilon(0.01));
}

// The boundary layer follows its ground, the free atmosphere is the planet's: a warm column's air is warmer under the
// inversion, the same over it.
TEST_CASE("WeatherForcing::level_equilibrium: the boundary layer follows its ground, the air aloft is the planet's") {
    Setup s = make(8, 299.0f, 0.0f, F::Params(), WeatherLevels::for_top(3000.0, 45.0, 1.12, WeatherFluid::MAX_LEVELS));
    const M::Sounding snd = s.drive->sounding(*s.fluid);
    const double inv = F::Params().inversion_m;
    for (int k = 0; k < s.fluid->levels(); k++) {
        double theta, qv;
        s.drive->level_equilibrium(snd, k, 2.0, 1.1, theta, qv);
        const double z = snd.z[size_t(k)];
        if (z < inv) {
            CHECK(theta == doctest::Approx(snd.theta[size_t(k)] + 2.0));
            CHECK(qv == doctest::Approx(snd.qv[size_t(k)] * 1.1));
        } else if (z > inv + F::INVERSION_DEPTH_M) {
            CHECK(theta == doctest::Approx(snd.theta[size_t(k)]));
            CHECK(qv == doctest::Approx(snd.qv[size_t(k)]));
        }
    }
}

// The typhoon benchmark's seed: a cyclone turns anticlockwise seen from above in the north, clockwise in the south, its
// wind vmax at its radius on the lowest level.
TEST_CASE("WeatherForcing::seed_vortex: a cyclone each side of the equator, vmax at its radius") {
    for (const double lat : { 20.0, -20.0 }) {
        CAPTURE(lat);
        Setup s = make(32, 299.0f, 0.0f);
        const WeatherGrid &g = *s.grid;
        const double r = 1000.0, a = lat * 3.14159265358979323846 / 180.0;
        const double centre[3] = { r * std::cos(a), r * std::sin(a), 0.0 };
        const double up[3] = { std::cos(a), std::sin(a), 0.0 };
        F::seed_vortex(*s.fluid, centre, 5.0, 150.0);
        double spin = 0.0, strongest = 0.0, at_m = 0.0;
        for (size_t in = 0; in < g.cells(); in++) {
            const int f = int(in / (size_t(g.width()) * size_t(g.height())));
            const int j = int((in / size_t(g.width())) % size_t(g.height()));
            const int i = int(in % size_t(g.width()));
            double pos[3];
            s.fluid->column_pos(in, pos);
            const float *w = &s.fluid->state()[g.index(f, i, j) * size_t(s.fluid->stride()) + size_t(s.fluid->wind_offset(0))];
            const double d[3] = { pos[0] - centre[0], pos[1] - centre[1], pos[2] - centre[2] };
            // (d x w) . up: positive anticlockwise round up.
            spin += ((d[1] * w[2] - d[2] * w[1]) * up[0] + (d[2] * w[0] - d[0] * w[2]) * up[1] + (d[0] * w[1] - d[1] * w[0]) * up[2]);
            const double speed = std::sqrt(double(w[0]) * w[0] + double(w[1]) * w[1] + double(w[2]) * w[2]);
            if (speed > strongest) {
                strongest = speed;
                at_m = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            }
        }
        MESSAGE("lat ", lat, ": strongest ", strongest, " m/s at ", at_m, " m");
        CHECK((spin > 0.0) == (lat > 0.0));
        CHECK(strongest == doctest::Approx(5.0 * (1.0 - s.fluid->level_m(0) / s.fluid->top_m())).epsilon(0.1));
        CHECK(at_m == doctest::Approx(150.0).epsilon(0.3));
    }
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

// The forcing's whole start over a sea 1.5 K warmer than its air (a planet of trade-wind sea, the same at every latitude:
// sun_k 0; the Coriolis on), sampled each minute from 15 to 40 (the convection comes in bursts while the layer adjusts:
// one instant says little): cumulus on average, broken, their base above the ground, nothing over the inversion.
TEST_CASE("WeatherForcing: a trade-wind sea makes broken cumulus") {
    F::Params uniform;
    uniform.sun_k = 0.0;
    Setup s = make(32, 300.5f, 0.0f, uniform);
    const M::Sounding snd = s.drive->sounding(*s.fluid);
    WeatherMoist moist(*s.fluid, snd, M::Params());
    s.drive->start(*s.fluid, snd);
    std::vector<M::Ground> ground;
    s.drive->grounds(ground);
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
