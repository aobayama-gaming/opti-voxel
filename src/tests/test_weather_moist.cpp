#include "test_common.hpp"

#include "weather_moist.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// The weather's moist physics (WeatherMoist, docs/weather.md "Moist physics"): its schemes against hand values, then
// the fluid and the physics together on the reference cases: trade-wind cumulus over a warm sea are broken (clear air
// between them, a base, a top under the inversion), moist air over cold ground makes fog low down.

namespace {

using M = WeatherMoist;

WeatherFluid::Params moist_params() {
    WeatherFluid::Params p;
    p.levels = 15;
    p.top_m = 600.0;
    p.scalars = M::SCALARS;
    return p;
}

uint32_t hash4(uint32_t p_a, uint32_t p_b, uint32_t p_c, uint32_t p_d) {
    uint32_t h = p_a * 0x9E3779B1u ^ (p_b + 0x7F4A7C15u) * 0x85EBCA77u ^ (p_c + 0x165667B1u) * 0xC2B2AE3Du ^ (p_d + 0x27D4EB2Fu) * 0x27D4EB2Fu;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

double noise(uint32_t p_a, uint32_t p_b, uint32_t p_c, uint32_t p_d) {
    return double(hash4(p_a, p_b, p_c, p_d) & 0xFFFFu) / 65535.0 * 2.0 - 1.0;
}

// The sounding everywhere, a little noise on the two lowest levels (the LES cases' trigger), the wind p_wind_mps east
// on face 0's x.
void set_air(WeatherFluid &p_fluid, const M::Sounding &p_s, double p_theta_noise, double p_qv_noise) {
    const WeatherGrid &g = p_fluid.grid();
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                float *here = &p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride())];
                for (int k = 0; k < p_fluid.levels(); k++) {
                    const double n = k < 2 ? 1.0 : 0.0;
                    here[p_fluid.scalar_offset(k, M::THETA)] = float(p_s.theta[size_t(k)] + n * p_theta_noise * noise(f, i, j, k));
                    here[p_fluid.scalar_offset(k, M::QV)] = float(p_s.qv[size_t(k)] + n * p_qv_noise * noise(f, i, j, k + 100));
                }
            }
        }
    }
    p_fluid.fill_halo();
}

// p_seconds of fluid and physics, in steps of at most 5 s (and max_dt).
void run(WeatherFluid &p_fluid, WeatherMoist &p_moist, const std::vector<M::Ground> &p_ground, double p_seconds) {
    std::vector<float> b(p_fluid.cell_count());
    double left = p_seconds;
    while (left > 0.0) {
        const double dt = std::min({ left, 5.0, p_fluid.max_dt() });
        p_fluid.advect(dt);
        p_moist.apply(p_fluid, p_ground, dt, b.data());
        p_fluid.force(dt, b.data());
        p_fluid.project(dt);
        left -= dt;
    }
}

// The share of columns whose cloud water at a level passes 0.01 g/kg (the LES cases' cloudy cell).
double cloud_fraction(const WeatherFluid &p_fluid, int p_level) {
    const WeatherGrid &g = p_fluid.grid();
    size_t cloudy = 0;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                cloudy += p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride()) + size_t(p_fluid.scalar_offset(p_level, M::QC))] > 1e-5f;
            }
        }
    }
    return double(cloudy) / double(g.cells());
}

} // namespace

TEST_CASE("WeatherMoist: the saturation is Tetens', 14.7 g/kg at 20 C and 1000 mb") {
    CHECK(M::saturation(293.15, 1000.0) == doctest::Approx(0.0147).epsilon(0.02));
    CHECK(M::saturation(303.15, 1000.0) == doctest::Approx(0.0276).epsilon(0.03));
    CHECK(M::saturation(273.15, 1000.0) == doctest::Approx(0.00379).epsilon(0.02));
}

TEST_CASE("WeatherMoist: the saturation adjustment keeps the water and the liquid potential temperature") {
    const int n = 15;
    std::vector<double> theta(n, 300.0), qv(n, 0.025), qc(n, 0.0), qr(n, 0.0), rho(n), exner(n), z(n);
    for (int k = 0; k < n; k++) {
        z[size_t(k)] = 20.0 + 40.0 * k;
        exner[size_t(k)] = 1.0 - M::GRAVITY / (M::CP * 300.0) * z[size_t(k)];
        rho[size_t(k)] = M::P0_MB * 100.0 * std::pow(exner[size_t(k)], 1.0 / M::KAPPA_EXNER) / (M::R_D * 300.0 * exner[size_t(k)]);
    }
    M::kessler(theta.data(), qv.data(), qc.data(), qr.data(), rho.data(), exner.data(), z.data(), n, 0.1);
    for (int k = 0; k < n; k++) {
        const size_t s = size_t(k);
        CHECK(qv[s] + qc[s] + qr[s] == doctest::Approx(0.025).epsilon(1e-4));
        CHECK(theta[s] - M::LATENT / (M::CP * exner[s]) * (qc[s] + qr[s]) == doctest::Approx(300.0).epsilon(1e-5));
        CHECK(qc[s] > 0.0);
        // One linearised step lands within a few percent of the saturation at the warmed temperature.
        CHECK(qv[s] == doctest::Approx(M::saturation(theta[s] * exner[s], M::P0_MB * std::pow(exner[s], 1.0 / M::KAPPA_EXNER))).epsilon(0.05));
    }
}

TEST_CASE("WeatherMoist: rain forms past 1 g/kg of cloud, falls out, and the column's water balances the rain") {
    const int n = 15;
    std::vector<double> theta(n, 300.0), qv(n), qc(n, 0.0), qr(n, 0.0), rho(n), exner(n), z(n);
    double before = 0.0;
    for (int k = 0; k < n; k++) {
        const size_t s = size_t(k);
        z[s] = 20.0 + 40.0 * k;
        exner[s] = 1.0 - M::GRAVITY / (M::CP * 300.0) * z[s];
        rho[s] = M::P0_MB * 100.0 * std::pow(exner[s], 1.0 / M::KAPPA_EXNER) / (M::R_D * 300.0 * exner[s]);
        qv[s] = M::saturation(theta[s] * exner[s], M::P0_MB * std::pow(exner[s], 1.0 / M::KAPPA_EXNER));
        qc[s] = k >= 5 && k <= 9 ? 0.003 : 0.0;
        before += rho[s] * (qv[s] + qc[s] + qr[s]) * 40.0;
    }
    double rained = 0.0; // kg/m^2
    for (int t = 0; t < 360; t++) { // 30 min
        rained += M::kessler(theta.data(), qv.data(), qc.data(), qr.data(), rho.data(), exner.data(), z.data(), n, 5.0) * 1000.0 * 5.0;
    }
    double after = 0.0;
    for (int k = 0; k < n; k++) {
        after += rho[size_t(k)] * (qv[size_t(k)] + qc[size_t(k)] + qr[size_t(k)]) * 40.0;
    }
    MESSAGE("column water ", before, " -> ", after, " kg/m^2, rained ", rained);
    CHECK(rained > 0.05);
    CHECK(before - after == doctest::Approx(rained).epsilon(0.1));
}

TEST_CASE("WeatherMoist: the longwave cools a cloud's top, nothing in clear air") {
    const int n = 15;
    std::vector<double> qc(n, 0.0), rho(n, 1.15), exner(n, 1.0), heating(n), thick(n, 40.0);
    for (int k = 5; k <= 8; k++) {
        qc[size_t(k)] = 0.0005;
    }
    M::longwave(qc.data(), rho.data(), exner.data(), thick.data(), n, heating.data());
    MESSAGE("cloud top ", heating[8] * 3600.0, " K/h, cloud base ", heating[5] * 3600.0, " K/h");
    CHECK(heating[8] * 3600.0 < -2.0); // DYCOMS' ~-4.7 K/h over the top 40 m
    CHECK(heating[5] > heating[8]);
    for (const int k : { 0, 3, 10, 14 }) {
        CHECK(std::fabs(heating[size_t(k)]) < 1e-12);
    }
}

// A BOMEX-like trade-wind boundary layer (Siebesma et al. 2003) fitted under the game's 600 m: a mixed layer at 298 K
// and 18 g/kg (91% humidity: a base near 160 m), a conditionally unstable cloud layer, an inversion of 4 K from 300 to
// 340 m, dry and stable above; a sea 1.5 K warmer under it. After 20 minutes: cumulus with clear air between them.
TEST_CASE("WeatherMoist: trade-wind cumulus over a warm sea are broken, with a base and a top") {
    const auto grid = std::make_shared<CubeSphereGrid>(32, 1000.0); // cells ~49 m
    WeatherFluid fluid(grid, moist_params());
    std::vector<double> theta, qv;
    for (int k = 0; k < fluid.levels(); k++) {
        const double z = fluid.level_m(k);
        theta.push_back(z < 160.0 ? 298.0 : z < 300.0 ? 298.0 + 0.004 * (z - 160.0)
                        : z < 340.0                   ? 298.56 + 0.1 * (z - 300.0)
                                                      : 302.56 + 0.003 * (z - 340.0));
        qv.push_back(z < 160.0 ? 0.018 : z < 300.0 ? 0.018 - 1.4e-5 * (z - 160.0)
                        : z < 340.0                ? 0.016 - 2e-4 * (z - 300.0)
                                                   : 0.008);
    }
    const M::Sounding s = M::sounding(fluid, theta, qv);
    WeatherMoist moist(fluid, s, M::Params());
    set_air(fluid, s, 0.1, 2.5e-5);
    std::vector<M::Ground> sea(grid->cells());
    for (M::Ground &gr : sea) {
        gr.temp_k = 299.5f;
    }
    // Each level's cloudy share averaged over a minute apart from 15 to 25 minutes (one snapshot is one cloud's life).
    run(fluid, moist, sea, 840.0);
    std::vector<double> mean(size_t(fluid.levels()), 0.0);
    double top_level = 0.0, ground_level = 0.0;
    const int samples = 11;
    for (int n = 0; n < samples; n++) {
        run(fluid, moist, sea, 60.0);
        for (int k = 0; k < fluid.levels(); k++) {
            mean[size_t(k)] += cloud_fraction(fluid, k) / samples;
        }
        ground_level = std::max(ground_level, cloud_fraction(fluid, 0));
        top_level = std::max(top_level, cloud_fraction(fluid, fluid.levels() - 1));
    }
    double most = 0.0;
    int most_k = 0;
    for (int k = 0; k < fluid.levels(); k++) {
        MESSAGE("level ", k, " (", fluid.level_m(k), " m): cloud ", mean[size_t(k)]);
        if (mean[size_t(k)] > most) {
            most = mean[size_t(k)];
            most_k = k;
        }
    }
    CHECK(most > 0.005); // cumulus formed (averaged over 10 minutes: 1.6% of a level at most)
    CHECK(most < 0.6); // and they are broken
    CHECK(fluid.level_m(most_k) > 100.0);
    CHECK(ground_level == 0.0); // a base above the ground
    CHECK(top_level == 0.0); // a top under the lid
}

// Moist air (98% humidity, its dew point ~6 K over the ground) over ground 6 K colder (sea fog over a cold current):
// the lowest level, cooled and losing vapour to the ground at the same rate, crosses saturation (the mixing line between
// the air and the ground's saturated state bows over the saturation curve near the ground's end); the fog
// stays low.
TEST_CASE("WeatherMoist: moist air over cold ground makes fog low down") {
    const auto grid = std::make_shared<CubeSphereGrid>(16, 1000.0);
    WeatherFluid fluid(grid, moist_params());
    std::vector<double> theta, qv;
    for (int k = 0; k < fluid.levels(); k++) {
        const double z = fluid.level_m(k);
        theta.push_back(290.0 + 0.004 * z);
        qv.push_back(0.98 * M::saturation(290.0, M::P0_MB) * (z < 200.0 ? 1.0 : 0.7));
    }
    const M::Sounding s = M::sounding(fluid, theta, qv);
    WeatherMoist moist(fluid, s, M::Params());
    set_air(fluid, s, 0.0, 0.0);
    std::vector<M::Ground> cold(grid->cells());
    for (M::Ground &gr : cold) {
        gr.temp_k = 284.0f;
    }
    run(fluid, moist, cold, 2400.0);
    const double low = cloud_fraction(fluid, 0);
    MESSAGE("fog: lowest level ", low, ", level 4 ", cloud_fraction(fluid, 4));
    CHECK(low > 0.8);
    CHECK(cloud_fraction(fluid, 4) == 0.0);
}
