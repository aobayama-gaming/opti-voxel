#include "test_common.hpp"

#include "weather_fluid.hpp"
#include "weather_moist.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// BOMEX (Siebesma et al. 2003, "A large eddy simulation intercomparison study of shallow cumulus convection", JAS 60,
// appendix B, Table B1): trade-wind cumulus over a sea in a steady state, ten large-eddy simulations compared. The
// case's own forcings on the model's own fluid and Kessler microphysics (docs/weather.md "Benchmarks"): its start
// profiles, surface fluxes, subsidence, radiative cooling and large-scale drying, prescribed as every model of the
// intercomparison had them. Its results (hours 3-6): the profiles hold (tendencies ~1 K and 1 g/kg a day), the cloud
// fraction peaks at ~0.06 near the ~500 m cloud base and falls with height to the inversion at ~1500 m, the total
// cloud cover ~0.13 (10-15% over the models).
// Not the case's: a 2 km planet (~98 m cells, 64 levels of 47 m to 3 km; the case 100 m and 40 m) instead of a 6.4 km
// square, no mean wind, no Coriolis (its geostrophic wind and u* need a flat domain), the ground's pressure 1000 hPa
// (1015). Slow (minutes): runs only when a filter names it, `tools/unit_test.sh "Slow: BOMEX*"`.

namespace {

// Table B1's piecewise-linear profiles: a value at each height, linear between.
double table(const double (*p_points)[2], int p_count, double p_z) {
    if (p_z <= p_points[0][0]) {
        return p_points[0][1];
    }
    for (int n = 1; n < p_count; n++) {
        if (p_z <= p_points[n][0]) {
            const double t = (p_z - p_points[n - 1][0]) / (p_points[n][0] - p_points[n - 1][0]);
            return p_points[n - 1][1] + t * (p_points[n][1] - p_points[n - 1][1]);
        }
    }
    return p_points[p_count - 1][1];
}

constexpr double THETA_L[][2] = { { 0, 298.7 }, { 520, 298.7 }, { 1480, 302.4 }, { 2000, 308.2 }, { 3000, 311.85 } };
constexpr double Q_T[][2] = { { 0, 17.0e-3 }, { 520, 16.3e-3 }, { 1480, 10.7e-3 }, { 2000, 4.2e-3 }, { 3000, 3.0e-3 } };
constexpr double SUBSIDENCE[][2] = { { 0, 0.0 }, { 1500, -0.0065 }, { 2100, 0.0 } }; // m/s
constexpr double RADIATION[][2] = { { 0, -2.0 / 86400.0 }, { 1500, -2.0 / 86400.0 }, { 3000, 0.0 } }; // K/s
constexpr double DRYING[][2] = { { 0, -1.2e-8 }, { 300, -1.2e-8 }, { 500, 0.0 } }; // 1/s
constexpr double THETA_FLUX = 8e-3; // K m/s
constexpr double QT_FLUX = 5.2e-5; // m/s

uint32_t hash32(uint32_t p_x) {
    p_x ^= p_x >> 16;
    p_x *= 0x7feb352du;
    p_x ^= p_x >> 15;
    p_x *= 0x846ca68bu;
    p_x ^= p_x >> 16;
    return p_x;
}

double signed_unit(uint32_t p_key) {
    return double(hash32(p_key) & 0xffffu) / 65535.0 * 2.0 - 1.0;
}

struct Profiles {
    std::vector<double> cloud; // the share of the cells holding cloud water, per level
    std::vector<double> theta_l, q_t; // the means
    double cover = 0.0; // the share of the columns holding cloud water at any level
};

// The case's forcings on one column for p_dt: the surface fluxes into the lowest level, the subsidence (upwind, from
// above), the radiation and the drying, then Kessler; the buoyancy against the sounding.
void force_column(WeatherFluid &p_fluid, const WeatherMoist::Sounding &p_sound, size_t p_in, float *p_here, double p_dt, std::vector<float> &r_b) {
    constexpr int MAX = WeatherFluid::MAX_LEVELS;
    const int levels = p_fluid.levels();
    const double dz = p_fluid.thick_m(0); // the surface fluxes go into the lowest level
    double col[WeatherMoist::SCALARS][MAX];
    for (int k = 0; k < levels; k++) {
        for (int s = 0; s < WeatherMoist::SCALARS; s++) {
            col[s][k] = p_here[p_fluid.scalar_offset(k, s)];
        }
    }
    col[WeatherMoist::THETA][0] += THETA_FLUX * p_dt / dz;
    col[WeatherMoist::QV][0] += QT_FLUX * p_dt / dz;
    for (int k = 0; k + 1 < levels; k++) {
        const double a = -table(SUBSIDENCE, 3, p_sound.z[size_t(k)]) * p_dt / (p_sound.z[size_t(k + 1)] - p_sound.z[size_t(k)]);
        for (const int s : { int(WeatherMoist::THETA), int(WeatherMoist::QV), int(WeatherMoist::QC) }) {
            col[s][k] += a * (col[s][k + 1] - col[s][k]);
        }
    }
    for (int k = 0; k < levels; k++) {
        col[WeatherMoist::THETA][k] += table(RADIATION, 3, p_sound.z[size_t(k)]) * p_dt;
        col[WeatherMoist::QV][k] = ov::max(col[WeatherMoist::QV][k] + table(DRYING, 3, p_sound.z[size_t(k)]) * p_dt, 0.0);
    }
    WeatherMoist::kessler(col[WeatherMoist::THETA], col[WeatherMoist::QV], col[WeatherMoist::QC], col[WeatherMoist::QR], p_sound.rho.data(),
            p_sound.exner.data(), p_sound.z.data(), levels, p_dt);
    for (int k = 0; k < levels; k++) {
        for (int s = 0; s < WeatherMoist::SCALARS; s++) {
            p_here[p_fluid.scalar_offset(k, s)] = float(col[s][k]);
        }
        r_b[p_fluid.cell_index(p_in, k)] = float(WeatherMoist::buoyancy(col[WeatherMoist::THETA][k], col[WeatherMoist::QV][k], col[WeatherMoist::QC][k],
                col[WeatherMoist::QR][k], double(float(p_sound.theta[size_t(k)])), double(float(p_sound.qv[size_t(k)]))));
    }
}

void accumulate(const WeatherFluid &p_fluid, const WeatherMoist::Sounding &p_sound, Profiles &r_sum) {
    const WeatherGrid &g = p_fluid.grid();
    const int levels = p_fluid.levels();
    const double n = double(g.cells());
    r_sum.cloud.resize(size_t(levels), 0.0);
    r_sum.theta_l.resize(size_t(levels), 0.0);
    r_sum.q_t.resize(size_t(levels), 0.0);
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const float *here = &p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride())];
                bool any = false;
                for (int k = 0; k < levels; k++) {
                    const double qc = here[p_fluid.scalar_offset(k, WeatherMoist::QC)];
                    const double theta = here[p_fluid.scalar_offset(k, WeatherMoist::THETA)];
                    const double qv = here[p_fluid.scalar_offset(k, WeatherMoist::QV)];
                    r_sum.cloud[size_t(k)] += qc > 0.0 ? 1.0 / n : 0.0;
                    r_sum.theta_l[size_t(k)] += (theta - WeatherMoist::LATENT / (WeatherMoist::CP * p_sound.exner[size_t(k)]) * qc) / n;
                    r_sum.q_t[size_t(k)] += (qv + qc) / n;
                    any = any || qc > 0.0;
                }
                r_sum.cover += any ? 1.0 / n : 0.0;
            }
        }
    }
}

} // namespace

TEST_CASE("Slow: BOMEX trade-wind cumulus (Siebesma et al. 2003)" * doctest::skip()) {
    const auto grid = std::make_shared<CubeSphereGrid>(32, 2000.0);
    WeatherFluid::Params fp;
    fp.levels = 64;
    fp.top_m = 3000.0;
    fp.scalars = WeatherMoist::SCALARS;
    WeatherFluid fluid(grid, fp);
    const WeatherGrid &g = fluid.grid();
    const int levels = fluid.levels();
    std::vector<double> theta0(size_t(levels), 0.0), qv0(size_t(levels), 0.0);
    for (int k = 0; k < levels; k++) {
        theta0[size_t(k)] = table(THETA_L, 5, fluid.level_m(k));
        qv0[size_t(k)] = table(Q_T, 5, fluid.level_m(k));
    }
    const WeatherMoist::Sounding sound = WeatherMoist::sounding(fluid, theta0, qv0);
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const size_t in = g.interior(f, i, j);
                float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                for (int k = 0; k < levels; k++) {
                    const bool low = fluid.level_m(k) < 1600.0; // the case's lowest 40 levels
                    const uint32_t key = uint32_t(in * 128 + size_t(k)) * 2u;
                    here[fluid.scalar_offset(k, WeatherMoist::THETA)] = float(theta0[size_t(k)] + (low ? 0.1 * signed_unit(key) : 0.0));
                    here[fluid.scalar_offset(k, WeatherMoist::QV)] = float(qv0[size_t(k)] + (low ? 0.025e-3 * signed_unit(key + 1u) : 0.0));
                }
            }
        }
    }
    fluid.vertical_from_centres();
    std::vector<float> b(fluid.cell_count(), 0.0f);
    Profiles at3, mean, at6;
    int samples = 0;
    double t = 0.0, next_sample = 3.0 * 3600.0;
    const double end_s = 6.0 * 3600.0;
    while (t < end_s) {
        const double dt = std::min(5.0, fluid.max_dt());
        fluid.advect(dt);
        for (int f = 0; f < g.faces(); f++) {
            for (int j = 0; j < g.height(); j++) {
                for (int i = 0; i < g.width(); i++) {
                    force_column(fluid, sound, g.interior(f, i, j), &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())], dt, b);
                }
            }
        }
        fluid.force(dt, b.data());
        fluid.project(dt);
        t += dt;
        if (t >= next_sample) {
            next_sample += 60.0;
            if (samples == 0) {
                accumulate(fluid, sound, at3);
            }
            accumulate(fluid, sound, mean);
            samples++;
        }
    }
    accumulate(fluid, sound, at6);
    CHECK_OR_RETURN(samples > 0);
    double peak = 0.0, peak_z = 0.0, drift_theta = 0.0, drift_qt = 0.0, above = 0.0;
    ov::str table_text = ov::format("hours 3-6, %d samples: total cover %.3f (the models ~0.13); z, cloud fraction, theta_l (K) and q_t (g/kg) at "
                                    "3 h -> 6 h:",
            samples, mean.cover / samples);
    for (int k = 0; k < levels; k++) {
        const double z = fluid.level_m(k);
        const double c = mean.cloud[size_t(k)] / samples;
        if (c > peak) {
            peak = c;
            peak_z = z;
        }
        if (z < 2500.0) {
            drift_theta = ov::max(drift_theta, std::fabs(at6.theta_l[size_t(k)] - at3.theta_l[size_t(k)]));
            drift_qt = ov::max(drift_qt, std::fabs(at6.q_t[size_t(k)] - at3.q_t[size_t(k)]));
        }
        if (z > 2000.0) {
            above = ov::max(above, c);
        }
        if (k % 4 == 0 || c == peak) {
            table_text += ov::format("\n  %6.0f  %.4f  %.2f -> %.2f  %.2f -> %.2f", z, c, at3.theta_l[size_t(k)], at6.theta_l[size_t(k)], at3.q_t[size_t(k)] * 1000.0,
                    at6.q_t[size_t(k)] * 1000.0);
        }
    }
    MESSAGE(table_text);
    MESSAGE("cloud fraction peak ", peak, " at ", peak_z, " m (the models ~0.06 near 500 m); drift over 3 h: theta_l ", drift_theta, " K, q_t ",
            drift_qt * 1000.0, " g/kg");
    CHECK(peak > 0.03);
    CHECK(peak < 0.12);
    CHECK(peak_z > 400.0);
    CHECK(peak_z < 800.0);
    CHECK(mean.cover / samples > 0.06);
    CHECK(mean.cover / samples < 0.3);
    CHECK(above < 0.002); // capped by the inversion
    // "The total tendencies for temperature and specific humidity over the last 3 h are typically of the order 1 K/day
    // and 1 g/kg/day": twice that over the 3 hours.
    CHECK(drift_theta < 0.25);
    CHECK(drift_qt < 0.25e-3);
}
