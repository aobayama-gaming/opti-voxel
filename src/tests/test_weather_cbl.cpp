#include "test_common.hpp"

#include "weather_fluid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// The dry convective boundary layer (docs/weather.md "Fluid"), the large-eddy simulations' first benchmark (Deardorff
// 1974, Schmidt & Schumann 1989, Moeng & Sullivan 1994): air heated from below at a fixed kinematic flux Q under an
// inversion at h makes thermals whose vertical velocity scales with w* = (g / theta Q h)^(1/3); the variance of w peaks
// at ~0.4 w*^2 near 0.35 h and the heat flux falls linearly to ~-0.2 Q at h (the entrainment). No rotation, no moisture,
// no kicks: the buoyancy alone must drive it.

namespace {

struct CBLStats {
    double w_star = 0.0;
    std::vector<double> w_rms; // per level, over the last third
    std::vector<double> flux; // the resolved w'theta' per level, over Q
};

uint32_t hash32(uint32_t p_x) {
    p_x ^= p_x >> 16;
    p_x *= 0x7feb352du;
    p_x ^= p_x >> 15;
    p_x *= 0x846ca68bu;
    p_x ^= p_x >> 16;
    return p_x;
}

// p_seconds of the benchmark on a 2 km planet of ~49 m cells and 50 m levels (the game's), the profiles over the
// last third.
CBLStats run_cbl(double p_seconds) {
    constexpr double H = 400.0, Q = 0.06, THETA = 300.0;
    const auto grid = std::make_shared<CubeSphereGrid>(32, 1000.0);
    WeatherFluid::Params fp;
    fp.levels = 16;
    fp.top_m = 800.0;
    fp.scalars = 1;
    WeatherFluid fluid(grid, fp);
    const WeatherGrid &g = fluid.grid();
    const int levels = fluid.levels();
    std::vector<double> ref(static_cast<size_t>(levels), 0.0);
    for (int k = 0; k < levels; k++) {
        const double z = fluid.level_m(k);
        ref[size_t(k)] = z < H ? THETA : THETA + 4.0 + 0.003 * (z - H);
    }
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                const size_t in = g.interior(f, i, j);
                for (int k = 0; k < levels; k++) {
                    const double noise = fluid.level_m(k) < H ? (double(hash32(uint32_t(in * 64 + size_t(k))) & 0xffff) / 65535.0 - 0.5) * 0.2 : 0.0;
                    here[fluid.scalar_offset(k, 0)] = float(ref[size_t(k)] + noise);
                }
            }
        }
    }
    fluid.vertical_from_centres();
    std::vector<float> b(fluid.cell_count());
    CBLStats out;
    out.w_star = std::cbrt(WeatherFluid::GRAVITY / THETA * Q * H);
    std::vector<double> w2(size_t(levels), 0.0), wt(size_t(levels), 0.0);
    int samples = 0;
    const double dz = fluid.dz_m();
    double t = 0.0, next_sample = p_seconds * 2.0 / 3.0;
    while (t < p_seconds) {
        const double dt = std::min(2.0, fluid.max_dt());
        fluid.advect(dt);
        for (int f = 0; f < g.faces(); f++) {
            for (int j = 0; j < g.height(); j++) {
                for (int i = 0; i < g.width(); i++) {
                    fluid.state()[g.index(f, i, j) * size_t(fluid.stride()) + size_t(fluid.scalar_offset(0, 0))] += float(Q * dt / dz);
                }
            }
        }
        fluid.dry_buoyancy(0, ref, b.data());
        fluid.force(dt, b.data());
        fluid.project(dt);
        t += dt;
        if (t < next_sample) {
            continue;
        }
        next_sample += 20.0;
        samples++;
        for (int k = 0; k < levels; k++) {
            double mean_w = 0.0, mean_t = 0.0;
            const double n = double(g.cells());
            std::vector<double> ws(g.cells()), ts(g.cells());
            for (int f = 0; f < g.faces(); f++) {
                for (int j = 0; j < g.height(); j++) {
                    for (int i = 0; i < g.width(); i++) {
                        const size_t in = g.interior(f, i, j);
                        const float *here = &fluid.state()[g.index(f, i, j) * size_t(fluid.stride())];
                        double up[3];
                        fluid.column_up(in, up);
                        const float *u = here + fluid.wind_offset(k);
                        ws[in] = u[0] * up[0] + u[1] * up[1] + u[2] * up[2];
                        ts[in] = here[fluid.scalar_offset(k, 0)];
                        mean_w += ws[in] / n;
                        mean_t += ts[in] / n;
                    }
                }
            }
            double v = 0.0, c = 0.0;
            for (size_t in = 0; in < g.cells(); in++) {
                v += (ws[in] - mean_w) * (ws[in] - mean_w) / n;
                c += (ws[in] - mean_w) * (ts[in] - mean_t) / n;
            }
            w2[size_t(k)] += v;
            wt[size_t(k)] += c;
        }
    }
    for (int k = 0; k < levels; k++) {
        out.w_rms.push_back(std::sqrt(w2[size_t(k)] / ov::max(samples, 1)));
        out.flux.push_back(wt[size_t(k)] / ov::max(samples, 1) / Q);
    }
    return out;
}

} // namespace

TEST_CASE("WeatherFluid: the dry convective boundary layer matches the large-eddy simulations' profiles") {
    const CBLStats s = run_cbl(1800.0);
    ov::str table = ov::format("w* %.3f m/s; z/h, w rms / w*, resolved w'theta' / Q:", s.w_star);
    double peak = 0.0, peak_z = 0.0, entrainment = 0.0;
    for (size_t k = 0; k < s.w_rms.size(); k++) {
        const double z = (double(k) + 0.5) * 50.0 / 400.0;
        table += ov::format("\n  %.2f  %.3f  %.3f", z, s.w_rms[k] / s.w_star, s.flux[k]);
        if (s.w_rms[k] / s.w_star > peak) {
            peak = s.w_rms[k] / s.w_star;
            peak_z = z;
        }
        entrainment = ov::min(entrainment, s.flux[k]);
    }
    MESSAGE(table);
    // Moeng & Sullivan 1994: w rms peaks at ~0.63 w* near 0.35 h, the heat flux reaches ~-0.2 Q at the inversion.
    CHECK(peak > 0.55);
    CHECK(peak < 0.75);
    CHECK(peak_z > 0.2);
    CHECK(peak_z < 0.5);
    CHECK(entrainment < -0.08);
    CHECK(s.w_rms[s.w_rms.size() - 1] / s.w_star < 0.05); // still air under the top
}
