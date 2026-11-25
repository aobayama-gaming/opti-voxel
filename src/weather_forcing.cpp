#include "weather_forcing.hpp"

#include "parallel_for.hpp"
#include "weather_vec.hpp"

#include "ov/math.hpp"

#include <cmath>

namespace {

using namespace weather_vec;

uint32_t hash3(uint32_t p_a, uint32_t p_b, uint32_t p_c) {
    uint32_t h = p_a * 0x9E3779B1u ^ (p_b + 0x7F4A7C15u) * 0x85EBCA77u ^ (p_c + 0x165667B1u) * 0xC2B2AE3Du;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

double unit(uint32_t p_h) {
    return double(p_h >> 8) / double(1u << 24);
}

} // namespace

void WeatherForcing::prevailing(const double p_up[3], double p_speed, double r_wind[3]) {
    const double pole[3] = { 0.0, 1.0, 0.0 };
    double east[3];
    cross3(pole, p_up, east);
    const double len = std::sqrt(dot3(east, east));
    const double lat = std::asin(ov::clamp(p_up[1], -1.0, 1.0));
    const double s = len > 1e-9 ? -std::sin(6.0 * std::fabs(lat)) * p_speed / len : 0.0;
    for (int k = 0; k < 3; k++) {
        r_wind[k] = east[k] * s;
    }
}

WeatherForcing::WeatherForcing(const WeatherFluid &p_fluid, const Params &p_params, std::vector<Climate::BiomeWeather> p_climate) :
        params(p_params), climate(std::move(p_climate)) {
    const size_t n = p_fluid.grid().cells();
    climate.resize(n);
    ups.resize(n * 3);
    double area = 0.0, ground = 0.0;
    for (size_t c = 0; c < n; c++) {
        const double a = p_fluid.column_area_m2(c);
        area += a;
        ground += a * double(climate[c].ground_k);
        any_storms = any_storms || climate[c].storms > 0.0f;
        p_fluid.column_up(c, &ups[c * 3]);
    }
    if (area > 0.0) {
        mean_ground_k = ground / area;
    }
    if (!(params.storm_m > 0.0)) {
        double pos[3];
        p_fluid.column_pos(0, pos);
        params.storm_m = std::sqrt(dot3(pos, pos)) / 8.0;
    }
}

WeatherMoist::Sounding WeatherForcing::sounding(const WeatherFluid &p_fluid) const {
    const double theta0 = mean_ground_k - AIR_BELOW_GROUND_K;
    const double inv = params.inversion_m;
    const double base = CLOUD_BASE * inv;
    // The mixed layer's vapour saturates a parcel lifted from the ground at the cloud layer's base (its lifting
    // condensation level: BOMEX's cloud base is its mixed layer's top).
    const double exner_base = 1.0 - WeatherMoist::GRAVITY / (WeatherMoist::CP * theta0) * base;
    const double q0 = WeatherMoist::saturation(theta0 * exner_base, WeatherMoist::P0_MB * std::pow(exner_base, 1.0 / WeatherMoist::KAPPA_EXNER));
    const double top = inv + INVERSION_DEPTH_M;
    const double theta_inv = theta0 + CLOUD_LAPSE_K_PER_M * (inv - base);
    const double q_inv = q0 - CLOUD_DRYING_PER_M * (inv - base);
    std::vector<double> theta, qv;
    for (int k = 0; k < p_fluid.levels(); k++) {
        const double z = p_fluid.level_m(k);
        if (z < base) {
            theta.push_back(theta0);
            qv.push_back(q0);
        } else if (z < inv) {
            theta.push_back(theta0 + CLOUD_LAPSE_K_PER_M * (z - base));
            qv.push_back(q0 - CLOUD_DRYING_PER_M * (z - base));
        } else if (z < top) {
            const double t = (z - inv) / INVERSION_DEPTH_M;
            theta.push_back(theta_inv + INVERSION_K * t);
            qv.push_back(q_inv + (ABOVE_HUMIDITY * q0 - q_inv) * t);
        } else {
            theta.push_back(theta_inv + INVERSION_K + ABOVE_LAPSE_K_PER_M * (z - top));
            qv.push_back(ABOVE_HUMIDITY * q0);
        }
    }
    return WeatherMoist::sounding(p_fluid, theta, qv);
}

void WeatherForcing::start(WeatherFluid &p_fluid, const WeatherMoist::Sounding &p_sounding) const {
    const WeatherGrid &g = p_fluid.grid();
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const size_t in = g.interior(f, i, j);
                float *here = &p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride())];
                double up[3], wind[3];
                p_fluid.column_up(in, up);
                prevailing(up, params.prevailing_mps, wind);
                for (int k = 0; k < p_fluid.levels(); k++) {
                    for (int a = 0; a < 3; a++) {
                        here[p_fluid.wind_offset(k) + a] = float(wind[a]);
                    }
                    const double noise = k < 2 ? NOISE_K * (2.0 * unit(hash3(params.seed, uint32_t(in), uint32_t(k))) - 1.0) : 0.0;
                    here[p_fluid.scalar_offset(k, WeatherMoist::THETA)] = float(p_sounding.theta[size_t(k)] + noise);
                    here[p_fluid.scalar_offset(k, WeatherMoist::QV)] = float(p_sounding.qv[size_t(k)]);
                    here[p_fluid.scalar_offset(k, WeatherMoist::QC)] = 0.0f;
                    here[p_fluid.scalar_offset(k, WeatherMoist::QR)] = 0.0f;
                }
            }
        }
    }
    p_fluid.fill_halo();
}

void WeatherForcing::grounds(const Float3 &p_towards_sun, std::vector<WeatherMoist::Ground> &r_ground) const {
    r_ground.resize(climate.size());
    const double sun[3] = { double(p_towards_sun.x), double(p_towards_sun.y), double(p_towards_sun.z) };
    constexpr double MEAN_DAY = 1.0 / 3.14159265358979323846; // max(cos, 0)'s mean over a day at the equator
    for (size_t c = 0; c < climate.size(); c++) {
        const double light = ov::max(dot3(&ups[c * 3], sun), 0.0);
        r_ground[c].temp_k = float(double(climate[c].ground_k) + double(climate[c].diurnal_k) * (light - MEAN_DAY));
        r_ground[c].wetness = climate[c].wetness;
        r_ground[c].z0_m = climate[c].z0_m;
    }
}

void WeatherForcing::apply(WeatherFluid &p_fluid, double p_dt, uint64_t p_step) {
    const WeatherGrid &g = p_fluid.grid();
    const int levels = p_fluid.levels();
    const int st = p_fluid.stride();
    const double omega = this->omega();
    std::vector<float> &state = p_fluid.state();
    // The large-scale pressure gradient: f up x u_g on every level.
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherForcing::apply", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            double up[3], ug[3], push[3];
            p_fluid.column_up(g.interior(face, i, j), up);
            prevailing(up, params.prevailing_mps, ug);
            cross3(up, ug, push);
            const double f = 2.0 * omega * up[1];
            float *here = &state[g.index(face, i, j) * size_t(st)];
            for (int k = 0; k < levels; k++) {
                for (int a = 0; a < 3; a++) {
                    here[p_fluid.wind_offset(k) + a] += float(f * push[a] * p_dt);
                }
            }
        } }, ov::Priority::LOW);
    // The storms' seeds.
    if (!any_storms) {
        return;
    }
    const double per_column = p_dt / (STORM_PERIOD_S * double(climate.size()));
    for (size_t c = 0; c < climate.size(); c++) {
        const double chance = double(climate[c].storms) * per_column;
        if (chance > 0.0 && unit(hash3(params.seed ^ 0x5EEDu, uint32_t(p_step), uint32_t(c))) < chance) {
            double pos[3];
            p_fluid.column_pos(c, pos);
            seed_storm(p_fluid, pos);
            storms_started++;
        }
    }
}

ov::Value WeatherForcing::summary(const WeatherFluid &p_fluid) const {
    ov::Value out = ov::Value::object();
    out["mean_ground_k"] = mean_ground_k;
    for (int north = 0; north < 2; north++) {
        double area = 0.0, ground = 0.0, wet = 0.0, lo = 1e9, hi = -1e9;
        for (size_t c = 0; c < climate.size(); c++) {
            if ((ups[c * 3 + 1] > 0.0) != (north == 1)) {
                continue;
            }
            const double a = p_fluid.column_area_m2(c);
            area += a;
            ground += a * double(climate[c].ground_k);
            wet += climate[c].wetness >= 0.9f ? a : 0.0;
            lo = ov::min(lo, double(climate[c].ground_k));
            hi = ov::max(hi, double(climate[c].ground_k));
        }
        ov::Value h = ov::Value::object();
        h["ground_k"] = ov::format("%.1f / %.1f / %.1f", lo, area > 0.0 ? ground / area : 0.0, hi);
        h["wet_share"] = area > 0.0 ? wet / area : 0.0;
        out[north == 1 ? "north" : "south"] = h;
    }
    return out;
}

void WeatherForcing::seed_storm(WeatherFluid &p_fluid, const double p_pos[3]) const {
    const WeatherGrid &g = p_fluid.grid();
    const double r2 = params.storm_m * params.storm_m;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                double pos[3];
                p_fluid.column_pos(g.interior(f, i, j), pos);
                const double d[3] = { pos[0] - p_pos[0], pos[1] - p_pos[1], pos[2] - p_pos[2] };
                const double d2 = dot3(d, d);
                if (d2 >= r2) {
                    continue;
                }
                const double c = std::cos(0.5 * 3.14159265358979323846 * std::sqrt(d2 / r2));
                const double shape = c * c;
                float *here = &p_fluid.state()[g.index(f, i, j) * size_t(p_fluid.stride())];
                for (int k = 0; k < (p_fluid.levels() + 1) / 2; k++) {
                    here[p_fluid.scalar_offset(k, WeatherMoist::THETA)] += float(STORM_THETA_K * shape);
                    here[p_fluid.scalar_offset(k, WeatherMoist::QV)] += float(STORM_QV * shape);
                }
            }
        }
    }
    p_fluid.fill_halo();
}
