#include "weather_sim.hpp"

#include "parallel_for.hpp"

#include "ov/math.hpp"

#include <cmath>

namespace {

constexpr int F = WeatherSim::FIELDS;
constexpr int D = WeatherSim::DIAGS;

double dot3(const double p_a[3], const double p_b[3]) {
    return p_a[0] * p_b[0] + p_a[1] * p_b[1] + p_a[2] * p_b[2];
}

void cross3(const double p_a[3], const double p_b[3], double r[3]) {
    r[0] = p_a[1] * p_b[2] - p_a[2] * p_b[1];
    r[1] = p_a[2] * p_b[0] - p_a[0] * p_b[2];
    r[2] = p_a[0] * p_b[1] - p_a[1] * p_b[0];
}

// p_v minus its component along the unit p_n.
void tangent(const double p_n[3], double r_v[3]) {
    const double d = dot3(p_n, r_v);
    for (int k = 0; k < 3; k++) {
        r_v[k] -= d * p_n[k];
    }
}

// The fraction of the way to its target a quantity relaxing over p_tau goes in p_dt.
double relax(double p_dt, double p_tau) {
    return 1.0 - std::exp(-p_dt / p_tau);
}

uint64_t mix64(uint64_t p_x) {
    p_x += 0x9E3779B97F4A7C15ull;
    p_x = (p_x ^ (p_x >> 30)) * 0xBF58476D1CE4E5B9ull;
    p_x = (p_x ^ (p_x >> 27)) * 0x94D049BB133111EBull;
    return p_x ^ (p_x >> 31);
}

} // namespace

WeatherSim::WeatherSim(const WeatherGrid &p_grid, const Params &p_params, const std::vector<Climate::BiomeWeather> &p_weather) :
        cells(p_grid), params(p_params) {
    const double storm_m = params.storm_m > 0.0 ? params.storm_m : 0.25 * params.radius_m;
    omega_rad_s = WAVE_MPS / (storm_m * std::sqrt(2.0));
    make_cells(p_weather);
    cur.assign(cells.padded_cells() * F, 0.0f);
    diag.assign(cells.padded_cells() * D, 0.0f);
    const int n = cells.size();
    for (int face = 0; face < WeatherGrid::FACES; face++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                const Cell &c = cell[interior(face, i, j)];
                float *s = &cur[cells.index(face, i, j) * F];
                s[WIND_X] = c.target[0];
                s[WIND_Y] = c.target[1];
                s[WIND_Z] = c.target[2];
                s[TEMP] = c.weather.heat;
                s[VAPOUR] = 0.9f * c.weather.moisture;
            }
        }
    }
    cells.fill_halo(cur.data(), F);
    next = cur;
}

size_t WeatherSim::interior(int p_face, int p_i, int p_j) const {
    const size_t n = size_t(cells.size());
    return (size_t(p_face) * n + size_t(p_j)) * n + size_t(p_i);
}

// Each cell's up, its central differences' dual vectors and its forcing; the substep and the diffusion from the smallest
// spacing.
void WeatherSim::make_cells(const std::vector<Climate::BiomeWeather> &p_weather) {
    const int n = cells.size();
    const double r = params.radius_m;
    cell.assign(cells.cells(), Cell{});
    double spacing = 1e30;
    for (int face = 0; face < WeatherGrid::FACES; face++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                Cell &c = cell[interior(face, i, j)];
                double up[3], xp[3], xm[3], yp[3], ym[3];
                cells.cell_direction(face, i, j, up);
                cells.cell_direction(face, i + 1, j, xp);
                cells.cell_direction(face, i - 1, j, xm);
                cells.cell_direction(face, i, j + 1, yp);
                cells.cell_direction(face, i, j - 1, ym);
                double ex[3], ey[3];
                for (int k = 0; k < 3; k++) {
                    ex[k] = r * (xp[k] - xm[k]);
                    ey[k] = r * (yp[k] - ym[k]);
                }
                tangent(up, ex);
                tangent(up, ey);
                spacing = ov::min(spacing, 0.5 * ov::min(std::sqrt(dot3(ex, ex)), std::sqrt(dot3(ey, ey))));
                // ax . ex = 1, ax . ey = 0, both in the tangent plane (ay likewise).
                double exy[3], ax[3], ay[3];
                cross3(ex, ey, exy);
                const double det = dot3(exy, up);
                cross3(ey, up, ax);
                cross3(up, ex, ay);
                double wind[3];
                const Climate::BiomeWeather &w = p_weather.size() == cell.size() ? p_weather[interior(face, i, j)] : Climate::BiomeWeather();
                prevailing(up, params.prevailing_mps * w.wind, wind);
                for (int k = 0; k < 3; k++) {
                    c.up[k] = float(up[k]);
                    c.ax[k] = float(ax[k] / det);
                    c.ay[k] = float(ay[k] / det);
                    c.target[k] = float(wind[k]);
                }
                c.f = float(2.0 * omega_rad_s * up[1]);
                c.weather = w;
            }
        }
    }
    substep_s = 0.5 * spacing / WAVE_MPS;
    diffusion_k = DIFFUSION * WAVE_MPS / spacing; // times dt: nu dt / spacing^2, nu = DIFFUSION spacing c
}

void WeatherSim::coriolis(const double p_up[3], double p_f, double p_dt, double r_wind[3]) {
    const double a = -p_f * p_dt;
    double nu[3];
    cross3(p_up, r_wind, nu);
    const double c = std::cos(a);
    const double s = std::sin(a);
    for (int k = 0; k < 3; k++) {
        r_wind[k] = r_wind[k] * c + nu[k] * s;
    }
}

void WeatherSim::prevailing(const double p_up[3], double p_speed, double r_wind[3]) {
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

void WeatherSim::step(double p_dt) {
    const int count = ov::max(1, int(std::ceil(p_dt / substep_s - 1e-9)));
    const double h = p_dt / count;
    for (int s = 0; s < count; s++) {
        pass_air(h);
        pass_pressure(h);
        pass_column(h);
        start_storms(h, uint32_t(s));
    }
    time += p_dt;
    tick++;
}

// 1. The air from upstream, turned by the Coriolis, pulled towards its forcing: cur into next.
void WeatherSim::pass_air(double p_dt) {
    const int n = cells.size();
    const double wind_a = relax(p_dt, WIND_TAU_S);
    const double heat_a = relax(p_dt, HEAT_TAU_S);
    const double moist_a = relax(p_dt, MOISTURE_TAU_S);
    const double step_r = p_dt / params.radius_m;
    parallel_for(uint32_t(WeatherGrid::FACES * n), "WeatherSim::air", [&](uint32_t p_row) {
        const int face = int(p_row) / n;
        const int j = int(p_row) % n;
        for (int i = 0; i < n; i++) {
            const Cell &c = cell[interior(face, i, j)];
            const float *here = &cur[cells.index(face, i, j) * F];
            const double up[3] = { c.up[0], c.up[1], c.up[2] };
            double from[3];
            for (int k = 0; k < 3; k++) {
                from[k] = up[k] - here[WIND_X + k] * step_r;
            }
            float s[F];
            cells.sample(cur.data(), F, from, s);
            double u[3] = { s[WIND_X], s[WIND_Y], s[WIND_Z] };
            tangent(up, u);
            coriolis(up, c.f, p_dt, u);
            const double speed = std::sqrt(dot3(u, u));
            const double drag = 1.0 / (1.0 + c.weather.drag * DRAG_PER_M * speed * p_dt);
            float *out = &next[cells.index(face, i, j) * F];
            for (int k = 0; k < 3; k++) {
                out[WIND_X + k] = float((u[k] + (c.target[k] - u[k]) * wind_a) * drag);
            }
            out[ETA] = s[ETA];
            out[TEMP] = float(s[TEMP] + (c.weather.heat - s[TEMP]) * heat_a);
            out[VAPOUR] = float(s[VAPOUR] + ov::max(double(c.weather.moisture) - s[VAPOUR], 0.0) * moist_a);
            out[CLOUD] = s[CLOUD];
        } }, ov::Priority::LOW);
    cells.fill_halo(next.data(), F);
}

// 2. The pressure pushes the wind: next into cur.
void WeatherSim::pass_pressure(double p_dt) {
    const int n = cells.size();
    const double c2 = WAVE_MPS * WAVE_MPS;
    parallel_for(uint32_t(WeatherGrid::FACES * n), "WeatherSim::pressure", [&](uint32_t p_row) {
        const int face = int(p_row) / n;
        const int j = int(p_row) % n;
        auto phi = [&](int p_i, int p_j) {
            const float *s = &next[cells.index(face, p_i, p_j) * F];
            return c2 * s[ETA] - BUOYANCY * s[TEMP];
        };
        for (int i = 0; i < n; i++) {
            const Cell &c = cell[interior(face, i, j)];
            const double dx = phi(i + 1, j) - phi(i - 1, j);
            const double dy = phi(i, j + 1) - phi(i, j - 1);
            const double up[3] = { c.up[0], c.up[1], c.up[2] };
            const float *in = &next[cells.index(face, i, j) * F];
            double u[3];
            for (int k = 0; k < 3; k++) {
                u[k] = in[WIND_X + k] - p_dt * (c.ax[k] * dx + c.ay[k] * dy);
            }
            tangent(up, u);
            float *out = &cur[cells.index(face, i, j) * F];
            for (int k = 0; k < F; k++) {
                out[k] = in[k];
            }
            for (int k = 0; k < 3; k++) {
                out[WIND_X + k] = float(u[k]);
            }
        } }, ov::Priority::LOW);
    cells.fill_halo(cur.data(), F);
}

// 3. The column: the thickness follows the divergence, the lift condenses vapour, cloud rains out, a little diffusion;
// the diagnostics. cur into next, then swapped.
void WeatherSim::pass_column(double p_dt) {
    const int n = cells.size();
    const double diffuse = ov::min(0.2, diffusion_k * p_dt);
    const double condense_a = relax(p_dt, CONDENSE_TAU_S);
    const double evaporate_a = relax(p_dt, EVAPORATE_TAU_S);
    const double rain_a = relax(p_dt, RAIN_TAU_S);
    parallel_for(uint32_t(WeatherGrid::FACES * n), "WeatherSim::column", [&](uint32_t p_row) {
        const int face = int(p_row) / n;
        const int j = int(p_row) % n;
        for (int i = 0; i < n; i++) {
            const Cell &c = cell[interior(face, i, j)];
            const float *here = &cur[cells.index(face, i, j) * F];
            const float *xp = &cur[cells.index(face, i + 1, j) * F];
            const float *xm = &cur[cells.index(face, i - 1, j) * F];
            const float *yp = &cur[cells.index(face, i, j + 1) * F];
            const float *ym = &cur[cells.index(face, i, j - 1) * F];
            double div = 0.0;
            for (int k = 0; k < 3; k++) {
                div += c.ax[k] * (xp[WIND_X + k] - xm[WIND_X + k]) + c.ay[k] * (yp[WIND_X + k] - ym[WIND_X + k]);
            }
            double v[F];
            for (int k = 0; k < F; k++) {
                v[k] = here[k] + diffuse * (0.25 * (xp[k] + xm[k] + yp[k] + ym[k]) - here[k]);
            }
            v[ETA] -= p_dt * div;
            const double lift = ov::clamp(-div * LIFT_S, 0.0, 1.0);
            const double saturation = 1.0 + SATURATION_HEAT * v[TEMP] - SATURATION_LIFT * lift;
            const double excess = v[VAPOUR] - saturation;
            const double moved = excess > 0.0 ? excess * condense_a : -ov::min(v[CLOUD], -excess) * evaporate_a;
            v[VAPOUR] -= moved;
            v[CLOUD] += moved;
            const double rain = ov::max(v[CLOUD] - RAIN_CLOUD, 0.0) * rain_a;
            v[CLOUD] -= rain;
            float *out = &next[cells.index(face, i, j) * F];
            for (int k = 0; k < F; k++) {
                out[k] = float(v[k]);
            }
            float *d = &diag[cells.index(face, i, j) * D];
            d[COVER] = float(ov::clamp(v[CLOUD] / FULL_CLOUD, 0.0, 1.0));
            d[TYPE] = float(ov::clamp(0.3 + 0.7 * lift + 0.3 * v[TEMP], 0.0, 1.0));
            d[RAIN] = float(rain / p_dt);
        } }, ov::Priority::LOW);
    cells.fill_halo(next.data(), F);
    cells.fill_halo(diag.data(), D);
    cur.swap(next);
}

// Each cell starts a storm with its biome's chance, from the seed, the tick, the substep and the cell (serial: storms
// overlap, so their order is fixed).
void WeatherSim::start_storms(double p_dt, uint32_t p_substep) {
    const int n = cells.size();
    const double per_cell = p_dt / (STORM_PERIOD_S * double(cells.cells()));
    const uint64_t key = mix64((uint64_t(params.seed) << 32) ^ (tick << 8) ^ uint64_t(p_substep));
    for (int face = 0; face < WeatherGrid::FACES; face++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                const size_t ci = interior(face, i, j);
                const double chance = cell[ci].weather.storms * per_cell;
                if (chance <= 0.0) {
                    continue;
                }
                const double u = double(mix64(key ^ (uint64_t(ci) * 0xD1B54A32D192ED03ull)) >> 11) * (1.0 / 9007199254740992.0);
                if (u < chance) {
                    double dir[3];
                    cells.cell_direction(face, i, j, dir);
                    inject_storm(dir, 1.0);
                }
            }
        }
    }
}

void WeatherSim::inject_storm(const double p_dir[3], double p_strength) {
    const int n = cells.size();
    const double len = std::sqrt(dot3(p_dir, p_dir));
    const double centre[3] = { p_dir[0] / len, p_dir[1] / len, p_dir[2] / len };
    const double radius = (params.storm_m > 0.0 ? params.storm_m : 0.25 * params.radius_m) / params.radius_m; // radians
    for (int face = 0; face < WeatherGrid::FACES; face++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                const Cell &c = cell[interior(face, i, j)];
                const double d[3] = { c.up[0] - centre[0], c.up[1] - centre[1], c.up[2] - centre[2] };
                const double r2 = dot3(d, d) / (radius * radius);
                if (r2 > 9.0) {
                    continue;
                }
                const double g = p_strength * std::exp(-r2);
                float *s = &cur[cells.index(face, i, j) * F];
                s[ETA] += float(STORM_ETA * g);
                s[TEMP] += float(STORM_TEMP * g);
                s[VAPOUR] += float(STORM_VAPOUR * g);
            }
        }
    }
    cells.fill_halo(cur.data(), F);
    storms++;
}
