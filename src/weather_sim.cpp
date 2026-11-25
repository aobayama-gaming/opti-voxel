#include "weather_sim.hpp"

#include "parallel_for.hpp"
#include "weather_vec.hpp"

#include "ov/math.hpp"

#include <cmath>

using weather_vec::cross3;
using weather_vec::dot3;
using weather_vec::relax;
using weather_vec::tangent;

namespace {

constexpr int F = WeatherSim::FIELDS;
constexpr int D = WeatherSim::DIAGS;

uint64_t mix64(uint64_t p_x) {
    p_x += 0x9E3779B97F4A7C15ull;
    p_x = (p_x ^ (p_x >> 30)) * 0xBF58476D1CE4E5B9ull;
    p_x = (p_x ^ (p_x >> 27)) * 0x94D049BB133111EBull;
    return p_x ^ (p_x >> 31);
}

} // namespace

WeatherSim::WeatherSim(std::shared_ptr<const WeatherGrid> p_grid, const Params &p_params, const std::vector<Climate::BiomeWeather> &p_weather) :
        cells(std::move(p_grid)), params(p_params) {
    level_count = ov::clamp(params.levels, 1, MAX_LEVELS);
    dz = params.top_m / level_count;
    const double storm_m = params.storm_m > 0.0 ? params.storm_m : 0.25 * params.radius_m;
    omega_rad_s = WAVE_MPS / (storm_m * std::sqrt(2.0));
    make_cells(p_weather);
    const WeatherGrid &g = *cells;
    const int st = stride();
    cur.assign(g.padded_cells() * size_t(st), 0.0f);
    phi.assign(g.padded_cells() * size_t(level_count), 0.0f);
    diag.assign(g.padded_cells() * D, 0.0f);
    // The biome's warmth on the ground level, the inversion above; the ground's vapour, drier with height.
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const Cell &c = cell[g.interior(face, i, j)];
                float *s = &cur[g.index(face, i, j) * size_t(st)];
                for (int k = 0; k < level_count; k++) {
                    float *l = s + level_offset(k);
                    const double z = level_m(k) / params.top_m;
                    l[WIND_X] = c.target[0];
                    l[WIND_Y] = c.target[1];
                    l[WIND_Z] = c.target[2];
                    l[TEMP] = float(k == 0 ? c.weather.heat : ambient_temp(k));
                    l[VAPOUR] = float(GROUND_HUMIDITY * c.weather.moisture * (z < INVERSION_BASE ? 1.0 : 0.0));
                }
            }
        }
    }
    g.fill_halo(cur.data(), st);
    next = cur;
}

// Each cell's up, its central differences' dual vectors and its forcing; the substep and the diffusion from the smallest
// spacing.
void WeatherSim::make_cells(const std::vector<Climate::BiomeWeather> &p_weather) {
    const WeatherGrid &g = *cells;
    cell.assign(g.cells(), Cell{});
    double spacing = 1e30;
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                Cell &c = cell[g.interior(face, i, j)];
                double pos[3], up[3], xp[3], xm[3], yp[3], ym[3], ignored[3];
                g.cell_surface(face, i, j, pos, up);
                g.cell_surface(face, i + 1, j, xp, ignored);
                g.cell_surface(face, i - 1, j, xm, ignored);
                g.cell_surface(face, i, j + 1, yp, ignored);
                g.cell_surface(face, i, j - 1, ym, ignored);
                double ex[3], ey[3];
                for (int k = 0; k < 3; k++) {
                    ex[k] = xp[k] - xm[k];
                    ey[k] = yp[k] - ym[k];
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
                const Climate::BiomeWeather &w = p_weather.size() == cell.size() ? p_weather[g.interior(face, i, j)] : Climate::BiomeWeather();
                prevailing(up, params.prevailing_mps * w.wind, wind);
                for (int k = 0; k < 3; k++) {
                    c.pos[k] = pos[k];
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
    weather_vec::coriolis(p_up, p_f, p_dt, r_wind);
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

// 1. Each level's air from upstream, turned by the Coriolis, pulled towards its forcing; the ground's on the lowest:
// cur into next.
void WeatherSim::pass_air(double p_dt) {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const double wind_a = relax(p_dt, WIND_TAU_S);
    const double heat_a = relax(p_dt, HEAT_TAU_S);
    const double radiate_a = relax(p_dt, RADIATE_TAU_S);
    const double moist_a = relax(p_dt, MOISTURE_TAU_S);
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherSim::air", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Cell &c = cell[g.interior(face, i, j)];
            const size_t at = g.index(face, i, j) * size_t(st);
            const float *here = &cur[at];
            float *out = &next[at];
            const double up[3] = { c.up[0], c.up[1], c.up[2] };
            for (int k = 0; k < level_count; k++) {
                const float *hk = here + level_offset(k);
                double from[3];
                for (int a = 0; a < 3; a++) {
                    from[a] = c.pos[a] - hk[WIND_X + a] * p_dt;
                }
                float s[F];
                g.sample(cur.data() + level_offset(k), st, F, from, s);
                double u[3] = { s[WIND_X], s[WIND_Y], s[WIND_Z] };
                tangent(up, u);
                coriolis(up, c.f, p_dt, u);
                const bool ground = k == 0;
                const double speed = std::sqrt(dot3(u, u));
                const double drag = ground ? 1.0 / (1.0 + c.weather.drag * DRAG_PER_M * speed * p_dt) : 1.0;
                float *o = out + level_offset(k);
                for (int a = 0; a < 3; a++) {
                    o[WIND_X + a] = float((u[a] + (c.target[a] - u[a]) * wind_a) * drag);
                }
                if (ground) {
                    o[TEMP] = float(s[TEMP] + (c.weather.heat - s[TEMP]) * heat_a);
                    o[VAPOUR] = float(s[VAPOUR] + ov::max(GROUND_HUMIDITY * c.weather.moisture - s[VAPOUR], 0.0) * moist_a);
                } else {
                    o[TEMP] = float(s[TEMP] + (ambient_temp(k) - s[TEMP]) * radiate_a);
                    o[VAPOUR] = s[VAPOUR];
                }
                o[CLOUD] = s[CLOUD];
            }
            out[eta_offset()] = here[eta_offset()];
        } }, ov::Priority::LOW);
    g.fill_halo(next.data(), st);
}

// 2. The pressure pushes each level's wind: next into cur. phi per padded cell and level first (the halo's from next's
// halo), the warmth above a level summed from the top down.
void WeatherSim::pass_pressure(double p_dt) {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int levels_n = level_count;
    const double c2 = WAVE_MPS * WAVE_MPS;
    // A single level: c^2 ETA - BUOYANCY TEMP (its half of a column twice as warm).
    const double b = BUOYANCY * 2.0 / levels_n;
    const uint32_t rows = uint32_t(g.faces() * g.stride_y());
    parallel_for(rows, "WeatherSim::phi", [&](uint32_t p_row) {
        const size_t first = size_t(p_row) * size_t(g.stride_x());
        for (size_t p = first; p < first + size_t(g.stride_x()); p++) {
            const float *s = &next[p * size_t(st)];
            const double base = c2 * s[eta_offset()];
            double above = 0.0;
            for (int k = levels_n - 1; k >= 0; k--) {
                const double t = s[level_offset(k) + TEMP];
                phi[p * size_t(levels_n) + size_t(k)] = float(base - b * (above + 0.5 * t));
                above += t;
            }
        } }, ov::Priority::LOW);
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherSim::pressure", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Cell &c = cell[g.interior(face, i, j)];
            const size_t at = g.index(face, i, j);
            const float *xp = &phi[g.index(face, i + 1, j) * size_t(levels_n)];
            const float *xm = &phi[g.index(face, i - 1, j) * size_t(levels_n)];
            const float *yp = &phi[g.index(face, i, j + 1) * size_t(levels_n)];
            const float *ym = &phi[g.index(face, i, j - 1) * size_t(levels_n)];
            const double up[3] = { c.up[0], c.up[1], c.up[2] };
            const float *in = &next[at * size_t(st)];
            float *out = &cur[at * size_t(st)];
            for (int k = 0; k < st; k++) {
                out[k] = in[k];
            }
            for (int k = 0; k < levels_n; k++) {
                const double dx = double(xp[k]) - double(xm[k]);
                const double dy = double(yp[k]) - double(ym[k]);
                double u[3];
                for (int a = 0; a < 3; a++) {
                    u[a] = in[level_offset(k) + WIND_X + a] - p_dt * (c.ax[a] * dx + c.ay[a] * dy);
                }
                tangent(up, u);
                for (int a = 0; a < 3; a++) {
                    out[level_offset(k) + WIND_X + a] = float(u[a]);
                }
            }
        } }, ov::Priority::LOW);
    g.fill_halo(cur.data(), st);
}

// 3. The column: the top follows the mean divergence, the levels' divergences lift the air between them, warm air
// under colder overturns, vapour condenses, cloud rains out; a little diffusion; the diagnostics. cur into next, then
// swapped.
void WeatherSim::pass_column(double p_dt) {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int eta = eta_offset();
    const double diffuse = ov::min(0.2, diffusion_k * p_dt);
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherSim::column", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        double v[MAX_LEVELS * F];
        double div[MAX_LEVELS];
        for (int i = 0; i < g.width(); i++) {
            const Cell &c = cell[g.interior(face, i, j)];
            const float *here = &cur[g.index(face, i, j) * size_t(st)];
            const float *xp = &cur[g.index(face, i + 1, j) * size_t(st)];
            const float *xm = &cur[g.index(face, i - 1, j) * size_t(st)];
            const float *yp = &cur[g.index(face, i, j + 1) * size_t(st)];
            const float *ym = &cur[g.index(face, i, j - 1) * size_t(st)];
            double mean_div = 0.0;
            for (int k = 0; k < level_count; k++) {
                const int o = level_offset(k);
                double d = 0.0;
                for (int a = 0; a < 3; a++) {
                    d += c.ax[a] * (xp[o + WIND_X + a] - xm[o + WIND_X + a]) + c.ay[a] * (yp[o + WIND_X + a] - ym[o + WIND_X + a]);
                }
                div[k] = d;
                mean_div += d;
                for (int f = 0; f < F; f++) {
                    v[o + f] = here[o + f] + diffuse * (0.25 * (xp[o + f] + xm[o + f] + yp[o + f] + ym[o + f]) - here[o + f]);
                }
            }
            mean_div /= level_count;
            const double top = here[eta] + diffuse * (0.25 * (xp[eta] + xm[eta] + yp[eta] + ym[eta]) - here[eta]) - p_dt * mean_div;
            lift(p_dt, div, v);
            convect(p_dt, v);
            double rain, cover, type;
            condense(p_dt, v, rain, cover, type);
            float *out = &next[g.index(face, i, j) * size_t(st)];
            for (int k = 0; k < level_count * F; k++) {
                out[k] = float(v[k]);
            }
            out[eta] = float(top);
            float *d = &diag[g.index(face, i, j) * D];
            d[COVER] = float(cover);
            d[TYPE] = float(type);
            d[RAIN] = float(rain);
        } }, ov::Priority::LOW);
    g.fill_halo(next.data(), st);
    g.fill_halo(diag.data(), D);
    cur.swap(next);
}

// The vertical wind through each level's top (the ground's 0): w_{k+1/2} = w_{k-1/2} - div_k dz (the air a level
// gathers rises out of its top); each interface carries its upstream level's fields into the downstream one, at most
// half a level a substep. From the column's values before the step, so the interfaces don't chain.
void WeatherSim::lift(double p_dt, const double *p_div, double *r_v) const {
    double old[MAX_LEVELS * F];
    for (int k = 0; k < level_count * F; k++) {
        old[k] = r_v[k];
    }
    double w = 0.0;
    for (int k = 0; k + 1 < level_count; k++) {
        w -= p_div[k] * dz;
        const double a = ov::clamp(w * p_dt / dz, -0.5, 0.5);
        const int lo = level_offset(k);
        const int hi = level_offset(k + 1);
        for (int f = 0; f < F; f++) {
            if (a > 0.0) {
                r_v[hi + f] += a * (old[lo + f] - old[hi + f]);
            } else if (a < 0.0) {
                r_v[lo + f] -= a * (old[hi + f] - old[lo + f]);
            }
        }
    }
}

// A parcel of the ground level's air rises while it is no cooler than the level it reaches less CONVECT_EXCESS, its
// vapour past each level's saturation condensing into latent warmth; the levels from the ground to its top mix their
// air (wind, vapour, cloud; not the temperature, the forcing's profile) towards its mean (CONVECT_TAU_S): the cloud
// mixed into the drier levels below evaporates, so a mixed layer holds its water less each level's saturation. Above its top the air sinks and dries (DRY_TAU_S): the clouds live in the
// convecting layer, as deep as the ground's warmth (a storm's) carries the thermals.
void WeatherSim::convect(double p_dt, double *r_v) const {
    double parcel_t = r_v[TEMP];
    double parcel_q = r_v[VAPOUR];
    int top = 0;
    for (int k = 1; k < level_count; k++) {
        const double *l = r_v + level_offset(k);
        const double excess = parcel_q - saturation(k, parcel_t);
        if (excess > 0.0) {
            parcel_q -= excess;
            parcel_t += LATENT * excess;
        }
        if (parcel_t + CONVECT_EXCESS < l[TEMP]) {
            break;
        }
        top = k;
    }
    // Above the thermals' top the air sinks and dries: its vapour and cloud go.
    const double dry_a = relax(p_dt, DRY_TAU_S);
    for (int k = top + 1; k < level_count; k++) {
        r_v[level_offset(k) + VAPOUR] *= 1.0 - dry_a;
        r_v[level_offset(k) + CLOUD] *= 1.0 - dry_a;
    }
    if (top > 0) {
        const double convect_a = relax(p_dt, CONVECT_TAU_S);
        for (int f = 0; f < F; f++) {
            if (f == TEMP) {
                continue; // the forcing's profile stays: the inversion
            }
            double mean = 0.0;
            for (int k = 0; k <= top; k++) {
                mean += r_v[level_offset(k) + f];
            }
            mean /= top + 1;
            for (int k = 0; k <= top; k++) {
                r_v[level_offset(k) + f] += (mean - r_v[level_offset(k) + f]) * convect_a;
            }
        }
    }
}

double WeatherSim::ambient_temp(int p_level) const {
    const double z = level_m(p_level) / params.top_m;
    return INVERSION_TEMP * ov::max(0.0, z - INVERSION_BASE) / (1.0 - INVERSION_BASE);
}

double WeatherSim::saturation(int p_level, double p_temp) const {
    return std::exp(SATURATION_RATE * (p_temp - LAPSE * level_m(p_level) / params.top_m));
}

void WeatherSim::condense(double p_dt, double *r_v, double &r_rain, double &r_cover, double &r_type) const {
    const double condense_a = relax(p_dt, CONDENSE_TAU_S);
    const double rain_a = relax(p_dt, RAIN_TAU_S);
    double rain = 0.0;
    double most = 0.0;
    int cloudy = 0;
    for (int k = 0; k < level_count; k++) {
        double *l = r_v + level_offset(k);
        const double excess = l[VAPOUR] - saturation(k, l[TEMP]);
        const double moved = excess > 0.0 ? excess * condense_a : -ov::min(l[CLOUD], -excess); // evaporates at once
        l[VAPOUR] -= moved;
        l[CLOUD] += moved;
        const double out = ov::max(l[CLOUD] - RAIN_CLOUD, 0.0) * rain_a;
        l[CLOUD] -= out;
        rain += out;
        most = ov::max(most, l[CLOUD]);
        cloudy += l[CLOUD] > 0.25 * FULL_CLOUD ? 1 : 0;
    }
    r_rain = rain / p_dt;
    r_cover = ov::clamp(most / FULL_CLOUD, 0.0, 1.0);
    // Half the column deep: a tower.
    r_type = ov::clamp(cloudy / ov::max(1.0, 0.5 * level_count), 0.0, 1.0);
}

// Each cell starts a storm with its biome's chance, from the seed, the tick, the substep and the cell (serial: storms
// overlap, so their order is fixed).
void WeatherSim::start_storms(double p_dt, uint32_t p_substep) {
    const WeatherGrid &g = *cells;
    const double per_cell = p_dt / (STORM_PERIOD_S * double(g.cells()));
    const uint64_t key = mix64((uint64_t(params.seed) << 32) ^ (tick << 8) ^ uint64_t(p_substep));
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const size_t ci = g.interior(face, i, j);
                const double chance = cell[ci].weather.storms * per_cell;
                if (chance <= 0.0) {
                    continue;
                }
                const double u = double(mix64(key ^ (uint64_t(ci) * 0xD1B54A32D192ED03ull)) >> 11) * (1.0 / 9007199254740992.0);
                if (u < chance) {
                    inject_storm(cell[ci].pos, 1.0);
                }
            }
        }
    }
}

void WeatherSim::inject_storm(const double p_pos[3], double p_strength) {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const WeatherGrid::Point at = g.locate(p_pos);
    double centre[3], centre_up[3];
    g.surface(at.face, at.x, at.y, centre, centre_up);
    const double radius = params.storm_m > 0.0 ? params.storm_m : 0.25 * params.radius_m;
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const Cell &c = cell[g.interior(face, i, j)];
                const double d[3] = { c.pos[0] - centre[0], c.pos[1] - centre[1], c.pos[2] - centre[2] };
                const double r2 = dot3(d, d) / (radius * radius);
                if (r2 > 9.0) {
                    continue;
                }
                const double gauss = p_strength * std::exp(-r2);
                float *s = &cur[g.index(face, i, j) * size_t(st)];
                s[eta_offset()] += float(STORM_ETA * gauss);
                for (int k = 0; k < level_count; k++) {
                    const double z = level_m(k) / params.top_m;
                    float *l = s + level_offset(k);
                    l[TEMP] += float(STORM_TEMP * gauss * (1.0 - z));
                    l[VAPOUR] += z <= 0.5 ? float(STORM_VAPOUR * gauss) : 0.0f;
                }
            }
        }
    }
    g.fill_halo(cur.data(), st);
    storms++;
}
