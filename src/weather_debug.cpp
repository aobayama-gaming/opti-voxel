#include "weather_debug.hpp"

#include "sdf_cascade.hpp"
#include "weather.hpp"
#include "weather_convection.hpp"
#include "weather_forcing.hpp"
#include "world_file.hpp"

#include "ov/math.hpp"

#include <cmath>

namespace WeatherDebug {

namespace {

constexpr FieldInfo FIELD_INFO[FIELDS] = {
    { "wind east", "m/s", true },
    { "wind north", "m/s", true },
    { "w (up)", "m/s", true },
    { "w (interface below)", "m/s", true },
    { "wind speed", "m/s", false },
    { "theta", "K", false },
    { "theta - sounding", "K", true },
    { "temperature", "C", false },
    { "qv", "g/kg", false },
    { "qs (saturation)", "g/kg", false },
    { "relative humidity", "", false },
    { "qc (cloud)", "g/kg", false },
    { "qr (rain)", "g/kg", false },
    { "cloudy share", "", false },
    { "Q1", "", true },
    { "sigma_s (spread)", "g/kg", false },
    { "updraft qc", "g/kg", false },
    { "buoyancy", "m/s^2", true },
    { "pressure", "m^2/s^2", true },
    { "ground T", "K", false },
    { "ground T mean (biome)", "K", false },
    { "ground - air", "K", true },
    { "wetness", "", false },
    { "z0", "m", false },
    { "thermal inertia", "", false },
    { "storms", "", false },
    { "sensible flux", "W/m^2", true },
    { "latent flux", "W/m^2", true },
    { "u*", "m/s", false },
    { "rain", "mm/h", false },
    { "updraft top", "m", false },
    { "cover (max share)", "", false },
    { "cloud water path", "g/m^2", false },
};

// The level fields of one column (interior cell p_in, padded p_c) into r_values (field-major, p_n cells).
void capture_levels(const WeatherFluid &p_fluid, const WeatherMoist &p_moist, const WeatherConvection &p_convection,
        const std::vector<float> &p_buoyancy, size_t p_in, size_t p_c, size_t p_n, std::vector<float> &r_values, std::vector<float> &r_wind) {
    const WeatherMoist::Sounding &ref = p_moist.reference();
    const int levels = p_fluid.levels();
    const float *s = &p_fluid.state()[p_c * size_t(p_fluid.stride())];
    double up[3];
    p_fluid.column_up(p_in, up);
    const double el = std::sqrt(up[2] * up[2] + up[0] * up[0]);
    const double east[3] = { el > 1e-9 ? up[2] / el : 1.0, 0.0, el > 1e-9 ? -up[0] / el : 0.0 };
    const double north[3] = { up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0] };
    const size_t per = p_n * size_t(levels);
    for (int k = 0; k < levels; k++) {
        const size_t ck = p_fluid.cell_index(p_in, k);
        float *v = &r_values[ck];
        const float *u = s + p_fluid.wind_offset(k);
        double we = 0.0, wn = 0.0, wu = 0.0;
        for (int a = 0; a < 3; a++) {
            we += u[a] * east[a];
            wn += u[a] * north[a];
            wu += u[a] * up[a];
            r_wind[ck * 3 + size_t(a)] = u[a];
        }
        const double theta = s[p_fluid.scalar_offset(k, WeatherMoist::THETA)];
        const double qv = s[p_fluid.scalar_offset(k, WeatherMoist::QV)];
        const double t = theta * ref.exner[size_t(k)];
        const double qs = WeatherMoist::saturation(t, WeatherMoist::P0_MB * std::pow(ref.exner[size_t(k)], 1.0 / WeatherMoist::KAPPA_EXNER));
        v[WIND_EAST * per] = float(we);
        v[WIND_NORTH * per] = float(wn);
        v[WIND_UP * per] = float(wu);
        v[WIND_UP_FACE * per] = p_fluid.vertical_w(p_c, k);
        v[SPEED * per] = float(std::sqrt(we * we + wn * wn));
        v[THETA * per] = float(theta);
        v[THETA_ANOMALY * per] = float(theta - ref.theta[size_t(k)]);
        v[TEMP_C * per] = float(t - 273.15);
        v[QV * per] = float(qv * 1000.0);
        v[QS * per] = float(qs * 1000.0);
        v[RH * per] = float(qv / qs);
        v[QC * per] = s[p_fluid.scalar_offset(k, WeatherMoist::QC)] * 1000.0f;
        v[QR * per] = s[p_fluid.scalar_offset(k, WeatherMoist::QR)] * 1000.0f;
        v[SHARE * per] = p_moist.share()[ck];
        v[Q1 * per] = p_moist.q1()[ck];
        v[SIGMA * per] = p_moist.spread()[ck] * 1000.0f;
        v[UPDRAFT_QC * per] = p_convection.cloud()[ck] * 1000.0f;
        v[BUOYANCY * per] = p_buoyancy[ck];
        v[PRESSURE * per] = p_fluid.pressure()[p_c * size_t(levels) + size_t(k)];
    }
}

// The column fields of interior cell p_in into r_col (FIELDS - LEVEL_FIELDS floats, GROUND_K first).
void capture_column(const WeatherFluid &p_fluid, const WeatherMoist &p_moist, const WeatherConvection &p_convection,
        const WeatherForcing &p_forcing, const WeatherMoist::Ground &p_ground, size_t p_in, size_t p_c, float *r_col) {
    const WeatherMoist::Sounding &ref = p_moist.reference();
    const float *s = &p_fluid.state()[p_c * size_t(p_fluid.stride())];
    const Climate::BiomeWeather &b = p_forcing.cell_climate()[p_in];
    const WeatherMoist::Surface &sf = p_moist.surface()[p_in];
    float cover = 0.0f;
    double lwp = 0.0;
    for (int k = 0; k < p_fluid.levels(); k++) {
        cover = ov::max(cover, p_moist.share()[p_fluid.cell_index(p_in, k)]);
        lwp += ref.rho[size_t(k)] * double(s[p_fluid.scalar_offset(k, WeatherMoist::QC)]) * p_fluid.thick_m(k) * 1000.0;
    }
    r_col[GROUND_K - LEVEL_FIELDS] = p_ground.temp_k;
    r_col[GROUND_MEAN_K - LEVEL_FIELDS] = b.ground_k;
    r_col[GROUND_AIR_K - LEVEL_FIELDS] = float(double(p_ground.temp_k) - double(s[p_fluid.scalar_offset(0, WeatherMoist::THETA)]) * ref.exner[0]);
    r_col[WETNESS - LEVEL_FIELDS] = p_ground.wetness;
    r_col[Z0_M - LEVEL_FIELDS] = p_ground.z0_m;
    r_col[INERTIA - LEVEL_FIELDS] = b.thermal_inertia;
    r_col[STORMS - LEVEL_FIELDS] = b.storms;
    r_col[SENSIBLE_W_M2 - LEVEL_FIELDS] = float(ref.rho[0] * WeatherMoist::CP * double(sf.theta_flux));
    r_col[LATENT_W_M2 - LEVEL_FIELDS] = float(ref.rho[0] * WeatherMoist::LATENT * double(sf.qv_flux));
    r_col[USTAR - LEVEL_FIELDS] = sf.ustar;
    r_col[RAIN_MM_H - LEVEL_FIELDS] = float(double(p_moist.rain()[p_in]) * 3.6e6);
    r_col[UPDRAFT_TOP_M - LEVEL_FIELDS] = p_convection.tops()[p_in];
    r_col[COVER - LEVEL_FIELDS] = cover;
    r_col[LWP_G_M2 - LEVEL_FIELDS] = float(lwp);
}

} // namespace

const FieldInfo &info(int p_field) {
    return FIELD_INFO[ov::clamp(p_field, 0, FIELDS - 1)];
}

const float *Snapshot::field(int p_field) const {
    const size_t n = cells();
    const size_t at = per_level(p_field) ? size_t(p_field) * n * size_t(levels) : size_t(LEVEL_FIELDS) * n * size_t(levels) + size_t(p_field - LEVEL_FIELDS) * n;
    return values.data() + at;
}

float Snapshot::at(int p_field, size_t p_interior, int p_level) const {
    return per_level(p_field) ? field(p_field)[p_interior * size_t(levels) + size_t(ov::clamp(p_level, 0, levels - 1))] : field(p_field)[p_interior];
}

std::vector<float> Snapshot::slice(int p_field, int p_level) const {
    std::vector<float> out(cells());
    for (size_t c = 0; c < out.size(); c++) {
        out[c] = at(p_field, c, p_level);
    }
    return out;
}

std::vector<float> Snapshot::section(int p_field, int p_face, int p_j) const {
    const int rows = per_level(p_field) ? levels : 1;
    std::vector<float> out(size_t(width) * size_t(rows));
    for (int k = 0; k < rows; k++) {
        for (int i = 0; i < width; i++) {
            out[size_t(k) * size_t(width) + size_t(i)] = at(p_field, interior(p_face, i, p_j), k);
        }
    }
    return out;
}

std::shared_ptr<const Snapshot> Snapshot::capture(const WeatherFluid &p_fluid, const WeatherMoist &p_moist, const WeatherConvection &p_convection,
        const WeatherForcing &p_forcing, const std::vector<WeatherMoist::Ground> &p_ground, const std::vector<float> &p_buoyancy, double p_game_s) {
    auto out = std::make_shared<Snapshot>();
    const WeatherGrid &g = p_fluid.grid();
    out->faces = g.faces();
    out->width = g.width();
    out->height = g.height();
    out->levels = p_fluid.levels();
    for (int m = 0; m <= p_fluid.levels(); m++) {
        out->faces_m.push_back(p_fluid.face_m(m));
    }
    out->game_s = p_game_s;
    const size_t n = out->cells();
    out->values.assign(size_t(LEVEL_FIELDS) * n * size_t(out->levels) + size_t(FIELDS - LEVEL_FIELDS) * n, 0.0f);
    out->wind.assign(n * size_t(out->levels) * 3, 0.0f);
    const size_t columns = size_t(LEVEL_FIELDS) * n * size_t(out->levels);
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const size_t in = g.interior(f, i, j);
                const size_t c = g.index(f, i, j);
                capture_levels(p_fluid, p_moist, p_convection, p_buoyancy, in, c, n, out->values, out->wind);
                float col[FIELDS - LEVEL_FIELDS];
                capture_column(p_fluid, p_moist, p_convection, p_forcing, p_ground[in], in, c, col);
                for (int k = 0; k < FIELDS - LEVEL_FIELDS; k++) {
                    out->values[columns + size_t(k) * n + in] = col[k];
                }
            }
        }
    }
    return out;
}

namespace {

// p_fn(level, value) over every cell of p_columns, per level of p_field (one for a column field).
template <class F>
void each(const Snapshot &p_snap, int p_field, Columns p_columns, F &&p_fn) {
    const int rows = per_level(p_field) ? p_snap.levels : 1;
    for (size_t c = 0; c < p_snap.cells(); c++) {
        const bool sea = p_snap.at(WETNESS, c, 0) >= 0.9f;
        if ((p_columns == SEA && !sea) || (p_columns == LAND && sea)) {
            continue;
        }
        for (int k = 0; k < rows; k++) {
            p_fn(k, double(p_snap.at(p_field, c, k)));
        }
    }
}

std::vector<double> reduce(const Snapshot &p_snap, int p_field, Columns p_columns, int p_kind, float p_above) {
    const size_t rows = size_t(per_level(p_field) ? p_snap.levels : 1);
    std::vector<double> sum(rows, p_kind == 2 ? -1e300 : 0.0), count(rows, 0.0);
    each(p_snap, p_field, p_columns, [&](int p_k, double p_v) {
        const size_t k = size_t(p_k);
        count[k] += 1.0;
        switch (p_kind) {
            case 0:
                sum[k] += p_v;
                break;
            case 1:
                sum[k] += p_v * p_v;
                break;
            case 2:
                sum[k] = ov::max(sum[k], p_v);
                break;
            default:
                sum[k] += p_v > double(p_above) ? 1.0 : 0.0;
                break;
        }
    });
    for (size_t k = 0; k < rows; k++) {
        if (count[k] == 0.0) {
            sum[k] = 0.0;
        } else if (p_kind != 2) {
            sum[k] /= count[k];
        }
        if (p_kind == 1) {
            sum[k] = std::sqrt(sum[k]);
        }
    }
    return sum;
}

} // namespace

std::vector<double> mean(const Snapshot &p_snap, int p_field, Columns p_columns) {
    return reduce(p_snap, p_field, p_columns, 0, 0.0f);
}

std::vector<double> rms(const Snapshot &p_snap, int p_field, Columns p_columns) {
    return reduce(p_snap, p_field, p_columns, 1, 0.0f);
}

std::vector<double> maximum(const Snapshot &p_snap, int p_field, Columns p_columns) {
    return reduce(p_snap, p_field, p_columns, 2, 0.0f);
}

std::vector<double> share_above(const Snapshot &p_snap, int p_field, Columns p_columns, float p_above) {
    return reduce(p_snap, p_field, p_columns, 3, p_above);
}

CloudObjects cloud_objects(const Snapshot &p_snap, Columns p_columns, float p_qc_g_kg) {
    const size_t n = p_snap.cells();
    std::vector<uint8_t> cloudy(n, 0);
    size_t in = 0, any = 0;
    for (size_t c = 0; c < n; c++) {
        const bool sea = p_snap.at(WETNESS, c, 0) >= 0.9f;
        if ((p_columns == SEA && !sea) || (p_columns == LAND && sea)) {
            continue;
        }
        in++;
        for (int k = 0; k < p_snap.levels && !cloudy[c]; k++) {
            cloudy[c] = p_snap.at(QC, c, k) > p_qc_g_kg ? 1 : 0;
        }
        any += cloudy[c];
    }
    CloudObjects out;
    out.cover = in > 0 ? double(any) / double(in) : 0.0;
    std::vector<uint32_t> stack;
    size_t cells = 0;
    for (size_t c = 0; c < n; c++) {
        if (cloudy[c] != 1) {
            continue;
        }
        int size = 0;
        stack.push_back(uint32_t(c));
        cloudy[c] = 2;
        while (!stack.empty()) {
            const size_t at = stack.back();
            stack.pop_back();
            size++;
            const int i = int(at % size_t(p_snap.width));
            const int j = int((at / size_t(p_snap.width)) % size_t(p_snap.height));
            const int f = int(at / (size_t(p_snap.width) * size_t(p_snap.height)));
            const int nb[4][2] = { { i + 1, j }, { i - 1, j }, { i, j + 1 }, { i, j - 1 } };
            for (const auto &q : nb) {
                if (q[0] < 0 || q[1] < 0 || q[0] >= p_snap.width || q[1] >= p_snap.height) {
                    continue;
                }
                const size_t o = p_snap.interior(f, q[0], q[1]);
                if (cloudy[o] == 1) {
                    cloudy[o] = 2;
                    stack.push_back(uint32_t(o));
                }
            }
        }
        out.count++;
        cells += size_t(size);
        out.largest_cells = ov::max(out.largest_cells, size);
    }
    out.mean_cells = out.count > 0 ? double(cells) / double(out.count) : 0.0;
    return out;
}

ov::str profile_table(const Snapshot &p_snap, Columns p_columns) {
    const std::vector<double> theta = mean(p_snap, THETA, p_columns), qv = mean(p_snap, QV, p_columns), rh = mean(p_snap, RH, p_columns);
    const std::vector<double> qc = mean(p_snap, QC, p_columns), qc_max = maximum(p_snap, QC, p_columns), share = mean(p_snap, SHARE, p_columns);
    const std::vector<double> cloudy = share_above(p_snap, QC, p_columns, 0.01f), w = rms(p_snap, WIND_UP, p_columns), w_max = maximum(p_snap, WIND_UP, p_columns);
    const std::vector<double> sigma = mean(p_snap, SIGMA, p_columns), updraft = mean(p_snap, UPDRAFT_QC, p_columns), speed = mean(p_snap, SPEED, p_columns);
    ov::str s = "| z m | theta K | qv g/kg | RH | qc g/kg | qc max | cloud frac (share) | cells qc>0.01 | w rms | w max | sigma_s g/kg | updraft qc | wind m/s |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (int k = p_snap.levels - 1; k >= 0; k--) {
        const size_t i = size_t(k);
        s += ov::format("| %.0f | %.2f | %.2f | %.3f | %.4f | %.3f | %.3f | %.3f | %.3f | %.2f | %.3f | %.4f | %.2f |\n", 0.5 * (p_snap.faces_m[i] + p_snap.faces_m[i + 1]), theta[i], qv[i], rh[i],
                qc[i], qc_max[i], share[i], cloudy[i], w[i], w_max[i], sigma[i], updraft[i], speed[i]);
    }
    s += "\nColumns:";
    for (int f = LEVEL_FIELDS; f < FIELDS; f++) {
        if (f == Z0_M || f == INERTIA || f == STORMS || f == GROUND_MEAN_K) {
            continue;
        }
        s += ov::format(" %s %.3g %s,", info(f).name, mean(p_snap, f, p_columns)[0], info(f).unit);
    }
    s += ov::format(" updraft columns %.3f.\n", share_above(p_snap, UPDRAFT_TOP_M, p_columns, 0.0f)[0]);
    const CloudObjects objects = cloud_objects(p_snap, p_columns, 0.01f);
    s += ov::format("Clouds from above (qc > 0.01 g/kg at any level): cover %.3f, %d clouds, %.1f columns each on average, the largest %d.\n",
            objects.cover, objects.count, objects.mean_cells, objects.largest_cells);
    // The vertical checkerboard: each column's field (its column mean out) projected on (-1)^k, its rms over the columns
    // against the field's own (noise alone gives about the field's over the root of the levels).
    s += "Checkerboard (-1)^k part / rms / noise level:";
    for (const int f : { WIND_UP_FACE, QV, QC, THETA }) {
        double mode = 0.0, all = 0.0, count = 0.0;
        for (size_t c = 0; c < p_snap.cells(); c++) {
            double m = 0.0;
            for (int k = 0; k < p_snap.levels; k++) {
                m += double(p_snap.at(f, c, k)) / double(p_snap.levels);
            }
            double a = 0.0;
            for (int k = 0; k < p_snap.levels; k++) {
                const double v = double(p_snap.at(f, c, k)) - m;
                a += (k & 1 ? -v : v) / double(p_snap.levels);
                all += v * v / double(p_snap.levels);
            }
            mode += a * a;
            count += 1.0;
        }
        s += ov::format(" %s %.4f / %.4f / %.4f,", info(f).name, std::sqrt(mode / count), std::sqrt(all / count), std::sqrt(all / count / double(p_snap.levels)));
    }
    s += "\n";
    return s;
}

Session::Session() = default;
Session::~Session() = default;

ov::str Session::open(const WorldFile &p_world, double p_spin_up_s, bool p_probing) {
    sim.reset();
    sdf = std::make_unique<SDFCascade>();
    ChunkParam param;
    ov::str error = sdf->configure_world(p_world, param);
    if (error.empty() && !p_world.weather.on) {
        error = "weather: the world has none (not a sphere, or \"weather\": {\"on\": false}).";
    }
    if (!error.empty()) {
        return error;
    }
    const Climate::Table *climate = sdf->get_climate();
    const SDFCascade *cascade = sdf.get();
    Weather::ForcingFn forcing;
    if (climate != nullptr) {
        forcing = [climate, cascade](const VoxelPos &p_voxel, int p_m) { return Weather::forcing_at(*climate, cascade->program(), p_voxel, p_m); };
    }
    WorldFile::Weather settings = p_world.weather;
    if (p_spin_up_s >= 0.0) {
        settings.spin_up_s = p_spin_up_s;
    }
    sim = std::make_unique<Weather>(settings, sdf->get_base(), p_world.voxel_m, p_world.seed, forcing, p_probing);
    period_usec = uint64_t(std::llround(1e6 / settings.tick_hz));
    now_usec = 1;
    sim->update(now_usec);
    return "";
}

void Session::step(int p_ticks) {
    if (!sim) {
        return;
    }
    if (!sim->ready()) {
        sim->finish(); // the forcing and the spin-up
        sim->update(now_usec);
    }
    for (int t = 0; t < p_ticks; t++) {
        now_usec += period_usec;
        sim->update(now_usec); // asks for this period's tick
        sim->finish();
        sim->update(now_usec); // collects it (the clock didn't move: no new tick)
    }
}

void Session::run(double p_real_s) {
    if (!sim) {
        return;
    }
    now_usec += uint64_t(std::llround(ov::clamp(p_real_s, 0.0, Weather::MAX_FRAME_S) * 1e6));
    sim->update(now_usec);
}

} // namespace WeatherDebug
