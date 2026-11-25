#include "weather_report.hpp"

#include "sdf_cascade.hpp"
#include "weather.hpp"
#include "world_file.hpp"

#include "ov/clock.hpp"
#include "ov/files.hpp"
#include "ov/format.hpp"
#include "ov/math.hpp"
#include "ov/png.hpp"

#include <cmath>
#include <cstdint>

namespace {

constexpr double PI = 3.141592653589793;

// The unit direction of an equirectangular pixel's centre (the +y pole at the top).
void pixel_dir(int p_x, int p_y, int p_w, int p_h, double r_dir[3]) {
    const double lon = (p_x + 0.5) / p_w * 2.0 * PI - PI;
    const double lat = 0.5 * PI - (p_y + 0.5) / p_h * PI;
    r_dir[0] = std::cos(lat) * std::cos(lon);
    r_dir[1] = std::sin(lat);
    r_dir[2] = std::cos(lat) * std::sin(lon);
}

uint8_t byte(double p_v) {
    return uint8_t(ov::clamp(std::lround(p_v * 255.0), 0L, 255L));
}

// A hue in [0, 1) at full saturation, times p_value.
void hue_rgb(double p_hue, double p_value, uint8_t r_rgb[3]) {
    for (int k = 0; k < 3; k++) {
        const double h = std::fmod(p_hue + 1.0 - k / 3.0, 1.0) * 6.0;
        const double c = ov::clamp(std::fabs(h - 3.0) - 1.0, 0.0, 1.0);
        r_rgb[k] = byte(c * p_value);
    }
}

// The cover and wind maps of the weather's latest state, and its numbers.
WeatherReport::Frame map_state(const Weather &p_weather, int p_w, double p_full_mps) {
    const int h = p_w / 2;
    WeatherReport::Frame f;
    std::vector<uint8_t> &cover = f.cover_rgb;
    std::vector<uint8_t> &wind = f.wind_rgb;
    cover.resize(size_t(p_w) * size_t(h) * 3);
    wind.resize(cover.size());
    double cover_sum = 0.0, wind_max = 0.0, raining = 0.0, weight = 0.0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < p_w; x++) {
            double d[3];
            pixel_dir(x, y, p_w, h, d);
            const double r = p_weather.radius_m();
            const double pos[3] = { d[0] * r, d[1] * r, d[2] * r };
            Weather::Sample s;
            p_weather.at(pos, s);
            const size_t o = (size_t(y) * size_t(p_w) + size_t(x)) * 3;
            const double rain = ov::clamp(double(s.rain) / 10.0, 0.0, 1.0); // mm/h, full at 10
            cover[o + 0] = byte(s.cover * (1.0 - rain));
            cover[o + 1] = byte(s.cover * (1.0 - 0.5 * rain));
            cover[o + 2] = byte(s.cover + rain * (1.0 - s.cover));
            // The wind in the local frame: east = pole x up, north = up x east.
            const double east_len = std::sqrt(d[2] * d[2] + d[0] * d[0]);
            const double east[3] = { east_len > 1e-9 ? d[2] / east_len : 1.0, 0.0, east_len > 1e-9 ? -d[0] / east_len : 0.0 };
            const double north[3] = { d[1] * east[2] - d[2] * east[1], d[2] * east[0] - d[0] * east[2], d[0] * east[1] - d[1] * east[0] };
            const double we = s.wind_mps.x * east[0] + s.wind_mps.y * east[1] + s.wind_mps.z * east[2];
            const double wn = s.wind_mps.x * north[0] + s.wind_mps.y * north[1] + s.wind_mps.z * north[2];
            const double speed = std::sqrt(we * we + wn * wn);
            hue_rgb(std::atan2(wn, we) / (2.0 * PI) + 1.0, ov::min(speed / p_full_mps, 1.0), &wind[o]);
            // Area weights (cos latitude) for the means.
            const double a = std::sqrt(1.0 - d[1] * d[1]);
            cover_sum += a * s.cover;
            raining += s.rain > 0.1f ? a : 0.0; // mm/h
            weight += a;
            wind_max = ov::max(wind_max, speed);
        }
    }
    // The equator's cross-section, the ground at the bottom; each level's cloudy share of the planet.
    const int levels = p_weather.levels();
    const int sh = levels * WeatherReport::SECTION_PX;
    f.section_rgb.assign(size_t(p_w) * size_t(sh) * 3, 0);
    std::vector<float> column(static_cast<size_t>(levels), 0.0f);
    for (int x = 0; x < p_w; x++) {
        double d[3];
        pixel_dir(x, h / 2, p_w, h, d);
        const double r = p_weather.radius_m();
        const double pos[3] = { d[0] * r, 0.0, d[2] * r };
        p_weather.clouds_at(pos, column.data());
        for (int y = 0; y < sh; y++) {
            const double c = ov::clamp(double(column[size_t(levels - 1 - y / WeatherReport::SECTION_PX)]), 0.0, 1.0);
            uint8_t *o = &f.section_rgb[(size_t(y) * size_t(p_w) + size_t(x)) * 3];
            o[0] = byte(0.08 + 0.92 * c);
            o[1] = byte(0.12 + 0.88 * c);
            o[2] = byte(0.25 + 0.75 * c);
        }
    }
    ov::Value cloudy = ov::Value::array();
    for (int k = 0; k < levels; k++) {
        double share = 0.0, all = 0.0;
        for (int y = 0; y < h; y += 2) {
            for (int x = 0; x < p_w; x += 2) {
                double d[3];
                pixel_dir(x, y, p_w, h, d);
                const double r = p_weather.radius_m();
                const double pos[3] = { d[0] * r, d[1] * r, d[2] * r };
                p_weather.clouds_at(pos, column.data());
                const double a = std::sqrt(1.0 - d[1] * d[1]);
                share += column[size_t(k)] > 0.25f ? a : 0.0;
                all += a;
            }
        }
        cloudy.push_back(share / all);
    }
    ov::Value m = ov::Value::object();
    m["cloudy_levels"] = cloudy;
    m["game_s"] = p_weather.game_s();
    m["storms"] = int64_t(p_weather.storms());
    m["mean_cover"] = cover_sum / weight;
    m["max_wind_mps"] = wind_max;
    m["raining"] = raining / weight;
    f.numbers = m;
    return f;
}

bool write_png(const ov::str &p_path, int p_w, int p_h, const std::vector<uint8_t> &p_rgb) {
    const ov::bytes png = ov::encode_png(uint32_t(p_w), uint32_t(p_h), 3, p_rgb.data());
    return ov::write_file(p_path, ov::str(png.begin(), png.end()));
}

// p_world's weather mapped at p_minutes.
WeatherReport::Frames run_world(const WorldFile &p_world, const std::vector<double> &p_minutes, int p_width) {
    WeatherReport::Frames out;
    SDFCascade sdf;
    ChunkParam param;
    out.error = sdf.configure_world(p_world, param);
    if (out.error.empty() && !p_world.weather.on) {
        out.error = "weather: the world has none (not a sphere, or \"weather\": {\"on\": false}).";
    }
    if (!out.error.empty()) {
        return out;
    }
    const Climate::Table *climate = sdf.get_climate();
    const SDFCascade *cascade = &sdf;
    Weather::ForcingFn forcing;
    if (climate != nullptr) {
        forcing = [climate, cascade](const VoxelPos &p_voxel, int p_m) { return Weather::forcing_at(*climate, cascade->program(), p_voxel, p_m); };
    }
    Weather weather(p_world.weather, sdf.get_base(), p_world.voxel_m, p_world.seed, forcing);
    // The report's own clock: one tick of real time per update, each waited for.
    const uint64_t period_usec = uint64_t(std::llround(1e6 / p_world.weather.tick_hz));
    uint64_t now = 1;
    weather.update(now);
    weather.finish();
    weather.update(now);
    const double start_s = weather.game_s();
    out.width = ov::clamp(p_width, 64, 4096);
    out.height = out.width / 2;
    out.section_height = weather.levels() * WeatherReport::SECTION_PX;
    for (const double minutes : p_minutes) {
        while (weather.game_s() < start_s + minutes * 60.0 - 1e-6) {
            now += period_usec;
            weather.update(now); // asks for this period's tick
            weather.finish();
            weather.update(now); // collects it (the clock didn't move: no new tick)
        }
        out.frames.push_back(map_state(weather, out.width, 2.0 * ov::max(p_world.weather.prevailing_mps, 1.0)));
        out.frames.back().minutes = minutes;
    }
    out.stats = weather.get_stats();
    return out;
}

} // namespace

WeatherReport::Frames WeatherReport::frames(const ov::str &p_json, const ov::str &p_dir, const std::vector<double> &p_minutes, int p_width) {
    ov::Value data;
    WorldFile world;
    Frames out;
    out.error = WorldFile::parse_object(p_json, "the world's text", data);
    out.error = out.error.empty() ? WorldFile::parse(data, p_dir, world) : out.error;
    return out.error.empty() ? run_world(world, p_minutes, p_width) : out;
}

ov::Value WeatherReport::run(const ov::str &p_out_dir, const ov::str &p_world_path, const std::vector<double> &p_minutes, int p_width) {
    ov::Value report = ov::Value::object();
    report["ok"] = false;
    ov::make_dirs(p_out_dir);
    WorldFile world;
    Frames f;
    f.error = WorldFile::load(p_world_path, world);
    if (f.error.empty()) {
        f = run_world(world, p_minutes, p_width);
    }
    if (!f.error.empty()) {
        report["error"] = f.error;
        return report;
    }
    ov::Value maps = ov::Value::array();
    for (const Frame &frame : f.frames) {
        const ov::str tag = ov::format("%03d", int(std::lround(frame.minutes)));
        write_png(ov::path_join(p_out_dir, "cover_" + tag + ".png"), f.width, f.height, frame.cover_rgb);
        write_png(ov::path_join(p_out_dir, "wind_" + tag + ".png"), f.width, f.height, frame.wind_rgb);
        write_png(ov::path_join(p_out_dir, "section_" + tag + ".png"), f.width, f.section_height, frame.section_rgb);
        ov::Value m = frame.numbers;
        m["minutes"] = frame.minutes;
        maps.push_back(m);
    }
    const ov::Value &stats = f.stats;
    ov::str md = ov::format("# Weather report: %s\n\nGrid %s, cells of %g m (%d cells), time scale %g, %g ticks a second, storm radius %s, Omega %.4f rad/s, substep %.3f s.\n\n",
            p_world_path, stats["grid"].as_str(), stats["cell_m"].as_double(), int(stats["cells"].as_int()), world.weather.time_scale, world.weather.tick_hz,
            world.weather.storm_m > 0.0 ? ov::format("%g m", world.weather.storm_m) : ov::str("1/8 of the radius"), stats["omega_rad_s"].as_double(),
            stats["substep_s"].as_double());
    md += ov::format("%d levels up to %g m. Cloudy levels: each level's share of the planet with cloud past a quarter of full cover, the ground first.\n\n",
            int(stats["levels"].as_int()), stats["top_m"].as_double());
    md += "| game min | storms | mean cover | max wind (m/s) | raining | cloudy levels |\n|---|---|---|---|---|---|\n";
    for (size_t i = 0; i < maps.size(); i++) {
        const ov::Value &m = maps[i];
        ov::str levels;
        for (size_t k = 0; k < m["cloudy_levels"].size(); k++) {
            levels += ov::format("%s%.0f%%", k == 0 ? "" : " ", 100.0 * m["cloudy_levels"][k].as_double());
        }
        md += ov::format("| %g | %d | %.2f | %.1f | %.1f%% | %s |\n", m["minutes"].as_double(), int(m["storms"].as_int()), m["mean_cover"].as_double(),
                m["max_wind_mps"].as_double(), 100.0 * m["raining"].as_double(), levels);
    }
    md += "\nForcing: " + ov::to_json(stats["forcing_summary"]) + "\n";
    md += ov::format("\nForcing and spin-up: %.0f ms. Tick: %.2f ms on average, %.2f ms at most (%d ticks).\n", stats["forcing"]["total_ms"].as_double(),
            stats["tick"]["avg_ms"].as_double(), stats["tick"]["max_ms"].as_double(), int(stats["tick"]["count"].as_int()));
    ov::write_file(ov::path_join(p_out_dir, "report.md"), md);
    report["maps"] = maps;
    report["stats"] = stats;
    report["ok"] = true;
    return report;
}
