#include "test_common.hpp"

#include "weather.hpp"
#include "weather_debug.hpp"

#include <cmath>

// The weather debugger's core (WeatherDebug, docs/weather.md "Debugger"): a probing weather keeps every field of its
// published state, laid out as the debugger reads it.

namespace {

WorldFile::Weather small_weather() {
    WorldFile::Weather w;
    w.on = true;
    w.cell_m = 196.0; // 16 cells a face side over 2 km
    w.tick_hz = 10.0;
    w.spin_up_s = 5.0;
    return w;
}

} // namespace

TEST_CASE("WeatherDebug: every field named, the level fields first") {
    for (int f = 0; f < WeatherDebug::FIELDS; f++) {
        CHECK(std::string(WeatherDebug::info(f).name).size() > 0);
        CHECK(WeatherDebug::per_level(f) == (f < WeatherDebug::GROUND_K));
    }
}

TEST_CASE("WeatherDebug::Snapshot: a probing weather's state, field by field, as the published one") {
    const BaseSurface base = BaseSurface::planet(2000.0, 0.05);
    Weather weather(small_weather(), base, 0.05, 3, Weather::ForcingFn(), true);
    uint64_t now = 1;
    weather.update(now);
    weather.finish();
    weather.update(now);
    for (int t = 0; t < 3; t++) {
        now += 100000;
        weather.update(now);
        weather.finish();
        weather.update(now);
    }
    const auto snap = weather.probe();
    CHECK(snap != nullptr);
    if (!snap) {
        return;
    }
    const WeatherGrid &g = weather.grid();
    CHECK(snap->faces == g.faces());
    CHECK(snap->width == g.width());
    CHECK(snap->levels == weather.levels());
    CHECK(snap->game_s == doctest::Approx(weather.game_s()));
    // The cloud water and the ground level's temperature are the published state's.
    const std::vector<float> &look = weather.latest();
    double worst_qc = 0.0, worst_t = 0.0;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const float *o = &look[g.index(f, i, j) * size_t(weather.stride())];
                const size_t in = snap->interior(f, i, j);
                for (int k = 0; k < snap->levels; k++) {
                    const double qc = double(snap->at(WeatherDebug::QC, in, k)) * 1e-3 / Weather::FULL_CLOUD_QC;
                    worst_qc = std::max(worst_qc, std::fabs(qc - double(o[Weather::LOOK + k])));
                }
                worst_t = std::max(worst_t, std::fabs(double(snap->at(WeatherDebug::TEMP_C, in, 0)) - double(o[Weather::TEMP])));
            }
        }
    }
    CHECK(worst_qc < 1e-4);
    CHECK(worst_t < 0.01);
    // Physical ranges: humid tropical air, a ground near the sounding's, relative humidity in (0, 1.2).
    const size_t mid = snap->interior(2, 8, 8);
    CHECK(snap->at(WeatherDebug::THETA, mid, 0) > 280.0f);
    CHECK(snap->at(WeatherDebug::THETA, mid, 0) < 320.0f);
    CHECK(snap->at(WeatherDebug::RH, mid, 0) > 0.0f);
    CHECK(snap->at(WeatherDebug::RH, mid, 0) < 1.2f);
    CHECK(snap->at(WeatherDebug::GROUND_K, mid, 0) > 280.0f);
    // A slice and a section read the same values as at().
    const std::vector<float> slice = snap->slice(WeatherDebug::QV, 1);
    CHECK(slice[mid] == snap->at(WeatherDebug::QV, mid, 1));
    const std::vector<float> section = snap->section(WeatherDebug::QV, 2, 8);
    CHECK(section.size() == size_t(snap->width * snap->levels));
    const int top = snap->levels - 1;
    CHECK(section[size_t(top * snap->width + 8)] == snap->at(WeatherDebug::QV, mid, top));
    CHECK(snap->section(WeatherDebug::WETNESS, 2, 8).size() == size_t(snap->width));
}

TEST_CASE("WeatherDebug: no snapshot unless probing") {
    const BaseSurface base = BaseSurface::planet(2000.0, 0.05);
    Weather weather(small_weather(), base, 0.05, 3, Weather::ForcingFn());
    weather.update(1);
    weather.finish();
    weather.update(1);
    CHECK(weather.ready());
    CHECK(weather.probe() == nullptr);
}

TEST_CASE("WeatherDebug::cloud_objects: clouds seen from above, grouped within a face") {
    WeatherDebug::Snapshot s;
    s.faces = 1;
    s.width = 8;
    s.height = 8;
    s.levels = 2;
    s.values.assign(size_t(WeatherDebug::LEVEL_FIELDS) * s.cells() * 2 + size_t(WeatherDebug::FIELDS - WeatherDebug::LEVEL_FIELDS) * s.cells(), 0.0f);
    auto set_qc = [&](int p_i, int p_j, int p_k) {
        const size_t at = size_t(WeatherDebug::QC) * s.cells() * 2 + s.interior(0, p_i, p_j) * 2 + size_t(p_k);
        s.values[at] = 0.5f;
    };
    // A 2 x 2 cloud on the lower level, a 3-column line on the upper, a lone column: 3 clouds of 8 columns.
    set_qc(1, 1, 0);
    set_qc(2, 1, 0);
    set_qc(1, 2, 0);
    set_qc(2, 2, 1);
    set_qc(5, 5, 1);
    set_qc(5, 6, 1);
    set_qc(5, 7, 0);
    set_qc(7, 0, 1);
    const WeatherDebug::CloudObjects o = WeatherDebug::cloud_objects(s, WeatherDebug::ALL, 0.01f);
    CHECK(o.count == 3);
    CHECK(o.cover == doctest::Approx(8.0 / 64.0));
    CHECK(o.largest_cells == 4);
    CHECK(o.mean_cells == doctest::Approx(8.0 / 3.0));
}
