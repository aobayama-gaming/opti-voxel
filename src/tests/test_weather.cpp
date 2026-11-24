#include "test_climate_common.hpp"

#include "weather.hpp"

#include <cmath>

// The weather's worker (Weather, docs/weather.md "Worker"): the forcing from the climate, the clock, the states handed to
// the frame thread.

using ClimateTest::ClimateSDF;
using ClimateTest::json;

namespace {

WorldFile::Weather small_weather() {
    WorldFile::Weather w;
    w.on = true;
    w.grid = 16;
    w.tick_hz = 10.0;
    w.spin_up_s = 5.0;
    return w;
}

} // namespace

// Every biome pushing alike: the blend is that push wherever the weights fall; the default biome's own block counts too.
TEST_CASE("Weather::forcing_at: the biomes' weather blended by their weights") {
    ov::Value data;
    CHECK(load_table("res://climate/example.json", data) == ov::str());
    for (size_t b = 0; b < data["biomes"].size(); b++) {
        data["biomes"].at(b)["weather"] = json(R"({"wind": 2, "moisture": 1.2, "storms": 4})");
    }
    ClimateSDF same(data, 15);
    CHECK_OR_RETURN(same.sdf.get_climate() != nullptr);
    for (int k = 0; k < 8; k++) {
        const Climate::BiomeWeather w = Weather::forcing_at(*same.sdf.get_climate(), same.sdf.program(), VoxelPos(k * 70001, 0, -k * 50003), 9);
        CHECK(w.wind == doctest::Approx(2.0f));
        CHECK(w.moisture == doctest::Approx(1.2f));
        CHECK(w.storms == doctest::Approx(4.0f));
        CHECK(w.drag == doctest::Approx(Climate::BiomeWeather().drag));
    }
    // One biome wetter than the rest: every blend lies between the two.
    data["biomes"].at(0)["weather"] = json(R"({"moisture": 2})");
    ClimateSDF mixed(data, 15);
    CHECK_OR_RETURN(mixed.sdf.get_climate() != nullptr);
    for (int k = 0; k < 16; k++) {
        const Climate::BiomeWeather w = Weather::forcing_at(*mixed.sdf.get_climate(), mixed.sdf.program(), VoxelPos(k * 90001, 0, k * 30011), 9);
        CHECK(w.moisture >= 1.2f - 1e-5f);
        CHECK(w.moisture <= 2.0f + 1e-5f);
    }
}

// A clock of its own: the first state after the spin-up, then one tick a tenth of a second, the frame between the last
// two states; the weather read anywhere.
TEST_CASE("Weather: the worker's states follow the clock") {
    const BaseSurface base = BaseSurface::planet(2000.0, 0.05);
    Weather weather(small_weather(), base, 0.05, 3, Weather::ForcingFn());
    uint64_t now = 1000;
    weather.update(now);
    weather.finish();
    CHECK(weather.update(now)); // the spin-up's state
    CHECK(weather.ready());
    CHECK(weather.game_s() == doctest::Approx(5.0));
    for (int k = 0; k < 10; k++) {
        now += 100000; // a tick
        weather.update(now);
        weather.finish();
    }
    weather.update(now);
    CHECK(weather.game_s() == doctest::Approx(6.0));
    CHECK(weather.blend() >= 0.0f);
    CHECK(weather.blend() <= 1.0f);
    CHECK(weather.latest().size() == weather.grid().padded_cells() * Weather::LOOK);
    CHECK(weather.previous().size() == weather.latest().size());
    const double dir[3] = { 0.3, 0.8, -0.5 };
    Weather::Sample s;
    CHECK(weather.at(dir, s));
    CHECK(std::isfinite(s.wind_mps.x));
    CHECK(s.cover >= 0.0f);
    CHECK(s.cover <= 1.0f);
    const ov::Value stats = weather.get_stats();
    CHECK(stats["ready"].as_bool());
    CHECK(stats["tick"]["count"].as_int() == 10);
    CHECK(stats["behind_ticks"].as_int() <= Weather::MAX_BEHIND);
}

// A clock far ahead of the worker (frames without the worker keeping up) never asks for more than MAX_BEHIND ticks: the
// weather slows, it doesn't run away.
TEST_CASE("Weather: a clock far ahead drops ticks past MAX_BEHIND") {
    const BaseSurface base = BaseSurface::planet(2000.0, 0.05);
    Weather weather(small_weather(), base, 0.05, 3, Weather::ForcingFn());
    uint64_t now = 1000;
    weather.update(now);
    weather.finish();
    weather.update(now);
    for (int k = 0; k < 100; k++) {
        now += 250000; // 2.5 ticks a frame: 250 in all
        weather.update(now);
    }
    const ov::Value stats = weather.get_stats();
    CHECK(stats["behind_ticks"].as_int() <= Weather::MAX_BEHIND);
    CHECK(stats["tick"]["count"].as_int() + stats["dropped_ticks"].as_int() + stats["behind_ticks"].as_int() >= 240);
    weather.finish();
}
