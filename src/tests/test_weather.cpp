#include "test_climate_common.hpp"

#include "clouds.hpp"
#include "weather.hpp"
#include "weather_report.hpp"

#include "ov/files.hpp"
#include "ov/json.hpp"

#include "ov/half.hpp"

#include <cmath>

// The weather's worker (Weather, docs/weather.md "Worker"): the forcing from the climate, the clock, the states handed to
// the frame thread.

using ClimateTest::ClimateSDF;
using ClimateTest::json;

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
    CHECK(weather.levels() == 2); // 300 m over 196 m cells
    CHECK(weather.stride() == Weather::LOOK + 2);
    CHECK(weather.latest().size() == weather.grid().padded_cells() * size_t(weather.stride()));
    CHECK(weather.previous().size() == weather.latest().size());
    const double pos[3] = { 600.0, 1600.0, -1000.0 };
    Weather::Sample s;
    CHECK(weather.at(pos, s));
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

// The weather texture's texels (Clouds) are halves.
TEST_CASE("ov::float_to_half: the nearest half, back through half_to_float") {
    for (const float v : { 0.0f, 1.0f, 0.5f, 0.25f, 0.3f, 0.65f, 0.999f, 2.0f, 1e-6f }) {
        const float back = ov::half_to_float(ov::float_to_half(v));
        CHECK(std::abs(back - v) <= std::ldexp(std::abs(v), -11) + 6.2e-5f); // half an ulp; 0 under the normals
    }
    CHECK(ov::float_to_half(1.0f) == 0x3C00u);
    CHECK(ov::float_to_half(-2.0f) == 0xC000u);
    CHECK(ov::float_to_half(1e6f) == 0x7C00u);
}

// The clouds read the weather (Clouds' passes): its lookup is in their text, every placeholder substituted.
TEST_CASE("Clouds: the passes read the weather texture, every placeholder substituted") {
    const CubeSphereGrid grid(16, 2000.0);
    for (const ov::str &glsl : { Clouds::glsl_pass_march(grid), Clouds::glsl_pass_shadow(grid) }) {
        CHECK(glsl.find("vec2 weather_uv(vec3 p)") != ov::str::npos);
        CHECK(glsl.find("weather_uv(p)") != ov::str::npos);
        CHECK(glsl.find("cloud_weather_blend") != ov::str::npos);
        CHECK_MESSAGE(glsl.find('$') == ov::str::npos, glsl.substr(glsl.find('$'), 40));
        CHECK(glsl.find("cloud_map") == ov::str::npos);
    }
}

// The mixer's time-lapse (WeatherReport::frames): a planet's maps at each minute asked for, the world's text as the
// mixer sends it; no weather, no maps and why.
TEST_CASE("WeatherReport::frames: a planet's maps from its text, none without weather") {
    ov::str text;
    CHECK_OR_RETURN(ov::read_file("res://worlds/ocean_test.planet.json", text));
    ov::Value world;
    CHECK_OR_RETURN(WorldFile::parse_object(text, "ocean_test", world).empty());
    world["weather"] = json(R"({"cell_m": 196, "spin_up_s": 5})");
    const WeatherReport::Frames f = WeatherReport::frames(ov::to_json(world), "res://worlds", { 0.0, 0.1 }, 64);
    CHECK_MESSAGE(f.error.empty(), f.error);
    CHECK_OR_RETURN(f.frames.size() == 2);
    CHECK(f.width == 64);
    CHECK(f.height == 32);
    for (const WeatherReport::Frame &frame : f.frames) {
        CHECK(frame.cover_rgb.size() == size_t(64 * 32 * 3));
        CHECK(frame.wind_rgb.size() == frame.cover_rgb.size());
        CHECK(frame.numbers["mean_cover"].as_double() >= 0.0);
        CHECK(frame.numbers["mean_cover"].as_double() <= 1.0);
    }
    CHECK(f.frames[1].numbers["game_s"].as_double() == doctest::Approx(f.frames[0].numbers["game_s"].as_double() + 6.0));
    world["weather"] = json(R"({"on": false})");
    const WeatherReport::Frames off = WeatherReport::frames(ov::to_json(world), "res://worlds", { 0.0 }, 64);
    CHECK(off.error.find("has none") != ov::str::npos);
    CHECK(off.frames.empty());
}

// The parser and the mixer share one table of the weather block's keys and ranges.
TEST_CASE("Climate::WEATHER_KEYS: every key parses at its bounds, refused past them") {
    for (const Climate::WeatherKey &k : Climate::WEATHER_KEYS) {
        for (const float v : { k.min, k.max }) {
            ov::Value data;
            CHECK(load_table("res://climate/example.json", data) == ov::str());
            data["biomes"].at(0)["weather"] = json(ov::format(R"({"%s": %g})", k.name, double(v)));
            ClimateSDF ok(data, 15);
            CHECK_OR_RETURN(ok.sdf.get_climate() != nullptr);
            CHECK(ok.sdf.get_climate()->biomes[0].weather.*k.member == doctest::Approx(v));
        }
        ov::Value data;
        CHECK(load_table("res://climate/example.json", data) == ov::str());
        data["biomes"].at(0)["weather"] = json(ov::format(R"({"%s": %g})", k.name, double(k.max) + 1.0));
        ClimateSDF bad(data, 15);
        CHECK_MESSAGE(bad.sdf.get_climate() == nullptr, k.name);
    }
}
