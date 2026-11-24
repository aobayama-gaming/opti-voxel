#pragma once

#include "ov/str.hpp"
#include "ov/value.hpp"

#include <cstdint>
#include <vector>

// The weather report (tools/weather_report.sh, docs/weather.md "Report") and the mixer's time-lapse: a world's weather
// run on the CPU alone (no GPU, no window) on a clock of its own, mapped at the game minutes asked for. Each map is the
// whole planet, equirectangular (longitude across, the +y pole at the top): the cover (grey, the rain blue) and the wind
// (the direction it blows to as a hue, east red, north yellow-green, west cyan, south violet; its speed as the
// brightness, full at twice the prevailing wind), and a cross-section along the equator (longitude across, the levels
// up from the ground, SECTION_PX rows each: the cloud water, white at full cover, the rain blue).
namespace WeatherReport {

constexpr int SECTION_PX = 12;

struct Frame {
    double minutes = 0.0; // after the spin-up
    std::vector<uint8_t> cover_rgb; // width x height x 3
    std::vector<uint8_t> wind_rgb;
    std::vector<uint8_t> section_rgb; // width x section_height x 3
    ov::Value numbers; // game_s, storms, mean_cover, max_wind_mps, raining (the share of the planet), cloudy_levels (each
                       // level's share of the planet with cloud past a quarter of full cover)
};

struct Frames {
    ov::str error; // the world has no weather (not a sphere, or off), or its file was refused
    int width = 0;
    int height = 0;
    int section_height = 0;
    std::vector<Frame> frames;
    ov::Value stats; // Weather::get_stats after the last frame
};

// The world file's text p_json (its relative paths from p_dir) mapped at p_minutes (ascending), p_width pixels wide.
Frames frames(const ov::str &p_json, const ov::str &p_dir, const std::vector<double> &p_minutes, int p_width);

// The world file at p_world_path mapped into p_out_dir (cover_<min>.png, wind_<min>.png, report.md: per map the ticks,
// storms, mean cover, largest wind and the share raining; the forcing's and the tick's times); the report's numbers.
ov::Value run(const ov::str &p_out_dir, const ov::str &p_world_path, const std::vector<double> &p_minutes, int p_width);

} // namespace WeatherReport
