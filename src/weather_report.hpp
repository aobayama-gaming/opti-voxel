#pragma once

#include "ov/str.hpp"
#include "ov/value.hpp"

#include <vector>

// The weather report (tools/weather_report.sh, docs/weather.md "Report"): a world file's weather run on the CPU alone (no
// GPU, no window) on a clock of its own, mapped at the game minutes asked for. Each map is the whole planet,
// equirectangular (longitude across, the +y pole at the top): cover_<min>.png (the cover grey, the rain blue) and
// wind_<min>.png (the direction the wind blows to as a hue, east red, north yellow-green, west cyan, south violet; its speed
// as the brightness, full at twice the prevailing wind). report.md: per map the ticks, storms, mean cover, largest wind and
// the share of the planet raining; the forcing's and the tick's times.
namespace WeatherReport {

// Writes into p_out_dir; the report's numbers (its "error" when the world has no weather: not a sphere, or off).
ov::Value run(const ov::str &p_out_dir, const ov::str &p_world_path, const std::vector<double> &p_minutes, int p_width);

} // namespace WeatherReport
