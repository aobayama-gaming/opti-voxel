#pragma once

#include "climate.hpp"
#include "weather_moist.hpp"

#include "ov/value.hpp"

#include <cstdint>
#include <vector>

// What drives the weather from outside the air (docs/weather.md "Forcing"): the climate's 2D ground under each column
// and the planet's large scale. Engine-free; relief ignored (the SDF as the ground boundary comes later).
// - The ground (WeatherMoist::Ground) per column: its biomes' blend of ground_k, wetness and z0_m, warmed by day and
//   cooled by night: ground_k + diurnal_k (max(up . sun, 0) - 1 / pi) (the day's mean is ground_k; the sun keeps the
//   game's day).
// - The large-scale wind: the pressure gradient that holds the prevailing wind u_g in geostrophic balance, f up x u_g
//   added to every level (u_g: east times -sin(6 |latitude|), trade easterlies at 15 degrees, westerlies at 45, y the
//   pole axis); the Coriolis turns the rest. f = 2 Omega up_y, Omega the weather's day (the twist: a day of minutes, so
//   storms spin at the game's scale).
// - Storms' seeds: in each step a column starts one with the chance its biomes' storms give (a whole planet of storms 1
//   starts one an hour; a hash of the seed, the step and the column): a warm, moist bubble of storm_m in the lower half
//   of the levels. What it grows into is the physics': the seed only triggers it.
// - The sounding the air starts from and is buoyant against: a trade-wind boundary layer (BOMEX's shape, Siebesma et
//   al. 2003) scaled to the inversion: a mixed layer AIR_BELOW_GROUND_K under the grounds' mean, its vapour
//   saturating at the mixed layer's top (the cloud base), a conditionally unstable cloud layer, a 4 K inversion, dry stable air above.
class WeatherForcing {
public:
    static constexpr double STORM_PERIOD_S = 3600.0;
    static constexpr double STORM_THETA_K = 2.0;
    static constexpr double STORM_QV = 0.002;
    static constexpr double AIR_BELOW_GROUND_K = 1.5; // the mixed layer under the grounds' mean (BOMEX: 1.7)
    static constexpr double CLOUD_BASE = 0.53; // the cloud layer's base over the inversion's height
    static constexpr double INVERSION_DEPTH_M = 40.0;
    static constexpr double INVERSION_K = 4.0;
    static constexpr double CLOUD_LAPSE_K_PER_M = 0.004; // theta, the cloud layer's
    static constexpr double ABOVE_LAPSE_K_PER_M = 0.003; // theta, above the inversion
    static constexpr double CLOUD_DRYING_PER_M = 1.4e-5; // q_v
    static constexpr double ABOVE_HUMIDITY = 0.45; // q_v above the inversion over the mixed layer's
    static constexpr double NOISE_K = 0.1; // the two lowest levels' start

    struct Params {
        double day_s = 300.0; // the weather's day (Omega = 2 pi / day_s)
        double prevailing_mps = 5.0;
        double storm_m = 0.0; // 0: 1 / 8 of the radius
        double inversion_m = 300.0;
        uint32_t seed = 0;
    };

    // p_climate: each interior cell's biome weather (Weather::forcing_at).
    WeatherForcing(const WeatherFluid &p_fluid, const Params &p_params, std::vector<Climate::BiomeWeather> p_climate);

    double omega() const { return 2.0 * 3.14159265358979323846 / params.day_s; }
    // The sounding of the grounds' mean temperature (WeatherMoist::Sounding).
    WeatherMoist::Sounding sounding(const WeatherFluid &p_fluid) const;
    // The fluid's start: the sounding, the prevailing wind, a little noise on the lowest levels.
    void start(WeatherFluid &p_fluid, const WeatherMoist::Sounding &p_sounding) const;
    // Each column's ground under the sun (towards it, planet axes).
    void grounds(const Float3 &p_towards_sun, std::vector<WeatherMoist::Ground> &r_ground) const;
    // The large-scale wind and the storms' seeds for p_dt of step p_step; the storms started so far.
    void apply(WeatherFluid &p_fluid, double p_dt, uint64_t p_step);
    uint64_t storms() const { return storms_started; }
    // The forcing over the planet: the grounds' temperatures (min, area mean, max), the wet share (wetness 0.9 and
    // more), the sounding's mixed layer, per hemisphere (y > 0 the north).
    ov::Value summary(const WeatherFluid &p_fluid) const;
    // A storm's seed under a position (metres from the base's centre).
    void seed_storm(WeatherFluid &p_fluid, const double p_pos[3]) const;

    // The prevailing wind at a unit up for p_speed m/s.
    static void prevailing(const double p_up[3], double p_speed, double r_wind[3]);

private:
    Params params;
    std::vector<Climate::BiomeWeather> climate;
    std::vector<double> ups; // 3 per column
    double mean_ground_k = 299.0;
    bool any_storms = false;
    uint64_t storms_started = 0;
};
