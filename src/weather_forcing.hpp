#pragma once

#include "climate.hpp"
#include "weather_moist.hpp"

#include "ov/value.hpp"

#include <cstdint>
#include <vector>

// What drives the weather from outside the air (docs/weather.md "Forcing"): the climate's 2D ground under each column
// and the planet's large scale. Engine-free; relief ignored (the SDF as the ground boundary comes later).
// - The sun turns round the pole (y) once a weather day, west as the planet spins east (turn_sun): the planet's day is
//   the Coriolis' (one Omega).
// - The ground (WeatherMoist::Ground) per column: its biomes' blend of ground_k, wetness and z0_m, its temperature a
//   baseline, which the sun sets directly: ground_k + sun_k (L - PLANET_LIGHT), L = cos(latitude) / pi the column's
//   year's mean of max(up . sun, 0) (taken as the equinox's day), PLANET_LIGHT its mean over the sphere, so the
//   planet's mean stays the climate's: a warm equator, cold poles, sea and land alike (a long mean does not depend on
//   the heat a ground holds). The day swings it round that: a force-restore slab (Deardorff 1978), dT/dt = 2 sqrt(pi) Q
//   / (I sqrt(P)) - 2 pi / P (T - baseline), P the day, I the biome's thermal inertia sqrt(k rho c), Q the sunlight it
//   absorbs over its baseline's: SUN_W_M2 (1 - GROUND_ALBEDO) ((1 - A) max(up . sun, 0) - L), A the albedo of the cloud
//   over it (its water path, droplets of DROPLET_RADIUS_M, Stephens 1978's two-stream). Land warms into the afternoon
//   (the slab lags the sun by an eighth of a day), a sea (an inertia of ~1e5: its mixed layer) barely moves, a cloud's
//   shade cools what is under it. 2026-11-26: the planet had no warm equator nor cold pole (each column's day's mean
//   was held at ground_k), so nothing made a wind larger than a thermal.
// - The air relaxes to its column's equilibrium, the start's (equilibrium()), over relax_days weather days: its theta
//   (the air's radiation, Held & Suarez 1994's Newtonian relaxation; they take 40 days aloft, a game's weather has
//   minutes) and its vapour (the exchange with the air over the lid: the boundary layer's moisture carried up, dry
//   air sinking in, BOMEX's large-scale drying). Measured on the fix world (2026-11-26): with neither, the cover fell
//   0.31 -> 0.17 in 20 game minutes (the circulation mixes the sun's gradient away); with theta alone it stayed ~0.3
//   for 30 minutes, then the seas' vapour, which only rain took out, filled the lidded layer: 0.79 and fog over a
//   quarter of the ground at 90.
// - No large-scale wind is imposed (2026-11-26: Earth's three cells, put in as a balanced pressure gradient, held the
//   air in steady bands round the axis and a cloud ring under the polar jet): the air starts at rest and every wind is
//   the sun's heating, the convection and the storms', turned by the Coriolis (f = 2 Omega up_y, Omega the weather's
//   day: a day of minutes, so storms spin at the game's scale).
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
    static constexpr double SUN_W_M2 = 1000.0; // the clear sky's sunlight at the ground, the sun overhead
    static constexpr double GROUND_ALBEDO = 0.15;
    static constexpr double PLANET_LIGHT = 0.25; // a sphere's mean of max(up . sun, 0), the sun anywhere
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double DROPLET_RADIUS_M = 1e-5; // the clouds' effective radius
    static constexpr double CLOUD_ASYMMETRY = 0.85; // g, the droplets' forward scattering
    static constexpr double AIR_DENSITY = 1.15; // kg/m^3, the cloud layer's (the sounding's at the ground; 6% less at 600 m)

    struct Params {
        double day_s = 300.0; // the weather's day (Omega = 2 pi / day_s)
        double sun_k = 40.0; // the baseline's warming per unit of a year's mean light (an equator-pole of sun_k / pi)
        double relax_days = 2.0; // the air's relaxation to its equilibrium (theta and vapour), in weather days
        double storm_m = 0.0; // 0: 1 / 8 of the radius
        double inversion_m = 300.0;
        uint32_t seed = 0;
    };

    // p_climate: each interior cell's biome weather (Weather::forcing_at).
    WeatherForcing(const WeatherFluid &p_fluid, const Params &p_params, std::vector<Climate::BiomeWeather> p_climate);

    // Column p_c's year's mean light (cos(latitude) / pi) and its ground's baseline (ground_k + sun_k (that - 1/4)).
    double year_light(size_t p_c) const;
    double baseline_k(size_t p_c) const;
    double omega() const { return 2.0 * 3.14159265358979323846 / params.day_s; }
    // The sounding of the grounds' mean temperature (WeatherMoist::Sounding).
    WeatherMoist::Sounding sounding(const WeatherFluid &p_fluid) const;
    // The fluid's start: the sounding moved to each column's baseline (warmer or colder by its baseline over the
    // planet's mean, its vapour keeping its relative humidity), the air at rest, a little noise on the lowest levels.
    void start(WeatherFluid &p_fluid, const WeatherMoist::Sounding &p_sounding) const;
    // Each column's ground at its baseline, its wetness and roughness.
    void grounds(std::vector<WeatherMoist::Ground> &r_ground) const;
    // The grounds' temperatures p_dt on under the sun (towards it, planet axes) and the cloud over them (p_rho: the
    // air's density per level, the sounding's; p_extra_cloud, null or per WeatherFluid::cell_index: cloud water the grid
    // does not hold, the updrafts').
    void warm(const Float3 &p_towards_sun, double p_dt, const WeatherFluid &p_fluid, const std::vector<double> &p_rho, const float *p_extra_cloud,
            std::vector<WeatherMoist::Ground> &r_ground);
    // A thermal inertia's slab (force-restore over a day of p_day_s) p_dt on, absorbing p_q W/m^2 over its day's mean,
    // restored to p_mean_k.
    static double warm_slab(double p_temp_k, double p_mean_k, double p_inertia, double p_q, double p_day_s, double p_dt);
    // The albedo of a cloud of p_lwp kg/m^2 of water.
    static double cloud_albedo(double p_lwp);
    // The extinction (1/m) of p_lwc kg/m^3 of cloud water: 3 LWC / (2 rho_w r_e) (droplets far larger than the light's
    // wavelength: an extinction efficiency of 2; Hansen & Travis 1974, Stephens 1978). Through a path the opacity is
    // 1 - exp(-integral of it), the visibility 3.912 / it (Koschmieder); 0.3 g/kg of air at AIR_DENSITY: 0.052 /m, ~75 m
    // (Kunkel 1984's fog fit, 144.7 LWC^0.88 /km in g/m^3: 0.057 /m).
    static double cloud_extinction(double p_lwc);
    // p_sun turned round the pole (y) by p_angle (radians, right-handed: the planet's spin turns it by -Omega t).
    static Float3 turn_sun(const Float3 &p_sun, double p_angle);
    // The large-scale wind and the storms' seeds for p_dt of step p_step; the storms started so far.
    void apply(WeatherFluid &p_fluid, double p_dt, uint64_t p_step);
    // Column p_c's equilibrium against p_reference: its theta offset (its baseline over the planet's mean) and its
    // vapour's factor (the saturation's ratio at the ground over that offset: the same relative humidity).
    void equilibrium(size_t p_c, const WeatherMoist::Sounding &p_reference, double &r_offset_k, double &r_vapour) const;
    // The air p_dt on towards its equilibrium, theta and vapour (implicit: any dt is stable).
    void relax(WeatherFluid &p_fluid, const WeatherMoist::Sounding &p_reference, double p_dt) const;
    uint64_t storms() const { return storms_started; }
    // Each interior cell's biome weather (the constructor's).
    const std::vector<Climate::BiomeWeather> &cell_climate() const { return climate; }
    // The forcing over the planet: the grounds' temperatures (min, area mean, max), the wet share (wetness 0.9 and
    // more), the sounding's mixed layer, per hemisphere (y > 0 the north).
    ov::Value summary(const WeatherFluid &p_fluid) const;
    // The grounds' mean temperature (K) on the sun's side (r_day) and the other (r_night), area-weighted.
    void day_night(const WeatherFluid &p_fluid, const Float3 &p_towards_sun, const std::vector<WeatherMoist::Ground> &p_ground, double &r_day,
            double &r_night) const;
    // A storm's seed under a position (metres from the base's centre).
    void seed_storm(WeatherFluid &p_fluid, const double p_pos[3]) const;

private:
    Params params;
    std::vector<Climate::BiomeWeather> climate;
    std::vector<double> ups; // 3 per column
    double mean_ground_k = 299.0;
    bool any_storms = false;
    uint64_t storms_started = 0;
};
