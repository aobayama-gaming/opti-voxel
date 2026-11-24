#pragma once

#include "climate.hpp"
#include "weather_grid.hpp"

#include <cstdint>
#include <memory>
#include <vector>

// The weather's simulation (docs/weather.md "Simulation"): the air over the whole planet in levels over a WeatherGrid,
// hydrostatic and rotating, carrying vapour and cloud water, forced by the climate. Engine-free, on the CPU,
// deterministic: a fixed dt, every pass writes each column from the previous state only (any thread count, the same bits).
// - Levels: `levels` slabs of top_m / levels from the ground up to top_m, level k centred at (k + 1/2) top_m / levels.
//   The shallow-atmosphere approximation: a level's cells share their column's surface metric.
// - The state per padded cell: per level (Field) the wind, a 3D vector tangent to the surface in the planet's axes
//   (no per-face frames at the seams), m/s; TEMP the air's potential temperature anomaly (about [-1, 1]); VAPOUR
//   (saturation 1 at the ground) and CLOUD the cloud water; then the column's ETA (its top's height anomaly: the
//   surface pressure, c^2 ETA). Diagnostics per padded cell (Diag): the column's cover and type, its rain.
// - A substep (step() takes as many as the gravity waves' CFL asks):
//   1. Each level's air comes from upstream (semi-Lagrangian: the level sampled at the point its wind brings here), the
//      Coriolis turns the wind (an exact rotation about the up by -f dt, f = 2 Omega n_y), the wind relaxes towards the
//      prevailing wind times the biome's `wind`. The ground touches the lowest level only: it drags it (quadratic, times
//      `drag`), warms it towards `heat` and gives it vapour up to GROUND_HUMIDITY `moisture`; the levels above relax
//      towards an inversion (INVERSION_TEMP).
//   2. The pressure pushes each level's wind: -grad(phi), phi = c^2 ETA - BUOYANCY (the warmth above the level, the
//      level's half included; hydrostatic: a warm column is a low at the ground).
//   3. The column: the top follows the column's mean divergence; the levels' divergences make the vertical wind through
//      their interfaces (zero at the ground), which carries every field upwind between levels; convection: a parcel of
//      the ground level's air rises while it is warmer than the level it reaches (CONVECT_EXCESS of overshoot), past
//      its saturation it condenses and its latent heat carries it higher (a cumulus tower), and the levels it rises
//      through mix their air towards its mean (CONVECT_TAU_S; the temperature stays the forcing's profile; cloud mixed
//      into the drier levels below evaporates, so each level holds the layer's water less its saturation), and above its top the air sinks and dries (DRY_TAU_S): clouds
//      live in the convecting layer, as deep as the ground's warmth (a storm's) carries it; a little horizontal
//      diffusion. Vapour past the level's saturation
//      (1 + SATURATION_HEAT TEMP - SATURATION_LAPSE altitude / top_m: rising air condenses) becomes cloud and warms
//      the air (LATENT), cloud under it evaporates and cools it, cloud past RAIN_CLOUD rains out. The cover is the
//      column's largest cloud over FULL_CLOUD, the type its cloudy depth (stratus 0, a tower 1).
// - Storms: in each substep a cell starts one with the chance its `storms` gives (a whole planet of storms 1 starts one
//   an hour; hash of the seed, the tick and the cell), a warm, moist low of the storm radius that the Coriolis spins up.
// - The prevailing wind: east (y the pole axis) times -sin(6 |latitude|): trade easterlies peaking at 15 degrees,
//   westerlies at 45, polar easterlies at 75, calm at the equator and the poles.
class WeatherSim {
public:
    enum Field {
        WIND_X,
        WIND_Y,
        WIND_Z,
        TEMP,
        VAPOUR,
        CLOUD,
        FIELDS
    };
    enum Diag {
        COVER,
        TYPE,
        RAIN,
        DIAGS
    };
    static constexpr int MAX_LEVELS = 32;

    // The gravity waves' speed (m/s): the pressure's stiffness; the storm radius is their Rossby radius at 45 degrees.
    static constexpr double WAVE_MPS = 20.0;
    // Warm air's low: a unit of TEMP over the whole column pushes like this much c^2 ETA (m^2/s^2).
    static constexpr double BUOYANCY = 20.0;
    // Relaxation times (game seconds): the wind to its target, the ground level's temperature to the biome's, the levels
    // above to the inversion, the vapour and cloud above the thermals' top to none (the sinking air dries them), the ground's vapour up to its moisture, condensation, the cloud's evaporation, the rain-out, the
    // mixing between levels, an overturning column.
    static constexpr double WIND_TAU_S = 900.0;
    static constexpr double HEAT_TAU_S = 600.0;
    static constexpr double RADIATE_TAU_S = 1800.0;
    static constexpr double DRY_TAU_S = 300.0;
    static constexpr double MOISTURE_TAU_S = 120.0;
    static constexpr double CONDENSE_TAU_S = 20.0;
    static constexpr double EVAPORATE_TAU_S = 10.0;
    static constexpr double RAIN_TAU_S = 120.0;
    static constexpr double CONVECT_TAU_S = 30.0;
    // A thermal from the ground rises through the levels it is at least this much cooler than (its overshoot).
    static constexpr double CONVECT_EXCESS = 0.05;
    // The air aloft relaxes to INVERSION_TEMP times its altitude over top_m: warmer with height (an inversion), so the
    // ground's convection stops where its air is no longer the warmer (cloud tops); a storm's warmth and the latent heat
    // carry it through.
    static constexpr double INVERSION_TEMP = 0.3;
    // Quadratic drag per metre at drag 1: 10 m/s of wind loses it over ~1000 s.
    static constexpr double DRAG_PER_M = 1e-4;
    // The ground's vapour: its moisture times this (the air over water is not saturated).
    static constexpr double GROUND_HUMIDITY = 0.75;
    // The saturation: 1 + SATURATION_HEAT TEMP - SATURATION_LAPSE altitude / top_m.
    static constexpr double SATURATION_HEAT = 0.3;
    static constexpr double SATURATION_LAPSE = 0.9;
    // Condensing a unit of vapour warms a rising parcel by LATENT (convect: its buoyancy, not the air round it).
    static constexpr double LATENT = 0.1;
    // Cloud water raining out over RAIN_CLOUD; the cover full at FULL_CLOUD.
    static constexpr double RAIN_CLOUD = 0.3;
    static constexpr double FULL_CLOUD = 0.2;
    // Diffusion (m^2/s) per metre of cell and m/s of wave: a stabiliser, not a physical term.
    static constexpr double DIFFUSION = 0.02;
    // A storm: its low (ETA), its warmth (TEMP, at the ground, fading to 0 at the top) and its vapour at the centre (the
    // lower half of the column), a Gaussian of the storm radius.
    static constexpr double STORM_ETA = -0.3;
    static constexpr double STORM_TEMP = 0.5;
    static constexpr double STORM_VAPOUR = 0.4;
    // A whole planet of storms 1 starts one in STORM_PERIOD_S game seconds.
    static constexpr double STORM_PERIOD_S = 3600.0;

    struct Params {
        double radius_m = 2000.0;
        double prevailing_mps = 8.0;
        double storm_m = 0.0; // 0: a quarter of the radius
        uint32_t seed = 0;
        int levels = 1; // in [1, MAX_LEVELS]
        double top_m = 300.0;
    };

    // p_weather: each interior cell's biome weather (WeatherGrid::interior), the climate under its centre.
    WeatherSim(std::shared_ptr<const WeatherGrid> p_grid, const Params &p_params, const std::vector<Climate::BiomeWeather> &p_weather);

    // p_dt game seconds, in substeps of at most the CFL's.
    void step(double p_dt);
    // A storm of p_strength (1: a biome storm's) centred under a position (metres from the base's centre).
    void inject_storm(const double p_pos[3], double p_strength);

    const WeatherGrid &grid() const { return *cells; }
    int levels() const { return level_count; }
    // A level's thickness and its centre's altitude (metres).
    double level_thickness_m() const { return dz; }
    double level_m(int p_level) const { return (p_level + 0.5) * dz; }
    // The state's floats per padded cell, a level's fields' offset in it and the column's ETA's.
    int stride() const { return level_count * FIELDS + 1; }
    static int level_offset(int p_level) { return p_level * FIELDS; }
    int eta_offset() const { return level_count * FIELDS; }
    // The state (stride() floats per padded cell) and the diagnostics (DIAGS), halos filled.
    const std::vector<float> &state() const { return cur; }
    const std::vector<float> &diagnostics() const { return diag; }
    // The Coriolis's Omega (rad/s, from the storm radius) and the substep's largest dt (game seconds).
    double omega() const { return omega_rad_s; }
    double max_substep_s() const { return substep_s; }
    uint64_t ticks() const { return tick; }
    uint64_t storms_started() const { return storms; }
    double time_s() const { return time; }

    // The wind's Coriolis turn: p_wind rotated about the unit up p_up by -p_f p_dt (exact, the speed kept).
    static void coriolis(const double p_up[3], double p_f, double p_dt, double r_wind[3]);
    // The prevailing wind's target at a unit up for p_speed m/s (east times -sin(6 |latitude|)).
    static void prevailing(const double p_up[3], double p_speed, double r_wind[3]);

private:
    // Per interior cell: its surface point (metres), its up, the dual vectors of its central differences (grad = ax d_i +
    // ay d_j, 1/m) and its forcing.
    struct Cell {
        double pos[3];
        float up[3];
        float ax[3];
        float ay[3];
        float target[3]; // the prevailing wind times the biome's wind
        float f; // the Coriolis parameter 2 Omega n_y
        Climate::BiomeWeather weather;
    };

    std::shared_ptr<const WeatherGrid> cells;
    Params params;
    int level_count = 1;
    double dz = 300.0;
    std::vector<Cell> cell;
    std::vector<float> cur;
    std::vector<float> next;
    std::vector<float> phi; // pass_pressure's: per padded cell and level
    std::vector<float> diag;
    double omega_rad_s = 0.0;
    double substep_s = 1.0;
    double diffusion_k = 0.0; // per substep of substep_s, of a cell's difference to its neighbours' mean
    double time = 0.0;
    uint64_t tick = 0;
    uint64_t storms = 0;

    void make_cells(const std::vector<Climate::BiomeWeather> &p_weather);
    void pass_air(double p_dt);
    void pass_pressure(double p_dt);
    void pass_column(double p_dt);
    void start_storms(double p_dt, uint32_t p_substep);
    // pass_column's vertical steps on one column's levels (v: levels x FIELDS, the divergences div).
    void lift(double p_dt, const double *p_div, double *r_v) const;
    void convect(double p_dt, double *r_v) const;
    // A level's saturation for air of p_temp.
    double saturation(int p_level, double p_temp) const;
    // The condensation, the evaporation and the rain of one column (r_rain its rate), its cover and type.
    void condense(double p_dt, double *r_v, double &r_rain, double &r_cover, double &r_type) const;
};
