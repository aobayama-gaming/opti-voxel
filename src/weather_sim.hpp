#pragma once

#include "climate.hpp"
#include "weather_grid.hpp"

#include <cstdint>
#include <vector>

// The weather's simulation (docs/weather.md "Simulation"): one layer of rotating shallow water over the whole planet on
// a WeatherGrid, carrying vapour and cloud water, forced by the climate. Engine-free, on the CPU, deterministic: a fixed
// dt, every pass writes each cell from the previous state only (any thread count, the same bits).
// - The state per padded cell (Field): the wind, a 3D vector tangent to the sphere in the planet's axes (no per-face
//   frames at the seams), m/s; ETA the layer's thickness anomaly (the pressure: c^2 ETA); TEMP the air's temperature
//   anomaly ([-1, 1] about the biome's); VAPOUR (saturation 1) and CLOUD the cloud water. Diagnostics per padded cell
//   (Diag): the cover and type the clouds draw, the rain.
// - A substep (step() takes as many as the gravity waves' CFL asks):
//   1. Each cell's air comes from upstream (semi-Lagrangian: the state sampled at the point the wind brings here), the
//      Coriolis turns the wind (an exact rotation about the up by -f dt, f = 2 Omega n_y), the wind relaxes towards the
//      prevailing wind times the biome's `wind`, the ground drags it (quadratic, times `drag`), the air's temperature
//      relaxes to `heat`, the ground gives vapour up to `moisture`.
//   2. The pressure pushes the wind: -grad(c^2 ETA - BUOYANCY TEMP) (warm air: a low).
//   3. The layer's thickness follows the wind's divergence; converging air rises, which lowers the saturation: vapour
//      over it condenses into cloud, cloud under it evaporates, cloud past RAIN_CLOUD rains out; a little diffusion. The
//      cover is the cloud water over FULL_CLOUD, the type rises with the lift and the heat (stratus 0, cumulus 1).
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
        ETA,
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

    // The gravity waves' speed (m/s): the pressure's stiffness; the storm radius is their Rossby radius at 45 degrees.
    static constexpr double WAVE_MPS = 20.0;
    // Warm air's low: a unit of TEMP pushes like this much c^2 ETA (m^2/s^2).
    static constexpr double BUOYANCY = 20.0;
    // Relaxation times (game seconds): the wind to its target, the air's temperature to the biome's, the vapour up to its
    // moisture, condensation, the cloud's evaporation, the rain-out.
    static constexpr double WIND_TAU_S = 900.0;
    static constexpr double HEAT_TAU_S = 600.0;
    static constexpr double MOISTURE_TAU_S = 300.0;
    static constexpr double CONDENSE_TAU_S = 20.0;
    static constexpr double EVAPORATE_TAU_S = 60.0;
    static constexpr double RAIN_TAU_S = 120.0;
    // Quadratic drag per metre at drag 1: 10 m/s of wind loses it over ~1000 s.
    static constexpr double DRAG_PER_M = 1e-4;
    // The saturation: 1 + SATURATION_HEAT TEMP - SATURATION_LIFT lift, the lift the convergence times LIFT_S, in [0, 1].
    static constexpr double SATURATION_HEAT = 0.3;
    static constexpr double SATURATION_LIFT = 0.4;
    static constexpr double LIFT_S = 200.0;
    // Cloud water raining out over RAIN_CLOUD; the cover full at FULL_CLOUD.
    static constexpr double RAIN_CLOUD = 0.3;
    static constexpr double FULL_CLOUD = 0.2;
    // Diffusion (m^2/s) per metre of cell and m/s of wave: a stabiliser, not a physical term.
    static constexpr double DIFFUSION = 0.02;
    // A storm: its low (ETA), its warmth (TEMP) and its vapour at the centre, a Gaussian of the storm radius.
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
    };

    // p_weather: each interior cell's biome weather (cell index face * N^2 + j * N + i), the climate under its centre.
    WeatherSim(const WeatherGrid &p_grid, const Params &p_params, const std::vector<Climate::BiomeWeather> &p_weather);

    // p_dt game seconds, in substeps of at most the CFL's.
    void step(double p_dt);
    // A storm of p_strength (1: a biome storm's) centred on a direction.
    void inject_storm(const double p_dir[3], double p_strength);

    const WeatherGrid &grid() const { return cells; }
    // The state and the diagnostics, FIELDS and DIAGS floats per padded cell, halos filled.
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
    // Per interior cell: its up, the dual vectors of its central differences (grad = ax d_i + ay d_j, 1/m) and its forcing.
    struct Cell {
        float up[3];
        float ax[3];
        float ay[3];
        float target[3]; // the prevailing wind times the biome's wind
        float f; // the Coriolis parameter 2 Omega n_y
        Climate::BiomeWeather weather;
    };

    WeatherGrid cells;
    Params params;
    std::vector<Cell> cell;
    std::vector<float> cur;
    std::vector<float> next;
    std::vector<float> diag;
    double omega_rad_s = 0.0;
    double substep_s = 1.0;
    double diffusion_k = 0.0; // per substep of substep_s, of a cell's difference to its neighbours' mean
    double time = 0.0;
    uint64_t tick = 0;
    uint64_t storms = 0;

    size_t interior(int p_face, int p_i, int p_j) const;
    void make_cells(const std::vector<Climate::BiomeWeather> &p_weather);
    void substep(double p_dt);
    void pass_air(double p_dt);
    void pass_pressure(double p_dt);
    void pass_column(double p_dt);
    void start_storms(double p_dt, uint32_t p_substep);
};
