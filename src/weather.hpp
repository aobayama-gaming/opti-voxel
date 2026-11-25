#pragma once

#include "debug_log.hpp"
#include "level_program.hpp"
#include "weather_convection.hpp"
#include "weather_forcing.hpp"
#include "world_file.hpp"

namespace WeatherDebug {
struct Snapshot;
}

#include "ov/executor.hpp"
#include "ov/value.hpp"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

// The weather over a planet (docs/weather.md "Worker"): the fluid (WeatherFluid), its moist physics (WeatherMoist) and
// its forcing (WeatherForcing) on a worker of its own (ov::executor's start_worker), ticking on a fixed clock away from
// the frame. The frame thread advances the clock (update) and collects each state the
// worker publishes; the renderer blends the last two (blend()). Engine-free.
// - The worker's first step builds the forcing (each cell's biome weather, from the climate under its centre: ForcingFn,
//   in parallel at LOW priority) and runs the spin-up; then one tick (time_scale / tick_hz game seconds, in steps the
//   fluid's max_dt and MAX_STEP_S allow) per step while the clock is ahead of it. More than MAX_BEHIND ticks behind (a stall, a slow machine), it drops the excess: the
//   weather slows, it never runs away.
// - Each step: the fluid's advection, the forcing, the shallow convection (WeatherConvection), the moist physics, the
//   buoyancy and the projection.
// - A published state is stride() floats per padded cell: the column's LOOK (Look: the ground level's wind (m/s, planet
//   axes), the cover (its levels' largest cloud), the type (how deep the cloud is: 0 a layer, 1 a tower over a third of
//   the levels), the rain (mm/h at the ground), the ground level's temperature (C) and vapour (g/kg), the wind aloft
//   (the levels' mean)), then each level's mean cloud water (the partial cloud's) over FULL_CLOUD_QC (the clouds'
//   density), the ground's first, then each level's Q1 (WeatherMoist::q1: where in its spread the cell's mean stands).
class Weather : public ov::WorkerLoop {
public:
    enum Look {
        WIND_X,
        WIND_Y,
        WIND_Z,
        COVER,
        TYPE,
        RAIN,
        TEMP,
        VAPOUR,
        ALOFT_X,
        ALOFT_Y,
        ALOFT_Z,
        LOOK
    };
    static constexpr int MAX_BEHIND = 20;
    static constexpr double FULL_CLOUD_QC = 3e-4; // kg/kg: a level's cloud 1
    static constexpr double MAX_STEP_S = 5.0;
    // The largest real time a frame advances the clock (seconds): a hitch is not a storm's worth of weather.
    static constexpr double MAX_FRAME_S = 0.25;

    // The biome weather at a cell's ground point (a LOD0 voxel), read at level p_m of the climate's lattice.
    using ForcingFn = std::function<Climate::BiomeWeather(const VoxelPos &, int)>;

    struct Sample {
        Float3 wind_mps; // at the ground level, planet (and render) axes, tangent to the surface
        Float3 aloft_mps; // the levels' mean
        float cover = 0.0f;
        float type = 0.0f;
        float rain = 0.0f;
        float temp = 0.0f;
        float vapour = 0.0f;
    };

    // The weather p_weather (WorldFile::Weather, on) over p_base (voxel p_voxel_m) on its grid (WeatherGrid::for_base, which
    // the world file checked), seeded by the planet's seed; p_probing: every published state also keeps the whole
    // simulation's (probe(), the debugger's).
    Weather(const WorldFile::Weather &p_weather, const BaseSurface &p_base, double p_voxel_m, uint32_t p_seed, ForcingFn p_forcing,
            bool p_probing = false);
    ~Weather() override;
    Weather(const Weather &) = delete;
    Weather &operator=(const Weather &) = delete;

    // The biome weather the climate gives at a LOD0 voxel: its biome weights at Λ_p_m (clamped to the program's levels)
    // blending their biomes' values (a weight's share of their sum; the neutral biome's where none weighs).
    static Climate::BiomeWeather forcing_at(const Climate::Table &p_table, const Terrain::LevelProgram &p_prog, const VoxelPos &p_voxel, int p_m);

    // Frame thread: advances the clock by the real time since the last call (MAX_FRAME_S at most) and asks the worker for
    // the ticks due; collects a published state (true: a new one, the last two are blend()'s).
    bool update(uint64_t p_now_usec);
    // Waits for the ticks asked for (the next update collects them); an inline executor pumps instead.
    void finish();
    // Whether a state has landed (the forcing and the spin-up done).
    bool ready() const { return has_live; }
    // Where the frame stands between the previous state (0) and the latest (1).
    float blend() const;
    // The latest state under a position (metres from the base's centre; false before the first).
    bool at(const double p_pos[3], Sample &r_sample) const;
    // Each level's cloud (over FULL_CLOUD) under a position: r_levels gets levels() floats (false before the first state).
    bool clouds_at(const double p_pos[3], float *r_levels) const;
    // The levels (the planet file's top_m over about its cells' width), each one top_m / levels thick.
    int levels() const { return level_count; }
    double top_m() const { return settings.top_m; }
    // A published state's floats per padded cell: LOOK, then a cloud per level.
    int stride() const { return LOOK + 2 * level_count; }
    // The last two states (stride() floats per padded cell of grid()): previous, latest.
    const std::vector<float> &previous() const { return prev_look; }
    const std::vector<float> &latest() const { return live_look; }
    const WeatherGrid &grid() const { return *cells; }
    // The sphere's radius (metres): the report's maps are equirectangular over it.
    double radius_m() const { return radius; }
    // The latest state's game time (the spin-up's included) and the storms started up to it.
    double game_s() const { return live_time_s; }
    uint64_t storms() const { return live_storms; }
    const WorldFile::Weather &get_settings() const { return settings; }
    // The direction towards the sun (planet axes, unit) at the end of the spin-up (the scene's sun): the weather turns it
    // round the pole once a day (day_angle) for the grounds' day from the next tick on.
    void set_sun(const Float3 &p_towards_sun);
    // How far the planet's spin has turned the sun at game time p_game_s (radians round +y: WeatherForcing::turn_sun):
    // 0 at the end of the spin-up, -Omega t after.
    double day_angle(double p_game_s) const;
    // The sun at game time p_game_s (planet axes, unit): set_sun's turned by day_angle.
    Float3 sun_at(double p_game_s) const;
    // The game time the frame shows: between the previous state's and the latest's (blend()); the spin-up's end before
    // the first.
    double shown_s() const;
    ov::Value get_stats() const;
    // The latest state's whole simulation (WeatherDebug::Snapshot; null unless probing).
    std::shared_ptr<const WeatherDebug::Snapshot> probe() const { return live_probe; }

    // The worker's: the forcing and the spin-up first, then one tick per call while ticks are due.
    bool step() override;

private:
    WorldFile::Weather settings;
    std::shared_ptr<const WeatherGrid> cells;
    int level_count = 1;
    double radius = 1.0;
    uint32_t seed = 0;
    BaseSurface base;
    double voxel_m = 1.0;
    ForcingFn forcing;
    int forcing_level = 0;
    bool probing = false;
    // The worker's.
    std::unique_ptr<WeatherFluid> fluid;
    std::unique_ptr<WeatherMoist> moist;
    std::unique_ptr<WeatherForcing> drive;
    std::unique_ptr<WeatherConvection> convection;
    std::vector<WeatherMoist::Ground> ground;
    std::vector<float> buoyancy;
    uint64_t steps = 0;
    double sim_time_s = 0.0;
    std::unique_ptr<ov::Worker> worker;

    mutable std::mutex mutex;
    std::condition_variable published;
    // Under mutex: the ticks the clock asks for and those done, a state published and not collected.
    uint64_t due = 0;
    uint64_t done = 0;
    uint64_t dropped = 0;
    bool ready_state = false;
    bool started = false;
    std::vector<float> next_look;
    uint64_t next_tick = 0;
    uint64_t next_done = 0;
    double next_time_s = 0.0;
    uint64_t next_storms = 0;
    std::shared_ptr<const WeatherDebug::Snapshot> next_probe;
    Float3 sun_dir{ 0.0f, 0.0f, 1.0f }; // an equinox sun until a host sets one (the report's)
    TimingStat tick_stat;
    TimingStat forcing_stat;
    double substep_s = 0.0;
    double omega_rad_s = 0.0;
    ov::Value forcing_summary;
    double ground_day_k = 0.0, ground_night_k = 0.0; // the latest tick's grounds, the sun's side and the other
    double convecting = 0.0, convection_top_m = 0.0; // the latest tick's share of columns with an updraft, their mean top

    // The frame thread's.
    uint64_t last_usec = 0;
    double clock_ticks = 0.0; // real seconds times tick_hz since the first state
    bool has_live = false;
    uint64_t live_tick = 0;
    uint64_t live_done = 0; // the clock's ticks the latest state has done (0: the spin-up's)
    double live_time_s = 0.0;
    uint64_t live_storms = 0;
    std::vector<float> live_look;
    std::vector<float> prev_look;
    std::shared_ptr<const WeatherDebug::Snapshot> live_probe;

    void build();
    // p_dt game seconds of fluid, physics and forcing.
    void advance(double p_dt);
    std::vector<float> look() const;
};
