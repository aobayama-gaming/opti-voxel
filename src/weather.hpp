#pragma once

#include "debug_log.hpp"
#include "level_program.hpp"
#include "weather_sim.hpp"
#include "world_file.hpp"

#include "ov/executor.hpp"
#include "ov/value.hpp"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

// The weather over a planet (docs/weather.md "Worker"): a WeatherSim on a worker of its own (ov::executor's start_worker),
// ticking on a fixed clock away from the frame. The frame thread advances the clock (update) and collects each state the
// worker publishes; the renderer blends the last two (blend()). Engine-free.
// - The worker's first step builds the forcing (each cell's biome weather, from the climate under its centre: ForcingFn,
//   in parallel at LOW priority) and runs the spin-up; then one tick (time_scale / tick_hz game seconds) per step while
//   the clock is ahead of it. More than MAX_BEHIND ticks behind (a stall, a slow machine), it drops the excess: the
//   weather slows, it never runs away.
// - A published state is stride() floats per padded cell: the column's LOOK (Look: the ground level's wind (m/s, planet
//   axes), the cover, the type, the rain, the ground level's temperature and vapour, the wind aloft (the levels' mean)),
//   then each level's cloud water over FULL_CLOUD (the clouds' density), the ground's first.
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
    // the world file checked), seeded by the planet's seed.
    Weather(const WorldFile::Weather &p_weather, const BaseSurface &p_base, double p_voxel_m, uint32_t p_seed, ForcingFn p_forcing);
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
    // The levels (WeatherSim's: the planet file's top_m over about its cells' width), each one top_m / levels thick.
    int levels() const { return params.levels; }
    double top_m() const { return params.top_m; }
    // A published state's floats per padded cell: LOOK, then a cloud per level.
    int stride() const { return LOOK + params.levels; }
    // The last two states (stride() floats per padded cell of grid()): previous, latest.
    const std::vector<float> &previous() const { return prev_look; }
    const std::vector<float> &latest() const { return live_look; }
    const WeatherGrid &grid() const { return *cells; }
    // The sphere's radius (metres): the report's maps are equirectangular over it.
    double radius_m() const { return params.radius_m; }
    // The latest state's game time (the spin-up's included) and the storms started up to it.
    double game_s() const { return live_time_s; }
    uint64_t storms() const { return live_storms; }
    const WorldFile::Weather &get_settings() const { return settings; }
    ov::Value get_stats() const;

    // The worker's: the forcing and the spin-up first, then one tick per call while ticks are due.
    bool step() override;

private:
    WorldFile::Weather settings;
    std::shared_ptr<const WeatherGrid> cells;
    WeatherSim::Params params;
    BaseSurface base;
    double voxel_m = 1.0;
    ForcingFn forcing;
    int forcing_level = 0;
    std::unique_ptr<WeatherSim> sim; // the worker's
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
    TimingStat tick_stat;
    TimingStat forcing_stat;
    double substep_s = 0.0;
    double omega_rad_s = 0.0;

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

    void build();
    std::vector<float> look() const;
};
