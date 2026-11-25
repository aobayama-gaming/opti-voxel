#pragma once

#include "weather_moist.hpp"

#include "ov/str.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class SDFCascade;
class Weather;
class WeatherConvection;
class WeatherForcing;
struct WorldFile;

// The weather debugger's core (docs/weather.md "Debugger", res://debug/weather_debug.tscn): what a tick leaves in the
// simulation, read field by field, cell by cell, level by level, and a world's weather on a clock of its own.
// Engine-free; the adapter's WeatherDebugNode only maps it onto Godot's types.
namespace WeatherDebug {

// The fields, per interior cell and level first, then per column (ground and its fluxes).
enum Field {
    WIND_EAST, // m/s, the tangent wind's east part (east = pole x up)
    WIND_NORTH,
    WIND_UP, // w at the centre (the mean of its interfaces')
    WIND_UP_FACE, // w on the interface under the level (the prognostic one; the ground's 0)
    SPEED, // the tangent wind's
    THETA, // K, potential temperature
    THETA_ANOMALY, // K, over the sounding's (the buoyancy's reference)
    TEMP_C, // C
    QV, // g/kg
    QS, // g/kg, saturation at the cell's temperature and the sounding's pressure (Kessler's Tetens)
    RH, // qv / qs
    QC, // g/kg, the partial cloud's mean
    QR, // g/kg
    SHARE, // the cloudy share (WeatherMoist::share)
    Q1, // the mean saturation excess over its spread
    SIGMA, // g/kg, the spread sigma_s
    UPDRAFT_QC, // g/kg, the shallow convection's updraft water
    BUOYANCY, // m/s^2
    PRESSURE, // m^2/s^2, the kinematic pressure perturbation
    GROUND_K, // K, the ground's temperature now
    GROUND_MEAN_K, // K, its biomes' day mean (ground_k)
    GROUND_AIR_K, // K, the ground over the lowest level's air
    WETNESS, // beta: 1 a sea, 0 a desert
    Z0_M, // m, roughness
    INERTIA, // J/(m^2 K s^1/2)
    STORMS, // the biomes' storm odds
    SENSIBLE_W_M2, // the surface's heat flux, up
    LATENT_W_M2, // the surface's vapour flux as heat, up
    USTAR, // m/s, friction velocity
    RAIN_MM_H, // at the ground
    UPDRAFT_TOP_M, // the shallow convection's top (0: none)
    COVER, // the column's largest cloudy share
    LWP_G_M2, // the column's cloud water path
    FIELDS,
};
constexpr int LEVEL_FIELDS = GROUND_K;

struct FieldInfo {
    const char *name;
    const char *unit;
    bool signed_field; // diverging around 0 (a wind, an anomaly)
};
const FieldInfo &info(int p_field);
inline bool per_level(int p_field) {
    return p_field < LEVEL_FIELDS;
}

// Every field at one tick.
struct Snapshot {
    int faces = 0;
    int width = 0;
    int height = 0;
    int levels = 0;
    double dz_m = 0.0;
    double game_s = 0.0;
    std::vector<float> values; // field-major; a level field interior cell-major (WeatherFluid::cell_index), a column field per interior cell
    std::vector<float> wind; // 3 per interior cell and level: the wind in planet axes

    size_t cells() const { return size_t(faces) * size_t(width) * size_t(height); }
    size_t interior(int p_face, int p_i, int p_j) const { return (size_t(p_face) * size_t(height) + size_t(p_j)) * size_t(width) + size_t(p_i); }
    // A field's values (cells() x levels or cells()).
    const float *field(int p_field) const;
    // A field at an interior cell (p_level ignored by a column field).
    float at(int p_field, size_t p_interior, int p_level) const;
    // A field's level (or its column values) per interior cell.
    std::vector<float> slice(int p_field, int p_level) const;
    // A field along row p_j of face p_face: width per level, the ground's first (a column field: one row).
    std::vector<float> section(int p_field, int p_face, int p_j) const;

    // The state the worker holds after a tick: p_climate the forcing's biome weather per interior cell.
    static std::shared_ptr<const Snapshot> capture(const WeatherFluid &p_fluid, const WeatherMoist &p_moist, const WeatherConvection &p_convection,
            const WeatherForcing &p_forcing, const std::vector<WeatherMoist::Ground> &p_ground, const std::vector<float> &p_buoyancy, double p_game_s);
};

// Which columns a profile averages: all, the seas (wetness 0.9 and more) or the land.
enum Columns {
    ALL,
    SEA,
    LAND,
};
// Per level (a column field: one value) over p_columns' cells (equal weights: the cube sphere's cells differ by up to
// ~30% in area): the mean of p_field, the root of its square's mean, its largest, the share of cells above p_above.
std::vector<double> mean(const Snapshot &p_snap, int p_field, Columns p_columns);
std::vector<double> rms(const Snapshot &p_snap, int p_field, Columns p_columns);
std::vector<double> maximum(const Snapshot &p_snap, int p_field, Columns p_columns);
std::vector<double> share_above(const Snapshot &p_snap, int p_field, Columns p_columns, float p_above);
// The large-eddy simulations' profiles (Siebesma et al. 2003's: the means per level, the cloud fraction, w's rms) over
// p_columns as a markdown table, then the column fields' means.
ov::str profile_table(const Snapshot &p_snap, Columns p_columns);

// A world's weather on a clock of its own (the report's and the debugger's): the world's SDF for the forcing, the
// weather, its ticks asked for by hand.
class Session {
public:
    Session();
    ~Session();
    // The world p_world's weather, p_spin_up_s of spin-up (negative: the world file's), every tick's Snapshot kept
    // when p_probing: "" or why not.
    ov::str open(const WorldFile &p_world, double p_spin_up_s, bool p_probing);
    bool is_open() const { return bool(sim); }
    Weather &weather() { return *sim; }
    const Weather &weather() const { return *sim; }
    // p_ticks more ticks, each waited for and collected (after the forcing and the spin-up).
    void step(int p_ticks);
    // p_real_s of the weather's clock: asks for the ticks due and collects what landed, without waiting (the worker
    // catches up, at most Weather::MAX_BEHIND behind).
    void run(double p_real_s);

private:
    std::unique_ptr<SDFCascade> sdf;
    std::unique_ptr<Weather> sim;
    uint64_t now_usec = 1;
    uint64_t period_usec = 200000;
};

} // namespace WeatherDebug
