#pragma once

#include "weather_debug.hpp"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>

using namespace godot;

// The weather debugger's node (res://debug/weather_debug.tscn, docs/weather.md "Debugger"): WeatherDebug::Session and
// its Snapshot mapped onto Godot's types; the scene draws them. Positions are metres from the planet's centre.
class WeatherDebugNode : public Node {
    GDCLASS(WeatherDebugNode, Node)

protected:
    static void _bind_methods();

public:
    // The world file at p_world_path's weather, p_spin_up_s of spin-up (negative: the file's): "" or why not.
    String open(const String &p_world_path, double p_spin_up_s);
    // p_ticks ticks, waited for.
    void step(int p_ticks);
    // p_real_s of the weather's clock, not waited for.
    void run(double p_real_s);
    // Whether a state landed (the forcing and the spin-up done).
    bool is_ready() const;
    // Weather::get_stats, plus the snapshot's game_s.
    Dictionary stats() const;
    // { faces, width, height, levels, dz_m, top_m, radius_m, cell_m }.
    Dictionary grid() const;
    // [{ name, unit, levels (per level), signed }], WeatherDebug::Field's order.
    static Array fields();
    // A field at a level per interior cell (face-major, then rows, then x: an image of width x faces * height).
    PackedFloat32Array slice(int p_field, int p_level) const;
    // A field along row p_j of face p_face: width per level, the ground's first.
    PackedFloat32Array section(int p_field, int p_face, int p_j) const;
    // Every field at a column: an array per field (levels values, or one).
    Array column(int p_face, int p_i, int p_j) const;
    // The cell under a position: (face, i, j).
    Vector3i pick(const Vector3 &p_pos) const;
    // The planet's cells as quads p_altitude_m over the ground: UV the cell's texel in slice()'s image (its centre),
    // UV2 the corner (0 or 1 per axis, for the cell's edges), normal the up.
    Ref<ArrayMesh> planet_mesh(double p_altitude_m) const;
    // The wind at a level as line pairs (every p_every-th cell per axis): the cell's centre p_altitude_m over the
    // ground, then that point plus the wind times p_seconds.
    PackedVector3Array wind_lines(int p_level, double p_altitude_m, double p_seconds, int p_every) const;

private:
    WeatherDebug::Session session;
};
