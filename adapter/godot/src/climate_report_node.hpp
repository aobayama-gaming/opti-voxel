#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

using namespace godot;

// ClimateReport (src/climate_report.hpp) as a node: res://debug/climate_report.tscn, tools/climate_report.sh.
class ClimateReportNode : public Node {
    GDCLASS(ClimateReportNode, Node)

protected:
    static void _bind_methods();

public:
    Dictionary run(const String &p_out_dir, const String &p_world_path, double p_extent_km, int p_size, bool p_surface,
            const Vector2 &p_center, int p_lattice = -1);
    // ClimateReport::check: "" if the table's text is accepted, else why not (static: ClassDB.class_call_static).
    static String check(const String &p_json, double p_voxel_size, int p_lods);
    // ClimateReport::preview of a world file's text (its relative paths from p_dir), static: { error, size, lattice,
    // spacing_m, us_per_point, relief_m, biomes: Image, shaded: Image (RGB8), altitude_m: PackedFloat32Array, fields:
    // PackedFloat32Array (point-major), field_names, biome_names } (the biome mixer's maps).
    static Dictionary preview(const String &p_json, const String &p_dir, double p_extent_km, int p_size, bool p_surface, const Vector2 &p_center);
    // A height curve node's R(h) at p_xs (metres) through p_points ([h, R] pairs, metres) at a voxel size: the engine's
    // curve (Climate::height_curve, eval_curve), drawn by the biome mixer's curve editor. Points it refuses: p_xs.
    static PackedFloat32Array curve(const Array &p_points, const PackedFloat32Array &p_xs, double p_voxel_m);
    // The template ground's notes at a voxel size and LOD count (Climate::template_graph; the biome mixer's notes are
    // its octaves): { seed, steepness, max_amp_m, k_min (the finest, BAND_SHIFT), k_top (its coarsest the LODs keep), octave_top (a note's or a field's coarsest at that LOD count), core (where a biome is whole on a field: within core of its span, Climate::KERNEL_FADE) }.
    static Dictionary template_notes(double p_voxel_m, int p_lods);
};
