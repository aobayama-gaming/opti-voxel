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
    // spacing_m, us_per_point, blend: Image, shaded: Image (RGB8), altitude_m: PackedFloat32Array, fields:
    // PackedFloat32Array (point-major), field_names, biome_names } (the biome mixer's maps).
    // Climate::make_curve through p_points ([x, y] pairs, x increasing, at least 2) evaluated at p_xs (eval_curve): the biome
    // mixer's curve editor draws the engine's cubic. Fewer or unordered points: p_xs unchanged (no remap).
    // The template ground's notes at a voxel size and LOD count (Climate::template_graph; the biome mixer's notes are
    // its octaves): { seed, steepness, max_amp_m, k_min (the finest, BAND_SHIFT), k_top (its coarsest the LODs keep), octave_top (a note's or a field's coarsest at that LOD count) }.
    static Dictionary template_notes(double p_voxel_m, int p_lods);
    static PackedFloat32Array curve(const Array &p_points, const PackedFloat32Array &p_xs);
    static Dictionary preview(const String &p_json, const String &p_dir, double p_extent_km, int p_size, bool p_surface, const Vector2 &p_center);
};
