#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
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
    // PackedFloat32Array (point-major), field_names, biome_names } (the biome mixer's maps). p_background: its loop at
    // ov::Priority::LOW, behind the frame's.
    static Dictionary preview(const String &p_json, const String &p_dir, double p_extent_km, int p_size, bool p_surface, const Vector2 &p_center,
            bool p_background = false);
    // A height curve node's R(h) at p_xs (metres) through p_points ([h, R] pairs, metres) at a voxel size: the engine's
    // curve (Climate::height_curve, eval_curve), drawn by the biome mixer's curve editor. Points it refuses: p_xs.
    static PackedFloat32Array curve(const Array &p_points, const PackedFloat32Array &p_xs, double p_voxel_m);
    // A biome graph's node kinds (Climate::graph_kinds): [{ kind, inputs, params: [{ name, type, required, default (null:
    // none), options }] }], what the biome mixer builds its nodes from.
    static Array graph_kinds();
    // A shape node's phi at p_ns (Climate::parse_shape, shape_phi; p_multifractal: a multifractal reads it): { error,
    // phi }, the biome mixer's plot.
    static Dictionary shape(const Dictionary &p_node, bool p_multifractal, const PackedFloat32Array &p_ns);
    // The template biome named p_name at a voxel size (Climate::template_biome): a new biome's graph.
    static Dictionary template_biome(double p_voxel_m, const String &p_name);
    // A biome's debug colour without its own: its name's hash (the parser's).
    static Color debug_color(const String &p_name);
    // Where a biome is whole on a field: within core of its span (smoothstep(clamp((1 - u^2) KERNEL_FADE)) is 1).
    static double kernel_core();
};
