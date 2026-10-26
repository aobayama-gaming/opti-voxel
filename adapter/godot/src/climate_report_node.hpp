#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/vector2.hpp>

using namespace godot;

// ClimateReport (src/climate_report.hpp) as a node: res://debug/climate_report.tscn, tools/climate_report.sh.
class ClimateReportNode : public Node {
    GDCLASS(ClimateReportNode, Node)

protected:
    static void _bind_methods();

public:
    Dictionary run(const String &p_out_dir, const String &p_climate_path, double p_voxel_size, int p_lods, double p_extent_km, int p_size, bool p_surface,
            const Vector2 &p_center, int p_lattice = -1);
};
