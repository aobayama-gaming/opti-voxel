#include "climate_report_node.hpp"

#include "climate_report.hpp"
#include "godot_conv.hpp"

#include <godot_cpp/core/class_db.hpp>

void ClimateReportNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "out_dir", "world_path", "extent_km", "size", "surface", "center", "lattice"),
            &ClimateReportNode::run);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("check", "json", "voxel_size", "lods"), &ClimateReportNode::check);
}

Dictionary ClimateReportNode::run(const String &p_out_dir, const String &p_world_path, double p_extent_km, int p_size,
        bool p_surface, const Vector2 &p_center, int p_lattice) {
    return godot_dict(ClimateReport::run(ov_str(p_out_dir), ov_str(p_world_path), p_extent_km, p_size, p_surface, p_center.x,
            p_center.y, p_lattice));
}

String ClimateReportNode::check(const String &p_json, double p_voxel_size, int p_lods) {
    return godot_str(ClimateReport::check(ov_str(p_json), p_voxel_size, p_lods));
}
