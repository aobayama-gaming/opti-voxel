#include "climate_report_node.hpp"

#include "climate_report.hpp"
#include "godot_conv.hpp"

#include <godot_cpp/core/class_db.hpp>

void ClimateReportNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "out_dir", "climate_path", "voxel_size", "lods", "extent_km", "size", "surface", "center", "lattice"),
            &ClimateReportNode::run);
}

Dictionary ClimateReportNode::run(const String &p_out_dir, const String &p_climate_path, double p_voxel_size, int p_lods, double p_extent_km, int p_size,
        bool p_surface, const Vector2 &p_center, int p_lattice) {
    return godot_dict(ClimateReport::run(ov_str(p_out_dir), ov_str(p_climate_path), p_voxel_size, p_lods, p_extent_km, p_size, p_surface, p_center.x,
            p_center.y, p_lattice));
}
