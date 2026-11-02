#include "climate_report_node.hpp"

#include "climate.hpp"
#include "climate_report.hpp"
#include "godot_conv.hpp"
#include "sdf_cascade.hpp"

#include "ov/json.hpp"

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

void ClimateReportNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "out_dir", "world_path", "extent_km", "size", "surface", "center", "lattice"),
            &ClimateReportNode::run);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("check", "json", "voxel_size", "lods"), &ClimateReportNode::check);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("curve", "points", "xs"), &ClimateReportNode::curve);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("template_notes", "voxel_m", "lods"), &ClimateReportNode::template_notes);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("preview", "json", "dir", "extent_km", "size", "surface", "center"), &ClimateReportNode::preview);
}

Dictionary ClimateReportNode::run(const String &p_out_dir, const String &p_world_path, double p_extent_km, int p_size,
        bool p_surface, const Vector2 &p_center, int p_lattice) {
    return godot_dict(ClimateReport::run(ov_str(p_out_dir), ov_str(p_world_path), p_extent_km, p_size, p_surface, p_center.x,
            p_center.y, p_lattice));
}

String ClimateReportNode::check(const String &p_json, double p_voxel_size, int p_lods) {
    return godot_str(ClimateReport::check(ov_str(p_json), p_voxel_size, p_lods));
}

Dictionary ClimateReportNode::preview(const String &p_json, const String &p_dir, double p_extent_km, int p_size, bool p_surface, const Vector2 &p_center) {
    Dictionary d;
    ov::str json_error;
    const ov::Value world = ov::parse_json(ov_str(p_json), json_error);
    if (!world.is_object()) {
        d["error"] = godot_str("world: not a JSON object. " + json_error);
        return d;
    }
    const ClimateReport::Preview p = ClimateReport::preview(world, ov_str(p_dir), p_extent_km, p_size, p_surface, p_center.x, p_center.y);
    d["error"] = godot_str(p.error);
    if (!p.error.empty()) {
        return d;
    }
    d["size"] = p.size;
    d["lattice"] = p.lattice;
    d["spacing_m"] = p.spacing_m;
    d["us_per_point"] = p.us_per_point;
    d["blend"] = Image::create_from_data(p.size, p.size, false, Image::FORMAT_RGB8, godot_bytes(p.blend));
    d["shaded"] = Image::create_from_data(p.size, p.size, false, Image::FORMAT_RGB8, godot_bytes(p.shaded));
    PackedFloat32Array altitude, fields;
    altitude.resize(int64_t(p.altitude_m.size()));
    std::copy(p.altitude_m.begin(), p.altitude_m.end(), altitude.ptrw());
    fields.resize(int64_t(p.fields.size()));
    std::copy(p.fields.begin(), p.fields.end(), fields.ptrw());
    d["altitude_m"] = altitude;
    d["fields"] = fields;
    PackedStringArray field_names, biome_names;
    for (const ov::str &n : p.field_names) {
        field_names.push_back(godot_str(n));
    }
    for (const ov::str &n : p.biome_names) {
        biome_names.push_back(godot_str(n));
    }
    d["field_names"] = field_names;
    d["biome_names"] = biome_names;
    return d;
}

PackedFloat32Array ClimateReportNode::curve(const Array &p_points, const PackedFloat32Array &p_xs) {
    std::vector<double> x, y;
    for (int64_t i = 0; i < p_points.size(); i++) {
        const Array p = p_points[i];
        x.push_back(double(p[0]));
        y.push_back(double(p[1]));
        if (x.size() > 1 && !(x.back() > x[x.size() - 2])) {
            return p_xs;
        }
    }
    if (x.size() < 2) {
        return p_xs;
    }
    const Climate::Curve c = Climate::make_curve(x, y);
    PackedFloat32Array out;
    out.resize(p_xs.size());
    for (int64_t i = 0; i < p_xs.size(); i++) {
        out[i] = Climate::eval_curve(c, ExactNoise::Float4{ 0.0f, 0.0f, 0.0f, p_xs[i] }).w;
    }
    return out;
}

Dictionary ClimateReportNode::template_notes(double p_voxel_m, int p_lods) {
    Dictionary d;
    d["seed"] = int64_t(Climate::TEMPLATE_SEED);
    d["steepness"] = Climate::TEMPLATE_STEEPNESS;
    d["max_amp_m"] = Climate::TEMPLATE_MAX_AMP_M;
    d["k_min"] = ExactNoise::BAND_SHIFT;
    d["k_top"] = ov::min(Climate::template_top(p_voxel_m), SDFCascade::top_octave(p_lods));
    d["octave_top"] = SDFCascade::top_octave(p_lods);
    return d;
}
