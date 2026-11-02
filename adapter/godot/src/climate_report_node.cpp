#include "climate_report_node.hpp"

#include "biome_compiler.hpp"
#include "climate.hpp"
#include "climate_report.hpp"
#include "godot_conv.hpp"
#include "world_file.hpp"

#include "ov/json.hpp"

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

#include <cmath>

void ClimateReportNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "out_dir", "world_path", "extent_km", "size", "surface", "center", "lattice"),
            &ClimateReportNode::run);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("check", "json", "voxel_size", "lods"), &ClimateReportNode::check);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("curve", "points", "xs", "voxel_m"), &ClimateReportNode::curve);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("graph_kinds"), &ClimateReportNode::graph_kinds);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("shape", "node", "multifractal", "ns"), &ClimateReportNode::shape);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("template_biome", "voxel_m", "name"), &ClimateReportNode::template_biome);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("debug_color", "name"), &ClimateReportNode::debug_color);
    ClassDB::bind_static_method("ClimateReportNode", D_METHOD("kernel_core"), &ClimateReportNode::kernel_core);
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
    ov::Value world;
    const ov::str err = WorldFile::parse_object(ov_str(p_json), "the world's text", world);
    if (!err.empty()) {
        d["error"] = godot_str(err);
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
    d["relief_m"] = p.relief_m;
    d["biomes"] = Image::create_from_data(p.size, p.size, false, Image::FORMAT_RGB8, godot_bytes(p.biomes));
    d["shaded"] = Image::create_from_data(p.size, p.size, false, Image::FORMAT_RGB8, godot_bytes(p.shaded));
    d["altitude_m"] = godot_floats(p.altitude_m);
    d["fields"] = godot_floats(p.fields);
    d["field_names"] = godot_strings(p.field_names);
    d["biome_names"] = godot_strings(p.biome_names);
    return d;
}

PackedFloat32Array ClimateReportNode::curve(const Array &p_points, const PackedFloat32Array &p_xs, double p_voxel_m) {
    std::vector<double> h, r;
    for (int64_t i = 0; i < p_points.size(); i++) {
        if (p_points[i].get_type() != Variant::ARRAY || Array(p_points[i]).size() != 2) {
            return p_xs;
        }
        const Array p = p_points[i];
        h.push_back(double(p[0]));
        r.push_back(double(p[1]));
    }
    Climate::Curve c;
    if (!Climate::height_curve(h, r, p_voxel_m, c).empty()) {
        return p_xs;
    }
    PackedFloat32Array out;
    out.resize(p_xs.size());
    for (int64_t i = 0; i < p_xs.size(); i++) {
        const float x = float(double(p_xs[i]) / p_voxel_m);
        out[i] = float(double(p_xs[i]) + double(Climate::eval_curve(c, ExactNoise::Float4{ 0.0f, 0.0f, 0.0f, x }).w) * p_voxel_m);
    }
    return out;
}

Array ClimateReportNode::graph_kinds() {
    Array out;
    for (const Climate::GraphKind &k : Climate::graph_kinds()) {
        Dictionary d;
        d["kind"] = String(k.kind);
        PackedStringArray inputs;
        for (const char *in : k.inputs) {
            inputs.push_back(String(in));
        }
        d["inputs"] = inputs;
        Array params;
        for (const Climate::GraphParam &p : k.params) {
            Dictionary pd;
            pd["name"] = String(p.name);
            pd["type"] = String(p.type);
            pd["required"] = p.required;
            pd["default"] = std::isnan(p.fallback) ? Variant() : Variant(p.fallback);
            pd["options"] = String(p.options).split("|", false);
            params.push_back(pd);
        }
        d["params"] = params;
        out.push_back(d);
    }
    return out;
}

Dictionary ClimateReportNode::shape(const Dictionary &p_node, bool p_multifractal, const PackedFloat32Array &p_ns) {
    Dictionary d;
    Climate::Shape s;
    d["error"] = godot_str(Climate::parse_shape(ov_value(p_node), p_multifractal, s));
    PackedFloat32Array phi;
    phi.resize(p_ns.size());
    for (int64_t i = 0; i < p_ns.size(); i++) {
        phi[i] = float(Climate::shape_phi(s, double(p_ns[i])));
    }
    d["phi"] = phi;
    return d;
}

Dictionary ClimateReportNode::template_biome(double p_voxel_m, const String &p_name) {
    return godot_dict(Climate::template_biome(p_voxel_m, ov_str(p_name)));
}

Color ClimateReportNode::debug_color(const String &p_name) {
    return Color::hex((Climate::name_hash(ov_str(p_name)) & 0xFFFFFFu) << 8 | 0xFFu);
}

double ClimateReportNode::kernel_core() {
    return std::sqrt(1.0 - 1.0 / double(Climate::KERNEL_FADE)); // smoothstep(clamp((1 - u^2) KERNEL_FADE)) is 1 up to there
}
