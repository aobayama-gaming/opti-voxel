#include "weather_debug_node.hpp"

#include "godot_conv.hpp"
#include "weather.hpp"
#include "world_file.hpp"

#include "ov/math.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>

#include <cmath>

void WeatherDebugNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("open", "world_path", "spin_up_s"), &WeatherDebugNode::open);
    ClassDB::bind_method(D_METHOD("step", "ticks"), &WeatherDebugNode::step);
    ClassDB::bind_method(D_METHOD("run", "real_s"), &WeatherDebugNode::run);
    ClassDB::bind_method(D_METHOD("seed_vortex", "lat_deg", "lon_deg", "vmax_mps", "radius_m"), &WeatherDebugNode::seed_vortex);
    ClassDB::bind_method(D_METHOD("storm", "radius_m"), &WeatherDebugNode::storm);
    ClassDB::bind_method(D_METHOD("is_ready"), &WeatherDebugNode::is_ready);
    ClassDB::bind_method(D_METHOD("stats"), &WeatherDebugNode::stats);
    ClassDB::bind_method(D_METHOD("grid"), &WeatherDebugNode::grid);
    ClassDB::bind_static_method("WeatherDebugNode", D_METHOD("fields"), &WeatherDebugNode::fields);
    ClassDB::bind_method(D_METHOD("slice", "field", "level"), &WeatherDebugNode::slice);
    ClassDB::bind_method(D_METHOD("section", "field", "face", "j"), &WeatherDebugNode::section);
    ClassDB::bind_method(D_METHOD("column", "face", "i", "j"), &WeatherDebugNode::column);
    ClassDB::bind_method(D_METHOD("pick", "pos"), &WeatherDebugNode::pick);
    ClassDB::bind_method(D_METHOD("planet_mesh", "altitude_m"), &WeatherDebugNode::planet_mesh);
    ClassDB::bind_method(D_METHOD("profile_table", "columns"), &WeatherDebugNode::profile_table);
    ClassDB::bind_method(D_METHOD("wind_lines", "level", "altitude_m", "seconds", "every"), &WeatherDebugNode::wind_lines);
}

namespace {

PackedFloat32Array packed(const std::vector<float> &p_v) {
    PackedFloat32Array out;
    out.resize(int64_t(p_v.size()));
    for (size_t i = 0; i < p_v.size(); i++) {
        out.set(int64_t(i), p_v[i]);
    }
    return out;
}

} // namespace

String WeatherDebugNode::open(const String &p_world_path, double p_spin_up_s) {
    WorldFile world;
    ov::str error = WorldFile::load(ov_str(p_world_path), world);
    if (error.empty()) {
        error = session.open(world, p_spin_up_s, true);
    }
    return godot_str(error);
}

void WeatherDebugNode::step(int p_ticks) {
    session.step(p_ticks);
}

void WeatherDebugNode::run(double p_real_s) {
    session.run(p_real_s);
}

void WeatherDebugNode::seed_vortex(double p_lat_deg, double p_lon_deg, double p_vmax_mps, double p_radius_m) {
    session.seed_vortex(p_lat_deg, p_lon_deg, p_vmax_mps, p_radius_m);
}

Dictionary WeatherDebugNode::storm(double p_radius_m) const {
    const WeatherDebug::Storm s = session.storm(p_radius_m);
    Dictionary d;
    d["drop"] = s.drop;
    d["max_wind"] = s.max_wind;
    d["rmw_m"] = s.rmw_m;
    d["lat_deg"] = s.lat_deg;
    d["lon_deg"] = s.lon_deg;
    return d;
}

bool WeatherDebugNode::is_ready() const {
    return session.is_open() && session.weather().ready() && session.weather().probe() != nullptr;
}

Dictionary WeatherDebugNode::stats() const {
    if (!session.is_open()) {
        return Dictionary();
    }
    Dictionary d = godot_dict(session.weather().get_stats());
    const auto snap = session.weather().probe();
    d["snapshot_game_s"] = snap ? snap->game_s : 0.0;
    return d;
}

Dictionary WeatherDebugNode::grid() const {
    Dictionary d;
    if (!session.is_open()) {
        return d;
    }
    const Weather &w = session.weather();
    const WeatherGrid &g = w.grid();
    d["faces"] = g.faces();
    d["width"] = g.width();
    d["height"] = g.height();
    d["levels"] = w.levels();
    PackedFloat64Array faces;
    for (int m = 0; m <= w.levels(); m++) {
        faces.push_back(w.levels_def().face_m(m));
    }
    d["faces_m"] = faces; // the levels' interfaces' heights (stretched: finer near the ground)
    d["top_m"] = w.top_m();
    d["radius_m"] = w.radius_m();
    d["cell_m"] = g.cell_m();
    return d;
}

Array WeatherDebugNode::fields() {
    Array out;
    for (int f = 0; f < WeatherDebug::FIELDS; f++) {
        const WeatherDebug::FieldInfo &fi = WeatherDebug::info(f);
        Dictionary d;
        d["name"] = String(fi.name);
        d["unit"] = String(fi.unit);
        d["levels"] = WeatherDebug::per_level(f);
        d["signed"] = fi.signed_field;
        out.push_back(d);
    }
    return out;
}

PackedFloat32Array WeatherDebugNode::slice(int p_field, int p_level) const {
    const auto snap = session.is_open() ? session.weather().probe() : nullptr;
    return snap ? packed(snap->slice(ov::clamp(p_field, 0, WeatherDebug::FIELDS - 1), p_level)) : PackedFloat32Array();
}

PackedFloat32Array WeatherDebugNode::section(int p_field, int p_face, int p_j) const {
    const auto snap = session.is_open() ? session.weather().probe() : nullptr;
    if (!snap || p_face < 0 || p_face >= snap->faces || p_j < 0 || p_j >= snap->height) {
        return PackedFloat32Array();
    }
    return packed(snap->section(ov::clamp(p_field, 0, WeatherDebug::FIELDS - 1), p_face, p_j));
}

Array WeatherDebugNode::column(int p_face, int p_i, int p_j) const {
    Array out;
    const auto snap = session.is_open() ? session.weather().probe() : nullptr;
    if (!snap || p_face < 0 || p_face >= snap->faces || p_i < 0 || p_i >= snap->width || p_j < 0 || p_j >= snap->height) {
        return out;
    }
    const size_t in = snap->interior(p_face, p_i, p_j);
    for (int f = 0; f < WeatherDebug::FIELDS; f++) {
        const int n = WeatherDebug::per_level(f) ? snap->levels : 1;
        PackedFloat32Array v;
        v.resize(n);
        for (int k = 0; k < n; k++) {
            v.set(k, snap->at(f, in, k));
        }
        out.push_back(v);
    }
    return out;
}

String WeatherDebugNode::profile_table(int p_columns) const {
    const auto snap = session.is_open() ? session.weather().probe() : nullptr;
    return snap ? godot_str(WeatherDebug::profile_table(*snap, WeatherDebug::Columns(ov::clamp(p_columns, 0, 2)))) : String();
}

Vector3i WeatherDebugNode::pick(const Vector3 &p_pos) const {
    if (!session.is_open()) {
        return Vector3i(-1, -1, -1);
    }
    const WeatherGrid &g = session.weather().grid();
    const double pos[3] = { p_pos.x, p_pos.y, p_pos.z };
    const WeatherGrid::Point pt = g.locate(pos);
    return Vector3i(pt.face, ov::clamp(int(std::floor(pt.x)), 0, g.width() - 1), ov::clamp(int(std::floor(pt.y)), 0, g.height() - 1));
}

Ref<ArrayMesh> WeatherDebugNode::planet_mesh(double p_altitude_m) const {
    Ref<ArrayMesh> mesh;
    mesh.instantiate();
    if (!session.is_open()) {
        return mesh;
    }
    const WeatherGrid &g = session.weather().grid();
    PackedVector3Array verts, normals;
    PackedVector2Array uv, uv2;
    const int rows = g.faces() * g.height();
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const Vector2 texel((i + 0.5f) / float(g.width()), (float(f * g.height() + j) + 0.5f) / float(rows));
                Vector3 p[4], n[4];
                const int corner[4][2] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
                for (int c = 0; c < 4; c++) {
                    double pos[3], up[3];
                    g.surface(f, i + corner[c][0], j + corner[c][1], pos, up);
                    n[c] = Vector3(float(up[0]), float(up[1]), float(up[2]));
                    p[c] = Vector3(float(pos[0]), float(pos[1]), float(pos[2])) + n[c] * float(p_altitude_m);
                }
                // Two triangles, wound so the face is seen from outside.
                const int order[6] = { 0, 2, 1, 0, 3, 2 };
                const bool flip = (p[1] - p[0]).cross(p[3] - p[0]).dot(n[0]) < 0.0f;
                for (int t = 0; t < 6; t++) {
                    const int c = order[flip ? 5 - t : t];
                    verts.push_back(p[c]);
                    normals.push_back(n[c]);
                    uv.push_back(texel);
                    uv2.push_back(Vector2(float(corner[c][0]), float(corner[c][1])));
                }
            }
        }
    }
    Array arrays;
    arrays.resize(Mesh::ARRAY_MAX);
    arrays[Mesh::ARRAY_VERTEX] = verts;
    arrays[Mesh::ARRAY_NORMAL] = normals;
    arrays[Mesh::ARRAY_TEX_UV] = uv;
    arrays[Mesh::ARRAY_TEX_UV2] = uv2;
    mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
    return mesh;
}

PackedVector3Array WeatherDebugNode::wind_lines(int p_level, double p_altitude_m, double p_seconds, int p_every) const {
    PackedVector3Array out;
    const auto snap = session.is_open() ? session.weather().probe() : nullptr;
    if (!snap) {
        return out;
    }
    const WeatherGrid &g = session.weather().grid();
    const int every = ov::max(p_every, 1);
    const int k = ov::clamp(p_level, 0, snap->levels - 1);
    for (int f = 0; f < g.faces(); f++) {
        for (int j = every / 2; j < g.height(); j += every) {
            for (int i = every / 2; i < g.width(); i += every) {
                double pos[3], up[3];
                g.cell_surface(f, i, j, pos, up);
                const Vector3 a = Vector3(float(pos[0] + up[0] * p_altitude_m), float(pos[1] + up[1] * p_altitude_m), float(pos[2] + up[2] * p_altitude_m));
                const float *w = &snap->wind[(snap->interior(f, i, j) * size_t(snap->levels) + size_t(k)) * 3];
                out.push_back(a);
                out.push_back(a + Vector3(w[0], w[1], w[2]) * float(p_seconds));
            }
        }
    }
    return out;
}
