#include "world_file.hpp"

#include "base_surface.hpp"
#include "chunk_math.hpp"

#include "ov/files.hpp"
#include "ov/format.hpp"
#include "ov/json.hpp"

#include <cmath>
#include <initializer_list>

namespace {

// The directory of p_path, with its last slash; empty for a bare name.
ov::str dir_of(const ov::str &p_path) {
    const size_t slash = p_path.rfind('/');
    return slash == ov::str::npos ? ov::str() : p_path.substr(0, slash + 1);
}

bool is_absolute(const ov::str &p_path) {
    return (!p_path.empty() && p_path[0] == '/') || p_path.find("://") != ov::str::npos;
}

ov::str read_object(const ov::str &p_path, ov::Value &r_data) {
    ov::str text;
    if (!ov::read_file(p_path, text)) {
        return "world: cannot read " + p_path + ov::str(".");
    }
    return WorldFile::parse_object(text, p_path, r_data);
}

ov::str resolve(const ov::str &p_dir, const ov::str &p_path) {
    return is_absolute(p_path) ? p_path : ov::path_join(p_dir, p_path);
}

// The object at p_data[p_key], or the file its string names (relative to p_dir), in r_part; empty if the key is absent.
ov::str object_or_file(const ov::Value &p_data, const char *p_key, const ov::str &p_dir, ov::Value &r_part) {
    r_part = ov::Value::object();
    if (!p_data.has(p_key)) {
        return ov::str();
    }
    const ov::Value &v = p_data[p_key];
    if (v.is_string()) {
        return read_object(resolve(p_dir, v.as_str()), r_part);
    }
    if (!v.is_object()) {
        return "world: '" + ov::str(p_key) + ov::str("' is not a path or an object.");
    }
    r_part = v;
    return ov::str();
}

// The list at p_data[p_key], or the file its string names (relative to p_dir, holding the list), in r_list; empty if
// the key is absent (r_list nil).
ov::str list_or_file(const ov::Value &p_data, const char *p_key, const ov::str &p_dir, ov::Value &r_list) {
    r_list = ov::Value();
    if (!p_data.has(p_key)) {
        return ov::str();
    }
    const ov::Value &v = p_data[p_key];
    if (v.is_string()) {
        return WorldFile::load_list(resolve(p_dir, v.as_str()), r_list);
    }
    if (!v.is_array()) {
        return "world: '" + ov::str(p_key) + ov::str("' is not a path or a list.");
    }
    r_list = v;
    return ov::str();
}

// Every key of p_d but "comment" is one of p_allowed: empty, or the first that isn't.
ov::str unknown_key(const ov::Value &p_d, const char *p_where, std::initializer_list<const char *> p_allowed) {
    for (const std::pair<ov::str, ov::Value> &entry : p_d.entries()) {
        bool known = entry.first == "comment";
        for (const char *key : p_allowed) {
            known = known || entry.first == key;
        }
        if (!known) {
            return "world: " + ov::str(p_where) + ov::str(": unknown key '") + entry.first + ov::str("'.");
        }
    }
    return ov::str();
}

// The first value of p_d (but "comment") that isn't a number: empty, or its error.
ov::str not_a_number(const ov::Value &p_d, const char *p_where) {
    for (const std::pair<ov::str, ov::Value> &entry : p_d.entries()) {
        if (entry.first != "comment" && !entry.second.is_number()) {
            return "world: " + ov::str(p_where) + ov::str(": '") + entry.first + ov::str("' is not a number.");
        }
    }
    return ov::str();
}

// p_data[p_key] as a section of p_allowed keys (none: an empty one), r_error set if it isn't.
ov::Value section(const ov::Value &p_data, const char *p_key, std::initializer_list<const char *> p_allowed, ov::str &r_error) {
    const ov::Value v = p_data.get(p_key, ov::Value::object());
    if (!r_error.empty()) {
        return v;
    }
    r_error = v.is_object() ? unknown_key(v, p_key, p_allowed) : "world: '" + ov::str(p_key) + ov::str("' is not an object.");
    return v;
}

} // namespace

ov::str WorldFile::parse(const ov::Value &p_data, const ov::str &p_dir, WorldFile &r_world) {
    ov::str err = unknown_key(p_data, "file", { "engine", "climate", "planet", "look", "zone", "instances", "features", "species", "biomes" });
    if (!err.empty()) {
        return err;
    }
    ov::Value engine, climate;
    err = object_or_file(p_data, "engine", p_dir, engine);
    if (err.empty()) {
        err = object_or_file(p_data, "climate", p_dir, climate);
    }
    if (err.empty()) {
        err = unknown_key(engine, "engine", { "voxel_m", "lods" });
    }
    if (err.empty()) {
        err = unknown_key(climate, "climate", { "fields" });
    }
    if (!err.empty()) {
        return err;
    }
    WorldFile w;
    const ov::Value planet = section(p_data, "planet", { "radius_m", "ring_m" }, err);
    const ov::Value look = section(p_data, "look", { "atmosphere", "atmosphere_scale", "clouds" }, err);
    ov::Value zone = section(p_data, "zone", { "side_m", "centre_m" }, err);
    const ov::Value centre = zone.get("centre_m", ov::Value::array());
    if (err.empty() && zone.has("centre_m") && !(centre.is_array() && centre.size() == 2 && centre[size_t(0)].is_number() && centre[size_t(1)].is_number())) {
        err = "world: zone: centre_m is [x, z] in metres.";
    }
    zone.erase("centre_m");
    for (const auto &part : { std::pair<const char *, ov::Value>{ "engine", engine }, { "planet", planet }, { "zone", zone } }) {
        err = err.empty() ? not_a_number(part.second, part.first) : err;
    }
    if (!err.empty()) {
        return err;
    }
    w.voxel_m = engine.get("voxel_m", w.voxel_m).as_double();
    w.lods = int(engine.get("lods", w.lods).as_int());
    if (!(w.voxel_m > 0.0)) {
        return "world: engine: voxel_m must be positive.";
    }
    if (w.lods < 1 || w.lods > ChunkMath::MAX_LOD_NUMBERS) {
        return ov::format("world: engine: lods must be in [1, %d].", ChunkMath::MAX_LOD_NUMBERS);
    }
    w.planet_radius_m = planet.get("radius_m", w.planet_radius_m).as_double();
    w.planet_ring_m = planet.get("ring_m", w.planet_ring_m).as_double();
    if (!(w.planet_radius_m >= 0.0 && std::isfinite(w.planet_radius_m)) || !(w.planet_ring_m >= 0.0 && std::isfinite(w.planet_ring_m))) {
        return "world: planet: radius_m and ring_m must be finite and >= 0.";
    }
    if (w.planet_ring_m > 0.0 && !(w.planet_radius_m > 0.0)) {
        return "world: planet: ring_m needs radius_m (the torus's tube).";
    }
    if (w.planet_radius_m > 0.0) {
        ov::str refusal;
        BaseSurface::planet(w.planet_radius_m, w.voxel_m, w.planet_ring_m, &refusal);
        if (!refusal.empty()) {
            return "world: planet: " + refusal;
        }
    }
    w.atmosphere = look.get("atmosphere", w.atmosphere).as_bool();
    w.atmosphere_scale = look.get("atmosphere_scale", w.atmosphere_scale).as_double();
    w.clouds = look.get("clouds", w.clouds).as_bool();
    w.zone.side_m = zone.get("side_m", w.zone.side_m).as_double();
    if (centre.size() == 2) {
        w.zone.x_m = centre[size_t(0)].as_double();
        w.zone.z_m = centre[size_t(1)].as_double();
    }
    if (!(w.zone.side_m >= 0.0 && std::isfinite(w.zone.side_m))) {
        return "world: zone: side_m must be >= 0.";
    }
    if (p_data.has("instances")) {
        err = Climate::parse_instances(p_data["instances"], "instances", w.instances);
        if (!err.empty()) {
            return "world: " + err;
        }
    }
    if (p_data.has("features")) {
        err = Climate::parse_features(p_data["features"], "features", w.features);
        if (!err.empty()) {
            return "world: " + err;
        }
    }
    ov::Value species, biomes;
    err = list_or_file(p_data, "species", p_dir, species);
    if (err.empty() && !species.is_nil()) {
        err = Archetype::parse_species(species, w.species);
        err = err.empty() ? err : "world: " + err;
    }
    err = err.empty() ? list_or_file(p_data, "biomes", p_dir, biomes) : err;
    if (!err.empty()) {
        return err;
    }
    // Biomes without a climate: no fields.
    if (p_data.has("climate") || p_data.has("biomes")) {
        w.climate_table["fields"] = climate.get("fields", ov::Value::array());
        w.climate_table["biomes"] = biomes.is_nil() ? ov::Value::array() : biomes;
    }
    r_world = w;
    return ov::str();
}

ov::str WorldFile::parse_object(const ov::str &p_text, const ov::str &p_name, ov::Value &r_data) {
    ov::str json_error;
    r_data = ov::parse_json(p_text, json_error);
    if (!r_data.is_object()) {
        return "world: " + p_name + ov::str(" is not a JSON object.") + (json_error.empty() ? ov::str() : " " + json_error);
    }
    return ov::str();
}

ov::str WorldFile::parse_text(const ov::str &p_text, const ov::str &p_dir, WorldFile &r_world) {
    ov::Value data;
    const ov::str err = parse_object(p_text, "the world's text", data);
    return err.empty() ? parse(data, p_dir, r_world) : err;
}

ov::str WorldFile::load_list(const ov::str &p_path, ov::Value &r_list) {
    ov::str text, json_error;
    if (!ov::read_file(p_path, text)) {
        return "world: cannot read " + p_path + ov::str(".");
    }
    const ov::Value v = ov::parse_json(text, json_error);
    if (!v.is_array()) {
        return "world: " + p_path + ov::str(" is not a JSON list.") + (json_error.empty() ? ov::str() : " " + json_error);
    }
    r_list = v;
    return ov::str();
}

ov::str WorldFile::ground_table(ov::Value &r_table) {
    ov::Value biomes;
    const ov::str err = load_list(GROUND_BIOMES, biomes);
    if (err.empty()) {
        r_table = ov::Value::object();
        r_table["fields"] = ov::Value::array();
        r_table["biomes"] = biomes;
    }
    return err;
}

ov::str WorldFile::load_species(const ov::str &p_path, std::vector<Archetype::Species> &r_species) {
    ov::Value list;
    ov::str err = load_list(p_path, list);
    err = err.empty() ? Archetype::parse_species(list, r_species) : err;
    return err.empty() ? err : p_path + ": " + err;
}

ov::str WorldFile::load(const ov::str &p_path, WorldFile &r_world) {
    ov::Value data;
    const ov::str err = read_object(p_path, data);
    return err.empty() ? parse(data, dir_of(p_path), r_world) : err;
}
