#include "world_file.hpp"

#include "base_surface.hpp"
#include "chunk_math.hpp"
#include "weather_grid.hpp"

#include "ov/files.hpp"
#include "ov/format.hpp"
#include "ov/json.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

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

// p_v as an int64 in decimal text (an anchor's coordinate: JSON numbers are doubles), false if it isn't one.
bool int64_text(const ov::Value &p_v, int64_t &r_out) {
    const ov::str &s = p_v.as_str();
    if (!p_v.is_string() || s.empty()) {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || end != s.c_str() + s.size()) {
        return false;
    }
    r_out = int64_t(v);
    return true;
}

// p_v as a list of p_min to p_max numbers in r_out (as floats), false if it isn't one.
bool float_list(const ov::Value &p_v, size_t p_min, size_t p_max, float *r_out) {
    if (!p_v.is_array() || p_v.size() < p_min || p_v.size() > p_max) {
        return false;
    }
    for (size_t i = 0; i < p_v.size(); i++) {
        if (!p_v[i].is_number()) {
            return false;
        }
        r_out[i] = float(p_v[i].as_double());
    }
    return true;
}

// The edits file's entry p_d (the p_index-th) in r_edit: empty, or why it was refused.
ov::str parse_edit(const ov::Value &p_d, size_t p_index, SDFTable::Edit &r_edit) {
    const ov::str where = ov::format("edits: entry %d", int64_t(p_index));
    if (!p_d.is_object()) {
        return "world: " + where + ": not an object.";
    }
    const ov::str err = unknown_key(p_d, where.c_str(), { "type", "op", "anchor", "rotation", "params", "k" });
    if (!err.empty()) {
        return err;
    }
    SDFTable::Edit e;
    if (!p_d["type"].is_string()) {
        return "world: " + where + ": 'type' is a library type's name.";
    }
    e.type = p_d["type"].as_str();
    if (!Archetype::combine_named(p_d["op"].as_str(), e.op)) {
        return "world: " + where + ": 'op' is union, smooth_union, subtract or smooth_subtract.";
    }
    const ov::Value &a = p_d["anchor"];
    if (!(a.is_array() && a.size() == 3 && int64_text(a[0], e.anchor.x) && int64_text(a[1], e.anchor.y) && int64_text(a[2], e.anchor.z))) {
        return "world: " + where + ": 'anchor' is [\"x\", \"y\", \"z\"], LOD0 voxels as int64 strings.";
    }
    if (p_d.has("rotation") && !float_list(p_d["rotation"], 9, 9, e.rotation)) {
        return "world: " + where + ": 'rotation' is 9 numbers, its rows.";
    }
    if (p_d.has("params") && !float_list(p_d["params"], 0, 8, e.params)) {
        return "world: " + where + ": 'params' is at most 8 numbers.";
    }
    if (p_d.has("k") && !p_d["k"].is_number()) {
        return "world: " + where + ": 'k' is a number.";
    }
    e.k = float(p_d.get("k", 0.0).as_double());
    r_edit = e;
    return ov::str();
}

// p_d[p_key] (absent: r_out untouched) as three numbers, r_error set if it isn't.
void colour(const ov::Value &p_d, const char *p_key, const char *p_where, Float3 &r_out, ov::str &r_error) {
    float c[3];
    if (!r_error.empty() || !p_d.has(p_key)) {
        return;
    }
    if (!float_list(p_d[p_key], 3, 3, c)) {
        r_error = "world: " + ov::str(p_where) + ": '" + p_key + ov::str("' is [r, g, b], three numbers.");
        return;
    }
    r_out = Float3{ c[0], c[1], c[2] };
}

// "shape" into r_world's planet: the kind agreeing with its radius and ring, BaseSurface::planet's limits.
ov::str parse_shape(const ov::Value &p_data, WorldFile &r_world) {
    if (p_data.has("planet")) {
        return "world: 'planet' is now 'shape': { \"kind\": \"flat\" | \"sphere\" | \"torus\", \"radius_m\", \"ring_m\" } (rename it and give its kind).";
    }
    ov::str err;
    const ov::Value shape = section(p_data, "shape", { "kind", "radius_m", "ring_m" }, err);
    ov::Value sizes = shape;
    sizes.erase("kind");
    err = err.empty() ? not_a_number(sizes, "shape") : err;
    if (!err.empty() || !p_data.has("shape")) {
        return err;
    }
    const ov::str kind = shape["kind"].as_str();
    const double radius = shape.get("radius_m", 0.0).as_double(), ring = shape.get("ring_m", 0.0).as_double();
    if (!(radius >= 0.0 && std::isfinite(radius)) || !(ring >= 0.0 && std::isfinite(ring))) {
        return "world: shape: radius_m and ring_m must be finite and >= 0.";
    }
    const bool agrees = (kind == "flat" && radius == 0.0 && ring == 0.0) || (kind == "sphere" && radius > 0.0 && ring == 0.0) || (kind == "torus" && radius > 0.0 && ring > 0.0);
    if (!agrees) {
        return "world: shape: 'kind' is flat (no radius_m, no ring_m), sphere (a radius_m, no ring_m) or torus (radius_m its tube, ring_m its ring).";
    }
    if (radius > 0.0) {
        ov::str refusal;
        BaseSurface::planet(radius, r_world.voxel_m, ring, &refusal);
        if (!refusal.empty()) {
            return "world: shape: " + refusal;
        }
    }
    r_world.planet_radius_m = radius;
    r_world.planet_ring_m = ring;
    return ov::str();
}

// "seed" into r_world: a whole number in [0, 2^32).
ov::str parse_seed(const ov::Value &p_data, WorldFile &r_world) {
    const double s = p_data.get("seed", 0.0).as_double();
    if (!p_data.get("seed", 0.0).is_number() || !(s >= 0.0 && s < 4294967296.0 && s == std::floor(s))) {
        return "world: 'seed' is a whole number in [0, 2^32).";
    }
    r_world.seed = uint32_t(s);
    return ov::str();
}

// "sea" into r_world.sea (on): its level, water and shore.
ov::str parse_sea(const ov::Value &p_data, WorldFile &r_world) {
    ov::str err;
    const ov::Value sea = section(p_data, "sea", { "level_m", "water", "shore" }, err);
    const ov::Value water = section(sea, "water", { "deep", "shallow", "clarity_m" }, err);
    const ov::Value shore = section(sea, "shore", { "below_m", "above_m", "sand" }, err);
    if (!err.empty() || !p_data.has("sea")) {
        return err;
    }
    if (r_world.planet_ring_m > 0.0) {
        // TODO: the sea pass finds the level along a ray in closed form (base_level_span_k); a torus's level is a quartic,
        // sphere-traced per pixel (base_traced_span), too costly for every water pixel. A root solver would close it.
        return "world: sea: a sea on a torus isn't drawn yet (its level along a ray has no closed form): give it a sphere or a flat shape.";
    }
    ov::Value level = sea, clarity = water, band = shore; // their numbers alone
    level.erase("water");
    level.erase("shore");
    clarity.erase("deep");
    clarity.erase("shallow");
    band.erase("sand");
    err = !sea.has("level_m") ? ov::str("world: sea: 'level_m' is required (the sea's altitude above the base).") : not_a_number(level, "sea");
    err = err.empty() ? not_a_number(clarity, "sea water") : err;
    err = err.empty() ? not_a_number(band, "sea shore") : err;
    if (!err.empty()) {
        return err;
    }
    WorldFile::Sea s;
    s.on = true;
    s.level_m = sea["level_m"].as_double();
    s.clarity_m = water.get("clarity_m", s.clarity_m).as_double();
    s.shore_below_m = shore.get("below_m", s.shore_below_m).as_double();
    s.shore_above_m = shore.get("above_m", s.shore_above_m).as_double();
    if (!std::isfinite(s.level_m) || !(s.clarity_m > 0.0 && std::isfinite(s.clarity_m)) || !(s.shore_below_m >= 0.0 && s.shore_above_m >= 0.0 && std::isfinite(s.shore_below_m + s.shore_above_m))) {
        return "world: sea: level_m is finite, clarity_m > 0, the shore's below_m and above_m >= 0.";
    }
    colour(water, "deep", "sea water", s.deep, err);
    colour(water, "shallow", "sea water", s.shallow, err);
    colour(shore, "sand", "sea shore", s.sand, err);
    r_world.sea = s;
    return err;
}

// "weather" into r_world.weather: on over a sphere unless "on" is false; refused over a flat shape or a torus.
ov::str parse_weather(const ov::Value &p_data, WorldFile &r_world) {
    ov::str err;
    const ov::Value d = section(p_data, "weather", { "on", "cell_m", "top_m", "tick_hz", "time_scale", "storm_m", "spin_up_s", "day_s", "inversion_m", "exchange", "convection", "cloud_spread" }, err);
    if (!err.empty()) {
        return err;
    }
    const bool sphere = r_world.planet_radius_m > 0.0 && r_world.planet_ring_m == 0.0;
    WorldFile::Weather w;
    for (const char *key : { "on", "convection", "cloud_spread" }) {
        if (d.has(key) && d[key].get_type() != ov::Value::BOOL) {
            return ov::format("world: weather: '%s' is true or false.", key);
        }
    }
    w.on = sphere && d.get("on", true).as_bool();
    if (!sphere) {
        return p_data.has("weather") ? ov::str("world: weather: the weather is simulated over a sphere only (give the shape a sphere, or drop \"weather\").") : ov::str();
    }
    ov::Value numbers = d;
    numbers.erase("on");
    numbers.erase("convection");
    numbers.erase("cloud_spread");
    err = not_a_number(numbers, "weather");
    if (!err.empty()) {
        return err;
    }
    w.cell_m = d.get("cell_m", w.cell_m).as_double();
    w.top_m = d.get("top_m", w.top_m).as_double();
    w.tick_hz = d.get("tick_hz", w.tick_hz).as_double();
    w.time_scale = d.get("time_scale", w.time_scale).as_double();
    w.storm_m = d.get("storm_m", w.storm_m).as_double();
    w.spin_up_s = d.get("spin_up_s", w.spin_up_s).as_double();
    w.day_s = d.get("day_s", w.day_s).as_double();
    w.inversion_m = d.get("inversion_m", w.inversion_m).as_double();
    w.exchange = d.get("exchange", w.exchange).as_double();
    w.convection = d.get("convection", w.convection).as_bool();
    w.cloud_spread = d.get("cloud_spread", w.cloud_spread).as_bool();
    const double side = CubeSphereGrid::side_for(r_world.planet_radius_m, WeatherGrid::cell_width(r_world.planet_radius_m, w.cell_m));
    const bool ok = w.cell_m >= 0.0 && w.top_m > 0.0 && w.top_m <= 100000.0 && side >= WeatherGrid::MIN_N && side <= WeatherGrid::MAX_N && w.tick_hz > 0.0 && w.tick_hz <= 120.0 && w.time_scale > 0.0 && w.time_scale <= 10000.0 && w.storm_m >= 0.0 && std::isfinite(w.storm_m) && w.spin_up_s >= 0.0 && w.spin_up_s <= 86400.0 && w.day_s >= 1.0 && w.day_s <= 1e7 && w.inversion_m > 0.0 && w.inversion_m < w.top_m && w.exchange > 0.0 && w.exchange <= 1000.0;
    if (!ok) {
        return ov::format("world: weather: cell_m giving a cube face of [%d, %d] cells (%g m: %g), top_m in (0, 100000], tick_hz in (0, 120], time_scale in (0, 10000], storm_m >= 0, spin_up_s in [0, 86400], day_s in [1, 1e7], inversion_m in (0, top_m), exchange in (0, 1000].",
                WeatherGrid::MIN_N, WeatherGrid::MAX_N, w.cell_m, side);
    }
    r_world.weather = w;
    return ov::str();
}

// "atmosphere" into r_world: the air's density and its Rayleigh tint.
ov::str parse_air(const ov::Value &p_data, WorldFile &r_world) {
    ov::str err;
    const ov::Value air = section(p_data, "atmosphere", { "density", "tint" }, err);
    if (err.empty() && air.has("density") && !(air["density"].is_number() && air["density"].as_double() >= 0.0 && std::isfinite(air["density"].as_double()))) {
        err = "world: atmosphere: 'density' is a number >= 0.";
    }
    r_world.atmosphere_density = air.get("density", r_world.atmosphere_density).as_double();
    colour(air, "tint", "atmosphere", r_world.atmosphere_tint, err);
    return err;
}

// The climate's per-field offsets (every key of p_climate but "fields") merged into r_fields' objects: { "mean",
// "poles" }, numbers; a name that isn't a field's, or poles on a base without poles, refused.
ov::str field_offsets(const ov::Value &p_climate, const WorldFile &p_world, ov::Value &r_fields) {
    for (const std::pair<ov::str, ov::Value> &entry : p_climate.entries()) {
        if (entry.first == "fields" || entry.first == "comment") {
            continue;
        }
        const ov::str where = "climate: '" + entry.first + ov::str("'");
        size_t f = 0;
        while (f < r_fields.size() && r_fields[f]["name"].as_str() != entry.first) {
            f++;
        }
        if (f == r_fields.size()) {
            return "world: " + where + " is not a field (a key of 'climate' besides 'fields' names one: { \"mean\", \"poles\" }).";
        }
        const ov::Value &o = entry.second;
        ov::str err = o.is_object() ? unknown_key(o, where.c_str(), { "mean", "poles" }) : "world: " + where + ": not an object.";
        err = err.empty() ? not_a_number(o, where.c_str()) : err;
        if (err.empty() && o.get("poles", 0.0).as_double() != 0.0 && p_world.planet_ring_m > 0.0) {
            // TODO: a torus's latitude is its tube's angle; its gradient needs a formula of its own (BaseSurface::pole_form).
            err = "world: " + where + ": poles on a torus aren't written yet (its latitude is the tube's angle).";
        } else if (err.empty() && o.get("poles", 0.0).as_double() != 0.0 && !(p_world.planet_radius_m > 0.0)) {
            err = "world: " + where + ": poles need a planet (the flat base has none).";
        }
        if (!err.empty()) {
            return err;
        }
        for (const char *key : { "mean", "poles" }) {
            if (o.has(key)) {
                r_fields.at(f)[key] = o[key];
            }
        }
    }
    return ov::str();
}

// The climate's fields (a list, or the file holding it, relative to p_dir) with their offsets (field_offsets).
ov::str climate_fields(const ov::Value &p_climate, const ov::str &p_dir, const WorldFile &p_world, ov::Value &r_fields) {
    r_fields = ov::Value::array();
    ov::str err = list_or_file(p_climate, "fields", p_dir, r_fields);
    if (r_fields.is_nil()) {
        r_fields = ov::Value::array();
    }
    return err.empty() ? field_offsets(p_climate, p_world, r_fields) : err;
}

// p_list's strings read as single-biome files (relative to p_dir), its default biome (no "at") moved last: two refused.
ov::str biome_files(const ov::str &p_dir, ov::Value &r_list) {
    ov::Value out = ov::Value::array(), last;
    for (const ov::Value &v : r_list.elements()) {
        ov::Value b = v;
        if (v.is_string()) {
            const ov::str err = read_object(resolve(p_dir, v.as_str()), b);
            if (!err.empty()) {
                return err;
            }
        }
        if (b.is_object() && !b.has("at")) {
            if (!last.is_nil()) {
                return "world: biomes: two default biomes (no 'at'), '" + last["name"].as_str() + "' and '" + b["name"].as_str() + "': one fills what the others leave.";
            }
            last = b;
            continue;
        }
        out.push_back(b);
    }
    if (!last.is_nil()) {
        out.push_back(last);
    }
    r_list = out;
    return ov::str();
}

// "biomes" (a list, or the file holding it) with its biome files read (biome_files), in r_list; nil if absent.
ov::str biome_list(const ov::Value &p_data, const ov::str &p_dir, ov::Value &r_list) {
    ov::str err = list_or_file(p_data, "biomes", p_dir, r_list);
    if (err.empty() && !r_list.is_nil()) {
        // A list file's biome files are relative to it.
        const ov::Value &v = p_data["biomes"];
        err = biome_files(v.is_string() ? dir_of(resolve(p_dir, v.as_str())) : p_dir, r_list);
    }
    return err;
}

} // namespace

ov::str WorldFile::parse_edits(const ov::Value &p_data, std::vector<SDFTable::Edit> &r_edits) {
    if (!p_data.is_object()) {
        return "world: edits: not a JSON object.";
    }
    const ov::str err = unknown_key(p_data, "edits", { "version", "edits" });
    if (!err.empty()) {
        return err;
    }
    if (!(p_data["version"].is_number() && p_data["version"].as_double() == 1.0)) {
        return "world: edits: \"version\": 1 is the only version.";
    }
    const ov::Value list = p_data.get("edits", ov::Value::array());
    if (!list.is_array()) {
        return "world: edits: 'edits' is a list.";
    }
    std::vector<SDFTable::Edit> out(list.size());
    for (size_t i = 0; i < list.size(); i++) {
        const ov::str entry = parse_edit(list[i], i, out[i]);
        if (!entry.empty()) {
            return entry;
        }
    }
    r_edits = out;
    return ov::str();
}

ov::str WorldFile::load_edits(const ov::str &p_path, std::vector<SDFTable::Edit> &r_edits) {
    ov::Value data;
    const ov::str err = read_object(p_path, data);
    return err.empty() ? parse_edits(data, r_edits) : err;
}

ov::str WorldFile::edits_text(const std::vector<SDFTable::Edit> &p_edits) {
    static const SDFTable::Edit identity;
    auto floats = [](const float *p_v, size_t p_count) {
        ov::Value list = ov::Value::array();
        for (size_t i = 0; i < p_count; i++) {
            list.push_back(p_v[i]);
        }
        return list;
    };
    ov::str out = "{\"version\": 1, \"edits\": [";
    for (size_t i = 0; i < p_edits.size(); i++) {
        const SDFTable::Edit &e = p_edits[i];
        ov::Value d = ov::Value::object();
        d["type"] = e.type;
        d["op"] = Archetype::combine_name(e.op);
        ov::Value anchor = ov::Value::array();
        for (const int64_t c : { e.anchor.x, e.anchor.y, e.anchor.z }) {
            anchor.push_back(std::to_string(c));
        }
        d["anchor"] = anchor;
        if (std::memcmp(e.rotation, identity.rotation, sizeof(e.rotation)) != 0) {
            d["rotation"] = floats(e.rotation, 9);
        }
        size_t params = 8; // the trailing +0s left out (read back as 0)
        while (params > 0 && std::memcmp(&e.params[params - 1], &identity.params[0], sizeof(float)) == 0) {
            params--;
        }
        d["params"] = floats(e.params, params);
        d["k"] = e.k;
        out += (i == 0 ? "\n  " : ",\n  ") + ov::to_json(d);
    }
    return out + (p_edits.empty() ? "]}\n" : "\n]}\n");
}

ov::str WorldFile::save_edits(const ov::str &p_path, const std::vector<SDFTable::Edit> &p_edits) {
    if (!ov::write_file(p_path, edits_text(p_edits))) {
        return "world: cannot write " + p_path + " (no such directory, or a host that writes no file: Dawn).";
    }
    return ov::str();
}

namespace {

// "engine" (an object or its file) into r_world: the voxel and the LOD count.
ov::str parse_engine(const ov::Value &p_data, const ov::str &p_dir, WorldFile &r_world) {
    ov::Value engine;
    ov::str err = object_or_file(p_data, "engine", p_dir, engine);
    err = err.empty() ? unknown_key(engine, "engine", { "voxel_m", "lods" }) : err;
    err = err.empty() ? not_a_number(engine, "engine") : err;
    if (!err.empty()) {
        return err;
    }
    r_world.voxel_m = engine.get("voxel_m", r_world.voxel_m).as_double();
    r_world.lods = int(engine.get("lods", r_world.lods).as_int());
    if (!(r_world.voxel_m > 0.0)) {
        return "world: engine: voxel_m must be positive.";
    }
    if (r_world.lods < 1 || r_world.lods > ChunkMath::MAX_LOD_NUMBERS) {
        return ov::format("world: engine: lods must be in [1, %d].", ChunkMath::MAX_LOD_NUMBERS);
    }
    return ov::str();
}

// "look" and "feature_region" into r_world.
ov::str parse_look(const ov::Value &p_data, WorldFile &r_world) {
    ov::str err;
    const ov::Value look = section(p_data, "look", { "atmosphere", "atmosphere_scale", "clouds" }, err);
    ov::Value region = section(p_data, "feature_region", { "side_m", "centre_m" }, err);
    const ov::Value centre = region.get("centre_m", ov::Value::array());
    if (err.empty() && region.has("centre_m") && !(centre.is_array() && centre.size() == 2 && centre[size_t(0)].is_number() && centre[size_t(1)].is_number())) {
        err = "world: feature_region: centre_m is [x, z] in metres.";
    }
    region.erase("centre_m");
    err = err.empty() ? not_a_number(region, "feature_region") : err;
    if (!err.empty()) {
        return err;
    }
    WorldFile &w = r_world;
    w.atmosphere = look.get("atmosphere", w.atmosphere).as_bool();
    w.atmosphere_scale = look.get("atmosphere_scale", w.atmosphere_scale).as_double();
    w.clouds = look.get("clouds", w.clouds).as_bool();
    w.feature_region.side_m = region.get("side_m", w.feature_region.side_m).as_double();
    if (centre.size() == 2) {
        w.feature_region.x_m = centre[size_t(0)].as_double();
        w.feature_region.z_m = centre[size_t(1)].as_double();
    }
    if (!(w.feature_region.side_m >= 0.0 && std::isfinite(w.feature_region.side_m))) {
        return "world: feature_region: side_m must be >= 0.";
    }
    return ov::str();
}

// "instances", "features", "species", "instance_band" and "edits" into r_world.
ov::str parse_contents(const ov::Value &p_data, const ov::str &p_dir, WorldFile &r_world) {
    WorldFile &w = r_world;
    ov::str err;
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
    ov::Value species;
    err = list_or_file(p_data, "species", p_dir, species);
    if (err.empty() && !species.is_nil()) {
        err = Archetype::parse_species(species, w.species);
        err = err.empty() ? err : "world: " + err;
    }
    if (!err.empty()) {
        return err;
    }
    if (p_data.has("instance_band")) {
        if (p_data["instance_band"].get_type() != ov::Value::BOOL) {
            return "world: 'instance_band' is true or false.";
        }
        w.instance_band = p_data["instance_band"].as_bool();
    }
    if (p_data.has("edits")) {
        ov::Value edits;
        err = object_or_file(p_data, "edits", p_dir, edits);
        err = err.empty() ? WorldFile::parse_edits(edits, w.edits) : err;
    }
    return err;
}

// "climate" (an object or its file: its fields, a list or their file, and its per-field offsets) and "biomes" (a list or
// their file, each a biome or its file) into r_world's climate table (biomes without a climate: no fields), after the
// shape (the offsets' poles need one).
ov::str parse_climate(const ov::Value &p_data, const ov::str &p_dir, WorldFile &r_world) {
    ov::Value climate, fields, biomes;
    ov::str err = object_or_file(p_data, "climate", p_dir, climate);
    // A climate file's fields file is relative to it.
    const ov::str climate_dir = p_data["climate"].is_string() ? dir_of(resolve(p_dir, p_data["climate"].as_str())) : p_dir;
    err = err.empty() ? climate_fields(climate, climate_dir, r_world, fields) : err;
    err = err.empty() ? biome_list(p_data, p_dir, biomes) : err;
    if (err.empty() && (p_data.has("climate") || p_data.has("biomes"))) {
        r_world.climate_table["fields"] = fields;
        r_world.climate_table["biomes"] = biomes.is_nil() ? ov::Value::array() : biomes;
    }
    return err;
}

} // namespace

ov::str WorldFile::parse(const ov::Value &p_data, const ov::str &p_dir, WorldFile &r_world) {
    // "planet" is known only to be refused with its new name (parse_shape).
    ov::str err = unknown_key(p_data, "file",
            { "engine", "seed", "shape", "planet", "sea", "climate", "atmosphere", "weather", "look", "instances", "features", "feature_region", "species", "biomes", "instance_band", "edits" });
    WorldFile w;
    err = err.empty() ? parse_engine(p_data, p_dir, w) : err;
    err = err.empty() ? parse_shape(p_data, w) : err;
    err = err.empty() ? parse_seed(p_data, w) : err;
    err = err.empty() ? parse_sea(p_data, w) : err;
    err = err.empty() ? parse_air(p_data, w) : err;
    err = err.empty() ? parse_weather(p_data, w) : err;
    err = err.empty() ? parse_look(p_data, w) : err;
    err = err.empty() ? parse_contents(p_data, p_dir, w) : err;
    err = err.empty() ? parse_climate(p_data, p_dir, w) : err;
    if (err.empty()) {
        r_world = w;
    }
    return err;
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
