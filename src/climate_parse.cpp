#include "climate.hpp"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/variant/array.hpp>

#include <algorithm>
#include <cmath>

// A climate table's validation and its JSON format (parse, load).
namespace Climate {

namespace {

// Coefficients, bounds and scales: |x| within [2^-40, 2^40], so their products with field values stay normal.
bool moderate(double p_v) {
    const double a = std::fabs(p_v);
    return std::isfinite(p_v) && a >= std::ldexp(1.0, -40) && a <= std::ldexp(1.0, 40);
}

const Cascade::Band *find_band(const std::vector<Cascade::Band> &p_bands, int p_k) {
    for (const Cascade::Band &b : p_bands) {
        if (b.k == p_k) {
            return &b;
        }
    }
    return nullptr;
}

// The bands a term or multiplier source names: the ground's, a field's, or null.
const std::vector<Cascade::Band> *bands_of(const Table &p_table, const Ground &p_ground, int p_source) {
    if (p_source == GROUND) {
        return p_ground.spec ? &p_ground.spec->bands : nullptr;
    }
    return p_source >= 0 && p_source < int(p_table.fields.size()) ? &p_table.fields[size_t(p_source)].bands : nullptr;
}

template <typename T>
bool name_taken_before(const std::vector<T> &p_list, size_t p_i) {
    for (size_t j = 0; j < p_i; j++) {
        if (p_list[j].name == p_list[p_i].name) {
            return true;
        }
    }
    return false;
}

// A shape's coefficients: each 0 or within [2^-40, 2^40]; a cell noise field's bands (p_cell) are plain.
String validate_shape(const Shape &p_shape, bool p_cell) {
    for (const float c : p_shape.c) {
        if (c != 0.0f && !moderate(c)) {
            return vformat("shape coefficient %s outside [2^-40, 2^40].", c);
        }
    }
    return p_cell && !p_shape.plain() ? String("a cell noise field's bands take no shape.") : String();
}

String validate_field(const Table &p_table, int p_i, const Ground &p_ground) {
    const Field &f = p_table.fields[size_t(p_i)];
    if (!f.name.is_valid_ascii_identifier() || f.name == "ground") {
        return vformat("climate field %d: the name must be an identifier (letters, digits, _), and 'ground' is reserved.", p_i);
    }
    const String where = "climate field '" + f.name + String("': ");
    if (name_taken_before(p_table.fields, size_t(p_i))) {
        return where + String("the name is used twice.");
    }
    for (const Cascade::Band &b : f.bands) {
        if (b.k < ExactNoise::BAND_SHIFT || b.k > p_ground.k_max) {
            return where + vformat("octave k = %d is outside the ground's range [%d, %d].", b.k, ExactNoise::BAND_SHIFT, p_ground.k_max);
        }
        if (find_band(f.bands, b.k) != &b) {
            return where + vformat("octave k = %d is listed twice.", b.k);
        }
        if (!ExactNoise::in_exact_range(b.amp)) {
            return where + vformat("octave k = %d: amplitude %s outside [2^-16, 2^40].", b.k, b.amp);
        }
    }
    for (const Shape &sh : f.shapes) {
        const String err = validate_shape(sh, f.cell);
        if (!err.is_empty()) {
            return where + err;
        }
    }
    for (const Term &t : f.terms) {
        if (t.source != GROUND && (t.source < 0 || t.source >= p_i)) {
            return where + String("a term reads the ground or an earlier field only.");
        }
        if (bands_of(p_table, p_ground, t.source) == nullptr) {
            return where + String("a term reads the ground, and there is none.");
        }
        if (!moderate(t.coef)) {
            return where + vformat("term coefficient %s outside [2^-40, 2^40].", t.coef);
        }
    }
    return String();
}

String validate_amp(const Table &p_table, const Ground &p_ground, const Amp &p_amp) {
    const std::vector<Cascade::Band> *bands = bands_of(p_table, p_ground, p_amp.field);
    const Cascade::Band *band = bands ? find_band(*bands, p_amp.k) : nullptr;
    if (band == nullptr) {
        return vformat("an amplitude multiplier names octave k = %d of a field that doesn't have it.", p_amp.k);
    }
    if (!std::isfinite(p_amp.mult) || p_amp.mult < 0.0f) {
        return vformat("octave k = %d: the multiplier must be >= 0.", p_amp.k);
    }
    if ((p_amp.gain != 0.0f || p_amp.warp != 0.0f) && p_amp.field != GROUND) {
        return vformat("octave k = %d: multifractal and warp shape ground bands only.", p_amp.k);
    }
    if ((p_amp.gain != 0.0f && !(moderate(p_amp.gain) && p_amp.gain > 0.0f)) ||
            (p_amp.warp != 0.0f && !(moderate(p_amp.warp) && std::fabs(p_amp.warp) <= Cascade::WARP_LIMIT))) {
        return vformat("octave k = %d: multifractal gain or warp out of range.", p_amp.k);
    }
    for (const int f : p_amp.warp_fields) {
        if ((p_amp.warp != 0.0f && f < 0) || f >= int(p_table.fields.size())) {
            return vformat("octave k = %d: a warp reads 3 fields that exist (warp_fields).", p_amp.k);
        }
    }
    if (p_amp.has_shape) {
        const String err = validate_shape(p_amp.shape, p_amp.field != GROUND && p_table.fields[size_t(p_amp.field)].cell);
        if (!err.is_empty()) {
            return err;
        }
    }
    // The inserted amplitude: the multiplier's, each shape term's in the exact range, the shaped band under the ground's cap.
    const double amp = double(p_amp.mult) * double(band->amp);
    const Shape shape = p_amp.has_shape ? p_amp.shape : band_shape(p_table, nullptr, p_amp.field, p_amp.k);
    bool ok = ExactNoise::in_exact_range(float(amp)) && (p_amp.field != GROUND || amp * double(shape_bound(shape)) <= p_ground.amp_cap);
    for (int t = 1; t < 4 && ok; t++) {
        ok = shape.c[t] == 0.0f || ExactNoise::in_exact_range(float(amp * std::fabs(double(shape.c[t]))));
    }
    return p_amp.mult == 0.0f || ok ? String() : vformat("octave k = %d: the modulated amplitude %s (shaped) is out of range.", p_amp.k, amp);
}

// A biome's or a region's edges: existing fields, bounds and fades in range.
String validate_edges(const Table &p_table, const std::vector<Edge> &p_edges) {
    for (const Edge &e : p_edges) {
        if (e.field < 0 || e.field >= int(p_table.fields.size())) {
            return "an edge reads a field that doesn't exist.";
        }
        if ((e.anchor != 0.0f && !moderate(e.anchor)) || !moderate(e.scale)) {
            return vformat("edge on '%s': bound or fade out of range.", p_table.fields[size_t(e.field)].name);
        }
    }
    return String();
}

String validate_region(const Table &p_table, size_t p_i) {
    const Region &r = p_table.regions[p_i];
    if (!r.name.is_valid_ascii_identifier()) {
        return vformat("climate region %d: the name must be an identifier (letters, digits, _).", int64_t(p_i));
    }
    const String where = "climate region '" + r.name + String("': ");
    if (name_taken_before(p_table.regions, p_i)) {
        return where + String("the name is used twice.");
    }
    if (r.edges.empty()) {
        return where + String("a region has at least one edge.");
    }
    const String err = validate_edges(p_table, r.edges);
    return err.is_empty() ? String() : where + err;
}

String validate_biome(const Table &p_table, size_t p_i, const Ground &p_ground) {
    const Biome &b = p_table.biomes[p_i];
    if (b.name.is_empty()) {
        return vformat("biome %d: a name is required.", int64_t(p_i));
    }
    const String where = "biome '" + b.name + String("': ");
    if (name_taken_before(p_table.biomes, p_i)) {
        return where + String("the name is used twice.");
    }
    const bool last = p_i + 1 == p_table.biomes.size();
    if (last != b.edges.empty()) {
        return where + String(last ? "the last biome is the default: no region." : "only the last biome (the default) has no region.");
    }
    const String err = validate_edges(p_table, b.edges);
    if (!err.is_empty()) {
        return where + err;
    }
    if (b.terrace.step != 0.0f && !(moderate(b.terrace.step) && b.terrace.step > 0.0f)) {
        return where + String("the terrace step must be > 0.");
    }
    for (size_t j = 0; j < b.amps.size(); j++) {
        for (size_t i = 0; i < j; i++) {
            if (b.amps[i].field == b.amps[j].field && b.amps[i].k == b.amps[j].k) {
                return where + vformat("octave k = %d of one field has two multipliers.", b.amps[j].k);
            }
        }
        const String err = validate_amp(p_table, p_ground, b.amps[j]);
        if (!err.is_empty()) {
            return where + err;
        }
    }
    return String();
}

} // namespace

String validate(const Table &p_table, const Ground &p_ground) {
    if (p_table.fields.size() > size_t(MAX_FIELDS)) {
        return vformat("climate: %d fields, at most %d.", int64_t(p_table.fields.size()), MAX_FIELDS);
    }
    if (p_table.biomes.empty() || p_table.biomes.size() > size_t(MAX_BIOMES)) {
        return vformat("climate: %d biomes, 1 to %d.", int64_t(p_table.biomes.size()), MAX_BIOMES);
    }
    if (p_table.regions.size() > size_t(MAX_REGIONS)) {
        return vformat("climate: %d regions, at most %d.", int64_t(p_table.regions.size()), MAX_REGIONS);
    }
    for (int i = 0; i < int(p_table.fields.size()); i++) {
        const String err = validate_field(p_table, i, p_ground);
        if (!err.is_empty()) {
            return err;
        }
    }
    for (size_t i = 0; i < p_table.regions.size(); i++) {
        const String err = validate_region(p_table, i);
        if (!err.is_empty()) {
            return err;
        }
    }
    for (size_t i = 0; i < p_table.biomes.size(); i++) {
        const String err = validate_biome(p_table, i, p_ground);
        if (!err.is_empty()) {
            return err;
        }
    }
    if (p_table.has_stretch() && (p_ground.spec == nullptr || p_ground.spec->set != Cascade::ChannelSet::LINEAR4)) {
        return "climate: stretched bands need a LINEAR4 ground.";
    }
    if (p_table.has_state() && p_table.fields.size() + 1 > size_t(Cascade::MAX_FIELD_GROUPS)) {
        return vformat("climate: a multifractal carries its state as a field group: at most %d fields.", Cascade::MAX_FIELD_GROUPS - 1);
    }
    for (const HeightTerm &h : p_table.heights) {
        if (h.field < 0 || h.field >= int(p_table.fields.size())) {
            return "climate: a height term reads a field that doesn't exist.";
        }
        if (!moderate(h.coef)) {
            return vformat("climate: height term coefficient %s outside [2^-40, 2^40] voxels.", h.coef);
        }
        if (h.region >= int(p_table.regions.size()) || (h.region >= 0 && !h.has_curve)) {
            return "climate: a height term's region must exist, and a gated term takes a curve (bounded: its skirt step).";
        }
    }
    return String();
}

// The format (JSON; distances in metres, rounded to whole octaves 2^k voxels; "comment" keys are ignored, any
// other unknown key is an error):
//   { "fields": [ field... ], "regions": [ region... ] (optional), "heights": [ height... ] (optional),
//     "ground_stretch": [ stretch... ] (optional), "biomes": [ biome..., default biome ] }
//   field:  { "name": "temperature", "seed": 7 (optional, else a hash of the name), "noise": "perlin" | "cell" (optional),
//             "bands": [ { "wavelength_m": 5000, "amp": 1, shape (optional), "stretch_y": 1 (optional) }
//                      | { "from_m": 1600, "to_m": 210000, "amp": 1, "persistence": 1 (optional), shape (optional) } ],
//             "terms": [ { "source": "ground", "per_m": -0.002 } | { "source": "<earlier field>", "x": 0.5 } ] }
//   stretch: { "from_m": 100, "to_m": 30000000 (or "wavelength_m"), "stretch_y": 4 }: the ground's octaves in range
//           read their noise at y / 4 (their wavelength along y 4 times theirs along x and z); a field band takes
//           "stretch_y" too. A power of two; octave k takes at most 2^(k - 3) of it (Cascade::stretched: a range's finest
//           octaves are stretched less), and no octave is stretched twice. The ground is a 3D noise
//           whose vertical slope is as large as its horizontal one: past dh/dy = 1 it folds into overhangs (steep
//           landforms: mountains, cliff faces, valley walls, tools/landforms); stretched octaves keep it a height field
//           where they dominate.
//   shape:  "shape": "plain" | "billow" | "ridged" | "square", "scale": a, "offset": o (both optional): a n + o,
//           2 |a n| - 1 + o, (o - |a n|)^2 (o default 1), (a n)^2 + o; or "coeffs": [c1, cn, cabs, csq] (Shape). A band
//           inserts its shape without its mean; |n| is smoothed below the lattice's resolution (Cascade::smooth_abs).
//   region: { "name": "cold", "where": [ edge... ] (a biome's "where") }: a part of the fields' space, for several
//           landforms in one table (macro climate fields as its edges). A biome's or a height's "region" names one.
//   height: { "field": "continent", "height_m": 2048, "curve": [[x, y], ...] (optional, x increasing),
//             "region": "cold" (optional, with a curve) }: the ground adds height_m * curve(field) at each sample
//           (libnoise's Curve), height_m * field without a curve, times the region's value with one (1 in its core,
//           0 outside its fades: a landform's cliff, trough or coast only in its region).
//     A cell noise field ("noise": "cell") is Worley F1 (ExactNoise::cell3), plain bands only: read through a height
//     term (dunes, cells).
//     A range is every octave from from_m to to_m that the ground has (its octaves depend on the LOD count: the
//     table covers the whole spectrum at any count), amp at to_m (the coarsest) times persistence per finer octave.
//     Octave k's seed and lattice are make_octave(field seed, k): editing one band leaves the others' unchanged.
//     A ground term reads the ground's height above the datum (per metre); a field term that field's value.
//   biome:  { "name": "volcanic", "region": "warm" (optional: its edges come before the biome's own),
//             "where": [ { "field": "temperature", "min": 0.5, "max": 2 (either or both), "fade": 0.1
//                          (or "fade_min" / "fade_max") } ],
//             "amp": [ { "field": "ground" | "<field>", "wavelength_m": 4000 | "from_m": 1000, "to_m": 20000, "x": 3 (default 1),
//                        shape (optional, else the band's own), "multifractal": 2 (gain, ground only),
//                        "warp_m": 80, "warp_fields": ["wx", "wy", "wz"] (ground only) } ],
//             "terrace": { "step_m": 40, "invert": false } (optional),
//             "surface": { "flat": "#2a2420", "steep": "#1a1614" (optional, else flat), "roughness": 1 },
//             "debug_color": "#c04020" }
//     "multifractal": the octave is weighted by clamp(gain (1 - c), 0, 1), c the multifractal state's complement (the
//     previous multifractal octave's weighted signal: Musgrave's ridged multifractal with gain 2 and the ridged shape,
//     hybrid with 1 and plain + offset). "warp_m": its noise is read displaced by warp_m times the three fields' values
//     (libnoise's turbulence). "terrace": libnoise's Terrace of the ground height every step_m, blended by the weight.
//     The region is 1 between min and max, fading to 0 over the fade outside. A range of "amp" covers the
//     octaves the field has in it (none, past the ground's octaves at this LOD count). Octave k reads the weights
//     from the fields' octaves coarser than k only (Cascade::Modulation): a region shapes the ground's octaves up to
//     about the size of its fields' octaves, so flattening the continental octaves takes continental regions. The
//     last biome is the default: no "where". "surface": the biome's ground colours
//     (sRGB) on gentle and steep slopes; without it, the procedural ground's.
namespace {

class Parser {
public:
    Parser(const Ground &p_ground, Table &r_table) :
            ground(p_ground), table(r_table) {}

    String error;

    void run(const Dictionary &p_data) {
        if (!keys(p_data, "climate", { "fields", "regions", "biomes", "heights", "ground_stretch" })) {
            return;
        }
        const Array stretch = array(p_data, "ground_stretch", "climate");
        for (int64_t i = 0; i < stretch.size() && error.is_empty(); i++) {
            ground_stretch(dict(stretch[i], vformat("ground_stretch %d", i)), vformat("ground_stretch %d", i));
        }
        const Array fields = array(p_data, "fields", "climate");
        for (int64_t i = 0; i < fields.size() && error.is_empty(); i++) {
            field(dict(fields[i], vformat("field %d", i)), i);
        }
        const Array regions = array(p_data, "regions", "climate");
        for (int64_t i = 0; i < regions.size() && error.is_empty(); i++) {
            region(dict(regions[i], vformat("region %d", i)), i);
        }
        const Array heights = array(p_data, "heights", "climate");
        for (int64_t i = 0; i < heights.size() && error.is_empty(); i++) {
            height(dict(heights[i], vformat("height %d", i)), vformat("height %d", i));
        }
        const Array biomes = array(p_data, "biomes", "climate");
        for (int64_t i = 0; i < biomes.size() && error.is_empty(); i++) {
            biome(dict(biomes[i], vformat("biome %d", i)), i);
        }
    }

private:
    const Ground &ground;
    Table &table;

    // The first error only: "climate: <where>: <msg>".
    void fail(const String &p_where, const String &p_msg) {
        if (error.is_empty()) {
            error = "climate: " + p_where + String(": ") + p_msg;
        }
    }

    void fail_key(const String &p_where, const char *p_key, const char *p_msg) {
        fail(p_where, "'" + String(p_key) + String("' ") + String(p_msg));
    }

    bool keys(const Dictionary &p_d, const String &p_where, std::initializer_list<const char *> p_allowed) {
        const Array k = p_d.keys();
        for (int64_t i = 0; i < k.size(); i++) {
            const String key = k[i];
            if (key != "comment" && std::none_of(p_allowed.begin(), p_allowed.end(), [&](const char *p_a) { return key == p_a; })) {
                fail(p_where, "unknown key '" + key + String("'."));
            }
        }
        return error.is_empty();
    }

    Dictionary dict(const Variant &p_v, const String &p_where) {
        if (p_v.get_type() != Variant::DICTIONARY) {
            fail(p_where, "an object is expected.");
            return Dictionary();
        }
        return p_v;
    }

    Array array(const Dictionary &p_d, const char *p_key, const String &p_where) {
        if (!p_d.has(p_key)) {
            return Array();
        }
        if (p_d[p_key].get_type() != Variant::ARRAY) {
            fail_key(p_where, p_key, "must be a list.");
            return Array();
        }
        return p_d[p_key];
    }

    static bool is_number(const Variant &p_v) { return p_v.get_type() == Variant::INT || p_v.get_type() == Variant::FLOAT; }

    // The number at p_key; absent: p_default, or an error if p_default is NaN (required).
    double number(const Dictionary &p_d, const char *p_key, const String &p_where, double p_default = NAN) {
        if (!p_d.has(p_key)) {
            if (std::isnan(p_default)) {
                fail_key(p_where, p_key, "is required.");
            }
            return p_default;
        }
        const Variant v = p_d[p_key];
        if (!is_number(v)) {
            fail_key(p_where, p_key, "must be a number.");
            return 0.0;
        }
        return double(v);
    }

    String text(const Dictionary &p_d, const char *p_key, const String &p_where) {
        if (!p_d.has(p_key) || p_d[p_key].get_type() != Variant::STRING) {
            fail_key(p_where, p_key, "must be a string.");
            return String();
        }
        return p_d[p_key];
    }

    // [lo, hi] octaves of a band or amplitude entry: wavelength_m, or from_m .. to_m.
    bool octave_range(const Dictionary &p_d, const String &p_where, int &r_lo, int &r_hi) {
        const bool single = p_d.has("wavelength_m");
        if (single && (p_d.has("from_m") || p_d.has("to_m"))) {
            fail(p_where, "wavelength_m or from_m / to_m, not both.");
            return false;
        }
        const double lo_m = number(p_d, single ? "wavelength_m" : "from_m", p_where);
        const double hi_m = single ? lo_m : number(p_d, "to_m", p_where);
        if (error.is_empty() && !(lo_m > 0.0 && hi_m >= lo_m && std::isfinite(hi_m))) {
            fail(p_where, "wavelengths must be > 0 and from_m <= to_m.");
        }
        if (!error.is_empty()) {
            return false;
        }
        r_lo = ExactNoise::lattice_k(lo_m, ground.voxel_size);
        r_hi = ExactNoise::lattice_k(hi_m, ground.voxel_size);
        // make_octave's shifts need k in [0, MAX_K]; validate() then holds bands to the ground's range.
        if (r_lo < 0 || r_hi > ExactNoise::MAX_K) {
            fail(p_where, vformat("octaves %d .. %d: outside [0, %d].", r_lo, r_hi, ExactNoise::MAX_K));
        }
        return error.is_empty();
    }

    // The field a name refers to: GROUND for "ground" (if p_ground_ok), else a field index below p_before (p_expect
    // names the allowed ones in the error).
    int source(const String &p_name, const String &p_where, bool p_ground_ok, int p_before, const char *p_expect) {
        if (p_name == "ground" && p_ground_ok) {
            return GROUND;
        }
        const int i = table.field_index(p_name);
        if (i < 0 || i >= p_before) {
            fail(p_where, "'" + p_name + String("' is not ") + String(p_expect) + String("."));
        }
        return i;
    }

    static uint32_t name_hash(const String &p_name) {
        const CharString s = p_name.utf8();
        uint32_t h = 2166136261u;
        for (int i = 0; i < s.length(); i++) {
            h = (h ^ uint8_t(s[i])) * 16777619u;
        }
        return h;
    }

    // A shape (shape, scale, offset or coeffs) of an entry: the plain one without any.
    bool shape(const Dictionary &p_d, const String &p_where, Shape &r_s) {
        if (p_d.has("coeffs")) {
            const Array c = array(p_d, "coeffs", p_where);
            if (c.size() != 4 || p_d.has("shape")) {
                fail(p_where, "'coeffs' is 4 numbers (on 1, n, |n|, n^2), without 'shape'.");
                return true;
            }
            for (int i = 0; i < 4; i++) {
                const bool num = is_number(c[i]);
                r_s.c[i] = num ? float(double(c[i])) : NAN;
                if (!num) {
                    fail(p_where, "'coeffs' is 4 numbers.");
                }
            }
            return true;
        }
        if (!p_d.has("shape") && !p_d.has("scale") && !p_d.has("offset")) {
            return false;
        }
        const String name = p_d.has("shape") ? text(p_d, "shape", p_where) : String("plain");
        const double a = number(p_d, "scale", p_where, 1.0);
        const bool ridged = name == "ridged";
        const double o = number(p_d, "offset", p_where, ridged ? 1.0 : 0.0);
        // plain a n + o, billow 2 |a n| - 1 + o, ridged (o - |a n|)^2, square (a n)^2 + o.
        if (name == "plain") {
            r_s = Shape{ { float(o), float(a), 0.0f, 0.0f } };
        } else if (name == "billow") {
            r_s = Shape{ { float(o - 1.0), 0.0f, float(2.0 * std::fabs(a)), 0.0f } };
        } else if (ridged) {
            r_s = Shape{ { float(o * o), 0.0f, float(-2.0 * o * std::fabs(a)), float(a * a) } };
        } else if (name == "square") {
            r_s = Shape{ { float(o), 0.0f, 0.0f, float(a * a) } };
        } else {
            fail(p_where, "'shape' is plain, billow, ridged or square.");
        }
        return true;
    }

    // "stretch_y": a power of two >= 1 (1: none), as log2; octave k's at most k - the insertion shift (its insertion
    // lattice divides exactly, Cascade::stretched).
    int stretch(const Dictionary &p_d, const String &p_where) {
        const double s = number(p_d, "stretch_y", p_where, 1.0);
        const int j = int(std::lround(std::log2(std::max(s, 1.0))));
        if (!(s >= 1.0 && s <= std::ldexp(1.0, MAX_STRETCH) && std::ldexp(1.0, j) == s)) {
            fail(p_where, vformat("'stretch_y' is a power of two in [1, 2^%d].", MAX_STRETCH));
            return 0;
        }
        return j;
    }

    static int octave_stretch(int p_j, int p_k) { return std::max(0, std::min(p_j, p_k - ExactNoise::BAND_SHIFT)); }

    // { "from_m": 100, "to_m": 30000000 | "wavelength_m": 400, "stretch_y": 4 }: the ground's octaves in range read
    // their noise at y / stretch.
    void ground_stretch(const Dictionary &p_d, const String &p_where) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "wavelength_m", "from_m", "to_m", "stretch_y" }) || !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const int j = stretch(p_d, p_where);
        for (int k = lo; k <= hi; k++) {
            for (const std::pair<int, int> &s : table.ground_stretch) {
                if (s.first == k) {
                    fail(p_where, vformat("octave k = %d is stretched twice.", k));
                    return;
                }
            }
            const int s = octave_stretch(j, k);
            if (s > 0) {
                table.ground_stretch.emplace_back(k, s);
            }
        }
    }

    void bands(const Dictionary &p_d, const String &p_where, Field &r_f) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "wavelength_m", "from_m", "to_m", "amp", "persistence", "shape", "scale", "offset", "coeffs", "stretch_y" }) ||
                !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const int j = stretch(p_d, p_where);
        const double amp = number(p_d, "amp", p_where);
        const double persistence = number(p_d, "persistence", p_where, 1.0);
        Shape sh;
        shape(p_d, p_where, sh);
        // A range keeps the ground's octaves (another LOD count keeps another part of it, each octave at the same
        // amplitude); a single wavelength outside them is refused by validate().
        const bool range = lo != hi;
        for (int k = hi; k >= lo; k--) {
            if (range && (k < ExactNoise::BAND_SHIFT || k > ground.k_max)) {
                continue;
            }
            r_f.bands.push_back(ExactNoise::make_octave(r_f.seed, uint32_t(k), k, float(amp * std::pow(persistence, hi - k))));
            r_f.bands.back().stretch = octave_stretch(j, k);
            r_f.shapes.push_back(sh);
        }
    }

    void term(const Dictionary &p_d, const String &p_where, int p_field, Field &r_f) {
        if (!keys(p_d, p_where, { "source", "per_m", "x" })) {
            return;
        }
        Term t;
        t.source = source(text(p_d, "source", p_where), p_where, true, p_field, "the ground or an earlier field");
        // The ground's value is in LOD0 voxels: per metre * metres per voxel.
        t.coef = float(t.source == GROUND ? number(p_d, "per_m", p_where) * ground.voxel_size : number(p_d, "x", p_where));
        r_f.terms.push_back(t);
    }

    void field(const Dictionary &p_d, int64_t p_i) {
        if (!keys(p_d, vformat("field %d", p_i), { "name", "seed", "noise", "bands", "terms" })) {
            return;
        }
        Field f;
        f.name = text(p_d, "name", vformat("field %d", p_i));
        const String where = "field '" + f.name + String("'");
        if (p_d.has("noise")) {
            const String noise = text(p_d, "noise", where);
            f.cell = noise == "cell";
            if (!f.cell && noise != "perlin") {
                fail(where, "'noise' is perlin or cell.");
            }
        }
        f.seed = name_hash(f.name);
        if (p_d.has("seed")) {
            const double seed = number(p_d, "seed", where);
            if (!(seed >= 0.0 && seed < 4294967296.0 && seed == std::floor(seed))) {
                fail(where, "the seed is a whole number in [0, 2^32).");
                return;
            }
            f.seed = uint32_t(seed);
        }
        const Array band_list = array(p_d, "bands", where);
        const String band_where = where + String(" band");
        for (int64_t i = 0; i < band_list.size() && error.is_empty(); i++) {
            bands(dict(band_list[i], band_where), band_where, f);
        }
        // Coarsest first, each band with its shape.
        std::vector<std::pair<Cascade::Band, Shape>> sorted;
        for (size_t i = 0; i < f.bands.size(); i++) {
            sorted.emplace_back(f.bands[i], f.shapes[i]);
        }
        std::stable_sort(sorted.begin(), sorted.end(), [](const auto &p_a, const auto &p_b) { return p_a.first.k > p_b.first.k; });
        for (size_t i = 0; i < sorted.size(); i++) {
            f.bands[i] = sorted[i].first;
            f.shapes[i] = sorted[i].second;
        }
        const Array terms = array(p_d, "terms", where);
        const String term_where = where + String(" term");
        for (int64_t i = 0; i < terms.size() && error.is_empty(); i++) {
            term(dict(terms[i], term_where), term_where, int(p_i), f);
        }
        table.fields.push_back(f);
    }

    // { "field": f, "height_m": H, "curve": [[x, y], ...] (optional) }: the ground adds H * curve(f) (H * f).
    void height(const Dictionary &p_d, const String &p_where) {
        if (!keys(p_d, p_where, { "field", "height_m", "curve", "region" })) {
            return;
        }
        HeightTerm h;
        h.field = source(text(p_d, "field", p_where), p_where, false, int(table.fields.size()), "a field");
        h.region = p_d.has("region") ? region_ref(text(p_d, "region", p_where), p_where) : -1;
        h.coef = float(number(p_d, "height_m", p_where) / ground.voxel_size);
        if (p_d.has("curve")) {
            const Array pts = array(p_d, "curve", p_where);
            std::vector<double> x, y;
            for (int64_t i = 0; i < pts.size(); i++) {
                const Array pt = pts[i].get_type() == Variant::ARRAY ? Array(pts[i]) : Array();
                const bool ok = pt.size() == 2 && is_number(pt[0]) && is_number(pt[1]);
                if (!ok) {
                    fail(p_where, "a curve point is [x, y].");
                    return;
                }
                x.push_back(double(pt[0]));
                y.push_back(double(pt[1]));
                if (x.size() > 1 && !(x.back() > x[x.size() - 2])) {
                    fail(p_where, "the curve's x must increase.");
                    return;
                }
                if (!std::isfinite(x.back()) || !std::isfinite(y.back()) || std::fabs(y.back()) > 1e6) {
                    fail(p_where, "curve points must be finite (|y| <= 1e6).");
                    return;
                }
            }
            if (x.size() < 2) {
                fail(p_where, "a curve has at least 2 points.");
                return;
            }
            h.has_curve = true;
            h.curve = make_curve(x, y);
        }
        table.heights.push_back(h);
    }

    void edges(const Dictionary &p_d, const String &p_where, std::vector<Edge> &r_edges) {
        if (!keys(p_d, p_where, { "field", "min", "max", "fade", "fade_min", "fade_max" })) {
            return;
        }
        const int f = source(text(p_d, "field", p_where), p_where, false, int(table.fields.size()), "a field");
        const double fade = number(p_d, "fade", p_where, 0.0);
        if (!p_d.has("min") && !p_d.has("max")) {
            fail(p_where, "'min', 'max' or both.");
        }
        // A side rises from bound - fade (sign 1, min) or falls to bound + fade (sign -1, max).
        auto side = [&](const char *p_bound, const char *p_fade, double p_sign) {
            if (!p_d.has(p_bound)) {
                return;
            }
            const double w = number(p_d, p_fade, p_where, fade);
            if (!(w > 0.0)) {
                fail(p_where, "the fade must be > 0.");
                return;
            }
            r_edges.push_back(Edge{ f, float(number(p_d, p_bound, p_where) - p_sign * w), float(p_sign / w) });
        };
        side("min", "fade_min", 1.0);
        side("max", "fade_max", -1.0);
    }

    void region(const Dictionary &p_d, int64_t p_i) {
        const String at = vformat("region %d", p_i);
        if (!keys(p_d, at, { "name", "where" })) {
            return;
        }
        Region r;
        r.name = text(p_d, "name", at);
        const String where = "region '" + r.name + String("' where");
        const Array sides = array(p_d, "where", where);
        for (int64_t i = 0; i < sides.size() && error.is_empty(); i++) {
            edges(dict(sides[i], where), where, r.edges);
        }
        table.regions.push_back(r);
    }

    // The region a name refers to (declared under "regions"), -1 and an error otherwise.
    int region_ref(const String &p_name, const String &p_where) {
        const int r = table.region_index(p_name);
        if (r < 0) {
            fail(p_where, "'" + p_name + String("' is not a region."));
        }
        return r;
    }

    void amps(const Dictionary &p_d, const String &p_where, Climate::Biome &r_b) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where,
                    { "field", "wavelength_m", "from_m", "to_m", "x", "shape", "scale", "offset", "coeffs", "multifractal", "warp_m", "warp_fields" }) ||
                !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        Amp proto;
        proto.field = source(text(p_d, "field", p_where), p_where, true, int(table.fields.size()), "the ground or a field");
        proto.mult = float(number(p_d, "x", p_where, 1.0));
        proto.has_shape = shape(p_d, p_where, proto.shape);
        proto.gain = float(number(p_d, "multifractal", p_where, 0.0));
        proto.warp = float(number(p_d, "warp_m", p_where, 0.0) / ground.voxel_size);
        const Array wf = array(p_d, "warp_fields", p_where);
        if (p_d.has("warp_fields") && wf.size() != 3) {
            fail(p_where, "'warp_fields' names 3 fields (x, y, z).");
        }
        for (int64_t i = 0; i < wf.size() && i < 3 && error.is_empty(); i++) {
            proto.warp_fields[i] = source(wf[i].get_type() == Variant::STRING ? String(wf[i]) : String(), p_where, false, int(table.fields.size()), "a field");
        }
        const std::vector<Cascade::Band> *bands = bands_of(table, ground, proto.field);
        if (!error.is_empty() || bands == nullptr) {
            fail(p_where, "no such field.");
            return;
        }
        auto add = [&](int p_k) {
            Amp a = proto;
            a.k = p_k;
            r_b.amps.push_back(a);
        };
        if (lo == hi) {
            add(lo); // validate() reports a missing octave
            return;
        }
        const size_t before = r_b.amps.size();
        for (const Cascade::Band &band : *bands) {
            if (band.k >= lo && band.k <= hi) {
                add(band.k);
            }
        }
        // A range wholly past the ground's octaves names none at this LOD count (like a field's band range).
        if (r_b.amps.size() == before && lo <= ground.k_max && hi >= ExactNoise::BAND_SHIFT) {
            fail(p_where, "the field has no octave in that range.");
        }
    }

    // A "#rrggbb" string at p_key as 0xRRGGBB.
    uint32_t hex_color(const Dictionary &p_d, const char *p_key, const String &p_where) {
        const String c = text(p_d, p_key, p_where);
        bool hex = c.length() == 7 && c[0] == '#';
        for (int i = 1; i < c.length() && hex; i++) {
            hex = is_hex_digit(c[i]);
        }
        if (!hex) {
            fail_key(p_where, p_key, "is \"#rrggbb\".");
            return 0;
        }
        return uint32_t(c.substr(1).hex_to_int());
    }

    // An sRGB "#rrggbb" at p_key as a linear colour (the page shader's albedos), p_default if absent.
    Color linear_color(const Dictionary &p_d, const char *p_key, const String &p_where, const Color &p_default) {
        if (!p_d.has(p_key)) {
            return p_default;
        }
        return srgb_to_linear(hex_color(p_d, p_key, p_where));
    }

    void surface(const Dictionary &p_d, const String &p_where, Surface &r_s) {
        if (!keys(p_d, p_where, { "flat", "steep", "roughness" })) {
            return;
        }
        r_s.flat = linear_color(p_d, "flat", p_where, r_s.flat);
        r_s.steep = linear_color(p_d, "steep", p_where, p_d.has("flat") ? r_s.flat : r_s.steep);
        const double roughness = number(p_d, "roughness", p_where, r_s.roughness);
        if (error.is_empty() && !(roughness >= 0.0 && roughness <= 1.0)) {
            fail(p_where, "roughness is in [0, 1].");
        }
        r_s.roughness = float(roughness);
    }

    void biome(const Dictionary &p_d, int64_t p_i) {
        if (!keys(p_d, vformat("biome %d", p_i), { "name", "region", "where", "amp", "terrace", "surface", "debug_color" })) {
            return;
        }
        Climate::Biome b;
        b.name = text(p_d, "name", vformat("biome %d", p_i));
        const String where = "biome '" + b.name + String("'");
        if (p_d.has("region")) {
            const int r = region_ref(text(p_d, "region", where), where);
            if (r >= 0) {
                b.edges = table.regions[size_t(r)].edges;
            }
        }
        const Array regions = array(p_d, "where", where);
        const String edge_where = where + String(" where");
        for (int64_t i = 0; i < regions.size() && error.is_empty(); i++) {
            edges(dict(regions[i], edge_where), edge_where, b.edges);
        }
        const Array amp = array(p_d, "amp", where);
        const String amp_where = where + String(" amp");
        for (int64_t i = 0; i < amp.size() && error.is_empty(); i++) {
            amps(dict(amp[i], amp_where), amp_where, b);
        }
        if (p_d.has("terrace")) {
            const String tw = where + String(" terrace");
            const Dictionary t = dict(p_d["terrace"], tw);
            if (keys(t, tw, { "step_m", "invert" })) {
                const double step = number(t, "step_m", tw) / ground.voxel_size;
                if (error.is_empty() && !(float(step) > 0.0f)) {
                    fail(tw, "the terrace step must be > 0.");
                }
                if (t.has("invert") && t["invert"].get_type() != Variant::BOOL) {
                    fail_key(tw, "invert", "must be true or false.");
                }
                b.terrace.step = float(step);
                b.terrace.inv_step = float(1.0 / step);
                b.terrace.invert = t.has("invert") && bool(t["invert"]);
            }
        }
        if (p_d.has("surface")) {
            surface(dict(p_d["surface"], where + String(" surface")), where + String(" surface"), b.surface);
        }
        b.debug_color = p_d.has("debug_color") ? hex_color(p_d, "debug_color", where) : name_hash(b.name) & 0xFFFFFFu;
        table.biomes.push_back(b);
    }
};

} // namespace

String parse(const Dictionary &p_data, const Ground &p_ground, Table &r_table) {
    r_table = Table();
    Parser parser(p_ground, r_table);
    parser.run(p_data);
    String err = parser.error;
    if (err.is_empty()) {
        err = validate(r_table, p_ground);
    }
    if (!err.is_empty()) {
        r_table = Table();
    }
    return err;
}

String load(const String &p_path, Dictionary &r_data) {
    const Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
    if (f.is_null()) {
        return "climate: cannot read " + p_path + String(".");
    }
    const Variant data = JSON::parse_string(f->get_as_text());
    if (data.get_type() != Variant::DICTIONARY || Dictionary(data).is_empty()) {
        return "climate: " + p_path + String(" is not a JSON object with a table.");
    }
    r_data = data;
    return String();
}

} // namespace Climate
