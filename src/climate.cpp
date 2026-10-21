#include "climate.hpp"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/variant/array.hpp>

#include <algorithm>
#include <cmath>

namespace Climate {

using Cascade::flush_value;

namespace {

Float4 flushed(const Float4 &p_a) {
    return Float4{ flush_value(p_a.x), flush_value(p_a.y), flush_value(p_a.z), flush_value(p_a.w) };
}

// Dual product, then the flush (see the header's exactness note).
Float4 dmul(const Float4 &p_a, const Float4 &p_b) {
    return flushed(Float4{ p_a.w * p_b.x + p_b.w * p_a.x, p_a.w * p_b.y + p_b.w * p_a.y, p_a.w * p_b.z + p_b.w * p_a.z, p_a.w * p_b.w });
}

Float4 dsub(const Float4 &p_a, const Float4 &p_b) {
    return flushed(Float4{ p_a.x - p_b.x, p_a.y - p_b.y, p_a.z - p_b.z, p_a.w - p_b.w });
}

// Cascade channels (VALUE, DX, DY, DZ) as a Float4 (gradient, value).
Float4 to_float4(const float p_ch[Cascade::CHANNELS]) {
    return Float4{ p_ch[Cascade::DX], p_ch[Cascade::DY], p_ch[Cascade::DZ], p_ch[Cascade::VALUE] };
}

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
    const double amp = double(p_amp.mult) * double(band->amp);
    const bool ok = p_amp.mult == 0.0f || (ExactNoise::in_exact_range(float(amp)) && (p_amp.field != GROUND || amp <= p_ground.amp_cap));
    return ok ? String() : vformat("octave k = %d: the modulated amplitude %s is out of range.", p_amp.k, amp);
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
    for (const Edge &e : b.edges) {
        if (e.field < 0 || e.field >= int(p_table.fields.size())) {
            return where + String("an edge reads a field that doesn't exist.");
        }
        if ((e.anchor != 0.0f && !moderate(e.anchor)) || !moderate(e.scale)) {
            return where + vformat("edge on '%s': bound or fade out of range.", p_table.fields[size_t(e.field)].name);
        }
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

Float4 edge_weight(const Edge &p_edge, const Float4 &p_x) {
    const float raw = flush_value((p_x.w - p_edge.anchor) * p_edge.scale);
    if (!(raw > 0.0f)) {
        return Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    if (raw >= 1.0f) {
        return Float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    const Float4 s = flushed(Float4{ p_x.x * p_edge.scale, p_x.y * p_edge.scale, p_x.z * p_edge.scale, raw });
    // s^2 (3 - 2 s); the x2 scalings are exact.
    const Float4 q{ s.x * -2.0f, s.y * -2.0f, s.z * -2.0f, 3.0f - s.w * 2.0f };
    return dmul(dmul(s, s), q);
}

} // namespace

int Table::field_index(const String &p_name) const {
    for (size_t i = 0; i < fields.size(); i++) {
        if (fields[i].name == p_name) {
            return int(i);
        }
    }
    return -1;
}

String validate(const Table &p_table, const Ground &p_ground) {
    if (p_table.fields.size() > size_t(MAX_FIELDS)) {
        return vformat("climate: %d fields, at most %d.", int64_t(p_table.fields.size()), MAX_FIELDS);
    }
    if (p_table.biomes.empty() || p_table.biomes.size() > size_t(MAX_BIOMES)) {
        return vformat("climate: %d biomes, 1 to %d.", int64_t(p_table.biomes.size()), MAX_BIOMES);
    }
    for (int i = 0; i < int(p_table.fields.size()); i++) {
        const String err = validate_field(p_table, i, p_ground);
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
    return String();
}

// The format (JSON; distances in metres, rounded to whole octaves 2^k voxels; "comment" keys are ignored, any
// other unknown key is an error):
//   { "fields": [ field... ], "biomes": [ biome..., default biome ] }
//   field:  { "name": "temperature", "seed": 7 (optional, else a hash of the name),
//             "bands": [ { "wavelength_m": 5000, "amp": 1 }
//                      | { "from_m": 1600, "to_m": 210000, "amp": 1, "persistence": 1 (optional) } ],
//             "terms": [ { "source": "ground", "per_m": -0.002 } | { "source": "<earlier field>", "x": 0.5 } ] }
//     A range is every octave from from_m to to_m, amp at to_m (the coarsest) times persistence per finer octave.
//     Octave k's seed and lattice are make_octave(field seed, k): editing one band leaves the others' unchanged.
//     A ground term reads the ground's height above the datum (per metre); a field term that field's value.
//   biome:  { "name": "volcanic",
//             "where": [ { "field": "temperature", "min": 0.5, "max": 2 (either or both), "fade": 0.1
//                          (or "fade_min" / "fade_max") } ],
//             "amp": [ { "field": "ground" | "<field>", "wavelength_m": 4000 | "from_m": 1000, "to_m": 20000, "x": 3 } ],
//             "surface": { "flat": "#2a2420", "steep": "#1a1614" (optional, else flat), "roughness": 1 },
//             "debug_color": "#c04020" }
//     The region is 1 between min and max, fading to 0 over the fade outside. A range of "amp" covers the
//     octaves the field has in it. The last biome is the default: no "where". "surface": the biome's ground colours
//     (sRGB) on gentle and steep slopes; without it, the procedural ground's.
namespace {

class Parser {
public:
    Parser(const Ground &p_ground, Table &r_table) :
            ground(p_ground), table(r_table) {}

    String error;

    void run(const Dictionary &p_data) {
        if (!keys(p_data, "climate", { "fields", "biomes" })) {
            return;
        }
        const Array fields = array(p_data, "fields", "climate");
        for (int64_t i = 0; i < fields.size() && error.is_empty(); i++) {
            field(dict(fields[i], vformat("field %d", i)), i);
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

    // The number at p_key; absent: p_default, or an error if p_default is NaN (required).
    double number(const Dictionary &p_d, const char *p_key, const String &p_where, double p_default = NAN) {
        if (!p_d.has(p_key)) {
            if (std::isnan(p_default)) {
                fail_key(p_where, p_key, "is required.");
            }
            return p_default;
        }
        const Variant v = p_d[p_key];
        if (v.get_type() != Variant::INT && v.get_type() != Variant::FLOAT) {
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

    void bands(const Dictionary &p_d, const String &p_where, Field &r_f) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "wavelength_m", "from_m", "to_m", "amp", "persistence" }) || !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const double amp = number(p_d, "amp", p_where);
        const double persistence = number(p_d, "persistence", p_where, 1.0);
        for (int k = hi; k >= lo; k--) {
            r_f.bands.push_back(ExactNoise::make_octave(r_f.seed, uint32_t(k), k, float(amp * std::pow(persistence, hi - k))));
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
        if (!keys(p_d, vformat("field %d", p_i), { "name", "seed", "bands", "terms" })) {
            return;
        }
        Field f;
        f.name = text(p_d, "name", vformat("field %d", p_i));
        const String where = "field '" + f.name + String("'");
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
        std::stable_sort(f.bands.begin(), f.bands.end(), [](const Cascade::Band &p_a, const Cascade::Band &p_b) { return p_a.k > p_b.k; });
        const Array terms = array(p_d, "terms", where);
        const String term_where = where + String(" term");
        for (int64_t i = 0; i < terms.size() && error.is_empty(); i++) {
            term(dict(terms[i], term_where), term_where, int(p_i), f);
        }
        table.fields.push_back(f);
    }

    void edges(const Dictionary &p_d, const String &p_where, Climate::Biome &r_b) {
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
            r_b.edges.push_back(Edge{ f, float(number(p_d, p_bound, p_where) - p_sign * w), float(p_sign / w) });
        };
        side("min", "fade_min", 1.0);
        side("max", "fade_max", -1.0);
    }

    void amps(const Dictionary &p_d, const String &p_where, Climate::Biome &r_b) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "field", "wavelength_m", "from_m", "to_m", "x" }) || !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const int f = source(text(p_d, "field", p_where), p_where, true, int(table.fields.size()), "the ground or a field");
        const float mult = float(number(p_d, "x", p_where));
        const std::vector<Cascade::Band> *bands = bands_of(table, ground, f);
        if (!error.is_empty() || bands == nullptr) {
            fail(p_where, "no such field.");
            return;
        }
        if (lo == hi) {
            r_b.amps.push_back(Amp{ f, lo, mult }); // validate() reports a missing octave
            return;
        }
        const size_t before = r_b.amps.size();
        for (const Cascade::Band &band : *bands) {
            if (band.k >= lo && band.k <= hi) {
                r_b.amps.push_back(Amp{ f, band.k, mult });
            }
        }
        if (r_b.amps.size() == before) {
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
        if (!keys(p_d, vformat("biome %d", p_i), { "name", "where", "amp", "surface", "debug_color" })) {
            return;
        }
        Climate::Biome b;
        b.name = text(p_d, "name", vformat("biome %d", p_i));
        const String where = "biome '" + b.name + String("'");
        const Array regions = array(p_d, "where", where);
        const String edge_where = where + String(" where");
        for (int64_t i = 0; i < regions.size() && error.is_empty(); i++) {
            edges(dict(regions[i], edge_where), edge_where, b);
        }
        const Array amp = array(p_d, "amp", where);
        const String amp_where = where + String(" amp");
        for (int64_t i = 0; i < amp.size() && error.is_empty(); i++) {
            amps(dict(amp[i], amp_where), amp_where, b);
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

void apply_terms(const Table &p_table, const float p_slots[Cascade::SLOTS], Float4 r_fields[MAX_FIELDS]) {
    for (int i = 0; i < MAX_FIELDS; i++) {
        r_fields[i] = Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    const Float4 ground = to_float4(p_slots);
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        Float4 f = to_float4(p_slots + Cascade::field_slot(int(i)));
        for (const Term &t : p_table.fields[i].terms) {
            const Float4 s = t.source == GROUND ? ground : r_fields[t.source];
            f = flushed(Float4{ f.x + s.x * t.coef, f.y + s.y * t.coef, f.z + s.z * t.coef, f.w + s.w * t.coef });
        }
        r_fields[i] = f;
    }
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

float biome_amp(const Biome &p_biome, int p_field, const Cascade::Band &p_band) {
    for (const Amp &e : p_biome.amps) {
        if (e.field == p_field && e.k == p_band.k) {
            return float(double(e.mult) * double(p_band.amp));
        }
    }
    return p_band.amp;
}

float max_amp(const Table &p_table, int p_field, const Cascade::Band &p_band) {
    float m = p_table.biomes.empty() ? p_band.amp : 0.0f;
    for (const Climate::Biome &b : p_table.biomes) {
        m = MAX(m, biome_amp(b, p_field, p_band));
    }
    return m;
}

void fields_at(const Table &p_table, const Cascade::Spec &p_combined, const Vector3i64 &p_p, int p_m, Float4 r_fields[MAX_FIELDS]) {
    float slots[Cascade::SLOTS];
    Cascade::point_slots(p_combined, p_p, p_m, slots, true);
    apply_terms(p_table, slots, r_fields);
}

void weights(const Table &p_table, const Float4 p_fields[MAX_FIELDS], Float4 r_w[MAX_BIOMES]) {
    for (int b = 0; b < MAX_BIOMES; b++) {
        r_w[b] = Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    Float4 rest{ 0.0f, 0.0f, 0.0f, 1.0f };
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b + 1 < n; b++) {
        Float4 r{ 0.0f, 0.0f, 0.0f, 1.0f };
        for (const Edge &e : p_table.biomes[b].edges) {
            r = dmul(r, edge_weight(e, p_fields[e.field]));
        }
        r_w[b] = dmul(rest, r);
        rest = dsub(rest, r_w[b]);
    }
    if (n > 0) {
        r_w[n - 1] = rest;
    }
}

int pick(const Table &p_table, const Float4 p_w[MAX_BIOMES], const Vector3i64 &p_p, int p_m) {
    int best[MAX_BIOMES];
    int count = 0;
    for (int b = 0; b < int(p_table.biomes.size()); b++) {
        if (count > 0 && ExactNoise::float_bits(p_w[b].w) == ExactNoise::float_bits(p_w[best[0]].w)) {
            best[count++] = b;
        } else if (count == 0 || p_w[b].w > p_w[best[0]].w) {
            best[0] = b;
            count = 1;
        }
    }
    if (count == 0) {
        return -1;
    }
    // The absolute voxel: the same point seen from two LODs breaks a tie the same way.
    const int64_t step = int64_t(1) << p_m;
    const Vector3i64 voxel(p_p.x * step, p_p.y * step, p_p.z * step);
    return best[ExactNoise::hash_cell(voxel, PICK_SEED) % uint32_t(count)];
}

int pick_slots(const Table &p_table, const float p_slots[Cascade::SLOTS], const Vector3i64 &p_p, int p_m) {
    Float4 f[MAX_FIELDS];
    apply_terms(p_table, p_slots, f);
    Float4 w[MAX_BIOMES];
    weights(p_table, f, w);
    return pick(p_table, w, p_p, p_m);
}

Color srgb_to_linear(uint32_t p_rgb) {
    return Color::hex((p_rgb << 8) | 0xFFu).srgb_to_linear();
}

Material::Params biome_material(const Biome &p_biome) {
    Material::Params m;
    m.albedo = p_biome.surface.flat;
    m.steep = p_biome.surface.steep;
    m.roughness = p_biome.surface.roughness;
    m.pattern = Material::PATTERN_GROUND;
    m.debug = srgb_to_linear(p_biome.debug_color);
    return m;
}

namespace {

// The climate's Cascade::Modulation (combined_spec): per modulated band (group, level), each biome's amplitude.
class Modulator : public Cascade::Modulation {
public:
    Modulator(const Table &p_table, const Cascade::Spec &p_spec) :
            table(p_table) {
        for (int g = 0; g < p_spec.group_count(); g++) {
            for (const Cascade::Band &band : p_spec.group_bands(g)) {
                Entry e{ g, std::vector<float>(table.biomes.size(), band.amp) };
                bool any = false;
                for (size_t b = 0; b < table.biomes.size(); b++) {
                    e.amp[b] = biome_amp(table.biomes[b], g - 1, band); // group g is field g - 1, the ground GROUND
                    for (const Amp &a : table.biomes[b].amps) {
                        any = any || (a.field == g - 1 && a.k == band.k);
                    }
                }
                const int m = p_spec.insert_level(band);
                if (any && m >= 0 && m <= Cascade::MAX_LEVEL) {
                    entries[m].push_back(e);
                    mask[m] |= 1u << g;
                    // Only the ground reaches the mesh: the fields are carried for the ground's modulated levels.
                    if (g == 0) {
                        range.floor = MIN(range.floor, m);
                        range.ceiling = MAX(range.ceiling, m);
                    }
                }
            }
        }
    }

    bool any() const {
        return std::any_of(std::begin(mask), std::end(mask), [](uint32_t p_m) { return p_m != 0u; });
    }

    uint32_t modulated(int p_m) const override { return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? mask[p_m] : 0u; }
    Cascade::FieldRange field_range() const override { return range; }

    void amplitudes(int p_m, const float p_pre[Cascade::SLOTS], Float4 r_amp[Cascade::MAX_GROUPS]) const override {
        Float4 f[MAX_FIELDS];
        apply_terms(table, p_pre, f);
        Float4 w[MAX_BIOMES];
        weights(table, f, w);
        for (const Entry &e : entries[p_m]) {
            Float4 a{ 0.0f, 0.0f, 0.0f, 0.0f };
            for (size_t b = 0; b < e.amp.size(); b++) {
                if (e.amp[b] != 0.0f) {
                    const float k = e.amp[b];
                    a = Float4{ a.x + w[b].x * k, a.y + w[b].y * k, a.z + w[b].z * k, a.w + w[b].w * k };
                }
            }
            r_amp[e.group] = flushed(a);
        }
    }

    String glsl(const Cascade::Spec &p_spec) const override;

private:
    struct Entry {
        int group = 0;
        std::vector<float> amp; // per biome
    };
    Table table;
    std::vector<Entry> entries[Cascade::MAX_LEVEL + 1];
    uint32_t mask[Cascade::MAX_LEVEL + 1] = {};
    Cascade::FieldRange range;
};

String Modulator::glsl(const Cascade::Spec &p_spec) const {
    using ExactNoise::glsl_float;
    String code = glsl_helpers();
    code += "\n// Climate::Modulator::amplitudes: the field terms, the biome weights, then each modulated band's amplitude.\n";
    code += vformat("uint cascade_amplitudes(int m, Ch pre, out vec4 amp[%d]) {\n", Cascade::MAX_GROUPS);
    code += vformat("    for (int g = 0; g < %d; g++) {\n        amp[g] = vec4(0.0);\n    }\n    uint mask = 0u;\n    switch (m) {\n", Cascade::MAX_GROUPS);
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (mask[m] != 0u) {
            code += vformat("    case %d: mask = %du; break;\n", m, int64_t(mask[m]));
        }
    }
    code += "    default: break;\n    }\n    if (mask == 0u) {\n        return 0u;\n    }\n";
    code += glsl_weights(table, p_spec);
    code += "    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (entries[m].empty()) {
            continue;
        }
        code += vformat("    case %d: {\n", m);
        for (const Entry &e : entries[m]) {
            code += "        { precise vec4 a = vec4(0.0);";
            for (size_t b = 0; b < e.amp.size(); b++) {
                if (e.amp[b] != 0.0f) {
                    code += vformat(" a = a + w%d * %s;", int64_t(b), glsl_float(e.amp[b]));
                }
            }
            code += vformat(" amp[%d] = cl_flush(a); }\n", e.group);
        }
        code += "        break;\n    }\n";
    }
    code += "    default: break;\n    }\n    return mask;\n}\n";
    return code;
}

} // namespace

Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground) {
    Cascade::Spec spec = p_ground;
    spec.fields.clear();
    spec.modulation.reset();
    for (const Field &f : p_table.fields) {
        spec.fields.push_back(f.bands);
    }
    std::shared_ptr<Modulator> mod = std::make_shared<Modulator>(p_table, spec);
    if (mod->any()) {
        ERR_FAIL_COND_V_MSG(spec.set != Cascade::ChannelSet::LINEAR4, spec, "Climate: amplitude multipliers need a LINEAR4 ground; the climate is carried unmodulated.");
        spec.modulation = mod;
    }
    return spec;
}

} // namespace Climate
