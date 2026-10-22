#include "climate.hpp"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/variant/array.hpp>

#include <algorithm>
#include <cmath>

// A climate table's JSON format (parse, load); validate() is in climate_validate.cpp.
namespace Climate {

// The format (JSON; distances in metres, rounded to whole octaves 2^k voxels; "comment" keys are ignored, any
// other unknown key is an error):
//   { "fields": [ field... ], "heights": [ height... ] (optional), "ground_stretch": [ stretch... ] (optional),
//     "biomes": [ biome..., default biome ] }
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
//   height: { "field": "continent", "height_m": 2048, "curve": [[x, y], ...] (optional, x increasing) }: the ground
//           adds height_m * curve(field) at each sample (libnoise's Curve), height_m * field without a curve.
//     A cell noise field ("noise": "cell") is Worley F1 (ExactNoise::cell3), plain bands only: read through a height
//     term (dunes, cells).
//     A range is every octave from from_m to to_m that the ground has (its octaves depend on the LOD count: the
//     table covers the whole spectrum at any count), amp at to_m (the coarsest) times persistence per finer octave.
//     Octave k's seed and lattice are make_octave(field seed, k): editing one band leaves the others' unchanged.
//     A ground term reads the ground's height above the datum (per metre); a field term that field's value.
//   biome:  { "name": "volcanic",
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
        if (!keys(p_data, "climate", { "fields", "biomes", "heights", "ground_stretch" })) {
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
        if (!keys(p_d, p_where, { "field", "height_m", "curve" })) {
            return;
        }
        HeightTerm h;
        h.field = source(text(p_d, "field", p_where), p_where, false, int(table.fields.size()), "a field");
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
        if (!keys(p_d, vformat("biome %d", p_i), { "name", "where", "amp", "terrace", "surface", "debug_color" })) {
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
