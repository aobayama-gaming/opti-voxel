#include "climate.hpp"

#include "ov/files.hpp"
#include "ov/format.hpp"
#include "ov/json.hpp"
#include "ov/value.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

// A climate table's JSON format (parse, load); validate() is in climate_validate.cpp.
namespace Climate {

// The format (JSON; distances in metres, rounded to whole octaves 2^k voxels; "comment" keys are ignored, any
// other unknown key is an error):
//   { "fields": [ field... ], "ground_stretch": [ stretch... ] (optional), "biomes": [ biome..., default biome ] }
//   field:  { "name": "temperature", "seed": 7 (optional, else a hash of the name),
//             "bands": [ { "wavelength_m": 5000, "amp": 1 } | { "from_m": 1600, "to_m": 210000, "amp": 1, "persistence": 1 (optional) } ],
//             "terms": [ { "source": "ground", "per_m": -0.002 } | { "source": "<earlier field>", "x": 0.5 } ] }
//     A range is every octave from from_m to to_m that the ground has (its octaves depend on the LOD count: the
//     table covers the whole spectrum at any count), amp at to_m (the coarsest) times persistence per finer octave.
//     Octave k's seed and lattice are make_octave(field seed, k): editing one band leaves the others' unchanged.
//     A ground term reads the ground's height above the datum (per metre: colder higher up); a field term that
//     field's value.
//   stretch: { "from_m": 100, "to_m": 30000000 (or "wavelength_m"), "stretch_y": 4 }: the ground's octaves in range
//           read their noise squeezed 4 times along the base's up (y on the flat world: their wavelength along it 4
//           times the tangent ones). A power of two; octave k takes at most 2^(k - 3) of it (BaseSurface::stretched: a
//           range's finest octaves are stretched less), and no octave is stretched twice. The ground is a 3D noise
//           whose vertical slope is as large as its horizontal one: past dh/dy = 1 it folds into overhangs; stretched
//           octaves keep it a height field where they dominate.
//   biome:  { "name": "volcanic",
//             "where": [ { "field": "temperature", "min": 0.5, "max": 2 (either or both), "fade": 0.1
//                          (or "fade_min" / "fade_max") } ],
//             "amp": [ { "field": "ground" (optional), "wavelength_m": 4000 | "from_m": 1000, "to_m": 20000, "x": 3 (default 1),
//                        shape (optional, else plain), "multifractal": 2 (gain), "warp_m": 80, "warp_fields": ["wx", "wy", "wz"] } ],
//             "curve": [[h, R], ...] (optional, h increasing, metres),
//             "surface": { "flat": "#2a2420", "steep": "#1a1614" (optional, else flat), "roughness": 1 },
//             "debug_color": "#c04020" }
//   shape:  "shape": "plain" | "billow" | "ridged" | "square", "scale": a, "offset": o (both optional): a n + o,
//           2 |a n| - 1 + o, (o - |a n|)^2 (o default 1), (a n)^2 + o; or "coeffs": [c1, cn, cabs, csq] (Shape). A band
//           inserts its shape without its mean; |n| is smoothed below the lattice's resolution (Cascade::smooth_abs).
//     "multifractal": the octave is weighted by clamp(gain (1 - c), 0, 1), c the multifractal state's complement (the
//     previous multifractal octave's weighted signal: Musgrave's ridged multifractal with gain 2 and the ridged shape,
//     hybrid with 1 and plain + offset). "warp_m": its noise is read displaced by warp_m times the three fields' values
//     (libnoise's turbulence). "curve": the ground height h becomes R(h) where the biome's weight is 1 (R - h through
//     libnoise's cubic, constant past the ends; Biome). The biome is 1 between min and max, fading to 0 over the fade
//     outside. An "amp" entry plays the ground's octaves (a biome's notes: climate fields are never multiplied); a range
//     covers the ones the ground has in it (none, past them at this LOD count). Octave k reads the weights from the fields' octaves coarser than k only (Cascade::Modulation). The last
//     biome is the default: no "where". "surface": the biome's ground colours (sRGB) on gentle and steep slopes;
//     without it, the procedural ground's.
namespace {

class Parser {
public:
    Parser(const Ground &p_ground, Table &r_table) :
            ground(p_ground), table(r_table) {}

    ov::str error;

    void run(const ov::Value &p_data) {
        if (!keys(p_data, "climate", { "fields", "biomes", "ground_stretch" })) {
            return;
        }
        const ov::Value stretch = array(p_data, "ground_stretch", "climate");
        for (int64_t i = 0; i < int64_t(stretch.size()) && error.empty(); i++) {
            ground_stretch(dict(stretch[i], ov::format("ground_stretch %d", i)), ov::format("ground_stretch %d", i));
        }
        const ov::Value fields = array(p_data, "fields", "climate");
        for (int64_t i = 0; i < int64_t(fields.size()) && error.empty(); i++) {
            field(dict(fields[i], ov::format("field %d", i)), i);
        }
        const ov::Value biomes = array(p_data, "biomes", "climate");
        for (int64_t i = 0; i < int64_t(biomes.size()) && error.empty(); i++) {
            biome(dict(biomes[i], ov::format("biome %d", i)), i);
        }
    }

private:
    const Ground &ground;
    Table &table;

    // The first error only: "climate: <where>: <msg>".
    void fail(const ov::str &p_where, const ov::str &p_msg) {
        if (error.empty()) {
            error = "climate: " + p_where + ov::str(": ") + p_msg;
        }
    }

    void fail_key(const ov::str &p_where, const char *p_key, const char *p_msg) {
        fail(p_where, "'" + ov::str(p_key) + ov::str("' ") + ov::str(p_msg));
    }

    bool keys(const ov::Value &p_d, const ov::str &p_where, std::initializer_list<const char *> p_allowed) {
        for (const std::pair<ov::str, ov::Value> &entry : p_d.entries()) {
            const ov::str &key = entry.first;
            if (key != "comment" && std::none_of(p_allowed.begin(), p_allowed.end(), [&](const char *p_a) { return key == p_a; })) {
                fail(p_where, "unknown key '" + key + ov::str("'."));
            }
        }
        return error.empty();
    }

    ov::Value dict(const ov::Value &p_v, const ov::str &p_where) {
        if (p_v.get_type() != ov::Value::OBJECT) {
            fail(p_where, "an object is expected.");
            return ov::Value();
        }
        return p_v;
    }

    ov::Value array(const ov::Value &p_d, const char *p_key, const ov::str &p_where) {
        if (!p_d.has(p_key)) {
            return ov::Value();
        }
        if (p_d[p_key].get_type() != ov::Value::ARRAY) {
            fail_key(p_where, p_key, "must be a list.");
            return ov::Value();
        }
        return p_d[p_key];
    }

    static bool is_number(const ov::Value &p_v) { return p_v.get_type() == ov::Value::INT || p_v.get_type() == ov::Value::FLOAT; }

    // The number at p_key; absent: p_default, or an error if p_default is NaN (required).
    double number(const ov::Value &p_d, const char *p_key, const ov::str &p_where, double p_default = NAN) {
        if (!p_d.has(p_key)) {
            if (std::isnan(p_default)) {
                fail_key(p_where, p_key, "is required.");
            }
            return p_default;
        }
        const ov::Value &v = p_d[p_key];
        if (!is_number(v)) {
            fail_key(p_where, p_key, "must be a number.");
            return 0.0;
        }
        return v.as_double();
    }

    ov::str text(const ov::Value &p_d, const char *p_key, const ov::str &p_where) {
        if (!p_d.has(p_key) || p_d[p_key].get_type() != ov::Value::STRING) {
            fail_key(p_where, p_key, "must be a string.");
            return ov::str();
        }
        return p_d[p_key].as_str();
    }

    // [lo, hi] octaves of a band or amplitude entry: wavelength_m, or from_m .. to_m.
    bool octave_range(const ov::Value &p_d, const ov::str &p_where, int &r_lo, int &r_hi) {
        const bool single = p_d.has("wavelength_m");
        if (single && (p_d.has("from_m") || p_d.has("to_m"))) {
            fail(p_where, "wavelength_m or from_m / to_m, not both.");
            return false;
        }
        const double lo_m = number(p_d, single ? "wavelength_m" : "from_m", p_where);
        const double hi_m = single ? lo_m : number(p_d, "to_m", p_where);
        if (error.empty() && !(lo_m > 0.0 && hi_m >= lo_m && std::isfinite(hi_m))) {
            fail(p_where, "wavelengths must be > 0 and from_m <= to_m.");
        }
        if (!error.empty()) {
            return false;
        }
        r_lo = ExactNoise::lattice_k(lo_m, ground.voxel_size);
        r_hi = ExactNoise::lattice_k(hi_m, ground.voxel_size);
        // make_octave's shifts need k in [0, MAX_K]; validate() then holds bands to the ground's range.
        if (r_lo < 0 || r_hi > ExactNoise::MAX_K) {
            fail(p_where, ov::format("octaves %d .. %d: outside [0, %d].", r_lo, r_hi, ExactNoise::MAX_K));
        }
        return error.empty();
    }

    // The field a name refers to: GROUND for "ground" (if p_ground_ok), else a field index below p_before (p_expect
    // names the allowed ones in the error).
    int source(const ov::str &p_name, const ov::str &p_where, bool p_ground_ok, int p_before, const char *p_expect) {
        if (p_name == "ground" && p_ground_ok) {
            return GROUND;
        }
        const int i = table.field_index(p_name);
        if (i < 0 || i >= p_before) {
            fail(p_where, "'" + p_name + ov::str("' is not ") + ov::str(p_expect) + ov::str("."));
        }
        return i;
    }

    static uint32_t name_hash(const ov::str &p_name) {
        const ov::str &s = p_name;
        uint32_t h = 2166136261u;
        for (size_t i = 0; i < s.size(); i++) {
            h = (h ^ uint8_t(s[i])) * 16777619u;
        }
        return h;
    }

    // A shape (shape, scale, offset or coeffs) of an entry: the plain one without any. Only a multifractal entry's state
    // reads a shape's constant term.
    bool shape(const ov::Value &p_d, const ov::str &p_where, Shape &r_s) {
        const bool state = p_d.has("multifractal");
        if (p_d.has("coeffs")) {
            const ov::Value c = array(p_d, "coeffs", p_where);
            if (c.size() != 4 || p_d.has("shape")) {
                fail(p_where, "'coeffs' is 4 numbers (on 1, n, |n|, n^2), without 'shape'.");
                return true;
            }
            for (int i = 0; i < 4; i++) {
                const bool num = is_number(c[i]);
                r_s.c[i] = num ? float(c[i].as_double()) : NAN;
                if (!num) {
                    fail(p_where, "'coeffs' is 4 numbers.");
                }
            }
            if (r_s.c[0] != 0.0f && !state) {
                fail(p_where, "the constant coefficient only counts in a multifractal (a band is inserted without its mean).");
            }
            return true;
        }
        if (!p_d.has("shape") && !p_d.has("scale") && !p_d.has("offset")) {
            return false;
        }
        const ov::str name = p_d.has("shape") ? text(p_d, "shape", p_where) : ov::str("plain");
        const double a = number(p_d, "scale", p_where, 1.0);
        const bool ridged = name == "ridged";
        const double o = number(p_d, "offset", p_where, ridged ? 1.0 : 0.0);
        // Plain, billow and square: the offset is the constant term, inserted without its mean (Cascade::NOISE_*_MEAN):
        // only a multifractal's state reads it. Ridged's (o - |a n|)^2 also scales |n| by it.
        if (p_d.has("offset") && !ridged && !state) {
            fail(p_where, "'offset' only counts in a multifractal (a plain, billow or square band is inserted without its mean).");
            return true;
        }
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

    // { "from_m": 100, "to_m": 30000000 | "wavelength_m": 400, "stretch_y": 4 }: the ground's octaves in range read
    // their noise at y / stretch. "stretch_y": a power of two >= 1 (1: none), as log2; octave k's at most k - the
    // insertion shift (its insertion lattice divides exactly, BaseSurface::stretched).
    void ground_stretch(const ov::Value &p_d, const ov::str &p_where) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "wavelength_m", "from_m", "to_m", "stretch_y" }) || !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const double s = number(p_d, "stretch_y", p_where, 1.0);
        const int j = int(std::lround(std::log2(std::max(s, 1.0))));
        if (!(s >= 1.0 && s <= std::ldexp(1.0, MAX_STRETCH) && std::ldexp(1.0, j) == s)) {
            fail(p_where, ov::format("'stretch_y' is a power of two in [1, 2^%d].", MAX_STRETCH));
            return;
        }
        for (int k = lo; k <= hi; k++) {
            for (const std::pair<int, int> &st : table.ground_stretch) {
                if (st.first == k) {
                    fail(p_where, ov::format("octave k = %d is stretched twice.", k));
                    return;
                }
            }
            const int sk = std::max(0, std::min(j, k - ExactNoise::BAND_SHIFT));
            if (sk > 0) {
                table.ground_stretch.emplace_back(k, sk);
            }
        }
    }

    void bands(const ov::Value &p_d, const ov::str &p_where, Field &r_f) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "wavelength_m", "from_m", "to_m", "amp", "persistence" }) || !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const double amp = number(p_d, "amp", p_where);
        const double persistence = number(p_d, "persistence", p_where, 1.0);
        // A range keeps the ground's octaves (another LOD count keeps another part of it, each octave at the same
        // amplitude); a single wavelength outside them is refused by validate().
        const bool range = lo != hi;
        for (int k = hi; k >= lo; k--) {
            if (range && (k < ExactNoise::BAND_SHIFT || k > ground.k_max)) {
                continue;
            }
            r_f.bands.push_back(ExactNoise::make_octave(r_f.seed, uint32_t(k), k, float(amp * std::pow(persistence, hi - k))));
        }
    }

    void term(const ov::Value &p_d, const ov::str &p_where, int p_field, Field &r_f) {
        if (!keys(p_d, p_where, { "source", "per_m", "x" })) {
            return;
        }
        Term t;
        t.source = source(text(p_d, "source", p_where), p_where, true, p_field, "the ground or an earlier field");
        // The ground's value is in LOD0 voxels: per metre * metres per voxel.
        t.coef = float(t.source == GROUND ? number(p_d, "per_m", p_where) * ground.voxel_size : number(p_d, "x", p_where));
        r_f.terms.push_back(t);
    }

    void field(const ov::Value &p_d, int64_t p_i) {
        if (!keys(p_d, ov::format("field %d", p_i), { "name", "seed", "bands", "terms" })) {
            return;
        }
        Field f;
        f.name = text(p_d, "name", ov::format("field %d", p_i));
        const ov::str where = "field '" + f.name + ov::str("'");
        f.seed = name_hash(f.name);
        if (p_d.has("seed")) {
            const double seed = number(p_d, "seed", where);
            if (!(seed >= 0.0 && seed < 4294967296.0 && seed == std::floor(seed))) {
                fail(where, "the seed is a whole number in [0, 2^32).");
                return;
            }
            f.seed = uint32_t(seed);
        }
        const ov::Value band_list = array(p_d, "bands", where);
        const ov::str band_where = where + ov::str(" band");
        for (int64_t i = 0; i < int64_t(band_list.size()) && error.empty(); i++) {
            bands(dict(band_list[i], band_where), band_where, f);
        }
        std::stable_sort(f.bands.begin(), f.bands.end(), [](const Cascade::Band &p_a, const Cascade::Band &p_b) { return p_a.k > p_b.k; }); // coarsest first
        const ov::Value terms = array(p_d, "terms", where);
        const ov::str term_where = where + ov::str(" term");
        for (int64_t i = 0; i < int64_t(terms.size()) && error.empty(); i++) {
            term(dict(terms[i], term_where), term_where, int(p_i), f);
        }
        table.fields.push_back(f);
    }

    // A biome's "curve": [[h, R], ...] (metres, h increasing): C = R - h through the points, in LOD0 voxels.
    void curve(const ov::Value &p_d, const ov::str &p_where, Biome &r_b) {
        const ov::Value pts = array(p_d, "curve", p_where);
        std::vector<double> x, y;
        for (int64_t i = 0; i < int64_t(pts.size()); i++) {
            const ov::Value pt = pts[i].get_type() == ov::Value::ARRAY ? pts[i] : ov::Value();
            if (!(pt.size() == 2 && is_number(pt[0]) && is_number(pt[1]))) {
                fail(p_where, "a curve point is [h, R(h)] in metres.");
                return;
            }
            const double h = pt[0].as_double(), r = pt[1].as_double();
            if (!std::isfinite(h) || !std::isfinite(r)) {
                fail(p_where, "curve points must be finite.");
                return;
            }
            x.push_back(h / ground.voxel_size);
            y.push_back((r - h) / ground.voxel_size);
            if (x.size() > 1 && !(x.back() > x[x.size() - 2])) {
                fail(p_where, "the curve's h must increase.");
                return;
            }
        }
        if (x.size() < 2) {
            fail(p_where, "a curve has at least 2 points.");
            return;
        }
        r_b.curve = make_curve(x, y);
    }

    void edges(const ov::Value &p_d, const ov::str &p_where, std::vector<Edge> &r_edges) {
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

    void amps(const ov::Value &p_d, const ov::str &p_where, Climate::Biome &r_b) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where,
                    { "field", "wavelength_m", "from_m", "to_m", "x", "shape", "scale", "offset", "coeffs", "multifractal", "warp_m", "warp_fields" }) ||
                !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        if (p_d.has("field") && text(p_d, "field", p_where) != "ground") {
            fail(p_where, "a biome plays the ground's octaves only ('field': 'ground').");
            return;
        }
        Amp proto;
        proto.mult = float(number(p_d, "x", p_where, 1.0));
        proto.has_shape = shape(p_d, p_where, proto.shape);
        proto.gain = float(number(p_d, "multifractal", p_where, 0.0));
        proto.warp = float(number(p_d, "warp_m", p_where, 0.0) / ground.voxel_size);
        const ov::Value wf = array(p_d, "warp_fields", p_where);
        if (p_d.has("warp_fields") && wf.size() != 3) {
            fail(p_where, "'warp_fields' names 3 fields (x, y, z).");
        }
        for (int64_t i = 0; i < int64_t(wf.size()) && i < 3 && error.empty(); i++) {
            proto.warp_fields[i] = source(wf[i].get_type() == ov::Value::STRING ? wf[i].as_str() : ov::str(), p_where, false, int(table.fields.size()), "a field");
        }
        const std::vector<Cascade::Band> *bands = bands_of(table, ground, GROUND);
        if (!error.empty() || bands == nullptr) {
            fail(p_where, "there is no ground.");
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
            fail(p_where, "the ground has no octave in that range.");
        }
    }

    // A "#rrggbb" string at p_key as 0xRRGGBB.
    uint32_t hex_color(const ov::Value &p_d, const char *p_key, const ov::str &p_where) {
        const ov::str c = text(p_d, p_key, p_where);
        bool hex = c.size() == 7 && c[0] == '#';
        for (size_t i = 1; i < c.size() && hex; i++) {
            hex = std::isxdigit(static_cast<unsigned char>(c[i])) != 0;
        }
        if (!hex) {
            fail_key(p_where, p_key, "is \"#rrggbb\".");
            return 0;
        }
        return uint32_t(std::stoul(c.substr(1), nullptr, 16));
    }

    // An sRGB "#rrggbb" at p_key as a linear colour (the page shader's albedos), p_default if absent.
    ov::Color linear_color(const ov::Value &p_d, const char *p_key, const ov::str &p_where, const ov::Color &p_default) {
        if (!p_d.has(p_key)) {
            return p_default;
        }
        return srgb_to_linear(hex_color(p_d, p_key, p_where));
    }

    void surface(const ov::Value &p_d, const ov::str &p_where, Surface &r_s) {
        if (!keys(p_d, p_where, { "flat", "steep", "roughness" })) {
            return;
        }
        r_s.flat = linear_color(p_d, "flat", p_where, r_s.flat);
        r_s.steep = linear_color(p_d, "steep", p_where, p_d.has("flat") ? r_s.flat : r_s.steep);
        const double roughness = number(p_d, "roughness", p_where, r_s.roughness);
        if (error.empty() && !(roughness >= 0.0 && roughness <= 1.0)) {
            fail(p_where, "roughness is in [0, 1].");
        }
        r_s.roughness = float(roughness);
    }

    void biome(const ov::Value &p_d, int64_t p_i) {
        if (!keys(p_d, ov::format("biome %d", p_i), { "name", "where", "amp", "curve", "surface", "debug_color" })) {
            return;
        }
        Climate::Biome b;
        b.name = text(p_d, "name", ov::format("biome %d", p_i));
        const ov::str where = "biome '" + b.name + ov::str("'");
        const ov::Value sides = array(p_d, "where", where);
        const ov::str edge_where = where + ov::str(" where");
        for (int64_t i = 0; i < int64_t(sides.size()) && error.empty(); i++) {
            edges(dict(sides[i], edge_where), edge_where, b.edges);
        }
        const ov::Value amp = array(p_d, "amp", where);
        const ov::str amp_where = where + ov::str(" amp");
        for (int64_t i = 0; i < int64_t(amp.size()) && error.empty(); i++) {
            amps(dict(amp[i], amp_where), amp_where, b);
        }
        if (p_d.has("curve") && error.empty()) {
            curve(p_d, where + ov::str(" curve"), b);
        }
        if (p_d.has("surface")) {
            surface(dict(p_d["surface"], where + ov::str(" surface")), where + ov::str(" surface"), b.surface);
        }
        b.debug_color = p_d.has("debug_color") ? hex_color(p_d, "debug_color", where) : name_hash(b.name) & 0xFFFFFFu;
        table.biomes.push_back(b);
    }
};

} // namespace

ov::str parse(const ov::Value &p_data, const Ground &p_ground, Table &r_table) {
    r_table = Table();
    Parser parser(p_ground, r_table);
    parser.run(p_data);
    ov::str err = parser.error;
    if (err.empty()) {
        err = validate(r_table, p_ground);
    }
    if (!err.empty()) {
        r_table = Table();
    }
    return err;
}

ov::str load(const ov::str &p_path, ov::Value &r_data) {
    ov::str text;
    if (!ov::read_file(p_path, text)) {
        return "climate: cannot read " + p_path + ov::str(".");
    }
    ov::str json_error;
    const ov::Value data = ov::parse_json(text, json_error);
    if (data.get_type() != ov::Value::OBJECT || data.is_empty()) {
        return "climate: " + p_path + ov::str(" is not a JSON object with a table.");
    }
    r_data = data;
    return ov::str();
}

} // namespace Climate
