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
//   { "fields": [ field... ], "regions": [ region... ] (optional), "heights": [ height... ] (optional),
//     "ground_stretch": [ stretch... ] (optional), "biomes": [ biome..., default biome ] }
//   field:  { "name": "temperature", "seed": 7 (optional, else a hash of the name), "noise": "perlin" | "cell" (optional),
//             "bands": [ { "wavelength_m": 5000, "amp": 1, shape (optional), "stretch_y": 1 (optional) }
//                      | { "from_m": 1600, "to_m": 210000, "amp": 1, "persistence": 1 (optional), shape (optional) } ],
//             "terms": [ { "source": "ground", "per_m": -0.002 } | { "source": "<earlier field>", "x": 0.5 } ] }
//   stretch: { "from_m": 100, "to_m": 30000000 (or "wavelength_m"), "stretch_y": 4 }: the ground's octaves in range
//           read their noise squeezed 4 times along the base's up (y on the flat world: their wavelength along it 4
//           times the tangent ones); a field band takes
//           "stretch_y" too. A power of two; octave k takes at most 2^(k - 3) of it (BaseSurface::stretched: a range's finest
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
//             "clouds": { "cover": 0.6, "type": 0.5 } (optional),
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
//     (sRGB) on gentle and steep slopes; without it, the procedural ground's. "clouds": the sky over the biome, its
//     cover and cloud type in [0, 1] (stratus 0, stratocumulus 0.5, cumulus 1; Climate::BiomeClouds); without it, the
//     sky's cover and the weather noise's type.
namespace {

class Parser {
public:
    Parser(const Ground &p_ground, Table &r_table) :
            ground(p_ground), table(r_table) {}

    ov::str error;

    void run(const ov::Value &p_data) {
        if (!keys(p_data, "climate", { "fields", "regions", "biomes", "heights", "ground_stretch" })) {
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
        const ov::Value regions = array(p_data, "regions", "climate");
        for (int64_t i = 0; i < int64_t(regions.size()) && error.empty(); i++) {
            region(dict(regions[i], ov::format("region %d", i)), i);
        }
        const ov::Value heights = array(p_data, "heights", "climate");
        for (int64_t i = 0; i < int64_t(heights.size()) && error.empty(); i++) {
            height(dict(heights[i], ov::format("height %d", i)), ov::format("height %d", i));
        }
        const ov::Value biomes = array(p_data, "biomes", "climate");
        for (int64_t i = 0; i < int64_t(biomes.size()) && error.empty(); i++) {
            biome(dict(biomes[i], ov::format("biome %d", i)), i);
        }
        table.gated = gated_regions(table);
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

    // A shape (shape, scale, offset or coeffs) of an entry: the plain one without any. p_state: the entry weights its
    // octave by the multifractal state, the only reader of a shape's constant term.
    bool shape(const ov::Value &p_d, const ov::str &p_where, bool p_state, Shape &r_s) {
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
            if (r_s.c[0] != 0.0f && !p_state) {
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
        if (p_d.has("offset") && !ridged && !p_state) {
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

    // "stretch_y": a power of two >= 1 (1: none), as log2; octave k's at most k - the insertion shift (its insertion
    // lattice divides exactly, BaseSurface::stretched).
    int stretch(const ov::Value &p_d, const ov::str &p_where) {
        const double s = number(p_d, "stretch_y", p_where, 1.0);
        const int j = int(std::lround(std::log2(std::max(s, 1.0))));
        if (!(s >= 1.0 && s <= std::ldexp(1.0, MAX_STRETCH) && std::ldexp(1.0, j) == s)) {
            fail(p_where, ov::format("'stretch_y' is a power of two in [1, 2^%d].", MAX_STRETCH));
            return 0;
        }
        return j;
    }

    static int octave_stretch(int p_j, int p_k) { return std::max(0, std::min(p_j, p_k - ExactNoise::BAND_SHIFT)); }

    // { "from_m": 100, "to_m": 30000000 | "wavelength_m": 400, "stretch_y": 4 }: the ground's octaves in range read
    // their noise at y / stretch.
    void ground_stretch(const ov::Value &p_d, const ov::str &p_where) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "wavelength_m", "from_m", "to_m", "stretch_y" }) || !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const int j = stretch(p_d, p_where);
        for (int k = lo; k <= hi; k++) {
            for (const std::pair<int, int> &s : table.ground_stretch) {
                if (s.first == k) {
                    fail(p_where, ov::format("octave k = %d is stretched twice.", k));
                    return;
                }
            }
            const int s = octave_stretch(j, k);
            if (s > 0) {
                table.ground_stretch.emplace_back(k, s);
            }
        }
    }

    void bands(const ov::Value &p_d, const ov::str &p_where, Field &r_f) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "wavelength_m", "from_m", "to_m", "amp", "persistence", "shape", "scale", "offset", "coeffs", "stretch_y" }) ||
                !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const int j = stretch(p_d, p_where);
        const double amp = number(p_d, "amp", p_where);
        const double persistence = number(p_d, "persistence", p_where, 1.0);
        Shape sh;
        shape(p_d, p_where, false, sh);
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
        if (!keys(p_d, ov::format("field %d", p_i), { "name", "seed", "noise", "bands", "terms" })) {
            return;
        }
        Field f;
        f.name = text(p_d, "name", ov::format("field %d", p_i));
        const ov::str where = "field '" + f.name + ov::str("'");
        if (p_d.has("noise")) {
            const ov::str noise = text(p_d, "noise", where);
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
        const ov::Value band_list = array(p_d, "bands", where);
        const ov::str band_where = where + ov::str(" band");
        for (int64_t i = 0; i < int64_t(band_list.size()) && error.empty(); i++) {
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
        const ov::Value terms = array(p_d, "terms", where);
        const ov::str term_where = where + ov::str(" term");
        for (int64_t i = 0; i < int64_t(terms.size()) && error.empty(); i++) {
            term(dict(terms[i], term_where), term_where, int(p_i), f);
        }
        table.fields.push_back(f);
    }

    // { "field": f, "height_m": H, "curve": [[x, y], ...] (optional), "region": r (optional) }: the ground adds
    // H * curve(f) (H * f), times region r.
    void height(const ov::Value &p_d, const ov::str &p_where) {
        if (!keys(p_d, p_where, { "field", "height_m", "curve", "region" })) {
            return;
        }
        HeightTerm h;
        h.field = source(text(p_d, "field", p_where), p_where, false, int(table.fields.size()), "a field");
        h.region = p_d.has("region") ? region_ref(text(p_d, "region", p_where), p_where) : -1;
        h.coef = float(number(p_d, "height_m", p_where) / ground.voxel_size);
        if (p_d.has("curve")) {
            const ov::Value pts = array(p_d, "curve", p_where);
            std::vector<double> x, y;
            for (int64_t i = 0; i < int64_t(pts.size()); i++) {
                const ov::Value pt = pts[i].get_type() == ov::Value::ARRAY ? pts[i] : ov::Value();
                const bool ok = pt.size() == 2 && is_number(pt[0]) && is_number(pt[1]);
                if (!ok) {
                    fail(p_where, "a curve point is [x, y].");
                    return;
                }
                x.push_back(pt[0].as_double());
                y.push_back(pt[1].as_double());
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

    void region(const ov::Value &p_d, int64_t p_i) {
        const ov::str at = ov::format("region %d", p_i);
        if (!keys(p_d, at, { "name", "where" })) {
            return;
        }
        Region r;
        r.name = text(p_d, "name", at);
        const ov::str where = "region '" + r.name + ov::str("' where");
        const ov::Value sides = array(p_d, "where", where);
        for (int64_t i = 0; i < int64_t(sides.size()) && error.empty(); i++) {
            edges(dict(sides[i], where), where, r.edges);
        }
        table.regions.push_back(r);
    }

    // The region a name refers to (declared under "regions"), -1 and an error otherwise.
    int region_ref(const ov::str &p_name, const ov::str &p_where) {
        const int r = table.region_index(p_name);
        if (r < 0) {
            fail(p_where, "'" + p_name + ov::str("' is not a region."));
        }
        return r;
    }

    void amps(const ov::Value &p_d, const ov::str &p_where, Climate::Biome &r_b) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where,
                    { "field", "wavelength_m", "from_m", "to_m", "x", "shape", "scale", "offset", "coeffs", "multifractal", "warp_m", "warp_fields" }) ||
                !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        Amp proto;
        proto.field = source(text(p_d, "field", p_where), p_where, true, int(table.fields.size()), "the ground or a field");
        proto.mult = float(number(p_d, "x", p_where, 1.0));
        proto.has_shape = shape(p_d, p_where, p_d.has("multifractal"), proto.shape);
        proto.gain = float(number(p_d, "multifractal", p_where, 0.0));
        proto.warp = float(number(p_d, "warp_m", p_where, 0.0) / ground.voxel_size);
        const ov::Value wf = array(p_d, "warp_fields", p_where);
        if (p_d.has("warp_fields") && wf.size() != 3) {
            fail(p_where, "'warp_fields' names 3 fields (x, y, z).");
        }
        for (int64_t i = 0; i < int64_t(wf.size()) && i < 3 && error.empty(); i++) {
            proto.warp_fields[i] = source(wf[i].get_type() == ov::Value::STRING ? wf[i].as_str() : ov::str(), p_where, false, int(table.fields.size()), "a field");
        }
        const std::vector<Cascade::Band> *bands = bands_of(table, ground, proto.field);
        if (!error.empty() || bands == nullptr) {
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

    void clouds(const ov::Value &p_d, const ov::str &p_where, Climate::BiomeClouds &r_c) {
        if (!keys(p_d, p_where, { "cover", "type" })) {
            return;
        }
        const double cover = number(p_d, "cover", p_where);
        const double type = number(p_d, "type", p_where);
        if (error.empty() && !(cover >= 0.0 && cover <= 1.0 && type >= 0.0 && type <= 1.0)) {
            fail(p_where, "cover and type are in [0, 1].");
        }
        r_c.set = true;
        r_c.cover = float(cover);
        r_c.type = float(type);
    }

    void biome(const ov::Value &p_d, int64_t p_i) {
        if (!keys(p_d, ov::format("biome %d", p_i), { "name", "region", "where", "amp", "terrace", "surface", "clouds", "debug_color" })) {
            return;
        }
        Climate::Biome b;
        b.name = text(p_d, "name", ov::format("biome %d", p_i));
        const ov::str where = "biome '" + b.name + ov::str("'");
        if (p_d.has("region")) {
            const int r = region_ref(text(p_d, "region", where), where);
            if (r >= 0) {
                b.edges = table.regions[size_t(r)].edges;
            }
        }
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
        if (p_d.has("terrace")) {
            const ov::str tw = where + ov::str(" terrace");
            const ov::Value t = dict(p_d["terrace"], tw);
            if (keys(t, tw, { "step_m", "invert" })) {
                const double step = number(t, "step_m", tw) / ground.voxel_size;
                if (error.empty() && !(float(step) > 0.0f)) {
                    fail(tw, "the terrace step must be > 0.");
                }
                if (t.has("invert") && t["invert"].get_type() != ov::Value::BOOL) {
                    fail_key(tw, "invert", "must be true or false.");
                }
                b.terrace.step = float(step);
                b.terrace.inv_step = float(1.0 / step);
                b.terrace.invert = t.has("invert") && t["invert"].as_bool();
            }
        }
        if (p_d.has("surface")) {
            surface(dict(p_d["surface"], where + ov::str(" surface")), where + ov::str(" surface"), b.surface);
        }
        if (p_d.has("clouds")) {
            clouds(dict(p_d["clouds"], where + ov::str(" clouds")), where + ov::str(" clouds"), b.clouds);
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
