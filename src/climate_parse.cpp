#include "climate.hpp"

#include "ov/containers.hpp"
#include "ov/files.hpp"
#include "ov/format.hpp"
#include "ov/json.hpp"
#include "ov/value.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

// A climate table's JSON format: the parser of its layout (fields, biomes' places, looks and contents) and of its graph
// nodes, which the cascade v2 compiler reads (graph_compile.cpp).
namespace Climate {

// The format (JSON; distances in metres, rounded to whole octaves 2^k voxels; "comment" keys are ignored, any
// other unknown key is an error):
//   { "fields": [ field... ], "biomes": [ biome..., default biome ] }
//   field:  { "name": "temperature", "seed": 7 (optional, else a hash of the name),
//             "bands": [ { "wavelength_m": 5000, "amp": 1 } | { "from_m": 1600, "to_m": 210000, "amp": 1, "persistence": 1 (optional) } ],
//             "terms": [ { "source": "<earlier field>", "x": 0.5 } ] }
//     A range is every octave from from_m to to_m that the ground has (its octaves depend on the LOD count: the
//     table covers the whole spectrum at any count), amp at to_m (the coarsest) times persistence per finer octave.
//     Octave k's seed and lattice are make_octave(field seed, k): editing one band leaves the others' unchanged.
//     A term adds x times an earlier field's value (the ground can't be one: it depends on the weights, which depend on
//     the fields).
//   biome:  { "name": "volcanic",
//             "at": { "temperature": [0.5, 0.3] (centre, span), "humidity": { "min": 0.1, "max": 0.5 } },
//             "graph": { "nodes": [ node... ] },
//             "surface": { "flat": "#2a2420", "steep": "#1a1614" (optional, else flat), "roughness": 1 },
//             "clouds": { "cover": 0.6, "type": 0.5 } (optional),
//             "instances": [ { "species": "conifer", "cell_m": 5, "share": 2 } ] (optional), "features": [ { "kind": "lava_lake", "cell_m": 640, "chance": 0.6, "radius_m": [18, 45] } ]
//             (optional),
//             "debug_color": "#c04020" }
//     "at": the biome's place in the fields' space, a centre and a span on each field it names ({min, max}: their
//     middle and half their gap); on each it is whole within 0.71 of the span and gone a span away, overlapping biomes
//     blend by their kernels (climate.hpp, Weights), a field it doesn't name doesn't matter to it. The last biome is the
//     default: no "at", it fills what the others leave. "surface": the biome's ground colours (sRGB) on gentle and
//     steep slopes; without it, the procedural ground's. "clouds": the sky over the biome, its cover and cloud type in
//     [0, 1] (stratus 0, stratocumulus 0.5, cumulus 1; Climate::BiomeClouds); without it, the sky's cover and the
//     weather noise's type. "instances": the world's species (species.hpp) its sites grow, one site per cell of cell_m
//     metres, each by its integer share among the biome's entries of its kind and cell (BiomePick); without the key
//     the world's "instances", [] none. "features": the features standing on it
//     (biome_features.hpp: a kind, its cell, the chance a cell holds one, its radius range); without the key the
//     world's "features".
//   node:   { "id": "n1", "kind": ..., inputs by id, parameters; "pos": [x, y] (the mixer's, ignored) }; a key its
//            kind doesn't list in graph_kinds (the catalogue the biome mixer builds its nodes from) is refused. The kinds:
//     "note":  one octave: "wavelength_m": 400 (its octave k, the nearest power of two of voxels), "seed": 59943, "amp_m":
//              50, "stretch_y": 4 (optional; a power of two: its noise squeezed along the base's up, y on the flat
//              world, its wavelength along it 4 times the tangent ones; octave k takes at most 2^(k - 3) of it,
//              BaseSurface::stretched), "warp": a warp node (optional input: the note read displaced). Its seed and
//              lattice are make_octave(seed, k): one note whatever biome mixes it. A note the LOD count doesn't have
//              (k outside [3, LODs + 4]) plays nothing: a world reads at any LOD count. The ground is a 3D noise whose
//              vertical slope is as large as its horizontal one: past dh/dy = 1 it folds into overhangs; stretched
//              octaves keep it a height field where they dominate.
//     "warp":  "in": any node strictly coarser than every note the warp moves (the gate, docs/cascade-v2.md), "amount_m":
//              80: a note reading it is read displaced by amount_m times (x, y, z), the input read three times with its
//              notes' seeds + 0, + 1 and + 2 (libnoise's turbulence; amp_m 0.5 is half of amount_m). Or, instead of
//              "in", the ports "x", "y", "z": a node per axis, each read once and each strictly coarser.
//     "shape": "in", "shape": "plain" | "billow" | "ridged" | "square", "scale": a, "offset": o (both optional): a n + o,
//              2 |a n| - 1 + o, (o - |a n|)^2 (o default 1), (a n)^2 + o; or "coeffs": [c1, cn, cabs, csq] (Shape). On a
//              note it shapes the noise without its mean (a coarser LOD dropping the note makes no step), its constant
//              term not counting; |n| is smoothed below the lattice's resolution (Cascade::smooth_abs). On any other node
//              it shapes the node's value, its constant term counting.
//     "multifractal": "in", "parent": any node, "low_m": -50, "high_m": 100: in weighted by where its parent's value
//              lies between low_m (weight 0) and high_m (weight 1), clamped; low_m above high_m inverts it.
//     "mix":   "rows": [ { "in": id, "x": 1.2 (volume, default 1) } ]: the rows' sum, each row any node.
//     "curve": "in", "points": [[h, R], ...] (h increasing, metres): in's value h becomes R(h) (R - h through libnoise's
//              cubic, constant past the ends).
//     "up":    the sample's height above the base (metres as LOD0 voxels): a cave is "smin" (k_m) of the mix and
//              up - C, C a carve density (> 0 in the air it adds, its band's tent built from the depth mix - up).
//     "constant" (value_m), "field" (a field's value), "add", "subtract", "multiply", "min", "max" (a, b), "smin", "smax"
//              (a, b, k_m).
//     "output": "height": any node.
//     Any node may read any other, without a loop. Each is computed at the level of the finest note it reads
//     (docs/cascade-v2.md, The rule).

const std::vector<GraphKind> &graph_kinds() {
    static const std::vector<GraphKind> KINDS = {
        { "note", { "warp" }, { { "wavelength_m", "number", true, NAN, "" }, { "seed", "seed", true, NAN, "" }, { "amp_m", "number", true, NAN, "" }, { "stretch_y", "pow2", false, 1.0, "" } } },
        { "warp", { "in", "x", "y", "z" }, { { "amount_m", "number", true, NAN, "" } } },
        { "shape", { "in" }, { { "shape", "shape", false, NAN, "plain|billow|ridged|square" }, { "scale", "number", false, 1.0, "" }, { "offset", "number", false, NAN, "" }, { "coeffs", "coeffs", false, NAN, "" } } },
        { "multifractal", { "in", "parent" }, { { "low_m", "number", true, NAN, "" }, { "high_m", "number", true, NAN, "" } } },
        { "mix", {}, { { "rows", "rows", true, NAN, "" } } },
        { "constant", {}, { { "value_m", "number", true, NAN, "" } } },
        { "field", {}, { { "field", "field", true, NAN, "" } } },
        { "add", { "a", "b" }, {} },
        { "subtract", { "a", "b" }, {} },
        { "multiply", { "a", "b" }, {} },
        { "min", { "a", "b" }, {} },
        { "max", { "a", "b" }, {} },
        { "smin", { "a", "b" }, { { "k_m", "number", true, NAN, "" } } },
        { "smax", { "a", "b" }, { { "k_m", "number", true, NAN, "" } } },
        { "curve", { "in" }, { { "points", "points", true, NAN, "" } } },
        { "up", {}, {} },
        { "output", { "height" }, {} },
    };
    return KINDS;
}

const GraphKind *graph_kind(const ov::str &p_kind) {
    for (const GraphKind &k : graph_kinds()) {
        if (p_kind == k.kind) {
            return &k;
        }
    }
    return nullptr;
}

uint32_t name_hash(const ov::str &p_name) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < p_name.size(); i++) {
        h = (h ^ uint8_t(p_name[i])) * 16777619u;
    }
    return h;
}

namespace {

// Parameter p_name's default in kind p_kind (graph_kinds).
double fallback(const char *p_kind, const char *p_name) {
    const GraphKind *kind = graph_kind(p_kind);
    if (kind == nullptr) {
        return NAN;
    }
    for (const GraphParam &p : kind->params) {
        if (ov::str(p_name) == p.name) {
            return p.fallback;
        }
    }
    return NAN;
}

class Parser {
public:
    Parser(const Ground &p_ground, Table &r_table) :
            ground(p_ground), table(r_table) {}

    ov::str error;

    // The table's fields and biomes, each biome's graph nodes into r_graphs (the compiler reads them).
    void run(const ov::Value &p_data, std::vector<ov::Value> &r_graphs) {
        graphs = &r_graphs;
        if (!keys(p_data, "climate", { "fields", "biomes" })) {
            return;
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

    // A list of instances or features alone (parse_list).
    void run_list(const ov::Value &p_list, const ov::str &p_where, std::vector<InstanceShare> &r_out) { instances(p_list, p_where, r_out); }
    void run_list(const ov::Value &p_list, const ov::str &p_where, std::vector<FeatureShare> &r_out) { features(p_list, p_where, r_out); }
    void shape_of(const ov::Value &p_node, bool p_constant, Shape &r_shape) { shape(p_node, "shape", p_constant, r_shape); }
    // The node readers (parse_note, parse_curve, parse_node_keys, parse_number).
    void note_of(const ov::Value &p_node, const ov::str &p_where, NoteNode &r_n) { note(p_node, p_where, r_n); }
    void curve_of(const ov::Value &p_node, const ov::str &p_where, Curve &r_c) { curve(p_node, p_where, r_c); }
    void keys_of(const ov::Value &p_node, const ov::str &p_where) {
        const GraphKind *k = graph_kind(p_node["kind"].as_str());
        if (k == nullptr) {
            fail(p_where, "'" + p_node["kind"].as_str() + ov::str("' is not a node kind."));
            return;
        }
        node_keys(p_node, p_where, k->kind);
    }
    double number_of(const ov::Value &p_node, const char *p_key, const ov::str &p_where) {
        return number(p_node, p_key, p_where, fallback(p_node["kind"].as_str().c_str(), p_key));
    }

private:
    const Ground &ground;
    Table &table;
    std::vector<ov::Value> *graphs = nullptr;

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
        return keys_in(p_d, p_where, p_allowed);
    }

    // A graph node's keys: its kind's (graph_kinds), its id, kind and pos.
    bool node_keys(const ov::Value &p_d, const ov::str &p_where, const char *p_kind) {
        std::vector<const char *> allowed = { "id", "kind", "pos" };
        const GraphKind *k = graph_kind(p_kind);
        allowed.insert(allowed.end(), k->inputs.begin(), k->inputs.end());
        for (const GraphParam &p : k->params) {
            allowed.push_back(p.name);
        }
        return keys_in(p_d, p_where, allowed);
    }

    template <class C>
    bool keys_in(const ov::Value &p_d, const ov::str &p_where, const C &p_allowed) {
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

    // The field a name refers to: a field index below p_before (p_expect names the allowed ones in the error).
    int source(const ov::str &p_name, const ov::str &p_where, int p_before, const char *p_expect) {
        const int i = table.field_index(p_name);
        if (i < 0 || i >= p_before) {
            fail(p_where, "'" + p_name + ov::str("' is not ") + ov::str(p_expect) + ov::str("."));
        }
        return i;
    }

    // A shape (shape, scale, offset or coeffs) of a node: the plain one without any. Only a shape at the samples
    // (p_constant) keeps its constant term: a band is inserted without its mean.
    bool shape(const ov::Value &p_d, const ov::str &p_where, bool p_constant, Shape &r_s) {
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
            if (r_s.c[0] != 0.0f && !p_constant) {
                fail(p_where, "the constant coefficient doesn't count on a note (a band is inserted without its mean).");
            }
            return true;
        }
        if (!p_d.has("shape") && !p_d.has("scale") && !p_d.has("offset")) {
            return false;
        }
        const ov::str name = p_d.has("shape") ? text(p_d, "shape", p_where) : ov::str("plain");
        const double a = number(p_d, "scale", p_where, fallback("shape", "scale"));
        const bool ridged = name == "ridged";
        const double o = number(p_d, "offset", p_where, ridged ? 1.0 : 0.0);
        // Plain, billow and square: the offset is the constant term, inserted without its mean (Cascade::NOISE_*_MEAN):
        // it doesn't count on a note. Ridged's (o - |a n|)^2 also scales |n| by it.
        if (p_d.has("offset") && !ridged && !p_constant) {
            fail(p_where, "'offset' doesn't count on a note (a plain, billow or square band is inserted without its mean).");
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

    // A whole number in [0, 2^32) at p_key (required).
    uint32_t seed(const ov::Value &p_d, const ov::str &p_where) {
        const double s = number(p_d, "seed", p_where);
        if (error.empty() && !(s >= 0.0 && s < 4294967296.0 && s == std::floor(s))) {
            fail(p_where, "the seed is a whole number in [0, 2^32).");
        }
        return error.empty() ? uint32_t(s) : 0u;
    }

    // A note node: its octave, seed, amplitude and stretch (NoteNode).
    void note(const ov::Value &p_d, const ov::str &p_where, NoteNode &r_n) {
        int hi = 0;
        if (!node_keys(p_d, p_where, "note") || !octave_range(p_d, p_where, r_n.k, hi)) {
            return;
        }
        if (r_n.k < ExactNoise::BAND_SHIFT) {
            fail(p_where, ov::format("octave k = %d is outside the cascade's range: a note's wavelength is at least %s m (%d voxels).", r_n.k, ov::num(std::ldexp(ground.voxel_size, ExactNoise::BAND_SHIFT), 2), 1 << ExactNoise::BAND_SHIFT));
            return;
        }
        r_n.seed = seed(p_d, p_where);
        r_n.amp_m = number(p_d, "amp_m", p_where);
        if (error.empty() && !(r_n.amp_m >= 0.0)) {
            fail(p_where, "a note's amplitude is >= 0.");
        }
        // "stretch_y": a power of two >= 1 (1: none), as log2.
        const double s = number(p_d, "stretch_y", p_where, fallback("note", "stretch_y"));
        r_n.stretch = int(std::lround(std::log2(std::max(s, 1.0))));
        if (!(s >= 1.0 && s <= std::ldexp(1.0, MAX_STRETCH) && std::ldexp(1.0, r_n.stretch) == s)) {
            fail(p_where, ov::format("'stretch_y' is a power of two in [1, 2^%d].", MAX_STRETCH));
        }
    }

    void bands(const ov::Value &p_d, const ov::str &p_where, Field &r_f) {
        int lo = 0, hi = 0;
        if (!keys(p_d, p_where, { "wavelength_m", "from_m", "to_m", "amp", "persistence" }) || !octave_range(p_d, p_where, lo, hi)) {
            return;
        }
        const double amp = number(p_d, "amp", p_where);
        const double persistence = number(p_d, "persistence", p_where, 1.0);
        // A range keeps the octaves the LOD count has (another LOD count keeps another part of it, each octave at the
        // same amplitude); a single wavelength outside them is refused by validate().
        const bool range = lo != hi;
        for (int k = hi; k >= lo; k--) {
            if (range && (k < ExactNoise::BAND_SHIFT || k > ground.k_max)) {
                continue;
            }
            r_f.bands.push_back(ExactNoise::make_octave(r_f.seed, uint32_t(k), k, float(amp * std::pow(persistence, hi - k))));
        }
    }

    void term(const ov::Value &p_d, const ov::str &p_where, int p_field, Field &r_f) {
        if (p_d["source"].as_str() == "ground") {
            fail(p_where, "a 'ground' term is gone (the ground depends on the weights, which depend on the fields): a field's terms read earlier fields only.");
            return;
        }
        if (!keys(p_d, p_where, { "source", "x" })) {
            return;
        }
        const ov::str name = text(p_d, "source", p_where);
        Term t;
        t.source = source(name, p_where, p_field, "an earlier field");
        t.coef = float(number(p_d, "x", p_where));
        r_f.terms.push_back(t);
    }

    void field(const ov::Value &p_d, int64_t p_i) {
        if (!keys(p_d, ov::format("field %d", p_i), { "name", "seed", "bands", "terms" })) {
            return;
        }
        Field f;
        f.name = text(p_d, "name", ov::format("field %d", p_i));
        const ov::str where = "field '" + f.name + ov::str("'");
        f.seed = p_d.has("seed") ? seed(p_d, where) : name_hash(f.name);
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

    // A curve node's "points": [[h, R], ...] (metres, h increasing): C = R - h through the points, in LOD0 voxels.
    void curve(const ov::Value &p_d, const ov::str &p_where, Curve &r_c) {
        const ov::Value pts = array(p_d, "points", p_where);
        std::vector<double> hs, rs;
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
            hs.push_back(h);
            rs.push_back(r);
        }
        const ov::str err = height_curve(hs, rs, ground.voxel_size, r_c);
        if (!err.empty()) {
            fail(p_where, err);
        }
    }

    // A biome's "at": per field, [centre, span] or { "min", "max" }; the axes sorted by field (the weights' op order).
    void axes(const ov::Value &p_d, const ov::str &p_where, std::vector<Axis> &r_at) {
        for (const std::pair<ov::str, ov::Value> &entry : p_d.entries()) {
            if (entry.first == "comment") {
                continue;
            }
            const ov::str where = p_where + ov::str(" ") + entry.first;
            const int f = source(entry.first, where, int(table.fields.size()), "a field");
            const ov::Value &v = entry.second;
            double centre = 0.0, span = 0.0;
            if (v.get_type() == ov::Value::ARRAY && v.size() == 2 && is_number(v[0]) && is_number(v[1])) {
                centre = v[0].as_double();
                span = v[1].as_double();
            } else if (v.get_type() == ov::Value::OBJECT && keys(v, where, { "min", "max" }) && v.has("min") && v.has("max")) {
                const double lo = number(v, "min", where), hi = number(v, "max", where);
                centre = 0.5 * (lo + hi);
                span = 0.5 * (hi - lo);
            } else {
                fail(where, "[centre, span] or { \"min\": a, \"max\": b }.");
            }
            if (!(span > 0.0)) {
                fail(where, "the span must be > 0 (max above min).");
            }
            if (!error.empty()) {
                return;
            }
            r_at.push_back(Axis{ f, float(centre), float(1.0 / span) });
        }
        std::stable_sort(r_at.begin(), r_at.end(), [](const Axis &p_a, const Axis &p_b) { return p_a.field < p_b.field; });
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
        if (p_d.has("where")) {
            fail(ov::format("biome %d", p_i), "'where' (edges) became 'at': a centre and span per field, e.g. \"at\": { \"temperature\": [0.3, 0.2] }.");
            return;
        }
        if (p_d.has("amp") || p_d.has("curve")) {
            fail(ov::format("biome %d", p_i), "'amp' and 'curve' became the biome's 'graph': its notes, a mix of them, a curve (climate_parse.cpp).");
            return;
        }
        if (!keys(p_d, ov::format("biome %d", p_i), { "name", "at", "graph", "surface", "clouds", "debug_color", "instances", "features" })) {
            return;
        }
        Climate::Biome b;
        b.name = text(p_d, "name", ov::format("biome %d", p_i));
        const ov::str where = "biome '" + b.name + ov::str("'");
        if (p_d.has("at")) {
            axes(dict(p_d["at"], where + ov::str(" at")), where + ov::str(" at"), b.at);
        }
        const ov::Value g = p_d.has("graph") ? dict(p_d["graph"], where + ov::str(" graph")) : ov::Value();
        if (p_d.has("graph") && keys(g, where + ov::str(" graph"), { "nodes" })) {
            graphs->push_back(array(g, "nodes", where + ov::str(" graph")));
        } else {
            graphs->push_back(ov::Value::array());
        }
        if (p_d.has("surface")) {
            surface(dict(p_d["surface"], where + ov::str(" surface")), where + ov::str(" surface"), b.surface);
        }
        if (p_d.has("clouds")) {
            clouds(dict(p_d["clouds"], where + ov::str(" clouds")), where + ov::str(" clouds"), b.clouds);
        }
        content(p_d, where, b);
        b.debug_color = p_d.has("debug_color") ? hex_color(p_d, "debug_color", where) : name_hash(b.name) & 0xFFFFFFu;
        table.biomes.push_back(b);
    }

    // A list of instances: [{ "species": name, "cell_m": metres > 0, "share": 1 (a whole number > 0) }] (the world's
    // species: SDFTable checks the names).
    void instances(const ov::Value &p_list, const ov::str &p_where, std::vector<InstanceShare> &r_out) {
        for (int64_t i = 0; i < int64_t(p_list.size()) && error.empty(); i++) {
            const ov::str where = p_where + ov::format(" instance %d", i);
            const ov::Value e = dict(p_list[i], where);
            if (!keys(e, where, { "species", "cell_m", "share" })) {
                return;
            }
            InstanceShare s;
            s.species = text(e, "species", where);
            s.cell_m = number(e, "cell_m", where);
            const double share = number(e, "share", where, 1.0);
            if (error.empty() && !(s.cell_m > 0.0 && s.cell_m <= MAX_CELL_M)) {
                fail(where, ov::format("'cell_m' is a size in metres > 0, at most %s.", ov::num(MAX_CELL_M)));
            }
            if (error.empty() && !(share >= 1.0 && share < 65536.0 && share == std::floor(share))) {
                fail(where, "a share is a whole number in [1, 65535].");
            }
            s.share = uint32_t(share);
            r_out.push_back(s);
        }
    }

    // A list of features: [{ "kind": name, "cell_m": metres > 0, "chance": 1 (in [0, 1]), "radius_m": [lo, hi] (0 < lo
    // <= hi) }] (the feature kinds: SDFTable checks the names).
    void features(const ov::Value &p_list, const ov::str &p_where, std::vector<FeatureShare> &r_out) {
        for (int64_t i = 0; i < int64_t(p_list.size()) && error.empty(); i++) {
            const ov::str where = p_where + ov::format(" feature %d", i);
            const ov::Value e = dict(p_list[i], where);
            if (!keys(e, where, { "kind", "cell_m", "chance", "radius_m" })) {
                return;
            }
            FeatureShare f;
            f.kind = text(e, "kind", where);
            f.cell_m = number(e, "cell_m", where);
            f.chance = number(e, "chance", where, 1.0);
            const ov::Value r = e.get("radius_m", ov::Value());
            if (error.empty() && !(r.is_array() && r.size() == 2 && is_number(r[size_t(0)]) && is_number(r[size_t(1)]))) {
                fail_key(where, "radius_m", "is [lo, hi] in metres.");
                return;
            }
            f.radius_lo_m = r[size_t(0)].as_double();
            f.radius_hi_m = r[size_t(1)].as_double();
            if (error.empty() && !(f.cell_m > 0.0 && f.cell_m <= MAX_CELL_M)) {
                fail(where, ov::format("'cell_m' is a size in metres > 0, at most %s.", ov::num(MAX_CELL_M)));
            }
            if (error.empty() && !(f.chance >= 0.0 && f.chance <= 1.0)) {
                fail_key(where, "chance", "is in [0, 1].");
            }
            if (error.empty() && !(f.radius_lo_m > 0.0 && f.radius_hi_m >= f.radius_lo_m && std::isfinite(f.radius_hi_m))) {
                fail_key(where, "radius_m", "is 0 < lo <= hi.");
            }
            r_out.push_back(f);
        }
    }

    // A biome's "instances" (instances()) and "features" (features()).
    void content(const ov::Value &p_d, const ov::str &p_where, Climate::Biome &r_b) {
        r_b.instances_listed = p_d.has("instances");
        instances(array(p_d, "instances", p_where), p_where, r_b.instances);
        r_b.features_listed = p_d.has("features");
        features(array(p_d, "features", p_where), p_where, r_b.features);
    }
};

// A list of instances or features alone (it reads no octave), r_out unchanged on an error.
template <class T>
ov::str parse_list(const ov::Value &p_list, const ov::str &p_where, const char *p_what, std::vector<T> &r_out) {
    if (!p_list.is_array()) {
        return p_where + ": a list of " + p_what + ".";
    }
    const Ground ground;
    Table table;
    Parser parser(ground, table);
    std::vector<T> out;
    parser.run_list(p_list, p_where, out);
    if (parser.error.empty()) {
        r_out = out;
    }
    return parser.error;
}

// Coefficients, bounds and scales: |x| within [2^-40, 2^40], so their products with field values stay normal.
bool moderate(double p_v) {
    const double a = std::fabs(p_v);
    return std::isfinite(p_v) && a >= std::ldexp(1.0, -40) && a <= std::ldexp(1.0, 40);
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

// A field: an identifier used once, each band's octave in the cascade's range once, its amplitude in the exact range,
// each term on an earlier field with a moderate coefficient.
ov::str validate_field(const Table &p_table, int p_i, const Ground &p_ground) {
    const Field &f = p_table.fields[size_t(p_i)];
    if (!ov::is_valid_ascii_identifier(f.name) || f.name == "ground") {
        return ov::format("climate field %d: the name must be an identifier (letters, digits, _), and 'ground' is reserved.", p_i);
    }
    const ov::str where = "climate field '" + f.name + ov::str("': ");
    if (name_taken_before(p_table.fields, size_t(p_i))) {
        return where + ov::str("the name is used twice.");
    }
    for (const Cascade::Band &b : f.bands) {
        if (b.k < ExactNoise::BAND_SHIFT || b.k > p_ground.k_max) {
            return where + ov::format("octave k = %d is outside the cascade's range [%d, %d].", b.k, ExactNoise::BAND_SHIFT, p_ground.k_max);
        }
        if (std::count_if(f.bands.begin(), f.bands.end(), [&](const Cascade::Band &p_o) { return p_o.k == b.k; }) > 1) {
            return where + ov::format("octave k = %d is listed twice.", b.k);
        }
        if (b.amp != 0.0f && !ExactNoise::in_exact_range(b.amp)) {
            return where + ov::format("octave k = %d: amplitude %s outside [2^-16, 2^40].", b.k, b.amp);
        }
    }
    for (const Term &t : f.terms) {
        if (t.source < 0 || t.source >= p_i) {
            return where + ov::str("a term reads an earlier field only.");
        }
        if (!moderate(t.coef)) {
            return where + ov::format("term coefficient %s outside [2^-40, 2^40].", t.coef);
        }
    }
    return ov::str();
}

// A biome: a name used once, the last one alone without 'at' (the default), each axis on an existing field once, in
// field order, its centre 0 or moderate and its 1 / span moderate and > 0.
ov::str validate_biome(const Table &p_table, size_t p_i) {
    const Biome &b = p_table.biomes[p_i];
    if (b.name.empty()) {
        return ov::format("biome %d: a name is required.", int64_t(p_i));
    }
    const ov::str where = "biome '" + b.name + ov::str("': ");
    if (name_taken_before(p_table.biomes, p_i)) {
        return where + ov::str("the name is used twice.");
    }
    const bool last = p_i + 1 == p_table.biomes.size();
    if (last != b.at.empty()) {
        return where + ov::str(last ? "the last biome is the default: no 'at'." : "only the last biome (the default) has no 'at'.");
    }
    for (size_t i = 0; i < b.at.size(); i++) {
        const Axis &a = b.at[i];
        if (a.field < 0 || a.field >= int(p_table.fields.size())) {
            return where + ov::str("'at' names a field that doesn't exist.");
        }
        if (i > 0 && b.at[i - 1].field >= a.field) {
            return where + ov::format("'at' names '%s' twice, or out of field order.", p_table.fields[size_t(a.field)].name);
        }
        if ((a.centre != 0.0f && !moderate(a.centre)) || !moderate(a.inv_span) || !(a.inv_span > 0.0f)) {
            return where + ov::format("'at' on '%s': centre or span out of range.", p_table.fields[size_t(a.field)].name);
        }
    }
    return ov::str();
}

// The layout's rules (the graphs' are the compiler's: graph_compile.cpp, graph_bounds.cpp): the field and biome counts,
// each field's and each biome's.
ov::str validate(const Table &p_table, const Ground &p_ground) {
    if (p_table.fields.size() > size_t(MAX_FIELDS)) {
        return ov::format("climate: %d fields, at most %d.", int64_t(p_table.fields.size()), MAX_FIELDS);
    }
    if (p_table.biomes.empty() || p_table.biomes.size() > size_t(MAX_BIOMES)) {
        return ov::format("climate: %d biomes, 1 to %d.", int64_t(p_table.biomes.size()), MAX_BIOMES);
    }
    for (int i = 0; i < int(p_table.fields.size()); i++) {
        const ov::str err = validate_field(p_table, i, p_ground);
        if (!err.empty()) {
            return err;
        }
    }
    for (size_t i = 0; i < p_table.biomes.size(); i++) {
        const ov::str err = validate_biome(p_table, i);
        if (!err.empty()) {
            return err;
        }
    }
    return ov::str();
}

} // namespace

ov::str height_curve(const std::vector<double> &p_h_m, const std::vector<double> &p_r_m, double p_voxel, Curve &r_c) {
    std::vector<double> x, y;
    for (size_t i = 0; i < p_h_m.size(); i++) {
        x.push_back(p_h_m[i] / p_voxel);
        y.push_back((p_r_m[i] - p_h_m[i]) / p_voxel);
        if (x.size() > 1 && !(x.back() > x[x.size() - 2])) {
            return "the curve's h must increase.";
        }
    }
    if (x.size() < 2) {
        return "a curve has at least 2 points.";
    }
    r_c = make_curve(x, y);
    return ov::str();
}

ov::str parse_shape(const ov::Value &p_node, bool p_constant, Shape &r_shape) {
    const Ground ground;
    Table table;
    Parser parser(ground, table);
    Shape s;
    parser.shape_of(p_node, p_constant, s);
    if (parser.error.empty()) {
        r_shape = s;
    }
    return parser.error;
}

ov::str parse_instances(const ov::Value &p_list, const ov::str &p_where, std::vector<InstanceShare> &r_out) {
    return parse_list(p_list, p_where, "instances", r_out);
}

ov::str parse_features(const ov::Value &p_list, const ov::str &p_where, std::vector<FeatureShare> &r_out) {
    return parse_list(p_list, p_where, "features", r_out);
}

ov::str parse_layout(const ov::Value &p_data, const Ground &p_ground, Table &r_table, std::vector<ov::Value> &r_graphs) {
    r_table = Table();
    r_graphs.clear();
    Parser parser(p_ground, r_table);
    parser.run(p_data, r_graphs);
    const ov::str err = parser.error.empty() ? validate(r_table, p_ground) : parser.error;
    if (!err.empty()) {
        r_table = Table();
        r_graphs.clear();
    }
    return err;
}

ov::str parse_node_keys(const ov::Value &p_node, const ov::str &p_where) {
    const Ground ground;
    Table table;
    Parser parser(ground, table);
    parser.keys_of(p_node, p_where);
    return parser.error;
}

ov::str parse_note(const ov::Value &p_node, const Ground &p_ground, const ov::str &p_where, NoteNode &r_note) {
    Table table;
    Parser parser(p_ground, table);
    parser.note_of(p_node, p_where, r_note);
    return parser.error;
}

ov::str parse_number(const ov::Value &p_node, const char *p_key, const ov::str &p_where, double &r_value) {
    const Ground ground;
    Table table;
    Parser parser(ground, table);
    r_value = parser.number_of(p_node, p_key, p_where);
    return parser.error;
}

ov::str parse_curve(const ov::Value &p_node, const Ground &p_ground, const ov::str &p_where, Curve &r_curve) {
    Table table;
    Parser parser(p_ground, table);
    parser.curve_of(p_node, p_where, r_curve);
    return parser.error;
}

} // namespace Climate