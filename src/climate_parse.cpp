#include "biome_compiler.hpp"
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

// A climate table's JSON format (parse, load); validate() is in climate_validate.cpp.
namespace Climate {

// The format (JSON; distances in metres, rounded to whole octaves 2^k voxels; "comment" keys are ignored, any
// other unknown key is an error):
//   { "fields": [ field... ], "biomes": [ biome..., default biome ] }
//   field:  { "name": "temperature", "seed": 7 (optional, else a hash of the name),
//             "bands": [ { "wavelength_m": 5000, "amp": 1 } | { "from_m": 1600, "to_m": 210000, "amp": 1, "persistence": 1 (optional) } ],
//             "terms": [ { "source": "ground", "per_m": -0.002 } | { "source": "<earlier field>", "x": 0.5 } ] }
//     A range is every octave from from_m to to_m that the ground has (its octaves depend on the LOD count: the
//     table covers the whole spectrum at any count), amp at to_m (the coarsest) times persistence per finer octave.
//     Octave k's seed and lattice are make_octave(field seed, k): editing one band leaves the others' unchanged.
//     A ground term reads the ground's height above the datum (per metre: colder higher up); a field term that
//     field's value.
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
//     blend by their kernels (Climate::weights), a field it doesn't name doesn't matter to it. The last biome is the
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
//     "warp":  "in": notes (a note or a mix of them, each coarser than every note the warp moves), "amount_m": 80: a
//              note reading it is read displaced by amount_m times (x, y, z), the input's notes read with their seed,
//              seed + 1 and seed + 2 (libnoise's turbulence; amp_m 0.5 is half of amount_m).
//     "shape": "in", "shape": "plain" | "billow" | "ridged" | "square", "scale": a, "offset": o (both optional): a n + o,
//              2 |a n| - 1 + o, (o - |a n|)^2 (o default 1), (a n)^2 + o; or "coeffs": [c1, cn, cabs, csq] (Shape). A
//              band inserts its shape without its mean; |n| is smoothed below the lattice's resolution (Cascade::smooth_abs).
//              Past a note's chain (a second shape, a mix of several notes, a node past the notes) it shapes their sum at
//              the samples, its constant term counting (SampleOp::SHAPE).
//     "multifractal": "in", "parent": a note strictly coarser than in's (no warp), "low_m": -50, "high_m": 100: in's note
//              weighted by where its parent's height (amp_m times its noise, stretched as the note says) lies between
//              low_m (weight 0) and high_m (weight 1), clamped; low_m above high_m inverts it. The parent plays nothing
//              itself (a stretched note's parent stretched alike keeps its weight's vertical slope down).
//     "mix":   "rows": [ { "in": id, "x": 1.2 (volume, default 1) } ]: the rows' sum, each row any node.
//     "curve": "in", "points": [[h, R], ...] (h increasing, metres): the ground height h becomes R(h) where the biome's
//              weight is 1 (R - h through libnoise's cubic, constant past the ends; Biome).
//     "up":    the sample's height above the base (metres as LOD0 voxels): a cave is "smin" (k_m) of the mix and
//              up - C, C a carve density (> 0 in the air it adds, its band's tent built from the depth mix - up).
//     "output": "height": any node.
//     Any node may read any other (no loop): the relaxer (expr) decides where each plays, per octave in the cascade or
//     at the samples, notes read past a node at the samples becoming a sub-mix field (docs/climate.md, "Parse, compile
//     and validate").
//     The biome compiler (biome_compiler.hpp) keeps each note once whatever biome mixes it. Octave k reads the weights
//     from the fields' octaves coarser than k only (Cascade::Modulation).
const std::vector<GraphKind> &graph_kinds() {
    static const std::vector<GraphKind> KINDS = {
        { "note", { "warp" }, { { "wavelength_m", "number", true, NAN, "" }, { "seed", "seed", true, NAN, "" }, { "amp_m", "number", true, NAN, "" }, { "stretch_y", "pow2", false, 1.0, "" } } },
        { "warp", { "in" }, { { "amount_m", "number", true, NAN, "" } } },
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

    void run(const ov::Value &p_data) {
        if (!keys(p_data, "climate", { "fields", "biomes" })) {
            return;
        }
        const ov::Value fields = array(p_data, "fields", "climate");
        for (int64_t i = 0; i < int64_t(fields.size()) && error.empty(); i++) {
            field(dict(fields[i], ov::format("field %d", i)), i);
        }
        named = int(table.fields.size());
        const ov::Value biomes = array(p_data, "biomes", "climate");
        std::vector<BiomeGraph> graphs;
        for (int64_t i = 0; i < int64_t(biomes.size()) && error.empty(); i++) {
            graphs.emplace_back();
            biome(dict(biomes[i], ov::format("biome %d", i)), i, graphs.back());
        }
        if (raw != nullptr) {
            return; // parse_layout: the graphs are the caller's
        }
        // A generic table keeps the warp fields of its capacity (the rest without a band), so a warp deleted compiles
        // nothing; never past the fields its sub-mixes' capacity leaves.
        while (!ground.specialize && int(table.fields.size()) - named < ground.caps.warps && int(table.fields.size()) + ground.caps.channels < MAX_FIELDS) {
            Field f;
            f.name = ov::format("~warp%d", int64_t(table.fields.size()));
            f.warp = true;
            table.fields.push_back(f);
        }
        const ov::str err = error.empty() ? compile(graphs, ground, table) : ov::str();
        if (!err.empty()) {
            error = "climate: " + err;
        }
    }

    // A list of instances or features alone (parse_list).
    void run_list(const ov::Value &p_list, const ov::str &p_where, std::vector<InstanceShare> &r_out) { instances(p_list, p_where, r_out); }
    void run_list(const ov::Value &p_list, const ov::str &p_where, std::vector<FeatureShare> &r_out) { features(p_list, p_where, r_out); }
    void shape_of(const ov::Value &p_node, bool p_constant, Shape &r_shape) { shape(p_node, "shape", p_constant, r_shape); }
    // parse_layout's: each biome's graph nodes go to r_graphs instead of the relaxer; the node readers.
    void set_raw(std::vector<ov::Value> &r_graphs) { raw = &r_graphs; }
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
    std::vector<ov::Value> *raw = nullptr;
    int named = 0; // the fields the format names (the climate's); the warps' follow (warp_field)

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
            const int f = source(entry.first, where, false, named, "a field");
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

    // A graph's node of id p_id, null (an error) if there is none.
    const ov::Value *node(const std::vector<ov::Value> &p_nodes, const ov::str &p_id, const ov::str &p_where) {
        for (const ov::Value &n : p_nodes) {
            if (n["id"].as_str() == p_id) {
                return &n;
            }
        }
        fail(p_where, "no node '" + p_id + ov::str("'."));
        return nullptr;
    }

    // The node an input of p_d names (its key p_key).
    const ov::Value *input(const std::vector<ov::Value> &p_nodes, const ov::Value &p_d, const char *p_key, const ov::str &p_where) {
        const ov::str id = text(p_d, p_key, p_where);
        return error.empty() ? node(p_nodes, id, p_where) : nullptr;
    }

    static bool is_kind(const ov::Value *p_n, const char *p_kind) { return p_n != nullptr && (*p_n)["kind"].as_str() == p_kind; }

    // The relaxer (docs/climate.md, "Biome graphs"): a graph's node is either notes the cascade inserts (rows, each a
    // note's per-octave chain at a volume: their sum) or a register of the biome's sample program. A note's chain plays
    // per octave while it reads note -> warp -> shape -> multifractal (each optional, at most once; a warp anywhere in it:
    // it only moves where the note is read); a node past that, or any node past a mix, reads their sum at the samples
    // (the notes lifted to a register: the ground's or a sub-mix's, graph()). Mixes are linear: a mix of mixes is one.
    struct Value {
        std::vector<GraphRow> rows;
        int reg = -1;
    };

    // How a graph is read back from its output: each node's value once (by id), the note sums lifted to registers (their rows and the instruction graph() makes the ground's or a
    // sub-mix's read), the depth (a loop).
    struct Reading {
        ov::HashMap<ov::str, Value, ov::DefaultHash> memo;
        std::vector<std::pair<std::vector<GraphRow>, int>> lifts;
        int depth = 0;
    };

    // Whether p_v is one note's chain a per-octave node may extend (at volume 1, no multifractal yet).
    static bool per_octave(const Value &p_v) {
        return p_v.reg < 0 && p_v.rows.size() == 1 && p_v.rows[0].x == 1.0f && p_v.rows[0].multifractal.field < 0;
    }

    int emit(Biome &r_b, SampleOp p_op, int p_a = -1, int p_b = -1, float p_value = 0.0f, int p_index = 0) {
        r_b.program.code.push_back(SampleInstr{ p_op, p_a, p_b, p_value, p_index });
        return int(r_b.program.code.size()) - 1;
    }

    // r_v as a register: its own, or its notes lifted (a placeholder graph() makes the ground's or a sub-mix's read).
    int lift(Value &r_v, Biome &r_b, Reading &r_read) {
        if (r_v.reg < 0 && error.empty()) {
            r_read.lifts.emplace_back(r_v.rows, int(r_b.program.code.size()));
            r_v.reg = emit(r_b, SampleOp::FIELD);
        }
        return r_v.reg;
    }

    // Node p_n's register (its notes lifted once, whatever reads it).
    int reg(const std::vector<ov::Value> &p_nodes, const ov::Value *p_n, const ov::str &p_where, Biome &r_b, Reading &r_read) {
        Value v = expr(p_nodes, p_n, p_where, r_b, r_read);
        if (v.reg >= 0 || !error.empty()) {
            return v.reg;
        }
        const int r = lift(v, r_b, r_read);
        if (Value *m = r_read.memo.getptr((*p_n)["id"].as_str())) {
            m->reg = r;
        }
        return r;
    }

    // Node p_n's value.
    Value expr(const std::vector<ov::Value> &p_nodes, const ov::Value *p_n, const ov::str &p_where, Biome &r_b, Reading &r_read) {
        if (!error.empty() || p_n == nullptr) {
            return Value();
        }
        const ov::str id = (*p_n)["id"].as_str();
        if (const Value *m = r_read.memo.getptr(id)) {
            return *m;
        }
        if (++r_read.depth > 4096) {
            fail(p_where + ov::str(" node '") + id + ov::str("'"), "the graph loops.");
            return Value();
        }
        const Value v = node_value(p_nodes, p_n, p_where, r_b, r_read);
        r_read.memo.insert(id, v);
        return v;
    }

    // A shape or a multifractal (p_kind): the note chain it extends per octave, or a shape at the samples.
    Value chain_value(const ov::str &kind, const std::vector<ov::Value> &p_nodes, const ov::Value *p_n, const ov::str &p_where, const ov::str &where, Biome &r_b,
            Reading &r_read) {
        Value out;
        if (kind == "shape") {
            out = expr(p_nodes, input(p_nodes, *p_n, "in", where), p_where, r_b, r_read);
            if (per_octave(out) && !out.rows[0].has_shape) {
                out.rows[0].has_shape = shape(*p_n, where, false, out.rows[0].shape);
                return out;
            }
            Shape s;
            shape(*p_n, where, true, s);
            const int a = reg(p_nodes, input(p_nodes, *p_n, "in", where), p_where, r_b, r_read);
            r_b.program.shapes.push_back(s);
            return Value{ {}, error.empty() ? emit(r_b, SampleOp::SHAPE, a, -1, 0.0f, int(r_b.program.shapes.size()) - 1) : -1 };
        }
        if (kind == "multifractal") {
            out = expr(p_nodes, input(p_nodes, *p_n, "in", where), p_where, r_b, r_read);
            if (error.empty() && !per_octave(out)) {
                fail(where, "a multifractal weighs one note: its input is a note (through a warp and a shape).");
            }
            if (error.empty()) {
                multifractal(*p_n, input(p_nodes, *p_n, "parent", where), where, out.rows[0]);
            }
            return out;
        }
        return out;
    }

    // Row r_row weighted by multifractal node p_mf: clamp(mul f + add, 0, 1), f its parent's noise at amplitude 1
    // (parent_field), 0 at the parent's height low_m, 1 at high_m (its amplitude under the note cap, as it would play). A
    // parent the LOD count doesn't have, or a silent one, is at height 0: the weight a constant, folded into the row's
    // volume.
    void multifractal(const ov::Value &p_mf, const ov::Value *p_parent, const ov::str &p_where, GraphRow &r_row) {
        const double low = number(p_mf, "low_m", p_where), high = number(p_mf, "high_m", p_where);
        if (!is_kind(p_parent, "note")) {
            fail(p_where, "a multifractal's parent is a note.");
            return;
        }
        NoteNode parent;
        note(*p_parent, p_where, parent);
        if (!error.empty()) {
            return;
        }
        if (p_parent->has("warp")) {
            fail(p_where, "a multifractal's parent is read without a warp.");
        } else if (parent.k <= r_row.note.k) {
            fail(p_where, ov::format("a multifractal's parent is coarser than its note: %s m isn't coarser than %s m.", ov::num(std::ldexp(ground.voxel_size, parent.k), 2), ov::num(std::ldexp(ground.voxel_size, r_row.note.k), 2)));
        } else if (!(std::isfinite(low) && std::isfinite(high) && low != high)) {
            fail(p_where, "a multifractal's low_m and high_m differ.");
        }
        if (!error.empty()) {
            return;
        }
        const double add = -low / (high - low);
        const double amp_m = double(note_amp(parent, ground)) * ground.voxel_size;
        if (!note_plays(parent, ground) || amp_m == 0.0) {
            r_row.x = r_row.x * float(ov::clamp(add, 0.0, 1.0));
            return;
        }
        r_row.multifractal = Multifractal{ parent_field(parent), float(amp_m / (high - low)), float(add) };
    }

    // The field of multifractal parent p_note: its noise at amplitude 1, stretched as a ground octave. A note of octave
    // k reads it before level k's insertions, whole: it is coarser.
    int parent_field(const NoteNode &p_note) {
        const int stretch = note_stretch(p_note, p_note.k);
        const ov::str name = ov::format("~parent%d_%d_%d", int64_t(p_note.k), int64_t(p_note.seed), int64_t(stretch));
        const int known = table.field_index(name);
        if (known >= 0) {
            return known;
        }
        Field f;
        f.name = name;
        f.warp = true;
        f.bands.push_back(ExactNoise::make_octave(p_note.seed, uint32_t(p_note.k), p_note.k, 1.0f));
        f.bands.back().stretch = stretch;
        table.fields.push_back(f);
        return int(table.fields.size()) - 1;
    }

    // Note row r_row read through warp node p_w: one term per note of its input (every one coarser than the note: what
    // the planner checks), its octave's fields (warp_field) at amount_m times the note's amplitude and volume. The fields
    // hold the noise alone, so a value edit never changes them: the table's structure follows the graph's.
    void warp(const std::vector<ov::Value> &p_nodes, const ov::Value *p_w, const ov::str &p_where, const ov::str &p_note, Biome &r_b, Reading &r_read, GraphRow &r_row) {
        if (!is_kind(p_w, "warp")) {
            fail(p_note, "'warp' is a warp node.");
            return;
        }
        const ov::str where = p_where + ov::str(" node '") + (*p_w)["id"].as_str() + ov::str("'");
        if (!node_keys(*p_w, where, "warp")) {
            return;
        }
        const Value in = expr(p_nodes, input(p_nodes, *p_w, "in", where), p_where, r_b, r_read);
        if (error.empty() && in.reg >= 0) {
            fail(where, "a warp's input is notes or a mix of them, not a node past them.");
        }
        const double amount = number(*p_w, "amount_m", where) / ground.voxel_size;
        for (size_t i = 0; i < in.rows.size() && error.empty(); i++) {
            const GraphRow &row = in.rows[i];
            if (row.has_shape || row.multifractal.field >= 0 || !row.warps.empty() || row.note.stretch != 0) {
                fail(where, "a warp's input is notes or a mix of them (no shape, multifractal, warp or stretch).");
            } else if (row.note.k <= r_row.note.k) {
                fail(where, ov::format("a warp is coarser than the note it moves: its note of %s m isn't coarser than %s m.", ov::num(std::ldexp(ground.voxel_size, row.note.k), 2), ov::num(std::ldexp(ground.voxel_size, r_row.note.k), 2)));
            } else if (note_plays(row.note, ground)) {
                const float scale = float(amount * double(row.x) * row.note.amp_m);
                if (scale != 0.0f) {
                    r_row.warps.push_back(WarpTerm{ warp_field(row.note), scale });
                }
            }
        }
    }

    // The first of the three fields (x, y, z) of warp note p_note's octave: its noise at amplitude 1, read with its
    // seed, seed + 1 and seed + 2. A note of octave k reads them before level k's insertions, whole: they are coarser.
    int warp_field(const NoteNode &p_note) {
        const int known = table.field_index(ov::format("~warp%d_%dx", int64_t(p_note.k), int64_t(p_note.seed)));
        if (known >= 0) {
            return known;
        }
        const int field = int(table.fields.size());
        for (int a = 0; a < 3; a++) {
            Field f;
            f.name = ov::format("~warp%d_%d%c", int64_t(p_note.k), int64_t(p_note.seed), "xyz"[a]);
            f.warp = true;
            f.bands.push_back(ExactNoise::make_octave(p_note.seed + uint32_t(a), uint32_t(p_note.k), p_note.k, 1.0f));
            table.fields.push_back(f);
        }
        return field;
    }

    // A mix: its rows' notes at their volumes, the rows past the notes summed at the samples.
    Value mix_value(const std::vector<ov::Value> &p_nodes, const ov::Value *p_n, const ov::str &p_where, const ov::str &where, Biome &r_b, Reading &r_read) {
        Value out;
        const ov::Value rows = array(*p_n, "rows", where);
        std::vector<std::pair<int, float>> regs;
        for (int64_t i = 0; i < int64_t(rows.size()) && error.empty(); i++) {
            const ov::str row_where = where + ov::format(" row %d", i);
            const ov::Value r = dict(rows[i], row_where);
            if (!keys(r, row_where, { "in", "x" })) {
                return out;
            }
            const float x = float(number(r, "x", row_where, 1.0));
            const Value v = expr(p_nodes, input(p_nodes, r, "in", row_where), p_where, r_b, r_read);
            if (v.reg >= 0) {
                regs.emplace_back(v.reg, x);
                continue;
            }
            for (GraphRow row : v.rows) {
                row.x = row.x * x;
                out.rows.push_back(row);
            }
        }
        if (regs.empty() || !error.empty()) {
            return out;
        }
        int acc = out.rows.empty() ? -1 : lift(out, r_b, r_read);
        for (const auto &[r, x] : regs) { // at its volume whatever its value: a volume edit keeps the program's structure
            const int t = emit(r_b, SampleOp::MUL, emit(r_b, SampleOp::CONST, -1, -1, x), r);
            acc = acc < 0 ? t : emit(r_b, SampleOp::ADD, acc, t);
        }
        return Value{ {}, acc };
    }

    // expr's kinds: "note" (its row, read through its warp), "shape" / "multifractal" (the chain, or a shape at the samples), "mix"
    // (rows, or a sum at the samples), and the samples' ops on registers: "constant" value_m, "field" (a climate
    // field's value), "add" / "subtract" / "multiply" / "min" / "max" (a, b), "smin" / "smax" (a, b, k_m), "curve" (in,
    // points), "up".
    Value node_value(const std::vector<ov::Value> &p_nodes, const ov::Value *p_n, const ov::str &p_where, Biome &r_b, Reading &r_read) {
        const ov::str kind = (*p_n)["kind"].as_str();
        const ov::str where = p_where + ov::str(" node '") + (*p_n)["id"].as_str() + ov::str("'");
        Value out;
        if (graph_kind(kind) != nullptr && kind != "output" && !node_keys(*p_n, where, kind.c_str())) {
            return out;
        }
        if (kind == "note") {
            out.rows.emplace_back();
            note(*p_n, where, out.rows.back().note);
            if (p_n->has("warp") && error.empty()) {
                warp(p_nodes, input(p_nodes, *p_n, "warp", where), p_where, where, r_b, r_read, out.rows.back());
            }
            return out;
        }
        if (kind == "warp") {
            fail(where, "a warp is a note's input ('warp'): it moves where that note is read.");
            return out;
        }
        if (kind == "shape" || kind == "multifractal") {
            return chain_value(kind, p_nodes, p_n, p_where, where, r_b, r_read);
        }
        if (kind == "mix") {
            return mix_value(p_nodes, p_n, p_where, where, r_b, r_read);
        }
        if (kind == "constant") {
            return Value{ {}, emit(r_b, SampleOp::CONST, -1, -1, float(number(*p_n, "value_m", where) / ground.voxel_size)) };
        }
        if (kind == "up") {
            return Value{ {}, emit(r_b, SampleOp::UP, -1, -1) };
        }
        if (kind == "field") {
            return Value{ {}, emit(r_b, SampleOp::FIELD, -1, -1, 0.0f, source(text(*p_n, "field", where), where, false, named, "a field")) };
        }
        static const std::pair<const char *, SampleOp> BINARY[] = { { "add", SampleOp::ADD }, { "subtract", SampleOp::SUB }, { "multiply", SampleOp::MUL }, { "min", SampleOp::MIN },
            { "max", SampleOp::MAX }, { "smin", SampleOp::SMIN }, { "smax", SampleOp::SMAX } };
        for (const auto &[name, op] : BINARY) {
            if (kind != name) {
                continue;
            }
            const bool smooth = op == SampleOp::SMIN || op == SampleOp::SMAX;
            const float k = smooth ? float(number(*p_n, "k_m", where) / ground.voxel_size) : 0.0f;
            const int a = reg(p_nodes, input(p_nodes, *p_n, "a", where), p_where, r_b, r_read);
            const int b = reg(p_nodes, input(p_nodes, *p_n, "b", where), p_where, r_b, r_read);
            return Value{ {}, error.empty() ? emit(r_b, op, a, b, k) : -1 };
        }
        if (kind == "curve") {
            const int a = reg(p_nodes, input(p_nodes, *p_n, "in", where), p_where, r_b, r_read);
            Curve c;
            curve(*p_n, where, c);
            r_b.program.curves.push_back(c);
            return Value{ {}, error.empty() ? emit(r_b, SampleOp::CURVE, a, -1, 0.0f, int(r_b.program.curves.size()) - 1) : -1 };
        }
        fail(where, "'" + kind + ov::str("' is not a node kind (note, warp, shape, multifractal, mix, constant, field, add, subtract, multiply, min, max, smin, smax, curve, up, output)."));
        return out;
    }

    // A lift's rank as the ground (graph(): the largest): its coarsest octave, then its octave count. Its structure, not
    // its values: an edit of a volume or an amplitude never makes another lift the ground.
    std::pair<int, int> rank(const std::vector<GraphRow> &p_rows) const {
        std::pair<int, int> r{ -1, 0 };
        for (const GraphRow &row : p_rows) {
            if (note_plays(row.note, ground)) {
                r.first = ov::max(r.first, row.note.k);
                r.second++;
            }
        }
        return r;
    }

    // A biome's "graph": its nodes by id, read back from the output's height (expr) into the biome's sample program
    // and r_g's rows: the lifted notes of the largest rank are the ground (GROUND, rows of channel 0; the first of
    // equals), the others the biome's sub-mixes 1, 2, ... (FIELD -c until the compiler gives each its field). A program
    // that is the ground alone is none.
    void graph(const ov::Value &p_d, const ov::str &p_where, Biome &r_b, BiomeGraph &r_g) {
        if (!keys(p_d, p_where, { "nodes" })) {
            return;
        }
        const ov::Value list = array(p_d, "nodes", p_where);
        std::vector<ov::Value> nodes;
        const ov::Value *out = nullptr;
        for (int64_t i = 0; i < int64_t(list.size()) && error.empty(); i++) {
            const ov::Value n = dict(list[i], p_where + ov::format(" node %d", i));
            const ov::str id = text(n, "id", p_where + ov::format(" node %d", i));
            text(n, "kind", p_where + ov::str(" node '") + id + ov::str("'"));
            for (const ov::Value &m : nodes) {
                if (m["id"].as_str() == id) {
                    fail(p_where, "two nodes are '" + id + ov::str("'."));
                }
            }
            nodes.push_back(n);
        }
        for (const ov::Value &n : nodes) {
            if (is_kind(&n, "output")) {
                if (out != nullptr) {
                    fail(p_where, "a graph has one output.");
                }
                out = &n;
            }
        }
        if (!error.empty()) {
            return;
        }
        if (out == nullptr || !node_keys(*out, p_where + ov::str(" output"), "output")) {
            fail(p_where, "a graph has an output node, its 'height' what the biome's ground is.");
            return;
        }
        Reading read;
        reg(nodes, input(nodes, *out, "height", p_where), p_where, r_b, read);
        if (!error.empty()) {
            return;
        }
        size_t top = 0;
        for (size_t i = 1; i < read.lifts.size(); i++) {
            top = rank(read.lifts[i].first) > rank(read.lifts[top].first) ? i : top;
        }
        int channel = 0;
        for (size_t i = 0; i < read.lifts.size(); i++) {
            SampleInstr &in = r_b.program.code[size_t(read.lifts[i].second)];
            const int c = i == top ? 0 : ++channel;
            in.op = c == 0 ? SampleOp::GROUND : SampleOp::FIELD;
            in.index = -c;
            for (GraphRow row : read.lifts[i].first) {
                row.channel = c;
                r_g.rows.push_back(row);
            }
        }
        if (r_b.program.code.size() == 1 && r_b.program.code[0].op == SampleOp::GROUND) {
            r_b.program = SampleProgram(); // the notes alone: no term at the samples
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

    void biome(const ov::Value &p_d, int64_t p_i, BiomeGraph &r_g) {
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
        if (raw != nullptr) {
            const ov::Value g = p_d.has("graph") ? dict(p_d["graph"], where + ov::str(" graph")) : ov::Value();
            if (p_d.has("graph") && keys(g, where + ov::str(" graph"), { "nodes" })) {
                raw->push_back(array(g, "nodes", where + ov::str(" graph")));
            } else {
                raw->push_back(ov::Value::array());
            }
        } else if (p_d.has("graph") && error.empty()) {
            graph(dict(p_d["graph"], where + ov::str(" graph")), where + ov::str(" graph"), b, r_g);
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
    parser.set_raw(r_graphs);
    parser.run(p_data);
    ov::str err = parser.error.empty() ? validate(r_table, p_ground) : parser.error;
    for (size_t f = 0; f < r_table.fields.size() && err.empty(); f++) {
        for (const Term &t : r_table.fields[f].terms) {
            if (t.source == GROUND) {
                err = "climate: field '" + r_table.fields[f].name + ov::str("': a 'ground' term is gone (the ground depends on the weights, which depend on the fields): a field's terms read earlier fields only.");
            }
        }
    }
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

} // namespace Climate
