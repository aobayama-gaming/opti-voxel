#include "graph_compile.hpp"

#include "ov/format.hpp"
#include "ov/math.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <utility>

namespace Terrain {

namespace {

using Climate::Ground;
using Term = std::pair<int, float>; // a LIN's input and its coefficient

// Whether p_ground has p_note's octave ([BAND_SHIFT, k_max]: a note the LOD count doesn't have plays nothing, so a
// world reads at any LOD count); its stretch on its octave (Band::stretch: octave k takes at most k - BAND_SHIFT of it);
// its amplitude (LOD0 voxels), under p_ground's note_cap.
bool note_plays(const Climate::NoteNode &p_note, const Ground &p_ground) {
    return p_note.k >= ExactNoise::BAND_SHIFT && p_note.k <= p_ground.k_max;
}

int note_stretch(const Climate::NoteNode &p_note) {
    return ov::max(0, ov::min(p_note.stretch, p_note.k - ExactNoise::BAND_SHIFT));
}

float note_amp(const Climate::NoteNode &p_note, const Ground &p_ground) {
    return float(ov::min(p_note.amp_m / p_ground.voxel_size, p_ground.note_cap));
}

// The program under construction: nodes hash-consed (one node per identity), their levels set and their constants
// folded as they are added; with `specialise`, the value-dependent folds too.
class Graph {
public:
    LevelProgram prog;
    bool specialise = true;

    const Node &node(int p_id) const { return prog.nodes[size_t(p_id)]; }
    bool is_constant(int p_id, float p_v) const { return node(p_id).op == Op::CONST && node(p_id).k[0] == p_v; }

    int add(Node p_n) {
        p_n.level = p_n.op == Op::NOTE ? p_n.band.k - ExactNoise::BAND_SHIFT : NO_LEVEL;
        bool constant = p_n.op != Op::NOTE && p_n.op != Op::CONST && p_n.op != Op::UP && p_n.op != Op::PICK && !p_n.in.empty();
        std::vector<Float4> in;
        for (const int i : p_n.in) {
            p_n.level = p_n.op == Op::NOTE ? p_n.level : ov::min(p_n.level, node(i).level);
            constant = constant && node(i).op == Op::CONST;
            in.push_back(Float4{ 0.0f, 0.0f, 0.0f, node(i).op == Op::CONST ? node(i).k[0] : 0.0f });
        }
        if (constant) {
            return this->constant(eval_node(prog, p_n, in.data(), OpPoint()).w);
        }
        std::vector<int64_t> key = { int64_t(p_n.op), p_n.curve, p_n.band.k, p_n.band.seed, p_n.band.offset.x, p_n.band.offset.y, p_n.band.offset.z, p_n.band.stretch };
        key.insert(key.end(), p_n.in.begin(), p_n.in.end());
        key.push_back(-1);
        for (const float k : p_n.k) {
            key.push_back(ExactNoise::float_bits(k));
        }
        const auto [it, fresh] = ids.emplace(key, int(prog.nodes.size()));
        if (fresh) {
            prog.nodes.push_back(p_n);
        }
        return it->second;
    }

    int constant(float p_v) {
        Node n;
        n.k = { p_v };
        return add(n);
    }

    int op(Op p_op, const std::vector<int> &p_in, const std::vector<float> &p_k = {}) {
        if (specialise && p_op == Op::MUL) {
            for (int i = 0; i < 2; i++) {
                if (is_constant(p_in[size_t(i)], 0.0f) || is_constant(p_in[size_t(i)], 1.0f)) {
                    return is_constant(p_in[size_t(i)], 0.0f) ? constant(0.0f) : p_in[size_t(1 - i)];
                }
            }
        }
        Node n;
        n.op = p_op;
        n.in = p_in;
        n.k = p_k;
        return add(n);
    }

    int lin(const std::vector<Term> &p_terms) {
        Node n;
        n.op = Op::LIN;
        for (const auto &[id, c] : p_terms) {
            if (!specialise || (c != 0.0f && !is_constant(id, 0.0f))) {
                n.in.push_back(id);
                n.k.push_back(c);
            }
        }
        if (n.in.empty()) {
            return constant(0.0f);
        }
        return specialise && n.in.size() == 1 && n.k[0] == 1.0f ? n.in[0] : add(n);
    }

    int curve(int p_in, const Climate::Curve &p_c) {
        Node n;
        n.op = Op::CURVE;
        n.in = { p_in };
        n.curve = int(prog.curves.size());
        prog.curves.push_back(p_c);
        return add(n);
    }

private:
    std::map<std::vector<int64_t>, int> ids;
};

// A note's unit noise: its octave on its own lattice, read displaced by p_warp's three nodes if any.
Node note_node(const Cascade::Band &p_band, const std::vector<int> &p_warp = {}) {
    Node n;
    n.op = Op::NOTE;
    n.band = p_band;
    n.band.amp = 1.0f;
    n.in = p_warp;
    return n;
}

// A wavelength in metres, for messages: level p_level's notes (octave level + BAND_SHIFT).
ov::str level_m(int p_level, const Ground &p_ground) {
    return ov::num(std::ldexp(p_ground.voxel_size, p_level + ExactNoise::BAND_SHIFT), 2) + ov::str(" m");
}

// One biome's graph read from its output back (each node once per seed offset: a warp's input is read three times,
// its notes' seeds + 0, 1, 2), into p_graph's nodes.
class Reader {
public:
    Reader(Graph &r_graph, const Ground &p_ground, const Climate::Table &p_table, const std::vector<int> &p_fields, const ov::Value &p_nodes, const ov::str &p_where) :
            g(r_graph), ground(p_ground), table(p_table), fields(p_fields), where(p_where) {
        for (const ov::Value &n : p_nodes.elements()) {
            const ov::str id = n["id"].as_str();
            if (!n.is_object() || !n["id"].is_string() || !n["kind"].is_string()) {
                fail(where, "a node is an object with an 'id' and a 'kind'.");
            } else if (!by_id.emplace(id, &n).second) {
                fail(where, "two nodes are '" + id + ov::str("'."));
            }
        }
    }

    ov::str error;

    // The graph's height (a constant 0 without nodes: a biome without a graph mixes no note).
    int output() {
        const ov::Value *out = nullptr;
        for (const auto &[id, n] : by_id) {
            if ((*n)["kind"].as_str() == "output") {
                if (out != nullptr) {
                    fail(where, "a graph has one output.");
                }
                out = n;
            }
        }
        if (by_id.empty() && error.empty()) {
            return g.constant(0.0f);
        }
        if (out == nullptr) {
            fail(where, "a graph has an output node, its 'height' what the biome's ground is.");
            return -1;
        }
        return keys(*out) ? read(input(*out, "height"), 0) : -1;
    }

private:
    Graph &g;
    const Ground &ground;
    const Climate::Table &table;
    const std::vector<int> &fields;
    ov::str where;
    std::map<ov::str, const ov::Value *> by_id;
    std::map<std::pair<ov::str, uint32_t>, int> memo;
    std::set<std::pair<ov::str, uint32_t>> active;

    bool fail(const ov::str &p_where, const ov::str &p_msg) {
        if (error.empty()) {
            error = "climate: " + p_where + ov::str(": ") + p_msg;
        }
        return false;
    }
    bool take(const ov::str &p_err) {
        if (error.empty()) {
            error = p_err;
        }
        return p_err.empty();
    }
    ov::str at(const ov::Value &p_n) const { return where + ov::str(" node '") + p_n["id"].as_str() + ov::str("'"); }
    bool keys(const ov::Value &p_n) { return take(Climate::parse_node_keys(p_n, at(p_n))); }
    double number(const ov::Value &p_n, const char *p_key) {
        double v = 0.0;
        take(Climate::parse_number(p_n, p_key, at(p_n), v));
        return v;
    }

    // The node input p_key of p_d names (p_d a node, or a mix's row: p_owner names it in the errors).
    const ov::Value *input(const ov::Value &p_d, const char *p_key, const ov::Value *p_owner = nullptr) {
        const ov::str owner = at(p_owner != nullptr ? *p_owner : p_d);
        if (!p_d[p_key].is_string()) {
            fail(owner, "'" + ov::str(p_key) + ov::str("' names a node."));
            return nullptr;
        }
        const auto it = by_id.find(p_d[p_key].as_str());
        if (it == by_id.end()) {
            fail(owner, "no node '" + p_d[p_key].as_str() + ov::str("'."));
            return nullptr;
        }
        return it->second;
    }

    int read(const ov::Value *p_n, uint32_t p_seed) {
        if (p_n == nullptr || !error.empty()) {
            return -1;
        }
        const std::pair<ov::str, uint32_t> key{ (*p_n)["id"].as_str(), p_seed };
        if (const auto it = memo.find(key); it != memo.end()) {
            return it->second;
        }
        if (!active.insert(key).second) {
            fail(at(*p_n), "the graph loops (the node reads itself through its inputs).");
            return -1;
        }
        const int id = value(*p_n, p_seed);
        active.erase(key);
        memo[key] = id;
        return error.empty() ? id : -1;
    }

    // A note's unit noise (r_unit, -1 if the LOD count doesn't have its octave: it plays nothing) and its amplitude.
    void note(const ov::Value &p_n, uint32_t p_seed, int &r_unit, float &r_amp) {
        r_unit = -1;
        r_amp = 0.0f;
        Climate::NoteNode nn;
        if (!take(Climate::parse_note(p_n, ground, at(p_n), nn)) || !note_plays(nn, ground)) {
            return;
        }
        Cascade::Band band = ExactNoise::make_octave(nn.seed + p_seed, uint32_t(nn.k), nn.k, 1.0f);
        band.stretch = note_stretch(nn);
        std::vector<int> disp;
        if (p_n.has("warp")) {
            const ov::Value *w = input(p_n, "warp");
            if (w != nullptr && (*w)["kind"].as_str() != "warp") {
                fail(at(p_n), "'warp' is a warp node.");
            }
            if (w == nullptr || !warp(*w, p_seed, nn.k - ExactNoise::BAND_SHIFT, at(p_n), disp)) {
                return;
            }
        }
        r_unit = g.add(note_node(band, disp));
        r_amp = note_amp(nn, ground);
    }

    // A warp's three displacements (LOD0 voxels): amount_m times its input read with seeds + 0, 1, 2, or times its x, y,
    // z ports. The gate: each strictly coarser than the note it moves (level p_level).
    bool warp(const ov::Value &p_w, uint32_t p_seed, int p_level, const ov::str &p_note, std::vector<int> &r_disp) {
        ov::Value plain = p_w;
        bool ports = false;
        for (const char *axis : { "x", "y", "z" }) {
            ports = plain.erase(axis) || ports;
        }
        if (!keys(plain)) {
            return false;
        }
        if (ports == p_w.has("in")) {
            return fail(at(p_w), "a warp reads 'in' (three times, its notes' seeds + 0, 1, 2) or the ports 'x', 'y', 'z'.");
        }
        const float amount = float(number(p_w, "amount_m"));
        static const char *const PORTS[3] = { "x", "y", "z" };
        for (uint32_t a = 0; a < 3 && error.empty(); a++) {
            const int in = ports ? read(input(p_w, PORTS[a]), p_seed) : read(input(p_w, "in"), p_seed + a);
            if (in < 0) {
                return false;
            }
            const int level = g.node(in).level;
            if (level != NO_LEVEL && level <= p_level) {
                return fail(p_note, ov::format("its warp '%s' isn't coarser than the note: it reads a note of %s, the note is %s (a warp's input is strictly coarser than the "
                                               "notes it moves: it must exist before they are inserted).",
                                            p_w["id"].as_str(), level_m(level, ground), level_m(p_level, ground)));
            }
            r_disp.push_back(g.lin({ { in, amount } }));
        }
        return error.empty();
    }

    // A shape: on a note, its noise shaped without the mean (Cascade::NOISE_*_MEAN; the format refuses a constant term
    // there) times the note's amplitude; on anything else, phi of its value, its constant term counting.
    int shape(const ov::Value &p_n, uint32_t p_seed) {
        const ov::Value *in = input(p_n, "in");
        const bool on_note = in != nullptr && (*in)["kind"].as_str() == "note";
        Climate::Shape s;
        if (in == nullptr || !take(Climate::parse_shape(p_n, !on_note, s))) {
            return -1;
        }
        if (!on_note) {
            const int a = read(in, p_seed);
            return a < 0 ? -1 : g.op(Op::SHAPE, { a }, { s.c[0], s.c[1], s.c[2], s.c[3] });
        }
        int unit = -1;
        float amp = 0.0f;
        note(*in, p_seed, unit, amp);
        if (unit < 0) {
            return error.empty() ? g.constant(0.0f) : -1;
        }
        const float mean = s.c[2] * Cascade::NOISE_ABS_MEAN + s.c[3] * Cascade::NOISE_SQ_MEAN;
        return g.lin({ { g.op(Op::SHAPE, { unit }, { -mean, s.c[1], s.c[2], s.c[3] }), amp } });
    }

    // multiply(in, ramp(parent)), ramp the two-point curve from (low_m, 0) to (high_m, 1), clamped.
    int multifractal(const ov::Value &p_n, uint32_t p_seed) {
        const double low = number(p_n, "low_m") / ground.voxel_size, high = number(p_n, "high_m") / ground.voxel_size;
        if (error.empty() && !(std::isfinite(low) && std::isfinite(high) && low != high)) {
            fail(at(p_n), "a multifractal's low_m and high_m differ.");
        }
        const int a = read(input(p_n, "in"), p_seed), parent = read(input(p_n, "parent"), p_seed);
        if (a < 0 || parent < 0) {
            return -1;
        }
        const Climate::Curve ramp = low < high ? Climate::make_curve({ low, high }, { 0.0, 1.0 }) : Climate::make_curve({ high, low }, { 1.0, 0.0 });
        return g.op(Op::MUL, { a, g.curve(parent, ramp) });
    }

    int mix(const ov::Value &p_n, uint32_t p_seed) {
        if (!p_n["rows"].is_array()) {
            fail(at(p_n), "'rows' must be a list.");
            return -1;
        }
        std::vector<Term> terms;
        for (const ov::Value &row : p_n["rows"].elements()) {
            for (const auto &[key, v] : row.entries()) {
                if (key != "in" && key != "x" && key != "comment") {
                    fail(at(p_n), "a row is { \"in\": id, \"x\": volume }, not '" + key + ov::str("'."));
                }
            }
            const double x = row.has("x") ? row["x"].as_double() : 1.0;
            if (!row.is_object() || (row.has("x") && !row["x"].is_number())) {
                fail(at(p_n), "a row is { \"in\": id, \"x\": volume }.");
            }
            terms.emplace_back(read(input(row, "in", &p_n), p_seed), float(x));
            if (!error.empty()) {
                return -1;
            }
        }
        return g.lin(terms);
    }

    int curve(const ov::Value &p_n, uint32_t p_seed) {
        Climate::Curve c;
        if (!take(Climate::parse_curve(p_n, ground, at(p_n), c))) {
            return -1;
        }
        if (double(Climate::curve_slope_bound(c)) > Climate::MAX_CURVE_SLOPE) {
            fail(at(p_n), ov::format("the curve's offset R - h has a slope up to %s, over %s: spread its points.", ov::num(Climate::curve_slope_bound(c), 2), Climate::MAX_CURVE_SLOPE));
            return -1;
        }
        const int a = read(input(p_n, "in"), p_seed);
        return a < 0 ? -1 : g.lin({ { a, 1.0f }, { g.curve(a, c), 1.0f } });
    }

    // add, subtract, multiply, min, max, smin, smax.
    int binary(const ov::Value &p_n, const ov::str &p_kind, uint32_t p_seed) {
        const bool smooth = p_kind == "smin" || p_kind == "smax";
        const double k = smooth ? number(p_n, "k_m") / ground.voxel_size : 0.0;
        if (smooth && error.empty() && !(k > 0.0 && std::isfinite(k))) {
            fail(at(p_n), "a smooth min / max's k_m is > 0.");
        }
        const int a = read(input(p_n, "a"), p_seed), b = read(input(p_n, "b"), p_seed);
        if (a < 0 || b < 0) {
            return -1;
        }
        if (p_kind == "add" || p_kind == "subtract") {
            return g.lin({ { a, 1.0f }, { b, p_kind == "add" ? 1.0f : -1.0f } });
        }
        if (smooth) {
            return g.op(p_kind == "smin" ? Op::SMIN : Op::SMAX, { a, b }, { float(k), float(0.5 / double(float(k))) });
        }
        return g.op(p_kind == "multiply" ? Op::MUL : (p_kind == "min" ? Op::MIN : Op::MAX), { a, b });
    }

    int value(const ov::Value &p_n, uint32_t p_seed) {
        const ov::str kind = p_n["kind"].as_str();
        if (kind == "warp") {
            fail(at(p_n), "a warp is a note's input ('warp'): it moves where that note is read.");
            return -1;
        }
        if (!keys(p_n)) {
            return -1;
        }
        if (kind == "note") {
            int unit = -1;
            float amp = 0.0f;
            note(p_n, p_seed, unit, amp);
            return unit < 0 ? (error.empty() ? g.constant(0.0f) : -1) : g.lin({ { unit, amp } });
        }
        if (kind == "shape") {
            return shape(p_n, p_seed);
        }
        if (kind == "multifractal") {
            return multifractal(p_n, p_seed);
        }
        if (kind == "mix") {
            return mix(p_n, p_seed);
        }
        if (kind == "curve") {
            return curve(p_n, p_seed);
        }
        if (kind == "constant") {
            const double v = number(p_n, "value_m");
            return error.empty() ? g.constant(float(v / ground.voxel_size)) : -1;
        }
        if (kind == "up") {
            return g.op(Op::UP, {});
        }
        if (kind == "field") {
            const int f = table.field_index(p_n["field"].as_str());
            if (f < 0) {
                fail(at(p_n), "'" + p_n["field"].as_str() + ov::str("' is not a field."));
                return -1;
            }
            return fields[size_t(f)];
        }
        if (kind == "output") {
            fail(at(p_n), "an output is read by no node.");
            return -1;
        }
        return binary(p_n, kind, p_seed);
    }
};

// Each field's node: the LIN of its bands (unit notes times their amplitudes) and its terms on earlier fields.
std::vector<int> field_nodes(Graph &r_g, const Climate::Table &p_table) {
    std::vector<int> fields;
    for (const Climate::Field &f : p_table.fields) {
        std::vector<Term> terms;
        for (const Cascade::Band &b : f.bands) {
            terms.emplace_back(r_g.add(note_node(b)), b.amp);
        }
        for (const Climate::Term &t : f.terms) {
            terms.emplace_back(fields[size_t(t.source)], t.coef);
        }
        fields.push_back(r_g.lin(terms));
    }
    return fields;
}

// The biome weights (the unit tests' oracle: ClimateTest::weights, op for op): each kernel the product of its axes' factors, the default's fill of
// their sum, each over the total.
std::vector<int> weight_nodes(Graph &r_g, const Climate::Table &p_table, const std::vector<int> &p_fields) {
    const size_t n = p_table.biomes.size();
    if (n == 1) {
        return { r_g.constant(1.0f) };
    }
    std::vector<int> k(n, -1);
    std::vector<Term> sum;
    for (size_t b = 0; b + 1 < n; b++) {
        for (const Climate::Axis &a : p_table.biomes[b].at) {
            const int f = r_g.op(Op::AXIS, { p_fields[size_t(a.field)] }, { a.centre, a.inv_span });
            k[b] = k[b] < 0 ? f : r_g.op(Op::MUL, { k[b], f });
        }
        sum.emplace_back(k[b], 1.0f);
    }
    const int total = r_g.lin(sum);
    k[n - 1] = r_g.op(Op::FILL, { total });
    const int rcp = r_g.op(Op::RCP, { r_g.lin({ { total, 1.0f }, { k[n - 1], 1.0f } }) });
    std::vector<int> w;
    for (size_t b = 0; b < n; b++) {
        w.push_back(r_g.op(Op::NORM, { k[b], rcp }));
    }
    return w;
}

// p_id as LIN terms, nested LINs expanded (blend factoring's view of a biome's output).
void expand(const Graph &p_g, int p_id, float p_coef, std::map<int, float> &r_terms) {
    const Node &n = p_g.node(p_id);
    if (n.op != Op::LIN) {
        r_terms[p_id] = r_terms[p_id] + p_coef;
        return;
    }
    for (size_t i = 0; i < n.in.size(); i++) {
        expand(p_g, n.in[i], p_coef * n.k[i], r_terms);
    }
}

// The ground, sum_b w_b out_b; specialised, the terms every biome's output has alike (same node, same coefficient) out
// of the blend.
int blend(Graph &r_g, const std::vector<int> &p_w, std::vector<int> p_out) {
    if (p_out.size() == 1) {
        return p_out[0];
    }
    std::vector<Term> terms;
    if (r_g.specialise) {
        std::vector<std::map<int, float>> exp(p_out.size());
        for (size_t b = 0; b < p_out.size(); b++) {
            expand(r_g, p_out[b], 1.0f, exp[b]);
        }
        for (const auto &[id, c] : exp[0]) {
            const auto alike = [&, id = id, c = c](const std::map<int, float> &p_e) {
                const auto it = p_e.find(id);
                return it != p_e.end() && ExactNoise::float_bits(it->second) == ExactNoise::float_bits(c);
            };
            if (std::all_of(exp.begin(), exp.end(), alike)) {
                terms.emplace_back(id, c);
            }
        }
        for (size_t b = 0; b < p_out.size() && !terms.empty(); b++) {
            std::vector<Term> rest;
            for (const auto &[id, c] : exp[b]) {
                if (std::none_of(terms.begin(), terms.end(), [&, id = id](const Term &p_t) { return p_t.first == id; })) {
                    rest.emplace_back(id, c);
                }
            }
            p_out[b] = r_g.lin(rest);
        }
    }
    for (size_t b = 0; b < p_out.size(); b++) {
        terms.emplace_back(r_g.op(Op::MUL, { p_w[b], p_out[b] }), 1.0f);
    }
    return r_g.lin(terms);
}

// How many times each node is read (the roots once each); 0: dead.
std::vector<int> use_counts(const LevelProgram &p_prog, const std::vector<int> &p_roots) {
    std::vector<int> use(p_prog.nodes.size(), 0);
    std::vector<bool> live(p_prog.nodes.size(), false);
    for (const int r : p_roots) {
        use[size_t(r)]++;
        live[size_t(r)] = true;
    }
    for (size_t i = p_prog.nodes.size(); i-- > 0;) {
        for (const int in : p_prog.nodes[i].in) {
            use[size_t(in)] += live[i] ? 1 : 0;
            live[size_t(in)] = live[size_t(in)] || live[i];
        }
    }
    return use;
}

// The live nodes of p_old in order, each through p_emit(node with its inputs renumbered, use counts, the new graph):
// its new id. The roots renumbered.
template <typename F>
Graph rebuild(const Graph &p_old, std::vector<int> &r_roots, F p_emit) {
    const std::vector<int> use = use_counts(p_old.prog, r_roots);
    Graph g;
    g.specialise = p_old.specialise;
    g.prog.curves = p_old.prog.curves;
    std::vector<int> id(p_old.prog.nodes.size(), -1);
    for (size_t i = 0; i < p_old.prog.nodes.size(); i++) {
        if (use[i] == 0) {
            continue;
        }
        Node n = p_old.prog.nodes[i];
        std::vector<int> old_in = n.in;
        for (int &in : n.in) {
            in = id[size_t(in)];
        }
        id[i] = p_emit(n, old_in, use, g);
    }
    for (int &r : r_roots) {
        r = id[size_t(r)];
    }
    return g;
}

// A LIN read once by another LIN merged into it (its terms times the reader's coefficient).
Graph merge_lins(const Graph &p_old, std::vector<int> &r_roots) {
    return rebuild(p_old, r_roots, [](const Node &p_n, const std::vector<int> &p_old_in, const std::vector<int> &p_use, Graph &r_g) {
        if (p_n.op != Op::LIN) {
            return r_g.add(p_n);
        }
        std::vector<Term> terms;
        for (size_t i = 0; i < p_n.in.size(); i++) {
            const Node &in = r_g.node(p_n.in[i]);
            if (in.op != Op::LIN || p_use[size_t(p_old_in[i])] != 1) {
                terms.emplace_back(p_n.in[i], p_n.k[i]);
                continue;
            }
            for (size_t j = 0; j < in.in.size(); j++) {
                terms.emplace_back(in.in[j], p_n.k[i] * in.k[j]);
            }
        }
        return r_g.lin(terms);
    });
}

// A LIN whose inputs sit on several levels as a chain of partial sums, coarsest first, each reading the one above: one
// channel crosses each level (Hermite subdivision is linear). A constant joins the finest sum.
Graph partial_sums(const Graph &p_old, std::vector<int> &r_roots) {
    return rebuild(p_old, r_roots, [](const Node &p_n, const std::vector<int> &, const std::vector<int> &, Graph &r_g) {
        if (p_n.op != Op::LIN) {
            return r_g.add(p_n);
        }
        std::map<int, std::vector<Term>, std::greater<int>> by_level;
        for (size_t i = 0; i < p_n.in.size(); i++) {
            const int level = r_g.node(p_n.in[i]).level;
            by_level[level == NO_LEVEL ? p_n.level : level].emplace_back(p_n.in[i], p_n.k[i]);
        }
        if (by_level.size() < 2) {
            return r_g.add(p_n);
        }
        int sum = -1;
        for (auto &[level, terms] : by_level) {
            if (sum >= 0) {
                terms.insert(terms.begin(), Term{ sum, 1.0f });
            }
            sum = r_g.lin(terms);
        }
        return sum;
    });
}

// Each node into its level's code (a node of no level into every level that reads it), liveness into each level's out.
ov::str lower(LevelProgram &r_prog) {
    const size_t n = r_prog.nodes.size();
    r_prog.top = 0;
    for (const Node &node : r_prog.nodes) {
        r_prog.top = node.level == NO_LEVEL ? r_prog.top : ov::max(r_prog.top, node.level);
    }
    if (r_prog.top > Cascade::MAX_LEVEL) {
        return ov::format("climate: a note on level %d, past the cascade's %d.", int64_t(r_prog.top), int64_t(Cascade::MAX_LEVEL));
    }
    r_prog.levels.assign(size_t(r_prog.top + 1), Level());
    std::vector<std::set<int>> need(n); // a node of no level: the levels computing it
    std::vector<int> last(n, NO_LEVEL); // a node with a level: the finest level reading it
    for (const int root : { r_prog.output, r_prog.pick }) {
        if (r_prog.nodes[size_t(root)].level == NO_LEVEL) {
            need[size_t(root)].insert(0);
        }
        last[size_t(root)] = 0;
    }
    for (size_t i = n; i-- > 0;) {
        const Node &node = r_prog.nodes[i];
        const std::set<int> at = node.level == NO_LEVEL ? need[i] : std::set<int>{ node.level };
        for (const int in : node.in) {
            for (const int l : at) {
                if (r_prog.nodes[size_t(in)].level == NO_LEVEL) {
                    need[size_t(in)].insert(l);
                } else {
                    last[size_t(in)] = ov::min(last[size_t(in)], l);
                }
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        const Node &node = r_prog.nodes[i];
        for (const int l : node.level == NO_LEVEL ? need[i] : std::set<int>{ node.level }) {
            r_prog.levels[size_t(l)].code.push_back(int(i));
        }
        for (int m = node.level; node.level != NO_LEVEL && m > last[i]; m--) {
            r_prog.levels[size_t(m)].out.push_back(int(i));
        }
    }
    for (int m = 0; m <= r_prog.top; m++) {
        const Level &l = r_prog.levels[size_t(m)];
        if (l.out.size() > size_t(MAX_CHANNELS) || l.code.size() > size_t(MAX_LEVEL_INSTRS)) {
            return ov::format("climate: level %d computes %d nodes and carries %d channels, at most %d and %d (fewer notes or biomes).", int64_t(m), int64_t(l.code.size()),
                    int64_t(l.out.size()), int64_t(MAX_LEVEL_INSTRS), int64_t(MAX_CHANNELS));
        }
    }
    return ov::str();
}

} // namespace

ov::str compile(const ov::Value &p_data, const Climate::Ground &p_ground, Mode p_mode, Climate::Table &r_table, LevelProgram &r_program) {
    r_program = LevelProgram();
    std::vector<ov::Value> graphs;
    ov::str err = Climate::parse_layout(p_data, p_ground, r_table, graphs);
    if (!err.empty()) {
        return err;
    }
    Graph g;
    g.specialise = p_mode == Mode::SPECIALISED;
    const std::vector<int> fields = field_nodes(g, r_table);
    std::vector<int> out;
    for (size_t b = 0; b < r_table.biomes.size(); b++) {
        Reader reader(g, p_ground, r_table, fields, graphs[b], "biome '" + r_table.biomes[b].name + ov::str("' graph"));
        out.push_back(reader.output());
        if (!reader.error.empty()) {
            r_table = Climate::Table();
            return reader.error;
        }
    }
    const std::vector<int> w = weight_nodes(g, r_table, fields);
    Node pick;
    pick.op = Op::PICK;
    pick.in = w;
    // The ground and the pick, then the fields and the weights (read on the CPU: BiomePick, CloudClimate, ClimateReport).
    std::vector<int> roots = { blend(g, w, out), w.size() == 1 ? g.constant(0.0f) : g.add(pick) };
    roots.insert(roots.end(), fields.begin(), fields.end());
    roots.insert(roots.end(), w.begin(), w.end());
    g = merge_lins(g, roots);
    g = partial_sums(g, roots);
    g = rebuild(g, roots, [](const Node &p_n, const std::vector<int> &, const std::vector<int> &, Graph &r_g) { return r_g.add(p_n); });
    r_program = std::move(g.prog);
    r_program.output = roots[0];
    r_program.pick = roots[1];
    r_program.fields.assign(roots.begin() + 2, roots.begin() + 2 + ptrdiff_t(fields.size()));
    r_program.weights.assign(roots.begin() + 2 + ptrdiff_t(fields.size()), roots.end());
    r_program.biomes = int(r_table.biomes.size());
    r_program.generic = p_mode == Mode::GENERIC;
    err = lower(r_program);
    err = err.empty() ? bound_program(p_ground, r_program) : err;
    if (!err.empty()) {
        r_program = LevelProgram();
        r_table = Climate::Table();
    }
    return err;
}

} // namespace Terrain
