#include "level_glsl.hpp"

#include "cascade_glsl.hpp"
#include "climate.hpp"
#include "exact_noise.hpp"

#include "ov/format.hpp"
#include "ov/math.hpp"

#include <algorithm>
#include <climits>
#include <iterator>

namespace Terrain {

using Cascade::ShaderParams;

namespace {

// A register's GLSL.
ov::str reg(int p_r) {
    return ov::format("r[%d]", p_r);
}

// Channel k of Ch as a dual (x, y, z: gradient, w: value; Ch holds Cascade::Channel order) and back.
ov::str ch_get(const ov::str &p_k) {
    return ov::format("vec4(ch.c[4 * %s + 1], ch.c[4 * %s + 2], ch.c[4 * %s + 3], ch.c[4 * %s])", p_k, p_k, p_k, p_k);
}

ov::str ch_put(const ov::str &p_k, const ov::str &p_v) {
    return ov::format("        { vec4 v = %s; o.c[4 * %s] = v.w; o.c[4 * %s + 1] = v.x; o.c[4 * %s + 2] = v.y; o.c[4 * %s + 3] = v.z; }\n", p_v, p_k, p_k, p_k, p_k);
}

// An op's GLSL form with its operands: p_in(i) input i, p_k(i) constant i, and the node's own (@level, @band, @curve).
template <typename In, typename K>
ov::str form(const char *p_form, In p_in, K p_k, const ov::str &p_level, const ov::str &p_band, const ov::str &p_curve) {
    ov::str s = p_form;
    for (int i = 0; i < 4; i++) {
        s = ov::replace_all(s, ov::format("$%d", i), p_in(i));
        s = ov::replace_all(s, ov::format("#%d", i), p_k(i));
    }
    return ov::replace_all(s, { { "@p", "p" }, { "@m", "m" }, { "@level", p_level }, { "@band", p_band }, { "@curve", p_curve } });
}

// The curves' records in the heap (Climate::curve_at_glsl's layout): each curve's index.
std::vector<int64_t> curve_records(const LevelProgram &p_prog, ShaderParams &r_params) {
    std::vector<int64_t> at;
    for (const Climate::Curve &c : p_prog.curves) {
        std::vector<float> rec = { float(c.x.size()), c.y_last };
        rec.insert(rec.end(), c.x.begin(), c.x.end());
        for (size_t i = 0; i < c.inv.size(); i++) {
            rec.insert(rec.end(), { c.inv[i], c.p[i], c.q[i], c.r[i], c.s[i] });
        }
        at.push_back(r_params.put(rec));
    }
    return at;
}

// Whether NOTE p_n of level p_s's code is computed in the level's note loop (the specialised text): its level is p_s
// and its inputs (a warp's, strictly coarser) are ready when the level starts, none computed by the level's own code.
bool looped(const LevelProgram &p_prog, int p_s, int p_n) {
    const Node &node = p_prog.nodes[size_t(p_n)];
    if (node.op != Op::NOTE || node.level != p_s) {
        return false;
    }
    const std::vector<int> &code = p_prog.levels[size_t(p_s)].code;
    for (const int in : node.in) {
        if (std::find(code.begin(), code.end(), in) != code.end()) {
            return false;
        }
    }
    return true;
}

// Each NOTE's entry in band_table (note_order's index).
std::vector<int> band_index(const LevelProgram &p_prog) {
    std::vector<int> index(p_prog.nodes.size(), -1);
    const std::vector<int> order = note_order(p_prog);
    for (size_t i = 0; i < order.size(); i++) {
        index[size_t(order[i])] = int(i);
    }
    return index;
}

// Node p_n's expression in the specialised text.
ov::str expression(const LevelProgram &p_prog, const GpuLayout &p_layout, int p_n, const std::vector<int> &p_band, const std::vector<int64_t> &p_curve,
        ShaderParams &r_params) {
    const Node &node = p_prog.nodes[size_t(p_n)];
    const auto in = [&](int p_i) { return p_i < int(node.in.size()) ? reg(p_layout.reg[size_t(node.in[size_t(p_i)])]) : ov::str("vec4(0.0)"); };
    const auto k = [&](int p_i) { return p_i < int(node.k.size()) ? r_params(node.k[size_t(p_i)]) : ov::str("0.0"); };
    const char *glsl = op_def(node.op).glsl;
    if (node.op == Op::LIN) {
        ov::str sum;
        for (size_t i = 0; i < node.in.size(); i++) {
            const ov::str t = ov::replace_all(glsl, { { "$i", in(int(i)) }, { "#i", k(int(i)) } });
            sum = i == 0 ? t : "cl_add(" + sum + ", " + t + ")";
        }
        return sum.empty() ? ov::str("vec4(0.0)") : sum;
    }
    if (node.op == Op::PICK) {
        ov::str all = ov::format("vec4[%d](", int64_t(Climate::MAX_BIOMES));
        for (int b = 0; b < Climate::MAX_BIOMES; b++) {
            all += (b ? ", " : "") + (b < int(node.in.size()) ? in(b) : ov::str("vec4(0.0)"));
        }
        return ov::replace_all(glsl, { { "$all", all + ")" }, { "#n", ov::itos(int64_t(node.in.size())) }, { "@p", "p" } });
    }
    const ov::str band = node.op == Op::NOTE ? ov::format("bands[%d]", int64_t(p_band[size_t(p_n)])) : ov::str();
    const ov::str curve = node.op == Op::CURVE ? ov::itos(p_curve[size_t(node.curve)]) : ov::str();
    return form(glsl, in, k, ov::itos(node.level), band, curve);
}

// A node's constant (Node::folded) as a dual: no gradient.
ov::str folded_literal(const Node &p_node, ShaderParams &r_params) {
    return "vec4(0.0, 0.0, 0.0, " + r_params(p_node.folded) + ")";
}

// LevelProgram::last_stop per LOD m = 0 .. MAX_LEVEL (both texts' table: where a run's last level stops).
std::vector<float> last_stops(const LevelProgram &p_prog) {
    std::vector<float> stops;
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        stops.push_back(float(p_prog.last_stop(m)));
    }
    return stops;
}

// Biome b's bit in the GLSL's masks (Terrain::biome_bit).
ov::str bit_glsl(int p_b) {
    return ov::format("%du", int64_t(biome_bit(p_b)));
}

// The specialised text's `on` word at level s's start (empty: no guards): above the last level the run's mask, on the
// last level the bits of the gating weights (Node::mark) not 0 at the point (registers computed on a level above s, or
// carried to it).
ov::str on_word(const LevelProgram &p_prog, const GpuLayout &p_layout) {
    ov::str bits;
    for (size_t n = 0; n < p_prog.nodes.size(); n++) {
        if (p_prog.nodes[n].mark != 0u) {
            bits += (bits.empty() ? "" : " | ") + ov::format("(%s != vec4(0.0) ? %du : 0u)", reg(p_layout.reg[n]), int64_t(p_prog.nodes[n].mark));
        }
    }
    return bits.empty() ? ov::str() : "        uint on = last ? (" + bits + ") : mask;\n";
}

// Node p_n's folded value on a run of LOD m (its constant from const_lod up), p_expr its expression.
ov::str folded_value(const Node &p_node, const ov::str &p_expr, ShaderParams &r_params) {
    const ov::str c = folded_literal(p_node, r_params);
    if (p_node.const_lod <= 0) {
        return c;
    }
    return p_node.const_lod < NO_LEVEL ? ov::format("(m >= %d ? %s : %s)", int64_t(p_node.const_lod), c, p_expr) : p_expr;
}

// p_lines inside `p_head {` ... `}`, indented.
std::vector<ov::str> wrapped(const ov::str &p_head, const std::vector<ov::str> &p_lines, const std::vector<ov::str> &p_else = {}) {
    std::vector<ov::str> t = { p_head + " {" };
    for (const ov::str &l : p_lines) {
        t.push_back("    " + l);
    }
    if (!p_else.empty()) {
        t.push_back("} else {");
        for (const ov::str &l : p_else) {
            t.push_back("    " + l);
        }
    }
    t.push_back("}");
    return t;
}

// Node p_n's statement in the specialised text, p_expr under its fold and guard, in compute()'s order: nothing on a
// last level of LOD > live_lod; 0 where its biome is off (guard: the `on` word); its constant on a run of LOD >=
// const_lod; else p_expr. A weight with a mark (Node::mark) then sets those bits in the run's mask, above the last level.
ov::str statement(const LevelProgram &p_prog, const GpuLayout &p_layout, int p_n, const ov::str &p_expr, ShaderParams &r_params) {
    const Node &node = p_prog.nodes[size_t(p_n)];
    const ov::str dst = reg(p_layout.reg[size_t(p_n)]);
    const ov::str c = dst + " = " + folded_literal(node, r_params) + ";", e = dst + " = " + p_expr + ";";
    std::vector<ov::str> lines = { e };
    if (node.const_lod <= 0) {
        lines = { c };
    } else if (node.const_lod < NO_LEVEL) {
        lines = wrapped(ov::format("if (m >= %d)", int64_t(node.const_lod)), { c }, { e });
    }
    if (node.guard >= 0) {
        lines = wrapped("if ((on & " + bit_glsl(node.guard) + ") == 0u)", { dst + " = vec4(0.0);" }, lines);
    }
    if (node.live_lod < NO_LEVEL) {
        lines = wrapped(ov::format("if (!last || m <= %d)", int64_t(node.live_lod)), lines);
    }
    if (node.mark != 0u) {
        // a plain read first: once a thread's atomic has landed, most others skip theirs
        const ov::str bits = ov::format("%du", int64_t(node.mark));
        const std::vector<ov::str> mark = wrapped("if (!last && " + dst + " != vec4(0.0) && (biome_mask[g_item] & " + bits + ") != " + bits + ")", { "atomicOr(biome_mask[g_item], " + bits + ");" });
        lines.insert(lines.end(), mark.begin(), mark.end());
    }
    ov::str t;
    for (const ov::str &l : lines) {
        t += "            " + l + "\n";
    }
    return t;
}

// The specialised text's note loop: level s's looped notes (one range of the band table, note_order) in one loop at the
// level's start, so the text holds one perlin3 whatever the note count (one inlined call per note took the NVIDIA
// compiler minutes and gigabytes for 80 notes); each note's statement then copies its nv slot. r_slot: per node, its
// slot in nv (-1: computed by its own statement).
ov::str note_loop(const LevelProgram &p_prog, const std::vector<int> &p_band, const GpuLayout &p_layout, const ov::str &p_on, ShaderParams &r_params, std::vector<int> &r_slot) {
    r_slot.assign(p_prog.nodes.size(), -1);
    std::vector<float> first(size_t(Cascade::MAX_LEVEL + 1), 0.0f), count(size_t(Cascade::MAX_LEVEL + 1), 0.0f);
    int most = 0;
    ov::str warps;
    for (int s = p_prog.top; s >= 0; s--) {
        int n = 0;
        ov::str gather;
        for (const int c : p_prog.levels[size_t(s)].code) {
            if (!looped(p_prog, s, c)) {
                continue;
            }
            first[size_t(s)] = n == 0 ? float(p_band[size_t(c)]) : first[size_t(s)];
            const Node &node = p_prog.nodes[size_t(c)];
            for (size_t i = 0; i < node.in.size(); i++) {
                gather += ov::format("            nw[%d] = %s;\n", int64_t(3 * n + int(i)), reg(p_layout.reg[size_t(node.in[i])]));
            }
            r_slot[size_t(c)] = n++;
        }
        count[size_t(s)] = float(n);
        most = ov::max(most, n);
        warps += gather.empty() ? ov::str() : ov::format("        case %d:\n", int64_t(s)) + gather + "            break;\n";
    }
    if (most == 0) {
        return ov::str();
    }
    ov::str t = p_on;
    t += "        int nf = int(" + r_params.table(first, "min(s, " + ov::itos(Cascade::MAX_LEVEL) + ")") + ");\n";
    // a coarser run's notes are absent (their statements folded to 0)
    t += "        int nn = m > s ? 0 : int(" + r_params.table(count, "min(s, " + ov::itos(Cascade::MAX_LEVEL) + ")") + ");\n";
    ov::str w = "vec4(0.0), vec4(0.0), vec4(0.0)";
    if (!warps.empty()) {
        t += ov::format("        vec4 nw[%d];\n        for (int e = 0; e < %d; e++) {\n            nw[e] = vec4(0.0);\n        }\n", int64_t(3 * most), int64_t(3 * most));
        t += "        switch (s) {\n" + warps + "        default:\n            break;\n        }\n";
        w = "nw[3 * e], nw[3 * e + 1], nw[3 * e + 2]";
    }
    t += ov::format("        vec4 nv[%d];\n        for (int e = 0; e < nn; e++) {\n", int64_t(most));
    const ov::str call = "nv[e] = tp_note(p, m, s, bands[nf + e], " + w + ");";
    if (p_on.empty()) {
        return t + "            " + call + "\n        }\n";
    }
    // Each band table entry's guard bit (0: none), a note of a biome off 0, as its statement would make it.
    std::vector<float> guards(note_order(p_prog).size(), 0.0f);
    for (size_t i = 0; i < p_prog.nodes.size(); i++) {
        if (p_band[i] >= 0 && p_prog.nodes[i].guard >= 0) {
            guards[size_t(p_band[i])] = float(biome_bit(p_prog.nodes[i].guard));
        }
    }
    t += "            uint ng = uint(" + r_params.table(guards, "nf + e") + ");\n";
    return t + "            if (ng == 0u || (on & ng) != 0u) {\n                " + call + "\n            } else {\n                nv[e] = vec4(0.0);\n            }\n        }\n";
}

ov::str specialised(const LevelProgram &p_prog, const GpuLayout &p_layout, ShaderParams &r_params) {
    const std::vector<int> band = band_index(p_prog);
    const std::vector<int64_t> curves = curve_records(p_prog, r_params);
    std::vector<int> slot;
    const ov::str on = on_word(p_prog, p_layout);
    const ov::str notes = note_loop(p_prog, band, p_layout, on, r_params, slot);
    ov::str loads, code, stores;
    for (int s = p_prog.top; s >= 0; s--) {
        const std::vector<int> &in = s < p_prog.top ? p_layout.chan[size_t(s + 1)] : std::vector<int>();
        if (!in.empty()) {
            loads += ov::format("    case %d:\n", s);
            for (size_t k = 0; k < in.size(); k++) {
                loads += "        " + reg(p_layout.reg[size_t(in[k])]) + " = " + ch_get(ov::itos(int64_t(k))) + ";\n";
            }
            loads += "        break;\n";
        }
        if (!p_prog.levels[size_t(s)].code.empty()) {
            code += ov::format("        case %d:\n", s);
            for (const int n : p_prog.levels[size_t(s)].code) {
                const ov::str expr = slot[size_t(n)] >= 0 ? ov::format("nv[%d]", int64_t(slot[size_t(n)])) : expression(p_prog, p_layout, n, band, curves, r_params);
                code += statement(p_prog, p_layout, n, expr, r_params);
            }
            code += "            break;\n";
        }
        const std::vector<int> &out = p_layout.chan[size_t(s)];
        if (!out.empty()) {
            stores += ov::format("        case %d:\n", s);
            for (size_t k = 0; k < out.size(); k++) {
                stores += "    " + ch_put(ov::itos(int64_t(k)), reg(p_layout.reg[size_t(out[k])]));
            }
            stores += "            break;\n";
        }
    }
    const ov::str empty = "    default:\n        break;\n";
    ov::str t = ov::format("\n// Terrain::level_run, specialised: level m's channels into registers, its code (a run's last level: levels m .. 0's), its\n"
                           "// channels out (the last level: the output, then the pick).\nCh level_run(Ch ch, WorldPos p, int m, bool last) {\n    vec4 r[%d];\n"
                           "    int lv = min(m, %d);\n",
            int64_t(p_layout.regs), int64_t(p_prog.top));
    if (!on.empty()) {
        t += "    uint mask = last ? 0u : biome_mask[g_item];\n";
    }
    t += "    switch (lv) {\n" + loads + empty + "    }\n";
    const ov::str stop = "int(" + r_params.table(last_stops(p_prog), ov::format("min(m, %d)", int64_t(Cascade::MAX_LEVEL))) + ")";
    t += "    for (int s = lv; s >= (last ? " + stop + " : lv); s--) {\n" + (notes.empty() ? on : notes) + "        switch (s) {\n" + code + "        default:\n            break;\n        }\n    }\n";
    const auto root = [&](int p_n) { return folded_value(p_prog.nodes[size_t(p_n)], reg(p_layout.reg[size_t(p_n)]), r_params); };
    t += "    Ch o = ch_zero();\n    if (last) {\n" + ch_put("0", root(p_prog.output)) + ov::format("        o.c[4] = %s.w;\n", root(p_prog.pick)) + "        return o;\n    }\n";
    return t + "    switch (lv) {\n" + stores + empty + "    }\n    return o;\n}\n";
}

// The generic interpreter's records (heap): level m's [inputs, outputs, instructions, first instruction, input registers,
// output registers] at the directory's entry m (values), each instruction [op, register, inputs, constants, aux0,
// aux1, const_lod, live_lod, folded, guard bit, mark bit, input registers, constants] (aux: a NOTE's band and level, a
// CURVE's record; the fold and the guard: Node's, as the specialised statement reads them; mark: a weight's bit).
// The floats of a generic record's head, before its input registers and constants (generic_records; the text gets it
// by substitution).
constexpr size_t RECORD_HEAD = 11;

struct GenericRecords {
    int64_t dir = 0, top = 0, output = 0, pick = 0, stop = 0, weights = 0;
};

GenericRecords generic_records(const LevelProgram &p_prog, const GpuLayout &p_layout, ShaderParams &r_params) {
    const std::vector<int> band = band_index(p_prog);
    const std::vector<int64_t> curves = curve_records(p_prog, r_params);
    std::vector<float> dir(size_t(Cascade::MAX_LEVEL + 1), 0.0f);
    for (int s = 0; s <= p_prog.top; s++) {
        std::vector<float> code;
        int64_t count = 0;
        for (const int n : p_prog.levels[size_t(s)].code) {
            const Node &node = p_prog.nodes[size_t(n)];
            const float aux0 = node.op == Op::NOTE ? float(band[size_t(n)]) : (node.op == Op::CURVE ? float(curves[size_t(node.curve)]) : 0.0f);
            const float head[] = { float(node.op), float(p_layout.reg[size_t(n)]), float(node.in.size()), float(node.k.size()), aux0, float(node.level), float(node.const_lod),
                float(node.live_lod), node.folded, node.guard >= 0 ? float(biome_bit(node.guard)) : 0.0f, float(node.mark) };
            static_assert(std::size(head) == RECORD_HEAD, "Terrain: a generic record's head is RECORD_HEAD floats.");
            code.insert(code.end(), std::begin(head), std::end(head));
            for (const int in : node.in) {
                code.push_back(float(p_layout.reg[size_t(in)]));
            }
            code.insert(code.end(), node.k.begin(), node.k.end());
            count++;
        }
        const int64_t first = r_params.put(code);
        const std::vector<int> &in = s < p_prog.top ? p_layout.chan[size_t(s + 1)] : std::vector<int>();
        const std::vector<int> &out = p_layout.chan[size_t(s)];
        std::vector<float> rec = { float(in.size()), float(out.size()), float(count), float(first) };
        for (const std::vector<int> *list : { &in, &out }) {
            for (const int n : *list) {
                rec.push_back(float(p_layout.reg[size_t(n)]));
            }
        }
        dir[size_t(s)] = float(r_params.put(rec));
    }
    GenericRecords g;
    g.dir = r_params.append(dir);
    g.top = r_params.append({ float(p_prog.top) });
    const auto root = [&](int p_n) {
        const Node &node = p_prog.nodes[size_t(p_n)];
        return r_params.append({ float(p_layout.reg[size_t(p_n)]), float(node.const_lod), node.folded }); // register, the fold
    };
    g.output = root(p_prog.output);
    g.pick = root(p_prog.pick);
    g.stop = r_params.append(last_stops(p_prog));
    // The on word's: count, then MAX_BIOMES registers (the unused ones 0), a fixed size like every appended list (the
    // text holds their offsets).
    std::vector<float> weights(size_t(Climate::MAX_BIOMES) + 1, 0.0f);
    for (size_t b = 0; b < p_prog.weights.size() && p_prog.weights.size() > 1; b++) {
        weights[0] = float(p_prog.weights.size());
        weights[b + 1] = float(p_layout.reg[size_t(p_prog.weights[b])]);
    }
    g.weights = r_params.append(weights);
    return g;
}

constexpr const char GENERIC_RUN[] = R"glsl(
// Terrain::level_run, generic: the program's records (ShaderParams' heap) interpreted, the same ops in the same order.
Ch level_run(Ch ch, WorldPos p, int m, bool last) {
    vec4 r[MAX_REGISTERS];
    int lv = min(m, int(cl_param(TOP)));
    int d = int(cl_param(DIR + lv));
    int nin = int(cl_heap(d));
    for (int k = 0; k < nin; k++) {
        r[int(cl_heap(d + 4 + k))] = CH_GET;
    }
    uint mask = last ? 0u : biome_mask[g_item];
    for (int s = lv; s >= (last ? int(cl_param(FINEST + min(m, MAX_LEVEL))) : lv); s--) {
        uint on = mask;
        if (last) {
            on = 0u;
            for (int b = 0; b < int(cl_param(WEIGHTS)); b++) {
                on |= r[int(cl_param(WEIGHTS + 1 + b))] != vec4(0.0) ? 1u << uint(b) : 0u;
            }
        }
        int e = int(cl_param(DIR + s));
        int q = int(cl_heap(e + 3));
        int count = int(cl_heap(e + 2));
        for (int i = 0; i < count; i++) {
            int op = int(cl_heap(q));
            int na = int(cl_heap(q + 2));
            int kq = q + RECORD_HEAD + na;
            int qn = kq + int(cl_heap(q + 3));
            uint gb = uint(cl_heap(q + 9));
            if (last && float(m) > cl_heap(q + 7)) {
                // unread on this last level
            } else if (gb != 0u && (on & gb) == 0u) {
                r[int(cl_heap(q + 1))] = vec4(0.0);
            } else if (float(m) >= cl_heap(q + 6)) {
                r[int(cl_heap(q + 1))] = vec4(0.0, 0.0, 0.0, cl_heap(q + 8));
            } else {
                vec4 a[3];
                for (int j = 0; j < 3; j++) {
                    a[j] = vec4(0.0);
                    if (j < na) {
                        a[j] = r[int(cl_heap(q + RECORD_HEAD + j))];
                    }
                }
                vec4 v = vec4(0.0);
                switch (op) {
CASES                default:
                    break;
                }
                r[int(cl_heap(q + 1))] = v;
            }
            uint mk = uint(cl_heap(q + 10));
            if (mk != 0u && !last && r[int(cl_heap(q + 1))] != vec4(0.0) && (biome_mask[g_item] & mk) != mk) {
                atomicOr(biome_mask[g_item], mk);
            }
            q = qn;
        }
    }
    Ch o = ch_zero();
    if (last) {
PUT_OUTPUT        o.c[4] = ROOT_PICK.w;
        return o;
    }
    int nout = int(cl_heap(d + 1));
    for (int k = 0; k < nout; k++) {
PUT_CHANNEL    }
    return o;
}
)glsl";

// A root's value in the generic text: its register, or its constant from its const_lod up (generic_records' root).
ov::str generic_root(const char *p_at) {
    return ov::replace_all("(float(m) >= cl_param(AT + 1) ? vec4(0.0, 0.0, 0.0, cl_param(AT + 2)) : r[int(cl_param(AT))])", "AT", p_at);
}

ov::str generic(const LevelProgram &p_prog, const GpuLayout &p_layout, ShaderParams &r_params) {
    const GenericRecords g = generic_records(p_prog, p_layout, r_params);
    const auto in = [](int p_i) { return ov::format("a[%d]", int64_t(p_i)); };
    const auto k = [](int p_i) { return ov::format("cl_heap(kq + %d)", int64_t(p_i)); };
    const ov::str in_j = "r[int(cl_heap(q + RECORD_HEAD + j))]";
    ov::str cases;
    for (int op = 0; op < int(Op::COUNT); op++) {
        const char *glsl = op_def(Op(op)).glsl;
        ov::str body;
        if (Op(op) == Op::LIN) {
            const ov::str t = ov::replace_all(glsl, { { "$i", in_j }, { "#i", "cl_heap(kq + j)" } });
            body = "                for (int j = 0; j < na; j++) {\n                    vec4 t = " + t + ";\n                    v = j == 0 ? t : cl_add(v, t);\n                }\n";
        } else if (Op(op) == Op::PICK) {
            body = ov::format("                vec4 w[%d];\n                for (int j = 0; j < %d; j++) {\n                    w[j] = vec4(0.0);\n", int64_t(Climate::MAX_BIOMES),
                    int64_t(Climate::MAX_BIOMES));
            body += "                    if (j < na) {\n                        w[j] = " + in_j + ";\n                    }\n                }\n";
            body += "                v = " + ov::replace_all(glsl, { { "$all", "w" }, { "#n", "na" }, { "@p", "p" } }) + ";\n";
        } else {
            body = "                v = " + form(glsl, in, k, "int(cl_heap(q + 5))", "bands[int(cl_heap(q + 4))]", "int(cl_heap(q + 4))") + ";\n";
        }
        cases += ov::format("            case %d: {\n", int64_t(op)) + body + "                break;\n            }\n";
    }
    ov::str t = ov::replace_all(GENERIC_RUN, { { "CASES", cases }, { "CH_GET", ch_get("k") }, { "PUT_OUTPUT", ch_put("0", generic_root("OUTPUT")) }, { "ROOT_PICK", generic_root("PICK") }, { "PUT_CHANNEL", ch_put("k", "r[int(cl_heap(d + 4 + nin + k))]") } });
    return ov::replace_all(t, { { "MAX_REGISTERS", ov::itos(MAX_REGISTERS) }, { "RECORD_HEAD", ov::itos(RECORD_HEAD) }, { "TOP", ov::itos(g.top) }, { "DIR", ov::itos(g.dir) }, { "OUTPUT", ov::itos(g.output) }, { "PICK", ov::itos(g.pick) }, { "FINEST", ov::itos(g.stop) }, { "WEIGHTS", ov::itos(g.weights) }, { "MAX_LEVEL", ov::itos(Cascade::MAX_LEVEL) } });
}

// Each node's first and last event over the stream (levels top .. 0: the inputs' loads, the code, the outputs' stores;
// then the last level's output and pick reads), and registers by linear scan over them.
void allocate(const LevelProgram &p_prog, GpuLayout &r_layout) {
    const size_t n = p_prog.nodes.size();
    std::vector<int> first(n, INT_MAX), last(n, -1);
    int pos = 0;
    const auto event = [&](int p_node, int p_at) {
        first[size_t(p_node)] = ov::min(first[size_t(p_node)], p_at);
        last[size_t(p_node)] = ov::max(last[size_t(p_node)], p_at);
    };
    for (int s = p_prog.top; s >= 0; s--) {
        for (const int in : s < p_prog.top ? r_layout.chan[size_t(s + 1)] : std::vector<int>()) {
            event(in, pos);
        }
        pos++;
        for (const int c : p_prog.levels[size_t(s)].code) {
            event(c, pos);
            for (const int in : p_prog.nodes[size_t(c)].in) {
                event(in, pos);
            }
            pos++;
        }
        for (const int out : r_layout.chan[size_t(s)]) {
            event(out, pos);
        }
        pos++;
    }
    event(p_prog.output, pos);
    event(p_prog.pick, pos);
    std::vector<int> order;
    for (size_t i = 0; i < n; i++) {
        if (last[i] >= 0) {
            order.push_back(int(i));
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](int p_a, int p_b) { return first[size_t(p_a)] < first[size_t(p_b)]; });
    r_layout.reg.assign(n, -1);
    std::vector<int> busy_until; // per register, its occupant's last event
    for (const int node : order) {
        int r = 0;
        while (r < int(busy_until.size()) && busy_until[size_t(r)] >= first[size_t(node)]) {
            r++;
        }
        if (r == int(busy_until.size())) {
            busy_until.push_back(0);
        }
        busy_until[size_t(r)] = last[size_t(node)];
        r_layout.reg[size_t(node)] = r;
    }
    r_layout.regs = ov::max(1, int(busy_until.size()));
}

// `int <p_name>(int m)`: p_values[m] for m = 0 .. MAX_LEVEL (a fixed-size table of r_params).
ov::str level_table(const char *p_name, const std::vector<float> &p_values, ShaderParams &r_params) {
    return ov::format("int %s(int m) {\n    return int(%s);\n}\n", p_name, r_params.table(p_values, ov::format("min(m, %d)", int64_t(Cascade::MAX_LEVEL))));
}

} // namespace

bool GpuLayout::pick_in(int p_m) const {
    return p_m + 1 < int(chan.size()) && !chan[size_t(p_m + 1)].empty() && chan[size_t(p_m + 1)][0] == pick_node;
}

int GpuLayout::inputs(int p_m) const {
    return p_m + 1 < int(chan.size()) ? int(chan[size_t(p_m + 1)].size()) : 0;
}

GpuLayout gpu_layout(const LevelProgram &p_prog, ov::str &r_error) {
    r_error = ov::str();
    GpuLayout l;
    l.pick_node = p_prog.nodes[size_t(p_prog.pick)].op == Op::PICK ? p_prog.pick : -1;
    l.chan.resize(size_t(p_prog.top + 1));
    for (int m = 0; m <= p_prog.top; m++) {
        std::vector<int> &c = l.chan[size_t(m)];
        c = p_prog.levels[size_t(m)].out;
        std::stable_partition(c.begin(), c.end(), [&](int p_n) { return p_n == l.pick_node; });
        l.channels = ov::max(l.channels, int(c.size()));
    }
    allocate(p_prog, l);
    if (p_prog.generic) {
        l.channels = MAX_CHANNELS;
        if (l.regs > MAX_REGISTERS) {
            r_error = ov::format("climate: the program needs %d registers, the generic text holds %d (fewer notes or biomes).", int64_t(l.regs), int64_t(MAX_REGISTERS));
        }
        l.regs = MAX_REGISTERS;
    }
    return l;
}

std::vector<int> note_order(const LevelProgram &p_prog) {
    std::vector<int> order;
    std::vector<bool> seen(p_prog.nodes.size(), false);
    const auto take = [&](int p_n) {
        if (!seen[size_t(p_n)] && p_prog.nodes[size_t(p_n)].op == Op::NOTE) {
            seen[size_t(p_n)] = true;
            order.push_back(p_n);
        }
    };
    for (int s = p_prog.top; s >= 0; s--) {
        for (const bool loop : { true, false }) {
            for (const int c : p_prog.levels[size_t(s)].code) {
                if (looped(p_prog, s, c) == loop) {
                    take(c);
                }
            }
        }
    }
    for (size_t i = 0; i < p_prog.nodes.size(); i++) {
        take(int(i));
    }
    return order;
}

ov::bytes band_table(const LevelProgram &p_prog) {
    using namespace WorldCoord;
    constexpr size_t BYTES = CascadeGLSL::BAND_BYTES;
    ov::bytes bytes(BYTES, 0);
    size_t at = 0;
    for (const int i : note_order(p_prog)) {
        const Node &n = p_prog.nodes[size_t(i)];
        bytes.resize(ov::max(bytes.size(), (at + 1) * BYTES), 0);
        uint8_t *slot = bytes.data() + at * BYTES;
        const Cascade::Band &b = n.band;
        for (int a = 0; a < 3; a++) {
            ov::put_u32(slot + 16 + 4 * a, lo32(b.offset[a]));
            ov::put_s32(slot + 32 + 4 * a, hi32(b.offset[a]));
        }
        ov::put_u32(slot + 48, 1);
        ov::put_u32(slot + 52, b.seed);
        ov::put_u32(slot + 56, uint32_t(b.k));
        ov::put_u32(slot + 60, uint32_t(b.stretch));
        at++;
    }
    return bytes;
}

ov::str level_flags_glsl(const LevelProgram &p_prog, const GpuLayout &p_layout, ShaderParams &r_params) {
    std::vector<float> in, pick, out;
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        in.push_back(m < p_prog.top ? float(p_layout.inputs(m)) : 0.0f);
        pick.push_back(m < p_prog.top && p_layout.pick_in(m) ? 1.0f : 0.0f);
        out.push_back(m <= p_prog.top ? float(p_layout.chan[size_t(m)].size()) : 0.0f);
    }
    ov::str code = "\n// Terrain::level_flags: what level m receives (g_in channels, the first the pick where g_pick) and stores (g_out).\n";
    code += "uint g_in = 0u;\nbool g_pick = false;\nuint g_out = 0u;\n\n";
    code += level_table("level_in", in, r_params) + level_table("level_pick", pick, r_params) + level_table("level_out", out, r_params);
    return code + "\nvoid level_flags(int m) {\n    g_in = uint(level_in(m));\n    g_pick = level_pick(m) != 0;\n    g_out = uint(level_out(m));\n}\n";
}

bool uses_biome_mask(const LevelProgram &p_prog) {
    return p_prog.generic || std::any_of(p_prog.nodes.begin(), p_prog.nodes.end(), [](const Node &p_n) { return p_n.guard >= 0; });
}

ov::str level_glsl(const LevelProgram &p_prog, const GpuLayout &p_layout, ShaderParams &r_params) {
    const ov::str mask = uses_biome_mask(p_prog)
            ? ov::format("\n// Terrain::guard_biomes: the run's biome mask per item (zeroed per batch), the item level_run reads it for.\n"
                         "layout(set = 0, binding = %d, std430) restrict buffer BiomeMask { uint biome_mask[]; };\nuint g_item = 0u;\n",
                      int64_t(CascadeGLSL::BIOME_MASK_BINDING))
            : ov::str();
    return mask + (p_prog.generic ? generic(p_prog, p_layout, r_params) : specialised(p_prog, p_layout, r_params));
}

} // namespace Terrain
