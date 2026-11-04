#include "level_glsl.hpp"

#include "climate.hpp"
#include "exact_noise.hpp"

#include "ov/format.hpp"
#include "ov/math.hpp"

#include <algorithm>
#include <climits>

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

// Each NOTE's entry in band_table.
std::vector<int> band_index(const LevelProgram &p_prog) {
    std::vector<int> index(p_prog.nodes.size(), -1);
    int n = 0;
    for (size_t i = 0; i < p_prog.nodes.size(); i++) {
        index[i] = p_prog.nodes[i].op == Op::NOTE ? n++ : -1;
    }
    return index;
}

// Node p_n's statement in the specialised text.
ov::str statement(const LevelProgram &p_prog, const GpuLayout &p_layout, int p_n, const std::vector<int> &p_band, const std::vector<int64_t> &p_curve,
        ShaderParams &r_params) {
    const Node &node = p_prog.nodes[size_t(p_n)];
    const ov::str dst = reg(p_layout.reg[size_t(p_n)]);
    const auto in = [&](int p_i) { return p_i < int(node.in.size()) ? reg(p_layout.reg[size_t(node.in[size_t(p_i)])]) : ov::str("vec4(0.0)"); };
    const auto k = [&](int p_i) { return p_i < int(node.k.size()) ? r_params(node.k[size_t(p_i)]) : ov::str("0.0"); };
    const char *glsl = op_def(node.op).glsl;
    if (node.op == Op::LIN) {
        ov::str sum;
        for (size_t i = 0; i < node.in.size(); i++) {
            const ov::str t = ov::replace_all(glsl, { { "$i", in(int(i)) }, { "#i", k(int(i)) } });
            sum = i == 0 ? t : "cl_add(" + sum + ", " + t + ")";
        }
        return "            " + dst + " = " + (sum.empty() ? ov::str("vec4(0.0)") : sum) + ";\n";
    }
    if (node.op == Op::PICK) {
        ov::str all = ov::format("vec4[%d](", int64_t(Climate::MAX_BIOMES));
        for (int b = 0; b < Climate::MAX_BIOMES; b++) {
            all += (b ? ", " : "") + (b < int(node.in.size()) ? in(b) : ov::str("vec4(0.0)"));
        }
        return "            " + dst + " = " + ov::replace_all(glsl, { { "$all", all + ")" }, { "#n", ov::itos(int64_t(node.in.size())) }, { "@p", "p" } }) + ";\n";
    }
    const ov::str band = node.op == Op::NOTE ? ov::format("bands[%d]", int64_t(p_band[size_t(p_n)])) : ov::str();
    const ov::str curve = node.op == Op::CURVE ? ov::itos(p_curve[size_t(node.curve)]) : ov::str();
    return "            " + dst + " = " + form(glsl, in, k, ov::itos(node.level), band, curve) + ";\n";
}

ov::str specialised(const LevelProgram &p_prog, const GpuLayout &p_layout, ShaderParams &r_params) {
    const std::vector<int> band = band_index(p_prog);
    const std::vector<int64_t> curves = curve_records(p_prog, r_params);
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
                code += statement(p_prog, p_layout, n, band, curves, r_params);
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
    t += "    switch (lv) {\n" + loads + empty + "    }\n";
    t += "    for (int s = lv; s >= (last ? 0 : lv); s--) {\n        switch (s) {\n" + code + "        default:\n            break;\n        }\n    }\n";
    t += "    Ch o = ch_zero();\n    if (last) {\n" + ch_put("0", reg(p_layout.reg[size_t(p_prog.output)])) +
            ov::format("        o.c[4] = %s.w;\n", reg(p_layout.reg[size_t(p_prog.pick)])) + "        return o;\n    }\n";
    return t + "    switch (lv) {\n" + stores + empty + "    }\n    return o;\n}\n";
}

// The generic interpreter's records (heap): level m's [inputs, outputs, instructions, first instruction, input registers,
// output registers] at the directory's entry m (values), each instruction [op, register, inputs, constants, aux0,
// aux1, input registers, constants] (aux: a NOTE's band and level, a CURVE's record).
struct GenericRecords {
    int64_t dir = 0, top = 0, output = 0, pick = 0;
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
            code.insert(code.end(), { float(node.op), float(p_layout.reg[size_t(n)]), float(node.in.size()), float(node.k.size()), aux0, float(ov::min(node.level, 1 << 20)) });
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
    g.output = r_params.append({ float(p_layout.reg[size_t(p_prog.output)]) });
    g.pick = r_params.append({ float(p_layout.reg[size_t(p_prog.pick)]) });
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
    for (int s = lv; s >= (last ? 0 : lv); s--) {
        int e = int(cl_param(DIR + s));
        int q = int(cl_heap(e + 3));
        int count = int(cl_heap(e + 2));
        for (int i = 0; i < count; i++) {
            int op = int(cl_heap(q));
            int na = int(cl_heap(q + 2));
            int kq = q + 6 + na;
            vec4 a[3];
            for (int j = 0; j < 3; j++) {
                a[j] = vec4(0.0);
                if (j < na) {
                    a[j] = r[int(cl_heap(q + 6 + j))];
                }
            }
            vec4 v = vec4(0.0);
            switch (op) {
CASES            default:
                break;
            }
            r[int(cl_heap(q + 1))] = v;
            q = kq + int(cl_heap(q + 3));
        }
    }
    Ch o = ch_zero();
    if (last) {
PUT_OUTPUT        o.c[4] = r[int(cl_param(PICK))].w;
        return o;
    }
    int nout = int(cl_heap(d + 1));
    for (int k = 0; k < nout; k++) {
PUT_CHANNEL    }
    return o;
}
)glsl";

ov::str generic(const LevelProgram &p_prog, const GpuLayout &p_layout, ShaderParams &r_params) {
    const GenericRecords g = generic_records(p_prog, p_layout, r_params);
    const auto in = [](int p_i) { return ov::format("a[%d]", int64_t(p_i)); };
    const auto k = [](int p_i) { return ov::format("cl_heap(kq + %d)", int64_t(p_i)); };
    const ov::str in_j = "r[int(cl_heap(q + 6 + j))]";
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
    ov::str t = ov::replace_all(GENERIC_RUN, { { "CASES", cases }, { "CH_GET", ch_get("k") }, { "PUT_OUTPUT", ch_put("0", "r[int(cl_param(OUTPUT))]") }, { "PUT_CHANNEL", ch_put("k", "r[int(cl_heap(d + 4 + nin + k))]") } });
    return ov::replace_all(t, { { "MAX_REGISTERS", ov::itos(MAX_REGISTERS) }, { "TOP", ov::itos(g.top) }, { "DIR", ov::itos(g.dir) }, { "OUTPUT", ov::itos(g.output) }, { "PICK", ov::itos(g.pick) } });
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

ov::bytes band_table(const LevelProgram &p_prog) {
    using namespace WorldCoord;
    constexpr size_t BYTES = 64; // CascadeGLSL::BAND_BYTES
    ov::bytes bytes(BYTES, 0);
    size_t at = 0;
    for (const Node &n : p_prog.nodes) {
        if (n.op != Op::NOTE) {
            continue;
        }
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

ov::str level_glsl(const LevelProgram &p_prog, const GpuLayout &p_layout, ShaderParams &r_params) {
    return p_prog.generic ? generic(p_prog, p_layout, r_params) : specialised(p_prog, p_layout, r_params);
}

} // namespace Terrain
