#include "climate.hpp"

#include "ov/math.hpp"
#include "ov/str.hpp"

#include <algorithm>
#include <cmath>

// Band shapes' bounds, the biomes' curves, their sample programs and the ground at a sample (ground_value), on the CPU
// and in generated GLSL, op for op: every product flushed under 2^-64 like the climate's dual numbers (climate.hpp's
// exactness note).
namespace Climate {

using Cascade::dadd;
using Cascade::dflush;
using Cascade::dmul;
using Cascade::dscale;
using Cascade::dsub;
using Cascade::flush_value;

double shape_phi(const Shape &p_shape, double p_n) {
    const double sabs = std::sqrt(p_n * p_n + double(Cascade::SMOOTH_ABS_E) * Cascade::SMOOTH_ABS_E) - Cascade::SMOOTH_ABS_E;
    return double(p_shape.c[0]) + double(p_shape.c[1]) * p_n + double(p_shape.c[2]) * sabs + double(p_shape.c[3]) * p_n * p_n;
}

float shape_bound(const Shape &p_shape) {
    double m = 0.0;
    const int steps = 2000;
    for (int i = 0; i <= steps; i++) {
        const double n = -1.0 + 2.0 * i / steps;
        const double sabs = std::sqrt(n * n + double(Cascade::SMOOTH_ABS_E) * Cascade::SMOOTH_ABS_E) - Cascade::SMOOTH_ABS_E;
        const double v = double(p_shape.c[1]) * n + double(p_shape.c[2]) * (sabs - Cascade::NOISE_ABS_MEAN) +
                double(p_shape.c[3]) * (n * n - Cascade::NOISE_SQ_MEAN);
        m = std::max(m, std::fabs(v));
    }
    return float(m);
}

Curve make_curve(const std::vector<double> &p_x, const std::vector<double> &p_y) {
    Curve c;
    const int n = int(p_x.size());
    auto y = [&](int p_i) { return p_y[size_t(ov::clamp(p_i, 0, n - 1))]; };
    for (int i = 0; i < n; i++) {
        c.x.push_back(float(p_x[size_t(i)]));
    }
    // libnoise's CubicInterp(n0, n1, n2, n3, a) on segment i: n0 .. n3 = y_{i-1} .. y_{i+2}, ends clamped.
    for (int i = 0; i + 1 < n; i++) {
        const double n0 = y(i - 1), n1 = y(i), n2 = y(i + 1), n3 = y(i + 2);
        const double p = (n3 - n2) - (n0 - n1);
        const double q = (n0 - n1) - p;
        c.inv.push_back(float(1.0 / (p_x[size_t(i + 1)] - p_x[size_t(i)])));
        c.p.push_back(float(p));
        c.q.push_back(float(q));
        c.r.push_back(float(n2 - n0));
        c.s.push_back(float(n1));
    }
    c.y_last = float(p_y.back());
    return c;
}

float curve_slope_bound(const Curve &p_curve) {
    double m = 0.0;
    for (size_t i = 0; i < p_curve.p.size(); i++) {
        for (int j = 0; j <= 64; j++) {
            const double a = j / 64.0;
            const double d = ((3.0 * p_curve.p[i] * a + 2.0 * p_curve.q[i]) * a + p_curve.r[i]) * p_curve.inv[i];
            m = std::max(m, std::fabs(d));
        }
    }
    return float(m);
}

float curve_bound(const Curve &p_curve) {
    double m = std::fabs(double(p_curve.y_last));
    for (size_t i = 0; i < p_curve.p.size(); i++) {
        for (int j = 0; j <= 64; j++) {
            const double a = j / 64.0;
            m = std::max(m, std::fabs(((p_curve.p[i] * a + p_curve.q[i]) * a + p_curve.r[i]) * a + p_curve.s[i]));
        }
    }
    return float(m);
}

bool Table::has_programs() const {
    return (generic && !biomes.empty()) || std::any_of(biomes.begin(), biomes.end(), [](const Biome &p_b) { return !p_b.program.empty(); });
}

int cap_for(int p_need, int p_floor, int p_max) {
    int c = 1;
    while (c < p_need) {
        c *= 2;
    }
    return ov::max(ov::min(ov::max(p_need > 0 ? c : 0, p_floor), p_max), p_need);
}

bool SampleProgram::reads_field(int p_field) const {
    return std::any_of(code.begin(), code.end(), [&](const SampleInstr &p_i) { return p_i.op == SampleOp::FIELD && p_i.index == p_field; });
}

bool SampleProgram::reads_up() const {
    return std::any_of(code.begin(), code.end(), [](const SampleInstr &p_i) { return p_i.op == SampleOp::UP; });
}

// The curve's value and slope at x (constant past the ends), the slope applied to x's gradient.
Float4 eval_curve(const Curve &p_c, const Float4 &p_x) {
    const size_t n = p_c.x.size();
    if (!(p_x.w > p_c.x[0])) {
        return Float4{ 0.0f, 0.0f, 0.0f, p_c.s[0] };
    }
    if (p_x.w >= p_c.x[n - 1]) {
        return Float4{ 0.0f, 0.0f, 0.0f, p_c.y_last };
    }
    size_t i = 0;
    while (i + 2 < n && p_x.w >= p_c.x[i + 1]) {
        i++;
    }
    const float a = flush_value((p_x.w - p_c.x[i]) * p_c.inv[i]);
    float v = flush_value(p_c.p[i] * a) + p_c.q[i];
    v = flush_value(v * a) + p_c.r[i];
    v = flush_value(v * a) + p_c.s[i];
    float d = flush_value((p_c.p[i] * 3.0f) * a) + p_c.q[i] * 2.0f;
    d = flush_value(d * a) + p_c.r[i];
    d = flush_value(d * p_c.inv[i]);
    return dflush(Float4{ p_x.x * d, p_x.y * d, p_x.z * d, v });
}

// Quilez's polynomial smooth min of p_a and p_b over p_k voxels (p_inv = 0.5 / p_k): h = clamp(1/2 + (b - a) / (2k), 0,
// 1), b + h (a - b) - k h (1 - h); outside the blend the smaller register itself. Not the archetype IR's smooth_min4: the
// climate's products flush under 2^-64 (its exactness note). Its dip, k / 4 at a = b, is biome_compiler.cpp's bound.
Float4 smooth_min(const Float4 &p_a, const Float4 &p_b, float p_k, float p_inv) {
    const Float4 t = dscale(dsub(p_b, p_a), p_inv);
    const float hw = 0.5f + t.w;
    if (!(hw > 0.0f)) {
        return p_b;
    }
    if (hw >= 1.0f) {
        return p_a;
    }
    const Float4 h{ t.x, t.y, t.z, hw };
    const Float4 one_h{ -t.x, -t.y, -t.z, 1.0f - hw };
    const Float4 m = dadd(p_b, dmul(h, dsub(p_a, p_b)));
    return dsub(m, dscale(dmul(h, one_h), p_k));
}

namespace {

// smooth_min in GLSL (after the shared helpers).
constexpr const char GLSL_SMOOTH_MIN[] = R"glsl(
vec4 cl_smin(vec4 a, vec4 b, float k, float inv) {
    vec4 t = cl_scale(cl_sub(b, a), inv);
    precise float hw = 0.5 + t.w;
    if (!(hw > 0.0)) {
        return b;
    }
    if (hw >= 1.0) {
        return a;
    }
    vec4 h = vec4(t.x, t.y, t.z, hw);
    precise float omh = 1.0 - hw;
    vec4 one_h = vec4(-t.x, -t.y, -t.z, omh);
    vec4 m = cl_add(b, cl_mul(h, cl_sub(a, b)));
    return cl_sub(m, cl_scale(cl_mul(h, one_h), k));
}
)glsl";

// The negation of a dual (exact).
Float4 neg(const Float4 &p_a) {
    return Float4{ -p_a.x, -p_a.y, -p_a.z, -p_a.w };
}

// A smooth op's 0.5 / k.
float half_inv(float p_k) {
    return float(0.5 / double(p_k));
}

// SampleOp::SHAPE: p_s's phi of p_a, each term where its coefficient isn't 0.
Float4 shape_value(const Shape &p_s, const Float4 &p_a) {
    Float4 r{ 0.0f, 0.0f, 0.0f, p_s.c[0] };
    if (p_s.c[1] != 0.0f) {
        r = dadd(r, dscale(p_a, p_s.c[1]));
    }
    if (p_s.c[2] != 0.0f) {
        r = dadd(r, dscale(p_a.w < 0.0f ? neg(p_a) : p_a, p_s.c[2]));
    }
    if (p_s.c[3] != 0.0f) {
        r = dadd(r, dscale(dmul(p_a, p_a), p_s.c[3]));
    }
    return r;
}

// shape_value in GLSL, the coefficients c (c[0] .. c[3] as x .. w).
constexpr const char GLSL_SHAPE[] = R"glsl(
vec4 cl_shape(vec4 a, vec4 c) {
    vec4 r = vec4(0.0, 0.0, 0.0, c.x);
    if (c.y != 0.0) {
        r = cl_add(r, cl_scale(a, c.y));
    }
    if (c.z != 0.0) {
        r = cl_add(r, cl_scale(a.w < 0.0 ? -a : a, c.z));
    }
    if (c.w != 0.0) {
        r = cl_add(r, cl_scale(cl_mul(a, a), c.w));
    }
    return r;
}
)glsl";

// Biome program p_p's term D at the ground p_ground (H), the up p_up and the point's slots p_slots (its carried fields,
// p_carry).
Float4 run_program(const SampleProgram &p_p, const Float4 &p_ground, const Float4 &p_up, const float p_slots[Cascade::SLOTS], const SampleCarry &p_carry) {
    if (p_p.curve_of_ground()) {
        return eval_curve(p_p.curves[0], p_ground);
    }
    Float4 r[MAX_SAMPLE_INSTRS];
    for (size_t i = 0; i < p_p.code.size(); i++) {
        const SampleInstr &in = p_p.code[i];
        const Float4 a = in.a >= 0 ? r[size_t(in.a)] : Float4{};
        const Float4 b = in.b >= 0 ? r[size_t(in.b)] : Float4{};
        switch (in.op) {
            case SampleOp::GROUND:
                r[i] = p_ground;
                break;
            case SampleOp::CONST:
                r[i] = Float4{ 0.0f, 0.0f, 0.0f, in.value };
                break;
            case SampleOp::FIELD:
                r[i] = to_float4(p_slots + Cascade::carry_slot(p_carry.field[in.index]));
                break;
            case SampleOp::ADD:
                r[i] = dadd(a, b);
                break;
            case SampleOp::SUB:
                r[i] = dsub(a, b);
                break;
            case SampleOp::MUL:
                r[i] = dmul(a, b);
                break;
            case SampleOp::MIN:
                r[i] = b.w < a.w ? b : a;
                break;
            case SampleOp::MAX:
                r[i] = b.w > a.w ? b : a;
                break;
            case SampleOp::SMIN:
                r[i] = smooth_min(a, b, in.value, half_inv(in.value));
                break;
            case SampleOp::SMAX:
                r[i] = neg(smooth_min(neg(a), neg(b), in.value, half_inv(in.value)));
                break;
            case SampleOp::CURVE:
                r[i] = dadd(a, eval_curve(p_p.curves[size_t(in.index)], a));
                break;
            case SampleOp::SHAPE:
                r[i] = shape_value(p_p.shapes[size_t(in.index)], a);
                break;
            case SampleOp::UP:
                r[i] = p_up;
                break;
        }
    }
    return dsub(r[p_p.code.size() - 1], p_ground);
}

// eval_curve's arithmetic once its segment is chosen (x0, inv, p, q, r, s declared), returning the curve at x.
ov::str curve_tail() {
    using Cascade::glsl_flush_of;
    ov::str code = "    precise float a0 = (x.w - x0) * inv;\n    precise float a = (" + glsl_flush_of("a0") + ov::str(");\n");
    code += "    precise float t0 = p * a;\n    precise float v = (" + glsl_flush_of("t0") + ov::str(") + q;\n");
    code += "    precise float t1 = v * a;\n    v = (" + glsl_flush_of("t1") + ov::str(") + r;\n");
    code += "    precise float t2 = v * a;\n    v = (" + glsl_flush_of("t2") + ov::str(") + s;\n");
    code += "    precise float u0 = (p * 3.0) * a;\n    precise float d = (" + glsl_flush_of("u0") + ov::str(") + q * 2.0;\n");
    code += "    precise float u1 = d * a;\n    d = (" + glsl_flush_of("u1") + ov::str(") + r;\n");
    code += "    precise float u2 = d * inv;\n    d = (" + glsl_flush_of("u2") + ov::str(");\n");
    return code + "    precise vec4 res = vec4(x.x * d, x.y * d, x.z * d, v);\n    return cl_flush(res);\n}\n";
}

// eval_curve as `vec4 <p_name>(vec4 x)`, the segment chosen by the same compares.
ov::str curve_glsl(const Curve &p_c, const ov::str &p_name, Cascade::ShaderParams &r_params) {
    const size_t n = p_c.x.size();
    ov::str code = "vec4 " + p_name + "(vec4 x) {\n";
    code += ov::format("    if (!(x.w > %s)) {\n        return vec4(0.0, 0.0, 0.0, %s);\n    }\n", r_params(p_c.x[0]), r_params(p_c.s[0]));
    code += ov::format("    if (x.w >= %s) {\n        return vec4(0.0, 0.0, 0.0, %s);\n    }\n", r_params(p_c.x[n - 1]), r_params(p_c.y_last));
    code += "    float x0, inv, p, q, r, s;\n";
    for (size_t i = 0; i + 1 < n; i++) {
        const bool last = i + 2 >= n;
        code += ov::str(i == 0 ? "    " : " else ") + (last ? ov::str("{") : ov::format("if (x.w < %s) {", r_params(p_c.x[i + 1])));
        code += ov::format(" x0 = %s; inv = %s; p = %s; q = %s; r = %s; s = %s; }", r_params(p_c.x[i]), r_params(p_c.inv[i]), r_params(p_c.p[i]),
                r_params(p_c.q[i]), r_params(p_c.r[i]), r_params(p_c.s[i]));
    }
    return code + "\n" + curve_tail();
}

// eval_curve over a curve's record in the params (a generic text's: n, y_last, x[n], then inv, p, q, r, s per
// segment), the segment chosen by curve_glsl's compares in a loop.
constexpr const char GLSL_CURVE_AT[] = R"glsl(
vec4 cl_curve_at(int cb, vec4 x) {
    int n = int(cl_heap(cb));
    int xs = cb + 2;
    int seg = xs + n;
    if (!(x.w > cl_heap(xs))) {
        return vec4(0.0, 0.0, 0.0, cl_heap(seg + 4));
    }
    if (x.w >= cl_heap(xs + n - 1)) {
        return vec4(0.0, 0.0, 0.0, cl_heap(cb + 1));
    }
    int i = 0;
    while (i + 2 < n && x.w >= cl_heap(xs + i + 1)) {
        i++;
    }
    int k = seg + 5 * i;
    float x0 = cl_heap(xs + i);
    float inv = cl_heap(k);
    float p = cl_heap(k + 1);
    float q = cl_heap(k + 2);
    float r = cl_heap(k + 3);
    float s = cl_heap(k + 4);
)glsl";

// A sample op's value (run_program's, op for op) as a GLSL expression of its registers p_a and p_b and its operands:
// p_value a constant's value or a smooth op's k, p_inv its 0.5 / k, p_field the carried field's read, p_curve the
// curve's call on p_a, p_shape a shape's coefficients (a vec4).
ov::str op_expr(SampleOp p_op, const ov::str &p_a, const ov::str &p_b, const ov::str &p_value, const ov::str &p_inv, const ov::str &p_field, const ov::str &p_curve,
        const ov::str &p_shape) {
    switch (p_op) {
        case SampleOp::GROUND:
            return "base";
        case SampleOp::CONST:
            return "vec4(0.0, 0.0, 0.0, " + p_value + ")";
        case SampleOp::FIELD:
            return p_field;
        case SampleOp::ADD:
            return "cl_add(" + p_a + ", " + p_b + ")";
        case SampleOp::SUB:
            return "cl_sub(" + p_a + ", " + p_b + ")";
        case SampleOp::MUL:
            return "cl_mul(" + p_a + ", " + p_b + ")";
        case SampleOp::MIN:
            return p_b + ".w < " + p_a + ".w ? " + p_b + " : " + p_a;
        case SampleOp::MAX:
            return p_b + ".w > " + p_a + ".w ? " + p_b + " : " + p_a;
        case SampleOp::SMIN:
            return "cl_smin(" + p_a + ", " + p_b + ", " + p_value + ", " + p_inv + ")";
        case SampleOp::SMAX:
            return "-cl_smin(-" + p_a + ", -" + p_b + ", " + p_value + ", " + p_inv + ")";
        case SampleOp::CURVE:
            return "cl_add(" + p_a + ", " + p_curve + ")";
        case SampleOp::SHAPE:
            return "cl_shape(" + p_a + ", " + p_shape + ")";
        case SampleOp::UP:
            return "up";
    }
    return ov::str();
}

// Biome p_b's program as `vec4 cl_sample<p_b>(vec4 base, vec4 up, Ch ch)`, D: run_program's ops (its curves cl_curve<b>_<i>
// before it); a curve of H alone is cl_curve<b> itself (its text before programs).
ov::str program_glsl(const SampleProgram &p_p, int p_b, const Cascade::Spec &p_spec, const SampleCarry &p_carry, Cascade::ShaderParams &r_params) {
    if (p_p.curve_of_ground()) {
        return curve_glsl(p_p.curves[0], ov::format("cl_curve%d", int64_t(p_b)), r_params);
    }
    ov::str code;
    for (size_t c = 0; c < p_p.curves.size(); c++) {
        code += curve_glsl(p_p.curves[c], ov::format("cl_curve%d_%d", int64_t(p_b), int64_t(c)), r_params);
    }
    code += ov::format("vec4 cl_sample%d(vec4 base, vec4 up, Ch ch) {\n", int64_t(p_b));
    for (size_t i = 0; i < p_p.code.size(); i++) {
        const SampleInstr &in = p_p.code[i];
        const ov::str a = ov::format("r%d", int64_t(in.a)), b = ov::format("r%d", int64_t(in.b));
        const bool k = in.op == SampleOp::SMIN || in.op == SampleOp::SMAX;
        const ov::str value = in.op == SampleOp::CONST || k ? r_params(in.value) : ov::str();
        const ov::str inv = k ? r_params(half_inv(in.value)) : ov::str();
        const ov::str field = in.op == SampleOp::FIELD ? carried_glsl(p_spec, "ch", p_carry.field[in.index]) : ov::str();
        const ov::str curve = in.op == SampleOp::CURVE ? ov::format("cl_curve%d_%d(%s)", int64_t(p_b), int64_t(in.index), a) : ov::str();
        ov::str shape;
        if (in.op == SampleOp::SHAPE) {
            const Shape &sh = p_p.shapes[size_t(in.index)];
            shape = ov::format("vec4(%s, %s, %s, %s)", r_params(sh.c[0]), r_params(sh.c[1]), r_params(sh.c[2]), r_params(sh.c[3]));
        }
        code += ov::format("    vec4 r%d = ", int64_t(i)) + op_expr(in.op, a, b, value, inv, field, curve, shape) + ";\n";
    }
    return code + ov::format("    return cl_sub(r%d, base);\n}\n", int64_t(p_p.code.size() - 1));
}

// The generic text's interpreter (Table::generic): `vec4 cl_run(int pb, vec4 base, vec4 up, Ch ch)`, run_program over the
// program record at pb in the params (its instruction count, or -1 for a curve of H alone and that curve's record;
// then op, a, b, value, aux per instruction: aux a smooth op's 0.5 / k, a field's carry group, a curve's or a shape's
// record), each op op_expr's.
ov::str interpreter_glsl(const Cascade::Spec &p_spec) {
    ov::str code = GLSL_CURVE_AT + curve_tail();
    code += "\nvec4 cl_run(int pb, vec4 base, vec4 up, Ch ch) {\n    int n = int(cl_heap(pb));\n    if (n < 0) {\n        return cl_curve_at(int(cl_heap(pb + 1)), base);\n    }\n";
    code += ov::format("    vec4 r[%d];\n    for (int i = 0; i < n; i++) {\n        int q = pb + 1 + 5 * i;\n", MAX_SAMPLE_INSTRS);
    code += "        vec4 a = r[max(int(cl_heap(q + 1)), 0)];\n        vec4 b = r[max(int(cl_heap(q + 2)), 0)];\n        float aux = cl_heap(q + 4);\n";
    code += ov::format("        int fc = %d + %d * int(aux);\n        vec4 v;\n        switch (int(cl_heap(q))) {\n", p_spec.gpu_carry(0), Cascade::FIELD_CHANNELS);
    for (int op = int(SampleOp::GROUND); op <= int(SampleOp::UP); op++) {
        const ov::str e = op_expr(SampleOp(op), "a", "b", "cl_heap(q + 3)", "aux", "vec4(ch.c[fc + 1], ch.c[fc + 2], ch.c[fc + 3], ch.c[fc])", "cl_curve_at(int(aux), a)",
                "vec4(cl_heap(int(aux)), cl_heap(int(aux) + 1), cl_heap(int(aux) + 2), cl_heap(int(aux) + 3))");
        code += ov::format("        case %d: v = %s; break;\n", op, e);
    }
    return code + "        default: v = vec4(0.0); break;\n        }\n        r[i] = v;\n    }\n    return cl_sub(r[n - 1], base);\n}\n";
}

// The programs' records (interpreter_glsl's), biome b's at the index the directory's entry b holds; the directory's index.
int64_t program_records(const Table &p_table, const SampleCarry &p_carry, Cascade::ShaderParams &r_params) {
    std::vector<float> dir(p_table.biomes.size());
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        const SampleProgram &p = p_table.biomes[b].program;
        std::vector<int64_t> curves;
        for (const Curve &c : p.curves) {
            curves.push_back(int64_t(r_params.heap.size()));
            std::vector<float> rec = { float(c.x.size()), c.y_last };
            rec.insert(rec.end(), c.x.begin(), c.x.end());
            for (size_t i = 0; i < c.inv.size(); i++) {
                rec.insert(rec.end(), { c.inv[i], c.p[i], c.q[i], c.r[i], c.s[i] });
            }
            r_params.put(rec);
        }
        std::vector<int64_t> shapes;
        for (const Shape &sh : p.shapes) {
            shapes.push_back(r_params.put({ sh.c[0], sh.c[1], sh.c[2], sh.c[3] }));
        }
        dir[b] = float(r_params.heap.size());
        if (p.curve_of_ground()) {
            r_params.put({ -1.0f, float(curves[0]) });
            continue;
        }
        std::vector<float> rec = { float(p.code.size()) };
        for (const SampleInstr &in : p.code) {
            const bool k = in.op == SampleOp::SMIN || in.op == SampleOp::SMAX;
            float aux = k ? half_inv(in.value) : 0.0f;
            aux = in.op == SampleOp::FIELD ? float(p_carry.field[in.index]) : aux;
            aux = in.op == SampleOp::CURVE ? float(curves[size_t(in.index)]) : aux;
            aux = in.op == SampleOp::SHAPE ? float(shapes[size_t(in.index)]) : aux;
            rec.insert(rec.end(), { float(in.op), float(in.a), float(in.b), in.value, aux });
        }
        r_params.put(rec);
    }
    return r_params.append(dir);
}

} // namespace

ov::str smooth_min_glsl() {
    return GLSL_SMOOTH_MIN;
}

ov::str curve_at_glsl() {
    return ov::replace_all(ov::str(GLSL_CURVE_AT) + curve_tail(), "FLUSH", Cascade::glsl_flush());
}

Float4 up_value(const Float4 &p_base, int p_lod) {
    return Cascade::dscale(p_base, ExactNoise::exact_pow2(p_lod));
}

Float4 ground_value(const Table &p_table, const SampleCarry &carry, const float p_slots[Cascade::SLOTS], const Float4 &p_up) {
    const Float4 base = to_float4(p_slots);
    Float4 h = base;
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (carry.weight[b] >= 0 && !p_table.biomes[b].program.empty()) {
            const Float4 w = to_float4(p_slots + Cascade::carry_slot(carry.weight[b]));
            h = dadd(h, dmul(w, run_program(p_table.biomes[b].program, base, p_up, p_slots, carry)));
        }
    }
    return h;
}

ov::str ground_glsl(const Table &p_table, const Cascade::Spec &p_combined, Cascade::ShaderParams &r_params) {
    const SampleCarry carry = sample_carry(p_table);
    ov::str code;
    bool smooth = p_table.generic, shape = p_table.generic, up = false;
    for (const Biome &b : p_table.biomes) {
        for (const SampleInstr &in : b.program.code) {
            smooth = smooth || in.op == SampleOp::SMIN || in.op == SampleOp::SMAX;
            shape = shape || in.op == SampleOp::SHAPE;
            up = up || in.op == SampleOp::UP;
        }
    }
    code += smooth ? ov::str(GLSL_SMOOTH_MIN) : ov::str();
    code += shape ? ov::str(GLSL_SHAPE) : ov::str();
    const int64_t dir = p_table.generic ? program_records(p_table, carry, r_params) : -1;
    if (p_table.generic) {
        code += ov::replace_all(interpreter_glsl(p_combined), "FLUSH", Cascade::glsl_flush());
    }
    for (size_t b = 0; b < p_table.biomes.size() && !p_table.generic; b++) {
        if (carry.weight[b] >= 0) {
            code += ov::replace_all(program_glsl(p_table.biomes[b].program, int(b), p_combined, carry, r_params), "FLUSH", Cascade::glsl_flush());
        }
    }
    code += "\n// Climate::ground_value: the cascade's ground h, plus each curved biome's weight times its curve of h; (value, gradient).\n";
    code += "vec4 climate_ground(Ch ch, WorldPos p, int lod) {\n";
    code += "    vec4 base = vec4(ch.c[1], ch.c[2], ch.c[3], ch.c[0]);\n    vec4 h = base;\n";
    const ov::str up_value = "cl_scale(base_at(p, lod), exact_pow2(lod))"; // up_value, where a program reads UP (a generic text: a param says)
    code += "    vec4 up = " + (p_table.generic ? ov::format("cl_param(%d) != 0.0 ? %s : vec4(0.0)", r_params.append({ up ? 1.0f : 0.0f }), up_value) : (up ? up_value : ov::str("vec4(0.0)"))) + ";\n";
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (carry.weight[b] < 0) {
            continue;
        }
        const ov::str w = carried_glsl(p_combined, "ch", carry.weight[b]);
        if (p_table.generic) {
            code += ov::format("    { int pb = int(cl_param(%d)); if (int(cl_heap(pb)) != 0) { h = cl_add(h, cl_mul(%s, cl_run(pb, base, up, ch))); } }\n", dir + int64_t(b), w);
            continue;
        }
        const ov::str term = p_table.biomes[b].program.curve_of_ground() ? ov::format("cl_curve%d(base)", int64_t(b)) : ov::format("cl_sample%d(base, up, ch)", int64_t(b));
        code += ov::format("    h = cl_add(h, cl_mul(%s, %s));\n", w, term);
    }
    code += "    return vec4(h.w, h.x, h.y, h.z);\n}\n";
    return code;
}

} // namespace Climate
