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

bool Table::has_state() const {
    return std::any_of(biomes.begin(), biomes.end(), [&](const Biome &p_b) {
        return std::any_of(p_b.amps.begin(), p_b.amps.end(), [](const Amp &p_a) { return p_a.multifractal; });
    });
}

bool Table::has_programs() const {
    return std::any_of(biomes.begin(), biomes.end(), [](const Biome &p_b) { return !p_b.program.empty(); });
}

bool SampleProgram::reads_field(int p_field) const {
    return std::any_of(code.begin(), code.end(), [&](const SampleInstr &p_i) { return p_i.op == SampleOp::FIELD && p_i.index == p_field; });
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

namespace {

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

// Biome program p_p's term D at the ground p_ground (H) and the point's slots p_slots (its carried fields, p_carry).
Float4 run_program(const SampleProgram &p_p, const Float4 &p_ground, const float p_slots[Cascade::SLOTS], const SampleCarry &p_carry) {
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
        }
    }
    return dsub(r[p_p.code.size() - 1], p_ground);
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
    code += "\n    precise float a0 = (x.w - x0) * inv;\n    precise float a = (" + Cascade::glsl_flush_of("a0") + ov::str(");\n");
    code += "    precise float t0 = p * a;\n    precise float v = (" + Cascade::glsl_flush_of("t0") + ov::str(") + q;\n");
    code += "    precise float t1 = v * a;\n    v = (" + Cascade::glsl_flush_of("t1") + ov::str(") + r;\n");
    code += "    precise float t2 = v * a;\n    v = (" + Cascade::glsl_flush_of("t2") + ov::str(") + s;\n");
    code += "    precise float u0 = (p * 3.0) * a;\n    precise float d = (" + Cascade::glsl_flush_of("u0") + ov::str(") + q * 2.0;\n");
    code += "    precise float u1 = d * a;\n    d = (" + Cascade::glsl_flush_of("u1") + ov::str(") + r;\n");
    code += "    precise float u2 = d * inv;\n    d = (" + Cascade::glsl_flush_of("u2") + ov::str(");\n");
    code += "    precise vec4 res = vec4(x.x * d, x.y * d, x.z * d, v);\n    return cl_flush(res);\n}\n";
    return code;
}

// Biome p_b's program as `vec4 cl_sample<p_b>(vec4 base, Ch ch)`, D: run_program's ops (its curves cl_curve<b>_<i>
// before it); a curve of H alone is cl_curve<b> itself (its text before programs).
ov::str program_glsl(const SampleProgram &p_p, int p_b, const Cascade::Spec &p_spec, const SampleCarry &p_carry, Cascade::ShaderParams &r_params) {
    if (p_p.curve_of_ground()) {
        return curve_glsl(p_p.curves[0], ov::format("cl_curve%d", int64_t(p_b)), r_params);
    }
    ov::str code;
    for (size_t c = 0; c < p_p.curves.size(); c++) {
        code += curve_glsl(p_p.curves[c], ov::format("cl_curve%d_%d", int64_t(p_b), int64_t(c)), r_params);
    }
    code += ov::format("vec4 cl_sample%d(vec4 base, Ch ch) {\n", int64_t(p_b));
    for (size_t i = 0; i < p_p.code.size(); i++) {
        const SampleInstr &in = p_p.code[i];
        const ov::str a = ov::format("r%d", int64_t(in.a)), b = ov::format("r%d", int64_t(in.b));
        ov::str e;
        switch (in.op) {
            case SampleOp::GROUND:
                e = "base";
                break;
            case SampleOp::CONST:
                e = "vec4(0.0, 0.0, 0.0, " + r_params(in.value) + ")";
                break;
            case SampleOp::FIELD:
                e = carried_glsl(p_spec, "ch", p_carry.field[in.index]);
                break;
            case SampleOp::ADD:
                e = "cl_add(" + a + ", " + b + ")";
                break;
            case SampleOp::SUB:
                e = "cl_sub(" + a + ", " + b + ")";
                break;
            case SampleOp::MUL:
                e = "cl_mul(" + a + ", " + b + ")";
                break;
            case SampleOp::MIN:
                e = b + ".w < " + a + ".w ? " + b + " : " + a;
                break;
            case SampleOp::MAX:
                e = b + ".w > " + a + ".w ? " + b + " : " + a;
                break;
            case SampleOp::SMIN:
                e = "cl_smin(" + a + ", " + b + ", " + r_params(in.value) + ", " + r_params(half_inv(in.value)) + ")";
                break;
            case SampleOp::SMAX:
                e = "-cl_smin(-" + a + ", -" + b + ", " + r_params(in.value) + ", " + r_params(half_inv(in.value)) + ")";
                break;
            case SampleOp::CURVE:
                e = "cl_add(" + a + ", cl_curve" + ov::format("%d_%d", int64_t(p_b), int64_t(in.index)) + "(" + a + "))";
                break;
        }
        code += ov::format("    vec4 r%d = ", int64_t(i)) + e + ";\n";
    }
    return code + ov::format("    return cl_sub(r%d, base);\n}\n", int64_t(p_p.code.size() - 1));
}

} // namespace

Float4 ground_value(const Table &p_table, const float p_slots[Cascade::SLOTS]) {
    return ground_value(p_table, sample_carry(p_table), p_slots);
}

Float4 ground_value(const Table &p_table, const SampleCarry &carry, const float p_slots[Cascade::SLOTS]) {
    const Float4 base = to_float4(p_slots);
    Float4 h = base;
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (carry.weight[b] >= 0) {
            const Float4 w = to_float4(p_slots + Cascade::carry_slot(carry.weight[b]));
            h = dadd(h, dmul(w, run_program(p_table.biomes[b].program, base, p_slots, carry)));
        }
    }
    return h;
}

ov::str ground_glsl(const Table &p_table, const Cascade::Spec &p_combined, Cascade::ShaderParams &r_params) {
    const SampleCarry carry = sample_carry(p_table);
    ov::str code;
    bool smooth = false;
    for (const Biome &b : p_table.biomes) {
        for (const SampleInstr &in : b.program.code) {
            smooth = smooth || in.op == SampleOp::SMIN || in.op == SampleOp::SMAX;
        }
    }
    code += smooth ? ov::str(GLSL_SMOOTH_MIN) : ov::str();
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (carry.weight[b] >= 0) {
            code += ov::replace_all(program_glsl(p_table.biomes[b].program, int(b), p_combined, carry, r_params), "FLUSH", Cascade::glsl_flush());
        }
    }
    code += "\n// Climate::ground_value: the cascade's ground h, plus each curved biome's weight times its curve of h; (value, gradient).\n";
    code += "vec4 climate_ground(Ch ch) {\n";
    code += "    vec4 base = vec4(ch.c[1], ch.c[2], ch.c[3], ch.c[0]);\n    vec4 h = base;\n";
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (carry.weight[b] >= 0) {
            const ov::str term = p_table.biomes[b].program.curve_of_ground() ? ov::format("cl_curve%d(base)", int64_t(b)) : ov::format("cl_sample%d(base, ch)", int64_t(b));
            code += ov::format("    h = cl_add(h, cl_mul(%s, %s));\n", carried_glsl(p_combined, "ch", carry.weight[b]), term);
        }
    }
    code += "    return vec4(h.w, h.x, h.y, h.z);\n}\n";
    return code;
}

} // namespace Climate
