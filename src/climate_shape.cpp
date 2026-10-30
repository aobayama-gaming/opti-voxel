#include "climate.hpp"

#include "ov/math.hpp"
#include "ov/str.hpp"

#include <algorithm>
#include <cmath>

// Band shapes' bounds, the biomes' curves and the ground at a sample (ground_value), on the CPU and in generated GLSL,
// op for op: every product flushed under 2^-64 like the climate's dual numbers (climate.hpp's exactness note).
namespace Climate {

using Cascade::dadd;
using Cascade::dflush;
using Cascade::dmul;
using Cascade::flush_value;

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
    return std::any_of(biomes.begin(), biomes.end(),
            [](const Biome &p_b) { return std::any_of(p_b.amps.begin(), p_b.amps.end(), [](const Amp &p_a) { return p_a.gain > 0.0f; }); });
}

bool Table::has_curves() const {
    return std::any_of(biomes.begin(), biomes.end(), [](const Biome &p_b) { return !p_b.curve.empty(); });
}

namespace {

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

// eval_curve as `vec4 cl_curve<p_i>(vec4 x)`, the segment chosen by the same compares.
ov::str curve_glsl(const Curve &p_c, int p_i) {
    using ExactNoise::glsl_float;
    const size_t n = p_c.x.size();
    ov::str code = ov::format("vec4 cl_curve%d(vec4 x) {\n", p_i);
    code += ov::format("    if (!(x.w > %s)) {\n        return vec4(0.0, 0.0, 0.0, %s);\n    }\n", glsl_float(p_c.x[0]), glsl_float(p_c.s[0]));
    code += ov::format("    if (x.w >= %s) {\n        return vec4(0.0, 0.0, 0.0, %s);\n    }\n", glsl_float(p_c.x[n - 1]), glsl_float(p_c.y_last));
    code += "    float x0, inv, p, q, r, s;\n";
    for (size_t i = 0; i + 1 < n; i++) {
        const bool last = i + 2 >= n;
        code += ov::str(i == 0 ? "    " : " else ") + (last ? ov::str("{") : ov::format("if (x.w < %s) {", glsl_float(p_c.x[i + 1])));
        code += ov::format(" x0 = %s; inv = %s; p = %s; q = %s; r = %s; s = %s; }", glsl_float(p_c.x[i]), glsl_float(p_c.inv[i]), glsl_float(p_c.p[i]),
                glsl_float(p_c.q[i]), glsl_float(p_c.r[i]), glsl_float(p_c.s[i]));
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

} // namespace

Float4 ground_value(const Table &p_table, const float p_slots[Cascade::SLOTS]) {
    const Float4 base = to_float4(p_slots);
    const SampleCarry carry = sample_carry(p_table);
    Float4 h = base;
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (carry.curve[b] >= 0) {
            const Float4 w = to_float4(p_slots + Cascade::carry_slot(carry.curve[b]));
            h = dadd(h, dmul(w, eval_curve(p_table.biomes[b].curve, base)));
        }
    }
    return h;
}

ov::str ground_glsl(const Table &p_table, const Cascade::Spec &p_combined) {
    const SampleCarry carry = sample_carry(p_table);
    ov::str code;
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (carry.curve[b] >= 0) {
            code += ov::replace_all(curve_glsl(p_table.biomes[b].curve, int(b)), "FLUSH", Cascade::glsl_flush());
        }
    }
    code += "\n// Climate::ground_value: the cascade's ground h, plus each curved biome's weight times its curve of h; (value, gradient).\n";
    code += "vec4 climate_ground(Ch ch) {\n";
    code += "    vec4 base = vec4(ch.c[1], ch.c[2], ch.c[3], ch.c[0]);\n    vec4 h = base;\n";
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (carry.curve[b] >= 0) {
            code += ov::format("    h = cl_add(h, cl_mul(%s, cl_curve%d(base)));\n", carried_glsl(p_combined, "ch", carry.curve[b]), int64_t(b));
        }
    }
    code += "    return vec4(h.w, h.x, h.y, h.z);\n}\n";
    return code;
}

} // namespace Climate
