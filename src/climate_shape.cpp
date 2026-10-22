#include "climate.hpp"

#include <algorithm>
#include <cmath>

// Band shapes' bounds, curves, terraces and the ground at a sample (ground_value), on the CPU and in generated GLSL, op
// for op: every product flushed under 2^-64 like the climate's dual numbers (climate.hpp's exactness note).
namespace Climate {

using Cascade::dadd;
using Cascade::dflush;
using Cascade::dmul;
using Cascade::dscale;
using Cascade::dsub;
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
    auto y = [&](int p_i) { return p_y[size_t(CLAMP(p_i, 0, n - 1))]; };
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

bool Table::has_state() const {
    return std::any_of(biomes.begin(), biomes.end(),
            [](const Biome &p_b) { return std::any_of(p_b.amps.begin(), p_b.amps.end(), [](const Amp &p_a) { return p_a.gain > 0.0f; }); });
}

bool Table::has_terraces() const {
    return std::any_of(biomes.begin(), biomes.end(), [](const Biome &p_b) { return p_b.terrace.step > 0.0f; });
}

bool Table::has_ground_terms() const {
    return !heights.empty() || has_terraces();
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

// libnoise's Terrace with points every p_t.step: v0 + step t^2 (inverted: v1 - step (1 - t)^2), slope 2 t (2 (1 - t)).
Float4 eval_terrace(const Terrace &p_t, const Float4 &p_h) {
    const float u = p_h.w * p_t.inv_step;
    float fl = std::floor(u);
    float a = u - fl;
    if (a >= 1.0f) {
        fl = fl + 1.0f;
        a = 0.0f;
    }
    const float v0 = fl * p_t.step;
    float g, d;
    if (p_t.invert) {
        const float b = 1.0f - a;
        g = (v0 + p_t.step) - p_t.step * flush_value(b * b);
        d = b * 2.0f;
    } else {
        g = v0 + p_t.step * flush_value(a * a);
        d = a * 2.0f;
    }
    return dflush(Float4{ p_h.x * d, p_h.y * d, p_h.z * d, g });
}

constexpr const char GLSL_TERRACE[] = R"glsl(
vec4 cl_terrace(vec4 h, float step, float inv_step, bool invert) {
    precise float u = h.w * inv_step;
    precise float fl = floor(u);
    precise float a = u - fl;
    if (a >= 1.0) {
        fl = fl + 1.0;
        a = 0.0;
    }
    precise float v0 = fl * step;
    precise float g;
    precise float d;
    if (invert) {
        precise float b = 1.0 - a;
        precise float b2 = b * b;
        g = (v0 + step) - step * (abs(b2) < FLUSH ? 0.0 : b2);
        d = b * 2.0;
    } else {
        precise float a2 = a * a;
        g = v0 + step * (abs(a2) < FLUSH ? 0.0 : a2);
        d = a * 2.0;
    }
    precise vec4 r = vec4(h.x * d, h.y * d, h.z * d, g);
    return cl_flush(r);
}
)glsl";

String glsl_flush_of(const String &p_expr) {
    return "(abs(" + p_expr + String(") < FLUSH ? 0.0 : ") + p_expr + String(")");
}

// eval_curve as `vec4 cl_curve<p_i>(vec4 x)`, the segment chosen by the same compares.
String curve_glsl(const Curve &p_c, int p_i) {
    using ExactNoise::glsl_float;
    const size_t n = p_c.x.size();
    String code = vformat("vec4 cl_curve%d(vec4 x) {\n", p_i);
    code += vformat("    if (!(x.w > %s)) {\n        return vec4(0.0, 0.0, 0.0, %s);\n    }\n", glsl_float(p_c.x[0]), glsl_float(p_c.s[0]));
    code += vformat("    if (x.w >= %s) {\n        return vec4(0.0, 0.0, 0.0, %s);\n    }\n", glsl_float(p_c.x[n - 1]), glsl_float(p_c.y_last));
    code += "    float x0, inv, p, q, r, s;\n";
    for (size_t i = 0; i + 1 < n; i++) {
        const bool last = i + 2 >= n;
        code += String(i == 0 ? "    " : " else ") + (last ? String("{") : vformat("if (x.w < %s) {", glsl_float(p_c.x[i + 1])));
        code += vformat(" x0 = %s; inv = %s; p = %s; q = %s; r = %s; s = %s; }", glsl_float(p_c.x[i]), glsl_float(p_c.inv[i]), glsl_float(p_c.p[i]),
                glsl_float(p_c.q[i]), glsl_float(p_c.r[i]), glsl_float(p_c.s[i]));
    }
    code += "\n    precise float a0 = (x.w - x0) * inv;\n    precise float a = " + glsl_flush_of("a0") + String(";\n");
    code += "    precise float t0 = p * a;\n    precise float v = " + glsl_flush_of("t0") + String(" + q;\n");
    code += "    precise float t1 = v * a;\n    v = " + glsl_flush_of("t1") + String(" + r;\n");
    code += "    precise float t2 = v * a;\n    v = " + glsl_flush_of("t2") + String(" + s;\n");
    code += "    precise float u0 = (p * 3.0) * a;\n    precise float d = " + glsl_flush_of("u0") + String(" + q * 2.0;\n");
    code += "    precise float u1 = d * a;\n    d = " + glsl_flush_of("u1") + String(" + r;\n");
    code += "    precise float u2 = d * inv;\n    d = " + glsl_flush_of("u2") + String(";\n");
    code += "    precise vec4 res = vec4(x.x * d, x.y * d, x.z * d, v);\n    return cl_flush(res);\n}\n";
    return code;
}

} // namespace

Float4 ground_value(const Table &p_table, const float p_slots[Cascade::SLOTS]) {
    Float4 h = to_float4(p_slots);
    if (!p_table.has_ground_terms()) {
        return h;
    }
    Float4 f[MAX_FIELDS];
    apply_terms(p_table, p_slots, f);
    for (const HeightTerm &t : p_table.heights) {
        const Float4 v = t.has_curve ? eval_curve(t.curve, f[t.field]) : f[t.field];
        h = dadd(h, dscale(v, t.coef));
    }
    if (!p_table.has_terraces()) {
        return h;
    }
    Float4 w[MAX_BIOMES];
    weights(p_table, f, w);
    const Float4 base = h;
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        const Terrace &t = p_table.biomes[b].terrace;
        if (t.step > 0.0f) {
            h = dadd(h, dmul(w[b], dsub(eval_terrace(t, base), base)));
        }
    }
    return h;
}

String ground_glsl(const Table &p_table, const Cascade::Spec &p_combined) {
    using ExactNoise::glsl_float;
    const String flush = Cascade::glsl_flush();
    String code;
    code += String(GLSL_TERRACE).replace("FLUSH", flush);
    for (size_t i = 0; i < p_table.heights.size(); i++) {
        if (p_table.heights[i].has_curve) {
            code += curve_glsl(p_table.heights[i].curve, int(i)).replace("FLUSH", flush);
        }
    }
    code += "\n// Climate::ground_value: the cascade's ground, the height terms, then the biomes' terraces; (value, gradient).\n";
    code += "vec4 climate_ground(Ch pre) {\n";
    code += glsl_weights(p_table, p_combined);
    code += "    vec4 h = g0;\n";
    for (size_t i = 0; i < p_table.heights.size(); i++) {
        const HeightTerm &t = p_table.heights[i];
        const String v = t.has_curve ? vformat("cl_curve%d(f%d)", int64_t(i), t.field) : vformat("f%d", t.field);
        code += vformat("    h = cl_add(h, cl_scale(%s, %s));\n", v, glsl_float(t.coef));
    }
    code += "    vec4 base = h;\n";
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        const Terrace &t = p_table.biomes[b].terrace;
        if (t.step > 0.0f) {
            code += vformat("    h = cl_add(h, cl_mul(w%d, cl_sub(cl_terrace(base, %s, %s, %s), base)));\n", int64_t(b), glsl_float(t.step), glsl_float(t.inv_step),
                    t.invert ? "true" : "false");
        }
    }
    code += "    return vec4(h.w, h.x, h.y, h.z);\n}\n";
    return code;
}

} // namespace Climate
