#include "climate.hpp"

#include "ov/math.hpp"
#include "ov/str.hpp"

#include <algorithm>
#include <cmath>

// The climate's math, on the CPU and in GLSL op for op (every product flushed under 2^-64: climate.hpp's exactness
// note): the weights' steps, the pick, the curves, the smooth min, and the biomes' materials.
namespace Climate {

using Cascade::dadd;
using Cascade::dflush;
using Cascade::dmul;
using Cascade::dscale;
using Cascade::dsub;
using Cascade::flush_value;

Float4 fill(const Float4 &p_x) {
    if (!(p_x.w < 1.0f)) {
        return Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    const Float4 t{ -p_x.x, -p_x.y, -p_x.z, 1.0f - p_x.w };
    return dmul(t, t);
}

Float4 axis_factor(const Axis &p_axis, const Float4 &p_x) {
    const Float4 u = dflush(Float4{ p_x.x * p_axis.inv_span, p_x.y * p_axis.inv_span, p_x.z * p_axis.inv_span, (p_x.w - p_axis.centre) * p_axis.inv_span });
    const Float4 uu = dmul(u, u);
    const float raw = flush_value((1.0f - uu.w) * KERNEL_FADE);
    if (!(raw > 0.0f)) {
        return Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    if (raw >= 1.0f) {
        return Float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    const Float4 s = dflush(Float4{ -uu.x * KERNEL_FADE, -uu.y * KERNEL_FADE, -uu.z * KERNEL_FADE, raw });
    // s^2 (3 - 2 s); the x2 scalings are exact.
    const Float4 t{ s.x * -2.0f, s.y * -2.0f, s.z * -2.0f, 3.0f - s.w * 2.0f };
    return dmul(dmul(s, s), t);
}

Float4 blend_weight(const Float4 &p_k, const Float4 &p_rcp) {
    Float4 w = dmul(p_k, p_rcp);
    w.w = ov::min(w.w, 1.0f); // rounding's last bit, never over 1
    return w;
}

Float4 blend_rcp(const Float4 &p_s) {
    // det_rsqrt(s)^2 and one Newton step (1 / 1 comes out exactly 1: a biome alone at its centre is whole).
    const float r = ExactNoise::det_rsqrt(p_s.w);
    const float inv0 = r * r;
    const float inv = inv0 * (2.0f - p_s.w * inv0);
    return dflush(Float4{ -(p_s.x * inv) * inv, -(p_s.y * inv) * inv, -(p_s.z * inv) * inv, inv });
}

int Table::field_index(const ov::str &p_name) const {
    for (size_t i = 0; i < fields.size(); i++) {
        if (fields[i].name == p_name) {
            return int(i);
        }
    }
    return -1;
}
int pick_of(int p_count, const Float4 *p_w, const Vector3i64 &p_p, int p_m) {
    int best[MAX_BIOMES];
    int count = 0;
    for (int b = 0; b < p_count; b++) {
        if (count > 0 && ExactNoise::float_bits(p_w[b].w) == ExactNoise::float_bits(p_w[best[0]].w)) {
            best[count++] = b;
        } else if (count == 0 || p_w[b].w > p_w[best[0]].w) {
            best[0] = b;
            count = 1;
        }
    }
    if (count == 0) {
        return -1;
    }
    // The absolute voxel: the same point seen from two LODs breaks a tie the same way.
    const int64_t step = int64_t(1) << p_m;
    const Vector3i64 voxel(p_p.x * step, p_p.y * step, p_p.z * step);
    return best[ExactNoise::hash_cell(voxel, PICK_SEED) % uint32_t(count)];
}

double shape_phi(const Shape &p_shape, double p_n) {
    const double sabs = std::sqrt(p_n * p_n + double(Cascade::SMOOTH_ABS_E) * Cascade::SMOOTH_ABS_E) - Cascade::SMOOTH_ABS_E;
    return double(p_shape.c[0]) + double(p_shape.c[1]) * p_n + double(p_shape.c[2]) * sabs + double(p_shape.c[3]) * p_n * p_n;
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
// climate's products flush under 2^-64 (its exactness note). Its dip, k / 4 at a = b, is graph_bounds.cpp's bound.
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

// eval_curve over a curve's record in the params' heap (n, y_last, x[n], then inv, p, q, r, s per segment), the
// segment chosen by eval_curve's compares in a loop.
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

// flushed, dmul, dsub, fill, axis_factor and the Newton reciprocal in GLSL, op for op.
constexpr const char GLSL_CLIMATE_HELPERS[] = R"glsl(
vec4 cl_flush(vec4 v) {
    return vec4(abs(v.x) < FLUSH ? 0.0 : v.x, abs(v.y) < FLUSH ? 0.0 : v.y, abs(v.z) < FLUSH ? 0.0 : v.z, abs(v.w) < FLUSH ? 0.0 : v.w);
}

vec4 cl_mul(vec4 a, vec4 b) {
    precise vec4 r;
    r.x = a.w * b.x + b.w * a.x;
    r.y = a.w * b.y + b.w * a.y;
    r.z = a.w * b.z + b.w * a.z;
    r.w = a.w * b.w;
    return cl_flush(r);
}

vec4 cl_sub(vec4 a, vec4 b) {
    precise vec4 r = a - b;
    return cl_flush(r);
}

// Climate::fill: the default's (1 - x)^2 for x < 1, else 0.
vec4 cl_fill(vec4 x) {
    if (!(x.w < 1.0)) {
        return vec4(0.0);
    }
    precise vec4 t = vec4(-x.x, -x.y, -x.z, 1.0 - x.w);
    return cl_mul(t, t);
}

// Climate::axis_factor: the smoothstep of clamp((1 - u^2) fade, 0, 1), u = (x - centre) * inv_span.
vec4 cl_axis(vec4 x, float centre, float inv_span, float fade) {
    precise vec4 u0 = vec4(x.x * inv_span, x.y * inv_span, x.z * inv_span, (x.w - centre) * inv_span);
    vec4 u = cl_flush(u0);
    vec4 uu = cl_mul(u, u);
    precise float raw0 = (1.0 - uu.w) * fade;
    float raw = abs(raw0) < FLUSH ? 0.0 : raw0;
    if (!(raw > 0.0)) {
        return vec4(0.0);
    }
    if (raw >= 1.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    precise vec4 s0 = vec4(-uu.x * fade, -uu.y * fade, -uu.z * fade, raw);
    vec4 s = cl_flush(s0);
    precise vec4 t = vec4(s.x * -2.0, s.y * -2.0, s.z * -2.0, 3.0 - s.w * 2.0);
    return cl_mul(cl_mul(s, s), t);
}

// The weights' normalisation (blend_rcp): (-s' / s^2, 1 / s), 1 / s = det_rsqrt(s)^2 and one Newton step.
vec4 cl_rcp(vec4 s) {
    precise float r = det_rsqrt(s.w);
    precise float inv0 = r * r;
    precise float inv = inv0 * (2.0 - s.w * inv0);
    precise vec4 v = vec4(-(s.x * inv) * inv, -(s.y * inv) * inv, -(s.z * inv) * inv, inv);
    return cl_flush(v);
}
)glsl";

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

ov::str shared_helpers_glsl() {
    return ov::replace_all(ov::str(GLSL_CLIMATE_HELPERS), "FLUSH", Cascade::glsl_flush()) + R"glsl(
vec4 cl_add(vec4 a, vec4 b) {
    precise vec4 r = a + b;
    return cl_flush(r);
}

vec4 cl_scale(vec4 a, float k) {
    precise vec4 r = a * k;
    return cl_flush(r);
}
)glsl";
}
ov::Color srgb_to_linear(uint32_t p_rgb) {
    return ov::Color::hex((p_rgb << 8) | 0xFFu).srgb_to_linear();
}

Material::Params biome_material(const Biome &p_biome) {
    Material::Params m;
    m.albedo = p_biome.surface.flat;
    m.steep = p_biome.surface.steep;
    m.roughness = p_biome.surface.roughness;
    m.pattern = Material::PATTERN_GROUND;
    m.debug = srgb_to_linear(p_biome.debug_color);
    return m;
}

} // namespace Climate