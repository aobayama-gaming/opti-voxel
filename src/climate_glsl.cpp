#include "climate.hpp"

#include "ov/math.hpp"
#include "ov/str.hpp"

// The climate's GLSL, generated from a table op for op with its CPU code in climate.cpp: the dual-number helpers, the
// field terms and biome weights, and the pick.
namespace Climate {

namespace {

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

// Climate::weights' normalisation: (-s' / s^2, 1 / s), 1 / s = det_rsqrt(s)^2 and one Newton step.
vec4 cl_rcp(vec4 s) {
    precise float r = det_rsqrt(s.w);
    precise float inv0 = r * r;
    precise float inv = inv0 * (2.0 - s.w * inv0);
    precise vec4 v = vec4(-(s.x * inv) * inv, -(s.y * inv) * inv, -(s.z * inv) * inv, inv);
    return cl_flush(v);
}
)glsl";

// A biome's kernel (Climate::kernel: its axes' factors multiplied in order) as statements defining `vec4 <p_var>` from
// the f<i> of glsl_weights.
ov::str glsl_kernel(const ov::str &p_var, const std::vector<Axis> &p_at, Cascade::ShaderParams &r_params) {
    ov::str code = "    vec4 " + p_var + ov::str(" = vec4(0.0, 0.0, 0.0, 1.0);\n");
    for (const Axis &a : p_at) {
        code += ov::format("    %s = cl_mul(%s, cl_axis(f%d, %s, %s, %s));\n", p_var, p_var, a.field, r_params(a.centre), r_params(a.inv_span),
                ExactNoise::glsl_float(KERNEL_FADE));
    }
    return code;
}

} // namespace

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

ov::str glsl_weights(const Table &p_table, const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) {
    ov::str code;
    auto channels = [](int p_base) { return ov::format("vec4(pre.c[%d], pre.c[%d], pre.c[%d], pre.c[%d])", p_base + 1, p_base + 2, p_base + 3, p_base); };
    code += "    vec4 g0 = " + channels(p_spec.gpu_base(0)) + ov::str(";\n");
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        code += ov::format("    vec4 f%d = %s;\n", int64_t(i), channels(p_spec.gpu_base(1 + int(i))));
        for (const Term &t : p_table.fields[i].terms) {
            const ov::str src = t.source == GROUND ? ov::str("g0") : ov::format("f%d", t.source);
            code += ov::format("    { precise vec4 t = f%d + %s * %s; f%d = cl_flush(t); }\n", int64_t(i), src, r_params(t.coef), int64_t(i));
        }
    }
    const int64_t n = int64_t(p_table.biomes.size());
    if (n == 1) {
        return code + ov::str("    vec4 w0 = vec4(0.0, 0.0, 0.0, 1.0);\n");
    }
    // Climate::weights: the kernels, their sum, the default's (1 - sum)^2, then each over the total.
    code += "    vec4 ksum = vec4(0.0);\n";
    for (int64_t b = 0; b + 1 < n; b++) {
        code += glsl_kernel(ov::format("w%d", b), p_table.biomes[size_t(b)].at, r_params);
        code += ov::format("    { precise vec4 t = ksum + w%d; ksum = cl_flush(t); }\n", b);
    }
    code += ov::format("    vec4 w%d = cl_fill(ksum);\n", n - 1);
    code += ov::format("    vec4 krcp;\n    { precise vec4 t = ksum + w%d; krcp = cl_rcp(cl_flush(t)); }\n", n - 1);
    for (int64_t b = 0; b < n; b++) {
        code += ov::format("    w%d = cl_mul(w%d, krcp);\n    w%d.w = min(w%d.w, 1.0);\n", b, b, b, b);
    }
    return code;
}

ov::str pick_statements_glsl(const Table &p_table, const ov::str &p_var) {
    // The largest weight's bits and how many biomes share them, then the (hash % count)-th of those in list order: the
    // CPU's incremental list holds exactly them (nothing exceeds the last best, so it is never reset after it).
    ov::str code = "    float " + p_var + ov::str(" = 0.0;\n    {\n        float best = w0.w;\n        uint count = 1u;\n");
    const int64_t n = int64_t(p_table.biomes.size());
    for (int64_t b = 1; b < n; b++) {
        code += ov::format("        if (floatBitsToUint(w%d.w) == floatBitsToUint(best)) {\n            count++;\n        } else if (w%d.w > best) {\n"
                           "            best = w%d.w;\n            count = 1u;\n        }\n",
                b, b, b);
    }
    code += ov::format("        uint k = count == 1u ? 0u : hash_cell(p.lo, p.hi, %du) %% count;\n        bool found = false;\n", int64_t(PICK_SEED));
    for (int64_t b = 0; b < n; b++) {
        code += ov::format("        if (!found && floatBitsToUint(w%d.w) == floatBitsToUint(best)) {\n            if (k == 0u) {\n                %s = %s;\n"
                           "                found = true;\n            } else {\n                k--;\n            }\n        }\n",
                b, p_var, ExactNoise::glsl_float(float(b)));
    }
    return code + "    }\n";
}

ov::str carried_glsl(const Cascade::Spec &p_spec, const ov::str &p_ch, int p_c) {
    const int b = p_spec.gpu_carry(p_c);
    return ov::format("vec4(%s.c[%d], %s.c[%d], %s.c[%d], %s.c[%d])", p_ch, b + 1, p_ch, b + 2, p_ch, b + 3, p_ch, b);
}

ov::str pick_glsl(const Table &p_table, const Cascade::Spec &p_combined) {
    const SampleCarry carry = sample_carry(p_table);
    ov::str code = "\n// Climate::pick_slots: the pick the level above derived (SampleCarry::pick).\nuint climate_biome(Ch ch) {\n";
    if (carry.pick < 0) {
        return code + ov::format("    return %du;\n}\n", int64_t(p_table.biomes.size()) - 1);
    }
    return code + ov::format("    return uint(ch.c[%d]);\n}\n", p_combined.gpu_carry(carry.pick));
}

} // namespace Climate
