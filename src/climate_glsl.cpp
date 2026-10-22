#include "climate.hpp"

// The climate's GLSL, generated from a table op for op with its CPU code in climate.cpp: the dual-number helpers, the
// field terms and biome weights, and the pick.
namespace Climate {

namespace {

// flushed, dmul, dsub and edge_weight in GLSL, op for op.
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

vec4 cl_edge(vec4 x, float anchor, float scale) {
    precise float raw0 = (x.w - anchor) * scale;
    float raw = abs(raw0) < FLUSH ? 0.0 : raw0;
    if (!(raw > 0.0)) {
        return vec4(0.0);
    }
    if (raw >= 1.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    precise vec4 s0 = vec4(x.x * scale, x.y * scale, x.z * scale, raw);
    vec4 s = cl_flush(s0);
    precise vec4 q = vec4(s.x * -2.0, s.y * -2.0, s.z * -2.0, 3.0 - s.w * 2.0);
    return cl_mul(cl_mul(s, s), q);
}
)glsl";

} // namespace

String glsl_helpers() {
    return String(GLSL_CLIMATE_HELPERS).replace("FLUSH", Cascade::glsl_flush());
}

bool splices_helpers(const Table &p_table, const Cascade::Spec &p_combined) {
    return (p_combined.modulation && p_combined.modulation->shaped()) || p_table.has_ground_terms();
}

String shared_helpers_glsl() {
    return glsl_helpers() + R"glsl(
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

String glsl_weights(const Table &p_table, const Cascade::Spec &p_spec) {
    using ExactNoise::glsl_float;
    String code;
    auto channels = [](int p_base) { return vformat("vec4(pre.c[%d], pre.c[%d], pre.c[%d], pre.c[%d])", p_base + 1, p_base + 2, p_base + 3, p_base); };
    code += "    vec4 g0 = " + channels(p_spec.gpu_base(0)) + String(";\n");
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        code += vformat("    vec4 f%d = %s;\n", int64_t(i), channels(p_spec.gpu_base(1 + int(i))));
        for (const Term &t : p_table.fields[i].terms) {
            const String src = t.source == GROUND ? String("g0") : vformat("f%d", t.source);
            code += vformat("    { precise vec4 t = f%d + %s * %s; f%d = cl_flush(t); }\n", int64_t(i), src, glsl_float(t.coef), int64_t(i));
        }
    }
    code += "    vec4 rest = vec4(0.0, 0.0, 0.0, 1.0);\n";
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b + 1 < n; b++) {
        code += glsl_edges_value(vformat("r%d", int64_t(b)), p_table.biomes[b].edges);
        code += vformat("    vec4 w%d = cl_mul(rest, r%d);\n    rest = cl_sub(rest, w%d);\n", int64_t(b), int64_t(b), int64_t(b));
    }
    code += vformat("    vec4 w%d = rest;\n", int64_t(n - 1));
    return code;
}

String glsl_edges_value(const String &p_var, const std::vector<Edge> &p_edges) {
    using ExactNoise::glsl_float;
    String code = "    vec4 " + p_var + String(" = vec4(0.0, 0.0, 0.0, 1.0);\n");
    for (const Edge &e : p_edges) {
        code += vformat("    %s = cl_mul(%s, cl_edge(f%d, %s, %s));\n", p_var, p_var, e.field, glsl_float(e.anchor), glsl_float(e.scale));
    }
    return code;
}

String pick_glsl(const Table &p_table, const Cascade::Spec &p_combined) {
    String code = p_combined.modulation || splices_helpers(p_table, p_combined) ? String() : glsl_helpers();
    code += "\n// Climate::pick of the weights of the channels pre at the absolute point p (a lattice point).\nuint climate_biome(Ch pre, WorldPos p) {\n";
    code += glsl_weights(p_table, p_combined);
    // The largest weight's bits and how many biomes share them, then the (hash % count)-th of those in list order: the
    // CPU's incremental list holds exactly them (nothing exceeds the last best, so it is never reset after it).
    code += "    float best = w0.w;\n    uint count = 1u;\n";
    const int64_t n = int64_t(p_table.biomes.size());
    for (int64_t b = 1; b < n; b++) {
        code += vformat("    if (floatBitsToUint(w%d.w) == floatBitsToUint(best)) {\n        count++;\n    } else if (w%d.w > best) {\n        best = w%d.w;\n"
                        "        count = 1u;\n    }\n",
                b, b, b);
    }
    code += vformat("    uint k = count == 1u ? 0u : hash_cell(p.lo, p.hi, %du) %% count;\n", int64_t(PICK_SEED));
    for (int64_t b = 0; b < n; b++) {
        code += vformat("    if (floatBitsToUint(w%d.w) == floatBitsToUint(best)) {\n        if (k == 0u) {\n            return %du;\n        }\n        k--;\n    }\n", b,
                b);
    }
    code += "    return 0u;\n}\n";
    return code;
}

} // namespace Climate
