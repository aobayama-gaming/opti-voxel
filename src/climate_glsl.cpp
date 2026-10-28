#include "climate.hpp"

#include "ov/math.hpp"
#include "ov/str.hpp"

#include <algorithm>
#include <cmath>

// The climate's GLSL, generated from a table op for op with its CPU code in climate.cpp: the dual-number helpers, the
// field terms and biome weights, and the pick.
namespace Climate {

namespace {

// edge_raw in GLSL, op for op: cl_edge's first lines, and cl_edge_off's (one text, so the guards stay its test).
constexpr const char GLSL_EDGE_RAW[] = R"glsl(    precise float raw0 = (x.w - anchor) * scale;
    float raw = abs(raw0) < FLUSH ? 0.0 : raw0;
)glsl";

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
EDGE_RAW
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

// edge_off in GLSL: whether cl_edge returns vec4(0.0).
constexpr const char GLSL_EDGE_OFF[] = R"glsl(
bool cl_edge_off(vec4 x, float anchor, float scale) {
EDGE_RAW
    return !(raw > 0.0);
}
)glsl";

ov::str with_edge_raw(const char *p_text) {
    return ov::replace_all(ov::replace_all(ov::str(p_text), "EDGE_RAW\n", GLSL_EDGE_RAW), "FLUSH", Cascade::glsl_flush());
}

} // namespace

ov::str glsl_helpers() {
    return with_edge_raw(GLSL_CLIMATE_HELPERS);
}

ov::str region_helpers_glsl() {
    return with_edge_raw(GLSL_EDGE_OFF);
}

bool splices_helpers(const Table &p_table, const Cascade::Spec &p_combined) {
    return (p_combined.modulation && p_combined.modulation->shaped()) || p_table.has_ground_terms();
}

ov::str shared_helpers_glsl() {
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

ov::str glsl_fields(const Table &p_table, const Cascade::Spec &p_spec) {
    using ExactNoise::glsl_float;
    ov::str code;
    auto channels = [](int p_base) { return ov::format("vec4(pre.c[%d], pre.c[%d], pre.c[%d], pre.c[%d])", p_base + 1, p_base + 2, p_base + 3, p_base); };
    code += "    vec4 g0 = " + channels(p_spec.gpu_base(0)) + ov::str(";\n");
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        code += ov::format("    vec4 f%d = %s;\n", int64_t(i), channels(p_spec.gpu_base(1 + int(i))));
        for (const Term &t : p_table.fields[i].terms) {
            const ov::str src = t.source == GROUND ? ov::str("g0") : ov::format("f%d", t.source);
            code += ov::format("    { precise vec4 t = f%d + %s * %s; f%d = cl_flush(t); }\n", int64_t(i), src, glsl_float(t.coef), int64_t(i));
        }
    }
    return code;
}

uint32_t guarded_regions(const Table &p_table) {
    uint32_t used = 0u;
    for (const Biome &b : p_table.biomes) {
        used |= b.region >= 0 ? 1u << b.region : 0u;
    }
    for (const int r : p_table.gated) {
        used |= 1u << r;
    }
    return used;
}

ov::str glsl_biome_weights(const Table &p_table, bool p_guards) {
    ov::str code = "    vec4 rest = vec4(0.0, 0.0, 0.0, 1.0);\n";
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b + 1 < n; b++) {
        const int r = p_table.biomes[b].region;
        if (p_guards && r >= 0) {
            // Where its region is off, r_b is +0, so w_b is +0 and rest -= w_b leaves rest as it is.
            code += ov::format("    vec4 w%d = vec4(0.0);\n    if (!rof%d) {\n", int64_t(b), int64_t(r));
            code += glsl_edges_value(ov::format("r%d", int64_t(b)), p_table.biomes[b].edges);
            code += ov::format("    w%d = cl_mul(rest, r%d);\n    rest = cl_sub(rest, w%d);\n    }\n", int64_t(b), int64_t(b), int64_t(b));
            continue;
        }
        code += glsl_edges_value(ov::format("r%d", int64_t(b)), p_table.biomes[b].edges);
        code += ov::format("    vec4 w%d = cl_mul(rest, r%d);\n    rest = cl_sub(rest, w%d);\n", int64_t(b), int64_t(b), int64_t(b));
    }
    code += ov::format("    vec4 w%d = rest;\n", int64_t(n - 1));
    return code;
}

namespace {

// Groups (bit g: the ground 0, field i at i + 1) of the fields p_fields (bit i), closed under the fields' terms' sources
// (apply_terms reads earlier fields: from the last field down, every source of a needed one).
uint32_t field_groups(const Table &p_table, uint32_t p_fields) {
    for (int i = int(p_table.fields.size()) - 1; i >= 0; i--) {
        if ((p_fields & (1u << i)) == 0u) {
            continue;
        }
        for (const Term &t : p_table.fields[size_t(i)].terms) {
            p_fields |= t.source == GROUND ? 0u : 1u << t.source;
        }
    }
    return p_fields << 1;
}

uint32_t edge_fields(const std::vector<Edge> &p_edges) {
    uint32_t f = 0u;
    for (const Edge &e : p_edges) {
        f |= 1u << e.field;
    }
    return f;
}

// The fields biome p_b reads: its edges and its warps'.
uint32_t biome_fields(const Biome &p_b) {
    uint32_t f = edge_fields(p_b.edges);
    for (const Amp &a : p_b.amps) {
        for (const int w : a.warp_fields) {
            f |= w >= 0 ? 1u << w : 0u;
        }
    }
    return f;
}

} // namespace

uint32_t region_field_groups(const Table &p_table, int p_region) {
    uint32_t f = 0u;
    for (const Biome &b : p_table.biomes) {
        f |= b.region == p_region ? biome_fields(b) : 0u;
    }
    for (const HeightTerm &t : p_table.heights) {
        f |= t.region == p_region ? 1u << t.field : 0u;
    }
    return field_groups(p_table, f);
}

uint32_t base_field_groups(const Table &p_table) {
    const uint32_t guarded = guarded_regions(p_table);
    uint32_t f = 0u;
    for (int r = 0; r < int(p_table.regions.size()); r++) {
        f |= (guarded & (1u << r)) != 0u ? edge_fields(p_table.regions[size_t(r)].edges) : 0u;
    }
    for (const Biome &b : p_table.biomes) {
        f |= b.region < 0 ? biome_fields(b) : 0u;
    }
    for (const HeightTerm &t : p_table.heights) {
        f |= t.region < 0 ? 1u << t.field : 0u;
    }
    const int n = int(p_table.fields.size());
    return 1u | field_groups(p_table, f) | (p_table.has_state() ? 1u << (n + 1) : 0u);
}

ov::str region_off_glsl(const Table &p_table) {
    using ExactNoise::glsl_float;
    const uint32_t guarded = guarded_regions(p_table);
    ov::str code;
    for (int r = 0; r < int(p_table.regions.size()); r++) {
        if ((guarded & (1u << r)) == 0u) {
            continue;
        }
        ov::str any;
        for (const Edge &e : p_table.regions[size_t(r)].edges) {
            any += ov::format("%scl_edge_off(f%d, %s, %s)", any.empty() ? "" : " || ", e.field, glsl_float(e.anchor), glsl_float(e.scale));
        }
        code += ov::format("    bool rof%d = %s;\n", r, any);
    }
    return code;
}

ov::str region_on_glsl(const Table &p_table) {
    const uint32_t guarded = guarded_regions(p_table);
    ov::str code;
    for (int r = 0; r < int(p_table.regions.size()); r++) {
        if ((guarded & (1u << r)) != 0u) {
            code += ov::format("%s(rof%d ? 0u : %du)", code.empty() ? "" : " | ", r, int64_t(1u << r));
        }
    }
    return code.empty() ? ov::str("0u") : code;
}

ov::str glsl_edges_value(const ov::str &p_var, const std::vector<Edge> &p_edges) {
    using ExactNoise::glsl_float;
    ov::str code = "    vec4 " + p_var + ov::str(" = vec4(0.0, 0.0, 0.0, 1.0);\n");
    for (const Edge &e : p_edges) {
        code += ov::format("    %s = cl_mul(%s, cl_edge(f%d, %s, %s));\n", p_var, p_var, e.field, glsl_float(e.anchor), glsl_float(e.scale));
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
