#include "climate.hpp"

#include "ov/math.hpp"
#include "ov/str.hpp"

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

ov::str glsl_helpers() {
    return ov::replace_all(ov::str(GLSL_CLIMATE_HELPERS), "FLUSH", Cascade::glsl_flush());
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

ov::str glsl_biome_weights(const Table &p_table, uint32_t p_live) {
    ov::str code = "    vec4 rest = vec4(0.0, 0.0, 0.0, 1.0);\n";
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b + 1 < n; b++) {
        if ((p_live & (1u << b)) == 0u) {
            // Its region is off here (RegionPath): r_b is +0, so w_b is +0 and rest -= w_b leaves rest as it is.
            code += ov::format("    vec4 w%d = vec4(0.0);\n", int64_t(b));
            continue;
        }
        code += glsl_edges_value(ov::format("r%d", int64_t(b)), p_table.biomes[b].edges);
        code += ov::format("    vec4 w%d = cl_mul(rest, r%d);\n    rest = cl_sub(rest, w%d);\n", int64_t(b), int64_t(b), int64_t(b));
    }
    code += ov::format("    vec4 w%d = rest;\n", int64_t(n - 1));
    return code;
}

ov::str glsl_weights(const Table &p_table, const Cascade::Spec &p_spec) {
    return glsl_fields(p_table, p_spec) + glsl_biome_weights(p_table, ALL_LIVE);
}

std::vector<RegionPath> region_paths(const Table &p_table) {
    uint32_t used = 0u;
    for (const Biome &b : p_table.biomes) {
        used |= b.region >= 0 ? 1u << b.region : 0u;
    }
    for (const int r : p_table.gated) {
        used |= 1u << r;
    }
    std::vector<RegionPath> paths;
    if (__builtin_popcount(used) < 2) {
        return paths; // one landform or none: nothing to skip
    }
    for (int r = 0; r < int(p_table.regions.size()); r++) {
        if ((used & (1u << r)) == 0u) {
            continue;
        }
        RegionPath path;
        path.region = r;
        path.off = used & ~(1u << r);
        for (size_t b = 0; b < p_table.biomes.size(); b++) {
            path.biomes |= p_table.biomes[b].region < 0 || p_table.biomes[b].region == r ? 1u << b : 0u;
        }
        path.gated = 1u << r;
        paths.push_back(path);
    }
    return paths;
}

uint32_t path_field_groups(const Table &p_table, const RegionPath &p_path) {
    uint32_t fields = 0u; // bit i: field i
    auto edges = [&](const std::vector<Edge> &p_edges) {
        for (const Edge &e : p_edges) {
            fields |= 1u << e.field;
        }
    };
    for (int r = 0; r < int(p_table.regions.size()); r++) {
        if (((p_path.off | (1u << p_path.region)) & (1u << r)) != 0u) {
            edges(p_table.regions[size_t(r)].edges);
        }
    }
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if ((p_path.biomes & (1u << b)) == 0u) {
            continue;
        }
        edges(p_table.biomes[b].edges);
        for (const Amp &a : p_table.biomes[b].amps) {
            for (const int f : a.warp_fields) {
                fields |= f >= 0 ? 1u << f : 0u;
            }
        }
    }
    for (const HeightTerm &t : p_table.heights) {
        if (t.region < 0 || (p_path.gated & (1u << t.region)) != 0u) {
            fields |= 1u << t.field;
        }
    }
    // A field's terms read earlier fields (apply_terms): from the last field down, every source of a needed one.
    for (int i = int(p_table.fields.size()) - 1; i >= 0; i--) {
        if ((fields & (1u << i)) == 0u) {
            continue;
        }
        for (const Term &t : p_table.fields[size_t(i)].terms) {
            fields |= t.source == GROUND ? 0u : 1u << t.source;
        }
    }
    const int n = int(p_table.fields.size());
    return 1u | (fields << 1) | (p_table.has_state() ? 1u << (n + 1) : 0u);
}

ov::str region_off_glsl(const Table &p_table, const std::vector<RegionPath> &p_paths) {
    using ExactNoise::glsl_float;
    ov::str code;
    for (const RegionPath &path : p_paths) {
        ov::str any;
        for (const Edge &e : p_table.regions[size_t(path.region)].edges) {
            any += ov::format("%scl_edge_off(f%d, %s, %s)", any.empty() ? "" : " || ", e.field, glsl_float(e.anchor), glsl_float(e.scale));
        }
        code += ov::format("    bool rof%d = %s;\n", path.region, any);
    }
    return code;
}

ov::str region_condition_glsl(const RegionPath &p_path) {
    ov::str code;
    for (int r = 0; r < MAX_REGIONS; r++) {
        if ((p_path.off & (1u << r)) != 0u) {
            code += ov::format("%srof%d", code.empty() ? "" : " && ", r);
        }
    }
    return code;
}

ov::str region_helpers_glsl() {
    // cl_edge's first lines, op for op: whether it returns vec4(0.0).
    constexpr const char GLSL_EDGE_OFF[] = R"glsl(
bool cl_edge_off(vec4 x, float anchor, float scale) {
    precise float raw0 = (x.w - anchor) * scale;
    float raw = abs(raw0) < FLUSH ? 0.0 : raw0;
    return !(raw > 0.0);
}
)glsl";
    return ov::replace_all(ov::str(GLSL_EDGE_OFF), "FLUSH", Cascade::glsl_flush());
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
