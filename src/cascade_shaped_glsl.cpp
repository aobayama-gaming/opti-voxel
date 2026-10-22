#include "cascade_shaped_glsl.hpp"

#include <godot_cpp/variant/utility_functions.hpp>

// The shaped bands' GLSL (Cascade::Modulation's BandMod features): CascadeGLSL's band functions and insert code call
// these for a shaped spec.
namespace CascadeShapedGLSL {

using namespace Cascade;

bool shaped(const Spec &p_spec) {
    return p_spec.modulation && p_spec.modulation->shaped();
}

// Group p_group's noise function: perlin3, or cell3 (Spec::cell_groups, ExactNoise::glsl_cell3).
const char *noise_function(const Spec &p_spec, int p_group) {
    return ((p_spec.cell_groups >> p_group) & 1u) ? "cell3" : "perlin3";
}

String noise_code(const Spec &p_spec, int p_group, const String &p_pos) {
    const String args = ", int(band.info.z), band.info.y, band.offset_lo.xyz, band.offset_hi.xyz);\n";
    if (!p_spec.stretched(p_group)) {
        return vformat("    vec4 g = %s(", noise_function(p_spec, p_group)) + p_pos + args;
    }
    return vformat("    vec4 g = %s(base_stretched(", noise_function(p_spec, p_group)) + p_pos + String(", int(band.info.w))") + args +
            "    g = base_stretch_gradient(g, int(band.info.w));\n";
}

String stretch_helper(const Spec &p_spec) {
    // TODO: a sphere's stretch needs its regions on the GPU (base_stretched_at with the band's base sample): the planet
    // PR that uploads them.
    ERR_FAIL_COND_V_MSG(p_spec.base.kind() != BaseSurface::Kind::FLAT, String(), "CascadeGLSL: a stretched band on a sphere base has no GPU path yet.");
    return p_spec.base.glsl_stretch();
}

// Cascade::BandMod, smooth_abs and warped (cascade.cpp), op for op; after the climate's shared helpers (cl_flush, cl_mul).
String shaped_helpers() {
    using ExactNoise::glsl_float;
    String code = R"glsl(
struct BandMod { uint features; vec4 amp[3]; vec4 state[4]; vec4 warp[3]; };

vec4 cl_smooth_abs(vec4 n, vec4 n2) {
    precise float q = n2.w + SABS_E2;
    precise float r = det_rsqrt(q);
    precise float d0 = n.w * r;
    float d = abs(d0) < FLUSH ? 0.0 : d0;
    precise vec4 res = vec4(n.x * d, n.y * d, n.z * d, q * r - SABS_E);
    return cl_flush(res);
}

WorldPos cascade_warped(WorldPos p, vec4 warp[3]) {
    WorldPos q = p;
    for (int a = 0; a < 3; a++) {
        precise float w = clamp(warp[a].w, -WARP_LIMIT, WARP_LIMIT);
        precise float fl = floor(w);
        precise float rest = w - fl;
        if (rest >= 1.0) {
            fl = fl + 1.0;
            rest = 0.0;
        }
        int di = int(fl);
        uint lo = q.lo[a] + uint(di);
        q.hi[a] = q.hi[a] + (di < 0 ? -1 : 0) + (lo < q.lo[a] ? 1 : 0);
        q.lo[a] = lo;
        q.frac[a] = rest;
    }
    return q;
}

)glsl";
    return code.replace("SABS_E2", glsl_float(SMOOTH_ABS_E * SMOOTH_ABS_E))
            .replace("SABS_E", glsl_float(SMOOTH_ABS_E))
            .replace("WARP_LIMIT", glsl_float(WARP_LIMIT))
            .replace("FLUSH", glsl_flush());
}

// Cascade::modulated_channels for group p_group's band: `Ch insert_band_shaped_<g>(Ch ch, WorldPos p, CascadeBand band,
// BandMod md, int m)`, the plain term first (insert_band_mod's order), then the shaped ones and the state (group 0). Only
// the code for the features some level of the group uses (Modulation::group_features): the rest never runs.
String shaped_band_function(const Spec &p_spec, int p_group) {
    using ExactNoise::glsl_float;
    const int base = p_spec.gpu_base(p_group);
    const uint32_t used = p_spec.modulation->group_features(p_group);
    const String slope = glsl_float(ExactNoise::exact_pow2(-p_spec.insert_shift));
    String code = vformat("Ch insert_band_shaped_%d(Ch ch, WorldPos p, CascadeBand band, BandMod md, int m) {\n", p_group);
    if (used & MOD_WARP) {
        code += "    bool warp = (md.features & " + itos(MOD_WARP) + String("u) != 0u;\n");
        code += noise_code(p_spec, p_group, "warp ? cascade_warped(p, md.warp) : p");
        code += "    if (warp) {\n        precise float tv = exact_pow2(-m);\n        vec3 d = g.xyz;\n        vec3 o;\n";
        code += "        for (int b = 0; b < 3; b++) {\n            precise float s = 0.0;\n            for (int a = 0; a < 3; a++) {\n";
        code += "                precise float j0 = md.warp[a][b] * tv;\n                float j = " + glsl_flush_of("j0") + String(";\n");
        code += "                precise float t = j * d[a];\n                s = s + (" + glsl_flush_of("t") + String(");\n            }\n");
        code += "            o[b] = d[b] + (" + glsl_flush_of("s") + String(");\n        }\n        g = vec4(o, g.w);\n    }\n");
    } else {
        code += noise_code(p_spec, p_group, "p");
    }
    code += "    precise float aws = md.amp[0].w * " + slope + String(";\n    precise vec4 o;\n");
    code += "    o.w = g.w * md.amp[0].w;\n    o.x = g.x * aws + g.w * md.amp[0].x;\n    o.y = g.y * aws + g.w * md.amp[0].y;\n    o.z = g.z * aws + g.w * md.amp[0].z;\n";
    // ch = ch + o, channel by channel (insert_band_mod's adds).
    auto adds = [&]() {
        static const char *COMP = "wxyz";
        String a;
        for (int c = 0; c < 4; c++) {
            a += vformat("    { precise float s = ch.c[%d] + o.%c; ch.c[%d] = s; }\n", base + c, COMP[c], base + c);
        }
        return a + "    return ch;\n}\n\n";
    };
    const uint32_t shaped_bits = MOD_ABS | MOD_SQ | MOD_STATE;
    if ((used & shaped_bits) == 0u) {
        return code + adds();
    }
    code += "    if ((md.features & " + itos(shaped_bits) + String("u) != 0u) {\n");
    code += "        precise vec4 nd0 = vec4(g.x * " + slope + String(", g.y * ") + slope + String(", g.z * ") + slope + String(", g.w);\n");
    code += "        vec4 nd = cl_flush(nd0);\n        vec4 n2 = cl_mul(nd, nd);\n        vec4 sh = vec4(0.0);\n";
    // |n| only where a feature reads it (the CPU's modulated_channels computes it under the same test).
    if (used & (MOD_ABS | MOD_STATE)) {
        code += "        vec4 an = (md.features & " + itos(MOD_ABS | MOD_STATE) + String("u) != 0u ? cl_smooth_abs(nd, n2) : vec4(0.0);\n");
    }
    if (used & MOD_ABS) {
        code += "        if ((md.features & " + itos(MOD_ABS) + String("u) != 0u) {\n            sh = cl_add(sh, cl_mul(md.amp[1], cl_sub(an, vec4(0.0, 0.0, 0.0, ") +
                glsl_float(NOISE_ABS_MEAN) + String("))));\n        }\n");
    }
    if (used & MOD_SQ) {
        code += "        if ((md.features & " + itos(MOD_SQ) + String("u) != 0u) {\n            sh = cl_add(sh, cl_mul(md.amp[2], cl_sub(n2, vec4(0.0, 0.0, 0.0, ") +
                glsl_float(NOISE_SQ_MEAN) + String("))));\n        }\n");
    }
    code += "        o = o + sh;\n";
    const int sg = p_spec.modulation->state_group();
    if (p_group == 0 && sg > 0 && (used & MOD_STATE)) {
        const int sb = p_spec.gpu_base(sg);
        code += "        if ((md.features & " + itos(MOD_STATE) + String("u) != 0u) {\n");
        code += "            vec4 st = md.state[0];\n            st = cl_add(st, cl_mul(md.state[1], nd));\n";
        code += "            st = cl_add(st, cl_mul(md.state[2], an));\n            st = cl_add(st, cl_mul(md.state[3], n2));\n";
        // Overwritten after this group's adds (below): the CPU writes it after every group's, the slots are disjoint.
        code += vformat("            ch.c[%d] = st.w;\n            ch.c[%d] = st.x;\n            ch.c[%d] = st.y;\n            ch.c[%d] = st.z;\n        }\n", sb, sb + 1,
                sb + 2, sb + 3);
    }
    code += "    }\n";
    return code + adds();
}

} // namespace CascadeShapedGLSL
