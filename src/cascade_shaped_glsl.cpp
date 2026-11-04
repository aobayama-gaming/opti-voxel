#include "cascade_shaped_glsl.hpp"

#include "level_program.hpp"

// The shaped bands' GLSL (Cascade::Modulation's BandMod features): CascadeGLSL's band functions and insert code call
// these for a shaped spec.
namespace CascadeShapedGLSL {

using namespace Cascade;

ov::str noise_code(const Spec &p_spec, int p_group, const ov::str &p_pos) {
    const ov::str args = ", int(band.info.z), band.info.y, band.offset_lo.xyz, band.offset_hi.xyz);\n";
    if (!p_spec.stretched(p_group)) {
        return "    vec4 g = perlin3(" + p_pos + args;
    }
    return "    vec4 g = perlin3(base_stretched(" + p_pos + ov::str(", int(band.info.w))") + args +
            "    g = base_stretch_gradient(g, int(band.info.w));\n";
}

ov::str stretch_helper(const Spec &p_spec) {
    // TODO: a curved base's stretch needs the regions of the band's insertion LOD on the GPU (base_stretched_at with the band's
    // base sample; BaseRegions uploads the chunk's LOD only). SDFCascade refuses such a climate on a curved base.
    OV_ERR_FAIL_COND_V_MSG(p_spec.base.curved(), ov::str(), "CascadeGLSL: a stretched band on a curved base has no GPU path yet.");
    return p_spec.base.glsl_stretch();
}

// Cascade::BandMod, smooth_abs and warped (cascade.cpp), op for op; after the climate's shared helpers (cl_flush, cl_mul).
ov::str shaped_helpers() {
    using ExactNoise::glsl_float;
    ov::str code = R"glsl(
struct BandMod { uint features; vec4 amp[3]; vec4 warp[3]; };

)glsl";
    return code + Terrain::glsl_warp_helpers(); // cl_smooth_abs, cascade_warped: the v2 op table's
}

// Cascade::modulated_channels for group p_group's band: `Ch insert_band_shaped_<g>(Ch ch, WorldPos p, CascadeBand band,
// BandMod md, int m)`, the plain term first (modulated_channels' order), then the shaped ones. Only
// the code for the features some level of the group uses (Modulation::group_features): the rest never runs.
ov::str shaped_band_function(const Spec &p_spec, int p_group) {
    using ExactNoise::glsl_float;
    const int base = p_spec.gpu_base(p_group);
    const uint32_t used = p_spec.modulation->group_features(p_group);
    const ov::str slope = glsl_float(ExactNoise::exact_pow2(-p_spec.insert_shift));
    ov::str code = ov::format("Ch insert_band_shaped_%d(Ch ch, WorldPos p, CascadeBand band, BandMod md, int m) {\n", p_group);
    if (used & MOD_WARP) {
        code += "    bool warp = (md.features & " + ov::itos(MOD_WARP) + ov::str("u) != 0u;\n");
        code += noise_code(p_spec, p_group, "warp ? cascade_warped(p, md.warp) : p");
        code += "    if (warp) {\n        precise float tv = exact_pow2(-m);\n        vec3 d = g.xyz;\n        vec3 o;\n";
        code += "        for (int b = 0; b < 3; b++) {\n            precise float s = 0.0;\n            for (int a = 0; a < 3; a++) {\n";
        code += "                precise float j0 = md.warp[a][b] * tv;\n                float j = " + glsl_flush_of("j0") + ov::str(";\n");
        code += "                precise float t = j * d[a];\n                s = s + (" + glsl_flush_of("t") + ov::str(");\n            }\n");
        code += "            o[b] = d[b] + (" + glsl_flush_of("s") + ov::str(");\n        }\n        g = vec4(o, g.w);\n    }\n");
    } else {
        code += noise_code(p_spec, p_group, "p");
    }
    code += "    precise float aws = md.amp[0].w * " + slope + ov::str(";\n    precise vec4 o;\n");
    code += "    o.w = g.w * md.amp[0].w;\n    o.x = g.x * aws + g.w * md.amp[0].x;\n    o.y = g.y * aws + g.w * md.amp[0].y;\n    o.z = g.z * aws + g.w * md.amp[0].z;\n";
    // ch = ch + o, channel by channel (two roundings, as insert_band's).
    auto adds = [&]() {
        static const char *COMP = "wxyz";
        ov::str a;
        for (int c = 0; c < 4; c++) {
            a += ov::format("    { precise float s = ch.c[%d] + o.%c; ch.c[%d] = s; }\n", base + c, COMP[c], base + c);
        }
        return a + "    return ch;\n}\n\n";
    };
    const uint32_t shaped_bits = MOD_ABS | MOD_SQ;
    if ((used & shaped_bits) == 0u) {
        return code + adds();
    }
    code += "    if ((md.features & " + ov::itos(shaped_bits) + ov::str("u) != 0u) {\n");
    code += "        precise vec4 nd0 = vec4(g.x * " + slope + ov::str(", g.y * ") + slope + ov::str(", g.z * ") + slope + ov::str(", g.w);\n");
    code += "        vec4 nd = cl_flush(nd0);\n        vec4 n2 = cl_mul(nd, nd);\n        vec4 sh = vec4(0.0);\n";
    // |n| only where a feature reads it (the CPU's modulated_channels computes it under the same test).
    if (used & MOD_ABS) {
        code += "        vec4 an = (md.features & " + ov::itos(MOD_ABS) + ov::str("u) != 0u ? cl_smooth_abs(nd, n2) : vec4(0.0);\n");
        code += "        if ((md.features & " + ov::itos(MOD_ABS) + ov::str("u) != 0u) {\n            sh = cl_add(sh, cl_mul(md.amp[1], cl_sub(an, vec4(0.0, 0.0, 0.0, ") +
                glsl_float(NOISE_ABS_MEAN) + ov::str("))));\n        }\n");
    }
    if (used & MOD_SQ) {
        code += "        if ((md.features & " + ov::itos(MOD_SQ) + ov::str("u) != 0u) {\n            sh = cl_add(sh, cl_mul(md.amp[2], cl_sub(n2, vec4(0.0, 0.0, 0.0, ") +
                glsl_float(NOISE_SQ_MEAN) + ov::str("))));\n        }\n");
    }
    code += "        o = o + sh;\n";
    code += "    }\n";
    return code + adds();
}

} // namespace CascadeShapedGLSL
