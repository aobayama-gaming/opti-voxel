#include "cascade_glsl.hpp"

#include "cascade_shaped_glsl.hpp"
#include "ov/str.hpp"

#include "exact_noise.hpp"
#include "exact_noise_glsl.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"

using namespace Cascade;

namespace {

constexpr const char LEVEL_TEMPLATE[] = R"glsl(#version 450

layout(local_size_x = /*LOCAL_SIZE*/, local_size_y = 1, local_size_z = 1) in;

/*STRUCTS*/

struct CascadeBand { vec4 factor; uvec4 offset_lo; ivec4 offset_hi; uvec4 info; };

layout(set = 0, binding = 0, std430) restrict readonly buffer InputChunk { ChunkInput input_chunk[]; };
layout(set = 0, binding = 1, std140) uniform ChunkParamUniform { ChunkParam chunk_param; };
/*OUTPUT_BUFFER*/
layout(set = 0, binding = 3, std430) restrict buffer ScratchA { vec4 scratch_a[]; };
layout(set = 0, binding = 4, std430) restrict buffer ScratchB { vec4 scratch_b[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer Bands { CascadeBand bands[]; };
layout(set = 0, binding = 6, std430) restrict writeonly buffer DebugChannels { vec4 debug_channels[]; };
/*TABLE_BUFFERS*/
/*MATERIAL_BUFFER*/

// One level of every chunk of a sub-batch: chunk i of the dispatch is input chunk chunk_base + i, scratch
// slot i. Workgroups: x over the level box bound n_child^3, y over the chunks.
layout(push_constant, std430) uniform CascadeParams {
    uint chunk_base;
    uint chunk_count;
    int j; // level above each chunk's LOD
    int top; // the spec's top lattice T
    uint n_child; // box bound of level j, points per axis (stride of its scratch)
    uint n_parent; // of level j + 1
    uint pad0;
    uint pad1;
} pc;

/*HELPERS*/

/*TERRAIN_SDF*/

/*CASCADE*/

void main() {
/*ITEM_INDEX*/    if (c >= pc.chunk_count || local >= n * n * n) {
        /*SKIP*/
    }
    uvec3 i = uvec3(local % n, (local / n) % n, local / (n * n));
/*ITEM_FRAME*/    int lod = f.lod;
    int m = lod + pc.j;
/*FIELDS_FLAG*/    int top = max(pc.top, lod);
    if (m > top) {
        /*SKIP*/ // above T: zero, nothing stored, the chunk starts lower
    }
    // The chunk's box on Lambda_L (Cascade::chunk_box): lo_L = anchor / 2^L - skirt, hi_L = lo_L + chunk_size - 1.
    uvec3 lo_lo = f.anchor_lo;
    ivec3 lo_hi = f.anchor_hi;
    i64_asr(lo_lo, lo_hi, lod);
    i64_add(lo_lo, lo_hi, ivec3(-chunk_param.skirt_size));
    uvec3 hi_lo = lo_lo;
    ivec3 hi_hi = lo_hi;
    i64_add(hi_lo, hi_hi, ivec3(/*ITEM_EXTENT*/));
    // Level m: lo_m = floor(lo_L / 2^j), hi_m = ceil(hi_L / 2^j) (Cascade::level_box).
    uvec3 m_lo = lo_lo;
    ivec3 m_hi = lo_hi;
    i64_asr(m_lo, m_hi, pc.j);
    i64_asr_ceil(hi_lo, hi_hi, pc.j);
    uvec3 size = hi_lo - m_lo + 1u;
    if (any(greaterThanEqual(i, size))) {
        /*SKIP*/
    }

    Ch ch;
    if (m < top) {
        // Parent box lo = floor(lo_m / 2): child lo_m + i has parent index (lo_m mod 2 + i) / 2, a midpoint
        // on the axes where that sum is odd.
        uvec3 par = (m_lo & 1u) + i;
        uvec3 p0 = par >> 1u;
        bvec3 mid = equal(par & 1u, uvec3(1u));
        uint np = pc.n_parent;
        uint pbase = c * np * np * np;
        // j + 1 odd: A, even: B.
        bool from_a = (pc.j & 1) == 0;
        Ch x00 = pass_x(from_a, pbase, np, p0, mid.x);
        Ch y0;
        if (mid.y) {
            y0 = hermite_y(x00, pass_x(from_a, pbase, np, p0 + uvec3(0u, 1u, 0u), mid.x), true);
        } else {
            y0 = hermite_y(x00, x00, false);
        }
        if (mid.z) {
            Ch x01 = pass_x(from_a, pbase, np, p0 + uvec3(0u, 0u, 1u), mid.x);
            Ch y1;
            if (mid.y) {
                y1 = hermite_y(x01, pass_x(from_a, pbase, np, p0 + uvec3(0u, 1u, 1u), mid.x), true);
            } else {
                y1 = hermite_y(x01, x01, false);
            }
            ch = hermite_z(y0, y1, true);
        } else {
            ch = hermite_z(y0, y0, false);
        }
    } else {
        ch = ch_zero();
    }

    // The level's absolute point: voxel (lo_m + i) << m.
    WorldPos p;
    p.lo = m_lo;
    p.hi = m_hi;
    i64_add(p.lo, p.hi, ivec3(i));
    i64_shl(p.lo, p.hi, m);
    p.frac = vec3(0.0);

/*INSERT_BANDS*/    ch = flush(ch);

    uint slot = c * n * n * n + local;
    if (pc.j > 0) {
        store(pc.j & 1, slot, ch);
        return;
    }
    // Lambda_L: the sample.
    uint sample_index = (pc.chunk_base + c) * n * n * n + local;
/*LEVEL0_OUTPUT*/
}
)glsl";

// The parts of LEVEL_TEMPLATE that differ between chunk items (the meshers': 32^3 samples, workgroups x over a
// level box, y over the chunks) and point items (one register per point, the items flattened over x). The chunk fills
// reproduce the template's text before point items, so the meshers' shaders are byte-identical to it.
struct ItemFills {
    const char *output_buffer;
    const char *item_index;
    const char *item_frame;
    const char *item_extent;
    const char *level0_output;
};

constexpr ItemFills CHUNK_FILLS = {
    "layout(set = 0, binding = 2, std430) restrict writeonly buffer SDFSample { vec4 sdf_sample[]; };",
    R"glsl(    uint c = gl_WorkGroupID.y;
    uint n = pc.n_child;
    uint local = gl_GlobalInvocationID.x;
)glsl",
    "    ChunkFrame f = chunk_frame(input_chunk[pc.chunk_base + c], chunk_param);\n",
    "chunk_param.chunk_size - 1",
    R"glsl(    /*DEBUG_STORE*/
    uint material = 0u; // the sample's palette index: ground unless the fold's winner says otherwise
    float skirt_extra = 0.0; // the moment skirt where a moment blob touches the sample (cascade_sample_sdf)
    precise vec4 r = /*SAMPLE_SDF*/;/*AFTER_SAMPLE*/
    // Skirt: past the chunk bounds the SDF is pushed towards air by up to the LOD's skirt depth, so the rim turns
    // down under the neighbour's surface; inside the bounds it is untouched. SDFCascade::with_skirt, same order, precise.
    if (chunk_param.skirt_size > 0 && chunk_param.skirt_depth > 0.0) {
        vec3 grid = vec3(i);
        precise float lo = float(chunk_param.skirt_size);
        precise float hi = lo + chunk_voxel_dimension(chunk_param);
        precise vec3 inside3 = min(grid - lo, hi - grid);
        precise float inside = min(inside3.x, min(inside3.y, inside3.z));
        precise float depth_m = cascade_skirt_depth_m(lod) + skirt_extra; // SDFCascade::skirt_depth_m + the moment skirt
        r.w = r.w + clamp(-inside, 0.0, 1.0) * depth_m;
    }
    sdf_sample[sample_index] = r;
    /*MATERIAL_STORE*/)glsl",
};

// A point item (GPUFrame::encode_point_item) is a chunk of one sample at any point of Λ_L, its frame from the item's
// own words (point_frame), skirt 0, box extent 0: level 0 writes the bare cascade's unit register there
// (cascade_terrain_unit: unit gradient, distance in voxels of the LOD; no instance folded: the discoverer never culls a
// node near an instance, SDFBase::cull_exempt), SDFCascade::ground_unit's bits, 16 B per point.
constexpr ItemFills POINT_FILLS = {
    "layout(set = 0, binding = 2, std430) restrict writeonly buffer PointValue { vec4 point_value[]; };",
    R"glsl(    uint n = pc.n_child;
    uint c = gl_GlobalInvocationID.x / (n * n * n);
    uint local = gl_GlobalInvocationID.x % (n * n * n);
)glsl",
    "    ChunkFrame f = point_frame(input_chunk[pc.chunk_base + c]);\n",
    "0",
    R"glsl(    precise vec4 r = cascade_terrain_unit(/*GROUND_C0*/, p, lod);
    point_value[sample_index] = r;)glsl",
};

// The fold variant's item index (it runs at level 0 only, record_level): workgroup x is a cube of FOLD_CUBE^3 samples,
// so that arch_fold tests the chunk's list once per cube. A sample's local index is the chunk fill's, so the scratch,
// sdf_sample and sdf_material keep their layout. The fold has barriers, so the template's returns turn into
// `sample_live = false` before it (/*SKIP*/) and a return after it (/*AFTER_SAMPLE*/): every invocation reaches them.
// A skipped lane still reads its chunk's frame: record_level dispatches exactly n^3 / LOCAL_SIZE = cubes^3 workgroups
// by chunk_count, so c and local stay in range (a padded dispatch would have to clamp them first).
constexpr const char FOLD_ITEM_INDEX[] = R"glsl(    uint c = gl_WorkGroupID.y;
    uint n = pc.n_child;
    uint cubes = n / CUBEu;
    uint cube_id = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationIndex;
    uvec3 cube = CUBEu * uvec3(cube_id % cubes, (cube_id / cubes) % cubes, cube_id / (cubes * cubes));
    uvec3 at = cube + uvec3(lane % CUBEu, (lane / CUBEu) % CUBEu, lane / (CUBEu * CUBEu));
    uint local = at.x + n * (at.y + n * at.z);
    bool sample_live = true;
)glsl";

// Point i of a level's box, (lo_m + i) << m: the template's p, for the corners of the fold variant's cube.
constexpr const char LEVEL_POINT[] = R"glsl(
WorldPos level_point(uvec3 m_lo, ivec3 m_hi, uvec3 i, int m) {
    WorldPos p;
    p.lo = m_lo;
    p.hi = m_hi;
    i64_add(p.lo, p.hi, ivec3(i));
    i64_shl(p.lo, p.hi, m);
    p.frac = vec3(0.0);
    return p;
}
)glsl";

ov::str flt(float p_f) {
    return ExactNoise::glsl_float(p_f);
}

// Ch and hermite_x/y/z for a spec's groups (from Cascade::PAIRS, the CPU's axis_rule per group).
ov::str hermite_functions(const Spec &p_spec) {
    const int nch = p_spec.gpu_channels();
    ov::str code = ov::format("struct Ch { float c[%d]; };\n\n", nch);
    code += "Ch ch_zero() {\n    Ch r;\n";
    for (int c = 0; c < nch; c++) {
        code += ov::format("    r.c[%d] = 0.0;\n", c);
    }
    code += "    return r;\n}\n\n";
    // Cascade::hermite_copy / hermite_mid, op for op.
    code += R"glsl(void hcopy(float a, float sa, out float v, out float s) {
    precise float ps = sa * 0.5;
    v = a;
    s = ps;
}

void hmid(float a, float b, float sa, float sb, out float v, out float s) {
    precise float pv = (a + b) * 0.5 + (sa - sb) * 0.125;
    precise float ps = (b - a) * 0.75 - (sa + sb) * 0.125;
    v = pv;
    s = ps;
}

)glsl";
    static const char *AXIS_NAMES[3] = { "x", "y", "z" };
    for (int axis = 0; axis < 3; axis++) {
        ov::str mid_code;
        ov::str copy_code;
        auto pair = [&](int p_v, int p_s) {
            mid_code += ov::format("        hmid(a.c[%d], b.c[%d], a.c[%d], b.c[%d], r.c[%d], r.c[%d]);\n", p_v, p_v, p_s, p_s, p_v, p_s);
            copy_code += ov::format("        hcopy(a.c[%d], a.c[%d], r.c[%d], r.c[%d]);\n", p_v, p_s, p_v, p_s);
        };
        for (int g = 0; g < p_spec.group_count(); g++) {
            const int base = p_spec.gpu_base(g);
            if (g == 1) {
                // The fields only where something reads them (Cascade::FieldRange).
                mid_code += "        if (g_fields) {\n";
                copy_code += "        if (g_fields) {\n";
            }
            if (p_spec.group_set(g) == ChannelSet::HERMITE8) {
                for (const auto &pr : PAIRS[axis]) {
                    pair(base + pr[0], base + pr[1]);
                }
                continue;
            }
            // LINEAR4: value + the axis slope by the Hermite rule, the transverse slopes linear.
            pair(base + VALUE, base + DX + axis);
            for (int s = base + DX; s <= base + DZ; s++) {
                if (s != base + DX + axis) {
                    mid_code += ov::format("        { precise float t = (a.c[%d] + b.c[%d]) * 0.5; r.c[%d] = t; }\n", s, s, s);
                    copy_code += ov::format("        r.c[%d] = a.c[%d];\n", s, s);
                }
            }
        }
        if (p_spec.group_count() > 1) {
            mid_code += "        }\n";
            copy_code += "        }\n";
        }
        code += ov::format("Ch hermite_%s(Ch a, Ch b, bool mid) {\n    Ch r;\n    if (mid) {\n%s    } else {\n%s    }\n    return r;\n}\n\n", AXIS_NAMES[axis],
                mid_code, copy_code);
    }
    return code;
}

// vec4s holding p_channels floats: the scratch's and Ch's packing.
uint32_t vec4s_for(int p_channels) {
    return uint32_t(p_channels + 3) / 4;
}

// Scratch and debug buffer access (nv vec4s per point) and the x pass at a parent position. The scratch is planar:
// vec4 v of slot s at v * plane + s, plane = the buffer's vec4s / p_stride, so the ground's loads stay contiguous
// whatever the fields; the debug channels are per point (slot * p_stride + v). p_stride: vec4s_per_point of the whole
// spec (the ground-only variant reads and writes the ground's planes of the same buffers).
ov::str storage_functions(const Spec &p_spec, int p_stride) {
    const int nch = p_spec.gpu_channels();
    const int nv = int(vec4s_for(nch)); // this variant's channels (the view: fields only in the fields variant)
    ov::str code = "Ch unpack(";
    for (int v = 0; v < nv; v++) {
        code += ov::format("%svec4 v%d", v ? ", " : "", v);
    }
    code += ") {\n    Ch r;\n";
    for (int c = 0; c < nch; c++) {
        code += ov::format("    r.c[%d] = v%d[%d];\n", c, c / 4, c % 4);
    }
    code += "    return r;\n}\n\n";
    code += "Ch load(bool from_a, uint slot) {\n";
    ov::str args_a;
    ov::str args_b;
    // vec4s from ground_vecs on are the fields' (4 channels each), read only where g_fields.
    const int ground_vecs = p_spec.gpu_base(1) / 4;
    for (int v = 0; v < nv; v++) {
        const char *sep = v ? ", " : "";
        if (v < ground_vecs) {
            args_a += ov::format("%sscratch_a[%du * plane_a + slot]", sep, v);
            args_b += ov::format("%sscratch_b[%du * plane_b + slot]", sep, v);
        } else {
            args_a += ov::format("%s(g_fields ? scratch_a[%du * plane_a + slot] : vec4(0.0))", sep, v);
            args_b += ov::format("%s(g_fields ? scratch_b[%du * plane_b + slot] : vec4(0.0))", sep, v);
        }
    }
    const ov::str planes = ov::format("    uint plane_a = uint(scratch_a.length()) / %du;\n    uint plane_b = uint(scratch_b.length()) / %du;\n", p_stride, p_stride);
    code += planes + ov::format("    if (from_a) {\n        return unpack(%s);\n    }\n    return unpack(%s);\n}\n\n", args_a, args_b);
    ov::str vecs;
    for (int v = 0; v < nv; v++) {
        vecs += ov::format("    vec4 v%d = vec4(r.c[%d], r.c[%d], r.c[%d], r.c[%d]);\n", v, 4 * v, 4 * v + 1, 4 * v + 2, 4 * v + 3);
    }
    code += "void store(int odd, uint slot, Ch r) {\n" + planes + vecs;
    for (int v = 0; v < nv; v++) {
        const ov::str guard = v < ground_vecs ? ov::str() : ov::str("    if (g_fields)\n");
        code += guard + ov::format("    if (odd != 0) {\n        scratch_a[%du * plane_a + slot] = v%d;\n    } else {\n        scratch_b[%du * plane_b + slot] = v%d;\n    }\n", v, v, v, v);
    }
    code += "}\n\n";
    code += "void store_debug(uint slot, Ch r) {\n" + vecs;
    for (int v = 0; v < nv; v++) {
        code += ov::format("    debug_channels[slot * %du + %du] = v%d;\n", p_stride, v, v);
    }
    code += "}\n\n";
    // The x pass at parent position q (and q + x on a midpoint).
    code += R"glsl(Ch pass_x(bool from_a, uint pbase, uint np, uvec3 q, bool mid) {
    uint slot = pbase + q.x + np * (q.y + np * q.z);
    Ch a = load(from_a, slot);
    if (mid) {
        return hermite_x(a, load(from_a, slot + 1u), true);
    }
    return hermite_x(a, a, false);
}

)glsl";
    return code;
}

using CascadeShapedGLSL::noise_code;
using CascadeShapedGLSL::shaped;
using CascadeShapedGLSL::shaped_band_function;
using CascadeShapedGLSL::shaped_helpers;

// Cascade's insert_point for group g's band: insert_band_<g> at its constant amplitude (o = channel * factor[order],
// ch = ch + o: two roundings, precise), insert_band_mod_<g> at a modulated one (Cascade::modulated_channels, the same
// order); and Cascade::flush_value.
ov::str band_functions(const Spec &p_spec) {
    const int nch = p_spec.gpu_channels();
    ov::str code = p_spec.stretched() ? CascadeShapedGLSL::stretch_helper(p_spec) : ov::str();
    for (int g = 0; g < p_spec.group_count(); g++) {
        const int base = p_spec.gpu_base(g);
        const ChannelSet set = p_spec.group_set(g);
        code += ov::format("Ch insert_band_%d(Ch ch, WorldPos p, CascadeBand band) {\n", g);
        if (set == ChannelSet::HERMITE8) {
            code += "    vec4 c0;\n    vec4 c1;\n    perlin3_cross(p, int(band.info.z), band.info.y, band.offset_lo.xyz, band.offset_hi.xyz, c0, c1);\n";
            code += "    float n[8] = float[8](c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);\n";
        } else {
            // perlin3 returns (d/dx, d/dy, d/dz, value): the same bits as perlin3_cross's first four channels.
            code += noise_code(p_spec, g, "p");
            code += "    float n[4] = float[4](g.w, g.x, g.y, g.z);\n";
        }
        for (int c = 0; c < channel_count(set); c++) {
            code += ov::format("    { precise float o = n[%d] * band.factor[%d]; precise float s = ch.c[%d] + o; ch.c[%d] = s; }\n", c, channel_order(c), base + c,
                    base + c);
        }
        code += "    return ch;\n}\n\n";
        if (shaped(p_spec) && set == ChannelSet::LINEAR4 && p_spec.modulation->group_features(g) != 0u) {
            code += shaped_band_function(p_spec, g);
        } else if (p_spec.modulation && set == ChannelSet::LINEAR4) {
            code += ov::format("Ch insert_band_mod_%d(Ch ch, WorldPos p, CascadeBand band, vec4 amp) {\n", g);
            code += noise_code(p_spec, g, "p");
            code += ov::format("    precise float aws = amp.w * %s;\n", flt(ExactNoise::exact_pow2(-p_spec.insert_shift)));
            code += ov::format("    { precise float o = g.w * amp.w; precise float s = ch.c[%d] + o; ch.c[%d] = s; }\n", base, base);
            static const char *AXES = "xyz";
            for (int a = 0; a < 3; a++) {
                code += ov::format("    { precise float o = g.%c * aws + g.w * amp.%c; precise float s = ch.c[%d] + o; ch.c[%d] = s; }\n", AXES[a], AXES[a], base + 1 + a,
                        base + 1 + a);
            }
            code += "    return ch;\n}\n\n";
        }
    }
    code += "Ch flush(Ch ch) {\n";
    for (int c = 0; c < nch; c++) {
        const ov::str v = ov::format("ch.c[%d]", c);
        code += "    " + v + " = " + glsl_flush_of(v) + ";\n";
    }
    code += "    return ch;\n}\n";
    return code;
}

// The level's insertions (the template's /*INSERT_BANDS*/): the modulated amplitudes from the channels before any
// insertion, then each group's band of level m (bands[group * BAND_ENTRIES + m]).
ov::str insert_code(const Spec &p_spec) {
    ov::str code;
    if (shaped(p_spec)) {
        code += ov::format("    BandMod mod[%d];\n    uint modulated = cascade_mods(m, ch, mod);\n", p_spec.group_count());
    } else if (p_spec.modulation) {
        code += ov::format("    vec4 amp[%d];\n    uint modulated = cascade_amplitudes(m, ch, amp);\n", p_spec.group_count());
    }
    for (int g = 0; g < p_spec.group_count(); g++) {
        if (p_spec.group_bands(g).empty()) {
            continue;
        }
        code += ov::format("    %s{\n        CascadeBand band = bands[%d + m];\n        if (band.info.x != 0u) {\n", g > 0 ? "if (g_fields) " : "",
                g * CascadeGLSL::BAND_ENTRIES);
        if (shaped(p_spec) && p_spec.group_set(g) == ChannelSet::LINEAR4) {
            const bool own = p_spec.modulation->group_features(g) != 0u;
            const ov::str call = own ? ov::format("insert_band_shaped_%d(ch, p, band, mod[%d], m)", g, g) : ov::format("insert_band_mod_%d(ch, p, band, mod[%d].amp[0])", g, g);
            code += ov::format("            ch = ((modulated >> %du) & 1u) != 0u ? %s : insert_band_%d(ch, p, band);\n", g, call, g);
        } else if (p_spec.modulation && p_spec.group_set(g) == ChannelSet::LINEAR4) {
            code += ov::format("            ch = ((modulated >> %du) & 1u) != 0u ? insert_band_mod_%d(ch, p, band, amp[%d]) : insert_band_%d(ch, p, band);\n", g, g, g, g);
        } else {
            code += ov::format("            ch = insert_band_%d(ch, p, band);\n", g);
        }
        code += "        }\n    }\n";
    }
    return code;
}

} // namespace

namespace CascadeGLSL {

namespace {

const char *const GLSL_VERTEX_GRADIENT = R"glsl(

// Cascade mode: the cascade only exists on lattice points, so the vertex normal is the trilinear blend of
// the samples' gradients (sdf_sample.xyz, the normalised cascade gradient) at the cell's 8 corners.
vec3 cascade_vertex_gradient(uvec3 gid, vec3 f) {
    vec3 g000 = sdf_sample[get_flat_index_z(gid)].xyz;
    vec3 g100 = sdf_sample[get_flat_index_z(gid + uvec3(1u, 0u, 0u))].xyz;
    vec3 g010 = sdf_sample[get_flat_index_z(gid + uvec3(0u, 1u, 0u))].xyz;
    vec3 g110 = sdf_sample[get_flat_index_z(gid + uvec3(1u, 1u, 0u))].xyz;
    vec3 g001 = sdf_sample[get_flat_index_z(gid + uvec3(0u, 0u, 1u))].xyz;
    vec3 g101 = sdf_sample[get_flat_index_z(gid + uvec3(1u, 0u, 1u))].xyz;
    vec3 g011 = sdf_sample[get_flat_index_z(gid + uvec3(0u, 1u, 1u))].xyz;
    vec3 g111 = sdf_sample[get_flat_index_z(gid + uvec3(1u, 1u, 1u))].xyz;
    vec3 y0 = mix(mix(g000, g100, f.x), mix(g010, g110, f.x), f.y);
    vec3 y1 = mix(mix(g001, g101, f.x), mix(g011, g111, f.x), f.y);
    return mix(y0, y1, f.z);
}

// The vertex's material (its chunk palette index): the material of the solid corner closest to the surface (the
// largest w among the corners with !(w > 0), the side rule every pass uses), ties to the first in x + 2y + 4z order.
// A cell with a vertex has a solid corner.
uint cascade_vertex_material(uvec3 gid) {
    uint best = get_flat_index_z(gid);
    float best_w = 0.0;
    bool found = false;
    for (uint k = 0u; k < 8u; k++) {
        uint idx = get_flat_index_z(gid + uvec3(k & 1u, (k >> 1u) & 1u, k >> 2u));
        float w = sdf_sample[idx].w;
        if (!(w > 0.0) && (!found || w > best_w)) {
            found = true;
            best_w = w;
            best = idx;
        }
    }
    return sdf_material[best];
}

)glsl";

} // namespace

ov::bytes band_table(const Spec &p_spec) {
    using namespace WorldCoord;
    ov::bytes bytes(size_t(p_spec.group_count()) * BAND_ENTRIES * BAND_BYTES, 0);
    for (int g = 0; g < p_spec.group_count(); g++) {
        for (const Band &b : p_spec.group_bands(g)) {
            const int m = p_spec.insert_level(b);
            OV_ERR_CONTINUE_MSG(m < 0 || m >= BAND_ENTRIES, ov::format("CascadeGLSL: band k=%d inserted outside the lattice table.", b.k));
            const int64_t base = (int64_t(g) * BAND_ENTRIES + m) * BAND_BYTES;
            OV_ERR_CONTINUE_MSG(ov::get_u32(&bytes[base + 48]) != 0, ov::format("CascadeGLSL: two bands on lattice %d (the GPU table holds one per level).", m));
            for (int o = 0; o < 4; o++) {
                // Cascade's BandFactors, the same float product.
                ov::put_f32(&bytes[base + 4 * o], b.amp * ExactNoise::exact_pow2(-p_spec.insert_shift * o));
            }
            ov::put_u32(&bytes[base + 16], lo32(b.offset.x));
            ov::put_u32(&bytes[base + 20], lo32(b.offset.y));
            ov::put_u32(&bytes[base + 24], lo32(b.offset.z));
            ov::put_s32(&bytes[base + 32], hi32(b.offset.x));
            ov::put_s32(&bytes[base + 36], hi32(b.offset.y));
            ov::put_s32(&bytes[base + 40], hi32(b.offset.z));
            ov::put_u32(&bytes[base + 48], 1);
            ov::put_u32(&bytes[base + 52], b.seed);
            ov::put_u32(&bytes[base + 56], uint32_t(b.k));
            ov::put_u32(&bytes[base + 60], uint32_t(b.stretch));
        }
    }
    return bytes;
}

uint32_t scratch_points_a(int p_item_size) {
    const uint32_t n = uint32_t(level_bound(p_item_size, 1));
    return n * n * n;
}

uint32_t scratch_points_b(int p_item_size) {
    const uint32_t n = uint32_t(level_bound(p_item_size, 2));
    return n * n * n;
}

uint32_t vec4s_per_point(const Spec &p_spec, const FieldRange &p_range) {
    // Fields carried nowhere (an empty FieldRange) take no scratch.
    const int channels = p_range.empty() ? channel_count(p_spec.set) : p_spec.gpu_channels();
    return vec4s_for(channels);
}

void scratch_bytes(const CascadeSource *p_sdf, int p_item_size, uint64_t p_items, uint64_t &r_a, uint64_t &r_b) {
    const uint64_t vec4_bytes = 16 * uint64_t(vec4s_per_point(*p_sdf->get_cascade(), field_range(p_sdf, p_item_size == 1)));
    r_a = p_items * scratch_points_a(p_item_size) * vec4_bytes;
    r_b = p_items * scratch_points_b(p_item_size) * vec4_bytes;
}

bool ground_materials(const CascadeSource *p_sdf) {
    return p_sdf->samples_carry_materials();
}

FieldRange field_range(const CascadeSource *p_sdf, bool p_points) {
    const Spec &spec = *p_sdf->get_cascade();
    // The climate variant (modulation, ground terms, the pick where there are fields) wherever a level needs it: a one-biome
    // table without fields still modulates the ground, and a terrace is a ground term. Ground terms read the fields at
    // every sample, point items too.
    const bool everywhere = (!p_points && ground_materials(p_sdf)) || p_sdf->has_ground_terms();
    return everywhere ? FieldRange{ 0, MAX_LEVEL } : spec.field_range();
}

ov::str level_shader(const CascadeSource *p_sdf, bool p_debug, bool p_points, bool p_fold, bool p_material, bool p_fields, bool p_cubes) {
    const Spec &whole = *p_sdf->get_cascade();
    // Without the fields: the ground alone (its Ch, its bands, no modulation) on the whole spec's scratch planes.
    const Spec view = p_fields ? whole : whole.ground_only();
    const Spec *spec = &view;
    OV_ERR_FAIL_COND_V_MSG(p_debug && p_points, ov::str(), "CascadeGLSL: point items have no debug variant.");
    OV_ERR_FAIL_COND_V_MSG((p_fold || p_material) && p_points, ov::str(), "CascadeGLSL: only chunk items fold or carry a material.");
    const ItemFills &fills = p_points ? POINT_FILLS : CHUNK_FILLS;
    // The fold per cube of samples (FOLD_ITEM_INDEX), or per sample (the chunk's item index) without whole cubes.
    const bool cubes = p_fold && p_cubes;
    ov::str code = LEVEL_TEMPLATE;
    code = ov::replace_all(code, "/*LOCAL_SIZE*/", ov::itos(LOCAL_SIZE));
    code = ov::replace_all(code, "/*OUTPUT_BUFFER*/", fills.output_buffer);
    code = ov::replace_all(code, "/*ITEM_INDEX*/", cubes ? ov::replace_all(ov::str(FOLD_ITEM_INDEX), "CUBE", ov::itos(FOLD_CUBE)) : ov::str(fills.item_index));
    code = ov::replace_all(code, "/*SKIP*/", cubes ? "sample_live = false;" : "return;");
    code = ov::replace_all(code, "/*ITEM_FRAME*/", fills.item_frame);
    code = ov::replace_all(code, "/*ITEM_EXTENT*/", fills.item_extent);
    code = ov::replace_all(code, "/*LEVEL0_OUTPUT*/", fills.level0_output);
    code = ov::replace_all(code, "/*AFTER_SAMPLE*/", cubes ? "\n    if (!sample_live) {\n        return; // past the fold's barriers\n    }" : "");
    // The sample's ground channels: the cascade's, or climate_ground's with ground terms (the fields variant only: their
    // field range is every level, so it is the only one compiled).
    const ov::str ground = p_fields && p_sdf->has_ground_terms() ? p_sdf->get_ground_glsl() : ov::str();
    const ov::str c0 = ground.empty() ? ov::str("vec4(ch.c[0], ch.c[1], ch.c[2], ch.c[3])") : ov::str("climate_ground(ch)");
    code = ov::replace_all(code, "/*GROUND_C0*/", c0);
    const ov::str channels = c0 + ov::str(", p, lod");
    // The fold's workgroup box: its cube's first and last samples (the sample itself per sample).
    const ov::str cube_box = cubes ? ov::format("level_point(m_lo, m_hi, cube, m), level_point(m_lo, m_hi, cube + %du, m)", FOLD_CUBE - 1) : ov::str("p, p");
    code = ov::replace_all(code, "/*SAMPLE_SDF*/", p_fold ? ov::format("cascade_sample_sdf(%s, %s, pc.chunk_base + c, material, skirt_extra)", channels, cube_box) : ov::format("cascade_terrain_sdf(%s)", channels));
    // Without ground materials only the fold writes materials, and only for chunks with a list: a chunk without one is
    // ground, and the meshers don't read its samples' materials (its palette is the ground block, or its list is empty).
    // With them every chunk's samples get one: the winner's, or the ground's pick, read from the fields (field_range
    // then carries them everywhere, so the fields variant is the only one CascadePasses compiles).
    const bool grounds = ground_materials(p_sdf) && !p_points;
    const bool store = p_material && (p_fold || grounds);
    const bool pick = grounds && p_fields && !whole.fields.empty();
    code = ov::replace_all(code, "/*MATERIAL_BUFFER*/", store ? glsl_material_buffer(MATERIAL_BINDING, "writeonly") : ov::str());
    ov::str material_store;
    if (store && grounds) {
        // Air samples (w > 0, after the skirt) keep 0: the meshers read a vertex's material from its cell's solid corners.
        material_store = ov::str(pick ? "if (material == 0u && !(r.w > 0.0)) {\n        material = cascade_ground_material(ch, p);\n    }\n    " : "") +
                "sdf_material[sample_index] = material;";
    } else if (store) {
        material_store = "if (g_sdf_chunk.anchor_hi.w > 0) {\n        sdf_material[sample_index] = material;\n    }";
    }
    code = ov::replace_all(code, "/*MATERIAL_STORE*/", material_store);
    code = ov::replace_all(code, "/*STRUCTS*/", ov::str(GLSL_STRUCT_DEFINITION) + ov::str(GLSL_FRAME_STRUCTS));
    code = ov::replace_all(code, "/*TABLE_BUFFERS*/", p_fold ? p_sdf->get_table_glsl(TABLE_RECORD_BINDING, TABLE_LIST_BINDING) : ov::str());
    code = ov::replace_all(code, "/*HELPERS*/", ov::str(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + (p_points ? GPUFrame::glsl_point_frame() : ov::str()) + GPUFrame::glsl_int64() + ov::str(GLSL_EXACT_NOISE) + ov::str(GLSL_PERLIN3_CROSS) + (spec->cell_groups ? glsl_cell3() : ov::str()));
    code = ov::replace_all(code, "/*TERRAIN_SDF*/", p_sdf->get_glsl_code() + (p_fold ? (cubes ? ov::str(LEVEL_POINT) : ov::str()) + p_sdf->get_sample_fold_glsl(cubes ? LOCAL_SIZE : 0) : ov::str()));
    const ov::str modulation = spec->modulation ? spec->modulation->glsl(*spec) : ov::str();
    const bool fields = !spec->fields.empty();
    const FieldRange range = field_range(p_sdf, p_points);
    const ov::str fields_flag = fields ? ov::str("bool g_fields = true; // the fields are carried at this level (Cascade::FieldRange)\n\n") : ov::str();
    // The climate's shared helpers (with a shaped modulation or ground terms) and the shaped inserts' (BandMod), before
    // every generated function that uses them.
    const ov::str helpers = (p_fields ? p_sdf->get_climate_helpers_glsl() : ov::str()) + (shaped(*spec) ? shaped_helpers() : ov::str());
    code = ov::replace_all(code, "/*CASCADE*/", fields_flag + helpers + hermite_functions(*spec) + storage_functions(*spec, int(vec4s_per_point(whole, range))) + band_functions(*spec) + modulation + (pick ? p_sdf->get_ground_material_glsl() : ov::str()) + ground);
    code = ov::replace_all(code, "/*INSERT_BANDS*/", insert_code(*spec));
    code = ov::replace_all(code, "/*FIELDS_FLAG*/", fields ? ov::format("    g_fields = m >= %d && lod <= %d; // FieldRange::carried\n", range.floor, range.ceiling) : ov::str());

    const ov::str debug_store = ov::str(cubes ? "if (sample_live) " : "") + "store_debug(sample_index, ch); // CascadeSpike variant";
    code = ov::replace_all(code, "/*DEBUG_STORE*/", p_debug ? debug_store : ov::str());
    return code;
}

ChunkParam point_param(const ChunkParam &p_param) {
    ChunkParam p = p_param;
    p.skirt_size = 0;
    return p;
}

ov::str glsl_material_buffer(int p_binding, const char *p_access) {
    return ov::format("layout(set = 0, binding = %d, std430) restrict %s buffer SDFMaterial { uint sdf_material[]; };\n", p_binding, p_access);
}

ov::str fill_mesher_pass(const ov::str &p_shader, const ov::str &p_pass) {
    return ov::replace_all(p_shader, { { "/*GLSL_VERTEX_GRADIENT*/", GLSL_VERTEX_GRADIENT }, { "/*GLSL_ALGORITHM_PASS*/", p_pass } });
}

} // namespace CascadeGLSL
