#include "cascade_glsl.hpp"

#include "climate.hpp"
#include "level_glsl.hpp"

#include "exact_noise.hpp"
#include "exact_noise_glsl.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"
#include "ov/bytes.hpp"
#include "ov/str.hpp"

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
/*PARAM_BLOCK*/
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
    int top = max(pc.top, lod);
    if (m > top) {
        /*SKIP*/ // above T: zero, nothing stored, the chunk starts lower
    }
    // The chunk's box on Λ_L (Cascade::chunk_box): lo_L = anchor / 2^L - skirt, hi_L = lo_L + chunk_size - 1.
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

    // Where subdivide reads the level above (read only below the top): parent box lo = floor(lo_m / 2), so child
    // lo_m + i has parent index (lo_m mod 2 + i) / 2, a midpoint on the axes where that sum is odd. j + 1 odd: A,
    // even: B; this level stores in the other one.
    uvec3 par = (m_lo & 1u) + i;
    g_p0 = par >> 1u;
    g_mid = equal(par & 1u, uvec3(1u));
    g_np = pc.n_parent;
    g_pbase = c * g_np * g_np * g_np;
    g_from_a = (pc.j & 1) == 0;
    g_slot = c * n * n * n + local;

    // The level's absolute point: voxel (lo_m + i) << m.
    WorldPos p;
    p.lo = m_lo;
    p.hi = m_hi;
    i64_add(p.lo, p.hi, ivec3(i));
    i64_shl(p.lo, p.hi, m);
    p.frac = vec3(0.0);

    vec4 ground;
    float pick;
    level_run(p, m, pc.j == 0, ground, pick);
    if (pc.j > 0) {
        return;
    }
    // Λ_L: the sample.
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
// node near an instance, CascadeSource::cull_exempt), SDFCascade::ground_unit's bits, 16 B per point.
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

// The subdivision of the level above, one channel at a time (Cascade::axis_rule, op for op), and the scratch access. A
// channel is a vec4 in Cascade::Channel order (x the value, y z w its slopes along x y z); a register is a dual (x y z
// the gradient, w the value). The scratch is planar: channel k of slot s at k * plane + s, plane = the buffer's vec4s /
// p_nch. main() sets the g_ globals: the point's floor parent g_p0 and its midpoint axes g_mid in the level above (g_np
// points per axis, the chunk's first at g_pbase, in A where g_from_a; this level stores in the other one) and its own
// slot g_slot.
ov::str subdivision_functions(int p_nch) {
    static_assert(VALUE == 0 && DX == 1 && DY == 2 && DZ == 3, "CascadeGLSL: a channel's vec4 is value, dx, dy, dz.");
    ov::str code = ov::replace_all(R"glsl(uvec3 g_p0 = uvec3(0u);
bvec3 g_mid = bvec3(false);
uint g_np = 0u;
uint g_pbase = 0u;
bool g_from_a = false;
uint g_slot = 0u;

vec4 ch_load(uint k, uint slot) {
    if (g_from_a) {
        return scratch_a[k * (uint(scratch_a.length()) / NCHu) + slot];
    }
    return scratch_b[k * (uint(scratch_b.length()) / NCHu) + slot];
}

// Register v as this level's channel k.
void ch_store(uint k, vec4 v) {
    if (g_from_a) {
        scratch_b[k * (uint(scratch_b.length()) / NCHu) + g_slot] = v.wxyz;
    } else {
        scratch_a[k * (uint(scratch_a.length()) / NCHu) + g_slot] = v.wxyz;
    }
}

// Cascade::hermite_copy / hermite_mid, op for op.
void hcopy(float a, float sa, out float v, out float s) {
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

)glsl",
            "NCH", ov::itos(p_nch));
    // Along one axis: the value and that axis' slope by the Hermite rule, the transverse slopes linear.
    for (int axis = 0; axis < 3; axis++) {
        const char slope = "yzw"[axis];
        ov::str mid = ov::format("        hmid(a.x, b.x, a.%c, b.%c, r.x, r.%c);\n", slope, slope, slope);
        ov::str copy = ov::format("        hcopy(a.x, a.%c, r.x, r.%c);\n", slope, slope);
        for (const char o : { 'y', 'z', 'w' }) {
            if (o != slope) {
                mid += ov::format("        { precise float t = (a.%c + b.%c) * 0.5; r.%c = t; }\n", o, o, o);
                copy += ov::format("        r.%c = a.%c;\n", o, o);
            }
        }
        code += ov::format("vec4 hermite_%c(vec4 a, vec4 b, bool mid) {\n    vec4 r;\n    if (mid) {\n%s    } else {\n%s    }\n    return r;\n}\n\n", "xyz"[axis], mid, copy);
    }
    ov::str flush;
    for (const char c : { 'x', 'y', 'z', 'w' }) {
        flush += ov::format("%s(%s)", c == 'x' ? "" : ", ", glsl_flush_of(ov::format("v.%c", c)));
    }
    // The x pass at parent position q (and q + x on a midpoint), then y, then z, then the flush.
    const ov::str tail = R"glsl(vec4 pass_x(uint k, uvec3 q) {
    uint slot = g_pbase + q.x + g_np * (q.y + g_np * q.z);
    vec4 a = ch_load(k, slot);
    if (g_mid.x) {
        return hermite_x(a, ch_load(k, slot + 1u), true);
    }
    return hermite_x(a, a, false);
}

// Channel k of the level above at this point, subdivided and flushed, as a register; the pick (pick) copied from the
// floor parent.
vec4 subdivide(uint k, bool pick) {
    vec4 v;
    if (pick) {
        v = ch_load(k, g_pbase + g_p0.x + g_np * (g_p0.y + g_np * g_p0.z));
    } else {
        vec4 x00 = pass_x(k, g_p0);
        vec4 y0 = g_mid.y ? hermite_y(x00, pass_x(k, g_p0 + uvec3(0u, 1u, 0u)), true) : hermite_y(x00, x00, false);
        if (g_mid.z) {
            vec4 x01 = pass_x(k, g_p0 + uvec3(0u, 0u, 1u));
            vec4 y1 = g_mid.y ? hermite_y(x01, pass_x(k, g_p0 + uvec3(0u, 1u, 1u)), true) : hermite_y(x01, x01, false);
            v = hermite_z(y0, y1, true);
        } else {
            v = hermite_z(y0, y0, false);
        }
    }
    v = vec4(/*FLUSH*/);
    return v.yzwx;
}

// The debug variant's level 0: the output channel, then the pick, at the slot's DEBUG_VEC4S.
void store_debug(uint slot, vec4 ground, float pick) {
    debug_channels[slot * /*DEBUG_VEC4S*/u] = ground;
    debug_channels[slot * /*DEBUG_VEC4S*/u + 1u] = vec4(pick, 0.0, 0.0, 0.0);
}

)glsl";
    return code + ov::replace_all(tail, { { "/*FLUSH*/", flush }, { "/*DEBUG_VEC4S*/", ov::itos(CascadeGLSL::DEBUG_VEC4S) } });
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

uint32_t scratch_points_a(int p_item_size) {
    const uint32_t n = uint32_t(level_bound(p_item_size, 1));
    return n * n * n;
}

uint32_t scratch_points_b(int p_item_size) {
    const uint32_t n = uint32_t(level_bound(p_item_size, 2));
    return n * n * n;
}

uint32_t vec4s_per_point(const CascadeSource *p_sdf) {
    ov::str err;
    return uint32_t(Terrain::gpu_layout(*p_sdf->get_program(), err).channels);
}

void scratch_bytes(const CascadeSource *p_sdf, int p_item_size, uint64_t p_items, uint64_t &r_a, uint64_t &r_b) {
    const uint64_t vec4_bytes = 16 * uint64_t(vec4s_per_point(p_sdf));
    r_a = p_items * scratch_points_a(p_item_size) * vec4_bytes;
    r_b = p_items * scratch_points_b(p_item_size) * vec4_bytes;
}

bool ground_materials(const CascadeSource *p_sdf) {
    return p_sdf->samples_carry_materials();
}

ov::str level_shader(const CascadeSource *p_sdf, bool p_debug, bool p_points, bool p_fold, bool p_material, bool p_cubes, ShaderParams *r_params, int p_upto) {
    ShaderParams own;
    ShaderParams &params = r_params != nullptr ? *r_params : own;
    const Terrain::LevelProgram &prog = *p_sdf->get_program();
    ov::str layout_error;
    const Terrain::GpuLayout layout = Terrain::gpu_layout(prog, layout_error);
    OV_ERR_FAIL_COND_V_MSG(!layout_error.empty(), ov::str(), "CascadeGLSL: " + layout_error);
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
    // A sphere's base_at reads the item's regions (BaseRegions): g_base_item names it. Flat: the text as before.
    const ov::str base_item = p_sdf->get_curved_base() != nullptr ? "    g_base_item = pc.chunk_base + c;\n" : "";
    // The biome pruning's mask is the item's (Terrain::uses_biome_mask; only levels above the last write it, never the fold's).
    const ov::str mask_item = Terrain::uses_biome_mask(prog) ? "    g_item = pc.chunk_base + c;\n" : "";
    code = ov::replace_all(code, "/*ITEM_FRAME*/", fills.item_frame + base_item + mask_item);
    code = ov::replace_all(code, "/*ITEM_EXTENT*/", fills.item_extent);
    code = ov::replace_all(code, "/*LEVEL0_OUTPUT*/", fills.level0_output);
    code = ov::replace_all(code, "/*AFTER_SAMPLE*/", cubes ? "\n    if (!sample_live) {\n        return; // past the fold's barriers\n    }" : "");
    // The run's last level returns the output channel and the pick (Terrain::level_run).
    code = ov::replace_all(code, "/*GROUND_C0*/", "ground");
    const ov::str channels = "ground, p, lod";
    // The fold's workgroup box: its cube's first and last samples (the sample itself per sample).
    const ov::str cube_box = cubes ? ov::format("level_point(m_lo, m_hi, cube, m), level_point(m_lo, m_hi, cube + %du, m)", FOLD_CUBE - 1) : ov::str("p, p");
    code = ov::replace_all(code, "/*SAMPLE_SDF*/", p_fold ? ov::format("cascade_sample_sdf(%s, %s, pc.chunk_base + c, material, skirt_extra)", channels, cube_box) : ov::format("cascade_terrain_sdf(%s)", channels));
    // Without ground materials only the fold writes materials, and only for chunks with a list: a chunk without one is
    // ground, and the meshers don't read its samples' materials (its palette is the ground block, or its list is empty).
    // With them every chunk's samples get one: the winner's, or the ground's pick (level_run's o_pick).
    const bool grounds = ground_materials(p_sdf) && !p_points;
    const bool store = p_material && (p_fold || grounds);
    const ov::str ground_material = grounds ? p_sdf->get_ground_material_glsl() : ov::str();
    code = ov::replace_all(code, "/*MATERIAL_BUFFER*/", store ? glsl_material_buffer(MATERIAL_BINDING, "writeonly") : ov::str());
    ov::str material_store;
    if (store && grounds) {
        // Air samples (w > 0, after the skirt) keep 0: the meshers read a vertex's material from its cell's solid corners.
        material_store = ov::str(!ground_material.empty() ? "if (material == 0u && !(r.w > 0.0)) {\n        material = cascade_ground_material(pick);\n    }\n    " : "") +
                "sdf_material[sample_index] = material;";
    } else if (store) {
        material_store = "if (g_sdf_chunk.anchor_hi.w > 0) {\n        sdf_material[sample_index] = material;\n    }";
    }
    code = ov::replace_all(code, "/*MATERIAL_STORE*/", material_store);
    code = ov::replace_all(code, "/*STRUCTS*/", ov::str(GLSL_STRUCT_DEFINITION) + ov::str(GLSL_FRAME_STRUCTS));
    code = ov::replace_all(code, "/*TABLE_BUFFERS*/", p_fold ? p_sdf->get_table_glsl(TABLE_RECORD_BINDING, TABLE_LIST_BINDING) : ov::str());
    code = ov::replace_all(code, "/*HELPERS*/", ov::str(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + (p_points ? GPUFrame::glsl_point_frame() : ov::str()) + GPUFrame::glsl_int64() + ov::str(GLSL_EXACT_NOISE));
    code = ov::replace_all(code, "/*TERRAIN_SDF*/", p_sdf->get_glsl_code(params) + (p_fold ? (cubes ? ov::str(LEVEL_POINT) : ov::str()) + p_sdf->get_sample_fold_glsl(cubes ? LOCAL_SIZE : 0) : ov::str()));
    // The op table's helpers (the climate's duals, smooth min, curves, the forms' own; the base's stretch is get_glsl_code's), the
    // subdivision, then the program.
    const ov::str helpers = Climate::shared_helpers_glsl() + Climate::smooth_min_glsl() + Climate::curve_at_glsl() + Terrain::glsl_helpers();
    const ov::str program = "\n" + helpers + subdivision_functions(layout.channels) + Terrain::level_glsl(prog, layout, params, p_upto < 0 ? Terrain::NO_LEVEL : p_upto) + ground_material;
    code = ov::replace_all(code, "/*CASCADE*/", program);
    const ov::str debug_store = ov::str(cubes ? "if (sample_live) " : "") + "store_debug(sample_index, ground, pick); // CascadeSpike variant";
    code = ov::replace_all(code, "/*DEBUG_STORE*/", p_debug ? debug_store : ov::str());
    // Last: the block holds every number the generators took (a generic program's at its largest: its size in the text).
    const int64_t heap = int64_t(params.values.size());
    params.values.insert(params.values.end(), params.heap.begin(), params.heap.end());
    params.heap.clear();
    if (prog.generic) {
        OV_ERR_FAIL_COND_V_MSG(params.values.size() * sizeof(float) > MAX_PARAM_BYTES, ov::str(), "CascadeGLSL: more shader params than a uniform block holds.");
        params.values.resize(MAX_PARAM_BYTES / sizeof(float), 0.0f);
    }
    const ov::str block = ov::format("layout(set = 0, binding = %d, std140) uniform ShaderParams { vec4 cascade_param[%d]; };\n", PARAM_BINDING, params.vec4s());
    return ov::replace_all(code, "/*PARAM_BLOCK*/",
            block + ov::format("float cl_param(int i) {\n    return cascade_param[i >> 2][i & 3];\n}\n\nfloat cl_heap(int i) {\n    return cl_param(%d + i);\n}\n", heap));
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
    return ov::replace_all(ov::replace_all(p_shader, "/*GLSL_VERTEX_GRADIENT*/", GLSL_VERTEX_GRADIENT), "/*GLSL_ALGORITHM_PASS*/", p_pass);
}

} // namespace CascadeGLSL
