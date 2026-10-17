#include "cascade_glsl.hpp"

#include "exact_noise.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"
#include "rd_utils.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace Cascade;

namespace {

constexpr const char LEVEL_TEMPLATE[] = R"glsl(#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

/*STRUCTS*/

struct CascadeBand { vec4 factor; uvec4 offset_lo; ivec4 offset_hi; uvec4 info; };

layout(set = 0, binding = 0, std430) restrict readonly buffer InputChunk { ChunkInput input_chunk[]; };
layout(set = 0, binding = 1, std140) uniform ChunkParamUniform { ChunkParam chunk_param; };
layout(set = 0, binding = 2, std430) restrict writeonly buffer SDFSample { vec4 sdf_sample[]; };
layout(set = 0, binding = 3, std430) restrict buffer ScratchA { vec4 scratch_a[]; };
layout(set = 0, binding = 4, std430) restrict buffer ScratchB { vec4 scratch_b[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer Bands { CascadeBand bands[]; };
layout(set = 0, binding = 6, std430) restrict writeonly buffer DebugChannels { vec4 debug_channels[]; };

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
    uint c = gl_WorkGroupID.y;
    uint n = pc.n_child;
    uint local = gl_GlobalInvocationID.x;
    if (c >= pc.chunk_count || local >= n * n * n) {
        return;
    }
    uvec3 i = uvec3(local % n, (local / n) % n, local / (n * n));
    ChunkFrame f = chunk_frame(input_chunk[pc.chunk_base + c], chunk_param);
    int lod = f.lod;
    int m = lod + pc.j;
    int top = max(pc.top, lod);
    if (m > top) {
        return; // above T: zero, nothing stored, the chunk starts lower
    }
    // The chunk's box on Λ_L (Cascade::chunk_box): lo_L = anchor / 2^L - skirt, hi_L = lo_L + chunk_size - 1.
    uvec3 lo_lo = f.anchor_lo;
    ivec3 lo_hi = f.anchor_hi;
    i64_asr(lo_lo, lo_hi, lod);
    i64_add(lo_lo, lo_hi, ivec3(-chunk_param.skirt_size));
    uvec3 hi_lo = lo_lo;
    ivec3 hi_hi = lo_hi;
    i64_add(hi_lo, hi_hi, ivec3(chunk_param.chunk_size - 1));
    // Level m: lo_m = floor(lo_L / 2^j), hi_m = ceil(hi_L / 2^j) (Cascade::level_box).
    uvec3 m_lo = lo_lo;
    ivec3 m_hi = lo_hi;
    i64_asr(m_lo, m_hi, pc.j);
    i64_asr_ceil(hi_lo, hi_hi, pc.j);
    uvec3 size = hi_lo - m_lo + 1u;
    if (any(greaterThanEqual(i, size))) {
        return;
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

    CascadeBand band = bands[m];
    if (band.info.x != 0u) {
        ch = insert_band(ch, p, band);
    }
    ch = flush(ch);

    uint slot = c * n * n * n + local;
    if (pc.j > 0) {
        store(pc.j & 1, slot, ch);
        return;
    }
    // Λ_L: the sample.
    uint sample_index = (pc.chunk_base + c) * n * n * n + local;
    /*DEBUG_STORE*/
    precise vec4 r = cascade_terrain_sdf(vec4(ch.c[0], ch.c[1], ch.c[2], ch.c[3]), p, lod);
    // skirt_sdf (gpu_glsl.hpp), the same formula and order, precise; the depth is per LOD.
    if (chunk_param.skirt_size > 0 && chunk_param.skirt_depth > 0.0) {
        vec3 grid = vec3(i);
        precise float lo = float(chunk_param.skirt_size);
        precise float hi = lo + chunk_voxel_dimension(chunk_param);
        precise vec3 inside3 = min(grid - lo, hi - grid);
        precise float inside = min(inside3.x, min(inside3.y, inside3.z));
        precise float depth_m = cascade_skirt_depth_m(lod); // SDFCascade::skirt_depth_m
        r.w = r.w + clamp(-inside, 0.0, 1.0) * depth_m;
    }
    sdf_sample[sample_index] = r;
}
)glsl";

String flt(float p_f) {
    return ExactNoise::glsl_float(p_f);
}

// Ch and hermite_x/y/z for a channel set (from Cascade::PAIRS).
String hermite_functions(ChannelSet p_set) {
    const int nch = channel_count(p_set);
    String code = vformat("struct Ch { float c[%d]; };\n\n", nch);
    code += "Ch ch_zero() {\n    Ch r;\n";
    for (int c = 0; c < nch; c++) {
        code += vformat("    r.c[%d] = 0.0;\n", c);
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
        String mid_code;
        String copy_code;
        auto pair = [&](int p_v, int p_s) {
            mid_code += vformat("        hmid(a.c[%d], b.c[%d], a.c[%d], b.c[%d], r.c[%d], r.c[%d]);\n", p_v, p_v, p_s, p_s, p_v, p_s);
            copy_code += vformat("        hcopy(a.c[%d], a.c[%d], r.c[%d], r.c[%d]);\n", p_v, p_s, p_v, p_s);
        };
        if (p_set == ChannelSet::HERMITE8) {
            for (const auto &pr : PAIRS[axis]) {
                pair(pr[0], pr[1]);
            }
        } else {
            // LINEAR4: value + the axis slope by the Hermite rule, the transverse slopes linear.
            pair(VALUE, DX + axis);
            for (int s = DX; s <= DZ; s++) {
                if (s != DX + axis) {
                    mid_code += vformat("        { precise float t = (a.c[%d] + b.c[%d]) * 0.5; r.c[%d] = t; }\n", s, s, s);
                    copy_code += vformat("        r.c[%d] = a.c[%d];\n", s, s);
                }
            }
        }
        code += vformat("Ch hermite_%s(Ch a, Ch b, bool mid) {\n    Ch r;\n    if (mid) {\n%s    } else {\n%s    }\n    return r;\n}\n\n", AXIS_NAMES[axis],
                mid_code, copy_code);
    }
    return code;
}

// Scratch and debug buffer access (nv vec4s per point) and the x pass at a parent position.
String storage_functions(ChannelSet p_set) {
    const int nch = channel_count(p_set);
    const int nv = int(CascadeGLSL::vec4s_per_point(p_set));
    String code = "Ch unpack(";
    for (int v = 0; v < nv; v++) {
        code += vformat("%svec4 v%d", v ? ", " : "", v);
    }
    code += ") {\n    Ch r;\n";
    for (int c = 0; c < nch; c++) {
        code += vformat("    r.c[%d] = v%d[%d];\n", c, c / 4, c % 4);
    }
    code += "    return r;\n}\n\n";
    code += "Ch load(bool from_a, uint slot) {\n";
    String args_a;
    String args_b;
    for (int v = 0; v < nv; v++) {
        args_a += vformat("%sscratch_a[slot * %du + %du]", v ? ", " : "", nv, v);
        args_b += vformat("%sscratch_b[slot * %du + %du]", v ? ", " : "", nv, v);
    }
    code += vformat("    if (from_a) {\n        return unpack(%s);\n    }\n    return unpack(%s);\n}\n\n", args_a, args_b);
    String vecs;
    for (int v = 0; v < nv; v++) {
        vecs += vformat("    vec4 v%d = vec4(r.c[%d], r.c[%d], r.c[%d], r.c[%d]);\n", v, 4 * v, 4 * v + 1, 4 * v + 2, 4 * v + 3);
    }
    code += "void store(int odd, uint slot, Ch r) {\n" + vecs;
    for (int v = 0; v < nv; v++) {
        code += vformat("    if (odd != 0) {\n        scratch_a[slot * %du + %du] = v%d;\n    } else {\n        scratch_b[slot * %du + %du] = v%d;\n    }\n", nv, v, v,
                nv, v, v);
    }
    code += "}\n\n";
    code += "void store_debug(uint slot, Ch r) {\n" + vecs;
    for (int v = 0; v < nv; v++) {
        code += vformat("    debug_channels[slot * %du + %du] = v%d;\n", nv, v, v);
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

// Cascade::insert_bands for one band (o = channel * factor[order], ch = ch + o: two roundings, precise)
// and Cascade::flush_value.
String band_functions(ChannelSet p_set) {
    const int nch = channel_count(p_set);
    String code = "Ch insert_band(Ch ch, WorldPos p, CascadeBand band) {\n";
    if (p_set == ChannelSet::HERMITE8) {
        code += "    vec4 c0;\n    vec4 c1;\n    perlin3_cross(p, int(band.info.z), band.info.y, band.offset_lo.xyz, band.offset_hi.xyz, c0, c1);\n";
        code += "    float n[8] = float[8](c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);\n";
    } else {
        // perlin3 returns (d/dx, d/dy, d/dz, value): the same bits as perlin3_cross's first four channels.
        code += "    vec4 g = perlin3(p, int(band.info.z), band.info.y, band.offset_lo.xyz, band.offset_hi.xyz);\n";
        code += "    float n[4] = float[4](g.w, g.x, g.y, g.z);\n";
    }
    for (int c = 0; c < nch; c++) {
        code += vformat("    { precise float o = n[%d] * band.factor[%d]; precise float s = ch.c[%d] + o; ch.c[%d] = s; }\n", c, channel_order(c), c, c);
    }
    code += "    return ch;\n}\n\n";
    code += "Ch flush(Ch ch) {\n";
    for (int c = 0; c < nch; c++) {
        code += vformat("    ch.c[%d] = abs(ch.c[%d]) < %s ? 0.0 : ch.c[%d];\n", c, c, flt(ExactNoise::bits_float(FLUSH_BITS)), c);
    }
    code += "    return ch;\n}\n";
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

)glsl";

} // namespace

PackedByteArray band_table(const Spec &p_spec) {
    using namespace WorldCoord;
    PackedByteArray bytes;
    bytes.resize(BAND_ENTRIES * BAND_BYTES);
    bytes.fill(0);
    for (const Band &b : p_spec.bands) {
        const int m = p_spec.insert_level(b);
        ERR_CONTINUE_MSG(m < 0 || m >= BAND_ENTRIES, vformat("CascadeGLSL: band k=%d inserted outside the lattice table.", b.k));
        const int64_t base = int64_t(m) * BAND_BYTES;
        ERR_CONTINUE_MSG(bytes.decode_u32(base + 48) != 0, vformat("CascadeGLSL: two bands on lattice %d (the GPU table holds one per level).", m));
        for (int o = 0; o < 4; o++) {
            // Cascade's BandFactors, the same float product.
            bytes.encode_float(base + 4 * o, b.amp * ExactNoise::exact_pow2(-p_spec.insert_shift * o));
        }
        bytes.encode_u32(base + 16, lo32(b.offset.x));
        bytes.encode_u32(base + 20, lo32(b.offset.y));
        bytes.encode_u32(base + 24, lo32(b.offset.z));
        bytes.encode_s32(base + 32, hi32(b.offset.x));
        bytes.encode_s32(base + 36, hi32(b.offset.y));
        bytes.encode_s32(base + 40, hi32(b.offset.z));
        bytes.encode_u32(base + 48, 1);
        bytes.encode_u32(base + 52, b.seed);
        bytes.encode_u32(base + 56, uint32_t(b.k));
    }
    return bytes;
}

uint32_t scratch_points_a(int p_chunk_size) {
    const uint32_t n = uint32_t(level_bound(p_chunk_size, 1));
    return n * n * n;
}

uint32_t scratch_points_b(int p_chunk_size) {
    const uint32_t n = uint32_t(level_bound(p_chunk_size, 2));
    return n * n * n;
}

uint32_t vec4s_per_point(ChannelSet p_set) {
    return uint32_t(channel_count(p_set) + 3) / 4;
}

String level_shader(const SDFBase *p_sdf, bool p_debug) {
    const Spec *spec = p_sdf->get_cascade();
    ERR_FAIL_NULL_V(spec, String());
    String code = LEVEL_TEMPLATE;
    code = code.replace("/*STRUCTS*/", String(GLSL_STRUCT_DEFINITION) + String(GLSL_FRAME_STRUCTS));
    code = code.replace("/*HELPERS*/", String(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + GPUFrame::glsl_int64() + String(GLSL_EXACT_NOISE) + String(GLSL_PERLIN3_CROSS));
    code = code.replace("/*TERRAIN_SDF*/", p_sdf->get_glsl_code());
    code = code.replace("/*CASCADE*/", hermite_functions(spec->set) + storage_functions(spec->set) + band_functions(spec->set));
    code = code.replace("/*DEBUG_STORE*/", p_debug ? "store_debug(sample_index, ch); // CascadeSpike variant" : "");
    return code;
}

String fill_mesher_pass(const String &p_shader, const SDFBase *p_sdf, const String &p_pass, bool p_vertex_pass) {
    if (p_sdf->get_cascade() == nullptr) {
        return p_shader.replace("/*GLSL_SKIRT_SDF*/", GLSL_SKIRT_SDF).replace("/*GLSL_ALGORITHM_PASS*/", p_pass);
    }
    String pass = p_pass;
    if (p_vertex_pass) {
        // GPUDirectMesher and GPUMesher spell the call differently.
        const char *calls[] = { "skirt_sdf(chunk, chunk_vertex, chunk_param).xyz", "skirt_sdf(chunk,chunk_vertex,chunk_param).xyz" };
        bool found = false;
        for (const char *call : calls) {
            if (pass.contains(call)) {
                pass = pass.replace(call, "cascade_vertex_gradient(gl_GlobalInvocationID, local_vertex)");
                found = true;
            }
        }
        ERR_FAIL_COND_V_MSG(!found, p_shader, "CascadeGLSL: the vertex pass has no skirt_sdf normal call to replace.");
    }
    return p_shader.replace("/*GLSL_SKIRT_SDF*/", String(GLSL_SKIRT_SDF) + GLSL_VERTEX_GRADIENT).replace("/*GLSL_ALGORITHM_PASS*/", pass);
}

} // namespace CascadeGLSL

bool CascadePasses::create(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_chunks, RID p_input, RID p_param_ubo,
        RID p_sdf_sample, RID p_debug, const CascadePasses *p_share_scratch) {
    const Spec *spec = p_sdf->get_cascade();
    ERR_FAIL_NULL_V(spec, false);
    rd = p_rd;
    top = spec->top_level();
    chunk_size = p_param.chunk_size;
    max_chunks = p_max_chunks;

    const String code = CascadeGLSL::level_shader(p_sdf, p_debug.is_valid());
    RDUtils::dump_shader(p_debug.is_valid() ? "cascade_levels_debug" : "cascade_levels", code);
    Ref<RDShaderSource> source;
    source.instantiate();
    source->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, code);
    Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(source);
    const String error = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!error.is_empty()) {
        UtilityFunctions::printerr("--- SDF SHADER COMPILATION FAILED (CascadePasses) ---");
        UtilityFunctions::printerr("ERROR MESSAGE:\n", error);
        const PackedStringArray lines = code.split("\n");
        for (int i = 0; i < lines.size(); i++) {
            UtilityFunctions::printerr(String::num_int64(i + 1) + " | " + lines[i]);
        }
        return false;
    }
    shader = rd->shader_create_from_spirv(spirv);
    pipeline = rd->compute_pipeline_create(shader);

    const uint64_t vec4_bytes = 16 * CascadeGLSL::vec4s_per_point(spec->set);
    const uint64_t bytes_a = uint64_t(max_chunks) * CascadeGLSL::scratch_points_a(chunk_size) * vec4_bytes;
    const uint64_t bytes_b = uint64_t(max_chunks) * CascadeGLSL::scratch_points_b(chunk_size) * vec4_bytes;
    if (p_share_scratch != nullptr) {
        ERR_FAIL_COND_V(p_share_scratch->max_chunks < max_chunks || p_share_scratch->scratch_bytes_total < bytes_a + bytes_b, false);
        scratch_a = p_share_scratch->scratch_a;
        scratch_b = p_share_scratch->scratch_b;
        owns_scratch = false;
    } else {
        scratch_a = rd->storage_buffer_create(bytes_a);
        scratch_b = rd->storage_buffer_create(bytes_b);
        scratch_bytes_total = bytes_a + bytes_b;
    }
    const PackedByteArray table = CascadeGLSL::band_table(*spec);
    bands = rd->storage_buffer_create(table.size(), table);
    RID debug = p_debug;
    if (!debug.is_valid()) {
        debug_placeholder = rd->storage_buffer_create(16);
        debug = debug_placeholder;
    }

    auto make_uniform = [](int p_binding, RID p_rid, RenderingDevice::UniformType p_type) {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(p_type);
        u->set_binding(p_binding);
        u->add_id(p_rid);
        return u;
    };
    TypedArray<RDUniform> uniforms;
    uniforms.push_back(make_uniform(0, p_input, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(1, p_param_ubo, RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER));
    uniforms.push_back(make_uniform(2, p_sdf_sample, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(3, scratch_a, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(4, scratch_b, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(5, bands, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(6, debug, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniform_set = rd->uniform_set_create(uniforms, shader, 0);
    return uniform_set.is_valid();
}

void CascadePasses::free() {
    if (rd == nullptr) {
        return;
    }
    for (const RID &rid : { uniform_set, pipeline, shader, bands, debug_placeholder }) {
        if (rid.is_valid()) {
            rd->free_rid(rid);
        }
    }
    if (owns_scratch) {
        for (const RID &rid : { scratch_a, scratch_b }) {
            if (rid.is_valid()) {
                rd->free_rid(rid);
            }
        }
    }
    uniform_set = RID();
    pipeline = RID();
    shader = RID();
    scratch_a = RID();
    scratch_b = RID();
    bands = RID();
    debug_placeholder = RID();
    rd = nullptr;
}

int CascadePasses::level_count(int p_min_lod) const {
    return MAX(top - p_min_lod, 0) + 1;
}

void CascadePasses::record_level(int p_j, uint32_t p_base, uint32_t p_count) {
    ERR_FAIL_COND(p_count == 0 || p_count > max_chunks);
    const uint32_t n = uint32_t(level_bound(chunk_size, p_j));
    PackedByteArray push;
    push.resize(CascadeGLSL::PUSH_BYTES);
    push.fill(0);
    push.encode_u32(0, p_base);
    push.encode_u32(4, p_count);
    push.encode_s32(8, p_j);
    push.encode_s32(12, top);
    push.encode_u32(16, n);
    push.encode_u32(20, uint32_t(level_bound(chunk_size, p_j + 1)));
    const int64_t list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(list, pipeline);
    rd->compute_list_bind_uniform_set(list, uniform_set, 0);
    rd->compute_list_set_push_constant(list, push, push.size());
    rd->compute_list_dispatch(list, (n * n * n + CascadeGLSL::LOCAL_SIZE - 1) / CascadeGLSL::LOCAL_SIZE, p_count, 1);
    rd->compute_list_end();
}

void CascadePasses::record_all(int p_min_lod, uint32_t p_count) {
    for (uint32_t base = 0; base < p_count; base += max_chunks) {
        const uint32_t count = MIN(max_chunks, p_count - base);
        for (int j = level_count(p_min_lod) - 1; j >= 0; j--) {
            record_level(j, base, count);
        }
    }
}
