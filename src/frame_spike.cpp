#include "frame_spike.hpp"

#include "archetype_probes.hpp"
#include "chunk_math.hpp"
#include "exact_noise.hpp"
#include "frame_probe.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"
#include "sdf_dummy.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

using namespace WorldCoord;
using ExactNoise::bits_float;
using ExactNoise::float_bits;

namespace {

// Plumbing check: an integer hash of every bit of the position.
uint32_t voxel_hash_cpu(const WorldPos &p_pos) {
    auto axis = [](int64_t p_v, float p_frac) {
        return lo32(p_v) * 1664525u + uint32_t(hi32(p_v)) * 22695477u + float_bits(p_frac) * 2891336453u;
    };
    const uint32_t vx = axis(p_pos.voxel.x, p_pos.frac.x);
    const uint32_t vy = axis(p_pos.voxel.y, p_pos.frac.y);
    const uint32_t vz = axis(p_pos.voxel.z, p_pos.frac.z);
    uint32_t h = vx ^ (vy * 0x9E3779B1u) ^ (vz * 0x85EBCA77u);
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

constexpr const char VOXEL_HASH_GLSL[] = R"glsl(
void probe_voxel_hash(WorldPos p, uint b) {
    uvec3 v = p.lo * 1664525u + uvec3(p.hi) * 22695477u + floatBitsToUint(p.frac) * 2891336453u;
    uint h = v.x ^ (v.y * 0x9E3779B1u) ^ (v.z * 0x85EBCA77u);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    results[b] = h;
}
)glsl";

// One word per octave_cell decomposition: cell low and high words and the fixed-point position.
uint32_t fold_cell(uint32_t p_lx, uint32_t p_ly, uint32_t p_lz, int32_t p_hx, int32_t p_hy, int32_t p_hz, const ExactNoise::UVec3 &p_q) {
    const ExactNoise::UVec3 a = ExactNoise::pcg3d(ExactNoise::UVec3{ p_lx ^ (uint32_t(p_hx) * 0x9E3779B1u), p_ly ^ (uint32_t(p_hy) * 0x9E3779B1u),
            p_lz ^ (uint32_t(p_hz) * 0x9E3779B1u) });
    return a.x ^ ExactNoise::pcg3d(p_q).y;
}

constexpr int CELL_KS[] = { 0, 3, 13, 24, 25, 31, 32, 33, 47, 62 };
constexpr int PERLIN3_KS[] = { 2, 9, 20, 33 };
constexpr int PERLIN2_KS[] = { 2, 9, 33 };
// perlin3_cross (the cascade's bands): perlin3's ks, and the band each LOD of PROBE_LODS inserts on its own
// lattice (k = L + 3).
constexpr int CROSS_KS[] = { 2, 3, 4, 8, 9, 17, 20, 27, 33 };
// The int64 helpers of GLSL_FRAME_INT64: shifts and small addends.
constexpr int I64_SHIFTS[] = { 0, 1, 5, 31, 32, 33, 47, 63 };
constexpr int I64_SHIFT_COUNT = int(sizeof(I64_SHIFTS) / sizeof(I64_SHIFTS[0]));
constexpr int32_t I64_ADDS[] = { 0, 1, -1, -896, 2147483647, -2147483647 - 1 };
constexpr int I64_ADD_COUNT = int(sizeof(I64_ADDS) / sizeof(I64_ADDS[0]));
// Hash-random points per origin, besides the fixed grid; fuzzed archetype programs and how many probes
// share one shader (one dispatch per set and origin).
constexpr int RANDOM_POINTS = 1024;
constexpr int FUZZ_PROGRAMS = 200;
constexpr size_t PROBES_PER_SET = 24;
constexpr int64_t MAX_SET_GLSL = 300000; // bytes of probe GLSL per shader (compile time)
// Size (voxels) of the archetype shapes probed (a copy near each origin): the grid and the random points
// fall inside and outside them.
constexpr float ARCH_SHAPE_SIZE = 2000.0f;

// A deterministic test offset for octave k, any value in [0, 2^(k+1)) (large ones exercise the carry).
Vector3i64 test_offset(int p_k) {
    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(p_k), 0xF00Du, 0xBEEFu });
    const uint64_t mask = (uint64_t(1) << MIN(p_k + 1, 61)) - 1;
    return Vector3i64(int64_t(((uint64_t(h.x) << 32) | h.y) & mask), int64_t(((uint64_t(h.y) << 32) | h.z) & mask),
            int64_t(((uint64_t(h.z) << 32) | h.x) & mask));
}

// Largest change of one octave (|gradient| <= p_g per cell) between neighbouring voxels. Above k = 24
// the position inside the cell moves in steps of 2^-24 every 2^(k-24) voxels, plus float rounding of
// a value ~1. A wrong corner hash shows up as a jump of order 0.1.
double octave_step_bound(int p_k, double p_g) {
    return p_g * std::ldexp(1.0, -MIN(p_k, 24)) + std::ldexp(1.0, -20);
}

// A probe set (a new noise or SDF term adds its entries here; one table, shared by every origin).
std::vector<Probe> make_probes(const SDFDummy &p_sdf) {
    using namespace ExactNoise;
    std::vector<Probe> probes;
    {
        Probe p;
        p.name = "voxel_hash";
        p.glsl = VOXEL_HASH_GLSL;
        p.cpu = [](const WorldPos &q, uint32_t *r) { r[0] = voxel_hash_cpu(q); };
        probes.push_back(p);
    }
    {
        Probe p;
        p.name = "hash_cell";
        p.glsl = "void probe_hash_cell(WorldPos p, uint b) {\n    results[b] = hash_cell(p.lo, p.hi, 0x00ABCDEFu);\n}\n";
        p.cpu = [](const WorldPos &q, uint32_t *r) { r[0] = hash_cell(q.voxel, 0x00ABCDEFu); };
        probes.push_back(p);
    }
    {
        Probe p;
        p.name = "octave_cell";
        p.words = int(sizeof(CELL_KS) / sizeof(CELL_KS[0]));
        p.glsl = R"glsl(
uint fold_cell(OctaveCell c) {
    uvec3 a = pcg3d(c.lo ^ (uvec3(c.hi) * 0x9E3779B1u));
    return a.x ^ pcg3d(c.q).y;
}
void probe_octave_cell(WorldPos p, uint b) {
)glsl";
        for (int i = 0; i < p.words; i++) {
            const Vector3i64 o = test_offset(CELL_KS[i]);
            p.glsl += vformat("    results[b + %du] = fold_cell(octave_cell(p, %d, %s, %s));\n", i, CELL_KS[i], glsl_lo(o), glsl_hi(o));
        }
        p.glsl += "}\n";
        p.cpu = [](const WorldPos &q, uint32_t *r) {
            for (size_t i = 0; i < sizeof(CELL_KS) / sizeof(CELL_KS[0]); i++) {
                const OctaveCell c = octave_cell(q, CELL_KS[i], test_offset(CELL_KS[i]));
                r[i] = fold_cell(lo32(c.cell.x), lo32(c.cell.y), lo32(c.cell.z), hi32(c.cell.x), hi32(c.cell.y), hi32(c.cell.z), c.q);
            }
        };
        probes.push_back(p);
    }
    for (int k : PERLIN3_KS) {
        Probe p;
        p.name = vformat("perlin3_k%d", k);
        p.words = 4;
        p.is_float = true;
        const Vector3i64 o = test_offset(k);
        const uint32_t seed = 0x1000u + uint32_t(k);
        p.glsl = vformat("void probe_%s(WorldPos p, uint b) {\n    vec4 v = perlin3(p, %d, %s, %s, %s);\n", p.name, k, glsl_uint(seed),
                glsl_lo(o), glsl_hi(o));
        p.glsl += "    for (uint i = 0u; i < 4u; i++) { results[b + i] = floatBitsToUint(v[i]); }\n}\n";
        p.cpu = [k, o, seed](const WorldPos &q, uint32_t *r) {
            const Float4 v = perlin3(q, k, seed, o);
            r[0] = float_bits(v.x);
            r[1] = float_bits(v.y);
            r[2] = float_bits(v.z);
            r[3] = float_bits(v.w);
        };
        p.continuity_bound = octave_step_bound(k, 4.0);
        p.continuity_word = 3;
        probes.push_back(p);
    }
    for (int k : PERLIN2_KS) {
        Probe p;
        p.name = vformat("perlin2_k%d", k);
        p.words = 4;
        p.is_float = true;
        const Vector3i64 o = test_offset(k);
        const uint32_t seed = 0x2000u + uint32_t(k);
        p.glsl = vformat("void probe_%s(WorldPos p, uint b) {\n    vec4 v = perlin2(p, %d, %s, %s, %s);\n", p.name, k, glsl_uint(seed),
                glsl_lo(o), glsl_hi(o));
        p.glsl += "    for (uint i = 0u; i < 4u; i++) { results[b + i] = floatBitsToUint(v[i]); }\n}\n";
        p.cpu = [k, o, seed](const WorldPos &q, uint32_t *r) {
            const Float4 v = perlin2(q, k, seed, o);
            r[0] = float_bits(v.x);
            r[1] = float_bits(v.y);
            r[2] = float_bits(v.z);
            r[3] = float_bits(v.w);
        };
        p.continuity_bound = octave_step_bound(k, 6.0);
        p.continuity_word = 3;
        probes.push_back(p);
    }
    for (int k : CROSS_KS) {
        Probe p;
        p.name = vformat("perlin3_cross_k%d", k);
        p.words = 8;
        p.is_float = true;
        const Vector3i64 o = test_offset(k);
        const uint32_t seed = 0x3000u + uint32_t(k);
        p.glsl = vformat("void probe_%s(WorldPos p, uint b) {\n    vec4 c0;\n    vec4 c1;\n    perlin3_cross(p, %d, %s, %s, %s, c0, c1);\n", p.name,
                k, glsl_uint(seed), glsl_lo(o), glsl_hi(o));
        p.glsl += "    for (uint i = 0u; i < 4u; i++) { results[b + i] = floatBitsToUint(c0[i]); results[b + 4u + i] = floatBitsToUint(c1[i]); }\n}\n";
        p.cpu = [k, o, seed](const WorldPos &q, uint32_t *r) {
            const Channels8 c = perlin3_cross(q, k, seed, o);
            for (int i = 0; i < 8; i++) {
                r[i] = float_bits(c.c[i]);
            }
        };
        p.continuity_bound = octave_step_bound(k, 4.0);
        p.continuity_word = 0;
        probes.push_back(p);
    }
    {
        // The voxel's three axes through every shift (asr, ceil asr, shl), then every addend: (lo, hi) words.
        Probe p;
        p.name = "i64";
        p.words = 3 * 6 * I64_SHIFT_COUNT + 6 * I64_ADD_COUNT;
        p.glsl = "void probe_i64_store(uvec3 lo, ivec3 hi, uint b) {\n"
                 "    for (uint i = 0u; i < 3u; i++) { results[b + 2u * i] = lo[i]; results[b + 2u * i + 1u] = uint(hi[i]); }\n}\n"
                 "void probe_i64(WorldPos p, uint b) {\n";
        for (int i = 0; i < I64_SHIFT_COUNT; i++) {
            const int s = I64_SHIFTS[i];
            p.glsl += vformat("    { uvec3 lo = p.lo; ivec3 hi = p.hi; i64_asr(lo, hi, %d); probe_i64_store(lo, hi, b + %du); }\n", s, 18 * i);
            p.glsl += vformat("    { uvec3 lo = p.lo; ivec3 hi = p.hi; i64_asr_ceil(lo, hi, %d); probe_i64_store(lo, hi, b + %du); }\n", s, 18 * i + 6);
            p.glsl += vformat("    { uvec3 lo = p.lo; ivec3 hi = p.hi; i64_shl(lo, hi, %d); probe_i64_store(lo, hi, b + %du); }\n", s, 18 * i + 12);
        }
        for (int i = 0; i < I64_ADD_COUNT; i++) {
            p.glsl += vformat("    { uvec3 lo = p.lo; ivec3 hi = p.hi; i64_add(lo, hi, ivec3(%s)); probe_i64_store(lo, hi, b + %du); }\n",
                    I64_ADDS[i] == INT32_MIN ? String("-2147483647 - 1") : itos(I64_ADDS[i]), 18 * I64_SHIFT_COUNT + 6 * i);
        }
        p.glsl += "}\n";
        p.cpu = [](const WorldPos &q, uint32_t *r) {
            auto store = [](const Vector3i64 &p_v, uint32_t *r_w) {
                for (int a = 0; a < 3; a++) {
                    r_w[2 * a] = lo32(p_v[a]);
                    r_w[2 * a + 1] = uint32_t(hi32(p_v[a]));
                }
            };
            for (int i = 0; i < I64_SHIFT_COUNT; i++) {
                const int s = I64_SHIFTS[i];
                Vector3i64 asr, ceil, shl;
                for (int a = 0; a < 3; a++) {
                    const int64_t v = q.voxel[a];
                    asr[a] = v >> s;
                    const uint64_t rest = uint64_t(v) & ((uint64_t(1) << s) - 1);
                    ceil[a] = asr[a] + (rest != 0 ? 1 : 0);
                    shl[a] = int64_t(uint64_t(v) << s);
                }
                store(asr, r + 18 * i);
                store(ceil, r + 18 * i + 6);
                store(shl, r + 18 * i + 12);
            }
            for (int i = 0; i < I64_ADD_COUNT; i++) {
                Vector3i64 sum;
                for (int a = 0; a < 3; a++) {
                    sum[a] = int64_t(uint64_t(q.voxel[a]) + uint64_t(int64_t(I64_ADDS[i])));
                }
                store(sum, r + 18 * I64_SHIFT_COUNT + 6 * i);
            }
        };
        probes.push_back(p);
    }
    // The terrain's two fBms and the terrain itself (gradient included), at every LOD of PROBE_LODS.
    // Continuity is checked on the value of LOD 0 (word 3), the same at every LOD.
    auto fbm_bound = [](const OctaveSet &p_set, double p_g) {
        double b = 0.0;
        for (const Octave &o : p_set.octaves) {
            b += double(gradient_factor(o, 0)) * p_g;
        }
        return b;
    };
    auto lod_probe = [](const String &p_name, const String &p_fn, std::function<Float4(const WorldPos &, int)> p_cpu) {
        Probe p;
        p.name = p_name;
        p.words = 4 * PROBE_LOD_COUNT;
        p.is_float = true;
        p.glsl = vformat("void probe_%s(WorldPos p, uint b) {\n", p_name);
        for (int i = 0; i < PROBE_LOD_COUNT; i++) {
            p.glsl += vformat("    vec4 v%d = %s(p, int(params.lod_bias) + %d);\n", i, p_fn, PROBE_LODS[i]);
            p.glsl += vformat("    for (uint i = 0u; i < 4u; i++) { results[b + %du + i] = floatBitsToUint(v%d[i]); }\n", 4 * i, i);
        }
        p.glsl += "}\n";
        p.cpu = [p_cpu](const WorldPos &q, uint32_t *r) {
            for (int i = 0; i < PROBE_LOD_COUNT; i++) {
                const Float4 v = p_cpu(q, PROBE_LODS[i]);
                r[4 * i + 0] = float_bits(v.x);
                r[4 * i + 1] = float_bits(v.y);
                r[4 * i + 2] = float_bits(v.z);
                r[4 * i + 3] = float_bits(v.w);
            }
        };
        return p;
    };
    const SDFDummy *sdf = &p_sdf;
    for (int which = 0; which < 2; which++) {
        const String name = which == 0 ? "terrain_height" : "terrain_caves";
        Probe p = lod_probe(name, name, [sdf, which](const WorldPos &q, int p_lod) {
            return fbm(which == 0 ? sdf->get_height() : sdf->get_caves(), q, p_lod);
        });
        p.continuity_bound = fbm_bound(which == 0 ? p_sdf.get_height() : p_sdf.get_caves(), which == 0 ? 6.0 : 4.0);
        p.continuity_word = 3;
        probes.push_back(p);
    }
    probes.push_back(lod_probe("terrain_sdf", "terrain_sdf", [sdf](const WorldPos &q, int p_lod) { return sdf->sdf(q, p_lod); }));
    return probes;
}

// Per sample: lo(3) hi(3) frac bits(3) anchor_lo(3), then the probe words.
constexpr uint32_t FIXED_WORDS = 12;
constexpr uint32_t SAMPLE_BYTES = 48; // ChunkInput (32) + vec4 grid

constexpr const char SHADER_TEMPLATE[] = R"glsl(#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

/*STRUCTS*/

struct Sample { ChunkInput chunk; vec4 grid; };

layout(set = 0, binding = 0, std430) restrict readonly buffer Samples { Sample samples[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer Results { uint results[]; };
layout(set = 0, binding = 2, std140) uniform ChunkParamUniform { ChunkParam chunk_param; };
layout(push_constant, std430) uniform Params { uint count; uint words; uint lod_bias; uint pad1; } params;

/*HELPERS*/

/*TERRAIN_SDF*/

/*PROBES*/

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= params.count) {
        return;
    }
    ChunkFrame f = chunk_frame(samples[i].chunk, chunk_param);
    WorldPos p = world_pos_from_grid(f, samples[i].grid.xyz, chunk_param);
    uint b = i * params.words;
    results[b + 0u] = p.lo.x;
    results[b + 1u] = p.lo.y;
    results[b + 2u] = p.lo.z;
    results[b + 3u] = uint(p.hi.x);
    results[b + 4u] = uint(p.hi.y);
    results[b + 5u] = uint(p.hi.z);
    results[b + 6u] = floatBitsToUint(p.frac.x);
    results[b + 7u] = floatBitsToUint(p.frac.y);
    results[b + 8u] = floatBitsToUint(p.frac.z);
    results[b + 9u] = f.anchor_lo.x;
    results[b + 10u] = f.anchor_lo.y;
    results[b + 11u] = f.anchor_lo.z;
    /*PROBE_CALLS*/
}
)glsl";

struct Case {
    ChunkId chunk;
    Float3 grid;
};

struct Pair {
    size_t a;
    size_t b;
    const char *what;
};

// One world origin: test vectors, GPU results and checks.
struct OriginRun {
    ChunkParam param;
    std::vector<Case> cases;
    std::vector<Pair> pairs;
    // Absolute test points (the same for every origin): cases[absolute_begin ..].
    size_t absolute_begin = 0;
    // The hash-random points (the same offsets from every origin): cases[random_begin, absolute_begin).
    size_t random_begin = 0;
    // Consecutive LOD0 voxels along x: index ranges into cases.
    std::vector<std::pair<size_t, size_t>> lines;
};

String case_str(const Case &p_case) {
    return vformat("chunk %s grid (%s, %s, %s)", to_string(p_case.chunk), p_case.grid.x, p_case.grid.y, p_case.grid.z);
}

// The LOD0 chunk holding voxel p_voxel, and the vertex-grid coordinate of that voxel in it.
float lod0_grid(int64_t p_voxel, int64_t p_half, int64_t &r_id) {
    r_id = floor_div(p_voxel, 2 * p_half) * 2 + 1;
    return float(2 + (p_voxel - (r_id - 1) * p_half));
}

Case lod0_case(const Vector3i64 &p_voxel, const Float3 &p_frac, int64_t p_half) {
    Case c;
    c.grid.x = lod0_grid(p_voxel.x, p_half, c.chunk.x) + p_frac.x;
    c.grid.y = lod0_grid(p_voxel.y, p_half, c.chunk.y) + p_frac.y;
    c.grid.z = lod0_grid(p_voxel.z, p_half, c.chunk.z) + p_frac.z;
    return c;
}

int64_t random_offset(uint32_t p_a, uint32_t p_b, uint32_t p_c, int64_t p_range) {
    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ p_a, p_b, p_c });
    return int64_t(((uint64_t(h.x) << 32) | h.y) % uint64_t(2 * p_range + 1)) - p_range;
}

// RANDOM_POINTS hash-random samples around the origin: a chunk of LOD 0, 1, 4 or 8 within 16 chunks of
// it on each axis, a vertex-grid position in [0, 32) in steps of 2^-6 (fractions of a voxel included).
void add_random_cases(OriginRun &r_run) {
    static const int LODS[] = { 0, 1, 4, 8 };
    const ChunkId origin = r_run.param.origin_chunk;
    for (uint32_t i = 0; i < uint32_t(RANDOM_POINTS); i++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, 0xC0FFEEu, 0x5A3Du });
        const int64_t s = int64_t(1) << LODS[h.x & 3u];
        Case c;
        for (int a = 0; a < 3; a++) {
            c.chunk[a] = origin[a] + (2 * random_offset(i, uint32_t(a), 0xD1CEu, 16) + 1) * s;
        }
        c.grid = Float3{ float(h.y & 2047u) * 0.015625f, float(h.z & 2047u) * 0.015625f, float((h.y >> 16) & 2047u) * 0.015625f };
        r_run.cases.push_back(c);
    }
}

void build_cases(OriginRun &r_run, const SDFDummy &p_sdf) {
    const ChunkParam &param = r_run.param;
    const ChunkId origin = param.origin_chunk;
    const VoxelPos origin_voxel = param.origin_voxel();
    const int64_t half = param.half_chunk_voxels();
    static const float GRID[] = { 0.0f, 1.0f, 2.0f, 3.0f, 15.0f, 28.0f, 29.0f, 30.0f, 31.0f, 0.5f, 0.0078125f, 1.999f, 30.25f,
        2.0f - 1.0f / 1048576.0f };

    auto add_chunk = [&](const ChunkId &p_chunk) {
        for (float g : GRID) {
            r_run.cases.push_back({ p_chunk, { g, g, g } });
            r_run.cases.push_back({ p_chunk, { g, 7.0f, 13.0f } });
            r_run.cases.push_back({ p_chunk, { 0.5f, g, 29.0f } });
        }
    };

    // Chunks around the origin, all signs (negative ids included).
    std::vector<ChunkId> near;
    for (int lod : { 0, 1, 4, 14 }) {
        const int64_t s = int64_t(1) << lod;
        for (const ChunkId &c : { ChunkId(1, 1, 1), ChunkId(-1, 1, -1), ChunkId(3, -3, 1), ChunkId(-3, -1, 3) }) {
            near.push_back(origin + c * s);
        }
    }
    for (const ChunkId &c : near) {
        add_chunk(c);
    }

    // Chunks whose sample grid crosses a 2^32-voxel boundary (the low word wraps inside them).
    const int64_t wrap = int64_t(1) << 32;
    for (int lod : { 0, 4 }) {
        const int64_t span_units = int64_t(2) << lod; // chunk-id units per chunk
        auto straddle = [&](int64_t p_origin_voxel, int p_j) {
            const int64_t boundary = (floor_div(p_origin_voxel, wrap) + p_j) * wrap;
            return floor_div(boundary, half * span_units) * span_units + (int64_t(1) << lod);
        };
        for (int j : { 0, 1 }) {
            add_chunk(ChunkId(straddle(origin_voxel.x, j), straddle(origin_voxel.y, j), straddle(origin_voxel.z, j)));
        }
    }

    // Pairs: the same voxel from the +x neighbour, and from the parent.
    for (const ChunkId &a : near) {
        const int lod = ChunkMath::get_lod(a);
        if (lod > 4) {
            continue;
        }
        const int64_t s = int64_t(1) << lod;
        const ChunkId b = a + ChunkId(2 * s, 0, 0);
        for (int g = 28; g <= 31; g++) {
            r_run.pairs.push_back({ r_run.cases.size(), r_run.cases.size() + 1, "neighbour" });
            r_run.cases.push_back({ a, { float(g), 5.0f, 9.0f } });
            r_run.cases.push_back({ b, { float(g - 28), 5.0f, 9.0f } });
        }
        const ChunkId parent = ChunkMath::get_parent_from_child(a);
        const VoxelPos min_a = ChunkMath::chunk_min_voxel(a, param);
        const VoxelPos min_p = ChunkMath::chunk_min_voxel(parent, param);
        // Per axis, child grid values that land on a parent grid point inside its sample grid.
        auto matches = [&](int64_t p_min_a, int64_t p_min_p) {
            std::vector<std::pair<int, int>> out;
            for (int g = 2; g < 32 && out.size() < 2; g++) {
                const int64_t v = p_min_a + (g - 2) * s;
                const int64_t d = v - p_min_p;
                if (d % (2 * s) == 0 && d / (2 * s) + 2 >= 0 && d / (2 * s) + 2 < 32) {
                    out.push_back({ g, int(d / (2 * s) + 2) });
                }
            }
            return out;
        };
        const auto mx = matches(min_a.x, min_p.x);
        const auto my = matches(min_a.y, min_p.y);
        const auto mz = matches(min_a.z, min_p.z);
        for (const auto &x : mx) {
            for (const auto &y : my) {
                for (const auto &z : mz) {
                    r_run.pairs.push_back({ r_run.cases.size(), r_run.cases.size() + 1, "parent" });
                    r_run.cases.push_back({ a, { float(x.first), float(y.first), float(z.first) } });
                    r_run.cases.push_back({ parent, { float(x.second), float(y.second), float(z.second) } });
                }
            }
        }
    }

    r_run.random_begin = r_run.cases.size();
    add_random_cases(r_run);

    // Absolute points, identical at every origin: the GPU results must not depend on the origin.
    // Values around 0, the 2^32 low-word wrap, cell wraps of high octaves and the int64 range, with
    // fractions (exact in the vertex grid: < 2^-18 of resolution needed).
    r_run.absolute_begin = r_run.cases.size();
    const int64_t P32 = int64_t(1) << 32;
    const int64_t VALUES[] = { 0, -1, 5, P32 - 1, P32, -P32 - 1, 2 * P32 + 7, (int64_t(1) << 45) - 1, -(int64_t(1) << 45),
        (int64_t(1) << 56) + 3, (int64_t(1) << 60) - 1, -(int64_t(1) << 60) };
    constexpr int NV = sizeof(VALUES) / sizeof(VALUES[0]);
    static const Float3 FRACS[] = { { 0.0f, 0.0f, 0.0f }, { 0.5f, 0.25f, 0.75f }, { 1.0f / 262144.0f, 0.0f, 1.0f - 1.0f / 262144.0f },
        { 1.0f - 1.0f / 262144.0f, 1.0f / 262144.0f, 0.5f } };
    for (int i = 0; i < NV; i++) {
        for (const Float3 &f : FRACS) {
            r_run.cases.push_back(lod0_case(Vector3i64(VALUES[i], VALUES[(i + 3) % NV], VALUES[(i + 7) % NV]), f, half));
        }
    }
    // Continuity lines: 81 consecutive LOD0 voxels along x at y near the datum, across voxel low-word
    // wraps and across the cell low-word wraps of the terrain's octaves (where (v + offset) >> k crosses
    // a multiple of 2^32), on both sides of 0.
    std::vector<int64_t> centres = { 0, P32, -P32 };
    for (const ExactNoise::OctaveSet *set : { &p_sdf.get_height(), &p_sdf.get_caves() }) {
        for (const ExactNoise::Octave &o : set->octaves) {
            if (o.k <= 28) {
                centres.push_back((int64_t(1) << (32 + o.k)) - o.offset.x);
                centres.push_back(-(int64_t(1) << (32 + o.k)) - o.offset.x);
            }
        }
    }
    const int64_t datum_voxel = std::llround(SDFDummy::DATUM_M / param.voxel_size);
    for (int64_t c : centres) {
        const size_t begin = r_run.cases.size();
        for (int64_t v = c - 40; v <= c + 40; v++) {
            r_run.cases.push_back(lod0_case(Vector3i64(v, datum_voxel - 3, 1234), Float3{}, half));
        }
        r_run.lines.push_back({ begin, r_run.cases.size() });
    }
}

struct Checks {
    int64_t samples = 0;
    int64_t worldpos_mismatch = 0;
    int64_t anchor_mismatch = 0;
    int64_t pairs = 0;
    int64_t pair_cpu_mismatch = 0; // test vector bug: the pair is not the same voxel on the CPU
    int64_t pair_gpu_mismatch = 0;
    int64_t continuity_steps = 0;
    int64_t continuity_violations = 0;
    int64_t cross_origin_points = 0;
    int64_t cross_origin_mismatch = 0;
    std::vector<int64_t> probe_mismatch;
    // Probe evaluations outside the exact domain (Probe::cpu_exact), and how many of them differ (not a failure).
    int64_t inexact_evaluations = 0;
    int64_t inexact_mismatch = 0;
    // Probe::origin_relative: (random point, probe) pairs compared with the first origin.
    int64_t relative_points = 0;
    int64_t relative_mismatch = 0;
    int64_t cpu_failed = 0;
    Array examples;

    void example(const String &p_text) {
        if (examples.size() < 16) {
            examples.push_back(p_text);
        }
    }
};

// p_first: index of p_probes[0] in r_checks.probe_mismatch.
void check_run(const OriginRun &p_run, const std::vector<Probe> &p_probes, size_t p_first, uint32_t p_words, const PackedByteArray &p_results,
        Checks &r_checks) {
    const ChunkParam &param = p_run.param;
    const uint32_t *res = reinterpret_cast<const uint32_t *>(p_results.ptr());
    const String origin = to_string(param.origin_chunk);
    std::vector<uint32_t> want(p_words);
    for (size_t i = 0; i < p_run.cases.size(); i++) {
        const Case &c = p_run.cases[i];
        const uint32_t *g = res + i * p_words;
        const WorldPos pos = ChunkMath::sample_pos(c.chunk, c.grid, param);
        const VoxelPos anchor = ChunkMath::chunk_min_voxel(c.chunk, param);
        r_checks.samples++;

        const uint32_t want_pos[9] = { lo32(pos.voxel.x), lo32(pos.voxel.y), lo32(pos.voxel.z), uint32_t(hi32(pos.voxel.x)),
            uint32_t(hi32(pos.voxel.y)), uint32_t(hi32(pos.voxel.z)), float_bits(pos.frac.x), float_bits(pos.frac.y),
            float_bits(pos.frac.z) };
        if (std::memcmp(want_pos, g, sizeof(want_pos)) != 0) {
            r_checks.worldpos_mismatch++;
            r_checks.example(vformat("origin %s %s: WorldPos gpu lo (%d %d %d) hi (%d %d %d) cpu voxel %s frac (%s %s %s)", origin,
                    case_str(c), g[0], g[1], g[2], as_i32(g[3]), as_i32(g[4]), as_i32(g[5]), to_string(pos.voxel), pos.frac.x,
                    pos.frac.y, pos.frac.z));
        }
        if (g[9] != lo32(anchor.x) || g[10] != lo32(anchor.y) || g[11] != lo32(anchor.z)) {
            r_checks.anchor_mismatch++;
            r_checks.example(vformat("origin %s %s: anchor_lo gpu (%d %d %d) cpu %s", origin, case_str(c), g[9], g[10], g[11],
                    to_string(anchor)));
        }
        uint32_t w = FIXED_WORDS;
        for (size_t k = 0; k < p_probes.size(); k++) {
            const Probe &probe = p_probes[k];
            bool exact = true;
            if (probe.cpu_exact) {
                exact = probe.cpu_exact(pos, want.data());
            } else {
                probe.cpu(pos, want.data());
            }
            if (!exact) {
                r_checks.inexact_evaluations++;
                r_checks.inexact_mismatch += std::memcmp(want.data(), g + w, probe.words * sizeof(uint32_t)) != 0 ? 1 : 0;
                w += probe.words;
                continue;
            }
            for (int j = 0; j < probe.words; j++) {
                if (want[j] != g[w + j]) {
                    r_checks.probe_mismatch[p_first + k]++;
                    if (probe.is_float) {
                        r_checks.example(vformat("origin %s %s voxel %s: probe %s[%d] gpu %s cpu %s (bits %x / %x)", origin, case_str(c),
                                to_string(pos.voxel), probe.name, j, bits_float(g[w + j]), bits_float(want[j]), g[w + j], want[j]));
                    } else {
                        r_checks.example(vformat("origin %s %s voxel %s: probe %s[%d] gpu %x cpu %x", origin, case_str(c), to_string(pos.voxel),
                                probe.name, j, g[w + j], want[j]));
                    }
                    break;
                }
            }
            w += probe.words;
        }
    }

    for (const Pair &p : p_run.pairs) {
        r_checks.pairs++;
        const Case &a = p_run.cases[p.a];
        const Case &b = p_run.cases[p.b];
        const WorldPos pa = ChunkMath::sample_pos(a.chunk, a.grid, param);
        const WorldPos pb = ChunkMath::sample_pos(b.chunk, b.grid, param);
        if (pa.voxel != pb.voxel || float_bits(pa.frac.x) != float_bits(pb.frac.x) || float_bits(pa.frac.y) != float_bits(pb.frac.y) ||
                float_bits(pa.frac.z) != float_bits(pb.frac.z)) {
            r_checks.pair_cpu_mismatch++;
            r_checks.example(vformat("origin %s %s pair %s / %s: not the same voxel on the CPU", origin, p.what, case_str(a), case_str(b)));
            continue;
        }
        const uint32_t *ga = res + p.a * p_words;
        const uint32_t *gb = res + p.b * p_words;
        // Everything but the chunk's own anchor must match bit for bit.
        bool same = std::memcmp(ga, gb, 9 * sizeof(uint32_t)) == 0 &&
                std::memcmp(ga + FIXED_WORDS, gb + FIXED_WORDS, (p_words - FIXED_WORDS) * sizeof(uint32_t)) == 0;
        if (!same) {
            r_checks.pair_gpu_mismatch++;
            r_checks.example(vformat("origin %s %s pair %s / %s: GPU results differ", origin, p.what, case_str(a), case_str(b)));
        }
    }

    // Lipschitz bounds of the probed noises along the lines.
    for (const std::pair<size_t, size_t> &line : p_run.lines) {
        uint32_t w = FIXED_WORDS;
        for (const Probe &probe : p_probes) {
            if (probe.continuity_bound >= 0.0) {
                for (size_t i = line.first + 1; i < line.second; i++) {
                    r_checks.continuity_steps++;
                    const float s0 = bits_float(res[(i - 1) * p_words + w + probe.continuity_word]);
                    const float s1 = bits_float(res[i * p_words + w + probe.continuity_word]);
                    if (!(std::abs(double(s1) - double(s0)) <= probe.continuity_bound)) {
                        r_checks.continuity_violations++;
                        r_checks.example(vformat("origin %s: %s jumps %s -> %s (bound %s) at %s", origin, probe.name, s0, s1,
                                probe.continuity_bound, case_str(p_run.cases[i])));
                    }
                }
            }
            w += probe.words;
        }
    }
}

// CPU-only checks of the int64 chunk math and the exact-noise helpers.
void check_cpu(Checks &r_checks) {
    auto expect = [&](bool p_ok, const String &p_what) {
        if (!p_ok) {
            r_checks.cpu_failed++;
            r_checks.example("cpu: " + p_what);
        }
    };
    expect(floor_div(-7, 2) == -4 && floor_div(7, 2) == 3 && floor_div(-8, 2) == -4 && floor_div(0, 5) == 0, "floor_div");
    // Parent spans contain their child's span, at negative and huge ids.
    const int64_t big = int64_t(1) << 58;
    for (const ChunkId &c : { ChunkId(1, -1, 3), ChunkId(-12, 4, -20), ChunkId(big + 1, -big - 1, 7), ChunkId(-big + 16, big - 48, -big + 48) }) {
        int parent_lod = -1;
        const ChunkId p = ChunkMath::get_parent_from_child(c, parent_lod);
        const int lod = ChunkMath::get_lod(c);
        const int64_t s = int64_t(1) << lod;
        const bool contained = std::abs(double(c.x - p.x)) + s <= 2 * s && std::abs(double(c.y - p.y)) + s <= 2 * s &&
                std::abs(double(c.z - p.z)) + s <= 2 * s;
        expect(contained && parent_lod == lod + 1 && ChunkMath::get_lod(p) == lod + 1, "parent of " + to_string(c) + " is " + to_string(p));
    }
    // chunk_at_render: the render position lies inside the chunk's bounds.
    ChunkParam param;
    param.origin_chunk = ChunkId(int64_t(7) << 31, -(int64_t(3) << 31), int64_t(66523) << 31);
    const Vector3 m(13.3f, 307.9f, -52.1f);
    const ChunkId c = ChunkMath::chunk_at_render(m, param);
    const Vector3 lo = ChunkMath::grid_to_render(c, Vector3(2, 2, 2), param);
    const Vector3 hi = ChunkMath::grid_to_render(c, Vector3(30, 30, 30), param);
    expect(ChunkMath::get_lod(c) == 0 && lo.x <= m.x && m.x < hi.x && lo.y <= m.y && m.y < hi.y && lo.z <= m.z && m.z < hi.z,
            "chunk_at_render " + to_string(c));
    // Octave decomposition: floor, negative values, k > 32.
    int64_t cell;
    uint32_t q;
    ExactNoise::octave_axis(-1, 0.0f, 3, 0, cell, q);
    expect(cell == -1 && q == (7u << 21), "octave_axis(-1, k=3)");
    ExactNoise::octave_axis(-1, 0.5f, 0, 0, cell, q);
    expect(cell == -1 && q == (1u << 23), "octave_axis(-1 + 0.5, k=0)");
    ExactNoise::octave_axis((int64_t(1) << 61) + (int64_t(5) << 37), 0.0f, 40, 0, cell, q);
    expect(cell == (int64_t(1) << 21) && q == (5u << 21), "octave_axis(2^61 + 5 * 2^37, k=40)");
    // Cells 2^32 apart hash differently.
    int same = 0;
    for (int i = 0; i < 1000; i++) {
        same += ExactNoise::hash_cell(Vector3i64(i, 3, 4), 1u) == ExactNoise::hash_cell(Vector3i64(i + (int64_t(1) << 32), 3, 4), 1u);
    }
    expect(same == 0, vformat("hash_cell: %d of 1000 cells equal to the cell 2^32 away", same));
    expect(WorldCoord::sat_diff(int64_t(1) << 40, 0, 1 << 24) == (1 << 24) && WorldCoord::sat_diff(-5, 7, 1 << 24) == -12, "sat_diff");
}

// The probe shader of one probe set: SHADER_TEMPLATE with the frame, the noise, the terrain and the probes.
String probe_shader(const SDFDummy &p_sdf, const std::vector<Probe> &p_probes, uint32_t &r_words) {
    r_words = FIXED_WORDS;
    String probes_glsl;
    String probe_calls;
    for (const Probe &probe : p_probes) {
        probes_glsl += probe.glsl;
        probe_calls += vformat("    probe_%s(p, b + %du);\n", probe.name, (int64_t)r_words);
        r_words += probe.words;
    }
    String code = String(SHADER_TEMPLATE);
    code = code.replace("/*STRUCTS*/", String(GLSL_STRUCT_DEFINITION) + String(GLSL_FRAME_STRUCTS));
    code = code.replace("/*HELPERS*/", String(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + GPUFrame::glsl_int64() + String(GLSL_EXACT_NOISE) + String(GLSL_PERLIN3_CROSS));
    code = code.replace("/*TERRAIN_SDF*/", p_sdf.get_glsl_code());
    code = code.replace("/*PROBES*/", probes_glsl);
    return code.replace("/*PROBE_CALLS*/", probe_calls);
}

// One dispatch of a probe pipeline over an origin's cases: the results, p_words per case.
PackedByteArray dispatch(RenderingDevice *p_rd, RID p_shader, RID p_pipeline, const OriginRun &p_run, uint32_t p_words) {
    const uint32_t count = (uint32_t)p_run.cases.size();
    PackedByteArray inputs;
    inputs.resize(count * GPUFrame::CHUNK_INPUT_BYTES);
    PackedByteArray samples;
    samples.resize(count * SAMPLE_BYTES);
    for (uint32_t i = 0; i < count; i++) {
        GPUFrame::encode_chunk_input(inputs, i, p_run.cases[i].chunk, p_run.param);
        std::memcpy(samples.ptrw() + i * SAMPLE_BYTES, inputs.ptr() + i * GPUFrame::CHUNK_INPUT_BYTES, GPUFrame::CHUNK_INPUT_BYTES);
        samples.encode_float(i * SAMPLE_BYTES + 32, p_run.cases[i].grid.x);
        samples.encode_float(i * SAMPLE_BYTES + 36, p_run.cases[i].grid.y);
        samples.encode_float(i * SAMPLE_BYTES + 40, p_run.cases[i].grid.z);
        samples.encode_float(i * SAMPLE_BYTES + 44, 0.0f);
    }
    const PackedByteArray param_bytes = GPUFrame::chunk_param_bytes(p_run.param);
    const RID buffer_samples = p_rd->storage_buffer_create(samples.size(), samples);
    const RID buffer_results = p_rd->storage_buffer_create(count * p_words * 4);
    const RID buffer_param = p_rd->uniform_buffer_create(param_bytes.size(), param_bytes);

    TypedArray<RDUniform> uniforms;
    const RID ids[3] = { buffer_samples, buffer_results, buffer_param };
    for (int b = 0; b < 3; b++) {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(b == 2 ? RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER : RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(b);
        u->add_id(ids[b]);
        uniforms.push_back(u);
    }
    const RID uniform_set = p_rd->uniform_set_create(uniforms, p_shader, 0);

    PackedByteArray push;
    push.resize(16);
    push.fill(0);
    push.encode_u32(0, count);
    push.encode_u32(4, p_words);
    push.encode_u32(8, 0); // lod_bias, see PROBE_LODS
    const int64_t list = p_rd->compute_list_begin();
    p_rd->compute_list_bind_compute_pipeline(list, p_pipeline);
    p_rd->compute_list_bind_uniform_set(list, uniform_set, 0);
    p_rd->compute_list_set_push_constant(list, push, push.size());
    p_rd->compute_list_dispatch(list, (count + 63) / 64, 1, 1);
    p_rd->compute_list_end();
    p_rd->submit();
    p_rd->sync();

    const PackedByteArray results = p_rd->buffer_get_data(buffer_results);
    p_rd->free_rid(uniform_set);
    p_rd->free_rid(buffer_samples);
    p_rd->free_rid(buffer_results);
    p_rd->free_rid(buffer_param);
    return results;
}

// The origin-relative probes' words at the random points, against the first origin's (r_ref, filled then).
void check_relative(const OriginRun &p_run, const std::vector<Probe> &p_probes, uint32_t p_words, const PackedByteArray &p_results,
        std::vector<uint32_t> &r_ref, Checks &r_checks) {
    const uint32_t *res = reinterpret_cast<const uint32_t *>(p_results.ptr());
    const bool first = r_ref.empty();
    size_t n = 0;
    uint32_t w = FIXED_WORDS;
    for (const Probe &probe : p_probes) {
        if (probe.origin_relative) {
            for (size_t i = p_run.random_begin; i < p_run.absolute_begin; i++) {
                const uint32_t *g = res + i * p_words + w;
                if (first) {
                    r_ref.insert(r_ref.end(), g, g + probe.words);
                } else {
                    r_checks.relative_points++;
                    if (std::memcmp(g, r_ref.data() + n, probe.words * sizeof(uint32_t)) != 0) {
                        r_checks.relative_mismatch++;
                        r_checks.example(vformat("origin %s: %s at random point %s differs from the first origin", to_string(p_run.param.origin_chunk),
                                probe.name, case_str(p_run.cases[i])));
                    }
                }
                n += probe.words;
            }
        }
        w += probe.words;
    }
}

// Runs one probe set at every origin and checks it (p_first: index of its first probe in the report).
// False if its shader doesn't compile.
bool run_set(RenderingDevice *p_rd, const SDFDummy &p_sdf, const std::vector<Probe> &p_probes, size_t p_first, const std::vector<OriginRun> &p_runs,
        Checks &r_checks) {
    const uint64_t t0 = Time::get_singleton()->get_ticks_msec();
    uint32_t words = 0;
    Ref<RDShaderSource> source;
    source.instantiate();
    source->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, probe_shader(p_sdf, p_probes, words));
    Ref<RDShaderSPIRV> spirv = p_rd->shader_compile_spirv_from_source(source);
    const String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!err.is_empty()) {
        UtilityFunctions::printerr("FrameSpike: SHADER COMPILATION FAILED: " + err);
        return false;
    }
    const RID shader = p_rd->shader_create_from_spirv(spirv);
    const RID pipeline = p_rd->compute_pipeline_create(shader);
    const uint64_t t1 = Time::get_singleton()->get_ticks_msec();

    // GPU results of the absolute points at the first origin, compared with every other origin; the same for
    // the random points of the origin-relative probes.
    std::vector<uint32_t> absolute_ref;
    std::vector<uint32_t> random_ref;
    for (const OriginRun &run : p_runs) {
        const PackedByteArray results = dispatch(p_rd, shader, pipeline, run, words);
        check_run(run, p_probes, p_first, words, results, r_checks);
        check_relative(run, p_probes, words, results, random_ref, r_checks);

        // Everything a sample yields is a function of its absolute position (and chunk id).
        const uint32_t *res = reinterpret_cast<const uint32_t *>(results.ptr());
        const std::vector<uint32_t> absolute(res + run.absolute_begin * words, res + run.cases.size() * words);
        if (absolute_ref.empty()) {
            absolute_ref = absolute;
            continue;
        }
        for (size_t i = 0; i < absolute.size() / words; i++) {
            r_checks.cross_origin_points++;
            if (std::memcmp(absolute.data() + i * words, absolute_ref.data() + i * words, words * sizeof(uint32_t)) != 0) {
                r_checks.cross_origin_mismatch++;
                r_checks.example(vformat("origin %s: absolute point %s differs from origin 0", WorldCoord::to_string(run.param.origin_chunk),
                        case_str(run.cases[run.absolute_begin + i])));
            }
        }
    }
    p_rd->free_rid(pipeline);
    p_rd->free_rid(shader);
    UtilityFunctions::print(vformat("FrameSpike: %d probes from %s, %d words per sample: compile %d ms, run and check %d ms", int64_t(p_probes.size()),
            p_probes[0].name, words, int64_t(t1 - t0), int64_t(Time::get_singleton()->get_ticks_msec() - t1)));
    return true;
}

} // namespace

void FrameSpike::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run"), &FrameSpike::run);
    ClassDB::bind_method(D_METHOD("set_voxel_size", "size"), &FrameSpike::set_voxel_size);
    ClassDB::bind_method(D_METHOD("get_voxel_size"), &FrameSpike::get_voxel_size);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size"), "set_voxel_size", "get_voxel_size");
}

Dictionary FrameSpike::run() {
    Dictionary report;
    report["ok"] = false;
    Checks checks;
    check_cpu(checks);

    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    if (!rd) {
        report["error"] = "no local RenderingDevice (headless or Compatibility renderer)";
        return report;
    }

    // Origins in chunk-id units, multiples of the root span (2^15), mixed signs per axis: 0, a
    // low-word wrap at the origin itself (7 * 2^32 voxels), -21 * 2^32 voxels, ~1e15 m at 0.5 m
    // voxels, and ~2^60 voxels.
    const int64_t r31 = int64_t(1) << 31;
    const int64_t root = int64_t(1) << 15;
    const int64_t far60 = floor_div(-(int64_t(1) << 60), 14 * root) * root;
    const ChunkId origins[] = {
        ChunkId(0, 0, 0),
        ChunkId(r31, -r31, r31),
        ChunkId(-3 * r31, 3 * r31, -3 * r31),
        ChunkId(66523 * r31, -r31, -66523 * r31),
        ChunkId(far60, -far60, far60),
    };

    // One terrain for all origins: it (debug shapes off) does not depend on the origin.
    ChunkParam sdf_param;
    sdf_param.voxel_size = voxel_size;
    SDFDummy sdf;
    sdf.configure(sdf_param, PROBE_LODS[PROBE_LOD_COUNT - 1] + 1);
    std::vector<OriginRun> runs;
    // The archetype programs' anchors: two near each origin, so every origin's points see unsaturated inputs.
    std::vector<VoxelPos> anchors;
    std::vector<VoxelPos> frame_anchors;
    for (const ChunkId &origin : origins) {
        OriginRun run;
        run.param.voxel_size = voxel_size;
        run.param.origin_chunk = origin;
        build_cases(run, sdf);
        for (uint32_t j = 0; j < 2; j++) {
            const uint32_t n = uint32_t(anchors.size());
            anchors.push_back(run.param.origin_voxel() + VoxelPos(random_offset(n, j, 1u, 2048), random_offset(n, j, 2u, 2048), random_offset(n, j, 3u, 2048)));
        }
        frame_anchors.push_back(run.param.origin_voxel() + VoxelPos(-1500, 700, 2100));
        runs.push_back(run);
    }

    // Probe sets, one shader each: the frame / noise / terrain table, then the archetype probes.
    std::vector<std::vector<Probe>> sets = { make_probes(sdf) };
    const std::vector<Probe> arch = ArchetypeProbes::make(anchors, frame_anchors, ARCH_SHAPE_SIZE, FUZZ_PROGRAMS);
    int64_t set_glsl = 0;
    for (const Probe &probe : arch) {
        if (sets.size() == 1 || sets.back().size() == PROBES_PER_SET || set_glsl + probe.glsl.length() > MAX_SET_GLSL) {
            sets.emplace_back();
            set_glsl = 0;
        }
        sets.back().push_back(probe);
        set_glsl += probe.glsl.length();
    }
    std::vector<String> names;
    for (const std::vector<Probe> &set : sets) {
        for (const Probe &probe : set) {
            names.push_back(probe.name);
        }
    }
    checks.probe_mismatch.assign(names.size(), 0);
    size_t first = 0;
    for (const std::vector<Probe> &set : sets) {
        if (!run_set(rd, sdf, set, first, runs, checks)) {
            report["error"] = "shader compilation failed";
            memdelete(rd);
            return report;
        }
        first += set.size();
    }
    memdelete(rd);

    report["origins"] = (int64_t)(sizeof(origins) / sizeof(origins[0]));
    int64_t cases = 0;
    for (const OriginRun &run : runs) {
        cases += int64_t(run.cases.size());
    }
    report["cases"] = cases; // over every origin; "samples" counts them once per probe set
    report["probe_sets"] = (int64_t)sets.size();
    report["arch_programs"] = (int64_t)arch.size();
    report["samples"] = checks.samples;
    report["worldpos_mismatch"] = checks.worldpos_mismatch;
    report["anchor_mismatch"] = checks.anchor_mismatch;
    report["pairs"] = checks.pairs;
    report["pair_cpu_mismatch"] = checks.pair_cpu_mismatch;
    report["pair_gpu_mismatch"] = checks.pair_gpu_mismatch;
    report["continuity_steps"] = checks.continuity_steps;
    report["continuity_violations"] = checks.continuity_violations;
    report["cross_origin_points"] = checks.cross_origin_points;
    report["cross_origin_mismatch"] = checks.cross_origin_mismatch;
    report["cpu_checks_failed"] = checks.cpu_failed;
    report["inexact_evaluations"] = checks.inexact_evaluations;
    report["inexact_mismatch"] = checks.inexact_mismatch;
    report["relative_points"] = checks.relative_points;
    report["relative_mismatch"] = checks.relative_mismatch;
    // Per probe; the fuzzed programs summed into one entry.
    int64_t probe_total = 0;
    Dictionary probe_report;
    for (size_t k = 0; k < names.size(); k++) {
        const String key = names[k].begins_with("arch_fuzz_") ? String("arch_fuzz") : names[k];
        probe_report[key] = int64_t(probe_report.get(key, 0)) + checks.probe_mismatch[k];
        probe_total += checks.probe_mismatch[k];
    }
    report["probe_mismatch"] = probe_report;
    report["examples"] = checks.examples;
    report["ok"] = checks.samples > 0 && checks.pairs > 0 && checks.continuity_steps > 0 && checks.cross_origin_points > 0 &&
            checks.worldpos_mismatch == 0 && checks.anchor_mismatch == 0 && checks.pair_cpu_mismatch == 0 && checks.pair_gpu_mismatch == 0 &&
            checks.continuity_violations == 0 && checks.cross_origin_mismatch == 0 && checks.cpu_failed == 0 && probe_total == 0 &&
            checks.relative_points > 0 && checks.relative_mismatch == 0;
    return report;
}
