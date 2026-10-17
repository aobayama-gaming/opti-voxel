#include "frame_spike.hpp"

#include "chunk_math.hpp"
#include "exact_noise.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"
#include "sdf_dummy.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
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

// A value computed from a WorldPos on the CPU and on the GPU, `words` uint32 (floats as their bits),
// compared bit for bit. A new noise adds its entries in make_probes().
struct Probe {
    String name;
    int words = 1;
    bool is_float = false;
    String glsl; // defines "void probe_<name>(WorldPos p, uint b)", writing results[b + 0 .. words - 1]
    std::function<void(const WorldPos &, uint32_t *)> cpu;
    // >= 0: word `continuity_word` may change by at most this much per LOD0 voxel along the
    // continuity lines (a wrong corner hash at a low-word carry shows up as a jump).
    double continuity_bound = -1.0;
    int continuity_word = 0;
};

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
    // The terrain's two fBms and the terrain itself (gradient included).
    auto fbm_bound = [](const OctaveSet &p_set, double p_g) {
        double b = 0.0;
        for (const Octave &o : p_set.octaves) {
            b += double(o.damp) * p_g;
        }
        return b;
    };
    const SDFDummy *sdf = &p_sdf;
    for (int which = 0; which < 2; which++) {
        Probe p;
        p.name = which == 0 ? "terrain_height" : "terrain_caves";
        p.words = 4;
        p.is_float = true;
        p.glsl = vformat("void probe_%s(WorldPos p, uint b) {\n    vec4 v = %s(p);\n", p.name, p.name);
        p.glsl += "    for (uint i = 0u; i < 4u; i++) { results[b + i] = floatBitsToUint(v[i]); }\n}\n";
        p.cpu = [sdf, which](const WorldPos &q, uint32_t *r) {
            const Float4 v = fbm(which == 0 ? sdf->get_height() : sdf->get_caves(), q);
            r[0] = float_bits(v.x);
            r[1] = float_bits(v.y);
            r[2] = float_bits(v.z);
            r[3] = float_bits(v.w);
        };
        p.continuity_bound = fbm_bound(which == 0 ? p_sdf.get_height() : p_sdf.get_caves(), which == 0 ? 6.0 : 4.0);
        p.continuity_word = 3;
        probes.push_back(p);
    }
    {
        Probe p;
        p.name = "terrain_sdf";
        p.words = 4;
        p.is_float = true;
        p.glsl = "void probe_terrain_sdf(WorldPos p, uint b) {\n    vec4 v = terrain_sdf(p);\n";
        p.glsl += "    for (uint i = 0u; i < 4u; i++) { results[b + i] = floatBitsToUint(v[i]); }\n}\n";
        p.cpu = [sdf](const WorldPos &q, uint32_t *r) {
            const Float4 v = sdf->sdf(q);
            r[0] = float_bits(v.x);
            r[1] = float_bits(v.y);
            r[2] = float_bits(v.z);
            r[3] = float_bits(v.w);
        };
        probes.push_back(p);
    }
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
layout(push_constant, std430) uniform Params { uint count; uint words; uint pad0; uint pad1; } params;

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
    int64_t cpu_failed = 0;
    Array examples;

    void example(const String &p_text) {
        if (examples.size() < 16) {
            examples.push_back(p_text);
        }
    }
};

void check_run(const OriginRun &p_run, const std::vector<Probe> &p_probes, uint32_t p_words, const PackedByteArray &p_results, Checks &r_checks) {
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
            p_probes[k].cpu(pos, want.data());
            for (int j = 0; j < probe.words; j++) {
                if (want[j] != g[w + j]) {
                    r_checks.probe_mismatch[k]++;
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

    // One shader for all origins: the terrain (debug shapes off) does not depend on the origin.
    ChunkParam sdf_param;
    sdf_param.voxel_size = voxel_size;
    SDFDummy sdf;
    sdf.configure(sdf_param);
    const std::vector<Probe> probes = make_probes(sdf);
    checks.probe_mismatch.assign(probes.size(), 0);
    uint32_t words = FIXED_WORDS;
    String probes_glsl;
    String probe_calls;
    for (const Probe &probe : probes) {
        probes_glsl += probe.glsl;
        probe_calls += vformat("    probe_%s(p, b + %du);\n", probe.name, (int64_t)words);
        words += probe.words;
    }
    String code = String(SHADER_TEMPLATE);
    code = code.replace("/*STRUCTS*/", String(GLSL_STRUCT_DEFINITION) + String(GLSL_FRAME_STRUCTS));
    code = code.replace("/*HELPERS*/", String(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + String(GLSL_EXACT_NOISE));
    code = code.replace("/*TERRAIN_SDF*/", sdf.get_glsl_code());
    code = code.replace("/*PROBES*/", probes_glsl);
    code = code.replace("/*PROBE_CALLS*/", probe_calls);

    Ref<RDShaderSource> source;
    source.instantiate();
    source->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, code);
    Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(source);
    const String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!err.is_empty()) {
        UtilityFunctions::printerr("FrameSpike: SHADER COMPILATION FAILED: " + err);
        report["error"] = "shader compilation failed";
        memdelete(rd);
        return report;
    }
    const RID shader = rd->shader_create_from_spirv(spirv);
    const RID pipeline = rd->compute_pipeline_create(shader);

    // GPU results of the absolute points at the first origin, compared with every other origin.
    std::vector<uint32_t> absolute_ref;
    for (const ChunkId &origin : origins) {
        OriginRun run;
        run.param.voxel_size = voxel_size;
        run.param.origin_chunk = origin;
        build_cases(run, sdf);
        const uint32_t count = (uint32_t)run.cases.size();

        PackedByteArray inputs;
        inputs.resize(count * GPUFrame::CHUNK_INPUT_BYTES);
        PackedByteArray samples;
        samples.resize(count * SAMPLE_BYTES);
        for (uint32_t i = 0; i < count; i++) {
            GPUFrame::encode_chunk_input(inputs, i, run.cases[i].chunk, run.param);
            std::memcpy(samples.ptrw() + i * SAMPLE_BYTES, inputs.ptr() + i * GPUFrame::CHUNK_INPUT_BYTES, GPUFrame::CHUNK_INPUT_BYTES);
            samples.encode_float(i * SAMPLE_BYTES + 32, run.cases[i].grid.x);
            samples.encode_float(i * SAMPLE_BYTES + 36, run.cases[i].grid.y);
            samples.encode_float(i * SAMPLE_BYTES + 40, run.cases[i].grid.z);
            samples.encode_float(i * SAMPLE_BYTES + 44, 0.0f);
        }
        const PackedByteArray param_bytes = GPUFrame::chunk_param_bytes(run.param);
        const RID buffer_samples = rd->storage_buffer_create(samples.size(), samples);
        const RID buffer_results = rd->storage_buffer_create(count * words * 4);
        const RID buffer_param = rd->uniform_buffer_create(param_bytes.size(), param_bytes);

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
        const RID uniform_set = rd->uniform_set_create(uniforms, shader, 0);

        PackedByteArray push;
        push.resize(16);
        push.fill(0);
        push.encode_u32(0, count);
        push.encode_u32(4, words);
        const int64_t list = rd->compute_list_begin();
        rd->compute_list_bind_compute_pipeline(list, pipeline);
        rd->compute_list_bind_uniform_set(list, uniform_set, 0);
        rd->compute_list_set_push_constant(list, push, push.size());
        rd->compute_list_dispatch(list, (count + 63) / 64, 1, 1);
        rd->compute_list_end();
        rd->submit();
        rd->sync();

        const PackedByteArray results = rd->buffer_get_data(buffer_results);
        check_run(run, probes, words, results, checks);

        // Everything a sample yields is a function of its absolute position (and chunk id).
        const uint32_t *res = reinterpret_cast<const uint32_t *>(results.ptr());
        const std::vector<uint32_t> absolute(res + run.absolute_begin * words, res + size_t(count) * words);
        if (absolute_ref.empty()) {
            absolute_ref = absolute;
        } else {
            for (size_t i = 0; i < absolute.size() / words; i++) {
                checks.cross_origin_points++;
                if (std::memcmp(absolute.data() + i * words, absolute_ref.data() + i * words, words * sizeof(uint32_t)) != 0) {
                    checks.cross_origin_mismatch++;
                    checks.example(vformat("origin %s: absolute point %s differs from origin 0", WorldCoord::to_string(origin),
                            case_str(run.cases[run.absolute_begin + i])));
                }
            }
        }

        rd->free_rid(uniform_set);
        rd->free_rid(buffer_samples);
        rd->free_rid(buffer_results);
        rd->free_rid(buffer_param);
    }
    rd->free_rid(pipeline);
    rd->free_rid(shader);
    memdelete(rd);

    report["origins"] = (int64_t)(sizeof(origins) / sizeof(origins[0]));
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
    int64_t probe_total = 0;
    Dictionary probe_report;
    for (size_t k = 0; k < probes.size(); k++) {
        probe_report[probes[k].name] = checks.probe_mismatch[k];
        probe_total += checks.probe_mismatch[k];
    }
    report["probe_mismatch"] = probe_report;
    report["examples"] = checks.examples;
    report["ok"] = checks.samples > 0 && checks.pairs > 0 && checks.continuity_steps > 0 && checks.cross_origin_points > 0 &&
            checks.worldpos_mismatch == 0 && checks.anchor_mismatch == 0 && checks.pair_cpu_mismatch == 0 && checks.pair_gpu_mismatch == 0 &&
            checks.continuity_violations == 0 && checks.cross_origin_mismatch == 0 && checks.cpu_failed == 0 && probe_total == 0;
    return report;
}
