#include "frame_spike.hpp"

#include "chunk_math.hpp"
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
#include <vector>

using namespace WorldCoord;

namespace {

uint32_t float_bits(float p_f) {
    uint32_t u;
    std::memcpy(&u, &p_f, sizeof(u));
    return u;
}

float bits_float(uint32_t p_u) {
    float f;
    std::memcpy(&f, &p_u, sizeof(f));
    return f;
}

// A value computed from a WorldPos on the CPU and on the GPU. The exact noise adds its entry here.
struct Probe {
    const char *name;
    const char *glsl; // defines "uint probe_<name>(WorldPos p)" (floats as floatBitsToUint)
    uint32_t (*cpu)(const WorldPos &p_pos, const ChunkParam &p_param);
    bool is_float;
    double tolerance; // float probes: largest accepted |cpu - gpu|
};

// Plumbing check: an integer hash of every bit of the position.
uint32_t voxel_hash_cpu(const WorldPos &p_pos, const ChunkParam &) {
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
uint probe_voxel_hash(WorldPos p) {
    uvec3 v = p.lo * 1664525u + uvec3(p.hi) * 22695477u + floatBitsToUint(p.frac) * 2891336453u;
    uint h = v.x ^ (v.y * 0x9E3779B1u) ^ (v.z * 0x85EBCA77u);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}
)glsl";

const Probe PROBES[] = {
    { "voxel_hash", VOXEL_HASH_GLSL, voxel_hash_cpu, false, 0.0 },
};
constexpr int PROBE_COUNT = sizeof(PROBES) / sizeof(PROBES[0]);

// Per sample: lo(3) hi(3) frac bits(3) anchor_lo(3) legacy_rel_voxels bits(3) terrain_sdf.w bits(1),
// then one word per probe.
constexpr uint32_t FIXED_WORDS = 16;
constexpr uint32_t WORDS = FIXED_WORDS + PROBE_COUNT;
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
    vec3 rel = legacy_rel_voxels(p, chunk_param);
    vec4 sdf = terrain_sdf(p);
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
    results[b + 12u] = floatBitsToUint(rel.x);
    results[b + 13u] = floatBitsToUint(rel.y);
    results[b + 14u] = floatBitsToUint(rel.z);
    results[b + 15u] = floatBitsToUint(sdf.w);
    /*PROBE_STORES*/
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
    std::vector<size_t> line; // consecutive voxels across a low-word wrap
};

String case_str(const Case &p_case) {
    return vformat("chunk %s grid (%s, %s, %s)", to_string(p_case.chunk), p_case.grid.x, p_case.grid.y, p_case.grid.z);
}

void build_cases(OriginRun &r_run) {
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

    // Consecutive LOD0 voxels across the low-word wrap nearest to the origin (only where the
    // origin-relative metre SDF is meaningful, i.e. close to the origin).
    const int64_t w = floor_div(origin_voxel.x + wrap / 2, wrap) * wrap;
    if (std::abs(double(w - origin_voxel.x)) <= 1.0e6) {
        auto lod0 = [&](int64_t p_voxel, int64_t &r_id) {
            r_id = floor_div(p_voxel, 2 * half) * 2 + 1;
            return float(2 + (p_voxel - (r_id - 1) * half));
        };
        for (int64_t v = w - 40; v <= w + 40; v++) {
            ChunkId id;
            Float3 grid;
            grid.x = lod0(v, id.x);
            grid.y = lod0(origin_voxel.y + 600, id.y);
            grid.z = lod0(origin_voxel.z, id.z);
            r_run.line.push_back(r_run.cases.size());
            r_run.cases.push_back({ id, grid });
        }
    }
}

struct Checks {
    int64_t samples = 0;
    int64_t worldpos_mismatch = 0;
    int64_t anchor_mismatch = 0;
    int64_t rel_mismatch = 0;
    int64_t pairs = 0;
    int64_t pair_cpu_mismatch = 0; // test vector bug: the pair is not the same voxel on the CPU
    int64_t pair_gpu_mismatch = 0;
    int64_t continuity_steps = 0;
    int64_t continuity_violations = 0;
    int64_t probe_mismatch[PROBE_COUNT] = {};
    int64_t cpu_failed = 0;
    Array examples;

    void example(const String &p_text) {
        if (examples.size() < 12) {
            examples.push_back(p_text);
        }
    }
};

void check_run(const OriginRun &p_run, const PackedByteArray &p_results, Checks &r_checks) {
    const ChunkParam &param = p_run.param;
    const uint32_t *res = reinterpret_cast<const uint32_t *>(p_results.ptr());
    const String origin = to_string(param.origin_chunk);
    for (size_t i = 0; i < p_run.cases.size(); i++) {
        const Case &c = p_run.cases[i];
        const uint32_t *g = res + i * WORDS;
        const WorldPos pos = ChunkMath::sample_pos(c.chunk, c.grid, param);
        const VoxelPos anchor = ChunkMath::chunk_min_voxel(c.chunk, param);
        const Float3 rel = ChunkMath::legacy_rel_voxels(pos, param);
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
        if (g[12] != float_bits(rel.x) || g[13] != float_bits(rel.y) || g[14] != float_bits(rel.z)) {
            r_checks.rel_mismatch++;
            r_checks.example(vformat("origin %s %s: legacy_rel_voxels gpu (%s %s %s) cpu (%s %s %s)", origin, case_str(c),
                    bits_float(g[12]), bits_float(g[13]), bits_float(g[14]), rel.x, rel.y, rel.z));
        }
        for (int k = 0; k < PROBE_COUNT; k++) {
            const uint32_t want = PROBES[k].cpu(pos, param);
            const uint32_t got = g[FIXED_WORDS + k];
            const bool ok = PROBES[k].is_float ? std::abs(double(bits_float(want)) - double(bits_float(got))) <= PROBES[k].tolerance
                                               : want == got;
            if (!ok) {
                r_checks.probe_mismatch[k]++;
                r_checks.example(vformat("origin %s %s: probe %s gpu %d cpu %d", origin, case_str(c), PROBES[k].name, got, want));
            }
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
        const uint32_t *ga = res + p.a * WORDS;
        const uint32_t *gb = res + p.b * WORDS;
        // Everything but the chunk's own anchor must match bit for bit.
        bool same = std::memcmp(ga, gb, 9 * sizeof(uint32_t)) == 0 && std::memcmp(ga + 12, gb + 12, (WORDS - 12) * sizeof(uint32_t)) == 0;
        if (!same) {
            r_checks.pair_gpu_mismatch++;
            r_checks.example(vformat("origin %s %s pair %s / %s: GPU results differ (sdf %s vs %s)", origin, p.what, case_str(a),
                    case_str(b), bits_float(ga[15]), bits_float(gb[15])));
        }
    }

    // Lipschitz bound of the metre SDF over one voxel, with margin.
    const double max_step = 2.0 * param.voxel_size + 1e-3;
    for (size_t k = 1; k < p_run.line.size(); k++) {
        r_checks.continuity_steps++;
        const float s0 = bits_float(res[p_run.line[k - 1] * WORDS + 15]);
        const float s1 = bits_float(res[p_run.line[k] * WORDS + 15]);
        if (!(std::abs(double(s1) - double(s0)) <= max_step)) {
            r_checks.continuity_violations++;
            r_checks.example(vformat("origin %s: terrain_sdf jumps %s -> %s at %s", origin, s0, s1, case_str(p_run.cases[p_run.line[k]])));
        }
    }
}

// CPU-only checks of the int64 chunk math.
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
    // chunk_at_render: the render position lies inside the chunk's (nudge-free) bounds.
    ChunkParam param;
    param.origin_chunk = ChunkId(int64_t(7) << 31, -(int64_t(3) << 31), int64_t(66523) << 31);
    const Vector3 m(13.3f, 307.9f, -52.1f);
    const ChunkId c = ChunkMath::chunk_at_render(m, param);
    const Vector3 nudge(LEGACY_NUDGE_M, LEGACY_NUDGE_M, LEGACY_NUDGE_M);
    const Vector3 lo = ChunkMath::grid_to_render(c, Vector3(2, 2, 2), param) - nudge;
    const Vector3 hi = ChunkMath::grid_to_render(c, Vector3(30, 30, 30), param) - nudge;
    expect(ChunkMath::get_lod(c) == 0 && lo.x <= m.x && m.x < hi.x && lo.y <= m.y && m.y < hi.y && lo.z <= m.z && m.z < hi.z,
            "chunk_at_render " + to_string(c));
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

    // One shader for all origins: the SDF code does not depend on the origin.
    SDFDummy sdf_for_glsl;
    String probes_glsl;
    String probe_stores;
    for (int k = 0; k < PROBE_COUNT; k++) {
        probes_glsl += PROBES[k].glsl;
        probe_stores += vformat("    results[b + %du] = probe_%s(p);\n", (int64_t)(FIXED_WORDS + k), PROBES[k].name);
    }
    String code = String(SHADER_TEMPLATE);
    code = code.replace("/*STRUCTS*/", String(GLSL_STRUCT_DEFINITION) + String(GLSL_FRAME_STRUCTS));
    code = code.replace("/*HELPERS*/", String(GLSL_HELPER_FUNCTION) + String(GLSL_NOISE) + GPUFrame::glsl_functions());
    code = code.replace("/*TERRAIN_SDF*/", sdf_for_glsl.get_glsl_code());
    code = code.replace("/*PROBES*/", probes_glsl);
    code = code.replace("/*PROBE_STORES*/", probe_stores);

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

    for (const ChunkId &origin : origins) {
        OriginRun run;
        run.param.voxel_size = voxel_size;
        run.param.origin_chunk = origin;
        build_cases(run);
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
        const RID buffer_results = rd->storage_buffer_create(count * WORDS * 4);
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
        push.encode_u32(4, WORDS);
        const int64_t list = rd->compute_list_begin();
        rd->compute_list_bind_compute_pipeline(list, pipeline);
        rd->compute_list_bind_uniform_set(list, uniform_set, 0);
        rd->compute_list_set_push_constant(list, push, push.size());
        rd->compute_list_dispatch(list, (count + 63) / 64, 1, 1);
        rd->compute_list_end();
        rd->submit();
        rd->sync();

        const PackedByteArray results = rd->buffer_get_data(buffer_results);
        check_run(run, results, checks);

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
    report["legacy_rel_mismatch"] = checks.rel_mismatch;
    report["pairs"] = checks.pairs;
    report["pair_cpu_mismatch"] = checks.pair_cpu_mismatch;
    report["pair_gpu_mismatch"] = checks.pair_gpu_mismatch;
    report["continuity_steps"] = checks.continuity_steps;
    report["continuity_violations"] = checks.continuity_violations;
    report["cpu_checks_failed"] = checks.cpu_failed;
    int64_t probe_total = 0;
    Dictionary probes;
    for (int k = 0; k < PROBE_COUNT; k++) {
        probes[PROBES[k].name] = checks.probe_mismatch[k];
        probe_total += checks.probe_mismatch[k];
    }
    report["probe_mismatch"] = probes;
    report["examples"] = checks.examples;
    report["ok"] = checks.samples > 0 && checks.pairs > 0 && checks.continuity_steps > 0 && checks.worldpos_mismatch == 0 &&
            checks.anchor_mismatch == 0 && checks.rel_mismatch == 0 && checks.pair_cpu_mismatch == 0 && checks.pair_gpu_mismatch == 0 &&
            checks.continuity_violations == 0 && checks.cpu_failed == 0 && probe_total == 0;
    return report;
}
