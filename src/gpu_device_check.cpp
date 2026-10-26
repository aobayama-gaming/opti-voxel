#include "gpu_device_check.hpp"

#include "exact_noise.hpp"
#include "exact_noise_glsl.hpp"
#include "frame_probe.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"

#include <cstring>

namespace GPUDeviceCheck {

namespace {

constexpr const char PLUMBING_GLSL[] = R"glsl(#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) restrict readonly buffer In { uint data[]; } input_words;
layout(set = 0, binding = 1, std430) restrict writeonly buffer Out { uint data[]; } output_words;
layout(set = 0, binding = 2, std140) uniform Mul { uint mul; } mul;
layout(set = 0, binding = 3, std430) restrict readonly buffer Unused { uint data[]; } unused;
layout(push_constant, std430) uniform Push { uint count; uint add; } push;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= push.count) {
        return;
    }
    output_words.data[i] = input_words.data[i] * mul.mul + push.add;
}
)glsl";

constexpr const char NOISE_GLSL[] = R"glsl(#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
/*STRUCTS*/
struct Point { uvec4 lo_lod; ivec4 hi; vec4 frac; };
layout(set = 0, binding = 0, std430) restrict readonly buffer Points { Point points[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer Values { vec4 values[]; };
layout(set = 0, binding = 2, std140) uniform ChunkParamUniform { ChunkParam chunk_param; };
layout(push_constant, std430) uniform Push { uint count; uint lod_bias; } push;
/*HELPERS*/
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= push.count) {
        return;
    }
    WorldPos p;
    p.lo = points[i].lo_lod.xyz;
    p.hi = points[i].hi.xyz;
    p.frac = points[i].frac.xyz;
    values[i] = check_fbm(p, int(points[i].lo_lod.w + push.lod_bias));
}
)glsl";

constexpr const char TEXTURE_GLSL[] = R"glsl(#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16ui) uniform restrict writeonly uimage2D image;
void main() {
    uvec2 p = gl_GlobalInvocationID.xy;
    imageStore(image, ivec2(p), uvec4(p.x, p.y, p.x ^ p.y, 7u));
}
)glsl";

constexpr uint32_t PLUMBING_WORDS = 1000;
constexpr uint32_t TEXTURE_W = 16;
constexpr uint32_t TEXTURE_H = 8;
constexpr uint32_t PLUMBING_MUL = 7;
constexpr uint32_t PLUMBING_ADD = 11;
constexpr uint32_t POINTS_PER_ORIGIN = 1024;
constexpr uint32_t POINT_BYTES = 48;
constexpr double CHECK_VOXEL_M = 0.05; // the 5 cm world: the finest octaves, the widest LOD range

struct Push {
    uint32_t count;
    uint32_t word;
};

void check_plumbing(ov::gpu::Device &p_device, Report &r_report) {
    using namespace ov::gpu;
    ov::str error;
    const Pipeline pipeline = p_device.create_compute_pipeline("gpu_check_plumbing", PLUMBING_GLSL, error);
    if (!pipeline.is_valid()) {
        r_report.error = "plumbing shader: " + error;
        return;
    }
    std::vector<uint32_t> input(PLUMBING_WORDS);
    for (uint32_t i = 0; i < PLUMBING_WORDS; i++) {
        input[i] = i * 3u;
    }
    const uint32_t mul[4] = { PLUMBING_MUL, 0, 0, 0 };
    const Buffer in = p_device.create_buffer(BufferKind::STORAGE, PLUMBING_WORDS * 4, input.data());
    const Buffer out = p_device.create_buffer(BufferKind::STORAGE, PLUMBING_WORDS * 4);
    const Buffer copy = p_device.create_buffer(BufferKind::STORAGE, PLUMBING_WORDS * 4);
    const Buffer uniform = p_device.create_buffer(BufferKind::UNIFORM, sizeof(mul), mul);
    const Buffer unused = p_device.create_buffer(BufferKind::STORAGE, 16);
    const BindSet set = p_device.create_bind_set(pipeline,
            { { 0, BufferKind::STORAGE, in }, { 1, BufferKind::STORAGE, out }, { 2, BufferKind::UNIFORM, uniform }, { 3, BufferKind::STORAGE, unused } });

    // The second half rewritten before the dispatch, the copy's first 4 words cleared after it.
    for (uint32_t i = PLUMBING_WORDS / 2; i < PLUMBING_WORDS; i++) {
        input[i] = 100000u + i;
    }
    p_device.update_buffer(in, PLUMBING_WORDS / 2 * 4, PLUMBING_WORDS / 2 * 4, input.data() + PLUMBING_WORDS / 2);
    const Push push{ PLUMBING_WORDS, PLUMBING_ADD };
    p_device.dispatch(pipeline, set, &push, sizeof(push), (PLUMBING_WORDS + 63) / 64, 1, 1);
    p_device.copy_buffer(out, copy, 0, 0, PLUMBING_WORDS * 4);
    p_device.clear_buffer(copy, 0, 16);
    ov::bytes async_copy;
    bool delivered = false;
    p_device.read_buffer_async(copy, 0, PLUMBING_WORDS * 4, [&](const ov::bytes &p_data) {
        async_copy = p_data;
        delivered = true;
    });
    p_device.sync();
    const ov::bytes direct = p_device.read_buffer(out, 0, PLUMBING_WORDS * 4);
    if (!delivered || async_copy.size() != PLUMBING_WORDS * 4 || direct.size() != PLUMBING_WORDS * 4) {
        r_report.error = ov::format("plumbing readback: async %s (%d bytes), direct %d bytes", delivered ? "delivered" : "missing",
                int64_t(async_copy.size()), int64_t(direct.size()));
    } else {
        for (uint32_t i = 0; i < PLUMBING_WORDS; i++) {
            const uint32_t want = input[i] * PLUMBING_MUL + PLUMBING_ADD;
            r_report.plumbing_checked += 2;
            r_report.plumbing_mismatch += ov::get_u32(direct.data() + i * 4) != want ? 1 : 0;
            r_report.plumbing_mismatch += ov::get_u32(async_copy.data() + i * 4) != (i < 4 ? 0u : want) ? 1 : 0;
        }
    }
    for (Handle h : { Handle(set), Handle(in), Handle(out), Handle(copy), Handle(uniform), Handle(unused), Handle(pipeline) }) {
        p_device.free(h);
    }
}

// A storage image written by a shader (A), a region of it copied into a texture made with data (B), a texture updated
// then cleared (C); B and C read back.
void check_textures(ov::gpu::Device &p_device, Report &r_report) {
    using namespace ov::gpu;
    ov::str error;
    const Pipeline pipeline = p_device.create_compute_pipeline("gpu_check_texture", TEXTURE_GLSL, error);
    if (!pipeline.is_valid()) {
        r_report.error = "texture shader: " + error;
        return;
    }
    const uint32_t texels = TEXTURE_W * TEXTURE_H;
    std::vector<uint16_t> start(texels * 4), update(texels * 4);
    for (uint32_t i = 0; i < texels * 4; i++) {
        start[i] = uint16_t(1000 + i);
        update[i] = uint16_t(5000 + i);
    }
    const Texture a = p_device.create_texture(TextureFormat::RGBA16_UINT, TEXTURE_W, TEXTURE_H, TEXTURE_STORAGE | TEXTURE_COPY_SRC | TEXTURE_COPY_DST);
    const Texture b = p_device.create_texture(TextureFormat::RGBA16_UINT, TEXTURE_W, TEXTURE_H, TEXTURE_COPY_SRC | TEXTURE_COPY_DST, start.data());
    const Texture c = p_device.create_texture(TextureFormat::RGBA16_UINT, TEXTURE_W, TEXTURE_H, TEXTURE_COPY_SRC | TEXTURE_COPY_DST | TEXTURE_UPDATE);
    const BindSet set = p_device.create_bind_set(pipeline, { Binding::storage_image(0, a) });
    p_device.clear_texture(a);
    p_device.dispatch(pipeline, set, nullptr, 0, TEXTURE_W / 8, TEXTURE_H / 8, 1);
    // A's 4 x 2 block at (2, 1) to (5, 3) of B.
    p_device.copy_texture(a, b, 2, 1, 5, 3, 4, 2);
    p_device.update_texture(c, update.data(), update.size() * 2);
    p_device.clear_texture(c);
    ov::bytes got_b, got_c;
    p_device.read_texture_async(b, [&](const ov::bytes &p_data) { got_b = p_data; });
    p_device.read_texture_async(c, [&](const ov::bytes &p_data) { got_c = p_data; });
    p_device.sync();
    if (got_b.size() != texels * 8 || got_c.size() != texels * 8) {
        r_report.error = ov::format("texture readback: %d and %d bytes of %d", int64_t(got_b.size()), int64_t(got_c.size()), int64_t(texels * 8));
    } else {
        for (uint32_t y = 0; y < TEXTURE_H; y++) {
            for (uint32_t x = 0; x < TEXTURE_W; x++) {
                const bool copied = x >= 5 && x < 9 && y >= 3 && y < 5;
                const uint32_t ax = x - 5 + 2, ay = y - 3 + 1; // the copy's source texel in A
                for (uint32_t k = 0; k < 4; k++) {
                    const uint32_t i = (y * TEXTURE_W + x) * 4 + k;
                    const uint16_t from_a[4] = { uint16_t(ax), uint16_t(ay), uint16_t(ax ^ ay), 7 };
                    uint16_t vb, vc;
                    std::memcpy(&vb, got_b.data() + i * 2, 2);
                    std::memcpy(&vc, got_c.data() + i * 2, 2);
                    r_report.plumbing_checked += 2;
                    r_report.plumbing_mismatch += vb != (copied ? from_a[k] : start[i]) ? 1 : 0;
                    r_report.plumbing_mismatch += vc != 0 ? 1 : 0;
                }
            }
        }
    }
    for (Handle h : { Handle(set), Handle(a), Handle(b), Handle(c), Handle(pipeline) }) {
        p_device.free(h);
    }
}

void check_noise(ov::gpu::Device &p_device, Report &r_report) {
    using namespace ov::gpu;
    const ExactNoise::OctaveSet set = ExactNoise::reference_fbm3d(CHECK_VOXEL_M);
    const ov::str range_error = ExactNoise::gradient_range_error(set, PROBE_LODS[PROBE_LOD_COUNT - 1] + 1);
    if (!range_error.empty()) {
        r_report.error = range_error;
        return;
    }
    const ov::str helpers = ov::str(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + GPUFrame::glsl_int64() + GLSL_EXACT_NOISE +
            ExactNoise::glsl_fbm("check_fbm", set);
    const ov::str code = ov::replace_all(ov::str(NOISE_GLSL),
            { { "/*STRUCTS*/", ov::str(GLSL_STRUCT_DEFINITION) + GLSL_FRAME_STRUCTS }, { "/*HELPERS*/", helpers } });
    ov::str error;
    const Pipeline pipeline = p_device.create_compute_pipeline("gpu_check_noise", code, error);
    if (!pipeline.is_valid()) {
        r_report.error = "noise shader: " + error;
        return;
    }

    // Hash-random points within 2^20 voxels of origins around 0, a low-word wrap, and far out, at fractions of 2^-20.
    static const int64_t ORIGINS[] = { 0, (int64_t(1) << 32) + 5, -(int64_t(1) << 45), (int64_t(1) << 60) - (int64_t(1) << 21) };
    std::vector<WorldPos> points;
    std::vector<int> lods;
    ov::bytes encoded;
    for (int64_t origin : ORIGINS) {
        for (uint32_t i = 0; i < POINTS_PER_ORIGIN; i++) {
            const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(origin >> 32), 0x6E0Du });
            WorldPos p;
            for (int a = 0; a < 3; a++) {
                p.voxel[a] = origin + ExactNoise::random_offset(i, uint32_t(a), uint32_t(origin), 1 << 20);
            }
            p.frac = Float3{ float(h.x >> 12) * ExactNoise::exact_pow2(-20), float(h.y >> 12) * ExactNoise::exact_pow2(-20),
                float(h.z >> 12) * ExactNoise::exact_pow2(-20) };
            const int lod = PROBE_LODS[i % PROBE_LOD_COUNT];
            uint8_t slot[POINT_BYTES] = {};
            const float frac[3] = { p.frac.x, p.frac.y, p.frac.z };
            for (int a = 0; a < 3; a++) {
                ov::put_u32(slot + 4 * a, WorldCoord::lo32(p.voxel[a]));
                ov::put_s32(slot + 16 + 4 * a, WorldCoord::hi32(p.voxel[a]));
                ov::put_f32(slot + 32 + 4 * a, frac[a]);
            }
            ov::put_u32(slot + 12, uint32_t(lod));
            encoded.insert(encoded.end(), slot, slot + POINT_BYTES);
            points.push_back(p);
            lods.push_back(lod);
        }
    }
    const uint32_t count = uint32_t(points.size());
    ChunkParam param;
    param.voxel_size = float(CHECK_VOXEL_M);
    const ov::bytes param_bytes = GPUFrame::chunk_param_bytes(param);
    const Buffer in = p_device.create_buffer(BufferKind::STORAGE, encoded.size(), encoded.data());
    const Buffer values = p_device.create_buffer(BufferKind::STORAGE, uint64_t(count) * 16);
    const Buffer uniform = p_device.create_buffer(BufferKind::UNIFORM, param_bytes.size(), param_bytes.data());
    const BindSet bind = p_device.create_bind_set(pipeline, { { 0, BufferKind::STORAGE, in }, { 1, BufferKind::STORAGE, values }, { 2, BufferKind::UNIFORM, uniform } });
    const Push push{ count, 0 }; // lod_bias: the LOD is a runtime value, as in the meshers
    p_device.dispatch(pipeline, bind, &push, sizeof(push), (count + 63) / 64, 1, 1);
    p_device.sync();
    const ov::bytes gpu = p_device.read_buffer(values, 0, uint64_t(count) * 16);
    if (gpu.size() != size_t(count) * 16) {
        r_report.error = "noise readback failed";
    } else {
        for (uint32_t i = 0; i < count; i++) {
            const ExactNoise::Float4 want = ExactNoise::fbm(set, points[i], lods[i]);
            r_report.noise_checked++;
            if (std::memcmp(&want, gpu.data() + size_t(i) * 16, 16) != 0) {
                r_report.noise_mismatch++;
                if (r_report.first_mismatch.empty()) {
                    const uint8_t *g = gpu.data() + size_t(i) * 16;
                    const float cpu[4] = { want.x, want.y, want.z, want.w };
                    r_report.first_mismatch = ov::format("fbm at voxel %s LOD %d (GPU / CPU bits):", WorldCoord::to_string(points[i].voxel), lods[i]);
                    for (int c = 0; c < 4; c++) {
                        r_report.first_mismatch += ov::format(" %x / %x", int64_t(ov::get_u32(g + 4 * c)), int64_t(ExactNoise::float_bits(cpu[c])));
                    }
                }
            }
        }
    }
    for (Handle h : { Handle(bind), Handle(in), Handle(values), Handle(uniform), Handle(pipeline) }) {
        p_device.free(h);
    }
}

} // namespace

ov::str Report::summary() const {
    ov::str s = ov::format("GPU device check %s: plumbing %d/%d words, noise %d/%d points bit-exact", ok() ? "PASS" : "FAIL",
            plumbing_checked - plumbing_mismatch, plumbing_checked, noise_checked - noise_mismatch, noise_checked);
    if (!error.empty()) {
        s += "; error: " + error;
    }
    if (!first_mismatch.empty()) {
        s += "; first mismatch: " + first_mismatch;
    }
    return s;
}

Report run(ov::gpu::Device &p_device) {
    Report report;
    check_plumbing(p_device, report);
    if (report.error.empty()) {
        check_textures(p_device, report);
    }
    if (report.error.empty()) {
        check_noise(p_device, report);
    }
    return report;
}

} // namespace GPUDeviceCheck
