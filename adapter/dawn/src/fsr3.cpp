// AMD FidelityFX Super Resolution 3.1 upscaler on Dawn (WebGPU): the viewer's (DawnUpscaler).
//
// The pass order, the resource lists, the constants block and the dispatch sizes are
// taken from AMD's reference implementation (ffx_fsr3upscaler.cpp) as mirrored by the
// Rust wgpu port this file was translated from. Every pass is a compute dispatch on a
// single bind group (group 0), with the binding indices of the reference shaders.

#include "fsr3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iterator>

#include "fsr3_shaders.hpp"
#include "shader_module.hpp"

namespace ov::dawn {
namespace {

// ---------------------------------------------------------------------------------------
// Constants block layout: std140, exactly what the precompiled shaders read.
// ---------------------------------------------------------------------------------------

static_assert(sizeof(Fsr3FsrConstants) == 160, "Fsr3FsrConstants must match the shader cbuffer");
static_assert(offsetof(Fsr3Constants, fsr) == 0, "bad Fsr3Constants layout");
static_assert(offsetof(Fsr3Constants, generate_auto_reactive) == 160, "bad Fsr3Constants layout");
static_assert(offsetof(Fsr3Constants, rcas) == 176, "bad Fsr3Constants layout");
static_assert(offsetof(Fsr3Constants, generate_reactive) == 192, "bad Fsr3Constants layout");
static_assert(offsetof(Fsr3Constants, spd) == 208, "bad Fsr3Constants layout");
static_assert(sizeof(Fsr3Constants) == 240, "Fsr3Constants must match the shader cbuffer");
static_assert(offsetof(Fsr3FsrConstants, device_to_view_depth) == 48, "bad Fsr3FsrConstants layout");
static_assert(offsetof(Fsr3FsrConstants, jitter_offset) == 64, "bad Fsr3FsrConstants layout");
static_assert(offsetof(Fsr3FsrConstants, tan_half_fov) == 104, "bad Fsr3FsrConstants layout");
static_assert(offsetof(Fsr3FsrConstants, delta_time) == 112, "bad Fsr3FsrConstants layout");
static_assert(offsetof(Fsr3FsrConstants, frame_index) == 124, "bad Fsr3FsrConstants layout");
static_assert(offsetof(Fsr3FsrConstants, velocity_factor) == 128, "bad Fsr3FsrConstants layout");
static_assert(offsetof(Fsr3FsrConstants, min_disocclusion_accumulation) == 144,
        "bad Fsr3FsrConstants layout");

constexpr float kPi = 3.14159265358979323846f;

// ---------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------

uint32_t CeilDiv(uint32_t p_value, uint32_t p_divisor) {
    return (p_value + p_divisor - 1u) / p_divisor;
}

// Number of mip levels a WebGPU texture of this size can have (1 + floor(log2(max(w, h)))).
uint32_t MaxMips(uint32_t p_width, uint32_t p_height) {
    uint32_t size = std::max(p_width, p_height);
    uint32_t mips = 1;
    while (size > 1) {
        size >>= 1;
        ++mips;
    }
    return mips;
}

// Jitter phase count of the reference implementation (jitter.rs): 8 * (display / render)^2.
int32_t GetJitterPhaseCount(int32_t p_render_width, int32_t p_display_width) {
    constexpr float kBasePhaseCount = 8.0f;
    return int32_t(kBasePhaseCount * std::pow(float(p_display_width) / float(p_render_width), 2.0f));
}

// Lanczos2 windowed sinc, used to build the 128 entry lookup table (lanczos2.rs).
float Lanczos2(float p_value) {
    constexpr float kEpsilon = 1e-6f;
    if (std::abs(p_value) < kEpsilon) {
        return 1.0f;
    }
    const float v_pi = kPi * p_value;
    const float half_v_pi = 0.5f * v_pi;
    return (std::sin(v_pi) / v_pi) * (std::sin(half_v_pi) / half_v_pi);
}

void GenerateLanczos2Lut(int16_t (&r_lut)[128]) {
    for (int i = 0; i < 128; ++i) {
        const float x = 2.0f * float(i) / 127.0f;
        r_lut[i] = int16_t(std::lround(Lanczos2(x) * 32767.0f));
    }
}

// IEEE-754 binary32 -> binary16 bits (round to nearest even). Only the normal range is
// needed here: the RCAS sharpness constant is always in [0.25, 1].
uint16_t FloatToHalf(float p_value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &p_value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 127;
    const uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent > 15) {
        return uint16_t(sign | 0x7C00u); // overflow: infinity
    }
    if (exponent < -14) {
        return uint16_t(sign); // underflow: zero
    }

    // Round the 23 bit mantissa down to 10 bits (round half to even).
    const uint32_t rounded = mantissa + 0x0FFFu + ((mantissa >> 13) & 1u);
    const uint32_t half = (uint32_t(exponent + 15) << 10) + (rounded >> 13);
    return uint16_t(sign | half);
}

uint32_t PackHalf2x16(float p_x, float p_y) {
    return uint32_t(FloatToHalf(p_x)) | (uint32_t(FloatToHalf(p_y)) << 16);
}

// RCAS constants (rcas.rs). p_sharpness is already remapped: 0 = maximum sharpness,
// 2 = no sharpening.
Fsr3RcasConstants MakeRcasConstants(float p_sharpness) {
    const float sharpness_linear = std::exp2(-p_sharpness);

    Fsr3RcasConstants rcas = {};
    uint32_t bits = 0;
    std::memcpy(&bits, &sharpness_linear, sizeof(bits));
    rcas.rcas_config[0] = bits;
    rcas.rcas_config[1] = PackHalf2x16(sharpness_linear, sharpness_linear);
    rcas.rcas_config[2] = 0;
    rcas.rcas_config[3] = 0;
    return rcas;
}

// SPD constants (spd.rs): one 64x64 tile per workgroup, mip count from the largest
// dimension and capped at 12 (the maximum SPD supports).
Fsr3SpdConstants MakeSpdConstants(uint32_t p_width, uint32_t p_height) {
    Fsr3SpdConstants spd = {};

    const uint32_t tiles_x = CeilDiv(p_width, 64);
    const uint32_t tiles_y = CeilDiv(p_height, 64);
    spd.num_work_groups = tiles_x * tiles_y;
    spd.work_group_offset[0] = 0;
    spd.work_group_offset[1] = 0;
    spd.render_size[0] = p_width;
    spd.render_size[1] = p_height;

    const float resolution = float(std::max(p_width, p_height));
    spd.mips = uint32_t(std::min(12.0f, std::floor(std::log2(resolution))));
    return spd;
}

// Device depth -> view space depth matrix of the reference implementation, for the
// non inverted, non infinite depth convention.
void SetupDeviceToViewDepth(Fsr3FsrConstants &r_fsr, uint32_t p_render_w, uint32_t p_render_h, const Fsr3Inputs &p_inputs) {
    const float f_min = std::min(p_inputs.camera_near, p_inputs.camera_far);
    const float f_max = std::max(p_inputs.camera_near, p_inputs.camera_far);
    const float f_q = f_max / (f_min - f_max);
    const float d = -1.0f;

    r_fsr.device_to_view_depth[0] = d * f_q;
    r_fsr.device_to_view_depth[1] = f_q * f_min;

    const float aspect = float(p_render_w) / float(p_render_h);
    const float cot_half_fov_y = std::cos(0.5f * p_inputs.fov_y) / std::sin(0.5f * p_inputs.fov_y);
    const float a = cot_half_fov_y / aspect;
    const float b = cot_half_fov_y;
    r_fsr.device_to_view_depth[2] = 1.0f / a;
    r_fsr.device_to_view_depth[3] = 1.0f / b;
}

// Clear shaders: one storage texture each, because a storage texture binding is tied to a
// single texel format. Only used to reset the temporal history.
constexpr char kClearR8Wgsl[] = R"(
@group(0) @binding(0) var r_texture: texture_storage_2d<r8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) p_id: vec3<u32>) {
    textureStore(r_texture, vec2<i32>(i32(p_id.x), i32(p_id.y)), vec4<f32>(0.0));
}
)";

constexpr char kClearR16Wgsl[] = R"(
@group(0) @binding(0) var r_texture: texture_storage_2d<r16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) p_id: vec3<u32>) {
    textureStore(r_texture, vec2<i32>(i32(p_id.x), i32(p_id.y)), vec4<f32>(0.0));
}
)";

constexpr char kClearRg16Wgsl[] = R"(
@group(0) @binding(0) var r_texture: texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) p_id: vec3<u32>) {
    textureStore(r_texture, vec2<i32>(i32(p_id.x), i32(p_id.y)), vec4<f32>(0.0));
}
)";

constexpr char kClearRgba16Wgsl[] = R"(
@group(0) @binding(0) var r_texture: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) p_id: vec3<u32>) {
    textureStore(r_texture, vec2<i32>(i32(p_id.x), i32(p_id.y)), vec4<f32>(0.0));
}
)";

// Fills the reconstructed previous depth buffer with one 32 bit pattern (1.0f for the
// standard depth convention), as the reference dispatch does every frame.
constexpr char kFillBufferWgsl[] = R"(
@group(0) @binding(0) var<storage, read_write> r_buffer: array<u32>;
@group(0) @binding(1) var<uniform> r_fill_value: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) p_id: vec3<u32>) {
    let index = p_id.x;
    if (index < arrayLength(&r_buffer)) {
        r_buffer[index] = r_fill_value.x;
    }
}
)";

} // namespace

// ---------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------

Fsr3Upscaler::Fsr3Upscaler(wgpu::Device p_device, wgpu::Instance p_instance, uint32_t p_render_w,
        uint32_t p_render_h, uint32_t p_upscale_w, uint32_t p_upscale_h) : device(std::move(p_device)),
                                                                           instance(std::move(p_instance)),
                                                                           render_w(p_render_w),
                                                                           render_h(p_render_h),
                                                                           upscale_w(p_upscale_w),
                                                                           upscale_h(p_upscale_h) {
    queue = device.GetQueue();

    // Half resolution textures need at least one texel; a 1x1 render target has no
    // meaningful pyramid, so require the same minimum the reference port does.
    if (!device || !queue || render_w < 2 || render_h < 2 || upscale_w < 1 || upscale_h < 1) {
        return;
    }

    // The FSR3 shaders declare r8unorm / r16float / rg16float / rgba16float storage images
    // as read-write (the SPIR-V carries no access decorations), which WebGPU only allows
    // with the tier 2 texture formats. Tier 2 implies tier 1.
    if (!device.HasFeature(wgpu::FeatureName::TextureFormatsTier2)) {
        return;
    }

    // Per pass resource counts (the pyramid passes are the widest): the luma pyramid binds 8
    // storage textures, the accumulate pass 9 sampled textures and 2 storage buffers.
    wgpu::Limits limits = {};
    if (device.GetLimits(&limits) != wgpu::Status::Success ||
            limits.maxStorageTexturesPerShaderStage < 8 || limits.maxSampledTexturesPerShaderStage < 9 ||
            limits.maxStorageBuffersPerShaderStage < 2) {
        return;
    }

    half_render_w = std::max(1u, render_w / 2u);
    half_render_h = std::max(1u, render_h / 2u);
    spd_mip_count = MaxMips(half_render_w, half_render_h);

    // Catch validation errors of the one-time setup so Dispatch() can report the upscaler
    // as unavailable instead of leaving broken objects behind.
    device.PushErrorScope(wgpu::ErrorFilter::Validation);

    bool ok = CreateResources();
    ok = CreateClearPipelines() && ok;
    ok = CreatePipelines() && ok;

    bool had_error = false;
    device.PopErrorScope(wgpu::CallbackMode::AllowSpontaneous,
            [&had_error](wgpu::PopErrorScopeStatus p_status, wgpu::ErrorType p_type,
                    wgpu::StringView p_message) {
                if (p_status == wgpu::PopErrorScopeStatus::Success &&
                        p_type != wgpu::ErrorType::NoError) {
                    had_error = true;
                    std::fprintf(stderr, "FSR3: setup failed: %.*s\n",
                            int(p_message.length), p_message.data);
                }
            });
#ifndef __EMSCRIPTEN__
    device.Tick(); // a browser delivers the scope's result between tasks, after this returns
#endif

    // Every pass is required: a missing one would silently corrupt the upscaled image, so the
    // upscaler reports itself unavailable and the caller falls back to a plain upscale.
    valid = ok && !had_error;
}

Fsr3Upscaler::~Fsr3Upscaler() = default;

bool Fsr3Upscaler::CreateResources() {
    const wgpu::TextureUsage kRenderUsage = wgpu::TextureUsage::RenderAttachment |
            wgpu::TextureUsage::StorageBinding |
            wgpu::TextureUsage::TextureBinding;
    const wgpu::TextureUsage kStorageUsage =
            wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding;
    const wgpu::TextureUsage kSampledUsage = wgpu::TextureUsage::TextureBinding;

    auto create_texture = [this](const char *p_label, wgpu::TextureFormat p_format, uint32_t p_width,
                                  uint32_t p_height, uint32_t p_mips, wgpu::TextureUsage p_usage) {
        wgpu::TextureDescriptor desc = {};
        desc.label = p_label;
        desc.usage = p_usage;
        desc.dimension = wgpu::TextureDimension::e2D;
        desc.size = { p_width, p_height, 1 };
        desc.format = p_format;
        desc.mipLevelCount = p_mips;
        desc.sampleCount = 1;
        return device.CreateTexture(&desc);
    };

    auto create_buffer = [this](const char *p_label, uint64_t p_size, wgpu::BufferUsage p_usage) {
        wgpu::BufferDescriptor desc = {};
        desc.label = p_label;
        desc.usage = p_usage;
        desc.size = p_size;
        desc.mappedAtCreation = false;
        return device.CreateBuffer(&desc);
    };

    // --- double buffered temporal resources (even frame in slot 0, odd frame in slot 1) ---
    accumulation[0] = create_texture("FSR3 Accumulation (even)", wgpu::TextureFormat::R8Unorm, render_w,
            render_h, 1, kRenderUsage);
    accumulation[1] = create_texture("FSR3 Accumulation (odd)", wgpu::TextureFormat::R8Unorm, render_w,
            render_h, 1, kRenderUsage);
    luma[0] = create_texture("FSR3 Luma (even)", wgpu::TextureFormat::R16Float, render_w, render_h, 1,
            kRenderUsage);
    luma[1] = create_texture("FSR3 Luma (odd)", wgpu::TextureFormat::R16Float, render_w, render_h, 1,
            kRenderUsage);
    luma_history[0] = create_texture("FSR3 Luma History (even)", wgpu::TextureFormat::RGBA16Float,
            render_w, render_h, 1, kRenderUsage);
    luma_history[1] = create_texture("FSR3 Luma History (odd)", wgpu::TextureFormat::RGBA16Float,
            render_w, render_h, 1, kRenderUsage);
    internal_upscaled[0] = create_texture("FSR3 Internal Upscaled (even)",
            wgpu::TextureFormat::RGBA16Float, upscale_w, upscale_h, 1,
            kRenderUsage);
    internal_upscaled[1] = create_texture("FSR3 Internal Upscaled (odd)",
            wgpu::TextureFormat::RGBA16Float, upscale_w, upscale_h, 1,
            kRenderUsage);

    // --- single instance internal resources ---
    intermediate_fp16x1 = create_texture("FSR3 Intermediate FP16x1", wgpu::TextureFormat::R16Float,
            render_w, render_h, 1, kRenderUsage);
    shading_change = create_texture("FSR3 Shading Change", wgpu::TextureFormat::R8Unorm,
            half_render_w, half_render_h, 1, kRenderUsage);
    new_locks = create_texture("FSR3 New Locks", wgpu::TextureFormat::R8Unorm, upscale_w, upscale_h, 1,
            wgpu::TextureUsage::RenderAttachment |
                    wgpu::TextureUsage::StorageBinding);
    spd_mips = create_texture("FSR3 SPD Mips", wgpu::TextureFormat::RG16Float, half_render_w,
            half_render_h, spd_mip_count, kStorageUsage);
    farthest_depth_mip1 = create_texture("FSR3 Farthest Depth Mip1", wgpu::TextureFormat::R16Float,
            half_render_w, half_render_h, 1, kStorageUsage);
    dilated_reactive_masks = create_texture("FSR3 Dilated Reactive Masks",
            wgpu::TextureFormat::RGBA8Unorm, render_w, render_h, 1,
            kRenderUsage);
    dilated_depth = create_texture("FSR3 Dilated Depth", wgpu::TextureFormat::R32Float, render_w,
            render_h, 1, kRenderUsage);
    dilated_motion_vectors = create_texture("FSR3 Dilated Motion Vectors",
            wgpu::TextureFormat::RG16Float, render_w, render_h, 1,
            kRenderUsage);

    // --- fixed size helper resources ---
    // CopyDst is only needed for the initial upload (the reference port uses
    // create_texture_with_data, which adds it implicitly).
    lanczos2_lut = create_texture("FSR3 Lanczos2 LUT", wgpu::TextureFormat::R16Snorm, 128, 1, 1,
            kSampledUsage | wgpu::TextureUsage::CopyDst);
    default_reactivity_mask = create_texture("FSR3 Default Reactivity Mask",
            wgpu::TextureFormat::R8Unorm, 1, 1, 1, kSampledUsage);
    // Zero initialised; the shaders substitute 1.0 when the exposure is 0.
    default_exposure = create_texture("FSR3 Default Exposure", wgpu::TextureFormat::RG32Float, 1, 1,
            1, kSampledUsage);
    frame_info = create_texture("FSR3 Frame Info", wgpu::TextureFormat::RGBA32Float, 1, 1, 1,
            wgpu::TextureUsage::StorageBinding |
                    wgpu::TextureUsage::TextureBinding |
                    wgpu::TextureUsage::CopyDst);

    if (!accumulation[0] || !accumulation[1] || !luma[0] || !luma[1] || !luma_history[0] ||
            !luma_history[1] || !internal_upscaled[0] || !internal_upscaled[1] ||
            !intermediate_fp16x1 || !shading_change || !new_locks || !spd_mips ||
            !farthest_depth_mip1 || !dilated_reactive_masks || !dilated_depth ||
            !dilated_motion_vectors || !lanczos2_lut || !default_reactivity_mask ||
            !default_exposure || !frame_info) {
        return false;
    }

    // --- views: one default view per texture, one single mip view per SPD output level ---
    accumulation_view[0] = accumulation[0].CreateView();
    accumulation_view[1] = accumulation[1].CreateView();
    luma_view[0] = luma[0].CreateView();
    luma_view[1] = luma[1].CreateView();
    luma_history_view[0] = luma_history[0].CreateView();
    luma_history_view[1] = luma_history[1].CreateView();
    internal_upscaled_view[0] = internal_upscaled[0].CreateView();
    internal_upscaled_view[1] = internal_upscaled[1].CreateView();
    intermediate_fp16x1_view = intermediate_fp16x1.CreateView();
    shading_change_view = shading_change.CreateView();
    new_locks_view = new_locks.CreateView();
    spd_mips_view = spd_mips.CreateView();
    farthest_depth_mip1_view = farthest_depth_mip1.CreateView();
    dilated_reactive_masks_view = dilated_reactive_masks.CreateView();
    dilated_depth_view = dilated_depth.CreateView();
    dilated_motion_vectors_view = dilated_motion_vectors.CreateView();
    lanczos2_lut_view = lanczos2_lut.CreateView();
    default_reactivity_mask_view = default_reactivity_mask.CreateView();
    default_exposure_view = default_exposure.CreateView();
    frame_info_view = frame_info.CreateView();

    // The SPD passes bind mip levels 0..5 individually. Tiny render sizes do not have six
    // levels; the missing bindings get their own 1x1 dummy (sharing a view between two
    // writable bindings of the same bind group is not allowed) and are never written,
    // because the shader stops before those levels when its mip count is below 6.
    const uint32_t spd_views = std::min(6u, spd_mip_count);
    for (uint32_t mip = 0; mip < spd_views; ++mip) {
        wgpu::TextureViewDescriptor mip_desc = {};
        mip_desc.label = "FSR3 SPD Mip";
        mip_desc.baseMipLevel = mip;
        mip_desc.mipLevelCount = 1;
        spd_mip_views[mip] = spd_mips.CreateView(&mip_desc);
    }
    for (uint32_t mip = spd_views; mip < 6; ++mip) {
        spd_dummy_mips[mip] = create_texture("FSR3 SPD Dummy Mip", wgpu::TextureFormat::RG16Float,
                1, 1, 1, kStorageUsage);
        if (!spd_dummy_mips[mip]) {
            return false;
        }
        spd_mip_views[mip] = spd_dummy_mips[mip].CreateView();
    }

    // --- buffers ---
    constants_buffer = create_buffer("FSR3 Constants", sizeof(Fsr3Constants),
            wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    spd_atomic_counter = create_buffer("FSR3 SPD Atomic Counter", 4,
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    reconstructed_previous_depth =
            create_buffer("FSR3 Reconstructed Previous Depth", uint64_t(render_w) * render_h * 4u,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
    fill_value_buffer = create_buffer("FSR3 Depth Fill Value", 16,
            wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    frame_info_staging = create_buffer("FSR3 Frame Info Reset", 256,
            wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);
    if (!constants_buffer || !spd_atomic_counter || !reconstructed_previous_depth ||
            !fill_value_buffer || !frame_info_staging) {
        return false;
    }

    // One time contents of the two small helper buffers:
    //  * the depth fill value is 1.0f (the standard, non inverted, depth convention),
    //  * the frame info reset is (log luma, exposure, scene average luma, unused) = (-1, 1, 0, 0).
    const float depth_fill = 1.0f;
    uint32_t depth_fill_bits = 0;
    std::memcpy(&depth_fill_bits, &depth_fill, sizeof(depth_fill_bits));
    queue.WriteBuffer(fill_value_buffer, 0, &depth_fill_bits, sizeof(depth_fill_bits));

    const float frame_info_reset[4] = { -1.0f, 1.0f, 0.0f, 0.0f };
    queue.WriteBuffer(frame_info_staging, 0, frame_info_reset, sizeof(frame_info_reset));

    // --- samplers: the two samplers every FSR3 pass has at the end of its bind group ---
    wgpu::SamplerDescriptor point_desc = {};
    point_desc.label = "FSR3 Sampler Point Clamp";
    point_desc.addressModeU = wgpu::AddressMode::ClampToEdge;
    point_desc.addressModeV = wgpu::AddressMode::ClampToEdge;
    point_desc.addressModeW = wgpu::AddressMode::ClampToEdge;
    point_desc.magFilter = wgpu::FilterMode::Nearest;
    point_desc.minFilter = wgpu::FilterMode::Nearest;
    point_desc.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
    sampler_point = device.CreateSampler(&point_desc);

    wgpu::SamplerDescriptor linear_desc = {};
    linear_desc.label = "FSR3 Sampler Linear Clamp";
    linear_desc.addressModeU = wgpu::AddressMode::ClampToEdge;
    linear_desc.addressModeV = wgpu::AddressMode::ClampToEdge;
    linear_desc.addressModeW = wgpu::AddressMode::ClampToEdge;
    linear_desc.magFilter = wgpu::FilterMode::Linear;
    linear_desc.minFilter = wgpu::FilterMode::Linear;
    linear_desc.mipmapFilter = wgpu::MipmapFilterMode::Linear;
    sampler_linear = device.CreateSampler(&linear_desc);
    if (!sampler_point || !sampler_linear) {
        return false;
    }

    // --- lanczos2 lookup table (128 x R16Snorm) ---
    int16_t lut[128] = {};
    GenerateLanczos2Lut(lut);

    wgpu::TexelCopyTextureInfo dst = {};
    dst.texture = lanczos2_lut;
    dst.mipLevel = 0;
    dst.origin = { 0, 0, 0 };
    dst.aspect = wgpu::TextureAspect::All;

    wgpu::TexelCopyBufferLayout layout = {};
    layout.offset = 0;
    layout.bytesPerRow = sizeof(lut);
    layout.rowsPerImage = 1;

    wgpu::Extent3D extent = { 128, 1, 1 };
    queue.WriteTexture(&dst, lut, sizeof(lut), &layout, &extent);

    // --- constants defaults (identical to the reference view setup) ---
    Fsr3FsrConstants &fsr = constants.fsr;
    fsr.max_render_size[0] = render_w;
    fsr.max_render_size[1] = render_h;
    fsr.max_upscale_size[0] = upscale_w;
    fsr.max_upscale_size[1] = upscale_h;
    fsr.velocity_factor = 1.0f;
    fsr.reactiveness_scale = 1.0f;
    fsr.shading_change_scale = 1.0f;
    fsr.accumulation_added_per_frame = 1.0f / 3.0f;
    fsr.min_disocclusion_accumulation = -1.0f / 3.0f;

    return true;
}

bool Fsr3Upscaler::CreateClearPipelines() {
    auto create_clear = [this](const char *p_label, const char *p_wgsl, wgpu::TextureFormat p_format,
                                ClearPass &r_clear) {
        wgpu::ShaderSourceWGSL source = {};
        source.code = p_wgsl;

        wgpu::ShaderModuleDescriptor module_desc = {};
        module_desc.nextInChain = &source;
        module_desc.label = p_label;
        wgpu::ShaderModule module = device.CreateShaderModule(&module_desc);
        if (!module) {
            return false;
        }

        wgpu::BindGroupLayoutEntry entry = {};
        entry.binding = 0;
        entry.visibility = wgpu::ShaderStage::Compute;
        entry.storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
        entry.storageTexture.format = p_format;
        entry.storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;

        wgpu::BindGroupLayoutDescriptor bgl_desc = {};
        bgl_desc.label = p_label;
        bgl_desc.entryCount = 1;
        bgl_desc.entries = &entry;
        r_clear.layout = device.CreateBindGroupLayout(&bgl_desc);
        if (!r_clear.layout) {
            return false;
        }

        wgpu::BindGroupLayout layouts[1] = { r_clear.layout };
        wgpu::PipelineLayoutDescriptor pl_desc = {};
        pl_desc.label = p_label;
        pl_desc.bindGroupLayoutCount = 1;
        pl_desc.bindGroupLayouts = layouts;
        wgpu::PipelineLayout pipeline_layout = device.CreatePipelineLayout(&pl_desc);
        if (!pipeline_layout) {
            return false;
        }

        wgpu::ComputePipelineDescriptor pipeline_desc = {};
        pipeline_desc.label = p_label;
        pipeline_desc.layout = pipeline_layout;
        pipeline_desc.compute.module = module;
        pipeline_desc.compute.entryPoint = "main";
        r_clear.pipeline = device.CreateComputePipeline(&pipeline_desc);
        return bool(r_clear.pipeline);
    };

    bool ok = create_clear("FSR3 Clear R8", kClearR8Wgsl, wgpu::TextureFormat::R8Unorm, clear_r8);
    ok = create_clear("FSR3 Clear R16", kClearR16Wgsl, wgpu::TextureFormat::R16Float, clear_r16) && ok;
    ok = create_clear("FSR3 Clear RG16", kClearRg16Wgsl, wgpu::TextureFormat::RG16Float, clear_rg16) && ok;
    ok = create_clear("FSR3 Clear RGBA16", kClearRgba16Wgsl, wgpu::TextureFormat::RGBA16Float,
                 clear_rgba16) &&
            ok;

    // Buffer fill pipeline (two bindings: the storage buffer and the value).
    wgpu::ShaderSourceWGSL source = {};
    source.code = kFillBufferWgsl;

    wgpu::ShaderModuleDescriptor module_desc = {};
    module_desc.nextInChain = &source;
    module_desc.label = "FSR3 Fill Buffer";
    wgpu::ShaderModule module = device.CreateShaderModule(&module_desc);
    if (!module) {
        return false;
    }

    wgpu::BindGroupLayoutEntry entries[2] = {};
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Compute;
    entries[0].buffer.type = wgpu::BufferBindingType::Storage;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = 0;
    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Compute;
    entries[1].buffer.type = wgpu::BufferBindingType::Uniform;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 16;

    wgpu::BindGroupLayoutDescriptor bgl_desc = {};
    bgl_desc.label = "FSR3 Fill Buffer";
    bgl_desc.entryCount = 2;
    bgl_desc.entries = entries;
    fill_buffer.layout = device.CreateBindGroupLayout(&bgl_desc);
    if (!fill_buffer.layout) {
        return false;
    }

    wgpu::BindGroupLayout layouts[1] = { fill_buffer.layout };
    wgpu::PipelineLayoutDescriptor pl_desc = {};
    pl_desc.label = "FSR3 Fill Buffer";
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = layouts;
    wgpu::PipelineLayout pipeline_layout = device.CreatePipelineLayout(&pl_desc);
    if (!pipeline_layout) {
        return false;
    }

    wgpu::ComputePipelineDescriptor pipeline_desc = {};
    pipeline_desc.label = "FSR3 Fill Buffer";
    pipeline_desc.layout = pipeline_layout;
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = "main";
    fill_buffer.pipeline = device.CreateComputePipeline(&pipeline_desc);

    return ok && bool(fill_buffer.pipeline);
}

bool Fsr3Upscaler::CreatePipelines() {
    using BT = BindType;
    using R = Resource;

    // Bindings of every pass, in the exact order of the reference pass resource lists
    // (binding index == position in the list). Samplers are appended to every pass.
    static const Binding kPrepareInputs[] = {
        { .binding = 0, .type = BT::SampledTexture, .resource = R::InputMotionVectors, .filterable = true },
        { .binding = 1, .type = BT::SampledTexture, .resource = R::InputDepth, .filterable = false },
        { .binding = 2, .type = BT::SampledTexture, .resource = R::InputColor, .filterable = true },
        { .binding = 3, .type = BT::StorageTexture, .resource = R::OutputDilatedMotionVectors, .format = wgpu::TextureFormat::RG16Float, .write_only = true },
        { .binding = 4, .type = BT::StorageTexture, .resource = R::OutputDilatedDepth, .format = wgpu::TextureFormat::R32Float, .write_only = true },
        { .binding = 5, .type = BT::StorageBuffer, .resource = R::OutputReconstructedPreviousDepth },
        { .binding = 6, .type = BT::StorageTexture, .resource = R::FarthestDepth, .format = wgpu::TextureFormat::R16Float },
        { .binding = 7, .type = BT::StorageTexture, .resource = R::Luma, .format = wgpu::TextureFormat::R16Float },
        { .binding = 8, .type = BT::UniformBuffer, .resource = R::Constants },
        { .binding = 9, .type = BT::SamplerPoint, .resource = R::SamplerPointClamp },
        { .binding = 10, .type = BT::SamplerLinear, .resource = R::SamplerLinearClamp },
    };

    static const Binding kLumaPyramid[] = {
        { .binding = 0, .type = BT::SampledTexture, .resource = R::Luma, .filterable = true },
        { .binding = 1, .type = BT::SampledTexture, .resource = R::FarthestDepth, .filterable = true },
        { .binding = 2, .type = BT::StorageBuffer, .resource = R::SpdAtomicCount },
        { .binding = 3, .type = BT::StorageTexture, .resource = R::FrameInfo, .format = wgpu::TextureFormat::RGBA32Float },
        { .binding = 4, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 0 },
        { .binding = 5, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 1 },
        { .binding = 6, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 2 },
        { .binding = 7, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 3 },
        { .binding = 8, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 4 },
        { .binding = 9, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 5 },
        { .binding = 10, .type = BT::StorageTexture, .resource = R::FarthestDepthMip1, .format = wgpu::TextureFormat::R16Float },
        { .binding = 11, .type = BT::UniformBuffer, .resource = R::Constants },
        { .binding = 12, .type = BT::SamplerPoint, .resource = R::SamplerPointClamp },
        { .binding = 13, .type = BT::SamplerLinear, .resource = R::SamplerLinearClamp },
    };

    static const Binding kShadingChangePyramid[] = {
        { .binding = 0, .type = BT::SampledTexture, .resource = R::Luma, .filterable = true },
        { .binding = 1, .type = BT::SampledTexture, .resource = R::PreviousLuma, .filterable = true },
        { .binding = 2, .type = BT::SampledTexture, .resource = R::OutputDilatedMotionVectors, .filterable = true },
        { .binding = 3, .type = BT::SampledTexture, .resource = R::InputExposure, .filterable = false },
        { .binding = 4, .type = BT::StorageBuffer, .resource = R::SpdAtomicCount },
        { .binding = 5, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 0 },
        { .binding = 6, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 1 },
        { .binding = 7, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 2 },
        { .binding = 8, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 3 },
        { .binding = 9, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 4 },
        { .binding = 10, .type = BT::StorageTexture, .resource = R::SpdMips, .format = wgpu::TextureFormat::RG16Float, .write_only = true, .mip = 5 },
        { .binding = 11, .type = BT::UniformBuffer, .resource = R::Constants },
        { .binding = 12, .type = BT::SamplerPoint, .resource = R::SamplerPointClamp },
        { .binding = 13, .type = BT::SamplerLinear, .resource = R::SamplerLinearClamp },
    };

    static const Binding kShadingChange[] = {
        { .binding = 0, .type = BT::SampledTexture, .resource = R::SpdMips, .filterable = true },
        { .binding = 1, .type = BT::StorageTexture, .resource = R::ShadingChange, .format = wgpu::TextureFormat::R8Unorm },
        { .binding = 2, .type = BT::UniformBuffer, .resource = R::Constants },
        { .binding = 3, .type = BT::SamplerPoint, .resource = R::SamplerPointClamp },
        { .binding = 4, .type = BT::SamplerLinear, .resource = R::SamplerLinearClamp },
    };

    static const Binding kPrepareReactivity[] = {
        { .binding = 0, .type = BT::StorageBuffer, .resource = R::OutputReconstructedPreviousDepth },
        { .binding = 1, .type = BT::SampledTexture, .resource = R::OutputDilatedMotionVectors, .filterable = true },
        { .binding = 2, .type = BT::SampledTexture, .resource = R::OutputDilatedDepth, .filterable = false },
        { .binding = 3, .type = BT::SampledTexture, .resource = R::InputReactiveMask, .filterable = true },
        { .binding = 4, .type = BT::SampledTexture, .resource = R::InputTransparencyAndComposition, .filterable = true },
        { .binding = 5, .type = BT::SampledTexture, .resource = R::AccumulationPrevious, .filterable = true },
        { .binding = 6, .type = BT::SampledTexture, .resource = R::ShadingChange, .filterable = true },
        { .binding = 7, .type = BT::SampledTexture, .resource = R::Luma, .filterable = true },
        { .binding = 8, .type = BT::SampledTexture, .resource = R::InputExposure, .filterable = false },
        { .binding = 9, .type = BT::StorageTexture, .resource = R::DilatedReactiveMasks, .format = wgpu::TextureFormat::RGBA8Unorm, .write_only = true },
        { .binding = 10, .type = BT::StorageTexture, .resource = R::NewLocks, .format = wgpu::TextureFormat::R8Unorm },
        { .binding = 11, .type = BT::StorageTexture, .resource = R::AccumulationCurrent, .format = wgpu::TextureFormat::R8Unorm },
        { .binding = 12, .type = BT::UniformBuffer, .resource = R::Constants },
        { .binding = 13, .type = BT::SamplerPoint, .resource = R::SamplerPointClamp },
        { .binding = 14, .type = BT::SamplerLinear, .resource = R::SamplerLinearClamp },
    };

    static const Binding kLumaInstability[] = {
        { .binding = 0, .type = BT::SampledTexture, .resource = R::InputExposure, .filterable = false },
        { .binding = 1, .type = BT::SampledTexture, .resource = R::DilatedReactiveMasks, .filterable = true },
        { .binding = 2, .type = BT::SampledTexture, .resource = R::OutputDilatedMotionVectors, .filterable = true },
        { .binding = 3, .type = BT::SampledTexture, .resource = R::FrameInfo, .filterable = false },
        { .binding = 4, .type = BT::SampledTexture, .resource = R::LumaHistoryPrevious, .filterable = true },
        { .binding = 5, .type = BT::SampledTexture, .resource = R::FarthestDepthMip1, .filterable = true },
        { .binding = 6, .type = BT::SampledTexture, .resource = R::Luma, .filterable = true },
        { .binding = 7, .type = BT::StorageTexture, .resource = R::LumaHistoryCurrent, .format = wgpu::TextureFormat::RGBA16Float },
        { .binding = 8, .type = BT::StorageTexture, .resource = R::LumaInstability, .format = wgpu::TextureFormat::R16Float },
        { .binding = 9, .type = BT::UniformBuffer, .resource = R::Constants },
        { .binding = 10, .type = BT::SamplerPoint, .resource = R::SamplerPointClamp },
        { .binding = 11, .type = BT::SamplerLinear, .resource = R::SamplerLinearClamp },
    };

    // The display-resolution motion vector variant: the accumulate pass reads the input motion
    // vectors at the output pixel, not the dilated ones.
    static const Binding kAccumulate[] = {
        { .binding = 0, .type = BT::SampledTexture, .resource = R::InputExposure, .filterable = false },
        { .binding = 1, .type = BT::SampledTexture, .resource = R::DilatedReactiveMasks, .filterable = true },
        { .binding = 2, .type = BT::SampledTexture, .resource = R::InputMotionVectors, .filterable = true },
        { .binding = 3, .type = BT::SampledTexture, .resource = R::InternalUpscaledPrevious, .filterable = true },
        { .binding = 4, .type = BT::SampledTexture, .resource = R::Lanczos2Lut, .filterable = false },
        { .binding = 5, .type = BT::SampledTexture, .resource = R::FarthestDepthMip1, .filterable = true },
        { .binding = 6, .type = BT::SampledTexture, .resource = R::Luma, .filterable = true },
        { .binding = 7, .type = BT::SampledTexture, .resource = R::LumaInstability, .filterable = true },
        { .binding = 8, .type = BT::SampledTexture, .resource = R::InputColor, .filterable = true },
        { .binding = 9, .type = BT::StorageTexture, .resource = R::InternalUpscaledCurrent, .format = wgpu::TextureFormat::RGBA16Float, .write_only = true },
        { .binding = 10, .type = BT::StorageTexture, .resource = R::OutputColor, .format = wgpu::TextureFormat::RGBA16Float, .write_only = true },
        { .binding = 11, .type = BT::StorageTexture, .resource = R::NewLocks, .format = wgpu::TextureFormat::R8Unorm },
        { .binding = 12, .type = BT::UniformBuffer, .resource = R::Constants },
        { .binding = 13, .type = BT::SamplerPoint, .resource = R::SamplerPointClamp },
        { .binding = 14, .type = BT::SamplerLinear, .resource = R::SamplerLinearClamp },
    };

    static const Binding kRcas[] = {
        { .binding = 0, .type = BT::SampledTexture, .resource = R::InputExposure, .filterable = false },
        { .binding = 1, .type = BT::SampledTexture, .resource = R::InternalUpscaledCurrent, .filterable = true },
        { .binding = 2, .type = BT::StorageTexture, .resource = R::OutputColor, .format = wgpu::TextureFormat::RGBA16Float, .write_only = true },
        { .binding = 3, .type = BT::UniformBuffer, .resource = R::Constants },
        { .binding = 4, .type = BT::SamplerPoint, .resource = R::SamplerPointClamp },
        { .binding = 5, .type = BT::SamplerLinear, .resource = R::SamplerLinearClamp },
    };

    auto count = [](const auto &p_array) { return uint32_t(std::size(p_array)); };

    bool ok = true;
    ok = CreatePass(fsr3_shaders::kPrepareInputs, uint32_t(std::size(fsr3_shaders::kPrepareInputs)),
                 kPrepareInputs, count(kPrepareInputs), "FSR3 Prepare Inputs",
                 pass_prepare_inputs) &&
            ok;
    // The two SPD pyramid passes are the upstream shaders with their unused "mips > 6" tail
    // removed (see fsr3_shaders.hpp): that tail contains a cross workgroup spin wait that WGSL
    // cannot express, while FSR3 only ever samples mip levels 0..5. Both passes need the device
    // to raise maxStorageTexturesPerShaderStage (see the class comment).
    ok = CreatePass(fsr3_shaders::kLumaPyramid, uint32_t(std::size(fsr3_shaders::kLumaPyramid)),
                 kLumaPyramid, count(kLumaPyramid), "FSR3 Luma Pyramid", pass_luma_pyramid) &&
            ok;
    ok = CreatePass(fsr3_shaders::kShadingChangePyramid,
                 uint32_t(std::size(fsr3_shaders::kShadingChangePyramid)), kShadingChangePyramid,
                 count(kShadingChangePyramid), "FSR3 Shading Change Pyramid",
                 pass_shading_change_pyramid) &&
            ok;
    ok = CreatePass(fsr3_shaders::kShadingChange, uint32_t(std::size(fsr3_shaders::kShadingChange)),
                 kShadingChange, count(kShadingChange), "FSR3 Shading Change",
                 pass_shading_change) &&
            ok;
    ok = CreatePass(fsr3_shaders::kPrepareReactivity,
                 uint32_t(std::size(fsr3_shaders::kPrepareReactivity)), kPrepareReactivity,
                 count(kPrepareReactivity), "FSR3 Prepare Reactivity",
                 pass_prepare_reactivity) &&
            ok;
    ok = CreatePass(fsr3_shaders::kLumaInstability,
                 uint32_t(std::size(fsr3_shaders::kLumaInstability)), kLumaInstability,
                 count(kLumaInstability), "FSR3 Luma Instability", pass_luma_instability) &&
            ok;
    ok = CreatePass(fsr3_shaders::kAccumulate, uint32_t(std::size(fsr3_shaders::kAccumulate)),
                 kAccumulate, count(kAccumulate), "FSR3 Accumulate", pass_accumulate) &&
            ok;
    ok = CreatePass(fsr3_shaders::kRcas, uint32_t(std::size(fsr3_shaders::kRcas)), kRcas,
                 count(kRcas), "FSR3 RCAS", pass_rcas) &&
            ok;
    return ok;
}

bool Fsr3Upscaler::CreatePass(const uint32_t *p_spirv, uint32_t p_spirv_words,
        const Binding *p_bindings, uint32_t p_binding_count,
        const char *p_label, Pass &r_pass) {
    wgpu::ShaderModule module = ov::dawn::create_spirv_module(device, p_spirv, p_spirv_words, p_label);
    if (!module) {
        return false;
    }

    std::vector<wgpu::BindGroupLayoutEntry> entries(p_binding_count);
    for (uint32_t i = 0; i < p_binding_count; ++i) {
        const Binding &binding = p_bindings[i];
        wgpu::BindGroupLayoutEntry &entry = entries[i];
        entry.binding = binding.binding;
        entry.visibility = wgpu::ShaderStage::Compute;
        switch (binding.type) {
            case BindType::SampledTexture:
                // The SPIR-V declares plain texture2D, so Dawn accepts either sample type; the
                // filtering sampler pairings of the shaders require the filterable one.
                entry.texture.sampleType = binding.filterable ? wgpu::TextureSampleType::Float
                                                              : wgpu::TextureSampleType::UnfilterableFloat;
                entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
                entry.texture.multisampled = false;
                break;
            case BindType::StorageTexture:
                entry.storageTexture.access = binding.write_only ? wgpu::StorageTextureAccess::WriteOnly
                                                                 : wgpu::StorageTextureAccess::ReadWrite;
                entry.storageTexture.format = binding.format;
                entry.storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;
                break;
            case BindType::UniformBuffer:
                entry.buffer.type = wgpu::BufferBindingType::Uniform;
                entry.buffer.hasDynamicOffset = false;
                entry.buffer.minBindingSize = sizeof(Fsr3Constants);
                break;
            case BindType::StorageBuffer:
                entry.buffer.type = wgpu::BufferBindingType::Storage;
                entry.buffer.hasDynamicOffset = false;
                entry.buffer.minBindingSize = 0;
                break;
            case BindType::SamplerPoint:
                entry.sampler.type = wgpu::SamplerBindingType::NonFiltering;
                break;
            case BindType::SamplerLinear:
                entry.sampler.type = wgpu::SamplerBindingType::Filtering;
                break;
        }
    }

    wgpu::BindGroupLayoutDescriptor bgl_desc = {};
    bgl_desc.label = p_label;
    bgl_desc.entryCount = entries.size();
    bgl_desc.entries = entries.data();
    r_pass.layout = device.CreateBindGroupLayout(&bgl_desc);
    if (!r_pass.layout) {
        return false;
    }

    wgpu::BindGroupLayout layouts[1] = { r_pass.layout };
    wgpu::PipelineLayoutDescriptor pl_desc = {};
    pl_desc.label = p_label;
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = layouts;
    wgpu::PipelineLayout pipeline_layout = device.CreatePipelineLayout(&pl_desc);
    if (!pipeline_layout) {
        return false;
    }

    wgpu::ComputePipelineDescriptor pipeline_desc = {};
    pipeline_desc.label = p_label;
    pipeline_desc.layout = pipeline_layout;
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = "main";
    r_pass.pipeline = device.CreateComputePipeline(&pipeline_desc);
    if (!r_pass.pipeline) {
        return false;
    }

    r_pass.bindings = p_bindings;
    r_pass.binding_count = p_binding_count;
    return true;
}

// ---------------------------------------------------------------------------------------
// Resource resolution
// ---------------------------------------------------------------------------------------

wgpu::TextureView Fsr3Upscaler::ResolveView(const Binding &p_binding,
        const Fsr3Inputs &p_inputs) const {
    switch (p_binding.resource) {
        case Resource::InputColor:
            return p_inputs.color;
        case Resource::InputDepth:
            return p_inputs.depth;
        case Resource::InputMotionVectors:
            return p_inputs.motion_vectors;
        case Resource::InputExposure:
            return default_exposure_view;
        case Resource::InputReactiveMask:
        case Resource::InputTransparencyAndComposition:
            // No reactive mask / transparency and composition mask is provided, so both bind
            // the zero initialised 1x1 mask (the reference port does the same).
            return default_reactivity_mask_view;
        case Resource::OutputColor:
            return p_inputs.output;
        case Resource::OutputDilatedDepth:
            return dilated_depth_view;
        case Resource::OutputDilatedMotionVectors:
            return dilated_motion_vectors_view;

        case Resource::AccumulationCurrent:
            return accumulation_view[frame_even ? 0 : 1];
        case Resource::AccumulationPrevious:
            return accumulation_view[frame_even ? 1 : 0];
        case Resource::Luma:
            return luma_view[frame_even ? 0 : 1];
        case Resource::PreviousLuma:
            return luma_view[frame_even ? 1 : 0];
        case Resource::LumaHistoryCurrent:
            return luma_history_view[frame_even ? 0 : 1];
        case Resource::LumaHistoryPrevious:
            return luma_history_view[frame_even ? 1 : 0];
        case Resource::InternalUpscaledCurrent:
            return internal_upscaled_view[frame_even ? 0 : 1];
        case Resource::InternalUpscaledPrevious:
            return internal_upscaled_view[frame_even ? 1 : 0];

        case Resource::LumaInstability:
        case Resource::FarthestDepth:
            // Both names alias the single R16Float intermediate of the reference port.
            return intermediate_fp16x1_view;
        case Resource::ShadingChange:
            return shading_change_view;
        case Resource::NewLocks:
            return new_locks_view;
        case Resource::SpdMips:
            return p_binding.mip >= 0 ? spd_mip_views[p_binding.mip] : spd_mips_view;
        case Resource::FarthestDepthMip1:
            return farthest_depth_mip1_view;
        case Resource::DilatedReactiveMasks:
            return dilated_reactive_masks_view;
        case Resource::Lanczos2Lut:
            return lanczos2_lut_view;
        case Resource::DefaultReactivityMask:
            return default_reactivity_mask_view;
        case Resource::DefaultExposure:
            return default_exposure_view;
        case Resource::FrameInfo:
            return frame_info_view;

        case Resource::OutputReconstructedPreviousDepth:
        case Resource::Constants:
        case Resource::SpdAtomicCount:
        case Resource::SamplerPointClamp:
        case Resource::SamplerLinearClamp:
            break;
    }
    return wgpu::TextureView();
}

wgpu::Buffer Fsr3Upscaler::ResolveBuffer(Resource p_resource) const {
    switch (p_resource) {
        case Resource::Constants:
            return constants_buffer;
        case Resource::OutputReconstructedPreviousDepth:
            return reconstructed_previous_depth;
        case Resource::SpdAtomicCount:
            return spd_atomic_counter;
        default:
            break;
    }
    return wgpu::Buffer();
}

wgpu::BindGroupEntry Fsr3Upscaler::MakeEntry(const Binding &p_binding,
        const Fsr3Inputs &p_inputs) const {
    wgpu::BindGroupEntry entry = {};
    entry.binding = p_binding.binding;
    switch (p_binding.type) {
        case BindType::SampledTexture:
        case BindType::StorageTexture:
            entry.textureView = ResolveView(p_binding, p_inputs);
            break;
        case BindType::UniformBuffer:
        case BindType::StorageBuffer:
            entry.buffer = ResolveBuffer(p_binding.resource);
            break;
        case BindType::SamplerPoint:
            entry.sampler = sampler_point;
            break;
        case BindType::SamplerLinear:
            entry.sampler = sampler_linear;
            break;
    }
    return entry;
}

// ---------------------------------------------------------------------------------------
// Per-frame dispatch
// ---------------------------------------------------------------------------------------

bool Fsr3Upscaler::Dispatch(wgpu::CommandEncoder &p_encoder, const Fsr3Inputs &p_inputs) {
    if (!valid || !p_encoder) {
        return false;
    }
    if (!p_inputs.color || !p_inputs.depth || !p_inputs.motion_vectors || !p_inputs.output) {
        return false;
    }

    // The first frame always behaves like a history reset (the internal textures have never
    // been written yet).
    const bool reset = p_inputs.reset || first_frame;
    first_frame = false;
    last_output = p_inputs.output;

    // --- constants of this frame ---
    Fsr3FsrConstants &fsr = constants.fsr;
    fsr.previous_frame_jitter_offset[0] = previous_jitter[0];
    fsr.previous_frame_jitter_offset[1] = previous_jitter[1];
    fsr.previous_frame_render_size[0] = render_w;
    fsr.previous_frame_render_size[1] = render_h;
    fsr.previous_frame_upscale_size[0] = upscale_w;
    fsr.previous_frame_upscale_size[1] = upscale_h;

    fsr.render_size[0] = render_w;
    fsr.render_size[1] = render_h;
    fsr.upscale_size[0] = upscale_w;
    fsr.upscale_size[1] = upscale_h;
    fsr.downscale_factor[0] = float(render_w) / float(upscale_w);
    fsr.downscale_factor[1] = float(render_h) / float(upscale_h);

    fsr.jitter_offset[0] = p_inputs.jitter_x;
    fsr.jitter_offset[1] = p_inputs.jitter_y;

    // Horizontal FOV from the vertical one (only used by the shading change heuristics).
    const float aspect_ratio = float(render_w) / float(render_h);
    const float camera_angle_horizontal = std::atan(std::tan(p_inputs.fov_y * 0.5f) * aspect_ratio) * 2.0f;
    fsr.tan_half_fov = std::tan(camera_angle_horizontal * 0.5f);
    fsr.view_space_to_meters_factor = 1.0f;
    SetupDeviceToViewDepth(fsr, render_w, render_h, p_inputs);

    // Motion vectors are output sized, in output pixels (the display-resolution variant): the shader scale is 1 / output.
    fsr.motion_vector_scale[0] = 1.0f / float(upscale_w);
    fsr.motion_vector_scale[1] = 1.0f / float(upscale_h);
    // Jitter cancellation is disabled (the shaders are the non jittered motion vector variant).
    fsr.motion_vector_jitter_cancellation[0] = 0.0f;
    fsr.motion_vector_jitter_cancellation[1] = 0.0f;

    // Pre exposure is fixed to 1.0 (Fsr3Inputs has no exposure control).
    fsr.delta_pre_exposure = 1.0f;

    const int32_t jitter_phase_count = GetJitterPhaseCount(int32_t(render_w), int32_t(upscale_w));
    if (reset || fsr.jitter_phase_count == 0.0f) {
        fsr.jitter_phase_count = float(jitter_phase_count);
    } else {
        const int32_t delta = jitter_phase_count - int32_t(fsr.jitter_phase_count);
        if (delta > 0) {
            fsr.jitter_phase_count += 1.0f;
        } else if (delta < 0) {
            fsr.jitter_phase_count -= 1.0f;
        }
    }

    fsr.delta_time = std::min(1.0f, std::max(0.0f, p_inputs.frame_time_ms * 0.001f));
    fsr.frame_index = reset ? 0.0f : fsr.frame_index + 1.0f;

    constants.spd = MakeSpdConstants(render_w, render_h);
    constants.rcas = MakeRcasConstants(-2.0f * p_inputs.sharpness + 2.0f);

    // Recorded into the encoder (not the queue) so that several frames can be recorded
    // into one command encoder without the later writes clobbering the earlier dispatches.
    // A browser has no encoder writes: the queue's, which land before this frame's
    // submission (the viewer records one frame per encoder).
#ifdef __EMSCRIPTEN__
    device.GetQueue().WriteBuffer(constants_buffer, 0, &constants, sizeof(constants));
#else
    p_encoder.WriteBuffer(constants_buffer, 0, &constants, sizeof(constants));
#endif

    // --- reset of the temporal history ---
    if (reset) {
        // Only the buffer that will be *read* this frame has to be cleared.
        RecordClearTexture(p_encoder, clear_r8, accumulation_view[frame_even ? 1 : 0], render_w, render_h);
        RecordClearTexture(p_encoder, clear_rgba16, luma_history_view[frame_even ? 1 : 0], render_w,
                render_h);
        for (uint32_t mip = 0; mip < spd_mip_count && mip < 6; ++mip) {
            RecordClearTexture(p_encoder, clear_rg16, spd_mip_views[mip],
                    std::max(1u, half_render_w >> mip), std::max(1u, half_render_h >> mip));
        }

        // Reset the auto exposure state (log luma, exposure, scene average luma).
        wgpu::TexelCopyBufferInfo src = {};
        src.buffer = frame_info_staging;
        src.layout.offset = 0;
        src.layout.bytesPerRow = 256;
        src.layout.rowsPerImage = 1;

        wgpu::TexelCopyTextureInfo dst = {};
        dst.texture = frame_info;
        dst.mipLevel = 0;
        dst.origin = { 0, 0, 0 };
        dst.aspect = wgpu::TextureAspect::All;

        wgpu::Extent3D extent = { 1, 1, 1 };
        p_encoder.CopyBufferToTexture(&src, &dst, &extent);
    }

    // The reconstructed previous depth is initialised to the far plane every frame; the
    // prepare_inputs pass atomically minimises it.
    RecordFillDepthBuffer(p_encoder);
    RecordClearTexture(p_encoder, clear_r16, farthest_depth_mip1_view, half_render_w, half_render_h);

    // The SPD atomic counter is a scratch counter for the cross workgroup wait of the
    // pyramid passes: it has to start at zero for every dispatch.
    p_encoder.ClearBuffer(spd_atomic_counter, 0, 4);

    // --- the FSR3 passes, in reference order ---
    const uint32_t wg_src_x = CeilDiv(render_w, 8);
    const uint32_t wg_src_y = CeilDiv(render_h, 8);
    const uint32_t wg_dst_x = CeilDiv(upscale_w, 8);
    const uint32_t wg_dst_y = CeilDiv(upscale_h, 8);
    const uint32_t wg_shading_change_x = CeilDiv(half_render_w, 8);
    const uint32_t wg_shading_change_y = CeilDiv(half_render_h, 8);
    const uint32_t wg_spd_x = CeilDiv(render_w, 64);
    const uint32_t wg_spd_y = CeilDiv(render_h, 64);
    const uint32_t wg_rcas_x = CeilDiv(upscale_w, 16);
    const uint32_t wg_rcas_y = CeilDiv(upscale_h, 16);

    wgpu::ComputePassEncoder pass = p_encoder.BeginComputePass();
    RecordPass(pass, pass_prepare_inputs, p_inputs, wg_src_x, wg_src_y);
    RecordPass(pass, pass_luma_pyramid, p_inputs, wg_spd_x, wg_spd_y);
    RecordPass(pass, pass_shading_change_pyramid, p_inputs, wg_spd_x, wg_spd_y);
    RecordPass(pass, pass_shading_change, p_inputs, wg_shading_change_x, wg_shading_change_y);
    RecordPass(pass, pass_prepare_reactivity, p_inputs, wg_src_x, wg_src_y);
    RecordPass(pass, pass_luma_instability, p_inputs, wg_src_x, wg_src_y);
    RecordPass(pass, pass_accumulate, p_inputs, wg_dst_x, wg_dst_y);
    RecordPass(pass, pass_rcas, p_inputs, wg_rcas_x, wg_rcas_y);
    pass.End();

    previous_jitter[0] = p_inputs.jitter_x;
    previous_jitter[1] = p_inputs.jitter_y;
    frame_even = !frame_even;
    return true;
}

wgpu::TextureView Fsr3Upscaler::OutputView() const {
    return last_output;
}

void Fsr3Upscaler::RecordPass(wgpu::ComputePassEncoder &p_pass, const Pass &p_fsr_pass,
        const Fsr3Inputs &p_inputs, uint32_t p_workgroups_x,
        uint32_t p_workgroups_y) const {
    if (!p_fsr_pass.pipeline) {
        return;
    }

    std::vector<wgpu::BindGroupEntry> entries(p_fsr_pass.binding_count);
    for (uint32_t i = 0; i < p_fsr_pass.binding_count; ++i) {
        entries[i] = MakeEntry(p_fsr_pass.bindings[i], p_inputs);
    }

    wgpu::BindGroupDescriptor bind_group_desc = {};
    bind_group_desc.layout = p_fsr_pass.layout;
    bind_group_desc.entryCount = entries.size();
    bind_group_desc.entries = entries.data();
    wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);
    if (!bind_group) {
        return;
    }

    p_pass.SetPipeline(p_fsr_pass.pipeline);
    p_pass.SetBindGroup(0, bind_group);
    p_pass.DispatchWorkgroups(p_workgroups_x, p_workgroups_y, 1);
}

void Fsr3Upscaler::RecordClearTexture(wgpu::CommandEncoder &p_encoder, const ClearPass &p_clear_pass,
        const wgpu::TextureView &p_view, uint32_t p_width,
        uint32_t p_height) const {
    if (!p_clear_pass.pipeline || !p_view) {
        return;
    }

    wgpu::BindGroupEntry entry = {};
    entry.binding = 0;
    entry.textureView = p_view;

    wgpu::BindGroupDescriptor bind_group_desc = {};
    bind_group_desc.layout = p_clear_pass.layout;
    bind_group_desc.entryCount = 1;
    bind_group_desc.entries = &entry;
    wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);
    if (!bind_group) {
        return;
    }

    wgpu::ComputePassEncoder pass = p_encoder.BeginComputePass();
    pass.SetPipeline(p_clear_pass.pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(CeilDiv(p_width, 8), CeilDiv(p_height, 8), 1);
    pass.End();
}

void Fsr3Upscaler::RecordFillDepthBuffer(wgpu::CommandEncoder &p_encoder) const {
    if (!fill_buffer.pipeline) {
        return;
    }

    wgpu::BindGroupEntry entries[2] = {};
    entries[0].binding = 0;
    entries[0].buffer = reconstructed_previous_depth;
    entries[1].binding = 1;
    entries[1].buffer = fill_value_buffer;

    wgpu::BindGroupDescriptor bind_group_desc = {};
    bind_group_desc.layout = fill_buffer.layout;
    bind_group_desc.entryCount = 2;
    bind_group_desc.entries = entries;
    wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);
    if (!bind_group) {
        return;
    }

    const uint32_t pixel_count = render_w * render_h;
    wgpu::ComputePassEncoder pass = p_encoder.BeginComputePass();
    pass.SetPipeline(fill_buffer.pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(CeilDiv(pixel_count, 64), 1, 1);
    pass.End();
}

} // namespace ov::dawn
