// AMD FidelityFX Super Resolution 3.1 upscaler on Dawn (WebGPU): the viewer's (DawnUpscaler).
//
// This is a C++/Dawn port of the FSR3 temporal upscaler: a lower resolution render
// (color + depth + motion vectors) is reconstructed at display resolution using
// temporal accumulation, with the same passes, resources and constants as AMD's
// reference implementation.
//
// Pipeline (same order as the reference dispatch):
//   prepare_inputs -> luma_pyramid -> shading_change_pyramid -> shading_change
//   -> prepare_reactivity -> luma_instability -> accumulate -> rcas
//
// Device requirements (checked by the constructor; Dispatch() returns false when they
// are missing):
//   * wgpu::InstanceFeatureName::ShaderSourceSPIRV on the instance that created the device.
//   * wgpu::FeatureName::TextureFormatsTier2 on the device. The FSR3 shaders declare their
//     R8Unorm / R16Float / RGBA16Float targets as read-write storage images, which WebGPU
//     only allows with the tier 2 texture formats (tier 1 is implied by tier 2).
//   * limits.maxStorageTexturesPerShaderStage >= 8 (the luma pyramid pass binds 8 storage
//     textures, the default WebGPU limit is 4), limits.maxSampledTexturesPerShaderStage >= 9
//     and limits.maxStorageBuffersPerShaderStage >= 2. Request them from the adapter limits
//     when creating the device, e.g. requiredLimits = adapter limits.
//
// The embedded SPIR-V (fsr3_shaders.hpp) is the upstream FSR3 shader set with the mechanical
// rewrites Dawn's SPIR-V -> WGSL path needs; see the comment block in that file.
//
// The caller keeps ownership of the textures; every texture view passed in Fsr3Inputs
// must already have the usage the pass needs (sampled textures: TEXTURE_BINDING,
// output: STORAGE_BINDING).

#pragma once

#include <cstdint>
#include <vector>

#include <dawn/webgpu_cpp.h>

namespace ov::dawn {

// std140 constants block shared by every FSR3 pass. The layout is byte-for-byte the
// one the precompiled shaders read (ffx_fsr3upscaler_callbacks_glsl.h, struct
// Fsr3UpscalerConstants); the static_asserts in fsr3.cpp pin it down.
struct Fsr3FsrConstants {
    uint32_t render_size[2];
    uint32_t previous_frame_render_size[2];

    uint32_t upscale_size[2];
    uint32_t previous_frame_upscale_size[2];

    uint32_t max_render_size[2];
    uint32_t max_upscale_size[2];

    float device_to_view_depth[4];

    float jitter_offset[2];
    float previous_frame_jitter_offset[2];

    float motion_vector_scale[2];
    float downscale_factor[2];

    float motion_vector_jitter_cancellation[2];
    float tan_half_fov;
    float jitter_phase_count;

    float delta_time;
    float delta_pre_exposure;
    float view_space_to_meters_factor;
    float frame_index;

    float velocity_factor;
    float reactiveness_scale;
    float shading_change_scale;
    float accumulation_added_per_frame;
    float min_disocclusion_accumulation;
    uint32_t padding[3];
};

struct Fsr3AutoReactiveConstants {
    float tc_threshold; // 0.1 is a good starting value, lower results in more TC pixels
    float tc_scale;
    float reactive_scale;
    float reactive_max;
};

struct Fsr3RcasConstants {
    uint32_t rcas_config[4];
};

struct Fsr3ReactiveConstants {
    float gen_reactive_scale;
    float gen_reactive_threshold;
    float gen_reactive_binary_value;
    uint32_t gen_reactive_flags;
};

struct Fsr3SpdConstants {
    uint32_t mips;
    uint32_t num_work_groups;
    uint32_t work_group_offset[2];
    uint32_t render_size[2];
    uint32_t padding[2];
};

struct Fsr3Constants {
    Fsr3FsrConstants fsr;
    Fsr3AutoReactiveConstants generate_auto_reactive;
    Fsr3RcasConstants rcas;
    Fsr3ReactiveConstants generate_reactive;
    Fsr3SpdConstants spd;
};

// Per-frame inputs of the upscaler. All views are owned by the caller and must stay
// alive until the command encoder recorded by Dispatch() has been submitted.
struct Fsr3Inputs {
    wgpu::TextureView color; // Rgba16Float, render size, sampled
    wgpu::TextureView depth; // Depth32Float, render size, sampled
    wgpu::TextureView motion_vectors; // Rg16Float, upscale (output) size, sampled (motion in output pixels)
    wgpu::TextureView output; // Rgba16Float, upscale size, storage
    float jitter_x = 0.0f, jitter_y = 0.0f;
    float frame_time_ms = 16.0f;
    bool reset = false;
    // The projection the depth was drawn with (standard depth: near 0, far 1), radians.
    float camera_near = 0.1f;
    float camera_far = 1000.0f;
    float fov_y = 1.0f;
    float sharpness = 0.2f; // RCAS: 0 none, 1 the most (the reference API's)
};

// FSR3.1 temporal upscaler: owns every internal texture / buffer / pipeline, records
// the compute passes of one frame into a caller supplied command encoder.
class Fsr3Upscaler {
public:
    Fsr3Upscaler(wgpu::Device p_device, wgpu::Instance p_instance, uint32_t p_render_w, uint32_t p_render_h,
            uint32_t p_upscale_w, uint32_t p_upscale_h);
    ~Fsr3Upscaler();

    Fsr3Upscaler(const Fsr3Upscaler &) = delete;
    Fsr3Upscaler &operator=(const Fsr3Upscaler &) = delete;

    // Records every FSR3 pass into p_encoder. Returns false if the shaders/resources are unavailable.
    bool Dispatch(wgpu::CommandEncoder &p_encoder, const Fsr3Inputs &p_inputs);

    // The final upscaled (RCAS-sharpened) color, Rgba16Float: the output view of the last
    // successful Dispatch() (an empty view before the first one).
    wgpu::TextureView OutputView() const;

private:
    // Which resource a pass binding refers to, mirroring FsrResourceName in the reference port.
    enum class Resource : uint32_t {
        InputColor,
        InputDepth,
        InputMotionVectors,
        InputExposure,
        InputReactiveMask,
        InputTransparencyAndComposition,

        OutputColor,
        OutputDilatedDepth,
        OutputDilatedMotionVectors,
        OutputReconstructedPreviousDepth,

        Constants,

        AccumulationCurrent,
        AccumulationPrevious,
        Luma,
        PreviousLuma,
        LumaInstability,
        ShadingChange,
        NewLocks,
        InternalUpscaledCurrent,
        InternalUpscaledPrevious,
        SpdMips,
        FarthestDepth,
        FarthestDepthMip1,
        LumaHistoryCurrent,
        LumaHistoryPrevious,
        SpdAtomicCount,
        DilatedReactiveMasks,
        Lanczos2Lut,
        DefaultReactivityMask,
        DefaultExposure,
        FrameInfo,

        SamplerPointClamp,
        SamplerLinearClamp,
    };

    enum class BindType : uint32_t {
        SampledTexture,
        StorageTexture,
        UniformBuffer,
        StorageBuffer,
        SamplerPoint,
        SamplerLinear,
    };

    // One entry of a pass' bind group layout / bind group (binding index == layout index,
    // exactly as the reference pass resource lists do it).
    struct Binding {
        uint32_t binding;
        BindType type;
        Resource resource;
        wgpu::TextureFormat format = wgpu::TextureFormat::Undefined; // StorageTexture only
        bool filterable = false; // SampledTexture only
        bool write_only = false; // StorageTexture only
        int32_t mip = -1; // >= 0: single mip view of SpdMips, -1: the whole texture
    };

    struct Pass {
        wgpu::BindGroupLayout layout;
        wgpu::ComputePipeline pipeline;
        const Binding *bindings = nullptr;
        uint32_t binding_count = 0;
    };

    // A tiny helper compute pipeline (clear a storage texture, or fill a storage buffer).
    struct ClearPass {
        wgpu::BindGroupLayout layout;
        wgpu::ComputePipeline pipeline;
    };

    // --- construction helpers ---
    bool CreateResources();
    bool CreatePipelines();
    bool CreatePass(const uint32_t *p_spirv, uint32_t p_spirv_words, const Binding *p_bindings,
            uint32_t p_binding_count, const char *p_label, Pass &r_pass);
    bool CreateClearPipelines();

    // --- recording helpers ---
    wgpu::TextureView ResolveView(const Binding &p_binding, const Fsr3Inputs &p_inputs) const;
    wgpu::Buffer ResolveBuffer(Resource p_resource) const;
    wgpu::BindGroupEntry MakeEntry(const Binding &p_binding, const Fsr3Inputs &p_inputs) const;
    void RecordPass(wgpu::ComputePassEncoder &p_pass, const Pass &p_fsr_pass, const Fsr3Inputs &p_inputs,
            uint32_t p_workgroups_x, uint32_t p_workgroups_y) const;
    void RecordClearTexture(wgpu::CommandEncoder &p_encoder, const ClearPass &p_clear_pass, const wgpu::TextureView &p_view,
            uint32_t p_width, uint32_t p_height) const;
    void RecordFillDepthBuffer(wgpu::CommandEncoder &p_encoder) const;

    wgpu::Device device;
    [[maybe_unused]] wgpu::Instance instance; // kept alive for the device, not otherwise needed
    wgpu::Queue queue;

    uint32_t render_w = 0;
    uint32_t render_h = 0;
    uint32_t upscale_w = 0;
    uint32_t upscale_h = 0;
    uint32_t half_render_w = 0;
    uint32_t half_render_h = 0;
    uint32_t spd_mip_count = 0; // mip levels of the half resolution SPD texture

    bool valid = false;
    bool first_frame = true;
    bool frame_even = true; // double buffering parity of the temporal resources
    float previous_jitter[2] = { 0.0f, 0.0f };

    wgpu::TextureView last_output;

    // --- internal textures (formats/sizes/usages identical to the reference port) ---
    wgpu::Texture accumulation[2]; // R8Unorm, render size
    wgpu::Texture luma[2]; // R16Float, render size
    wgpu::Texture luma_history[2]; // Rgba16Float, render size
    wgpu::Texture internal_upscaled[2]; // Rgba16Float, upscale size
    wgpu::Texture intermediate_fp16x1; // R16Float, render size (luma instability + farthest depth)
    wgpu::Texture shading_change; // R8Unorm, half render size
    wgpu::Texture new_locks; // R8Unorm, upscale size
    wgpu::Texture spd_mips; // Rg16Float, half render size, full mip chain
    wgpu::Texture farthest_depth_mip1; // R16Float, half render size
    wgpu::Texture dilated_reactive_masks; // Rgba8Unorm, render size
    wgpu::Texture dilated_depth; // R32Float, render size (output of prepare_inputs)
    wgpu::Texture dilated_motion_vectors; // Rg16Float, render size (output of prepare_inputs)
    wgpu::Texture lanczos2_lut; // R16Snorm, 128x1
    wgpu::Texture default_reactivity_mask; // R8Unorm, 1x1
    wgpu::Texture default_exposure; // Rg32Float, 1x1
    wgpu::Texture frame_info; // Rgba32Float, 1x1

    wgpu::TextureView accumulation_view[2];
    wgpu::TextureView luma_view[2];
    wgpu::TextureView luma_history_view[2];
    wgpu::TextureView internal_upscaled_view[2];
    wgpu::TextureView intermediate_fp16x1_view;
    wgpu::TextureView shading_change_view;
    wgpu::TextureView new_locks_view;
    wgpu::TextureView spd_mips_view;
    wgpu::TextureView spd_mip_views[6]; // one single-mip view per SPD output level
    wgpu::Texture spd_dummy_mips[6]; // stand-ins for levels a small render size does not have
    wgpu::TextureView farthest_depth_mip1_view;
    wgpu::TextureView dilated_reactive_masks_view;
    wgpu::TextureView dilated_depth_view;
    wgpu::TextureView dilated_motion_vectors_view;
    wgpu::TextureView lanczos2_lut_view;
    wgpu::TextureView default_reactivity_mask_view;
    wgpu::TextureView default_exposure_view;
    wgpu::TextureView frame_info_view;

    // --- internal buffers ---
    wgpu::Buffer constants_buffer; // Fsr3Constants, uniform
    wgpu::Buffer spd_atomic_counter; // one uint, storage
    wgpu::Buffer reconstructed_previous_depth; // one uint per render pixel, storage
    wgpu::Buffer fill_value_buffer; // 16 bytes, uniform, value of the depth buffer fill
    wgpu::Buffer frame_info_staging; // 16 bytes, copy source of the frame info reset

    wgpu::Sampler sampler_point;
    wgpu::Sampler sampler_linear;

    Fsr3Constants constants = {};

    Pass pass_prepare_inputs;
    Pass pass_luma_pyramid;
    Pass pass_shading_change_pyramid;
    Pass pass_shading_change;
    Pass pass_prepare_reactivity;
    Pass pass_luma_instability;
    Pass pass_accumulate;
    Pass pass_rcas;

    // Clear / fill helpers (WGSL, not part of the FSR3 shader set).
    ClearPass clear_r8;
    ClearPass clear_r16;
    ClearPass clear_rg16;
    ClearPass clear_rgba16;
    ClearPass fill_buffer;
};

} // namespace ov::dawn
