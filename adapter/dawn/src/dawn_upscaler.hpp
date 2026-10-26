#pragma once

// The viewer's upscaling, after Godot's `--upscale=fsr2 --render_scale=F --pixelate=N` (player.gd): the 3D renders at
// F of the output, AMD's FSR 3.1 upscaler (fsr3.hpp, the vendored reference shaders) brings it to the output (the
// window over N per axis), and a last pass copies each output pixel over an N x N block of the window. Without FSR
// (pixelate only) the 3D renders at the output size.
//
// FSR's inputs come from the 3D's color and depth: the projection is Halton-jittered (FSR's phase count), and two passes
// turn the reversed infinite depth into FSR's standard depth and derive the output-sized motion vectors from it (the terrain is
// static: the camera's motion is all the motion there is).

#include "fsr3.hpp"

#include <webgpu/webgpu_cpp.h>

#include <memory>

namespace ov::dawn {

class DawnUpscaler {
public:
    static constexpr wgpu::TextureFormat COLOR_FORMAT = wgpu::TextureFormat::RGBA16Float; // the 3D render's
    static constexpr wgpu::TextureFormat DEPTH_FORMAT = wgpu::TextureFormat::Depth32Float;

    struct Settings {
        bool fsr = false;
        float render_scale = 0.5f; // of the output, per axis (FSR only)
        int pixelate = 1; // the output is the window over this per axis
        float sharpness = 0.2f; // RCAS, FSR's API: 0 none .. 1 the most
    };

    // The 3D's projection (reversed infinite depth, FlyCamera::view_proj): FSR's depth and view-space conversions.
    struct Camera {
        float near_m = 0.1f;
        float fov_y = 1.0f; // radians
    };

    // p_target_format: the window's (or the shot's) format. FSR needs the device's TextureFormatsTier2: without it,
    // an error and the 3D at the output size.
    DawnUpscaler(const wgpu::Device &p_device, const wgpu::Instance &p_instance, wgpu::TextureFormat p_target_format, const Settings &p_settings,
            const Camera &p_camera);
    ~DawnUpscaler();

    // The window's size: the render and output sizes follow, the history starts over.
    void resize(int p_width, int p_height);
    int render_width() const { return render_w; }
    int render_height() const { return render_h; }
    bool ready() const { return present_pipeline != nullptr && (!settings.fsr || (depth_pipeline != nullptr && motion_pipeline != nullptr)); }

    // This frame's projection (column-major, clip = m * world): r_jittered is what the 3D is drawn with.
    void begin_frame(const float p_view_proj[16], float r_jittered[16]);
    // The 3D's targets for this frame (depth stored: FSR reads it).
    wgpu::TextureView color_view() const { return color.CreateView(); }
    wgpu::TextureView depth_view() const { return depth.CreateView(); }
    // After the 3D pass: FSR's passes, then the present pass into p_target (a window-sized p_target_format view).
    void encode(const wgpu::CommandEncoder &p_encoder, const wgpu::TextureView &p_target, float p_frame_seconds);

private:
    // std140, the prepare shaders' `Prepare` block.
    struct PrepareUniforms {
        float reproject[16]; // this frame's (unjittered) clip space to the last frame's
        float params[4]; // output w, h, render w, h
        float depth_scale[4]; // x: far / (far - near)
    };
    // std140, the present shader's `Present` block.
    struct PresentUniforms {
        float params[4]; // pixel block, source w, h
    };

    void make_pipelines(wgpu::TextureFormat p_target_format);

    wgpu::Device device;
    wgpu::Instance instance;
    Settings settings;
    Camera camera;
    int render_w = 0;
    int render_h = 0;
    int output_w = 0;
    int output_h = 0;
    wgpu::Texture color;
    wgpu::Texture depth;
    wgpu::Texture fsr_depth; // R32Float: standard depth
    wgpu::Texture motion; // RG16Float, the output size: output pixels, this frame to the last
    wgpu::Texture fsr_output; // RGBA16Float, the output size
    std::unique_ptr<Fsr3Upscaler> fsr;
    int frame = 0; // the jitter phase
    bool reset = true;
    double prev_view_proj[16] = {};
    float jitter[2] = {};

    wgpu::RenderPipeline depth_pipeline;
    wgpu::RenderPipeline motion_pipeline;
    wgpu::RenderPipeline present_pipeline;
    wgpu::BindGroupLayout prepare_layout;
    wgpu::BindGroupLayout present_layout;
    wgpu::Buffer prepare_uniforms;
    wgpu::Buffer present_uniforms;
    wgpu::BindGroup prepare_group;
    wgpu::BindGroup present_group;
};

} // namespace ov::dawn
