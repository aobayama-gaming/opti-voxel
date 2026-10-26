#pragma once

// The viewer's upscaling, after Godot's `--upscale=fsr2 --render_scale=F --pixelate=N` (player.gd): the 3D renders at
// F of the output, a temporal upscaler brings it to the output (the window over N per axis), and a last pass copies
// each output pixel over an N x N block of the window.
//
// The temporal upscaler follows FSR 2 / 3's upscaling pass in a compact form (not AMD's shaders): a Halton-jittered
// projection, every output pixel reprojected into the last frame by its depth (the terrain is static: the camera's
// motion is all the motion there is), the history resampled (Catmull-Rom) and clamped to this frame's 3 x 3
// neighbourhood in YCoCg, blended with this frame's Lanczos-2 reconstruction by the samples it has accumulated, then
// RCAS sharpening. Without the temporal pass (pixelate only) the 3D renders at the output size.

#include <dawn/webgpu_cpp.h>

namespace ov::dawn {

class DawnUpscaler {
public:
    static constexpr wgpu::TextureFormat COLOR_FORMAT = wgpu::TextureFormat::RGBA8Unorm; // the 3D render's
    static constexpr wgpu::TextureFormat DEPTH_FORMAT = wgpu::TextureFormat::Depth32Float;

    struct Settings {
        bool temporal = false;
        float render_scale = 0.5f; // of the output, per axis (temporal only)
        int pixelate = 1; // the output is the window over this per axis
        float sharpness = 0.2f; // RCAS, in stops (0 sharpest); negative: off
    };

    // p_target_format: the window's (or the shot's) format.
    DawnUpscaler(const wgpu::Device &p_device, wgpu::TextureFormat p_target_format, const Settings &p_settings);

    // The window's size: the render and output sizes follow, the history starts over.
    void resize(int p_width, int p_height);
    int render_width() const { return render_w; }
    int render_height() const { return render_h; }
    bool ready() const { return present_pipeline != nullptr && (!settings.temporal || resolve_pipeline != nullptr); }

    // This frame's projection (column-major, clip = m * world): r_jittered is what the 3D is drawn with.
    void begin_frame(const float p_view_proj[16], float r_jittered[16]);
    // The 3D's targets for this frame (depth stored: the reprojection reads it).
    wgpu::TextureView color_view() const { return color.CreateView(); }
    wgpu::TextureView depth_view() const { return depth.CreateView(); }
    // After the 3D pass: the resolve and the present pass into p_target (a window-sized p_target_format view).
    void encode(const wgpu::CommandEncoder &p_encoder, const wgpu::TextureView &p_target);

private:
    // std140, the resolve shader's `Resolve` block.
    struct ResolveUniforms {
        float reproject[16]; // this frame's (unjittered) clip space to the last frame's
        float sizes[4]; // render w, h, output w, h
        float params[4]; // jitter x, y (render pixels, y down), reset (1: no history)
    };
    // std140, the present shader's `Present` block.
    struct PresentUniforms {
        float params[4]; // pixel block, sharpness (negative: off), source w, h
    };

    void make_pipelines(wgpu::TextureFormat p_target_format);

    wgpu::Device device;
    Settings settings;
    int render_w = 0;
    int render_h = 0;
    int output_w = 0;
    int output_h = 0;
    wgpu::Texture color;
    wgpu::Texture depth;
    wgpu::Texture history[2]; // RGBA16Float at the output size: rgb, a = the samples accumulated
    int frame = 0; // the jitter phase and the history's parity
    bool reset = true;
    double prev_view_proj[16] = {};
    float jitter[2] = {};

    wgpu::RenderPipeline resolve_pipeline;
    wgpu::RenderPipeline present_pipeline;
    wgpu::BindGroupLayout resolve_layout;
    wgpu::BindGroupLayout present_layout;
    wgpu::Buffer resolve_uniforms;
    wgpu::Buffer present_uniforms;
    wgpu::Sampler linear;
    wgpu::BindGroup resolve_groups[2]; // by the parity of the history written
    wgpu::BindGroup present_groups[2]; // temporal: the history written; else [0], the color
};

} // namespace ov::dawn
