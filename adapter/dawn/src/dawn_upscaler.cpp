#include "dawn_upscaler.hpp"

#include "dawn_page_host.hpp"
#include "wgpu_util.hpp"

#include "ov/log.hpp"
#include "ov/on_error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace ov::dawn {

namespace {

// FSR 3's inputs from the 3D's reversed infinite depth. The vendored shaders are the display-resolution motion vector
// variant: their passes read the motion at the output pixel (prepare_inputs maps each render pixel to it), so the
// motion is an output-sized target and the depth a render-sized one, two passes on the same block.
constexpr const char PREPARE_BLOCK[] = R"glsl(
layout(set = 0, binding = 0, std140) uniform Prepare {
    mat4 reproject; // this frame's unjittered clip space to the last frame's
    vec4 params; // output w, h, render w, h
    vec4 depth_scale; // x: far / (far - near)
};
layout(set = 0, binding = 1) uniform texture2D depth_tex;
)glsl";

// The standard depth of the same projection with a far plane (near 0, far 1; the vendored shaders' convention).
constexpr const char DEPTH_FRAGMENT[] = R"glsl(
layout(location = 0) out float out_depth;

void main() {
    out_depth = (1.0 - texelFetch(depth_tex, ivec2(gl_FragCoord.xy), 0).r) * depth_scale.x;
}
)glsl";

// The camera's motion alone (the terrain is static): the point under each output pixel centre (unjittered; its depth
// from the render pixel over it) reprojected into the last frame, in output pixels from this frame to the last.
constexpr const char MOTION_FRAGMENT[] = R"glsl(
layout(location = 0) out vec2 out_motion;

void main() {
    vec2 size = params.xy;
    ivec2 texel = min(ivec2(gl_FragCoord.xy * params.zw / size), ivec2(params.zw) - 1);
    float d = texelFetch(depth_tex, texel, 0).r;
    vec2 ndc = vec2(gl_FragCoord.x / size.x * 2.0 - 1.0, 1.0 - gl_FragCoord.y / size.y * 2.0);
    vec4 prev = reproject * vec4(ndc, d, 1.0);
    vec2 prev_ndc = prev.w > 0.0 ? prev.xy / prev.w : ndc;
    out_motion = (prev_ndc - ndc) * 0.5 * size * vec2(1.0, -1.0);
}
)glsl";

// Each source pixel over a block x block square of the target.
constexpr const char PRESENT_FRAGMENT[] = R"glsl(
layout(set = 0, binding = 0, std140) uniform Present {
    vec4 params; // pixel block, source w, h
};
layout(set = 0, binding = 1) uniform texture2D source_tex;
layout(location = 0) out vec4 out_color;

void main() {
    ivec2 t = min(ivec2(gl_FragCoord.xy / params.x), ivec2(params.yz) - 1);
    out_color = vec4(clamp(texelFetch(source_tex, t, 0).rgb, 0.0, 1.0), 1.0);
}
)glsl";

// The far plane of FSR's standard depth: past the farthest terrain drawn (its view-space depth only feeds heuristics).
constexpr float FSR_FAR_M = 1.0e6f;

float halton(int p_index, int p_base) {
    float f = 1.0f;
    float r = 0.0f;
    for (int i = p_index; i > 0; i /= p_base) {
        f /= float(p_base);
        r += f * float(i % p_base);
    }
    return r;
}

// r_out = p_a * p_b, column-major.
void multiply(const double p_a[16], const double p_b[16], double r_out[16]) {
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            double s = 0.0;
            for (int k = 0; k < 4; k++) {
                s += p_a[k * 4 + row] * p_b[c * 4 + k];
            }
            r_out[c * 4 + row] = s;
        }
    }
}

// Gauss-Jordan with partial pivoting; false when singular.
bool invert(const double p_m[16], double r_inv[16]) {
    double a[4][8];
    for (int row = 0; row < 4; row++) {
        for (int c = 0; c < 4; c++) {
            a[row][c] = p_m[c * 4 + row];
            a[row][c + 4] = row == c ? 1.0 : 0.0;
        }
    }
    for (int c = 0; c < 4; c++) {
        int pivot = c;
        for (int row = c + 1; row < 4; row++) {
            if (std::abs(a[row][c]) > std::abs(a[pivot][c])) {
                pivot = row;
            }
        }
        if (std::abs(a[pivot][c]) < 1e-30) {
            return false;
        }
        std::swap(a[c], a[pivot]);
        const double scale = 1.0 / a[c][c];
        for (double &v : a[c]) {
            v *= scale;
        }
        for (int row = 0; row < 4; row++) {
            if (row != c) {
                const double k = a[row][c];
                for (int i = 0; i < 8; i++) {
                    a[row][i] -= k * a[c][i];
                }
            }
        }
    }
    for (int row = 0; row < 4; row++) {
        for (int c = 0; c < 4; c++) {
            r_inv[c * 4 + row] = a[row][c + 4];
        }
    }
    return true;
}

wgpu::RenderPipeline make_pipeline(const wgpu::Device &p_device, const wgpu::BindGroupLayout &p_layout, const char *p_fragment,
        const std::vector<wgpu::TextureFormat> &p_formats, const char *p_label) {
    const wgpu::ShaderModule vs = make_render_module(p_device, std::string(GLSL_HEADER) + FULLSCREEN_VERTEX, ShaderStage::VERTEX);
    const wgpu::ShaderModule fs = make_render_module(p_device, std::string(GLSL_HEADER) + p_fragment, ShaderStage::FRAGMENT);
    if (!vs || !fs) {
        return nullptr;
    }
    wgpu::PipelineLayoutDescriptor layout_desc{};
    layout_desc.bindGroupLayoutCount = 1;
    layout_desc.bindGroupLayouts = &p_layout;
    std::vector<wgpu::ColorTargetState> targets(p_formats.size());
    for (size_t i = 0; i < p_formats.size(); i++) {
        targets[i].format = p_formats[i];
    }
    wgpu::FragmentState fragment{};
    fragment.module = fs;
    fragment.entryPoint = "main";
    fragment.targetCount = targets.size();
    fragment.targets = targets.data();
    wgpu::RenderPipelineDescriptor desc{};
    desc.label = p_label;
    desc.layout = p_device.CreatePipelineLayout(&layout_desc);
    desc.vertex.module = vs;
    desc.vertex.entryPoint = "main";
    desc.fragment = &fragment;
    return p_device.CreateRenderPipeline(&desc);
}

void fullscreen_pass(const wgpu::CommandEncoder &p_encoder, const std::vector<wgpu::TextureView> &p_targets, const wgpu::RenderPipeline &p_pipeline,
        const wgpu::BindGroup &p_group) {
    std::vector<wgpu::RenderPassColorAttachment> colors(p_targets.size());
    for (size_t i = 0; i < p_targets.size(); i++) {
        colors[i].view = p_targets[i];
        colors[i].loadOp = wgpu::LoadOp::Clear;
        colors[i].storeOp = wgpu::StoreOp::Store;
    }
    wgpu::RenderPassDescriptor desc{};
    desc.colorAttachmentCount = colors.size();
    desc.colorAttachments = colors.data();
    const wgpu::RenderPassEncoder pass = p_encoder.BeginRenderPass(&desc);
    pass.SetPipeline(p_pipeline);
    pass.SetBindGroup(0, p_group);
    pass.Draw(3);
    pass.End();
}

} // namespace

DawnUpscaler::DawnUpscaler(const wgpu::Device &p_device, const wgpu::Instance &p_instance, wgpu::TextureFormat p_target_format,
        const Settings &p_settings, const Camera &p_camera) :
        device(p_device), instance(p_instance), settings(p_settings), camera(p_camera) {
    settings.pixelate = std::max(settings.pixelate, 1);
    settings.render_scale = std::clamp(settings.render_scale, 0.25f, 1.0f);
    if (settings.fsr && !device.HasFeature(wgpu::FeatureName::TextureFormatsTier2)) {
        OV_ERR_PRINT("DawnUpscaler: no TextureFormatsTier2 on this device: no FSR, the 3D at the output size.");
        settings.fsr = false;
    }
    make_pipelines(p_target_format);
}

DawnUpscaler::~DawnUpscaler() = default;

void DawnUpscaler::make_pipelines(wgpu::TextureFormat p_target_format) {
    present_layout = make_layout(device, { uniform_entry(0), texture_entry(1, wgpu::TextureSampleType::UnfilterableFloat) });
    present_pipeline = make_pipeline(device, present_layout, PRESENT_FRAGMENT, { p_target_format }, "upscaler present");
    present_uniforms = make_uniforms(device, sizeof(PresentUniforms));
    if (!settings.fsr) {
        return;
    }
    prepare_layout = make_layout(device, { uniform_entry(0), texture_entry(1, wgpu::TextureSampleType::UnfilterableFloat) });
    depth_pipeline = make_pipeline(device, prepare_layout, (std::string(PREPARE_BLOCK) + DEPTH_FRAGMENT).c_str(), { wgpu::TextureFormat::R32Float },
            "upscaler fsr depth");
    motion_pipeline = make_pipeline(device, prepare_layout, (std::string(PREPARE_BLOCK) + MOTION_FRAGMENT).c_str(), { wgpu::TextureFormat::RG16Float },
            "upscaler fsr motion");
    prepare_uniforms = make_uniforms(device, sizeof(PrepareUniforms));
}

void DawnUpscaler::resize(int p_width, int p_height) {
    output_w = std::max((p_width + settings.pixelate - 1) / settings.pixelate, 1);
    output_h = std::max((p_height + settings.pixelate - 1) / settings.pixelate, 1);
    const float scale = settings.fsr ? settings.render_scale : 1.0f;
    render_w = std::max(int(std::lround(output_w * scale)), 2);
    render_h = std::max(int(std::lround(output_h * scale)), 2);
    const wgpu::TextureUsage target = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
    color = make_texture(device, COLOR_FORMAT, render_w, render_h, target);
    depth = make_texture(device, DEPTH_FORMAT, render_w, render_h, target);
    reset = true;
    if (!ready()) {
        return;
    }
    const wgpu::BindGroupEntry present_uniform = buffer_bind(0, present_uniforms, sizeof(PresentUniforms));
    if (!settings.fsr) {
        present_group = make_group(device, present_layout, { present_uniform, texture_bind(1, color) });
        return;
    }
    fsr_depth = make_texture(device, wgpu::TextureFormat::R32Float, render_w, render_h, target);
    motion = make_texture(device, wgpu::TextureFormat::RG16Float, output_w, output_h, target);
    fsr_output = make_texture(device, wgpu::TextureFormat::RGBA16Float, output_w, output_h,
            wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding);
    fsr = std::make_unique<Fsr3Upscaler>(device, instance, uint32_t(render_w), uint32_t(render_h), uint32_t(output_w), uint32_t(output_h));
    prepare_group = make_group(device, prepare_layout, { buffer_bind(0, prepare_uniforms, sizeof(PrepareUniforms)), texture_bind(1, depth) });
    present_group = make_group(device, present_layout, { present_uniform, texture_bind(1, fsr_output) });
}

void DawnUpscaler::shift_history(const Float3 &p_shift) {
    // A point p now was p + shift then: prev_view_proj * translate(shift). Column-major: only the last column changes.
    const double t[3] = { p_shift.x, p_shift.y, p_shift.z };
    for (int row = 0; row < 4; row++) {
        prev_view_proj[12 + row] += prev_view_proj[row] * t[0] + prev_view_proj[4 + row] * t[1] + prev_view_proj[8 + row] * t[2];
    }
}

void DawnUpscaler::begin_frame(const float p_view_proj[16], float r_jittered[16]) {
    std::memcpy(r_jittered, p_view_proj, sizeof(float) * 16);
    if (!settings.fsr) {
        return;
    }
    // FSR's jitter: Halton (2, 3) over 8 phases per output pixel per render pixel, in render pixels (y down).
    const float ratio = float(output_w) / float(render_w);
    const int phases = std::max(int(8.0f * ratio * ratio), 1);
    const int phase = frame % phases + 1;
    jitter[0] = halton(phase, 2) - 0.5f;
    jitter[1] = halton(phase, 3) - 0.5f;
    // The scene moves by the jitter: clip x += dx w, y += dy w.
    const float dx = 2.0f * jitter[0] / float(render_w);
    const float dy = -2.0f * jitter[1] / float(render_h);
    for (int c = 0; c < 4; c++) {
        r_jittered[c * 4 + 0] += dx * p_view_proj[c * 4 + 3];
        r_jittered[c * 4 + 1] += dy * p_view_proj[c * 4 + 3];
    }

    double current[16];
    double inverse[16];
    double reproject[16];
    for (int i = 0; i < 16; i++) {
        current[i] = p_view_proj[i];
    }
    if (!invert(current, inverse)) {
        reset = true;
    }
    multiply(prev_view_proj, inverse, reproject);
    PrepareUniforms u{};
    for (int i = 0; i < 16; i++) {
        u.reproject[i] = float(reproject[i]);
        prev_view_proj[i] = current[i];
    }
    u.params[0] = float(output_w);
    u.params[1] = float(output_h);
    u.params[2] = float(render_w);
    u.params[3] = float(render_h);
    u.depth_scale[0] = FSR_FAR_M / (FSR_FAR_M - camera.near_m);
    device.GetQueue().WriteBuffer(prepare_uniforms, 0, &u, sizeof(u));
}

void DawnUpscaler::encode(const wgpu::CommandEncoder &p_encoder, const wgpu::TextureView &p_target, float p_frame_seconds) {
    PresentUniforms p{};
    p.params[0] = float(settings.pixelate);
    p.params[1] = float(settings.fsr ? output_w : render_w);
    p.params[2] = float(settings.fsr ? output_h : render_h);
    device.GetQueue().WriteBuffer(present_uniforms, 0, &p, sizeof(p));
    if (settings.fsr) {
        fullscreen_pass(p_encoder, { fsr_depth.CreateView() }, depth_pipeline, prepare_group);
        fullscreen_pass(p_encoder, { motion.CreateView() }, motion_pipeline, prepare_group);
        Fsr3Inputs inputs;
        inputs.color = color.CreateView();
        inputs.depth = fsr_depth.CreateView();
        inputs.motion_vectors = motion.CreateView();
        inputs.output = fsr_output.CreateView();
        inputs.jitter_x = jitter[0];
        inputs.jitter_y = jitter[1];
        inputs.frame_time_ms = p_frame_seconds * 1000.0f;
        inputs.reset = reset;
        inputs.camera_near = camera.near_m;
        inputs.camera_far = FSR_FAR_M;
        inputs.fov_y = camera.fov_y;
        inputs.sharpness = settings.sharpness;
        wgpu::CommandEncoder encoder = p_encoder;
        OV_ERR_FAIL_COND_MSG(!fsr->Dispatch(encoder, inputs), "DawnUpscaler: FSR 3 did not run (its shaders or resources failed).");
        reset = false;
    }
    fullscreen_pass(p_encoder, { p_target }, present_pipeline, present_group);
    frame++;
}

} // namespace ov::dawn
