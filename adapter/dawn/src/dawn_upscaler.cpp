#include "dawn_upscaler.hpp"

#include "dawn_page_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace ov::dawn {

namespace {

constexpr const char HEADER[] = "#version 450\n#extension GL_EXT_samplerless_texture_functions : require\n";

// One triangle over the target.
constexpr const char FULLSCREEN_VERTEX[] = R"glsl(
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

// Per output pixel: this frame's Lanczos-2 reconstruction from the jittered render, the history reprojected by the
// nearest depth of the 3 x 3 around it (FSR's dilated depth), clamped to the 3 x 3's YCoCg box, blended by the samples
// accumulated (a: their count, capped at MAX_SAMPLES; a clamped history loses them).
constexpr const char RESOLVE_FRAGMENT[] = R"glsl(
layout(set = 0, binding = 0, std140) uniform Resolve {
    mat4 reproject;
    vec4 sizes; // render w, h, output w, h
    vec4 params; // jitter x, y (render pixels), reset
};
layout(set = 0, binding = 1) uniform texture2D color_tex;
layout(set = 0, binding = 2) uniform texture2D depth_tex;
layout(set = 0, binding = 3) uniform texture2D history_tex;
layout(set = 0, binding = 4) uniform sampler linear_sampler;
layout(location = 0) out vec4 out_history;

const float MAX_SAMPLES = 16.0;

vec3 to_ycocg(vec3 c) {
    return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

vec3 to_rgb(vec3 c) {
    return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

float lanczos2(float x) {
    x = abs(x);
    if (x >= 2.0) {
        return 0.0;
    }
    if (x < 1e-4) {
        return 1.0;
    }
    float px = 3.14159265 * x;
    return 2.0 * sin(px) * sin(px * 0.5) / (px * px);
}

vec4 history_at(vec2 uv) {
    return textureLod(sampler2D(history_tex, linear_sampler), uv, 0.0);
}

// Catmull-Rom in 5 bilinear taps (the corners' weights dropped).
vec4 history_catmull_rom(vec2 uv, vec2 size) {
    vec2 pos = uv * size;
    vec2 t1 = floor(pos - 0.5) + 0.5;
    vec2 f = pos - t1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0 = (t1 - 1.0) / size;
    vec2 t3 = (t1 + 2.0) / size;
    vec2 t12 = (t1 + w2 / w12) / size;
    float a = w12.x * w0.y, b = w0.x * w12.y, c = w12.x * w12.y, d = w3.x * w12.y, e = w12.x * w3.y;
    vec4 sum = history_at(vec2(t12.x, t0.y)) * a + history_at(vec2(t0.x, t12.y)) * b + history_at(t12) * c +
            history_at(vec2(t3.x, t12.y)) * d + history_at(vec2(t12.x, t3.y)) * e;
    return max(sum / (a + b + c + d + e), vec4(0.0));
}

void main() {
    vec2 render_size = sizes.xy;
    vec2 output_size = sizes.zw;
    vec2 uv = gl_FragCoord.xy / output_size;
    // Render pixel i holds the scene at i + 0.5 - jitter: the output pixel's centre in those coordinates.
    vec2 q = uv * render_size + params.xy;
    ivec2 last = ivec2(render_size) - 1;
    ivec2 base = ivec2(floor(q - 0.5));
    vec3 sum = vec3(0.0);
    float weight = 0.0;
    for (int y = -1; y <= 2; y++) {
        for (int x = -1; x <= 2; x++) {
            ivec2 t = base + ivec2(x, y);
            vec2 d = vec2(t) + 0.5 - q;
            float w = lanczos2(d.x) * lanczos2(d.y);
            sum += texelFetch(color_tex, clamp(t, ivec2(0), last), 0).rgb * w;
            weight += w;
        }
    }
    ivec2 nearest = ivec2(floor(q));
    vec3 lo = vec3(1e9);
    vec3 hi = vec3(-1e9);
    vec3 rgb_lo = vec3(1e9);
    vec3 rgb_hi = vec3(-1e9);
    float depth = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            ivec2 t = clamp(nearest + ivec2(x, y), ivec2(0), last);
            vec3 c = texelFetch(color_tex, t, 0).rgb;
            rgb_lo = min(rgb_lo, c);
            rgb_hi = max(rgb_hi, c);
            vec3 y_c = to_ycocg(c);
            lo = min(lo, y_c);
            hi = max(hi, y_c);
            depth = max(depth, texelFetch(depth_tex, t, 0).r); // reversed depth: the nearest is the largest
        }
    }
    // Lanczos rings: kept within the neighbourhood (FSR's deringing).
    vec3 current = clamp(sum / max(weight, 1e-4), rgb_lo, rgb_hi);
    vec2 dn = vec2(nearest) + 0.5 - q;
    float sample_weight = max(lanczos2(dn.x) * lanczos2(dn.y), 0.05); // how near this frame's sample is

    vec4 prev = reproject * vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    vec2 prev_uv = vec2(prev.x / prev.w * 0.5 + 0.5, 0.5 - prev.y / prev.w * 0.5);
    bool valid = params.z < 0.5 && prev.w > 0.0 && all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThanEqual(prev_uv, vec2(1.0)));
    float count = 0.0;
    vec3 history = to_ycocg(current);
    if (valid) {
        vec4 h = history_catmull_rom(prev_uv, output_size);
        vec3 hy = to_ycocg(h.rgb);
        history = clamp(hy, lo, hi);
        float rejected = clamp(length(hy - history) * 8.0, 0.0, 1.0);
        count = mix(min(h.a, MAX_SAMPLES), min(h.a, 1.0), rejected);
    }
    float alpha = sample_weight / (count + sample_weight);
    out_history = vec4(to_rgb(mix(history, to_ycocg(current), alpha)), min(count + sample_weight, MAX_SAMPLES));
}
)glsl";

// Each source pixel over a block x block square of the target, RCAS-sharpened (FSR 1's robust contrast-adaptive
// sharpening: the strongest negative lobe that keeps the 4 neighbours' cross within [0, 1]).
constexpr const char PRESENT_FRAGMENT[] = R"glsl(
layout(set = 0, binding = 0, std140) uniform Present {
    vec4 params; // pixel block, sharpness (negative: off), source w, h
};
layout(set = 0, binding = 1) uniform texture2D source_tex;
layout(location = 0) out vec4 out_color;

const float RCAS_LIMIT = 0.25 - 1.0 / 16.0;

vec3 source(ivec2 t) {
    return texelFetch(source_tex, clamp(t, ivec2(0), ivec2(params.zw) - 1), 0).rgb;
}

void main() {
    ivec2 t = ivec2(gl_FragCoord.xy / params.x);
    vec3 e = source(t);
    if (params.y >= 0.0) {
        vec3 b = source(t + ivec2(0, -1));
        vec3 d = source(t + ivec2(-1, 0));
        vec3 f = source(t + ivec2(1, 0));
        vec3 h = source(t + ivec2(0, 1));
        vec3 mn4 = min(min(b, d), min(f, h));
        vec3 mx4 = max(max(b, d), max(f, h));
        vec3 hit_min = min(mn4, e) / max(4.0 * mx4, vec3(1e-5));
        vec3 hit_max = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3(-1e-5));
        vec3 lobe3 = max(-hit_min, hit_max);
        float lobe = max(-RCAS_LIMIT, min(max(lobe3.r, max(lobe3.g, lobe3.b)), 0.0)) * exp2(-params.y);
        e = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
    }
    out_color = vec4(clamp(e, 0.0, 1.0), 1.0);
}
)glsl";

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

wgpu::Texture make_texture(const wgpu::Device &p_device, wgpu::TextureFormat p_format, int p_width, int p_height, wgpu::TextureUsage p_usage) {
    wgpu::TextureDescriptor desc{};
    desc.size = { uint32_t(p_width), uint32_t(p_height), 1 };
    desc.format = p_format;
    desc.usage = p_usage;
    return p_device.CreateTexture(&desc);
}

wgpu::BindGroupLayoutEntry uniform_entry(uint32_t p_binding) {
    wgpu::BindGroupLayoutEntry e{};
    e.binding = p_binding;
    e.visibility = wgpu::ShaderStage::Fragment;
    e.buffer.type = wgpu::BufferBindingType::Uniform;
    return e;
}

wgpu::BindGroupLayoutEntry texture_entry(uint32_t p_binding, wgpu::TextureSampleType p_type) {
    wgpu::BindGroupLayoutEntry e{};
    e.binding = p_binding;
    e.visibility = wgpu::ShaderStage::Fragment;
    e.texture.sampleType = p_type;
    e.texture.viewDimension = wgpu::TextureViewDimension::e2D;
    return e;
}

wgpu::BindGroupEntry buffer_bind(uint32_t p_binding, const wgpu::Buffer &p_buffer, uint64_t p_size) {
    wgpu::BindGroupEntry e{};
    e.binding = p_binding;
    e.buffer = p_buffer;
    e.size = p_size;
    return e;
}

wgpu::BindGroupEntry texture_bind(uint32_t p_binding, const wgpu::Texture &p_texture) {
    wgpu::BindGroupEntry e{};
    e.binding = p_binding;
    e.textureView = p_texture.CreateView();
    return e;
}

wgpu::BindGroup make_group(const wgpu::Device &p_device, const wgpu::BindGroupLayout &p_layout, const std::vector<wgpu::BindGroupEntry> &p_entries) {
    wgpu::BindGroupDescriptor desc{};
    desc.layout = p_layout;
    desc.entryCount = p_entries.size();
    desc.entries = p_entries.data();
    return p_device.CreateBindGroup(&desc);
}

wgpu::RenderPipeline make_pipeline(const wgpu::Device &p_device, const wgpu::BindGroupLayout &p_layout, const char *p_fragment,
        wgpu::TextureFormat p_format, const char *p_label) {
    const wgpu::ShaderModule vs = make_render_module(p_device, std::string(HEADER) + FULLSCREEN_VERTEX, ShaderStage::VERTEX);
    const wgpu::ShaderModule fs = make_render_module(p_device, std::string(HEADER) + p_fragment, ShaderStage::FRAGMENT);
    if (!vs || !fs) {
        return nullptr;
    }
    wgpu::PipelineLayoutDescriptor layout_desc{};
    layout_desc.bindGroupLayoutCount = 1;
    layout_desc.bindGroupLayouts = &p_layout;
    wgpu::ColorTargetState color{};
    color.format = p_format;
    wgpu::FragmentState fragment{};
    fragment.module = fs;
    fragment.entryPoint = "main";
    fragment.targetCount = 1;
    fragment.targets = &color;
    wgpu::RenderPipelineDescriptor desc{};
    desc.label = p_label;
    desc.layout = p_device.CreatePipelineLayout(&layout_desc);
    desc.vertex.module = vs;
    desc.vertex.entryPoint = "main";
    desc.fragment = &fragment;
    return p_device.CreateRenderPipeline(&desc);
}

void fullscreen_pass(const wgpu::CommandEncoder &p_encoder, const wgpu::TextureView &p_target, const wgpu::RenderPipeline &p_pipeline,
        const wgpu::BindGroup &p_group) {
    wgpu::RenderPassColorAttachment color{};
    color.view = p_target;
    color.loadOp = wgpu::LoadOp::Clear;
    color.storeOp = wgpu::StoreOp::Store;
    wgpu::RenderPassDescriptor desc{};
    desc.colorAttachmentCount = 1;
    desc.colorAttachments = &color;
    const wgpu::RenderPassEncoder pass = p_encoder.BeginRenderPass(&desc);
    pass.SetPipeline(p_pipeline);
    pass.SetBindGroup(0, p_group);
    pass.Draw(3);
    pass.End();
}

} // namespace

DawnUpscaler::DawnUpscaler(const wgpu::Device &p_device, wgpu::TextureFormat p_target_format, const Settings &p_settings) :
        device(p_device), settings(p_settings) {
    settings.pixelate = std::max(settings.pixelate, 1);
    settings.render_scale = std::clamp(settings.render_scale, 0.25f, 1.0f);
    make_pipelines(p_target_format);
}

void DawnUpscaler::make_pipelines(wgpu::TextureFormat p_target_format) {
    const wgpu::BindGroupLayoutEntry present_entries[] = { uniform_entry(0), texture_entry(1, wgpu::TextureSampleType::UnfilterableFloat) };
    wgpu::BindGroupLayoutDescriptor present_desc{};
    present_desc.entryCount = 2;
    present_desc.entries = present_entries;
    present_layout = device.CreateBindGroupLayout(&present_desc);
    present_pipeline = make_pipeline(device, present_layout, PRESENT_FRAGMENT, p_target_format, "upscaler present");
    wgpu::BufferDescriptor present_buffer{};
    present_buffer.size = sizeof(PresentUniforms);
    present_buffer.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    present_uniforms = device.CreateBuffer(&present_buffer);
    if (!settings.temporal) {
        return;
    }
    wgpu::BindGroupLayoutEntry sampler_entry{};
    sampler_entry.binding = 4;
    sampler_entry.visibility = wgpu::ShaderStage::Fragment;
    sampler_entry.sampler.type = wgpu::SamplerBindingType::Filtering;
    const wgpu::BindGroupLayoutEntry resolve_entries[] = { uniform_entry(0), texture_entry(1, wgpu::TextureSampleType::UnfilterableFloat),
        texture_entry(2, wgpu::TextureSampleType::UnfilterableFloat), texture_entry(3, wgpu::TextureSampleType::Float), sampler_entry };
    wgpu::BindGroupLayoutDescriptor resolve_desc{};
    resolve_desc.entryCount = 5;
    resolve_desc.entries = resolve_entries;
    resolve_layout = device.CreateBindGroupLayout(&resolve_desc);
    resolve_pipeline = make_pipeline(device, resolve_layout, RESOLVE_FRAGMENT, wgpu::TextureFormat::RGBA16Float, "upscaler resolve");
    wgpu::BufferDescriptor resolve_buffer{};
    resolve_buffer.size = sizeof(ResolveUniforms);
    resolve_buffer.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    resolve_uniforms = device.CreateBuffer(&resolve_buffer);
    wgpu::SamplerDescriptor sampler_desc{};
    sampler_desc.magFilter = wgpu::FilterMode::Linear;
    sampler_desc.minFilter = wgpu::FilterMode::Linear;
    linear = device.CreateSampler(&sampler_desc);
}

void DawnUpscaler::resize(int p_width, int p_height) {
    output_w = std::max((p_width + settings.pixelate - 1) / settings.pixelate, 1);
    output_h = std::max((p_height + settings.pixelate - 1) / settings.pixelate, 1);
    const float scale = settings.temporal ? settings.render_scale : 1.0f;
    render_w = std::max(int(std::lround(output_w * scale)), 1);
    render_h = std::max(int(std::lround(output_h * scale)), 1);
    color = make_texture(device, COLOR_FORMAT, render_w, render_h, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding);
    depth = make_texture(device, DEPTH_FORMAT, render_w, render_h, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding);
    reset = true;
    if (!ready()) {
        return;
    }
    if (!settings.temporal) {
        present_groups[0] = make_group(device, present_layout, { buffer_bind(0, present_uniforms, sizeof(PresentUniforms)), texture_bind(1, color) });
        return;
    }
    for (wgpu::Texture &h : history) {
        h = make_texture(device, wgpu::TextureFormat::RGBA16Float, output_w, output_h,
                wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding);
    }
    wgpu::BindGroupEntry sampler_bind{};
    sampler_bind.binding = 4;
    sampler_bind.sampler = linear;
    for (int i = 0; i < 2; i++) {
        // Writing history[i]: reading history[1 - i].
        resolve_groups[i] = make_group(device, resolve_layout,
                { buffer_bind(0, resolve_uniforms, sizeof(ResolveUniforms)), texture_bind(1, color), texture_bind(2, depth),
                        texture_bind(3, history[1 - i]), sampler_bind });
        present_groups[i] = make_group(device, present_layout, { buffer_bind(0, present_uniforms, sizeof(PresentUniforms)), texture_bind(1, history[i]) });
    }
}

void DawnUpscaler::begin_frame(const float p_view_proj[16], float r_jittered[16]) {
    std::memcpy(r_jittered, p_view_proj, sizeof(float) * 16);
    if (!settings.temporal) {
        return;
    }
    // FSR's phase count: 8 per output pixel per render pixel.
    const float ratio = float(output_w) / float(render_w);
    const int phases = std::max(int(std::ceil(8.0f * ratio * ratio)), 8);
    const int phase = frame % phases + 1;
    jitter[0] = halton(phase, 2) - 0.5f;
    jitter[1] = halton(phase, 3) - 0.5f;
    // The scene moves by the jitter in render pixels (y down): clip x += dx w, y += dy w.
    const float dx = 2.0f * jitter[0] / float(render_w);
    const float dy = -2.0f * jitter[1] / float(render_h);
    for (int c = 0; c < 4; c++) {
        r_jittered[c * 4 + 0] += dx * p_view_proj[c * 4 + 3];
        r_jittered[c * 4 + 1] += dy * p_view_proj[c * 4 + 3];
    }

    ResolveUniforms u{};
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
    for (int i = 0; i < 16; i++) {
        u.reproject[i] = float(reproject[i]);
        prev_view_proj[i] = current[i];
    }
    u.sizes[0] = float(render_w);
    u.sizes[1] = float(render_h);
    u.sizes[2] = float(output_w);
    u.sizes[3] = float(output_h);
    u.params[0] = jitter[0];
    u.params[1] = jitter[1];
    u.params[2] = reset ? 1.0f : 0.0f;
    device.GetQueue().WriteBuffer(resolve_uniforms, 0, &u, sizeof(u));
    reset = false;
}

void DawnUpscaler::encode(const wgpu::CommandEncoder &p_encoder, const wgpu::TextureView &p_target) {
    const int written = frame & 1;
    PresentUniforms p{};
    p.params[0] = float(settings.pixelate);
    p.params[1] = settings.temporal ? settings.sharpness : -1.0f;
    p.params[2] = float(settings.temporal ? output_w : render_w);
    p.params[3] = float(settings.temporal ? output_h : render_h);
    device.GetQueue().WriteBuffer(present_uniforms, 0, &p, sizeof(p));
    if (settings.temporal) {
        fullscreen_pass(p_encoder, history[written].CreateView(), resolve_pipeline, resolve_groups[written]);
        fullscreen_pass(p_encoder, p_target, present_pipeline, present_groups[written]);
    } else {
        fullscreen_pass(p_encoder, p_target, present_pipeline, present_groups[0]);
    }
    frame++;
}

} // namespace ov::dawn
