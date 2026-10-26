#include "renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ov::dawn {

namespace {

struct Mat4 {
    float m[16] = {};
};

Mat4 multiply(const Mat4 &p_a, const Mat4 &p_b) {
    Mat4 out;
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            float v = 0.0f;
            for (int k = 0; k < 4; k++) {
                v += p_a.m[k * 4 + r] * p_b.m[c * 4 + k];
            }
            out.m[c * 4 + r] = v;
        }
    }
    return out;
}

Mat4 perspective(float p_fov_y, float p_aspect, float p_near, float p_far) {
    Mat4 out;
    const float f = 1.0f / std::tan(p_fov_y * 0.5f);
    out.m[0] = f / p_aspect;
    out.m[5] = f;
    out.m[10] = p_far / (p_near - p_far);
    out.m[11] = -1.0f;
    out.m[14] = (p_near * p_far) / (p_near - p_far);
    return out;
}

Mat4 look_rotation(float p_yaw, float p_pitch) {
    const float cy = std::cos(p_yaw);
    const float sy = std::sin(p_yaw);
    const float cp = std::cos(p_pitch);
    const float sp = std::sin(p_pitch);
    // Forward, right and up in world space (y up).
    const float fx = sy * cp;
    const float fy = sp;
    const float fz = -cy * cp;
    float rx = cy;
    float ry = 0.0f;
    float rz = sy;
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;
    const float ul = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (ul > 1e-6f) {
        ux /= ul;
        uy /= ul;
        uz /= ul;
    }
    Mat4 out;
    out.m[0] = rx;
    out.m[1] = ux;
    out.m[2] = -fx;
    out.m[4] = ry;
    out.m[5] = uy;
    out.m[6] = -fy;
    out.m[8] = rz;
    out.m[9] = uz;
    out.m[10] = -fz;
    out.m[15] = 1.0f;
    return out;
}

const char *TERRAIN_WGSL = R"wgsl(
struct Camera {
    view_proj: mat4x4f,
    prev_view_proj: mat4x4f,
    light: vec4f,
    params: vec4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) world: vec3f,
};

struct FSOut {
    @location(0) color: vec4f,
    @location(1) motion: vec2f,
    @location(2) depth: f32,
};

@vertex
fn vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> VSOut {
    var out: VSOut;
    out.world = position;
    out.position = camera.view_proj * vec4f(position, 1.0);
    out.normal = normal;
    return out;
}

@fragment
fn fs(in: VSOut) -> FSOut {
    let n = normalize(in.normal);
    let l = normalize(camera.light.xyz);
    let diffuse = max(dot(n, l), 0.0);
    let sky = vec3f(0.42, 0.55, 0.75);
    let ground = vec3f(0.16, 0.13, 0.10);
    let ambient = mix(ground, sky, clamp(n.y * 0.5 + 0.5, 0.0, 1.0)) * 0.55;
    let albedo = vec3f(0.34, 0.42, 0.22) * (0.75 + 0.25 * clamp(n.y, 0.0, 1.0));
    let color = albedo * (ambient + vec3f(1.0, 0.96, 0.88) * diffuse * 0.95);
    let prev = camera.prev_view_proj * vec4f(in.world, 1.0);
    let cur = camera.view_proj * vec4f(in.world, 1.0);
    let prev_ndc = prev.xy / max(prev.w, 1e-6);
    let cur_ndc = cur.xy / max(cur.w, 1e-6);
    var out: FSOut;
    out.color = vec4f(color, 1.0);
    out.motion = (prev_ndc - cur_ndc) * 0.5 * camera.params.xy * vec2f(1.0, -1.0);
    out.depth = in.position.z;
    return out;
}
)wgsl";

const char *PRESENT_WGSL = R"wgsl(
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
struct Params { size: vec2f, block: f32, flip: f32 };
@group(0) @binding(2) var<uniform> params: Params;

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VSOut {
    var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var out: VSOut;
    out.position = vec4f(positions[index], 0.0, 1.0);
    out.uv = (positions[index] + vec2f(1.0)) * 0.5;
    return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
    var uv = in.uv;
    if (params.flip > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    if (params.block > 1.0) {
        uv = (floor(uv * params.size / params.block) + vec2f(0.5)) * params.block / params.size;
    }
    let c = textureSample(source, source_sampler, uv);
    return vec4f(c.rgb, 1.0);
}
)wgsl";

wgpu::ShaderModule shader(const wgpu::Device &p_device, const char *p_code, const char *p_label) {
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = p_code;
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &wgsl;
    desc.label = p_label;
    return p_device.CreateShaderModule(&desc);
}

} // namespace

bool Renderer::initialize(GpuContext &p_gpu, Terrain &p_terrain) {
    gpu = &p_gpu;
    terrain = &p_terrain;
    width = p_gpu.width;
    height = p_gpu.height;
    render_w = width;
    render_h = height;

    wgpu::BindGroupLayoutEntry camera_entry{};
    camera_entry.binding = 0;
    camera_entry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    camera_entry.buffer.type = wgpu::BufferBindingType::Uniform;
    wgpu::BindGroupLayoutDescriptor camera_layout_desc{};
    camera_layout_desc.entryCount = 1;
    camera_layout_desc.entries = &camera_entry;
    camera_layout = gpu->device.CreateBindGroupLayout(&camera_layout_desc);

    wgpu::BindGroupLayoutEntry present_entries[3] = {};
    present_entries[0].binding = 0;
    present_entries[0].visibility = wgpu::ShaderStage::Fragment;
    present_entries[0].texture.sampleType = wgpu::TextureSampleType::Float;
    present_entries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    present_entries[1].binding = 1;
    present_entries[1].visibility = wgpu::ShaderStage::Fragment;
    present_entries[1].sampler.type = wgpu::SamplerBindingType::Filtering;
    present_entries[2].binding = 2;
    present_entries[2].visibility = wgpu::ShaderStage::Fragment;
    present_entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
    wgpu::BindGroupLayoutDescriptor present_layout_desc{};
    present_layout_desc.entryCount = 3;
    present_layout_desc.entries = present_entries;
    present_layout = gpu->device.CreateBindGroupLayout(&present_layout_desc);

    wgpu::PipelineLayoutDescriptor terrain_layout_desc{};
    terrain_layout_desc.bindGroupLayoutCount = 1;
    terrain_layout_desc.bindGroupLayouts = &camera_layout;
    terrain_pipeline_layout = gpu->device.CreatePipelineLayout(&terrain_layout_desc);

    wgpu::PipelineLayoutDescriptor present_layout_desc2{};
    present_layout_desc2.bindGroupLayoutCount = 1;
    present_layout_desc2.bindGroupLayouts = &present_layout;
    present_pipeline_layout = gpu->device.CreatePipelineLayout(&present_layout_desc2);

    camera_buffer = create_buffer(gpu->device, 160, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "camera");
    wgpu::BindGroupEntry camera_binding{};
    camera_binding.binding = 0;
    camera_binding.buffer = camera_buffer;
    camera_binding.size = 160;
    wgpu::BindGroupDescriptor camera_bg_desc{};
    camera_bg_desc.layout = camera_layout;
    camera_bg_desc.entryCount = 1;
    camera_bg_desc.entries = &camera_binding;
    camera_bind_group = gpu->device.CreateBindGroup(&camera_bg_desc);

    present_params = create_buffer(gpu->device, 16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "present params");

    wgpu::SamplerDescriptor sampler_desc{};
    sampler_desc.minFilter = wgpu::FilterMode::Linear;
    sampler_desc.magFilter = wgpu::FilterMode::Linear;
    sampler_desc.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
    sampler_desc.addressModeU = wgpu::AddressMode::ClampToEdge;
    sampler_desc.addressModeV = wgpu::AddressMode::ClampToEdge;
    present_sampler = gpu->device.CreateSampler(&sampler_desc);

    create_offscreen();

    // Terrain pipeline.
    wgpu::ShaderModule terrain_module = shader(gpu->device, TERRAIN_WGSL, "terrain");
    wgpu::VertexAttribute attributes[2] = {};
    attributes[0].format = wgpu::VertexFormat::Float32x3;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;
    attributes[1].format = wgpu::VertexFormat::Float32x3;
    attributes[1].offset = 12;
    attributes[1].shaderLocation = 1;
    wgpu::VertexBufferLayout vertex_layout{};
    vertex_layout.arrayStride = 24;
    vertex_layout.attributeCount = 2;
    vertex_layout.attributes = attributes;
    wgpu::VertexState vertex_state{};
    vertex_state.module = terrain_module;
    vertex_state.entryPoint = "vs";
    vertex_state.bufferCount = 1;
    vertex_state.buffers = &vertex_layout;

    wgpu::ColorTargetState color_targets[3] = {};
    color_targets[0].format = wgpu::TextureFormat::RGBA16Float;
    color_targets[0].writeMask = wgpu::ColorWriteMask::All;
    color_targets[1].format = wgpu::TextureFormat::RG16Float;
    color_targets[1].writeMask = wgpu::ColorWriteMask::All;
    color_targets[2].format = wgpu::TextureFormat::R32Float;
    color_targets[2].writeMask = wgpu::ColorWriteMask::All;
    wgpu::FragmentState fragment_state{};
    fragment_state.module = terrain_module;
    fragment_state.entryPoint = "fs";
    fragment_state.targetCount = 3;
    fragment_state.targets = color_targets;

    wgpu::DepthStencilState depth_state{};
    depth_state.format = wgpu::TextureFormat::Depth32Float;
    depth_state.depthWriteEnabled = true;
    depth_state.depthCompare = wgpu::CompareFunction::Less;

    wgpu::RenderPipelineDescriptor terrain_desc{};
    terrain_desc.layout = terrain_pipeline_layout;
    terrain_desc.vertex = vertex_state;
    terrain_desc.fragment = &fragment_state;
    terrain_desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    terrain_desc.primitive.frontFace = wgpu::FrontFace::CCW;
    terrain_desc.primitive.cullMode = wgpu::CullMode::None;
    terrain_desc.depthStencil = &depth_state;
    terrain_pipeline = gpu->device.CreateRenderPipeline(&terrain_desc);

    // Present pipeline.
    wgpu::ShaderModule present_module = shader(gpu->device, PRESENT_WGSL, "present");
    wgpu::VertexState present_vertex{};
    present_vertex.module = present_module;
    present_vertex.entryPoint = "vs";
    present_vertex.bufferCount = 0;
    wgpu::ColorTargetState present_target{};
    present_target.format = gpu->format;
    present_target.writeMask = wgpu::ColorWriteMask::All;
    wgpu::FragmentState present_fragment{};
    present_fragment.module = present_module;
    present_fragment.entryPoint = "fs";
    present_fragment.targetCount = 1;
    present_fragment.targets = &present_target;
    wgpu::RenderPipelineDescriptor present_desc{};
    present_desc.layout = present_pipeline_layout;
    present_desc.vertex = present_vertex;
    present_desc.fragment = &present_fragment;
    present_desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    present_pipeline = gpu->device.CreateRenderPipeline(&present_desc);

    present_source = color_view;
    wgpu::BindGroupEntry present_bindings[3] = {};
    present_bindings[0].binding = 0;
    present_bindings[0].textureView = present_source;
    present_bindings[1].binding = 1;
    present_bindings[1].sampler = present_sampler;
    present_bindings[2].binding = 2;
    present_bindings[2].buffer = present_params;
    present_bindings[2].size = 16;
    wgpu::BindGroupDescriptor present_bg_desc{};
    present_bg_desc.layout = present_layout;
    present_bg_desc.entryCount = 3;
    present_bg_desc.entries = present_bindings;
    present_bind_group = gpu->device.CreateBindGroup(&present_bg_desc);

    if (!gpu->device.HasFeature(wgpu::FeatureName::TextureFormatsTier2)) {
        std::fprintf(stderr, "dawn: TextureFormatsTier2 missing: FSR3 disabled (bilinear upscale)\n");
        fsr3_failed = true;
    }
    return true;
}

void Renderer::create_offscreen() {
    color = create_texture(gpu->device, render_w, render_h, wgpu::TextureFormat::RGBA16Float, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding, 1, "color");
    color_view = color.CreateView();
    depth = create_texture(gpu->device, render_w, render_h, wgpu::TextureFormat::Depth32Float, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding, 1, "depth");
    depth_view = depth.CreateView();
    motion = create_texture(gpu->device, render_w, render_h, wgpu::TextureFormat::RG16Float, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding, 1, "motion");
    motion_view = motion.CreateView();
    depth_color = create_texture(gpu->device, render_w, render_h, wgpu::TextureFormat::R32Float, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding, 1, "depth color");
    depth_color_view = depth_color.CreateView();
}

void Renderer::resize(uint32_t p_width, uint32_t p_height) {
    if (p_width == width && p_height == height) {
        return;
    }
    width = p_width;
    height = p_height;
    gpu->resize(width, height);
    render_w = std::max(1u, width / 2);
    render_h = std::max(1u, height / 2);
    create_offscreen();
    fsr3.reset();
    fsr_output = {};
    fsr_output_view = {};
    present_source = color_view;
    wgpu::BindGroupEntry present_bindings[3] = {};
    present_bindings[0].binding = 0;
    present_bindings[0].textureView = present_source;
    present_bindings[1].binding = 1;
    present_bindings[1].sampler = present_sampler;
    present_bindings[2].binding = 2;
    present_bindings[2].buffer = present_params;
    present_bindings[2].size = 16;
    wgpu::BindGroupDescriptor desc{};
    desc.layout = present_layout;
    desc.entryCount = 3;
    desc.entries = present_bindings;
    present_bind_group = gpu->device.CreateBindGroup(&desc);
}

void Renderer::update_meshes() {
    for (const auto &entry : terrain->chunks()) {
        const Terrain::Chunk &chunk = entry.second;
        GpuMesh &mesh = meshes[entry.first];
        if (mesh.revision == chunk.revision && mesh.index_count > 0) {
            continue;
        }
        mesh.revision = chunk.revision;
        mesh.index_count = uint32_t(chunk.mesh.indices.size());
        if (mesh.index_count == 0) {
            mesh.vertices = {};
            mesh.indices = {};
            continue;
        }
        const uint64_t vertex_bytes = uint64_t(chunk.mesh.vertices.size()) * sizeof(MeshVertex);
        const uint64_t index_bytes = uint64_t(chunk.mesh.indices.size()) * sizeof(uint32_t);
        mesh.vertices = create_buffer(gpu->device, vertex_bytes, wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst, "chunk vertices");
        mesh.indices = create_buffer(gpu->device, index_bytes, wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst, "chunk indices");
        write_buffer(gpu->queue, mesh.vertices, chunk.mesh.vertices.data(), vertex_bytes);
        write_buffer(gpu->queue, mesh.indices, chunk.mesh.indices.data(), index_bytes);
    }
    for (auto it = meshes.begin(); it != meshes.end();) {
        if (terrain->chunks().find(it->first) == terrain->chunks().end()) {
            it = meshes.erase(it);
        } else {
            ++it;
        }
    }
}

void Renderer::update_camera(const Camera &p_camera) {
    const Mat4 view = look_rotation(p_camera.yaw, p_camera.pitch);
    const Mat4 proj = perspective(p_camera.fov_y, float(render_w) / float(std::max(render_h, 1u)), p_camera.near_m, p_camera.far_m);
    Mat4 view_proj = multiply(proj, view);
    // Translation: the view matrix's last column is -R * eye.
    const float tx = -(view.m[0] * p_camera.position.x + view.m[4] * p_camera.position.y + view.m[8] * p_camera.position.z);
    const float ty = -(view.m[1] * p_camera.position.x + view.m[5] * p_camera.position.y + view.m[9] * p_camera.position.z);
    const float tz = -(view.m[2] * p_camera.position.x + view.m[6] * p_camera.position.y + view.m[10] * p_camera.position.z);
    Mat4 translate;
    translate.m[0] = translate.m[5] = translate.m[10] = translate.m[15] = 1.0f;
    translate.m[12] = tx;
    translate.m[13] = ty;
    translate.m[14] = tz;
    view_proj = multiply(proj, multiply(view, translate));

    float data[40] = {};
    std::memcpy(data, view_proj.m, sizeof(float) * 16);
    std::memcpy(data + 16, has_previous ? previous_view_proj : view_proj.m, sizeof(float) * 16);
    data[32] = 0.35f;
    data[33] = 0.85f;
    data[34] = 0.45f;
    data[35] = 0.0f;
    data[36] = float(render_w);
    data[37] = float(render_h);
    data[38] = 0.0f;
    data[39] = 0.0f;
    write_buffer(gpu->queue, camera_buffer, data, sizeof(data));
    std::memcpy(previous_view_proj, view_proj.m, sizeof(float) * 16);
    has_previous = true;

    float params[4] = { float(width), float(height), 1.0f, 0.0f };
    write_buffer(gpu->queue, present_params, params, sizeof(params));
}

void Renderer::render(Camera &p_camera, bool p_upscale, bool p_fsr3, int p_pixelate) {
    const uint32_t wanted_w = p_upscale ? std::max(1u, width / 2) : width;
    const uint32_t wanted_h = p_upscale ? std::max(1u, height / 2) : height;
    if (wanted_w != render_w || wanted_h != render_h) {
        render_w = wanted_w;
        render_h = wanted_h;
        create_offscreen();
        fsr3.reset();
        fsr_output = {};
        fsr_output_view = {};
        present_source = color_view;
        wgpu::BindGroupEntry bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].textureView = present_source;
        bindings[1].binding = 1;
        bindings[1].sampler = present_sampler;
        bindings[2].binding = 2;
        bindings[2].buffer = present_params;
        bindings[2].size = 16;
        wgpu::BindGroupDescriptor desc{};
        desc.layout = present_layout;
        desc.entryCount = 3;
        desc.entries = bindings;
        present_bind_group = gpu->device.CreateBindGroup(&desc);
    }
    update_meshes();
    update_camera(p_camera);

    wgpu::SurfaceTexture surface_texture = gpu->acquire();
    if (!surface_texture.texture) {
        return;
    }
    wgpu::TextureView surface_view = surface_texture.texture.CreateView();

    wgpu::CommandEncoder encoder = gpu->device.CreateCommandEncoder();
    wgpu::RenderPassColorAttachment color_attachment{};
    color_attachment.view = color_view;
    color_attachment.loadOp = wgpu::LoadOp::Clear;
    color_attachment.storeOp = wgpu::StoreOp::Store;
    color_attachment.clearValue = { 0.52, 0.62, 0.78, 1.0 };
    wgpu::RenderPassColorAttachment motion_attachment{};
    motion_attachment.view = motion_view;
    motion_attachment.loadOp = wgpu::LoadOp::Clear;
    motion_attachment.storeOp = wgpu::StoreOp::Store;
    motion_attachment.clearValue = { 0.0, 0.0, 0.0, 0.0 };
    wgpu::RenderPassColorAttachment depth_color_attachment{};
    depth_color_attachment.view = depth_color_view;
    depth_color_attachment.loadOp = wgpu::LoadOp::Clear;
    depth_color_attachment.storeOp = wgpu::StoreOp::Store;
    depth_color_attachment.clearValue = { 1.0, 0.0, 0.0, 0.0 };
    wgpu::RenderPassDepthStencilAttachment depth_attachment{};
    depth_attachment.view = depth_view;
    depth_attachment.depthLoadOp = wgpu::LoadOp::Clear;
    depth_attachment.depthStoreOp = wgpu::StoreOp::Store;
    depth_attachment.depthClearValue = 1.0f;
    wgpu::RenderPassDescriptor pass_desc{};
    wgpu::RenderPassColorAttachment attachments[3] = { color_attachment, motion_attachment, depth_color_attachment };
    pass_desc.colorAttachmentCount = 3;
    pass_desc.colorAttachments = attachments;
    pass_desc.depthStencilAttachment = &depth_attachment;

    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&pass_desc);
    pass.SetPipeline(terrain_pipeline);
    pass.SetBindGroup(0, camera_bind_group);
    for (const auto &entry : meshes) {
        if (entry.second.index_count == 0) {
            continue;
        }
        pass.SetVertexBuffer(0, entry.second.vertices);
        pass.SetIndexBuffer(entry.second.indices, wgpu::IndexFormat::Uint32);
        pass.DrawIndexed(entry.second.index_count, 1, 0, 0, 0);
    }
    pass.End();

    bool fsr_ok = false;
    if (p_upscale && p_fsr3 && !fsr3_failed && !fsr3) {
        fsr3 = std::make_unique<Fsr3Upscaler>(gpu->device, gpu->instance, render_w, render_h, width, height);
    }
    if (p_upscale && p_fsr3 && fsr3 && !fsr3_failed) {
        if (!fsr_output) {
            fsr_output = create_texture(gpu->device, width, height, wgpu::TextureFormat::RGBA16Float,
                    wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding, 1, "fsr3 output");
            fsr_output_view = fsr_output.CreateView();
        }
        Fsr3Inputs inputs{};
        inputs.color = color_view;
        inputs.depth = depth_color_view;
        inputs.motion_vectors = motion_view;
        inputs.output = fsr_output_view;
        inputs.frame_time_ms = 16.0f;
        fsr_ok = fsr3->Dispatch(encoder, inputs);
        fsr_last_ok = fsr_ok;
        if (fsr_ok) {
            present_source = fsr3->OutputView();
        }
    }
    if (!fsr_ok) {
        present_source = color_view;
    }
    {
        // The offscreen render target is stored top-left; the FSR3 output is already in present order.
        float params[4] = { float(width), float(height), float(std::max(p_pixelate, 1)), fsr_ok ? 0.0f : 1.0f };
        write_buffer(gpu->queue, present_params, params, sizeof(params));
    }
    // The present bind group follows the source (color or the FSR3 output).
    {
        wgpu::BindGroupEntry bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].textureView = present_source;
        bindings[1].binding = 1;
        bindings[1].sampler = present_sampler;
        bindings[2].binding = 2;
        bindings[2].buffer = present_params;
        bindings[2].size = 16;
        wgpu::BindGroupDescriptor desc{};
        desc.layout = present_layout;
        desc.entryCount = 3;
        desc.entries = bindings;
        present_bind_group = gpu->device.CreateBindGroup(&desc);
    }

    wgpu::RenderPassColorAttachment present_attachment{};
    present_attachment.view = surface_view;
    present_attachment.loadOp = wgpu::LoadOp::Clear;
    present_attachment.storeOp = wgpu::StoreOp::Store;
    present_attachment.clearValue = { 0.0, 0.0, 0.0, 1.0 };
    wgpu::RenderPassDescriptor present_desc{};
    present_desc.colorAttachmentCount = 1;
    present_desc.colorAttachments = &present_attachment;
    wgpu::RenderPassEncoder present_pass = encoder.BeginRenderPass(&present_desc);
    present_pass.SetPipeline(present_pipeline);
    present_pass.SetBindGroup(0, present_bind_group);
    present_pass.Draw(3, 1, 0, 0);
    present_pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    gpu->queue.Submit(1, &commands);
    gpu->present(surface_texture);
}

void Renderer::shutdown() {
    meshes.clear();
}

} // namespace ov::dawn
