#include "dawn_page_host.hpp"

#include "page_format.hpp"
#include "quad_page_pool.hpp"

#include "ov/log.hpp"
#include "ov/on_error.hpp"

#include <cstring>

namespace ov::dawn {

namespace {

constexpr const char FRAME_BLOCK[] = R"glsl(
layout(set = 0, binding = 0, std140) uniform Frame {
    mat4 view_proj;
    ivec4 origin_chunk; // w: chunk_size
    vec4 sun;
    vec4 camera; // w: fog distance (0 none)
    ivec4 ints; // skirt_size, debug_view
    vec4 floats; // pos_scale, voxel_size
};
)glsl";

// Corners decoded like Godot's page shader (godot_page_host.cpp) through the same PageFormat functions; a quad whose 4
// corners lie in hidden coverage sub-cubes collapses to a point.
constexpr const char VERTEX_SHADER[] = R"glsl(
layout(set = 1, binding = 0) uniform utexture2D quad_tex;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_world;
layout(location = 2) flat out int v_lod;

void main() {
    int vertex = gl_VertexIndex;
    int group = gl_InstanceIndex;
    int chunk_size = origin_chunk.w;
    int skirt_size = ints.x;
    float pos_scale = floats.x;
    float voxel_size = floats.y;
    uvec4 h0 = texelFetch(quad_tex, ivec2(0, group), 0);
    uvec4 h1 = texelFetch(quad_tex, ivec2(1, group), 0);
    uvec4 c = texelFetch(quad_tex, ivec2(int(HEADER_TEXELS) + vertex, group), 0);
    int lod = page_chunk_lod(page_header_chunk(h0, h1));
    float voxel_dim = float(chunk_size - 2 * skirt_size);
    vec3 chunk_min = page_chunk_min(page_header_chunk(h0, h1) - origin_chunk.xyz, lod, voxel_dim, voxel_size);
    vec3 world = page_corner_world(chunk_min, lod, page_decode_local(c, pos_scale), skirt_size, voxel_dim, voxel_size);
    if (page_header_has_coverage(h1)) {
        uvec4 h2 = texelFetch(quad_tex, ivec2(int(PAGE_COVERAGE_TEXEL), group), 0);
        int first = int(HEADER_TEXELS) + (vertex & ~3);
        bool hidden = true;
        for (int k = 0; k < 4; k++) {
            vec3 lk = page_decode_local(texelFetch(quad_tex, ivec2(first + k, group), 0), pos_scale);
            hidden = hidden && page_subcube_hidden(h2, page_corner_subcube(lk, skirt_size, voxel_dim));
        }
        if (hidden) {
            world = chunk_min;
        }
    }
    v_normal = c != uvec4(0u) ? page_decode_normal(c) : vec3(0.0, 1.0, 0.0);
    v_world = world;
    v_lod = lod;
    gl_Position = view_proj * vec4(world, 1.0);
}
)glsl";

// 0 shaded (Godot's ground colours: flat on gentle slopes, steep on the others), 1 normals, 2 LOD.
constexpr const char FRAGMENT_SHADER[] = R"glsl(
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world;
layout(location = 2) flat in int v_lod;
layout(location = 0) out vec4 out_color;

const vec3 SKY = vec3(0.55, 0.68, 0.85);

void main() {
    vec3 n = normalize(v_normal);
    vec3 color;
    if (ints.y == 1) {
        color = n * 0.5 + 0.5;
    } else {
        vec3 albedo = mix(GROUND_STEEP, GROUND_FLAT, smoothstep(0.8, 0.95, clamp(n.y, 0.0, 1.0)));
        if (ints.y == 2) {
            albedo = vec3(fract(float(v_lod) * 0.37), fract(float(v_lod) * 0.61 + 0.3), fract(float(v_lod) * 0.13 + 0.6));
        }
        float sun_light = max(dot(n, sun.xyz), 0.0);
        color = albedo * (sun_light * vec3(1.0, 0.95, 0.85) + SKY * (0.25 + 0.1 * n.y));
        if (camera.w > 0.0) {
            float d = length(v_world - camera.xyz);
            color = mix(color, SKY, 1.0 - exp2(-d / camera.w));
        }
    }
    out_color = vec4(pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
)glsl";

wgpu::ShaderModule make_module(const wgpu::Device &p_device, const std::string &p_glsl, ShaderStage p_stage) {
    const CompiledShader shader = compile_glsl(p_glsl, p_stage);
    OV_ERR_FAIL_COND_V_MSG(!shader.error.empty(), wgpu::ShaderModule(), "DawnPageHost: page shader: " + shader.error);
    wgpu::ShaderSourceSPIRV spirv{};
    spirv.codeSize = uint32_t(shader.spirv.size());
    spirv.code = shader.spirv.data();
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &spirv;
    return p_device.CreateShaderModule(&desc);
}

} // namespace

DawnPageHost::DawnPageHost(DawnDevice &p_device, wgpu::TextureFormat p_color_format) :
        device(p_device) {
    const wgpu::Device &dev = device.get_context().device;
    const std::string header = "#version 450\n#extension GL_EXT_samplerless_texture_functions : require\n";
    const wgpu::ShaderModule vs = make_module(dev, header + PageFormat::glsl_decode() + FRAME_BLOCK + VERTEX_SHADER, ShaderStage::VERTEX);
    const wgpu::ShaderModule fs = make_module(dev, header + Material::glsl_constants() + FRAME_BLOCK + FRAGMENT_SHADER, ShaderStage::FRAGMENT);
    if (!vs || !fs) {
        return;
    }

    wgpu::BindGroupLayoutEntry frame_entry{};
    frame_entry.binding = 0;
    frame_entry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    frame_entry.buffer.type = wgpu::BufferBindingType::Uniform;
    wgpu::BindGroupLayoutDescriptor frame_layout_desc{};
    frame_layout_desc.entryCount = 1;
    frame_layout_desc.entries = &frame_entry;
    const wgpu::BindGroupLayout frame_layout = dev.CreateBindGroupLayout(&frame_layout_desc);

    wgpu::BindGroupLayoutEntry page_entry{};
    page_entry.binding = 0;
    page_entry.visibility = wgpu::ShaderStage::Vertex;
    page_entry.texture.sampleType = wgpu::TextureSampleType::Uint;
    page_entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
    wgpu::BindGroupLayoutDescriptor page_layout_desc{};
    page_layout_desc.entryCount = 1;
    page_layout_desc.entries = &page_entry;
    page_layout = dev.CreateBindGroupLayout(&page_layout_desc);

    const wgpu::BindGroupLayout layouts[] = { frame_layout, page_layout };
    wgpu::PipelineLayoutDescriptor layout_desc{};
    layout_desc.bindGroupLayoutCount = 2;
    layout_desc.bindGroupLayouts = layouts;

    wgpu::ColorTargetState color{};
    color.format = p_color_format;
    wgpu::FragmentState fragment{};
    fragment.module = fs;
    fragment.entryPoint = "main";
    fragment.targetCount = 1;
    fragment.targets = &color;
    wgpu::DepthStencilState depth{};
    depth.format = DEPTH_FORMAT;
    depth.depthWriteEnabled = true;
    depth.depthCompare = wgpu::CompareFunction::Greater;
    wgpu::RenderPipelineDescriptor desc{};
    desc.label = "page shader";
    desc.layout = dev.CreatePipelineLayout(&layout_desc);
    desc.vertex.module = vs;
    desc.vertex.entryPoint = "main";
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.frontFace = wgpu::FrontFace::CW; // Godot's: the meshers' winding
    desc.primitive.cullMode = wgpu::CullMode::Back;
    desc.depthStencil = &depth;
    desc.fragment = &fragment;
    pipeline = dev.CreateRenderPipeline(&desc);

    wgpu::BufferDescriptor uniforms_desc{};
    uniforms_desc.size = sizeof(FrameUniforms);
    uniforms_desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniforms = dev.CreateBuffer(&uniforms_desc);
    wgpu::BindGroupEntry frame_bind{};
    frame_bind.binding = 0;
    frame_bind.buffer = uniforms;
    frame_bind.size = sizeof(FrameUniforms);
    wgpu::BindGroupDescriptor frame_group_desc{};
    frame_group_desc.layout = frame_layout;
    frame_group_desc.entryCount = 1;
    frame_group_desc.entries = &frame_bind;
    frame_group = dev.CreateBindGroup(&frame_group_desc);

    std::vector<uint32_t> index_data;
    for (uint32_t q = 0; q < QuadPagePool::QUAD_GROUP; q++) {
        for (uint32_t i : { 0u, 1u, 3u, 0u, 3u, 2u }) {
            index_data.push_back(q * 4 + i);
        }
    }
    wgpu::BufferDescriptor index_desc{};
    index_desc.size = index_data.size() * sizeof(uint32_t);
    index_desc.usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;
    indices = dev.CreateBuffer(&index_desc);
    device.get_context().queue.WriteBuffer(indices, 0, index_data.data(), index_desc.size);
}

void DawnPageHost::draw(const wgpu::RenderPassEncoder &p_pass, const PageView &p_view) {
    last_drawn = 0;
    if (!pipeline) {
        return;
    }
    FrameUniforms u{};
    std::memcpy(u.view_proj, p_view.view_proj, sizeof(u.view_proj));
    u.origin_chunk[0] = decode.origin_chunk_lo.x;
    u.origin_chunk[1] = decode.origin_chunk_lo.y;
    u.origin_chunk[2] = decode.origin_chunk_lo.z;
    u.origin_chunk[3] = decode.chunk_size;
    u.sun[0] = p_view.sun.x;
    u.sun[1] = p_view.sun.y;
    u.sun[2] = p_view.sun.z;
    u.camera[0] = p_view.camera.x;
    u.camera[1] = p_view.camera.y;
    u.camera[2] = p_view.camera.z;
    u.camera[3] = p_view.fog_m;
    u.ints[0] = decode.skirt_size;
    u.ints[1] = debug_view;
    u.floats[0] = decode.pos_scale;
    u.floats[1] = decode.voxel_size;
    device.get_context().queue.WriteBuffer(uniforms, 0, &u, sizeof(u));

    p_pass.SetPipeline(pipeline);
    p_pass.SetBindGroup(0, frame_group);
    p_pass.SetIndexBuffer(indices, wgpu::IndexFormat::Uint32);
    // Every visible page: the instance count is the GPU's (0 draws nothing). No frustum culling yet (the bounds are
    // kept for it).
    for (const Page &page : pages) {
        if (!page.visible || !page.group) {
            continue;
        }
        p_pass.SetBindGroup(1, page.group);
        p_pass.DrawIndexedIndirect(page.command_buffer, 0);
        last_drawn++;
    }
}

void DawnPageHost::create_page(int p_page, uint32_t p_rows) {
    (void)p_rows; // the texture carries them
    if (int(pages.size()) <= p_page) {
        pages.resize(size_t(p_page) + 1);
    }
    pages[p_page] = Page();
}

void DawnPageHost::free_page(int p_page) {
    pages[p_page].visible = false;
}

void DawnPageHost::set_page_visible(int p_page, bool p_visible) {
    pages[p_page].visible = p_visible;
}

void DawnPageHost::set_page_bounds(int p_page, const RenderBox &p_bounds) {
    pages[p_page].bounds = p_bounds;
}

void DawnPageHost::set_decode(const PageDecode &p_decode) {
    decode = p_decode;
}

void DawnPageHost::set_materials(const Material::Library::Image &p_image) {
    (void)p_image; // TODO: the materials' colours (sample the image like Godot's material_tex) once the palette is bound
}

void DawnPageHost::set_debug_view(int p_view) {
    debug_view = p_view;
}

gpu::Buffer DawnPageHost::rt_attach_page(int p_page, gpu::Texture p_texture) {
    Page &page = pages[p_page];
    // VkDrawIndexedIndirectCommand: index count, instance count (the core's), first index, base vertex, first instance.
    const uint32_t command[5] = { QuadPagePool::QUAD_GROUP * 6, 0, 0, 0, 0 };
    page.command = device.create_indirect_buffer(sizeof(command), command);
    page.command_buffer = device.get_buffer(page.command);
    const wgpu::Texture texture = device.get_texture(p_texture);
    OV_ERR_FAIL_COND_V_MSG(!texture, page.command, "DawnPageHost: attach of an unknown texture.");
    wgpu::BindGroupEntry entry{};
    entry.binding = 0;
    entry.textureView = texture.CreateView();
    wgpu::BindGroupDescriptor desc{};
    desc.layout = page_layout;
    desc.entryCount = 1;
    desc.entries = &entry;
    page.group = device.get_context().device.CreateBindGroup(&desc);
    return page.command;
}

void DawnPageHost::rt_detach_page(int p_page) {
    Page &page = pages[p_page];
    device.free(page.command);
    page = Page();
}

void DawnPageHost::rt_set_palette_texture(gpu::Texture p_texture) {
    (void)p_texture; // TODO: with set_materials
}

} // namespace ov::dawn
