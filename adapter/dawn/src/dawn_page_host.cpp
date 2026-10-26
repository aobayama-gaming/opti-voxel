#include "dawn_page_host.hpp"

#include "dawn_look_glsl.hpp"
#include "shader_module.hpp"
#include "wgpu_util.hpp"

#include "page_format.hpp"
#include "page_shading_glsl.hpp"
#include "quad_page_pool.hpp"
#include "sky_glsl.hpp"

#include "ov/log.hpp"
#include "ov/on_error.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace ov::dawn {

namespace {

// The sun's shadow map: SHADOW_SIZE texels a side over a square of PageView::shadow_m around the camera, SHADOW_DEPTH_M
// metres deep along the sun on each side of it (what lies farther towards the sun casts no shadow).
constexpr uint32_t SHADOW_SIZE = 4096;
constexpr float SHADOW_DEPTH_M = 2000.0f;

constexpr const char FRAME_BLOCK[] = R"glsl(
layout(set = 0, binding = 0, std140) uniform Frame {
    mat4 view_proj;
    mat4 shadow_proj; // render metres to the shadow map's clip space
    ivec4 origin_chunk; // w: chunk_size
    vec4 sun; // xyz: towards the sun; w: its disk's size (radians)
    vec4 sun_color; // rgb; w: energy
    vec4 camera; // w: fog distance (0 none)
    vec4 forward; // the camera's; w: metres per pixel per metre of depth
    vec4 view_right; // the camera's right, times the tangent of half the horizontal field of view
    vec4 view_up; // its up, times the tangent of half the vertical one
    vec4 shadow; // x: 1 on; y: normal offset (metres); z: a texel in uv
    vec4 look; // x: 1 graded (a LUT); y: the sky light's energy; z: the sky's exposure
    ivec4 ints; // skirt_size, debug_view
    vec4 floats; // pos_scale, voxel_size, time (seconds)
};
)glsl";

// The page, the chunks' palettes and the materials' parameters (page_shading_glsl.hpp).
constexpr const char PAGE_TEXTURES[] = R"glsl(
layout(set = 1, binding = 0) uniform utexture2D quad_tex;
layout(set = 2, binding = 0) uniform utexture2D palette_tex;
layout(set = 2, binding = 1) uniform texture2D material_tex;
)glsl";

// Corners decoded like Godot's page shader (godot_page_host.cpp) through the same PageFormat functions; a quad whose 4
// corners lie in hidden coverage sub-cubes collapses to a point. The quad's materials as Godot's: from corner A.
constexpr const char VERTEX_SHADER[] = R"glsl(
layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_world;
layout(location = 2) flat out int v_lod;
layout(location = 3) out vec2 v_quad_uv;
layout(location = 4) flat out ivec2 v_materials;

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
    int corner = vertex % 4; // base quad corners A, B, C, D
    v_materials = ivec2(0);
    if (corner == 0) {
        v_materials = quad_materials(h1, page_decode_palette(c), int(HEADER_TEXELS) + vertex, group);
    }
    v_normal = c != uvec4(0u) ? page_decode_normal(c) : vec3(0.0, 1.0, 0.0);
    v_world = world;
    v_lod = lod;
    v_quad_uv = vec2(float(corner & 1), float(corner >> 1));
    gl_Position = view_proj * vec4(world, 1.0);
}
)glsl";

// 0 shaded (Godot's ground: flat on gentle slopes, steep on the others, its surface detail, no contour lines; the
// materials through page_shading_glsl.hpp, still air; the sun with its shadow, the sky's light, the fog, then
// dawn_look_glsl.hpp's tonemapper and grade), 1 normals, 2 LOD. A flat world: up is +y.
constexpr const char FRAGMENT_SHADER[] = R"glsl(
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world;
layout(location = 2) flat in int v_lod;
layout(location = 3) in vec2 v_quad_uv;
layout(location = 4) flat in ivec2 v_materials;
layout(location = 0) out vec4 out_color;

const vec3 UP = vec3(0.0, 1.0, 0.0);

void main() {
    vec3 n = normalize(v_normal);
    if (ints.y == 1) {
        out_color = vec4(n * 0.5 + 0.5, 1.0);
        return;
    }
    vec3 albedo;
    float roughness;
    vec3 emission = vec3(0.0);
    if (ints.y == 2) {
        albedo = vec3(fract(float(v_lod) * 0.37), fract(float(v_lod) * 0.61 + 0.3), fract(float(v_lod) * 0.13 + 0.6));
    } else {
        float flat_w = ground_flat_weight(v_normal, UP); // the interpolated normal, as Godot's
        float footprint = max(dot(v_world - camera.xyz, forward.xyz), 0.0) * forward.w;
        uint mat_id = fragment_material(v_materials, v_quad_uv, v_world, floats.y * exp2(float(v_lod)));
        // The ground's detail only where it shows: the ground's and a biome ground's (bark and leaves have their own).
        vec3 shade = vec3(1.0);
        if (mat_id == MATERIAL_GROUND || material_pattern(mat_id) == MATERIAL_PATTERN_GROUND) {
            shade = ground_detail(v_world, footprint);
        }
        vec3 ground = mix(GROUND_STEEP, GROUND_FLAT, flat_w) * shade;
        material_shading(mat_id, ground, v_world, v_normal, UP, footprint, flat_w, shade, floats.z, 0.0, albedo, roughness, emission);
    }
    float n_dot_l = max(dot(n, sun.xyz), 0.0);
    float sun_light = n_dot_l > 0.0 ? n_dot_l * sun_shadow(v_world, n) : 0.0;
    // The sky's light: the haze's colour (the fog's light), less on the sides and under (no occlusion term).
    vec3 sky_light = FOG_LIGHT_COLOR * look.y * (0.6 + 0.4 * n.y);
    vec3 color = albedo * (sun_light * sun_color.rgb * sun_color.w + sky_light) + emission;
    if (camera.w > 0.0) {
        vec3 to_point = v_world - camera.xyz;
        float d = length(to_point);
        vec3 dir = to_point / max(d, 1e-6);
        color = mix(color, fog_color(dir, sky_color(dir)), 1.0 - exp2(-d / camera.w));
    }
    out_color = vec4(display_color(color), 1.0);
}
)glsl";

} // namespace

wgpu::ShaderModule make_render_module(const wgpu::Device &p_device, const std::string &p_glsl, ShaderStage p_stage) {
    const CompiledShader shader = compile_glsl(p_glsl, p_stage);
    OV_ERR_FAIL_COND_V_MSG(!shader.error.empty(), wgpu::ShaderModule(), "Dawn: shader: " + shader.error);
    return create_spirv_module(p_device, shader.spirv.data(), shader.spirv.size(), "render");
}

namespace {

wgpu::PipelineLayout make_pipeline_layout(const wgpu::Device &p_device, const std::vector<wgpu::BindGroupLayout> &p_layouts) {
    wgpu::PipelineLayoutDescriptor desc{};
    desc.bindGroupLayoutCount = p_layouts.size();
    desc.bindGroupLayouts = p_layouts.data();
    return p_device.CreatePipelineLayout(&desc);
}

// The shadow map's projection for p_view (column-major, render metres to clip space): an orthographic view along -sun
// of a square shadow_m wide centred a quarter of it ahead of the camera (where most of the view lies), depth 0 at
// SHADOW_DEPTH_M towards the sun, 1 as far the other way. The square moves by whole texels, so the shadows' edges don't
// crawl as the camera moves.
void shadow_projection(const PageView &p_view, float r_m[16]) {
    const float half_m = p_view.shadow_m * 0.5f;
    const Float3 center{ p_view.camera.x + p_view.forward.x * half_m * 0.5f, p_view.camera.y, p_view.camera.z + p_view.forward.z * half_m * 0.5f };
    const Float3 &sun = p_view.sun;
    const Float3 side = std::abs(sun.y) < 0.99f ? Float3{ 0.0f, 1.0f, 0.0f } : Float3{ 1.0f, 0.0f, 0.0f };
    const Float3 r = cross(sun, side).normalized();
    const Float3 u = cross(r, sun);
    const float texel = p_view.shadow_m / float(SHADOW_SIZE);
    const float a = std::floor(dot(r, center) / texel) * texel;
    const float b = std::floor(dot(u, center) / texel) * texel;
    const float k = 1.0f / (2.0f * SHADOW_DEPTH_M);
    const float rows[3][4] = { { r.x / half_m, r.y / half_m, r.z / half_m, -a / half_m }, { u.x / half_m, u.y / half_m, u.z / half_m, -b / half_m },
        { -sun.x * k, -sun.y * k, -sun.z * k, 0.5f + dot(sun, center) * k } };
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 3; row++) {
            r_m[c * 4 + row] = rows[row][c];
        }
        r_m[c * 4 + 3] = c == 3 ? 1.0f : 0.0f;
    }
}

// Whether p_box can cast into the shadow map of projection p_m (a square 2 p_half_m wide): its bounding sphere against
// the square's sides, along the sun everything within the depth range counts.
bool in_shadow_map(const RenderBox &p_box, const float p_m[16], float p_half_m) {
    const Float3 c{ p_box.position.x + p_box.size.x * 0.5f, p_box.position.y + p_box.size.y * 0.5f, p_box.position.z + p_box.size.z * 0.5f };
    const float radius = 0.5f * std::sqrt(dot(p_box.size, p_box.size)) / p_half_m;
    const float x = p_m[0] * c.x + p_m[4] * c.y + p_m[8] * c.z + p_m[12];
    const float y = p_m[1] * c.x + p_m[5] * c.y + p_m[9] * c.z + p_m[13];
    return std::abs(x) <= 1.0f + radius && std::abs(y) <= 1.0f + radius;
}

} // namespace

DawnPageHost::DawnPageHost(DawnDevice &p_device, wgpu::TextureFormat p_color_format) :
        device(p_device) {
    const wgpu::Device &dev = device.get_context().device;
    const std::string common = std::string(GLSL_HEADER) + PageFormat::glsl_decode() + Material::glsl_constants() + FRAME_BLOCK + PAGE_TEXTURES + PageShading::glsl();
    const std::string look = SkyGLSL::glsl() + LookGLSL::glsl();
    const wgpu::ShaderModule vs = make_render_module(dev, common + VERTEX_SHADER, ShaderStage::VERTEX);
    const wgpu::ShaderModule fs = make_render_module(dev, common + LookGLSL::declarations(3) + look + FRAGMENT_SHADER, ShaderStage::FRAGMENT);
    const wgpu::ShaderModule sky_vs = make_render_module(dev, std::string(GLSL_HEADER) + FULLSCREEN_VERTEX, ShaderStage::VERTEX);
    const wgpu::ShaderModule sky_fs = make_render_module(dev, GLSL_HEADER + (FRAME_BLOCK + LookGLSL::declarations(1)) + look + LookGLSL::SKY_FRAGMENT, ShaderStage::FRAGMENT);
    if (!vs || !fs || !sky_vs || !sky_fs) {
        return;
    }

    wgpu::BindGroupLayoutEntry frame_entry{};
    frame_entry.binding = 0;
    frame_entry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    frame_entry.buffer.type = wgpu::BufferBindingType::Uniform;
    const wgpu::BindGroupLayout frame_layout = make_layout(dev, { frame_entry });

    wgpu::BindGroupLayoutEntry page_entry{};
    page_entry.binding = 0;
    page_entry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment; // the shared text declares it in both stages
    page_entry.texture.sampleType = wgpu::TextureSampleType::Uint;
    page_entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
    page_layout = make_layout(dev, { page_entry });

    std::vector<wgpu::BindGroupLayoutEntry> material_entries(2);
    for (int i = 0; i < 2; i++) {
        material_entries[i].binding = uint32_t(i);
        material_entries[i].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        material_entries[i].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    }
    material_entries[0].texture.sampleType = wgpu::TextureSampleType::Uint;
    material_entries[1].texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
    material_layout = make_layout(dev, material_entries);
    // Until the mesher hands its own: one texel each (no palette, no material: the ground everywhere).
    palette_placeholder = make_texture(wgpu::TextureFormat::RGBA16Uint, 1, 1);
    palette_texture = palette_placeholder;
    material_texture = make_texture(wgpu::TextureFormat::RGBA32Float, 1, 1);
    _update_material_group();

    // The look's set (dawn_look_glsl.hpp): the shadow map, its comparison sampler, the grading cube and its sampler.
    // One texel of each until the first shadow pass and set_grade.
    std::vector<wgpu::BindGroupLayoutEntry> look_entries(4);
    for (int i = 0; i < 4; i++) {
        look_entries[i].binding = uint32_t(i);
        look_entries[i].visibility = wgpu::ShaderStage::Fragment;
    }
    look_entries[0].texture.sampleType = wgpu::TextureSampleType::Depth;
    look_entries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    look_entries[1].sampler.type = wgpu::SamplerBindingType::Comparison;
    look_entries[2].texture.sampleType = wgpu::TextureSampleType::Float;
    look_entries[2].texture.viewDimension = wgpu::TextureViewDimension::e3D;
    look_entries[3].sampler.type = wgpu::SamplerBindingType::Filtering;
    look_layout = make_layout(dev, look_entries);
    wgpu::SamplerDescriptor sampler_desc{};
    sampler_desc.magFilter = wgpu::FilterMode::Linear;
    sampler_desc.minFilter = wgpu::FilterMode::Linear;
    lut_sampler = dev.CreateSampler(&sampler_desc);
    sampler_desc.compare = wgpu::CompareFunction::LessEqual;
    shadow_sampler = dev.CreateSampler(&sampler_desc);
    shadow_view = ov::dawn::make_texture(dev, DEPTH_FORMAT, 1, 1, wgpu::TextureUsage::TextureBinding).CreateView();
    lut_view = ov::dawn::make_texture(dev, wgpu::TextureFormat::RGBA8Unorm, 1, 1, wgpu::TextureUsage::TextureBinding, 1).CreateView();
    _update_look_group();

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
    desc.layout = make_pipeline_layout(dev, { frame_layout, page_layout, material_layout, look_layout });
    desc.vertex.module = vs;
    desc.vertex.entryPoint = "main";
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.frontFace = wgpu::FrontFace::CW; // Godot's: the meshers' winding
    desc.primitive.cullMode = wgpu::CullMode::Back;
    desc.depthStencil = &depth;
    desc.fragment = &fragment;
    pipeline = dev.CreateRenderPipeline(&desc);

    // The shadow map: the same corners from the sun, depth only (standard depth: 0 towards the sun), both faces (the
    // sun's view may mirror the winding), a slope bias on top of the receiver's normal offset.
    wgpu::DepthStencilState shadow_depth{};
    shadow_depth.format = DEPTH_FORMAT;
    shadow_depth.depthWriteEnabled = true;
    shadow_depth.depthCompare = wgpu::CompareFunction::Less;
    shadow_depth.depthBiasSlopeScale = 1.5f;
    wgpu::RenderPipelineDescriptor shadow_desc = desc;
    shadow_desc.label = "page shadow";
    shadow_desc.layout = make_pipeline_layout(dev, { frame_layout, page_layout, material_layout });
    shadow_desc.primitive.cullMode = wgpu::CullMode::None;
    shadow_desc.depthStencil = &shadow_depth;
    shadow_desc.fragment = nullptr;
    shadow_pipeline = dev.CreateRenderPipeline(&shadow_desc);

    // The sky where no page is: the far depth (0) still there.
    wgpu::DepthStencilState sky_depth{};
    sky_depth.format = DEPTH_FORMAT;
    sky_depth.depthWriteEnabled = false;
    sky_depth.depthCompare = wgpu::CompareFunction::Equal;
    wgpu::FragmentState sky_fragment = fragment;
    sky_fragment.module = sky_fs;
    wgpu::RenderPipelineDescriptor sky_desc = desc;
    sky_desc.label = "sky";
    sky_desc.layout = make_pipeline_layout(dev, { frame_layout, look_layout });
    sky_desc.vertex.module = sky_vs;
    sky_desc.primitive.cullMode = wgpu::CullMode::None;
    sky_desc.depthStencil = &sky_depth;
    sky_desc.fragment = &sky_fragment;
    sky_pipeline = dev.CreateRenderPipeline(&sky_desc);

    uniforms = make_uniforms(dev, sizeof(FrameUniforms));
    shadow_uniforms = make_uniforms(dev, sizeof(FrameUniforms));
    frame_group = make_group(dev, frame_layout, { buffer_bind(0, uniforms, sizeof(FrameUniforms)) });
    shadow_frame_group = make_group(dev, frame_layout, { buffer_bind(0, shadow_uniforms, sizeof(FrameUniforms)) });

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

DawnPageHost::FrameUniforms DawnPageHost::_frame_uniforms(const PageView &p_view, const float p_view_proj[16]) const {
    FrameUniforms u{};
    std::memcpy(u.view_proj, p_view_proj, sizeof(u.view_proj));
    if (p_view.shadow_m > 0.0f) {
        shadow_projection(p_view, u.shadow_proj);
    }
    u.origin_chunk[0] = decode.origin_chunk_lo.x;
    u.origin_chunk[1] = decode.origin_chunk_lo.y;
    u.origin_chunk[2] = decode.origin_chunk_lo.z;
    u.origin_chunk[3] = decode.chunk_size;
    const auto put = [](float r_v[4], const Float3 &p_xyz, float p_w) {
        r_v[0] = p_xyz.x;
        r_v[1] = p_xyz.y;
        r_v[2] = p_xyz.z;
        r_v[3] = p_w;
    };
    put(u.sun, p_view.sun, p_view.sun_size);
    put(u.sun_color, p_view.sun_color, p_view.sun_energy);
    put(u.camera, p_view.camera, p_view.fog_m);
    put(u.forward, p_view.forward, p_view.pixel_m);
    put(u.view_right, p_view.view_right, 0.0f);
    put(u.view_up, p_view.view_up, 0.0f);
    // Normal offset: a texel and a half, against the acne of a surface at a grazing sun.
    put(u.shadow, Float3{ p_view.shadow_m > 0.0f ? 1.0f : 0.0f, 1.5f * p_view.shadow_m / float(SHADOW_SIZE), 1.0f / float(SHADOW_SIZE) }, 0.0f);
    put(u.look, Float3{ graded ? 1.0f : 0.0f, p_view.sky_energy, p_view.sky_exposure }, 0.0f);
    u.ints[0] = decode.skirt_size;
    u.ints[1] = debug_view;
    u.floats[0] = decode.pos_scale;
    u.floats[1] = decode.voxel_size;
    u.floats[2] = p_view.time_s;
    return u;
}

int DawnPageHost::_draw_pages(const wgpu::RenderPassEncoder &p_pass, const float *p_shadow_proj, float p_half_m) {
    p_pass.SetBindGroup(2, material_group);
    p_pass.SetIndexBuffer(indices, wgpu::IndexFormat::Uint32);
    // Every visible page (the camera's pass: no frustum culling yet), or those the shadow map can hold: the instance
    // count is the GPU's (0 draws nothing).
    int drawn = 0;
    for (const Page &page : pages) {
        if (!page.visible || !page.group || (p_shadow_proj != nullptr && !in_shadow_map(page.bounds, p_shadow_proj, p_half_m))) {
            continue;
        }
        p_pass.SetBindGroup(1, page.group);
        p_pass.DrawIndexedIndirect(page.command_buffer, 0);
        drawn++;
    }
    return drawn;
}

void DawnPageHost::encode_shadow(const wgpu::CommandEncoder &p_encoder, const PageView &p_view) {
    if (!pipeline || p_view.shadow_m <= 0.0f) {
        return;
    }
    if (!shadow_map) { // the map on the first frame that asks for it
        shadow_map = ov::dawn::make_texture(device.get_context().device, DEPTH_FORMAT, int(SHADOW_SIZE), int(SHADOW_SIZE),
                wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding);
        shadow_view = shadow_map.CreateView();
        _update_look_group();
    }
    FrameUniforms u = _frame_uniforms(p_view, p_view.view_proj);
    std::memcpy(u.view_proj, u.shadow_proj, sizeof(u.view_proj)); // the sun's view
    device.get_context().queue.WriteBuffer(shadow_uniforms, 0, &u, sizeof(u));
    wgpu::RenderPassDepthStencilAttachment depth{};
    depth.view = shadow_view;
    depth.depthLoadOp = wgpu::LoadOp::Clear;
    depth.depthStoreOp = wgpu::StoreOp::Store;
    depth.depthClearValue = 1.0f;
    wgpu::RenderPassDescriptor pass_desc{};
    pass_desc.depthStencilAttachment = &depth;
    const wgpu::RenderPassEncoder pass = p_encoder.BeginRenderPass(&pass_desc);
    pass.SetPipeline(shadow_pipeline);
    pass.SetBindGroup(0, shadow_frame_group);
    _draw_pages(pass, u.shadow_proj, p_view.shadow_m * 0.5f);
    pass.End();
}

void DawnPageHost::draw(const wgpu::RenderPassEncoder &p_pass, const PageView &p_view) {
    last_drawn = 0;
    if (!pipeline) {
        return;
    }
    const FrameUniforms u = _frame_uniforms(p_view, p_view.view_proj);
    device.get_context().queue.WriteBuffer(uniforms, 0, &u, sizeof(u));
    p_pass.SetPipeline(pipeline);
    p_pass.SetBindGroup(0, frame_group);
    p_pass.SetBindGroup(3, look_group);
    last_drawn = _draw_pages(p_pass, nullptr, 0.0f);
    p_pass.SetPipeline(sky_pipeline);
    p_pass.SetBindGroup(0, frame_group);
    p_pass.SetBindGroup(1, look_group);
    p_pass.Draw(3);
}

bool DawnPageHost::set_grade(const ov::str &p_rgb) {
    if (!pipeline) {
        return false;
    }
    // A cube of N^3 RGB texels: N from the size.
    const uint32_t n = uint32_t(std::lround(std::cbrt(double(p_rgb.size() / 3))));
    OV_ERR_FAIL_COND_V_MSG(n < 2 || size_t(n) * n * n * 3 != p_rgb.size(), false,
            ov::format("DawnPageHost: a grade is a cube of RGB texels, not %u bytes.", uint32_t(p_rgb.size())));
    std::vector<uint8_t> rgba(size_t(n) * n * n * 4, 255);
    for (size_t i = 0; i < rgba.size() / 4; i++) {
        std::memcpy(&rgba[i * 4], &p_rgb[i * 3], 3);
    }
    const wgpu::Texture lut = ov::dawn::make_texture(device.get_context().device, wgpu::TextureFormat::RGBA8Unorm, int(n), int(n),
            wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst, n);
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = lut;
    wgpu::TexelCopyBufferLayout layout{};
    layout.bytesPerRow = n * 4;
    layout.rowsPerImage = n;
    const wgpu::Extent3D size = { n, n, n };
    device.get_context().queue.WriteTexture(&dst, rgba.data(), rgba.size(), &layout, &size);
    lut_view = lut.CreateView();
    _update_look_group();
    graded = true;
    return true;
}

void DawnPageHost::_update_look_group() {
    std::vector<wgpu::BindGroupEntry> entries(4);
    for (int i = 0; i < 4; i++) {
        entries[i].binding = uint32_t(i);
    }
    entries[0].textureView = shadow_view;
    entries[1].sampler = shadow_sampler;
    entries[2].textureView = lut_view;
    entries[3].sampler = lut_sampler;
    look_group = make_group(device.get_context().device, look_layout, entries);
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

wgpu::Texture DawnPageHost::make_texture(wgpu::TextureFormat p_format, uint32_t p_width, uint32_t p_height) const {
    wgpu::TextureDescriptor desc{};
    desc.size = { p_width, p_height, 1 };
    desc.format = p_format;
    desc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    return device.get_context().device.CreateTexture(&desc);
}

void DawnPageHost::_update_material_group() {
    wgpu::BindGroupEntry entries[2]{};
    entries[0].binding = 0;
    entries[0].textureView = palette_texture.CreateView();
    entries[1].binding = 1;
    entries[1].textureView = material_texture.CreateView();
    wgpu::BindGroupDescriptor desc{};
    desc.layout = material_layout;
    desc.entryCount = 2;
    desc.entries = entries;
    material_group = device.get_context().device.CreateBindGroup(&desc);
}

void DawnPageHost::set_materials(const Material::Library::Image &p_image) {
    if (!pipeline) {
        return; // the page shader didn't compile (logged): nothing draws
    }
    wgpu::Limits limits{};
    device.get_context().device.GetLimits(&limits);
    OV_ERR_FAIL_COND_MSG(uint32_t(p_image.width) > limits.maxTextureDimension2D,
            ov::format("DawnPageHost: the material image is %d texels wide, the device takes %u.", p_image.width, limits.maxTextureDimension2D));
    if (p_image.width == 0 || p_image.height == 0) {
        return;
    }
    material_texture = make_texture(wgpu::TextureFormat::RGBA32Float, uint32_t(p_image.width), uint32_t(p_image.height));
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = material_texture;
    wgpu::TexelCopyBufferLayout layout{};
    layout.bytesPerRow = uint32_t(p_image.width) * 4 * sizeof(float);
    layout.rowsPerImage = uint32_t(p_image.height);
    const wgpu::Extent3D size = { uint32_t(p_image.width), uint32_t(p_image.height), 1 };
    device.get_context().queue.WriteTexture(&dst, p_image.rgba.data(), p_image.rgba.size() * sizeof(float), &layout, &size);
    _update_material_group();
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
    if (!pipeline) {
        return;
    }
    // The mesher's (it grows by a new texture), or none (the mesher's shutdown): the placeholder again.
    const wgpu::Texture texture = p_texture.is_valid() ? device.get_texture(p_texture) : wgpu::Texture();
    palette_texture = texture ? texture : palette_placeholder;
    _update_material_group();
}

} // namespace ov::dawn
