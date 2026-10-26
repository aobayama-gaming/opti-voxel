#pragma once

// The Dawn adapter's small WebGPU helpers: the shaders' GLSL header, a triangle over the target, textures, bind group
// layouts and groups, uniform buffers (the upscaler's passes, the page host's).

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <vector>

namespace ov::dawn {

// The engine's GLSL dialect: every render shader starts with it.
inline constexpr const char GLSL_HEADER[] = "#version 450\n#extension GL_EXT_samplerless_texture_functions : require\n";

// One triangle over the target, at the far depth of a reversed depth (0); v_ndc: the pixel's clip-space xy.
inline constexpr const char FULLSCREEN_VERTEX[] = R"glsl(
layout(location = 0) out vec2 v_ndc;

void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_ndc = p * 2.0 - 1.0;
    gl_Position = vec4(v_ndc, 0.0, 1.0);
}
)glsl";

// p_depth > 0: a 3D texture of that many slices.
inline wgpu::Texture make_texture(const wgpu::Device &p_device, wgpu::TextureFormat p_format, int p_width, int p_height, wgpu::TextureUsage p_usage,
        uint32_t p_depth = 0) {
    wgpu::TextureDescriptor desc{};
    desc.dimension = p_depth > 0 ? wgpu::TextureDimension::e3D : wgpu::TextureDimension::e2D;
    desc.size = { uint32_t(p_width), uint32_t(p_height), p_depth > 0 ? p_depth : 1 };
    desc.format = p_format;
    desc.usage = p_usage;
    return p_device.CreateTexture(&desc);
}

inline wgpu::BindGroupLayoutEntry uniform_entry(uint32_t p_binding) {
    wgpu::BindGroupLayoutEntry e{};
    e.binding = p_binding;
    e.visibility = wgpu::ShaderStage::Fragment;
    e.buffer.type = wgpu::BufferBindingType::Uniform;
    return e;
}

inline wgpu::BindGroupLayoutEntry texture_entry(uint32_t p_binding, wgpu::TextureSampleType p_type) {
    wgpu::BindGroupLayoutEntry e{};
    e.binding = p_binding;
    e.visibility = wgpu::ShaderStage::Fragment;
    e.texture.sampleType = p_type;
    e.texture.viewDimension = wgpu::TextureViewDimension::e2D;
    return e;
}

inline wgpu::BindGroupEntry buffer_bind(uint32_t p_binding, const wgpu::Buffer &p_buffer, uint64_t p_size) {
    wgpu::BindGroupEntry e{};
    e.binding = p_binding;
    e.buffer = p_buffer;
    e.size = p_size;
    return e;
}

inline wgpu::BindGroupEntry texture_bind(uint32_t p_binding, const wgpu::Texture &p_texture) {
    wgpu::BindGroupEntry e{};
    e.binding = p_binding;
    e.textureView = p_texture.CreateView();
    return e;
}

inline wgpu::BindGroup make_group(const wgpu::Device &p_device, const wgpu::BindGroupLayout &p_layout, const std::vector<wgpu::BindGroupEntry> &p_entries) {
    wgpu::BindGroupDescriptor desc{};
    desc.layout = p_layout;
    desc.entryCount = p_entries.size();
    desc.entries = p_entries.data();
    return p_device.CreateBindGroup(&desc);
}

inline wgpu::BindGroupLayout make_layout(const wgpu::Device &p_device, const std::vector<wgpu::BindGroupLayoutEntry> &p_entries) {
    wgpu::BindGroupLayoutDescriptor desc{};
    desc.entryCount = p_entries.size();
    desc.entries = p_entries.data();
    return p_device.CreateBindGroupLayout(&desc);
}

inline wgpu::Buffer make_uniforms(const wgpu::Device &p_device, uint64_t p_size) {
    wgpu::BufferDescriptor desc{};
    desc.size = p_size;
    desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    return p_device.CreateBuffer(&desc);
}

} // namespace ov::dawn
