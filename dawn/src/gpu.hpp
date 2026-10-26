#pragma once

// The Dawn side of the adapter: one instance, adapter, device, queue and X11 surface, plus the small helpers the
// mesher and renderer share (buffers, textures, readback, compute pipelines from GLSL).

#include <dawn/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ov::dawn {

struct GpuContext {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Surface surface;
    wgpu::TextureFormat format = wgpu::TextureFormat::BGRA8Unorm;
    uint32_t width = 0;
    uint32_t height = 0;

    bool initialize(void *p_xdisplay, uint64_t p_xwindow, uint32_t p_width, uint32_t p_height);
    void resize(uint32_t p_width, uint32_t p_height);
    wgpu::SurfaceTexture acquire();
    void present(wgpu::SurfaceTexture &p_texture);
};

// A buffer with an optional CPU shadow for the small uploads (uniforms, chunk inputs).
wgpu::Buffer create_buffer(const wgpu::Device &p_device, uint64_t p_size, wgpu::BufferUsage p_usage, const char *p_label = nullptr);
void write_buffer(const wgpu::Queue &p_queue, const wgpu::Buffer &p_buffer, const void *p_data, uint64_t p_size, uint64_t p_offset = 0);

// A texture with a view. p_mips: 1 unless the caller needs a chain.
wgpu::Texture create_texture(const wgpu::Device &p_device, uint32_t p_width, uint32_t p_height, wgpu::TextureFormat p_format,
        wgpu::TextureUsage p_usage, uint32_t p_mips = 1, const char *p_label = nullptr);

// Copies p_size bytes out of p_buffer and waits. Empty on failure. p_buffer needs COPY_SRC.
std::vector<uint8_t> read_buffer(const GpuContext &p_gpu, const wgpu::Buffer &p_buffer, uint64_t p_size);

// Compiles GLSL text to a Dawn compute pipeline. The bind group layout is explicit; immediateSize is the push
// constant block's size (0: none). r_error gets the compiler/linker message on failure.
wgpu::ComputePipeline create_compute_pipeline(const GpuContext &p_gpu, const std::string &p_glsl,
        const wgpu::BindGroupLayout &p_layout, uint32_t p_immediate_size, std::string &r_error, const char *p_label = nullptr);

} // namespace ov::dawn
