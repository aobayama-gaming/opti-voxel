#pragma once

#include "dawn_device.hpp"

#include "page_host.hpp"

#include <dawn/webgpu_cpp.h>

#include <string>
#include <vector>

namespace ov::dawn {

// A vertex or fragment shader of the engine's GLSL dialect; null (an error logged) when it doesn't compile.
wgpu::ShaderModule make_render_module(const wgpu::Device &p_device, const std::string &p_glsl, ShaderStage p_stage);

// What a frame of pages is drawn with: the camera and the light.
struct PageView {
    float view_proj[16]; // column-major, render metres to clip space (WebGPU's: depth in [0, 1])
    Float3 camera; // render metres (the fog's distance)
    Float3 sun; // towards the sun, normalized
    float fog_m = 0.0f; // distance where the fog reaches half; 0 none
};

// PageHost on Dawn: a page is an indexed indirect draw of QUAD_GROUP-quad instances (the core writes the instance count
// into its command buffer) with the page shader (dawn_page_host.cpp), which pulls each corner from the page texture
// through PageFormat::glsl_decode. Shading: the ground's slope colours under a sun and a sky light, no materials yet.
// The device's thread is the main thread (DawnDevice::post runs at once).
class DawnPageHost : public PageHost {
public:
    static constexpr wgpu::TextureFormat DEPTH_FORMAT = wgpu::TextureFormat::Depth32Float;

    // p_color_format: the render target's (a surface's, or the offscreen shot's).
    DawnPageHost(DawnDevice &p_device, wgpu::TextureFormat p_color_format);

    // Draws the visible pages into p_pass: a pass on a p_color_format target with a DEPTH_FORMAT depth cleared to 0
    // (reversed depth: near is 1).
    void draw(const wgpu::RenderPassEncoder &p_pass, const PageView &p_view);
    int drawn_pages() const { return last_drawn; }
    bool ready() const { return pipeline != nullptr; }

    void create_page(int p_page, uint32_t p_rows) override;
    void free_page(int p_page) override;
    void set_page_visible(int p_page, bool p_visible) override;
    void set_page_bounds(int p_page, const RenderBox &p_bounds) override;
    void set_decode(const PageDecode &p_decode) override;
    void set_materials(const Material::Library::Image &p_image) override;
    void set_debug_view(int p_view) override;

    gpu::Buffer rt_attach_page(int p_page, gpu::Texture p_texture) override;
    void rt_detach_page(int p_page) override;
    void rt_set_palette_texture(gpu::Texture p_texture) override;

private:
    struct Page {
        bool visible = false;
        RenderBox bounds;
        gpu::Buffer command; // ours, written by the core
        wgpu::Buffer command_buffer;
        wgpu::BindGroup group; // set 1: the page texture
    };

    // The page shader's set 0 (std140, the GLSL `Frame` block).
    struct FrameUniforms {
        float view_proj[16];
        int32_t origin_chunk[4]; // xyz: the world origin's chunk id, low words; w: chunk_size
        float sun[4]; // xyz: towards the sun
        float camera[4]; // xyz: render metres; w: fog distance (0 none)
        int32_t ints[4]; // skirt_size, debug_view
        float floats[4]; // pos_scale, voxel_size
    };

    DawnDevice &device;
    wgpu::RenderPipeline pipeline;
    wgpu::BindGroupLayout page_layout;
    wgpu::Buffer uniforms;
    wgpu::BindGroup frame_group;
    wgpu::Buffer indices; // QUAD_GROUP quads: corners A B C D = 4q+0..3, triangles A,B,D / A,D,C
    PageDecode decode;
    int debug_view = 0;
    std::vector<Page> pages;
    int last_drawn = 0;
};

} // namespace ov::dawn
