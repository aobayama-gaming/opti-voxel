#pragma once

#include "dawn_device.hpp"

#include "page_host.hpp"

#include "ov/str.hpp"

#include <webgpu/webgpu_cpp.h>

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
    Float3 sun_color;
    float sun_energy = 1.0f;
    float sun_size = 0.0f; // the sun's disk in the sky, radians across
    float sky_energy = 1.0f; // the sky's light on the pages
    float sky_exposure = 1.0f; // the sky's brightness (Godot's PhysicalSkyMaterial.energy_multiplier)
    float shadow_m = 0.0f; // the side of the square around the camera the sun's shadows cover; 0 none
    float fog_m = 0.0f; // distance where the fog reaches half; 0 none
    Float3 forward; // the camera's view direction (normalized)
    Float3 view_right; // the camera's right times the tangent of half the horizontal field of view (the sky's rays)
    Float3 view_up; // its up times the tangent of half the vertical one
    float pixel_m = 0.0f; // metres per pixel of the drawn target per metre of depth (the materials' detail fades)
    float time_s = 0.0f; // seconds since the start (the lava's drift)
};

// PageHost on Dawn: a page is an indexed indirect draw of QUAD_GROUP-quad instances (the core writes the instance count
// into its command buffer) with the page shader (dawn_page_host.cpp), which pulls each corner from the page texture
// through PageFormat::glsl_decode. Shading: the materials of page_shading_glsl.hpp (Godot's, from the mesher's palettes)
// under a sun with its shadow map and the sky's light, then the sky where no page is, the fog, the tonemapper and the
// grade (dawn_look_glsl.hpp).
// The device's thread is the main thread (DawnDevice::post runs at once).
class DawnPageHost : public PageHost {
public:
    static constexpr wgpu::TextureFormat DEPTH_FORMAT = wgpu::TextureFormat::Depth32Float;

    // p_color_format: the render target's (a surface's, or the offscreen shot's).
    DawnPageHost(DawnDevice &p_device, wgpu::TextureFormat p_color_format);

    // The sun's shadow map of this frame (its own pass), before draw; nothing when p_view.shadow_m is 0.
    void encode_shadow(const wgpu::CommandEncoder &p_encoder, const PageView &p_view);
    // Draws the visible pages, then the sky, into p_pass: a pass on a p_color_format target with a DEPTH_FORMAT depth
    // cleared to 0 (reversed depth: near is 1).
    void draw(const wgpu::RenderPassEncoder &p_pass, const PageView &p_view);
    // The grading cube (tools/make_lut.py's grade_lut.rgb: 32^3 RGB bytes, red fastest); ungraded until then. False
    // (an error logged) when it isn't one.
    bool set_grade(const ov::str &p_rgb);
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
        float shadow_proj[16];
        int32_t origin_chunk[4]; // xyz: the world origin's chunk id, low words; w: chunk_size
        float sun[4]; // xyz: towards the sun; w: its disk's size
        float sun_color[4]; // w: energy
        float camera[4]; // xyz: render metres; w: fog distance (0 none)
        float forward[4]; // xyz: the camera's; w: metres per pixel per metre of depth
        float view_right[4];
        float view_up[4];
        float shadow[4]; // on, normal offset (metres), a texel in uv
        float look[4]; // graded, the sky light's energy, the sky's exposure
        int32_t ints[4]; // skirt_size, debug_view
        float floats[4]; // pos_scale, voxel_size, time
    };

    wgpu::Texture make_texture(wgpu::TextureFormat p_format, uint32_t p_width, uint32_t p_height) const;
    // Set 2 from palette_texture and material_texture.
    void _update_material_group();
    // The look's set from shadow_view and lut_view.
    void _update_look_group();
    FrameUniforms _frame_uniforms(const PageView &p_view, const float p_view_proj[16]) const;
    // The visible pages into p_pass (its pipeline and set 0 bound); with p_shadow_proj, only those that can cast into
    // that shadow map (a square 2 p_half_m wide). Returns how many were drawn.
    int _draw_pages(const wgpu::RenderPassEncoder &p_pass, const float *p_shadow_proj, float p_half_m);

    DawnDevice &device;
    wgpu::RenderPipeline pipeline;
    wgpu::RenderPipeline shadow_pipeline;
    wgpu::RenderPipeline sky_pipeline;
    wgpu::BindGroupLayout page_layout;
    wgpu::Buffer uniforms;
    wgpu::BindGroup frame_group;
    wgpu::Buffer shadow_uniforms; // the shadow pass's Frame: view_proj is the sun's
    wgpu::BindGroup shadow_frame_group;
    wgpu::BindGroupLayout look_layout;
    wgpu::BindGroup look_group; // set 3 of the pages, set 1 of the sky
    wgpu::Texture shadow_map; // made by the first shadow pass
    wgpu::TextureView shadow_view; // the map's, or one texel until then
    wgpu::Sampler shadow_sampler;
    wgpu::TextureView lut_view; // the grade's cube, or one texel
    wgpu::Sampler lut_sampler;
    bool graded = false;
    wgpu::BindGroupLayout material_layout;
    wgpu::Texture palette_placeholder; // one texel: no palette yet
    wgpu::Texture palette_texture; // the mesher's palettes, or the placeholder
    wgpu::Texture material_texture; // Material::Library::make_image
    wgpu::BindGroup material_group; // set 2
    wgpu::Buffer indices; // QUAD_GROUP quads: corners A B C D = 4q+0..3, triangles A,B,D / A,D,C
    PageDecode decode;
    int debug_view = 0;
    std::vector<Page> pages;
    int last_drawn = 0;
};

} // namespace ov::dawn
