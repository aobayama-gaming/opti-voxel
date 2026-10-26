#pragma once

// The Dawn renderer: the terrain mesh drawn into an offscreen HDR target, upscaled by FSR3 (or a bilinear
// fallback), then presented with the pixelate pass on top. One pipeline for the terrain, one for the blit.

#include "fsr3.hpp"
#include "gpu.hpp"
#include "terrain.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace ov::dawn {

struct Camera {
    Float3 position{ 0.0f, 40.0f, 0.0f };
    float yaw = 0.0f; // radians, 0 looks down -z
    float pitch = 0.0f;
    float fov_y = 1.15f;
    float near_m = 0.1f;
    float far_m = 400.0f;
};

class Renderer {
public:
    bool initialize(GpuContext &p_gpu, Terrain &p_terrain);
    void shutdown();
    void resize(uint32_t p_width, uint32_t p_height);
    // Draws one frame into the surface. p_upscale: false renders at full resolution (no FSR3).
    void render(Camera &p_camera, bool p_upscale, bool p_fsr3, int p_pixelate);

    bool fsr_active() const { return fsr3 && !fsr3_failed && fsr_last_ok; }
    int render_width() const { return int(render_w); }
    int render_height() const { return int(render_h); }

private:
    struct GpuMesh {
        wgpu::Buffer vertices;
        wgpu::Buffer indices;
        uint32_t index_count = 0;
        uint64_t revision = ~uint64_t(0);
    };

    void create_offscreen();
    void update_meshes();
    void update_camera(const Camera &p_camera);

    GpuContext *gpu = nullptr;
    Terrain *terrain = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t render_w = 0;
    uint32_t render_h = 0;

    wgpu::Texture color;
    wgpu::TextureView color_view;
    wgpu::Texture depth;
    wgpu::TextureView depth_view;
    wgpu::Texture motion;
    wgpu::TextureView motion_view;
    wgpu::Texture depth_color;
    wgpu::TextureView depth_color_view;

    wgpu::BindGroupLayout camera_layout;
    wgpu::BindGroupLayout present_layout;
    wgpu::PipelineLayout terrain_pipeline_layout;
    wgpu::PipelineLayout present_pipeline_layout;
    wgpu::RenderPipeline terrain_pipeline;
    wgpu::RenderPipeline present_pipeline;
    wgpu::Buffer camera_buffer;
    wgpu::BindGroup camera_bind_group;
    wgpu::BindGroup present_bind_group;
    wgpu::Sampler present_sampler;
    wgpu::Buffer present_params;
    wgpu::TextureView present_source;
    wgpu::Texture fsr_output;
    wgpu::TextureView fsr_output_view;
    std::unique_ptr<Fsr3Upscaler> fsr3;
    bool fsr3_failed = false;
    bool fsr_last_ok = false;
    bool fsr_reported = false;

    std::unordered_map<ChunkId, GpuMesh, ChunkIdHash> meshes;
    float previous_view_proj[16] = {};
    bool has_previous = false;
};

} // namespace ov::dawn
