#pragma once

#include "page_host.hpp"
#include "rd_device.hpp"

#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <vector>

// PageHost on Godot's RenderingServer: a page is a MultiMesh of QUAD_GROUP-quad instances drawn indirectly (the core
// writes the instance count into its command buffer) with the page shader (godot_page_host.cpp), in the terrain's
// scenario. Never call multimesh_set_buffer / multimesh_instance_* / multimesh_get_buffer on a page: it creates a CPU
// cache that overwrites the GPU-written data.
class GodotPageHost : public PageHost {
public:
    // p_device: the main device (the pages' textures and command buffers are its objects).
    GodotPageHost(godot::RID p_scenario, RDDevice *p_device);
    // Main thread, after the mesher's destroy(): frees the shader and the base mesh, then deletes this object on the render
    // thread, after the pool's teardown posted before.
    void destroy();

    // The page shader's other uniforms (VoxelTerrain: sea level, skirt depths, base up, the sky's clouds), on every page.
    void set_material_param(const godot::StringName &p_name, const godot::Variant &p_value);

    void create_page(int p_page, uint32_t p_rows) override;
    void free_page(int p_page) override;
    void set_page_visible(int p_page, bool p_visible) override;
    void set_page_bounds(int p_page, const RenderBox &p_bounds) override;
    void set_decode(const PageDecode &p_decode) override;
    void set_materials(const Material::Library::Image &p_image) override;
    void set_debug_view(int p_view) override;

    ov::gpu::Buffer rt_attach_page(int p_page, ov::gpu::Texture p_texture) override;
    void rt_detach_page(int p_page) override;
    void rt_set_palette_texture(ov::gpu::Texture p_texture) override;

private:
    struct Page {
        godot::RID multimesh; // invalid once freed
        godot::RID instance;
        godot::RID material;
    };
    // The render thread's copy (the main thread may be growing `pages`).
    struct PageRT {
        godot::RID multimesh;
        godot::RID material;
        godot::RID rs_texture; // the page texture's RenderingServer wrapper, sampled as quad_tex
        ov::gpu::Buffer command; // the MultiMesh's indirect command buffer, imported into the device
    };

    godot::RID scenario;
    RDDevice *device = nullptr;
    godot::RID base_mesh;
    godot::RID shader;
    godot::Dictionary material_params;
    PageDecode decode; // the last set_decode's
    bool has_decode = false;
    godot::Ref<godot::ImageTexture> material_texture; // material_tex (a material param holds its RID only)
    std::vector<Page> pages;
    std::vector<PageRT> rt_pages;
    godot::RID rt_palette_texture;

    ~GodotPageHost() override = default;
};
