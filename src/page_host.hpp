#pragma once

#include "gpu_device.hpp"
#include "material_library.hpp"
#include "world_coord.hpp"

#include "ov/types.hpp"

// An axis-aligned box in render metres: a page's culling bounds. godot's AABB in float (merge_with,
// get_longest_axis_size), the same operations.
struct RenderBox {
    Float3 position;
    Float3 size;

    void merge_with(const RenderBox &p_box) {
        const Float3 end_1{ size.x + position.x, size.y + position.y, size.z + position.z };
        const Float3 end_2{ p_box.size.x + p_box.position.x, p_box.size.y + p_box.position.y, p_box.size.z + p_box.position.z };
        const Float3 lo{ position.x < p_box.position.x ? position.x : p_box.position.x, position.y < p_box.position.y ? position.y : p_box.position.y,
            position.z < p_box.position.z ? position.z : p_box.position.z };
        const Float3 hi{ end_1.x > end_2.x ? end_1.x : end_2.x, end_1.y > end_2.y ? end_1.y : end_2.y, end_1.z > end_2.z ? end_1.z : end_2.z };
        position = lo;
        size = Float3{ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z };
    }
    RenderBox merge(const RenderBox &p_box) const {
        RenderBox out = *this;
        out.merge_with(p_box);
        return out;
    }
    float longest_axis_size() const {
        float m = size.x;
        if (size.y > m) {
            m = size.y;
        }
        if (size.z > m) {
            m = size.z;
        }
        return m;
    }
};

// What the page shader decodes a page with (PageFormat::glsl_decode): the rows' chunk-local positions back to render
// space.
struct PageDecode {
    int chunk_size = 0;
    int skirt_size = 0;
    float pos_scale = 0.0f;
    float voxel_size = 0.0f;
    ov::ivec3 origin_chunk_lo; // the world origin's chunk id, low words (render positions are relative to it)
};

// The host's side of the quad pages (QuadPagePool): drawing them. The core fills each page's texture (PageFormat rows,
// one per group of QUAD_GROUP quads) and the instance count of its indirect draw on the main device; the host draws page
// p as that many instances of a QUAD_GROUP-quad mesh whose page shader pulls each corner from the texture. Godot's:
// a MultiMesh per page (adapter/godot/src/godot_page_host.*).
class PageHost {
public:
    virtual ~PageHost() = default;

    // ---- main thread ----
    // A hidden drawable for page p_page of p_rows groups (an index free_page released may come back).
    virtual void create_page(int p_page, uint32_t p_rows) = 0;
    virtual void free_page(int p_page) = 0;
    virtual void set_page_visible(int p_page, bool p_visible) = 0;
    virtual void set_page_bounds(int p_page, const RenderBox &p_bounds) = 0;
    virtual void set_decode(const PageDecode &p_decode) = 0;
    // The materials' parameters (Material::Library::make_image), read by id through the chunks' palettes.
    virtual void set_materials(const Material::Library::Image &p_image) = 0;
    virtual void set_debug_view(int p_view) = 0;

    // ---- device thread (after the create_page it follows) ----
    // Page p_page's texture, made by the core: the host samples it. Returns the page's indirect command buffer on the
    // device (VkDrawIndexedIndirectCommand: the instance count at QuadPagePool::INDIRECT_INSTANCE_COUNT_OFFSET).
    virtual ov::gpu::Buffer rt_attach_page(int p_page, ov::gpu::Texture p_texture) = 0;
    // Before the page's texture is freed (after free_page).
    virtual void rt_detach_page(int p_page) = 0;
    // The chunks' palettes (PalettePool), sampled by the page shader.
    virtual void rt_set_palette_texture(ov::gpu::Texture p_texture) = 0;
};
