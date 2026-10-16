#pragma once

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/local_vector.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <vector>

using namespace godot;

// A pool of "pages" that compute shaders fill directly on the GPU, so drawing never needs the CPU
// to see the geometry.
//
// Page = one MultiMesh whose instances are *groups* of QUAD_GROUP quads (the base mesh holds
// QUAD_GROUP quads) + one RGBA32F texture holding the quads, one texel per corner:
// xyz = world position, w = packed normal + 1 (0 = empty). One texture row is one group, so the
// page shader pulls its corner with texelFetch(quad_tex, ivec2(VERTEX_ID, INSTANCE_ID)). The GPU
// writes the number of used groups into the MultiMesh indirect command buffer.
//
// Chunks start on a group boundary, so a chunk is a block of whole rows: moving it to another page
// (compaction) or zeroing it (kill) is a single texture copy.
//
// Why groups and not one instance per quad: Godot keeps three CPU-side interpolation arrays per
// MultiMesh instance (~189 B RAM per instance with TRANSFORM_3D + custom, measured). Per group of
// 64 quads with TRANSFORM_2D that is ~1.5 B per quad. The instance data itself is unused.
//
// Threading: everything except rt_* runs on the main thread; rt_* must run on the render thread.
// Never call multimesh_set_buffer / multimesh_instance_* / multimesh_get_buffer on a page.
class QuadPagePool : public Object {
public:
    static constexpr uint32_t QUAD_GROUP = 64;        // quads per MultiMesh instance
    static constexpr uint32_t TEX_WIDTH = QUAD_GROUP * 4; // texels per row of a page texture: one group
    static constexpr uint32_t CORNER_BYTES = 16;      // one RGBA32F texel
    static constexpr uint32_t QUAD_BYTES = 4 * CORNER_BYTES;
    // VkDrawIndexedIndirectCommand: index_count, instance_count, first_index, vertex_offset, first_instance.
    static constexpr uint32_t INDIRECT_INSTANCE_COUNT_OFFSET = 4;

    struct PageRD {
        RID texture; // RD texture holding the page's quads
        RID command; // indirect command buffer (instance_count = used groups)
    };

    // p_page_capacity must be a multiple of QUAD_GROUP.
    void initialize(RID p_scenario, uint32_t p_page_capacity, uint32_t p_initial_pages, float p_voxel_size);
    // Frees everything and deletes this object (on the render thread, after pending work).
    void destroy();

    // Main thread. Returns a page index; grows the pool when the free list is empty.
    int acquire_page();
    // Main thread. The page stops drawing (hidden, group count reset on the render thread).
    void release_page(int p_page);
    // Main thread. Frees free pages beyond p_keep_free (at most p_max per call) so VRAM follows the
    // pages in use instead of the high-water mark. Their indices are reused by later growth.
    int trim(int p_keep_free, int p_max);

    void set_page_aabb(int p_page, const AABB &p_aabb);

    RID get_page_multimesh(int p_page) const { return pages[p_page].multimesh; }
    uint32_t get_page_capacity() const { return page_capacity; }
    uint32_t get_page_rows() const { return page_capacity * 4 / TEX_WIDTH; }
    int get_page_count() const { return pages.size(); } // index space, freed pages included

    void set_debug_view(int p_view);
    void set_material_param(const StringName &p_name, const Variant &p_value);

    Dictionary get_stats() const;

    // ---- render thread ----
    // The multimesh RID is passed in (not read from `pages`) because the main thread may be growing
    // `pages` concurrently.
    PageRD rt_get_page_rd(int p_page, RID p_multimesh);
    void rt_set_group_count(int p_page, RID p_multimesh, uint32_t p_groups);
    // Copies a whole page worth of rows of `p_src`, starting at `p_src_row`, into the page texture.
    void rt_copy_page_from(int p_page, RID p_multimesh, RID p_src, uint32_t p_src_row);
    // Zeroes quads [offset, offset + count): corners collapse to the origin, never rasterised.
    void rt_kill_range(int p_page, RID p_multimesh, uint32_t p_offset, uint32_t p_count);
    // Copies p_rows groups (texture rows) from one page to another (different) page.
    void rt_copy_rows(int p_src_page, int p_dst_page, uint32_t p_src_row, uint32_t p_dst_row, uint32_t p_rows);

private:
    struct Page {
        RID multimesh; // invalid once trimmed
        RID instance;
        RID material;
        bool in_use = false;
    };

    struct PageRT {
        RID texture;    // RD
        RID rs_texture; // RenderingServer wrapper sampled by the material
        RID command;
    };

    RID scenario;
    RID base_mesh;
    RID shader;
    uint32_t page_capacity = 0;
    float voxel_size = 1.0f;
    int debug_view = 0;
    Dictionary material_params;

    std::vector<Page> pages;
    LocalVector<int> free_pages;
    LocalVector<int> trimmed_pages; // indices whose resources were freed, reused first on growth
    uint64_t growth_events = 0;
    uint64_t pages_trimmed = 0;
    int pages_in_use = 0;

    // render thread only
    std::vector<PageRT> rt_pages;
    RID rt_zero_texture;

    int _create_page(int p_index = -1);
    void _rt_create_page(int p_page, RID p_multimesh, RID p_material);
    void _rt_reset_page(int p_page, RID p_multimesh);
    void _rt_free_page(int p_page);
    void _rt_destroy();
};
