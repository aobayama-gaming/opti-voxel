#pragma once

#include "gpu_device.hpp"
#include "page_format.hpp"
#include "page_host.hpp"

#include "ov/value.hpp"

#include <cstdint>
#include <vector>

// A pool of "pages" that compute shaders fill directly on the GPU, so drawing never needs the CPU
// to see the geometry.
//
// Page = one RGBA16UI texture holding the quads, one row per group of QUAD_GROUP quads: a header with the owning
// chunk's id, then 8 bytes per corner (chunk-local fixed-point position, palette index and octahedral normal, see
// page_format.hpp), and the instance count of an indirect draw: the groups used. The host draws it (PageHost: Godot's
// is a MultiMesh whose instances are the groups, its page shader pulling each corner with
// texelFetch(quad_tex, ivec2(HEADER_TEXELS + VERTEX_ID, INSTANCE_ID)) and rebuilding the world position from the
// header).
//
// Chunks start on a group boundary, so a chunk is a block of whole rows: moving it to another page
// (compaction) or zeroing it (kill) is a single texture copy.
//
// Why groups and not one instance per quad: Godot keeps three CPU-side interpolation arrays per
// MultiMesh instance (~189 B RAM per instance with TRANSFORM_3D + custom, measured). Per group of
// 64 quads with TRANSFORM_2D that is ~1.5 B per quad. The instance data itself is unused.
//
// Threading: everything except rt_* runs on the main thread; rt_* on the device's thread (Device::post).
class QuadPagePool {
public:
    static constexpr uint32_t QUAD_GROUP = PageFormat::QUAD_GROUP; // quads per drawn instance
    static constexpr uint32_t TEX_WIDTH = PageFormat::TEX_WIDTH; // texels per row: one group
    static constexpr ov::gpu::TextureFormat TEX_FORMAT = ov::gpu::TextureFormat::RGBA16_UINT;
    // VkDrawIndexedIndirectCommand: index_count, instance_count, first_index, vertex_offset, first_instance.
    static constexpr uint32_t INDIRECT_INSTANCE_COUNT_OFFSET = 4;

    struct PageRD {
        ov::gpu::Texture texture; // the page's quads
        ov::gpu::Buffer command; // indirect command buffer (instance_count = used groups)
    };

    // p_page_capacity must be a multiple of QUAD_GROUP. p_device: the main device (the host draws its textures).
    void initialize(ov::gpu::Device *p_device, PageHost *p_host, uint32_t p_page_capacity, uint32_t p_initial_pages);
    // Frees everything and deletes this object (on the device's thread, after pending work).
    void destroy();

    // Main thread. Returns a page index; grows the pool when the free list is empty.
    int acquire_page();
    // Main thread. The page stops drawing (hidden, group count reset on the device's thread).
    void release_page(int p_page);
    // Main thread. Frees free pages beyond p_keep_free (at most p_max per call) so VRAM follows the
    // pages in use instead of the high-water mark. Their indices are reused by later growth.
    int trim(int p_keep_free, int p_max);

    void set_page_aabb(int p_page, const RenderBox &p_aabb);

    uint32_t get_page_capacity() const { return page_capacity; }
    uint32_t get_page_rows() const { return page_capacity / QUAD_GROUP; }
    int get_page_count() const { return int(pages.size()); } // index space, freed pages included
    PageHost *get_host() const { return host; }
    ov::gpu::Device *get_device() const { return device; }

    ov::Value get_stats() const;

    // ---- device thread ----
    // These only read `rt_pages`, never `pages`, which the main thread may be growing concurrently.
    PageRD rt_get_page_rd(int p_page);
    void rt_set_group_count(int p_page, uint32_t p_groups);
    // Copies a whole page worth of rows of `p_src`, starting at `p_src_row`, into the page texture.
    void rt_copy_page_from(int p_page, ov::gpu::Texture p_src, uint32_t p_src_row);
    // Zeroes the groups covering quads [offset, offset + count), headers included: the corners
    // collapse to one point, never rasterised. p_offset must start a group (chunks do).
    void rt_kill_range(int p_page, uint32_t p_offset, uint32_t p_count);
    // Copies p_rows groups (texture rows) from one page to another (different) page.
    void rt_copy_rows(int p_src_page, int p_dst_page, uint32_t p_src_row, uint32_t p_dst_row, uint32_t p_rows);
    // Writes header texels 1 and 2 (PageFormat::encode_coverage) of chunks' rows: (page, first row, rows, 64-bit mask,
    // z low word | palette << 32) per chunk. A chunk's rows are at most a page's.
    static constexpr int HIDDEN_WRITE_WORDS = 5;
    void rt_set_hidden(const std::vector<int64_t> &p_writes);

private:
    struct Page {
        bool alive = false; // false once trimmed
        bool in_use = false;
    };

    struct PageRT {
        ov::gpu::Texture texture;
        ov::gpu::Buffer command; // the host's
    };

    ov::gpu::Device *device = nullptr;
    PageHost *host = nullptr;
    uint32_t page_capacity = 0;

    std::vector<Page> pages;
    std::vector<int> free_pages;
    std::vector<int> trimmed_pages; // indices whose resources were freed, reused first on growth
    uint64_t growth_events = 0;
    uint64_t pages_trimmed = 0;
    int pages_in_use = 0;
    int64_t pages_alive_peak = 0; // high-water mark of allocated pages (VRAM peak)

    // device thread only
    std::vector<PageRT> rt_pages;
    ov::gpu::Texture rt_zero_texture;
    // Header texels 1 and 2 per row, a page's rows tall, one per round of an rt_set_hidden call (created on demand):
    // filled whole from rt_hidden_bytes, then each chunk's run is copied into its rows.
    std::vector<ov::gpu::Texture> rt_hidden_staging;
    std::vector<uint16_t> rt_hidden_texels;

    int _create_page(int p_index = -1);
    int64_t _alive_pages() const { return int64_t(pages.size()) - int64_t(trimmed_pages.size()); } // with resources
    void _rt_create_page(int p_page);
    void _rt_free_page(int p_page);
    void _rt_destroy();
};
