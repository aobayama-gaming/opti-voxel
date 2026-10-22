#pragma once

#include "chunk_palette.hpp"
#include "material_library.hpp"
#include "page_format.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>

#include <memory>
#include <vector>

using namespace godot;

// The loaded chunks' material palettes (chunk_palette.hpp) on the main RenderingDevice, for the page shader: one
// RGBA16UI texture, PALETTE_TEX_WIDTH texels (one whole palette) per row, 2 global ids per texel (lo, hi words of
// each). A chunk's block offset (in entries) rides in its rows' group headers (PageFormat), a corner's palette
// index is relative to it. A chunk whose palette is just the base one (SDFGpuTable::base_palette) uses the shared ground
// block (no allocation, its entries written at rt_create), and so does a chunk whose palette found no room: its vertices
// then take the base's entries, and a record's index past them material 0 (the direct vertex pass).
//
// Main thread: place_batch / release (the allocator). Render thread: rt_* (the texture). The direct mesher's emit
// pass writes the batch's texels (glsl_batch) as it writes the headers that name them.
class PalettePool {
public:
    // A chunk's block: where its palette is in the texture, and the palette (the CPU mirror the validator and the
    // auditor read; a block holds palette->size() entries).
    struct Block {
        uint32_t offset = PaletteAlloc::GROUND_BLOCK;
        std::shared_ptr<const std::vector<uint32_t>> palette;
    };

    // The batch payload's layout (uint words): per chunk of the batch its block offset, then the number of texels
    // to write, then BATCH_TEXEL_WORDS words per texel (its even entry, the id of that entry, the id of the next).
    static constexpr uint32_t BATCH_TEXEL_WORDS = 3;
    static uint64_t batch_bytes(uint32_t p_batch_size);
    // The payload's reader for the emit pass: `void write_palette_texel(uint t)` (texel t of the batch's, if any),
    // after the palette_image / palette_batch bindings and PageFormat::glsl_encode().
    static String glsl_batch();

    // p_capacity: palette entries (a power of two).
    void initialize(uint32_t p_capacity, const std::vector<uint32_t> &p_base);
    // Places each chunk's palette (consumed) and returns the batch payload. A palette that finds no room draws as
    // ground (ERR once, counted).
    PackedByteArray place_batch(std::vector<std::vector<uint32_t>> &p_palettes, std::vector<Block> &r_blocks);
    void release(const Block &p_block);
    uint32_t rows() const { return alloc.capacity() / PageFormat::PALETTE_SIZE; }
    Dictionary get_stats() const;

    // ---- render thread ----
    // The texture (zeroed: the ground block reads GROUND) and its RenderingServer wrapper for the page shader.
    void rt_create(RenderingDevice *p_rd);
    void rt_free(RenderingDevice *p_rd);
    RID rt_texture() const { return texture; }
    RID rt_shader_texture() const { return rs_texture; }

private:
    PaletteAlloc alloc;
    int64_t blocks_live = 0;
    int64_t full = 0; // palettes that found no room
    // The base palette (SDFGpuTable::base_palette): the ground block's, and the Block of every chunk on it.
    std::shared_ptr<const std::vector<uint32_t>> base = std::make_shared<const std::vector<uint32_t>>(1, Material::GROUND);
    RID texture;
    RID rs_texture;
};
