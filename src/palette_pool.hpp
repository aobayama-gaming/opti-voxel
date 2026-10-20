#pragma once

#include "chunk_palette.hpp"
#include "page_format.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/rid.hpp>

#include <memory>
#include <vector>

using namespace godot;

// The loaded chunks' material palettes (chunk_palette.hpp) on the main RenderingDevice, for the page shader: one
// RGBA16UI texture, PALETTE_TEX_WIDTH texels (one whole palette) per row, 2 global ids per texel (lo, hi words of
// each). A chunk's block offset (in entries) rides in its rows' group headers (PageFormat), a corner's palette
// index is relative to it. A chunk whose palette is just {GROUND} uses the shared ground block (no allocation).
//
// Main thread: place_batch / release (the allocator). Render thread: rt_* (the texture). The batch payload is
// uploaded by the direct mesher's reserve pass (GLSL_DIRECT_PASS_RESERVE), so a block is written before the emit
// pass of the same batch names it in a header.
class PalettePool {
public:
    // A chunk's block: where its palette is in the texture, and the palette (the CPU mirror the validator and the
    // auditor read).
    struct Block {
        uint32_t offset = PaletteAlloc::GROUND_BLOCK;
        uint32_t entries = 1;
        std::shared_ptr<const std::vector<uint32_t>> palette;
    };

    // The batch payload's layout (uint words): per chunk of the batch its block offset, then the number of texels
    // to write, then BATCH_TEXEL_WORDS words per texel (texel index, id of entry 2t, id of entry 2t + 1, 0).
    static constexpr uint32_t BATCH_TEXEL_WORDS = 4;
    static uint64_t batch_bytes(uint32_t p_batch_size);

    // p_capacity: palette entries (a power of two).
    void initialize(uint32_t p_capacity);
    // Places each chunk's palette (consumed) and returns the batch payload. A palette that finds no room draws as
    // ground (ERR once, counted).
    PackedByteArray place_batch(std::vector<std::vector<uint32_t>> &p_palettes, uint32_t p_batch_size, std::vector<Block> &r_blocks);
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
    RID texture;
    RID rs_texture;
};
