#pragma once

#include "chunk_palette.hpp"
#include "gpu_device.hpp"
#include "material_library.hpp"
#include "page_format.hpp"
#include "palette_block.hpp"

#include "ov/bytes.hpp"
#include "ov/value.hpp"

#include <memory>
#include <vector>

// The loaded chunks' material palettes (chunk_palette.hpp) on the main device, for the page shader: one
// RGBA16UI texture, PALETTE_TEX_WIDTH texels (one whole palette) per row, 2 global ids per texel (lo, hi words of
// each). A chunk's block offset (in entries) rides in its rows' group headers (PageFormat), a corner's palette
// index is relative to it. A chunk whose palette is just the base one (SDFGpuTable::base_palette) uses the shared ground
// block (no allocation, its entries written at rt_create), and so does a chunk whose palette found no room: its vertices
// then take the base's entries, and a record's index past them material 0 (the direct vertex pass).
//
// Main thread: place_batch / release (the allocator). Device thread: rt_* (the texture). The direct mesher's emit
// pass writes the batch's texels (glsl_batch) as it writes the headers that name them.
class PalettePool {
public:
    using Block = PaletteBlock;

    // The batch payload's layout (uint words): per chunk of the batch its block offset, then the number of texels
    // to write, then BATCH_TEXEL_WORDS words per texel (its even entry, the id of that entry, the id of the next).
    static constexpr uint32_t BATCH_TEXEL_WORDS = 3;
    static uint64_t batch_bytes(uint32_t p_batch_size);
    // The payload's reader for the emit pass: `void write_palette_texel(uint t)` (texel t of the batch's, if any),
    // after the palette_image / palette_batch bindings and PageFormat::glsl_encode().
    static ov::str glsl_batch();

    // p_capacity: palette entries (a power of two).
    void initialize(uint32_t p_capacity, const std::vector<uint32_t> &p_base);
    // Places each chunk's palette (consumed) and returns the batch payload. A palette that finds no room draws as
    // ground (ERR once, counted).
    ov::bytes place_batch(std::vector<std::vector<uint32_t>> &p_palettes, std::vector<Block> &r_blocks);
    void release(const Block &p_block);
    uint32_t rows() const { return alloc.capacity() / PageFormat::PALETTE_SIZE; }
    ov::Value get_stats() const;

    // ---- device thread ----
    // The texture (zeroed: the ground block reads GROUND), sampled by the host's page shader (PageHost).
    void rt_create(ov::gpu::Device *p_device);
    void rt_free(ov::gpu::Device *p_device);
    ov::gpu::Texture rt_texture() const { return texture; }

private:
    PaletteAlloc alloc;
    int64_t blocks_live = 0;
    int64_t full = 0; // palettes that found no room
    // The base palette (SDFGpuTable::base_palette): the ground block's, and the Block of every chunk on it.
    std::shared_ptr<const std::vector<uint32_t>> base = std::make_shared<const std::vector<uint32_t>>(1, Material::GROUND);
    ov::gpu::Texture texture;
};
