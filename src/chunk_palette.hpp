#pragma once

#include "page_format.hpp"

#include <cstdint>
#include <set>
#include <vector>

// A chunk's material palette: the distinct global materials (material_library.hpp) its samples can take, so a
// corner stores a PageFormat::PALETTE_BITS index into it instead of a global id. Entry 0 is always
// Material::GROUND (a sample no instance won), then the listed records' materials in list order (lists are in
// record id order, so a chunk's palette is deterministic).
namespace ChunkPalette {

// A list word's palette index for a Material::INHERIT record: the fold skips it, so no sample and no corner holds it.
// The palette itself has one entry less.
static constexpr uint32_t INHERIT_INDEX = PageFormat::PALETTE_SIZE - 1;
static constexpr uint32_t MAX_ENTRIES = PageFormat::PALETTE_SIZE - 1;

// p_materials[i]: the material of the chunk's list entry i. r_palette: the global ids; r_index[i]: entry i's
// palette index (INHERIT_INDEX for Material::INHERIT, which takes no entry). Returns false when the chunk names
// more than MAX_ENTRIES materials: the entries past the palette take index 0 (they draw as ground).
bool build(const uint32_t *p_materials, uint32_t p_count, std::vector<uint32_t> &r_palette, std::vector<uint16_t> &r_index);

} // namespace ChunkPalette

// Allocates the palettes' blocks in the GPU palette texture: a buddy allocator over [0, capacity) entries, blocks
// of MIN_BLOCK << order entries, aligned to their size (a block never straddles a texture row of MAX_BLOCK
// entries). The first MIN_BLOCK entries are reserved: the shared {GROUND} palette (GROUND_BLOCK), which needs no
// upload since the texture starts zeroed.
class PaletteAlloc {
public:
    static constexpr uint32_t MIN_BLOCK = 16;
    static constexpr uint32_t MAX_BLOCK = PageFormat::PALETTE_SIZE; // a whole palette
    static constexpr uint32_t GROUND_BLOCK = 0;

    // p_capacity: a power of two, at least 2 * MAX_BLOCK entries.
    void reset(uint32_t p_capacity);
    // The offset of a block holding p_entries (1..MAX_BLOCK), or -1 when full.
    int64_t alloc(uint32_t p_entries);
    // Frees a block from alloc() (the same p_entries). GROUND_BLOCK is never freed.
    void free(uint32_t p_offset, uint32_t p_entries);

    static uint32_t block_size(uint32_t p_entries) { return MIN_BLOCK << order_for(p_entries); }
    uint32_t capacity() const { return cap; }
    uint32_t used() const { return used_entries; } // in block entries, the reserved block included

private:
    // The block order holding p_entries: ceil(log2(p_entries / MIN_BLOCK)), at least 0.
    static int order_for(uint32_t p_entries);
    uint32_t cap = 0;
    uint32_t used_entries = 0;
    std::vector<std::set<uint32_t>> free_lists; // per order: free block offsets
};
