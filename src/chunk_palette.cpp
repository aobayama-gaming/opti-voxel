#include "chunk_palette.hpp"

#include "material_library.hpp"
#include "page_format.hpp"

#include <godot_cpp/core/error_macros.hpp>

static_assert(PaletteAlloc::MAX_BLOCK == PageFormat::PALETTE_SIZE, "a block holds a whole palette");

namespace ChunkPalette {

bool build(const uint32_t *p_materials, uint32_t p_count, std::vector<uint32_t> &r_palette, std::vector<uint16_t> &r_index) {
    r_palette.assign(1, Material::GROUND);
    r_index.resize(p_count);
    bool fits = true;
    for (uint32_t i = 0; i < p_count; i++) {
        const uint32_t m = p_materials[i];
        uint32_t at = 0;
        while (at < r_palette.size() && r_palette[at] != m) { // palettes are short: a scan beats a map
            at++;
        }
        if (at == r_palette.size()) {
            if (r_palette.size() == PageFormat::PALETTE_SIZE) {
                fits = false;
                at = 0;
            } else {
                r_palette.push_back(m);
            }
        }
        r_index[i] = uint16_t(at);
    }
    return fits;
}

} // namespace ChunkPalette

uint32_t PaletteAlloc::block_size(uint32_t p_entries) {
    uint32_t size = MIN_BLOCK;
    while (size < p_entries) {
        size <<= 1;
    }
    return size;
}

int PaletteAlloc::order_of(uint32_t p_size) const {
    int order = 0;
    while ((MIN_BLOCK << order) < p_size) {
        order++;
    }
    return order;
}

void PaletteAlloc::reset(uint32_t p_capacity) {
    ERR_FAIL_COND_MSG(p_capacity < 2 * MAX_BLOCK || (p_capacity & (p_capacity - 1)) != 0, "PaletteAlloc: capacity must be a power of two >= 2 blocks of MAX_BLOCK.");
    cap = p_capacity;
    used_entries = 0;
    free_lists.assign(order_of(cap) + 1, std::set<uint32_t>());
    free_lists.back().insert(0);
    const int64_t ground = alloc(MIN_BLOCK); // the lowest free block: offset 0
    DEV_ASSERT(ground == GROUND_BLOCK);
    (void)ground;
}

int64_t PaletteAlloc::alloc(uint32_t p_entries) {
    ERR_FAIL_COND_V(p_entries == 0 || p_entries > MAX_BLOCK, -1);
    const int want = order_of(block_size(p_entries));
    int order = want;
    while (order < int(free_lists.size()) && free_lists[order].empty()) {
        order++;
    }
    if (order == int(free_lists.size())) {
        return -1;
    }
    uint32_t offset = *free_lists[order].begin(); // the lowest: keeps the used blocks packed
    free_lists[order].erase(free_lists[order].begin());
    while (order > want) { // split, keep the low half
        order--;
        free_lists[order].insert(offset + (MIN_BLOCK << order));
    }
    used_entries += MIN_BLOCK << want;
    return offset;
}

void PaletteAlloc::free(uint32_t p_offset, uint32_t p_entries) {
    ERR_FAIL_COND(p_offset == GROUND_BLOCK || p_offset >= cap);
    int order = order_of(block_size(p_entries));
    used_entries -= MIN_BLOCK << order;
    uint32_t offset = p_offset;
    while (order + 1 < int(free_lists.size())) { // merge with the free buddy
        const uint32_t buddy = offset ^ (MIN_BLOCK << order);
        auto it = free_lists[order].find(buddy);
        if (it == free_lists[order].end()) {
            break;
        }
        free_lists[order].erase(it);
        offset = offset < buddy ? offset : buddy;
        order++;
    }
    free_lists[order].insert(offset);
}
