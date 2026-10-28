#include "climate_chunk_modes.hpp"

#include "cascade.hpp"
#include "cascade_glsl.hpp"
#include "chunk_math.hpp"

void ClimateChunkModes::configure(bool p_active) {
    active_ = p_active;
    on.clear();
    stats = Stats();
}

uint32_t ClimateChunkModes::mode_of(const ChunkId &p_chunk) const {
    ChunkId c = p_chunk;
    for (int up = 0; up <= ANCESTORS; up++) {
        if (const uint32_t *r = on.getptr(c)) {
            return Cascade::CLIMATE_REDUCE | (*r & Cascade::CLIMATE_REGION_MASK);
        }
        c = ChunkMath::get_parent_from_child(c);
    }
    return Cascade::CLIMATE_FULL;
}

void ClimateChunkModes::modes(const std::vector<ChunkId> &p_chunks, std::vector<uint32_t> &r_modes) {
    r_modes.clear();
    if (!active_) {
        return;
    }
    r_modes.reserve(p_chunks.size());
    for (const ChunkId &c : p_chunks) {
        const uint32_t m = mode_of(c);
        r_modes.push_back(m);
        ((m & Cascade::CLIMATE_REDUCE) != 0u ? stats.reduced : stats.full)++;
    }
}

void ClimateChunkModes::learn(const std::vector<ChunkId> &p_chunks, const std::vector<uint32_t> &p_modes, const ov::bytes &p_words) {
    if (!active_ || p_words.size() < p_chunks.size() * CascadeGLSL::CLIMATE_ITEM_BYTES) {
        return;
    }
    if (on.size() > MAX_ENTRIES) {
        on.clear();
    }
    for (size_t i = 0; i < p_chunks.size(); i++) {
        const uint8_t *w = p_words.data() + i * CascadeGLSL::CLIMATE_ITEM_BYTES;
        const uint32_t regions = ov::get_u32(w + 4);
        stats.reports++;
        stats.regions_on += uint64_t(WorldCoord::popcount64(regions));
        if ((p_modes[i] & Cascade::CLIMATE_REDUCE) != 0u && ov::get_u32(w + 8) != 0u) {
            stats.failed++;
        }
        on.insert(p_chunks[i], regions); // what it really had on: the next mode of it and its children
    }
}
