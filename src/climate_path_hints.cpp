#include "climate_path_hints.hpp"

#include "cascade.hpp"
#include "cascade_glsl.hpp"
#include "chunk_math.hpp"

void ClimatePathHints::configure(int p_paths, Level p_level) {
    paths = p_paths;
    level = p_level;
    clean.clear();
    stats = Stats();
}

uint32_t ClimatePathHints::mode_of(const ChunkId &p_chunk) const {
    if (level == OFF) {
        return Cascade::CLIMATE_UNKNOWN;
    }
    ChunkId c = p_chunk;
    for (int up = 0; up <= ANCESTORS; up++) {
        if (const Entry *e = clean.getptr(c)) {
            if (e->some == 0u) {
                return Cascade::CLIMATE_BORDER; // no point took a path: the tests are pure cost
            }
            if (e->every == 0u) {
                return Cascade::CLIMATE_UNKNOWN; // some points did: per point
            }
            const uint32_t path = uint32_t(__builtin_ctz(e->every)) + 1u;
            return level == REDUCE ? path | Cascade::CLIMATE_REDUCE : path;
        }
        c = ChunkMath::get_parent_from_child(c);
    }
    return Cascade::CLIMATE_UNKNOWN;
}

void ClimatePathHints::modes(const std::vector<ChunkId> &p_chunks, std::vector<uint32_t> &r_modes) {
    r_modes.clear();
    if (!active()) {
        return;
    }
    r_modes.reserve(p_chunks.size());
    for (const ChunkId &c : p_chunks) {
        const uint32_t m = mode_of(c);
        r_modes.push_back(m);
        const uint32_t path = m & Cascade::CLIMATE_PATH_MASK;
        if (path == Cascade::CLIMATE_UNKNOWN) {
            stats.unknown++;
        } else if (path == Cascade::CLIMATE_BORDER) {
            stats.border++;
        } else if ((m & Cascade::CLIMATE_REDUCE) != 0u) {
            stats.reduced++;
        } else {
            stats.path++;
        }
    }
}

void ClimatePathHints::learn(const std::vector<ChunkId> &p_chunks, const std::vector<uint32_t> &p_modes, const ov::bytes &p_words) {
    if (!active() || p_modes.size() != p_chunks.size() || p_words.size() < p_chunks.size() * CascadeGLSL::CLIMATE_ITEM_BYTES) {
        return;
    }
    if (clean.size() > MAX_ENTRIES) {
        clean.clear();
    }
    const uint32_t all = paths >= 32 ? ~0u : (1u << paths) - 1u;
    for (size_t i = 0; i < p_chunks.size(); i++) {
        const uint8_t *w = p_words.data() + i * CascadeGLSL::CLIMATE_ITEM_BYTES;
        const uint32_t every = ov::get_u32(w + 4);
        const bool fail = ov::get_u32(w + 8) != 0u;
        const uint32_t some = ov::get_u32(w + 12);
        const uint32_t path = p_modes[i] & Cascade::CLIMATE_PATH_MASK;
        if (path == Cascade::CLIMATE_UNKNOWN) {
            if (every == ~0u) {
                continue; // its level 1 read no climate: nothing learned
            }
            stats.reports++;
            stats.clean += (every & all) != 0u ? 1u : 0u;
            stats.none += (some & all) == 0u ? 1u : 0u;
            clean.insert(p_chunks[i], Entry{ every & all, some & all });
        } else if (path != Cascade::CLIMATE_BORDER && fail) {
            stats.failed++;
            stats.failed_reduced += (p_modes[i] & Cascade::CLIMATE_REDUCE) != 0u ? 1u : 0u;
            clean.insert(p_chunks[i], Entry{ 0u, ~0u }); // some point held no path: per point from now on
        }
    }
}
