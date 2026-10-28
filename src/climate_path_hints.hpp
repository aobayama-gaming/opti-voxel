#pragma once

#include "lod_structures.hpp"
#include "ov/bytes.hpp"
#include "world_coord.hpp"

#include <cstdint>
#include <vector>

// The climate modes a mesher gives its chunks (Cascade::CLIMATE_*, the region paths' GLSL): learned from what the GPU
// reported for the chunks meshed before. A chunk meshed in CLIMATE_UNKNOWN reports the paths whose test held at every
// point of its level 1; a hinted one, whether its path failed somewhere. A chunk takes the mode of the nearest of itself
// and its ancestors (up to ANCESTORS LODs up) with a report: its first clean path (CLIMATE_REDUCE too when asked), the
// whole table where none was clean (CLIMATE_BORDER), CLIMATE_UNKNOWN without one. While flying, the ground ahead was
// meshed coarser before, so its chunks' parents have reports. The modes only choose the code: a path's test is still made
// at every point (A1 exact), a reduced chunk whose path failed is counted (`failed`).
class ClimatePathHints {
public:
    static constexpr int ANCESTORS = 4;
    static constexpr size_t MAX_ENTRIES = size_t(1) << 20; // reports kept, then forgotten at once

    enum Level {
        OFF = 0, // every chunk CLIMATE_UNKNOWN: each tries every path
        HINTS = 1, // learned modes, the whole table where no path was clean
        REDUCE = 2, // and a clean path's chunks carry its groups only (A2)
    };

    // p_paths: the climate's region path count (0: no climate words, modes() leaves r_modes empty).
    void configure(int p_paths, Level p_level);
    bool active() const { return paths > 0; }

    // Each chunk's mode, in order.
    void modes(const std::vector<ChunkId> &p_chunks, std::vector<uint32_t> &r_modes);
    // The batch's climate words (CascadeGLSL::CLIMATE_ITEM_BYTES per chunk, in p_chunks' order) and the modes it ran with.
    void learn(const std::vector<ChunkId> &p_chunks, const std::vector<uint32_t> &p_modes, const ov::bytes &p_words);

    struct Stats {
        uint64_t unknown = 0, path = 0, reduced = 0, border = 0; // chunks given each mode
        uint64_t reports = 0; // CLIMATE_UNKNOWN chunks that reported
        uint64_t clean = 0; // of them, with a clean path
        uint64_t failed = 0; // hinted chunks whose path failed (reduced ones: wrong samples)
        uint64_t failed_reduced = 0;
    };
    const Stats &get_stats() const { return stats; }

private:
    int paths = 0;
    Level level = OFF;
    ov::HashMap<ChunkId, uint32_t, Vector3i64Hasher> clean; // per chunk: the paths clean at it (bit p), 0: none
    Stats stats;

    uint32_t mode_of(const ChunkId &p_chunk) const;
};
