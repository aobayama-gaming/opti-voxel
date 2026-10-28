#pragma once

#include "lod_structures.hpp"
#include "ov/bytes.hpp"
#include "world_coord.hpp"

#include <cstdint>
#include <vector>

// The climate modes a mesher gives its chunks (Cascade::CLIMATE_*, the region guards' GLSL at a chunk's level 1): every
// chunk reports the guarded regions on at some point of its level 1; with `reduce`, a chunk then takes CLIMATE_REDUCE
// with the regions its nearest reported self or ancestor (up to ANCESTORS LODs up) had on, CLIMATE_FULL without one.
// While flying, the ground ahead was meshed coarser before, so its chunks' parents have reports. A reduced chunk with
// another region on is counted (`failed`: its samples are wrong, the experiment's cost) and forgotten.
class ClimateChunkModes {
public:
    static constexpr int ANCESTORS = 4;
    static constexpr size_t MAX_ENTRIES = size_t(1) << 20; // reports kept, then forgotten at once

    // p_active: the climate has region guards (else modes() leaves r_modes empty: no climate words).
    void configure(bool p_active, bool p_reduce);
    bool active() const { return active_; }

    // Each chunk's mode, in order.
    void modes(const std::vector<ChunkId> &p_chunks, std::vector<uint32_t> &r_modes);
    // The batch's climate words (CascadeGLSL::CLIMATE_ITEM_BYTES per chunk, in p_chunks' order) and the modes it ran with.
    void learn(const std::vector<ChunkId> &p_chunks, const std::vector<uint32_t> &p_modes, const ov::bytes &p_words);

    struct Stats {
        uint64_t full = 0, reduced = 0; // chunks given each mode
        uint64_t reports = 0; // chunks whose words came back
        uint64_t regions_on = 0; // the regions on at their level 1, summed
        uint64_t failed = 0; // reduced chunks with another region on (wrong samples)
    };
    const Stats &get_stats() const { return stats; }

private:
    bool active_ = false;
    bool reduce = false;
    ov::HashMap<ChunkId, uint32_t, Vector3i64Hasher> on; // per chunk: the regions on at its level 1 (bit r)
    Stats stats;

    uint32_t mode_of(const ChunkId &p_chunk) const;
};
