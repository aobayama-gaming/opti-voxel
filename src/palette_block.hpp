#pragma once

#include <cstdint>
#include <memory>
#include <vector>

// A loaded chunk's block in the palette texture (PalettePool): where its material palette is (in entries) and the
// palette itself (the CPU mirror the validator and the auditor read; a block holds palette->size() entries). Core
// side of a chunk's output (chunk_output.hpp); the texture and its allocator are the engine's (palette_pool.hpp).
struct PaletteBlock {
    // The first block, reserved for the shared base palette (SDFGpuTable::base_palette): never allocated or freed.
    static constexpr uint32_t GROUND = 0;

    uint32_t offset = GROUND;
    std::shared_ptr<const std::vector<uint32_t>> palette;
};
