#pragma once

#include <cstdint>

// A chunk split in SUBCUBE_AXIS^3 sub-cubes, one bit each in a 64-bit mask: the coverage of a stalled chunk
// (VoxelState, ChunkMath::subcube_bits) and the hidden mask of its page rows (page_format.hpp).
namespace Subcube {

static constexpr uint32_t AXIS_BITS = 2;
static constexpr uint32_t AXIS = 1u << AXIS_BITS; // sub-cubes per axis
static_assert(AXIS * AXIS * AXIS == 64, "the sub-cubes are one 64-bit mask");

// Bit of sub-cube (x, y, z).
constexpr uint32_t bit(uint32_t p_x, uint32_t p_y, uint32_t p_z) {
    return p_x + AXIS * (p_y + AXIS * p_z);
}

} // namespace Subcube
