#pragma once

#include <cstdint>

// Global materials: what a chunk palette entry names (chunk_palette.hpp). A record carries one (TableRecord::material)
// or INHERIT, the ground is GROUND. The page shader reads a material's parameters from make_image() by its id.
namespace Material {

// Built-in materials, registered in this order by Library's constructor. GROUND is the page shader's procedural
// ground colours (slope, height, contours).
enum Builtin : uint32_t {
    GROUND = 0,
    BARK,
    LEAVES,
    LAVA,
    SCORCHED_ROCK,
    ISLAND_ROCK,
    BUILTIN_COUNT,
};

// Ids stay below 2^ID_BITS: the legacy mesher (the oracle) carries them through a float vertex channel, and the page
// shader packs a 4-bit corner mask above one (v_materials).
static constexpr uint32_t ID_BITS = 24;
static constexpr uint32_t MAX_MATERIALS = 1u << ID_BITS;
static_assert(ID_BITS + 4 <= 31, "the page shader's id | corner mask << ID_BITS fits a positive int");
// A record's material when it takes its library type's (DefLibrary::material), resolved by SDFTable::finish_record.
static constexpr uint32_t FROM_TYPE = UINT32_MAX;
// A cutter that colours nothing: where it wins a sample, the sample keeps the material it had (its face shows what it
// cut). The generic shape types' default (DefLibrary::add); on a union it is ground (Archetype::record_material).
static constexpr uint32_t INHERIT = UINT32_MAX - 1;

} // namespace Material
