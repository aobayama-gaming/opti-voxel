#pragma once

// Engine-free vocabulary types for the core (world_coord, page_format, chunk_math, mesh-independent
// code). Nothing in src/ov/ may include godot_cpp: the same core compiles into other engines and into
// a web build. Enforced by tools/check_core_purity.sh.

#include <cstdint>

namespace ov {

// int32 pair: a (smallest, largest) range where godot had a Vector2i (ChunkMath::lod_range).
struct ivec2 {
    int32_t x = 0;
    int32_t y = 0;

    constexpr ivec2() = default;
    constexpr ivec2(int32_t p_x, int32_t p_y) :
            x(p_x), y(p_y) {}

    constexpr bool operator==(const ivec2 &p_o) const { return x == p_o.x && y == p_o.y; }
    constexpr bool operator!=(const ivec2 &p_o) const { return !(*this == p_o); }
};

// int32 triple, the core's word for what used to be godot's Vector3i at pure-core edges (page
// headers, shader uniforms). Same field names and comparison ops as godot's Vector3i: only the
// arithmetic godot gave it is not carried over, because the core never adds two int32 triples. The
// conversion back to godot's Vector3i stays manual at engine call sites (explicit field copies), in
// the spirit of WorldCoord::lo32_i ("a leftover int32 use must fail to compile").
struct ivec3 {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    constexpr ivec3() = default;
    constexpr ivec3(int32_t p_x, int32_t p_y, int32_t p_z) :
            x(p_x), y(p_y), z(p_z) {}

    constexpr bool operator==(const ivec3 &p_o) const { return x == p_o.x && y == p_o.y && z == p_o.z; }
    constexpr bool operator!=(const ivec3 &p_o) const { return !(*this == p_o); }
};

} // namespace ov
