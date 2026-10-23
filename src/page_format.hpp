#pragma once

#include "world_coord.hpp"

#include <cmath>
#include <cstdint>

// Page texture layout (8 bytes per corner), shared by the compute passes (GLSL_PAGE_FORMAT), the
// page shader (quad_page_pool.cpp) and the CPU-side readers (validator, auditor, spike).
//
// One RGBA16UI texture per page (RenderingServer cannot wrap 32-bit integer RD textures), one row
// per group of QUAD_GROUP quads:
//   texel 0          (x lo, x hi, y lo, y hi)  group header: low 32 bits of the id of the chunk owning
//   texel 1          (z lo, z hi, 0, 0)        every quad of the row (chunks start on a group, so rows
//                                               never mix chunks)
//   texel 2 + 4q + k corner k of quad q
// A corner is its position in the chunk's vertex grid (the grid the SDF is sampled on, skirt
// included) as 3 x 14-bit fixed point, and its normal n = nx | ny << 11 as 2 x 11-bit octahedral:
//   (x | (n & 3) << 14, y | (n >> 2 & 3) << 14, z | (n >> 4 & 3) << 14, n >> 6)
// Render positions are rebuilt from the header (relative to the world origin, wrapping int32) with
// the same formula as ChunkMath::grid_to_render_lo32. An all-zero row is empty: its corners collapse
// to one point.
namespace PageFormat {

static constexpr uint32_t QUAD_GROUP = 64;
static constexpr uint32_t HEADER_TEXELS = 2;
static constexpr uint32_t TEX_WIDTH = HEADER_TEXELS + QUAD_GROUP * 4; // texels per row
static constexpr uint32_t TEXEL_CHANNELS = 4; // uint16 each
static constexpr uint32_t CORNER_BYTES = 8;
static constexpr uint32_t ROW_BYTES = TEX_WIDTH * CORNER_BYTES;
static constexpr uint32_t POS_BITS = 14;
static constexpr uint32_t NORMAL_BITS = 11;

// Fractional bits of the fixed-point position: vertex coordinates are in [0, chunk_size - 1].
// chunk_size 32 -> 9 bits, a step of 1/512 cell.
inline int pos_frac_bits(int p_chunk_size) {
    int int_bits = 0;
    while ((1 << int_bits) <= p_chunk_size - 1) {
        int_bits++;
    }
    return POS_BITS - int_bits;
}

inline float pos_scale(int p_chunk_size) {
    return float(1 << pos_frac_bits(p_chunk_size));
}

inline const uint16_t *row(const uint16_t *p_page, uint32_t p_group) {
    return p_page + p_group * TEX_WIDTH * TEXEL_CHANNELS;
}

inline ov::ivec3 header_chunk(const uint16_t *p_row) {
    return ov::ivec3((int32_t)(p_row[0] | (uint32_t)p_row[1] << 16), (int32_t)(p_row[2] | (uint32_t)p_row[3] << 16),
            (int32_t)(p_row[4] | (uint32_t)p_row[5] << 16));
}

// Corner k of quad q (q = slot within the page) as its four channels.
inline const uint16_t *corner(const uint16_t *p_page, uint32_t p_quad, int p_k) {
    return row(p_page, p_quad / QUAD_GROUP) + (HEADER_TEXELS + (p_quad % QUAD_GROUP) * 4 + p_k) * TEXEL_CHANNELS;
}

inline Float3 decode_local(const uint16_t *p_corner, float p_pos_scale) {
    return Float3{ float(p_corner[0] & 16383u) / p_pos_scale, float(p_corner[1] & 16383u) / p_pos_scale,
        float(p_corner[2] & 16383u) / p_pos_scale };
}

inline Float3 decode_normal(const uint16_t *p_corner) {
    const uint32_t p = (p_corner[0] >> 14) | (p_corner[1] >> 14) << 2 | (p_corner[2] >> 14) << 4 | (uint32_t)p_corner[3] << 6;
    const float fx = float(p & 2047u) / 2047.0f * 2.0f - 1.0f;
    const float fy = float(p >> 11) / 2047.0f * 2.0f - 1.0f;
    const float fz = 1.0f - std::abs(fx) - std::abs(fy);
    const float t = fz < 0.0f ? -fz : 0.0f;
    const float nx = fx + (fx >= 0.0f ? -t : t);
    const float ny = fy + (fy >= 0.0f ? -t : t);
    return Float3{ nx, ny, fz }.normalized();
}

inline bool is_zero(const uint16_t *p_corner) {
    return p_corner[0] == 0 && p_corner[1] == 0 && p_corner[2] == 0 && p_corner[3] == 0;
}

} // namespace PageFormat

// GLSL side: the staging image binding and the encoders. Needs QUAD_GROUP defined, and
// pos_scale from page_pos_scale(chunk_size).
inline constexpr const char GLSL_PAGE_FORMAT[] = R"glsl(

#define HEADER_TEXELS 2u
#define TEX_WIDTH (HEADER_TEXELS + QUAD_GROUP * 4u)

float page_pos_scale(int chunk_size) {
    return float(1 << (14 - (findMSB(chunk_size - 1) + 1)));
}

vec2 _page_oct_sign(vec2 v) {
    return vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

uint pack_normal_oct22(vec3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
    vec2 o = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * _page_oct_sign(n.xy);
    uvec2 q = uvec2(round(clamp(o * 0.5 + 0.5, 0.0, 1.0) * 2047.0));
    return q.x | (q.y << 11u);
}

uvec4 encode_corner(vec3 local, uint normal22, float pos_scale) {
    uvec3 q = uvec3(clamp(round(local * pos_scale), 0.0, 16383.0));
    return uvec4(q.x | ((normal22 & 3u) << 14u), q.y | (((normal22 >> 2u) & 3u) << 14u),
            q.z | (((normal22 >> 4u) & 3u) << 14u), normal22 >> 6u);
}

void store_corner(uint slot, uint k, uvec4 corner) {
    imageStore(staging, ivec2(HEADER_TEXELS + (slot % QUAD_GROUP) * 4u + k, slot / QUAD_GROUP), corner);
}

void store_group_header(uint group, ivec3 chunk) {
    uvec3 c = uvec3(chunk);
    imageStore(staging, ivec2(0, group), uvec4(c.x & 65535u, c.x >> 16u, c.y & 65535u, c.y >> 16u));
    imageStore(staging, ivec2(1, group), uvec4(c.z & 65535u, c.z >> 16u, 0u, 0u));
}

)glsl";
