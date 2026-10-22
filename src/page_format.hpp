#pragma once

#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <cmath>
#include <cstdint>

using namespace godot;

// Page texture layout (8 bytes per corner), shared by the compute passes (PageFormat::glsl_encode), the
// page shader (PageFormat::glsl_decode, quad_page_pool.cpp) and the CPU-side readers (validator, auditor, spike).
//
// One RGBA16UI texture per page (RenderingServer cannot wrap 32-bit integer RD textures), one row
// per group of QUAD_GROUP quads:
//   texel 0          (x lo, x hi, y lo, y hi)  group header: low 32 bits of the id of the chunk owning
//   texel 1          (z lo, z hi, b lo, b hi)  every quad of the row (chunks start on a group, so rows
//                                               never mix chunks), and b the offset of the chunk's palette
//                                               block in the palette texture (palette_pool.hpp)
//   texel 2 + 4q + k corner k of quad q
// A corner is its position in the chunk's vertex grid (the grid the SDF is sampled on, skirt
// included) as 3 x POS_BITS fixed point, a PALETTE_BITS palette index p spread over the top bits of the
// three position channels, and its normal n = nx | ny << 8 as 2 x 8-bit octahedral:
//   (x | (p & 7) << 13, y | (p >> 3 & 7) << 13, z | (p >> 6 & 7) << 13, n)
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
static constexpr uint32_t POS_BITS = 13;
static constexpr uint32_t POS_MASK = (1u << POS_BITS) - 1;
static constexpr uint32_t NORMAL_AXIS_BITS = 8;
static constexpr uint32_t NORMAL_AXIS_MAX = (1u << NORMAL_AXIS_BITS) - 1;
static constexpr uint32_t NORMAL_BITS = 2 * NORMAL_AXIS_BITS;
// Worst round-trip angle of an octahedral normal at NORMAL_AXIS_BITS (measured on 2M random directions: 0.94
// degrees max, 0.34 mean at 8 bits).
static constexpr float NORMAL_MAX_ERROR_DEG = 1.0f;
static constexpr uint32_t PALETTE_BITS = 9;
static constexpr uint32_t PALETTE_LANE_BITS = PALETTE_BITS / 3; // per position channel
static constexpr uint32_t PALETTE_SIZE = 1u << PALETTE_BITS;
// The palette texture (palette_pool.hpp): RGBA16UI, two 32-bit global ids per texel, a whole palette per row.
static constexpr uint32_t PALETTE_TEX_WIDTH = PALETTE_SIZE / 2;
static_assert(POS_BITS + PALETTE_LANE_BITS == 16 && NORMAL_BITS == 16 && 3 * PALETTE_LANE_BITS == PALETTE_BITS,
        "a corner is exactly four 16-bit channels");

// Fractional bits of the fixed-point position: vertex coordinates are in [0, chunk_size - 1].
// chunk_size 32 -> 8 bits, a step of 1/256 cell.
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

inline Vector3i header_chunk(const uint16_t *p_row) {
    return Vector3i((int32_t)(p_row[0] | (uint32_t)p_row[1] << 16), (int32_t)(p_row[2] | (uint32_t)p_row[3] << 16),
            (int32_t)(p_row[4] | (uint32_t)p_row[5] << 16));
}

inline uint32_t header_palette(const uint16_t *p_row) {
    return uint32_t(p_row[6]) | uint32_t(p_row[7]) << 16;
}

// The palette texture's channels (TEXEL_CHANNELS per texel, row-major rows of PALETTE_TEX_WIDTH): entry p_e holds p_id,
// two 16-bit words per id, two ids per texel (the GLSL page_palette_texel_value's layout).
inline void encode_palette_entry(uint16_t *r_texture, uint32_t p_e, uint32_t p_id) {
    r_texture[2 * size_t(p_e)] = uint16_t(p_id & 0xFFFFu);
    r_texture[2 * size_t(p_e) + 1] = uint16_t(p_id >> 16);
}

// Corner k of quad q (q = slot within the page) as its four channels.
inline const uint16_t *corner(const uint16_t *p_page, uint32_t p_quad, int p_k) {
    return row(p_page, p_quad / QUAD_GROUP) + (HEADER_TEXELS + (p_quad % QUAD_GROUP) * 4 + p_k) * TEXEL_CHANNELS;
}

inline Vector3 decode_local(const uint16_t *p_corner, float p_pos_scale) {
    return Vector3(float(p_corner[0] & POS_MASK), float(p_corner[1] & POS_MASK), float(p_corner[2] & POS_MASK)) / p_pos_scale;
}

inline uint32_t decode_palette(const uint16_t *p_corner) {
    return uint32_t(p_corner[0] >> POS_BITS) | uint32_t(p_corner[1] >> POS_BITS) << PALETTE_LANE_BITS |
            uint32_t(p_corner[2] >> POS_BITS) << (2 * PALETTE_LANE_BITS);
}

inline Vector3 decode_normal(const uint16_t *p_corner) {
    const uint32_t p = p_corner[3];
    const float fx = float(p & NORMAL_AXIS_MAX) / float(NORMAL_AXIS_MAX) * 2.0f - 1.0f;
    const float fy = float(p >> NORMAL_AXIS_BITS) / float(NORMAL_AXIS_MAX) * 2.0f - 1.0f;
    Vector3 n(fx, fy, 1.0f - std::abs(fx) - std::abs(fy));
    const float t = n.z < 0.0f ? -n.z : 0.0f;
    n.x += n.x >= 0.0f ? -t : t;
    n.y += n.y >= 0.0f ? -t : t;
    return n.normalized();
}

inline bool is_zero(const uint16_t *p_corner) {
    return p_corner[0] == 0 && p_corner[1] == 0 && p_corner[2] == 0 && p_corner[3] == 0;
}

// The constants above as GLSL (compute shaders and the page shader: `const`, no #define).
inline String glsl_constants() {
    String s;
    const struct {
        const char *name;
        uint32_t value;
    } constants[] = { { "QUAD_GROUP", QUAD_GROUP }, { "HEADER_TEXELS", HEADER_TEXELS }, { "TEX_WIDTH", TEX_WIDTH },
        { "PAGE_POS_BITS", POS_BITS }, { "PAGE_POS_MASK", POS_MASK }, { "PAGE_NORMAL_AXIS_BITS", NORMAL_AXIS_BITS },
        { "PAGE_NORMAL_AXIS_MAX", NORMAL_AXIS_MAX }, { "PAGE_NORMAL_BITS", NORMAL_BITS },
        { "PAGE_PALETTE_LANE_BITS", PALETTE_LANE_BITS }, { "PALETTE_TEX_WIDTH", PALETTE_TEX_WIDTH } };
    for (const auto &c : constants) {
        s += String("const uint ") + c.name + " = " + itos(c.value) + "u;\n";
    }
    return s;
}

} // namespace PageFormat

// Page shader side (after PageFormat::glsl_constants()): the corner decoders.
inline constexpr const char GLSL_PAGE_DECODE[] = R"glsl(

vec3 page_decode_local(uvec4 c, float pos_scale) {
    return vec3(float(c.x & PAGE_POS_MASK), float(c.y & PAGE_POS_MASK), float(c.z & PAGE_POS_MASK)) / pos_scale;
}

vec3 page_decode_normal(uvec4 c) {
    vec2 f = vec2(float(c.w & PAGE_NORMAL_AXIS_MAX), float(c.w >> PAGE_NORMAL_AXIS_BITS)) / float(PAGE_NORMAL_AXIS_MAX) * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

uint page_decode_palette(uvec4 c) {
    return (c.x >> PAGE_POS_BITS) | ((c.y >> PAGE_POS_BITS) << PAGE_PALETTE_LANE_BITS) |
            ((c.z >> PAGE_POS_BITS) << (2u * PAGE_PALETTE_LANE_BITS));
}

// The palette block offset in header texel 1 (palette_pool.hpp), the GLSL header_palette.
uint page_header_palette(uvec4 h1) {
    return h1.z | (h1.w << 16u);
}

// Palette entry e (block offset + palette index): the palette texel holding it, and its id in that texel.
ivec2 page_palette_texel(uint e) {
    uint t = e >> 1u;
    return ivec2(int(t % PALETTE_TEX_WIDTH), int(t / PALETTE_TEX_WIDTH));
}

uint page_palette_id(uvec4 texel, uint e) {
    return (e & 1u) == 0u ? (texel.x | (texel.y << 16u)) : (texel.z | (texel.w << 16u));
}

)glsl";

// Compute side (after PageFormat::glsl_constants() and the staging image binding): the encoders. pos_scale
// comes from page_pos_scale(chunk_size).
inline constexpr const char GLSL_PAGE_ENCODE[] = R"glsl(

float page_pos_scale(int chunk_size) {
    return float(1 << (int(PAGE_POS_BITS) - (findMSB(chunk_size - 1) + 1)));
}

vec2 _page_oct_sign(vec2 v) {
    return vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

uint pack_normal_oct(vec3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
    vec2 o = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * _page_oct_sign(n.xy);
    uvec2 q = uvec2(round(clamp(o * 0.5 + 0.5, 0.0, 1.0) * float(PAGE_NORMAL_AXIS_MAX)));
    return q.x | (q.y << PAGE_NORMAL_AXIS_BITS);
}

uvec4 encode_corner(vec3 local, uint normal, uint palette, float pos_scale) {
    uvec3 q = uvec3(clamp(round(local * pos_scale), 0.0, float(PAGE_POS_MASK)));
    uint lane = (1u << PAGE_PALETTE_LANE_BITS) - 1u;
    uvec3 p = uvec3(palette & lane, (palette >> PAGE_PALETTE_LANE_BITS) & lane, (palette >> (2u * PAGE_PALETTE_LANE_BITS)) & lane);
    return uvec4(q | (p << PAGE_POS_BITS), normal);
}

// The palette texel holding ids id0 (even entry) and id1 (odd entry): page_palette_id's inverse.
uvec4 page_palette_texel_value(uint id0, uint id1) {
    return uvec4(id0 & 65535u, id0 >> 16u, id1 & 65535u, id1 >> 16u);
}

void store_corner(uint slot, uint k, uvec4 corner) {
    imageStore(staging, ivec2(HEADER_TEXELS + (slot % QUAD_GROUP) * 4u + k, slot / QUAD_GROUP), corner);
}

void store_group_header(uint group, ivec3 chunk, uint palette_block) {
    uvec3 c = uvec3(chunk);
    imageStore(staging, ivec2(0, group), uvec4(c.x & 65535u, c.x >> 16u, c.y & 65535u, c.y >> 16u));
    imageStore(staging, ivec2(1, group), uvec4(c.z & 65535u, c.z >> 16u, palette_block & 65535u, palette_block >> 16u));
}

)glsl";

namespace PageFormat {

// Everything a compute pass writing pages needs, after its staging image binding (the decoders too: palette addressing).
inline String glsl_encode() {
    return glsl_constants() + GLSL_PAGE_DECODE + GLSL_PAGE_ENCODE;
}

// Everything the page shader needs to read corners.
inline String glsl_decode() {
    return glsl_constants() + GLSL_PAGE_DECODE;
}

} // namespace PageFormat
