#pragma once

#include "world_coord.hpp"

#include "subcube.hpp"

#include <cmath>
#include <cstdint>

// Page texture layout (8 bytes per corner), shared by the compute passes (PageFormat::glsl_encode), the
// page shader (PageFormat::glsl_decode, quad_page_pool.cpp) and the CPU-side readers (validator, auditor, spike).
//
// One RGBA16UI texture per page (RenderingServer cannot wrap 32-bit integer RD textures), one row
// per group of QUAD_GROUP quads:
//   texel 0          (x lo, x hi, y lo, y hi)  group header: low 32 bits of the id of the chunk owning
//   texel 1          (z lo, z hi, b lo, b hi)  every quad of the row (chunks start on a group, so rows
//                                               never mix chunks), and b the offset of the chunk's palette
//                                               block in the palette texture (palette_pool.hpp), its bit 31
//                                               (COVERAGE_FLAG) set when texel 2 hides something
//   texel 2          (h0, h1, h2, h3)          the chunk's hidden sub-cubes, 64 bits h0 | h1 << 16 | ...: the
//                                               chunk split in sub-cubes (subcube.hpp), bit Subcube::bit set
//                                               where its replacements have landed (coverage, VoxelState); a
//                                               quad whose 4 corners are all in hidden sub-cubes isn't drawn.
//                                               0 (what the emit pass, kills and fresh pages write) draws all.
//                                               Only read when the flag is set: rows without one cost no fetch
//   texel 3 + 4q + k corner k of quad q
// A corner is its position in the chunk's vertex grid (the grid the SDF is sampled on, skirt
// included) as 3 x POS_BITS fixed point, a PALETTE_BITS palette index p spread over the top bits of the
// three position channels, and its normal n = nx | ny << 8 as 2 x 8-bit octahedral:
//   (x | (p & 7) << 13, y | (p >> 3 & 7) << 13, z | (p >> 6 & 7) << 13, n)
// Render positions are rebuilt from the header (relative to the world origin, wrapping int32) with
// the same formula as ChunkMath::grid_to_render_lo32. An all-zero row is empty: its corners collapse
// to one point.
namespace PageFormat {

static constexpr uint32_t QUAD_GROUP = 64;
static constexpr uint32_t QUAD_INDEX_BITS = 6; // a quad's index within its group
static_assert(1u << QUAD_INDEX_BITS == QUAD_GROUP, "QUAD_INDEX_BITS indexes a group");
static constexpr uint32_t HEADER_TEXELS = 3;
static constexpr uint32_t COVERAGE_TEXEL = 2; // the hidden sub-cubes (above)
// In texel 1's palette word: palette offsets stay far below it (a texture is at most 16,384 rows of 512 entries).
static constexpr uint32_t COVERAGE_FLAG = 1u << 31;
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

inline ov::ivec3 header_chunk(const uint16_t *p_row) {
    return ov::ivec3((int32_t)(p_row[0] | (uint32_t)p_row[1] << 16), (int32_t)(p_row[2] | (uint32_t)p_row[3] << 16),
            (int32_t)(p_row[4] | (uint32_t)p_row[5] << 16));
}

inline uint32_t header_palette(const uint16_t *p_row) {
    return (uint32_t(p_row[6]) | uint32_t(p_row[7]) << 16) & ~COVERAGE_FLAG;
}

// The palette texture's channels (TEXEL_CHANNELS per texel, row-major rows of PALETTE_TEX_WIDTH): entry p_e holds p_id,
// two 16-bit words per id, two ids per texel (the GLSL page_palette_texel_value's layout).
inline void encode_palette_entry(uint16_t *r_texture, uint32_t p_e, uint32_t p_id) {
    r_texture[2 * size_t(p_e)] = uint16_t(p_id & 0xFFFFu);
    r_texture[2 * size_t(p_e) + 1] = uint16_t(p_id >> 16);
}

inline uint64_t header_hidden(const uint16_t *p_row) {
    if ((p_row[7] & (COVERAGE_FLAG >> 16)) == 0) {
        return 0;
    }
    const uint16_t *h = p_row + COVERAGE_TEXEL * TEXEL_CHANNELS;
    return uint64_t(h[0]) | uint64_t(h[1]) << 16 | uint64_t(h[2]) << 32 | uint64_t(h[3]) << 48;
}

// Header texels 1 and 2 of a chunk's rows (z low word p_z_lo, palette block p_palette) hiding p_mask: what
// QuadPagePool::rt_set_hidden writes, the flag set iff the mask isn't 0.
inline void encode_coverage(uint16_t r_texels[2 * TEXEL_CHANNELS], int32_t p_z_lo, uint32_t p_palette, uint64_t p_mask) {
    const uint32_t z = uint32_t(p_z_lo);
    const uint32_t b = p_palette | (p_mask != 0 ? COVERAGE_FLAG : 0);
    r_texels[0] = uint16_t(z & 65535u);
    r_texels[1] = uint16_t(z >> 16);
    r_texels[2] = uint16_t(b & 65535u);
    r_texels[3] = uint16_t(b >> 16);
    for (uint32_t i = 0; i < TEXEL_CHANNELS; i++) {
        r_texels[TEXEL_CHANNELS + i] = uint16_t(p_mask >> (16 * i));
    }
}
static_assert(COVERAGE_TEXEL == 2, "encode_coverage writes texels 1 and 2");

// Corner k of quad q (q = slot within the page) as its four channels.
inline const uint16_t *corner(const uint16_t *p_page, uint32_t p_quad, int p_k) {
    return row(p_page, p_quad / QUAD_GROUP) + (HEADER_TEXELS + (p_quad % QUAD_GROUP) * 4 + p_k) * TEXEL_CHANNELS;
}

inline Float3 decode_local(const uint16_t *p_corner, float p_pos_scale) {
    return Float3{ float(p_corner[0] & POS_MASK) / p_pos_scale, float(p_corner[1] & POS_MASK) / p_pos_scale,
        float(p_corner[2] & POS_MASK) / p_pos_scale };
}

inline uint32_t decode_palette(const uint16_t *p_corner) {
    return uint32_t(p_corner[0] >> POS_BITS) | uint32_t(p_corner[1] >> POS_BITS) << PALETTE_LANE_BITS |
            uint32_t(p_corner[2] >> POS_BITS) << (2 * PALETTE_LANE_BITS);
}

inline Float3 decode_normal(const uint16_t *p_corner) {
    const uint32_t p = p_corner[3];
    const float fx = float(p & NORMAL_AXIS_MAX) / float(NORMAL_AXIS_MAX) * 2.0f - 1.0f;
    const float fy = float(p >> NORMAL_AXIS_BITS) / float(NORMAL_AXIS_MAX) * 2.0f - 1.0f;
    const float fz = 1.0f - std::abs(fx) - std::abs(fy);
    const float t = fz < 0.0f ? -fz : 0.0f;
    const float nx = fx + (fx >= 0.0f ? -t : t);
    const float ny = fy + (fy >= 0.0f ? -t : t);
    return Float3{ nx, ny, fz }.normalized();
}

inline bool is_zero(const uint16_t *p_corner) {
    return p_corner[0] == 0 && p_corner[1] == 0 && p_corner[2] == 0 && p_corner[3] == 0;
}

// The constants above as GLSL (compute shaders and the page shader: `const`, no #define).
inline ov::str glsl_constants() {
    ov::str s;
    const struct {
        const char *name;
        uint32_t value;
    } constants[] = { { "QUAD_GROUP", QUAD_GROUP }, { "QUAD_INDEX_BITS", QUAD_INDEX_BITS }, { "HEADER_TEXELS", HEADER_TEXELS }, { "TEX_WIDTH", TEX_WIDTH },
        { "PAGE_COVERAGE_TEXEL", COVERAGE_TEXEL }, { "PAGE_COVERAGE_AXIS", Subcube::AXIS },
        { "PAGE_COVERAGE_FLAG", COVERAGE_FLAG },
        { "PAGE_POS_BITS", POS_BITS }, { "PAGE_POS_MASK", POS_MASK }, { "PAGE_NORMAL_AXIS_BITS", NORMAL_AXIS_BITS },
        { "PAGE_NORMAL_AXIS_MAX", NORMAL_AXIS_MAX }, { "PAGE_NORMAL_BITS", NORMAL_BITS },
        { "PAGE_PALETTE_LANE_BITS", PALETTE_LANE_BITS }, { "PALETTE_TEX_WIDTH", PALETTE_TEX_WIDTH } };
    for (const auto &c : constants) {
        s += ov::format("const uint %s = %uu;\n", c.name, c.value);
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
    return (h1.z | (h1.w << 16u)) & ~PAGE_COVERAGE_FLAG;
}

// Whether header texel 2 hides something (PageFormat::COVERAGE_FLAG in texel 1).
bool page_header_has_coverage(uvec4 h1) {
    return (h1.w & (PAGE_COVERAGE_FLAG >> 16u)) != 0u;
}

// Sub-cube s (in [0, PAGE_COVERAGE_AXIS)^3) hidden in header texel 2's mask (Subcube::bit).
bool page_subcube_hidden(uvec4 h2, uvec3 s) {
    uint bit = s.x + PAGE_COVERAGE_AXIS * (s.y + PAGE_COVERAGE_AXIS * s.z);
    uint word = bit < 32u ? (h2.x | (h2.y << 16u)) : (h2.z | (h2.w << 16u));
    return ((word >> (bit & 31u)) & 1u) != 0u;
}

// Header texels 0 and 1: the group's chunk id (low words). LOD from its x (an empty row reads 0: LOD 0).
ivec3 page_header_chunk(uvec4 h0, uvec4 h1) {
    return ivec3(int(h0.x | (h0.y << 16u)), int(h0.z | (h0.w << 16u)), int(h1.x | (h1.y << 16u)));
}

int page_chunk_lod(ivec3 chunk) {
    return chunk.x == 0 ? 0 : findLSB(chunk.x);
}

// ChunkMath::vertices_to_world in render metres. rel: the chunk id minus the world origin's, wrapping int32 (exact while
// the chunk is within 2^31 ids of the origin; the origin is a multiple of the root chunk span, so the low word keeps the
// LOD bit). voxel_dim: chunk_size - 2 * skirt_size. Empty / killed rows are all zero: every corner lands on one point.
vec3 page_chunk_min(ivec3 rel, int lod, float voxel_dim, float voxel_size) {
    float chunk_world_dim = voxel_dim * voxel_size;
    return vec3(rel) * (chunk_world_dim * 0.5) - vec3(float(1 << lod) * chunk_world_dim * 0.5);
}

vec3 page_corner_world(vec3 chunk_min, int lod, vec3 local, int skirt_size, float voxel_dim, float voxel_size) {
    float size = float(1 << lod) * voxel_dim * voxel_size;
    return chunk_min + (local - float(skirt_size)) * size / voxel_dim;
}

// The coverage sub-cube (in [0, PAGE_COVERAGE_AXIS)^3) holding a chunk-local corner.
uvec3 page_corner_subcube(vec3 local, int skirt_size, float voxel_dim) {
    ivec3 sub = ivec3(floor((local - float(skirt_size)) * float(PAGE_COVERAGE_AXIS) / voxel_dim));
    return uvec3(clamp(sub, ivec3(0), ivec3(int(PAGE_COVERAGE_AXIS) - 1)));
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
    imageStore(staging, ivec2(PAGE_COVERAGE_TEXEL, group), uvec4(0u)); // nothing hidden
}

)glsl";

namespace PageFormat {

// Everything a compute pass writing pages needs, after its staging image binding (the decoders too: palette addressing).
inline ov::str glsl_encode() {
    return glsl_constants() + GLSL_PAGE_DECODE + GLSL_PAGE_ENCODE;
}

// Everything the page shader needs to read corners.
inline ov::str glsl_decode() {
    return glsl_constants() + GLSL_PAGE_DECODE;
}

} // namespace PageFormat
