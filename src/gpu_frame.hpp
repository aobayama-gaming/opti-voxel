#pragma once

#include "chunk_math.hpp"
#include "ov/bytes.hpp"

// The byte layouts are engine-free (ov::bytes); the GLSL text builders (glsl_functions, glsl_int64)
// still speak godot's String until the GLSL text pipeline converts as one unit.
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>

using godot::String;

// The integer frame (world_coord.hpp) on the GPU, shared by both meshers and the FrameSpike harness.
//
// ChunkParam UBO (std140, 48 B): chunk_size, voxel_size, skirt_size, skirt_depth, then the world origin
// in LOD0 voxels: low words (uvec4 at offset 16) and high words (ivec4 at offset 32).
//
// Per-chunk input (std430, 32 B): ChunkInput { uvec4 id; ivec4 anchor_hi; } holds the low 32 bits of
// the chunk id and the high 32 bits of the chunk's minimum voxel. The low word of the minimum voxel
// is derived on the GPU: (id_lo - lod_size) * half_chunk in wrapping uint arithmetic is exactly the
// low word of the int64 product. The .w components are reserved.
//
// Every absolute position is a WorldPos: the LOD0 voxel as low word (wrapping) + high word, plus a
// fraction in [0, 1). Differences of low words are exact within 2^31 voxels; the high word is only
// needed for absolute identity (noise lattices, exact_noise.hpp). The CPU mirror of the float math is
// ChunkMath::sample_pos and must stay op for op identical; `precise` keeps the GPU from fusing it into
// FMAs.
namespace GPUFrame {

static constexpr uint32_t CHUNK_PARAM_BYTES = 48;
static constexpr uint32_t CHUNK_INPUT_BYTES = 32;
// A ChunkInput's .w words: its instance list's offset (id.w) and length (anchor_hi.w), written by build_lists.
static constexpr uint32_t CHUNK_LIST_OFFSET_AT = 12;
static constexpr uint32_t CHUNK_LIST_COUNT_AT = 28;

inline ov::bytes chunk_param_bytes(const ChunkParam &p_param) {
    using namespace WorldCoord;
    const VoxelPos origin = p_param.origin_voxel();
    ov::bytes bytes(CHUNK_PARAM_BYTES, 0);
    ov::put_s32(bytes.data() + 0, p_param.chunk_size);
    ov::put_f32(bytes.data() + 4, p_param.voxel_size);
    ov::put_s32(bytes.data() + 8, p_param.skirt_size);
    ov::put_f32(bytes.data() + 12, p_param.skirt_depth);
    ov::put_u32(bytes.data() + 16, lo32(origin.x));
    ov::put_u32(bytes.data() + 20, lo32(origin.y));
    ov::put_u32(bytes.data() + 24, lo32(origin.z));
    ov::put_s32(bytes.data() + 32, hi32(origin.x));
    ov::put_s32(bytes.data() + 36, hi32(origin.y));
    ov::put_s32(bytes.data() + 40, hi32(origin.z));
    return bytes;
}

// Writes one ChunkInput (CHUNK_INPUT_BYTES) at r_slot. THE one layout of the GPU's ChunkInput.
inline void encode_chunk_input(uint8_t *r_slot, const ChunkId &p_chunk, const ChunkParam &p_param) {
    using namespace WorldCoord;
    const VoxelPos anchor = ChunkMath::chunk_min_voxel(p_chunk, p_param);
    ov::put_u32(r_slot + 0, lo32(p_chunk.x));
    ov::put_u32(r_slot + 4, lo32(p_chunk.y));
    ov::put_u32(r_slot + 8, lo32(p_chunk.z));
    ov::put_u32(r_slot + CHUNK_LIST_OFFSET_AT, 0);
    ov::put_s32(r_slot + 16, hi32(anchor.x));
    ov::put_s32(r_slot + 20, hi32(anchor.y));
    ov::put_s32(r_slot + 24, hi32(anchor.z));
    ov::put_s32(r_slot + CHUNK_LIST_COUNT_AT, 0);
}

// Chunk p_index of a batch's PackedByteArray (r_bytes must hold the batch).
inline void encode_chunk_input(godot::PackedByteArray &r_bytes, int64_t p_index, const ChunkId &p_chunk, const ChunkParam &p_param) {
    encode_chunk_input(r_bytes.ptrw() + p_index * CHUNK_INPUT_BYTES, p_chunk, p_param);
}

} // namespace GPUFrame

// Before the bindings (ChunkInput is a buffer element type).
inline constexpr const char GLSL_FRAME_STRUCTS[] = R"glsl(

struct ChunkInput { uvec4 id; ivec4 anchor_hi; };

// An absolute position: LOD0 voxel (lo: low 32 bits, wrapping; hi: high 32 bits) + fraction in [0, 1).
struct WorldPos { uvec3 lo; ivec3 hi; vec3 frac; };

// A chunk of the batch: low words of its id, its LOD (and 2^LOD; the low word holds the id's lowest
// set bit for LODs below 32) and its minimum voxel.
struct ChunkFrame { ivec3 id_lo; int lod; int lod_size; uvec3 anchor_lo; ivec3 anchor_hi; };

)glsl";

// After the LOD helpers (get_lod), before the terrain SDF. Use GPUFrame::glsl_functions().
inline constexpr const char GLSL_FRAME_FUNCTIONS[] = R"glsl(

ChunkFrame chunk_frame(ChunkInput c, ChunkParam cp) {
    ChunkFrame f;
    f.id_lo = ivec3(c.id.xyz);
    f.lod = get_lod(f.id_lo);
    f.lod_size = 1 << f.lod;
    uint half_chunk = uint(cp.chunk_size - 2 * cp.skirt_size) >> 1u;
    f.anchor_lo = (c.id.xyz - uvec3(uint(f.lod_size))) * half_chunk;
    f.anchor_hi = c.anchor_hi.xyz;
    return f;
}

// Absolute position of a point of the chunk's vertex grid (grid index skirt_size is the chunk's
// minimum corner). The integer part goes into the voxel, so the same point reached from two chunks
// gives bit-identical members.
WorldPos world_pos_from_grid(ChunkFrame f, vec3 grid, ChunkParam cp) {
    precise vec3 off = (grid - float(cp.skirt_size)) * float(f.lod_size);
    precise vec3 fl = floor(off);
    WorldPos p;
    p.frac = off - fl;
    ivec3 ioff = ivec3(fl);
    p.lo = f.anchor_lo + uvec3(ioff);
    // Carry out of the low word, minus the sign extension of a negative offset.
    p.hi = f.anchor_hi + ivec3(lessThan(p.lo, f.anchor_lo)) - ivec3(lessThan(ioff, ivec3(0)));
    return p;
}

// An int64 (low word, high word) saturated to [-limit, limit] (limit < 2^31). CPU mirror: the clamp of
// WorldCoord::sat_diff.
ivec3 sat_words(uvec3 lo, ivec3 hi, int limit) {
    ivec3 s = ivec3(lo);
    // The value fits in an int32 iff the high word is the sign extension of the low word.
    bvec3 fits = equal(hi, s >> 31);
    ivec3 inside = clamp(s, ivec3(-limit), ivec3(limit));
    ivec3 outside = mix(ivec3(-limit), ivec3(limit), greaterThanEqual(hi, ivec3(0)));
    return mix(outside, inside, fits);
}

// p - anchor in voxels (integer part), saturated to [-limit, limit] (limit < 2^31): the exact 64-bit
// difference wherever it is small. CPU mirror: WorldCoord::sat_diff.
ivec3 sat_diff(WorldPos p, uvec3 anchor_lo, ivec3 anchor_hi, int limit) {
    uvec3 dlo = p.lo - anchor_lo;
    ivec3 dhi = p.hi - anchor_hi - ivec3(lessThan(p.lo, anchor_lo));
    return sat_words(dlo, dhi, limit);
}

)glsl";

// 64-bit integer helpers on (low word, high word) pairs, one per axis: the GPU has no int64
// (GL_ARB_gpu_shader_int64 is not used). Only the cascade's shaders and FrameSpike include them, after
// GLSL_FRAME_FUNCTIONS. CPU mirror: plain int64 arithmetic (FrameSpike probes i64_* against it).
inline constexpr const char GLSL_FRAME_INT64[] = R"glsl(

// v >> s, arithmetic (floor(v / 2^s)), 0 <= s <= 63.
void i64_asr(inout uvec3 lo, inout ivec3 hi, int s) {
    if (s == 0) {
        return;
    }
    if (s < 32) {
        lo = (lo >> uint(s)) | (uvec3(hi) << uint(32 - s));
        hi = hi >> s;
    } else {
        lo = uvec3(hi >> (s - 32));
        hi = hi >> 31;
    }
}

// ceil(v / 2^s), 0 <= s <= 63.
void i64_asr_ceil(inout uvec3 lo, inout ivec3 hi, int s) {
    // The bits shifted out: any set bit rounds up.
    uvec3 rest;
    if (s < 32) {
        rest = lo & uvec3((1u << uint(s)) - 1u);
    } else {
        rest = lo | (uvec3(hi) & uvec3((1u << uint(s - 32)) - 1u));
    }
    i64_asr(lo, hi, s);
    i64_add(lo, hi, ivec3(notEqual(rest, uvec3(0u))));
}

// v << s (wrapping), 0 <= s <= 63.
void i64_shl(inout uvec3 lo, inout ivec3 hi, int s) {
    if (s == 0) {
        return;
    }
    if (s < 32) {
        hi = ivec3((uvec3(hi) << uint(s)) | (lo >> uint(32 - s)));
        lo = lo << uint(s);
    } else {
        hi = ivec3(lo << uint(s - 32));
        lo = uvec3(0u);
    }
}

)glsl";

// i64_add comes first (i64_asr_ceil calls it).
inline constexpr const char GLSL_FRAME_INT64_ADD[] = R"glsl(

// v + d (wrapping), d a signed 32-bit value.
void i64_add(inout uvec3 lo, inout ivec3 hi, ivec3 d) {
    uvec3 r = lo + uvec3(d);
    hi = hi + (d >> 31) + ivec3(lessThan(r, lo));
    lo = r;
}

// v - w (wrapping), w a (low, high) word pair (the record's anchor). Convention: lo is the low 32
// bits and hi the arithmetic high word of an int64, so the pair (lo, hi) *is* the value; the borrow
// out of the low word is lessThan(lo, wlo). CPU mirror: plain int64 subtraction (archetype_table.cpp).
void i64_sub(inout uvec3 lo, inout ivec3 hi, uvec3 wlo, ivec3 whi) {
    uvec3 r = lo - wlo;
    hi = hi - whi - ivec3(lessThan(lo, wlo));
    lo = r;
}

)glsl";

namespace GPUFrame {

inline String glsl_functions() {
    return String(GLSL_FRAME_FUNCTIONS);
}

// The int64 helpers (after glsl_functions()).
inline String glsl_int64() {
    return String(GLSL_FRAME_INT64_ADD) + String(GLSL_FRAME_INT64);
}

} // namespace GPUFrame
