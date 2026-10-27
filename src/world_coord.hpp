#pragma once

#include "ov/format.hpp"
#include "ov/murmur3.hpp"
#include "ov/types.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// Integer world frame. Every absolute position is an int64: chunk ids (chunk-id units, see
// chunk_math.hpp) and LOD0 voxel coordinates. Floats only hold small offsets: WorldPos::frac, and
// render positions relative to the world origin. The GPU gets the low 32 bits of a coordinate
// (wrapping arithmetic, exact for differences under 2^31) plus, where it needs absolute identity,
// the high 32 bits.

struct Vector3i64 {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;

    constexpr Vector3i64() = default;
    constexpr Vector3i64(int64_t p_x, int64_t p_y, int64_t p_z) :
            x(p_x), y(p_y), z(p_z) {}
    // No conversion back to an int32 triple: a leftover int32 use must fail to compile (see lo32_i).

    constexpr Vector3i64 operator+(const Vector3i64 &p_o) const { return Vector3i64(x + p_o.x, y + p_o.y, z + p_o.z); }
    constexpr Vector3i64 operator-(const Vector3i64 &p_o) const { return Vector3i64(x - p_o.x, y - p_o.y, z - p_o.z); }
    constexpr Vector3i64 operator*(int64_t p_s) const { return Vector3i64(x * p_s, y * p_s, z * p_s); }
    Vector3i64 &operator+=(const Vector3i64 &p_o) {
        x += p_o.x;
        y += p_o.y;
        z += p_o.z;
        return *this;
    }
    constexpr int64_t operator[](int p_axis) const { return p_axis == 0 ? x : (p_axis == 1 ? y : z); }
    int64_t &operator[](int p_axis) { return p_axis == 0 ? x : (p_axis == 1 ? y : z); }
    constexpr bool operator==(const Vector3i64 &p_o) const { return x == p_o.x && y == p_o.y && z == p_o.z; }
    constexpr bool operator!=(const Vector3i64 &p_o) const { return !(*this == p_o); }

    // Euclidean distance, from the exact integer difference.
    double distance_to(const Vector3i64 &p_o) const {
        const double dx = double(x - p_o.x);
        const double dy = double(y - p_o.y);
        const double dz = double(z - p_o.z);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

using ChunkId = Vector3i64;
using VoxelPos = Vector3i64;

// An axis-aligned box in LOD0 voxels (inclusive): an instance edit's region, a query box.
struct VoxelBox {
    VoxelPos lo;
    VoxelPos hi;
    bool valid = false;

    bool intersects(const VoxelBox &p_o) const {
        return valid && p_o.valid && lo.x <= p_o.hi.x && p_o.lo.x <= hi.x && lo.y <= p_o.hi.y && p_o.lo.y <= hi.y && lo.z <= p_o.hi.z &&
                p_o.lo.z <= hi.z;
    }
    bool contains(const VoxelPos &p_v) const {
        return valid && lo.x <= p_v.x && p_v.x <= hi.x && lo.y <= p_v.y && p_v.y <= hi.y && lo.z <= p_v.z && p_v.z <= hi.z;
    }
    // The cube of side p_side whose index is p_index (in units of p_side): voxels p_index * p_side .. + p_side - 1.
    static VoxelBox cube(const VoxelPos &p_index, int64_t p_side) {
        return VoxelBox{ p_index * p_side, p_index * p_side + VoxelPos(p_side - 1, p_side - 1, p_side - 1), true };
    }
    // The part of this box inside p_o (invalid if they don't meet).
    VoxelBox clipped(const VoxelBox &p_o) const {
        VoxelBox r;
        r.valid = intersects(p_o);
        for (int a = 0; a < 3; a++) {
            r.lo[a] = lo[a] > p_o.lo[a] ? lo[a] : p_o.lo[a];
            r.hi[a] = hi[a] < p_o.hi[a] ? hi[a] : p_o.hi[a];
        }
        return r;
    }
    // The smallest box holding this one and p_o (either one if the other is invalid).
    VoxelBox merged(const VoxelBox &p_o) const {
        if (!valid || !p_o.valid) {
            return valid ? *this : p_o;
        }
        VoxelBox r = *this;
        for (int a = 0; a < 3; a++) {
            r.lo[a] = lo[a] < p_o.lo[a] ? lo[a] : p_o.lo[a];
            r.hi[a] = hi[a] > p_o.hi[a] ? hi[a] : p_o.hi[a];
        }
        return r;
    }
    // This box grown by p_by voxels on every side.
    VoxelBox grown(int64_t p_by) const { return VoxelBox{ lo - VoxelPos(p_by, p_by, p_by), hi + VoxelPos(p_by, p_by, p_by), valid }; }
};

// For the core's hash containers. Folds each int64 to 32 bits (high word scrambled in), then the
// same murmur chain the fold feeds (ov/murmur3.hpp, vendored from godot's hashfuncs): the hash
// values are exactly those of the pre-de-Godot code.
struct Vector3i64Hasher {
    static inline uint32_t fold(int64_t p_v) {
        const uint64_t u = uint64_t(p_v);
        return uint32_t(u) ^ (uint32_t(u >> 32) * 0x9E3779B1u);
    }
    static inline uint32_t hash(const Vector3i64 &p_v) {
        uint32_t h = ov::murmur3_one_32(fold(p_v.x));
        h = ov::murmur3_one_32(fold(p_v.y), h);
        h = ov::murmur3_one_32(fold(p_v.z), h);
        return ov::fmix32(h);
    }
};

// A unit direction in absolute axes, double (SDFBase::up_at, BaseSurface::up_dir).
using UpAxis = std::array<double, 3>;
inline constexpr UpAxis UP_Y{ 0.0, 1.0, 0.0 };

// p_from moved p_h voxels along p_up, each axis rounded to the voxel (exactly p_h on y along UP_Y).
inline VoxelPos along_up(const VoxelPos &p_from, const UpAxis &p_up, double p_h) {
    return p_from + VoxelPos(std::llround(p_h * p_up[0]), std::llround(p_h * p_up[1]), std::llround(p_h * p_up[2]));
}

// Plain float32 triple: the CPU mirror of GPU float math must stay float32 even in
// precision=double builds (where godot's Vector3 is double).
struct Float3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    constexpr bool operator==(const Float3 &p_o) const { return x == p_o.x && y == p_o.y && z == p_o.z; }
    constexpr Float3 operator-(const Float3 &p_o) const { return Float3{ x - p_o.x, y - p_o.y, z - p_o.z }; }
    constexpr bool operator!=(const Float3 &p_o) const { return !(*this == p_o); }

    // godot Vector3::normalized's float form: divide each component by the length, no reciprocal
    // trick (the CPU mirrors of the page shader must stay op for op, see PageFormat::decode_normal).
    Float3 normalized() const {
        const float l = std::sqrt(x * x + y * y + z * z);
        return Float3{ x / l, y / l, z / l };
    }
};

inline float dot(const Float3 &p_a, const Float3 &p_b) {
    return p_a.x * p_b.x + p_a.y * p_b.y + p_a.z * p_b.z;
}

inline Float3 cross(const Float3 &p_a, const Float3 &p_b) {
    return Float3{ p_a.y * p_b.z - p_a.z * p_b.y, p_a.z * p_b.x - p_a.x * p_b.z, p_a.x * p_b.y - p_a.y * p_b.x };
}

// An absolute position: an integer LOD0 voxel plus a fraction of a voxel in [0, 1). Canonical: the
// same point reached from two chunks has bit-identical members (sample points have frac == 0).
struct WorldPos {
    VoxelPos voxel;
    Float3 frac;
};

namespace WorldCoord {

// World origins are bounded so that int64 voxel arithmetic around them never overflows.
inline constexpr int64_t MAX_ABS_ORIGIN_VOXEL = int64_t(1) << 60;

// PI as godot's Math_PI carried it (the core's trig constants don't come from the engine).
inline constexpr double PI = 3.14159265358979323846;

inline uint32_t lo32(int64_t p_v) {
    return uint32_t(uint64_t(p_v));
}

inline int32_t as_i32(uint32_t p_v) {
    int32_t r;
    std::memcpy(&r, &p_v, sizeof(r));
    return r;
}

inline int32_t hi32(int64_t p_v) {
    return as_i32(uint32_t(uint64_t(p_v) >> 32));
}

// The low 32 bits as an int32 bit pattern: what page headers and shader uniforms carry.
inline ov::ivec3 lo32_i(const Vector3i64 &p_v) {
    return ov::ivec3(as_i32(lo32(p_v.x)), as_i32(lo32(p_v.y)), as_i32(lo32(p_v.z)));
}

inline ov::ivec3 hi32_i(const Vector3i64 &p_v) {
    return ov::ivec3(hi32(p_v.x), hi32(p_v.y), hi32(p_v.z));
}

// Floor division, p_b > 0.
inline int64_t floor_div(int64_t p_a, int64_t p_b) {
    int64_t q = p_a / p_b;
    if ((p_a % p_b) != 0 && p_a < 0) {
        q--;
    }
    return q;
}

// The cell of side p_b (> 0) holding p_a, per axis.
inline Vector3i64 floor_div(const Vector3i64 &p_a, int64_t p_b) {
    return Vector3i64(floor_div(p_a.x, p_b), floor_div(p_a.y, p_b), floor_div(p_a.z, p_b));
}

// p_a - p_b saturated to [-p_limit, p_limit] (p_limit < 2^31): the exact int64 difference wherever it
// is small, CPU mirror of sat_diff (gpu_frame.hpp). Below 2^24 it converts to float exactly.
inline int32_t sat_diff(int64_t p_a, int64_t p_b, int32_t p_limit) {
    const int64_t d = p_a - p_b;
    return int32_t(d < -p_limit ? -p_limit : (d > p_limit ? p_limit : d));
}

// Number of set bits.
inline int popcount64(uint64_t p_v) {
#if defined(_MSC_VER)
    return int(__popcnt64(p_v));
#else
    return __builtin_popcountll(p_v);
#endif
}

// Index of the lowest set bit, p_v != 0.
inline int ctz64(uint64_t p_v) {
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, p_v);
    return int(index);
#else
    return __builtin_ctzll(p_v);
#endif
}

// p_anchor plus an offset in voxels (double), canonical: the floor goes to the voxel, the rest to the fraction,
// kept below 1 (a fraction that rounds up to 1.0f in float is clamped).
inline WorldPos offset_pos(const VoxelPos &p_anchor, const double p_offset[3]) {
    WorldPos p;
    float *frac[3] = { &p.frac.x, &p.frac.y, &p.frac.z };
    for (int a = 0; a < 3; a++) {
        const double fl = std::floor(p_offset[a]);
        p.voxel[a] = p_anchor[a] + int64_t(fl);
        *frac[a] = float(p_offset[a] - fl) < 1.0f ? float(p_offset[a] - fl) : 0.99999994f;
    }
    return p;
}

// "(x, y, z)", the same text godot's Vector3i shows. Reports carry int64 ids as strings because
// godot's JSON parser reads numbers back as doubles.
inline ov::str to_string(const Vector3i64 &p_v) {
    return ov::format("(%d, %d, %d)", p_v.x, p_v.y, p_v.z);
}

} // namespace WorldCoord
