#pragma once

#include "world_coord.hpp"

#include <godot_cpp/variant/string.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// Exact noise: a pure function of the absolute integer cell, bit-identical on the CPU and the GPU.
//
//   hash_cell(cell, seed)  64-bit cell (CPU int64, GPU lo/hi words) + seed -> uint32. PCG3D (Jarzynski
//                          & Olano, "Hash Functions for GPU Rendering", 2020) on the low words, salted
//                          by a PCG3D of the high words, so cells 2^32 apart don't collide.
//   perlin3 / perlin2      one octave of gradient noise (quintic fade, analytic derivatives) with
//                          wavelength 2^k LOD0 voxels: cell = (voxel + offset) >> k, and the float math
//                          only sees the position inside the cell, as 24-bit fixed point.
//   fbm (OctaveSet)        sum of octaves, each with its own k, seed, integer offset and amplitude, at
//                          a LOD L: the value is unscaled, the gradient is per voxel of LOD L (2^L
//                          LOD0 voxels), each octave's derivative per cell times amp * 2^(L-k).
//
// Exactness rules (the CPU mirror below is op for op the GLSL of GLSL_EXACT_NOISE, exact_noise_glsl.hpp):
//   - uint32 wrapping arithmetic only for the hash (CPU uint32_t, never int: signed overflow is UB).
//   - Float math uses only correctly rounded operations: + - *, compares, min/max/clamp, floor, and
//     int <-> float of values below 2^24. No mix / dot / length / normalize / sqrt / division builtins
//     (their GPU precision is implementation-defined): dots and lerps are written out, in the same order.
//   - GPU: `precise` on every float (no FMA contraction). CPU: float32 statements, built with
//     -ffp-contract=off (SConstruct).
//   - No denormal ever: the GPU may flush them, the CPU keeps them. Cell positions are multiples of
//     2^-24 and the fades are flushed to 0 below 2^-24 (an error below the octave's float resolution),
//     so every product stays >= 2^-96 or is 0; amplitudes and gradient factors amp * 2^(L-k) are
//     kept in [2^-16, 2^40] (make_fbm, gradient_range_error).
//   - Float constants reach the GLSL as raw bits (glsl_float), never as decimal text.
namespace ExactNoise {

struct UVec3 {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
};

struct Float4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

inline uint32_t float_bits(float p_f) {
    uint32_t u;
    std::memcpy(&u, &p_f, sizeof(u));
    return u;
}

inline float bits_float(uint32_t p_u) {
    float f;
    std::memcpy(&f, &p_u, sizeof(f));
    return f;
}

// Every bit of a value and its gradient (the sign of a zero included).
inline bool same_bits(const Float4 &p_a, const Float4 &p_b) {
    return float_bits(p_a.x) == float_bits(p_b.x) && float_bits(p_a.y) == float_bits(p_b.y) && float_bits(p_a.z) == float_bits(p_b.z) &&
            float_bits(p_a.w) == float_bits(p_b.w);
}

// 2^e as a float, -126 <= e <= 127 (exact: built from the exponent bits, like exact_pow2 in GLSL).
inline float exact_pow2(int p_e) {
    return bits_float(uint32_t(p_e + 127) << 23);
}

inline constexpr uint32_t SALT_Y = 0x9E3779B9u;
inline constexpr uint32_t SALT_Z = 0x85EBCA6Bu;
// Third input word of the 2D hash.
inline constexpr uint32_t PLANE_2D = 0x6A09E667u;
// Fades below this are flushed to 0 (see the denormal rule above).
inline constexpr uint32_t FADE_FLUSH_BITS = 0x33800000u; // 2^-24
inline constexpr int MAX_K = 62;

inline UVec3 pcg3d(UVec3 v) {
    v.x = v.x * 1664525u + 1013904223u;
    v.y = v.y * 1664525u + 1013904223u;
    v.z = v.z * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.x ^= v.x >> 16;
    v.y ^= v.y >> 16;
    v.z ^= v.z >> 16;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// A deterministic integer in [-p_range, p_range] from three words: the spikes' hash-random test points.
inline int64_t random_offset(uint32_t p_a, uint32_t p_b, uint32_t p_c, int64_t p_range) {
    const UVec3 h = pcg3d(UVec3{ p_a, p_b, p_c });
    return int64_t(((uint64_t(h.x) << 32) | h.y) % uint64_t(2 * p_range + 1)) - p_range;
}

// CPU placement (trees, biome features): a hash of three words, and its top 24 bits as a double in [0, 1).
inline UVec3 hash3(uint32_t p_a, uint32_t p_b, uint32_t p_c) {
    return pcg3d(UVec3{ p_a, p_b, p_c });
}
inline double unit_of(uint32_t p_bits) {
    return double(p_bits >> 8) * 0x1p-24;
}
// The k of a 2^k-voxel lattice nearest p_metres at voxel size p_voxel (bump and cell sizes in metres).
inline int lattice_k(double p_metres, double p_voxel) {
    return int(std::lround(std::log2(p_metres / p_voxel)));
}

// Salt of a block of 2^32 x 2^32 x 2^32 cells: its high words and the seed.
inline UVec3 cell_salt(int32_t p_hx, int32_t p_hy, int32_t p_hz, uint32_t p_seed) {
    return pcg3d(UVec3{ uint32_t(p_hx) ^ p_seed, uint32_t(p_hy) ^ (p_seed ^ SALT_Y), uint32_t(p_hz) ^ (p_seed ^ SALT_Z) });
}

inline uint32_t hash_words(uint32_t p_lx, uint32_t p_ly, uint32_t p_lz, int32_t p_hx, int32_t p_hy, int32_t p_hz, uint32_t p_seed) {
    const UVec3 s = cell_salt(p_hx, p_hy, p_hz, p_seed);
    return pcg3d(UVec3{ p_lx ^ s.x, p_ly ^ s.y, p_lz ^ s.z }).x;
}

inline uint32_t hash_cell(const Vector3i64 &p_cell, uint32_t p_seed) {
    using namespace WorldCoord;
    return hash_words(lo32(p_cell.x), lo32(p_cell.y), lo32(p_cell.z), hi32(p_cell.x), hi32(p_cell.y), hi32(p_cell.z), p_seed);
}

inline uint32_t hash_cell_2d(int64_t p_x, int64_t p_z, uint32_t p_seed) {
    using namespace WorldCoord;
    return hash_words(lo32(p_x), lo32(p_z), PLANE_2D, hi32(p_x), hi32(p_z), 0, p_seed);
}

// Octave input along one axis: v = voxel + offset, cell = v >> k (floor), q = the position inside the
// cell as 24-bit fixed point (q / 2^24 in [0, 1)). Above k = 24 the fraction and the low bits are
// below float resolution and dropped.
inline void octave_axis(int64_t p_voxel, float p_frac, int p_k, int64_t p_offset, int64_t &r_cell, uint32_t &r_q) {
    const int64_t v = p_voxel + p_offset;
    r_cell = v >= 0 ? (v >> p_k) : ~((~v) >> p_k);
    const uint64_t uv = uint64_t(v);
    if (p_k <= 24) {
        const uint32_t rem = uint32_t(uv & ((uint64_t(1) << p_k) - 1));
        const float scaled = p_frac * exact_pow2(24 - p_k);
        r_q = (rem << (24 - p_k)) | uint32_t(std::floor(scaled));
    } else {
        r_q = uint32_t(uv >> (p_k - 24)) & 0xFFFFFFu;
    }
}

struct OctaveCell {
    Vector3i64 cell;
    UVec3 q;
};

inline OctaveCell octave_cell(const WorldPos &p_pos, int p_k, const Vector3i64 &p_offset) {
    OctaveCell c;
    octave_axis(p_pos.voxel.x, p_pos.frac.x, p_k, p_offset.x, c.cell.x, c.q.x);
    octave_axis(p_pos.voxel.y, p_pos.frac.y, p_k, p_offset.y, c.cell.y, c.q.y);
    octave_axis(p_pos.voxel.z, p_pos.frac.z, p_k, p_offset.z, c.cell.z, c.q.z);
    return c;
}

// Corner gradients, built from the hash's top bits rather than looked up: on the GPU a table indexed
// by a per-thread value serializes (it cost 15% of the SDF pass).
struct Grad3 {
    float v[3];
    float operator[](int p_i) const { return v[p_i]; }
};
struct Grad2 {
    float v[2];
    float operator[](int p_i) const { return v[p_i]; }
};

// Ken Perlin's 12 cube edges for i in [0, 16): bit 0 / bit 1 flip the two non-zero components, i >> 2
// picks the plane (xy, xz, yz), and 12-15 repeat (±1, 1, 0) and (0, -1, ±1).
inline Grad3 grad3(uint32_t p_i) {
    const float sa = (p_i & 1u) ? -1.0f : 1.0f;
    const float sb = (p_i & 2u) ? -1.0f : 1.0f;
    const uint32_t plane = p_i >> 2;
    if (plane == 1u) {
        return { { sa, 0.0f, sb } };
    }
    if (plane == 2u) {
        return { { 0.0f, sa, sb } };
    }
    return p_i < 14u ? Grad3{ { sa, sb, 0.0f } } : Grad3{ { 0.0f, sb, sa } };
}

// 8 equal-length 2D gradients (±1, ±2) and (±2, ±1) for i in [0, 8): bit 0 swaps, bits 1 / 2 flip.
inline Grad2 grad2(uint32_t p_i) {
    const float a = (p_i & 1u) ? 2.0f : 1.0f;
    const float b = (p_i & 1u) ? 1.0f : 2.0f;
    return { { (p_i & 2u) ? -a : a, (p_i & 4u) ? -b : b } };
}

inline float fade(float f) {
    const float u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    return u < bits_float(FADE_FLUSH_BITS) ? 0.0f : u;
}

inline float dfade(float f) {
    const float du = 30.0f * f * f * (f * (f - 2.0f) + 1.0f);
    return du < bits_float(FADE_FLUSH_BITS) ? 0.0f : du;
}

// The cell setup of one 3D octave at a position: fades, corner gradient terms and the coefficients of
// the multilinear form value = k0 + k1 ux + k2 uy + k3 uz + k4 ux uy + k5 uy uz + k6 uz ux + k7 ux uy uz,
// where each k_j is linear in the in-cell position with d k_j / d f_i = g[i][j].
struct Perlin3Cell {
    float u[3];
    float du[3];
    float k[8];
    float g[3][8];
};

inline Perlin3Cell perlin3_cell(const WorldPos &p_pos, int p_k, uint32_t p_seed, const Vector3i64 &p_offset) {
    const OctaveCell c = octave_cell(p_pos, p_k, p_offset);
    const float scale = bits_float(FADE_FLUSH_BITS); // 2^-24
    const float fx = float(c.q.x) * scale;
    const float fy = float(c.q.y) * scale;
    const float fz = float(c.q.z) * scale;
    Perlin3Cell r;
    r.u[0] = fade(fx);
    r.u[1] = fade(fy);
    r.u[2] = fade(fz);
    r.du[0] = dfade(fx);
    r.du[1] = dfade(fy);
    r.du[2] = dfade(fz);

    auto grad = [&](int p_i, int p_j, int p_l) {
        return grad3(hash_cell(c.cell + Vector3i64(p_i, p_j, p_l), p_seed) >> 28);
    };
    const Grad3 ga = grad(0, 0, 0);
    const Grad3 gb = grad(1, 0, 0);
    const Grad3 gc = grad(0, 1, 0);
    const Grad3 gd = grad(1, 1, 0);
    const Grad3 ge = grad(0, 0, 1);
    const Grad3 gf = grad(1, 0, 1);
    const Grad3 gg = grad(0, 1, 1);
    const Grad3 gh = grad(1, 1, 1);
    const float fx1 = fx - 1.0f;
    const float fy1 = fy - 1.0f;
    const float fz1 = fz - 1.0f;
    const float va = ga[0] * fx + ga[1] * fy + ga[2] * fz;
    const float vb = gb[0] * fx1 + gb[1] * fy + gb[2] * fz;
    const float vc = gc[0] * fx + gc[1] * fy1 + gc[2] * fz;
    const float vd = gd[0] * fx1 + gd[1] * fy1 + gd[2] * fz;
    const float ve = ge[0] * fx + ge[1] * fy + ge[2] * fz1;
    const float vf = gf[0] * fx1 + gf[1] * fy + gf[2] * fz1;
    const float vg = gg[0] * fx + gg[1] * fy1 + gg[2] * fz1;
    const float vh = gh[0] * fx1 + gh[1] * fy1 + gh[2] * fz1;

    r.k[0] = va;
    r.k[1] = vb - va;
    r.k[2] = vc - va;
    r.k[3] = ve - va;
    r.k[4] = va - vb - vc + vd;
    r.k[5] = va - vc - ve + vg;
    r.k[6] = va - vb - ve + vf;
    r.k[7] = -va + vb + vc - vd + ve - vf - vg + vh;
    for (int i = 0; i < 3; i++) {
        float *g = r.g[i];
        g[0] = ga[i];
        g[1] = gb[i] - ga[i];
        g[2] = gc[i] - ga[i];
        g[3] = ge[i] - ga[i];
        g[4] = ga[i] - gb[i] - gc[i] + gd[i];
        g[5] = ga[i] - gc[i] - ge[i] + gg[i];
        g[6] = ga[i] - gb[i] - ge[i] + gf[i];
        g[7] = -ga[i] + gb[i] + gc[i] - gd[i] + ge[i] - gf[i] - gg[i] + gh[i];
    }
    return r;
}

// Value and gradient per cell of a cell setup: (d/dx, d/dy, d/dz, value).
inline Float4 perlin3_value_gradient(const Perlin3Cell &p_c) {
    const float ux = p_c.u[0];
    const float uy = p_c.u[1];
    const float uz = p_c.u[2];
    const float *k = p_c.k;
    Float4 r;
    r.w = k[0] + k[1] * ux + k[2] * uy + k[3] * uz + k[4] * ux * uy + k[5] * uy * uz + k[6] * uz * ux + k[7] * ux * uy * uz;
    // Per component: gradient interpolation + fade derivative terms (GLSL: vector ops, same order).
    float d[3];
    const float uyzx[3] = { uy, uz, ux };
    const float uzxy[3] = { uz, ux, uy };
    const float k123[3] = { k[1], k[2], k[3] };
    const float k456[3] = { k[4], k[5], k[6] };
    const float k645[3] = { k[6], k[4], k[5] };
    for (int i = 0; i < 3; i++) {
        const float *g = p_c.g[i];
        const float interp = g[0] + g[1] * ux + g[2] * uy + g[3] * uz + g[4] * ux * uy + g[5] * uy * uz + g[6] * uz * ux + g[7] * ux * uy * uz;
        const float fd = k123[i] + uyzx[i] * k456[i] + uzxy[i] * k645[i] + k[7] * uyzx[i] * uzxy[i];
        d[i] = interp + p_c.du[i] * fd;
    }
    r.x = d[0];
    r.y = d[1];
    r.z = d[2];
    return r;
}

// One 3D octave at wavelength 2^k voxels: (d/dx, d/dy, d/dz per cell, value), value in about [-1, 1]
// (std 0.27).
inline Float4 perlin3(const WorldPos &p_pos, int p_k, uint32_t p_seed, const Vector3i64 &p_offset) {
    return perlin3_value_gradient(perlin3_cell(p_pos, p_k, p_seed, p_offset));
}

// The 8 channels of one 3D octave, per cell: value, d/dx, d/dy, d/dz, d2/dxdy, d2/dxdz, d2/dydz,
// d3/dxdydz (the order of Cascade::Channel). The first four are perlin3's, bit for bit. A mixed
// derivative differentiates each axis once, through its fade (du) or through a coefficient k_j (g):
// no second derivative of the fade appears. Products of at most four factors from {u, du, k_j,
// integer g}, each 0 or >= 2^-24: the denormal rule holds as for perlin3.
// GLSL twin: GLSL_PERLIN3_CROSS, built from the same cell setup and value / gradient code as GLSL perlin3
// (FrameSpike probes perlin3_cross_k*).
struct Channels8 {
    float c[8];
};

inline Channels8 perlin3_cross(const WorldPos &p_pos, int p_k, uint32_t p_seed, const Vector3i64 &p_offset) {
    const Perlin3Cell p = perlin3_cell(p_pos, p_k, p_seed, p_offset);
    const Float4 vg = perlin3_value_gradient(p);
    const float ux = p.u[0];
    const float uy = p.u[1];
    const float uz = p.u[2];
    const float dux = p.du[0];
    const float duy = p.du[1];
    const float duz = p.du[2];
    const float *k = p.k;
    const float *gx = p.g[0];
    const float *gy = p.g[1];
    const float *gz = p.g[2];
    Channels8 r;
    r.c[0] = vg.w;
    r.c[1] = vg.x;
    r.c[2] = vg.y;
    r.c[3] = vg.z;
    r.c[4] = duy * (gx[2] + gx[4] * ux + gx[5] * uz + gx[7] * ux * uz) + dux * (gy[1] + gy[4] * uy + gy[6] * uz + gy[7] * uy * uz) +
            dux * duy * (k[4] + k[7] * uz);
    r.c[5] = duz * (gx[3] + gx[6] * ux + gx[5] * uy + gx[7] * ux * uy) + dux * (gz[1] + gz[4] * uy + gz[6] * uz + gz[7] * uy * uz) +
            dux * duz * (k[6] + k[7] * uy);
    r.c[6] = duz * (gy[3] + gy[5] * uy + gy[6] * ux + gy[7] * ux * uy) + duy * (gz[2] + gz[4] * ux + gz[5] * uz + gz[7] * ux * uz) +
            duy * duz * (k[5] + k[7] * ux);
    r.c[7] = duy * duz * (gx[5] + gx[7] * ux) + dux * duz * (gy[6] + gy[7] * uy) + dux * duy * (gz[4] + gz[7] * uz) + dux * duy * duz * k[7];
    return r;
}

// One 2D octave over (x, z): (d/dx, 0, d/dz per cell, value), value in about [-1.5, 1.5] (std 0.48).
inline Float4 perlin2(const WorldPos &p_pos, int p_k, uint32_t p_seed, const Vector3i64 &p_offset) {
    int64_t cx, cz;
    uint32_t qx, qz;
    octave_axis(p_pos.voxel.x, p_pos.frac.x, p_k, p_offset.x, cx, qx);
    octave_axis(p_pos.voxel.z, p_pos.frac.z, p_k, p_offset.z, cz, qz);
    const float scale = bits_float(FADE_FLUSH_BITS);
    const float fx = float(qx) * scale;
    const float fz = float(qz) * scale;
    const float ux = fade(fx);
    const float uz = fade(fz);
    const float dux = dfade(fx);
    const float duz = dfade(fz);

    const Grad2 ga = grad2(hash_cell_2d(cx, cz, p_seed) >> 29);
    const Grad2 gb = grad2(hash_cell_2d(cx + 1, cz, p_seed) >> 29);
    const Grad2 gc = grad2(hash_cell_2d(cx, cz + 1, p_seed) >> 29);
    const Grad2 gd = grad2(hash_cell_2d(cx + 1, cz + 1, p_seed) >> 29);
    const float fx1 = fx - 1.0f;
    const float fz1 = fz - 1.0f;
    const float va = ga[0] * fx + ga[1] * fz;
    const float vb = gb[0] * fx1 + gb[1] * fz;
    const float vc = gc[0] * fx + gc[1] * fz1;
    const float vd = gd[0] * fx1 + gd[1] * fz1;

    const float k1 = vb - va;
    const float k2 = vc - va;
    const float k3 = va - vb - vc + vd;

    Float4 r;
    r.w = va + k1 * ux + k2 * uz + k3 * ux * uz;
    const float u_swap[2] = { uz, ux };
    const float k12[2] = { k1, k2 };
    const float du[2] = { dux, duz };
    float d[2];
    for (int i = 0; i < 2; i++) {
        const float g1 = gb[i] - ga[i];
        const float g2 = gc[i] - ga[i];
        const float g3 = ga[i] - gb[i] - gc[i] + gd[i];
        const float interp = ga[i] + g1 * ux + g2 * uz + g3 * ux * uz;
        const float fd = k12[i] + k3 * u_swap[i];
        d[i] = interp + du[i] * fd;
    }
    r.x = d[0];
    r.y = 0.0f;
    r.z = d[1];
    return r;
}

// 1 / sqrt(x) for a normal x > 0, deterministic: integer seed + 3 Newton steps (mul/add only).
inline float det_rsqrt(float p_x) {
    float y = bits_float(0x5F375A86u - (float_bits(p_x) >> 1));
    const float hx = p_x * 0.5f;
    y = y * (1.5f - hx * y * y);
    y = y * (1.5f - hx * y * y);
    y = y * (1.5f - hx * y * y);
    return y;
}

// Fractal sum. Octave i: wavelength 2^k voxels, value * amp. At LOD L the gradient is per voxel of
// that LOD: derivative per cell * amp * 2^(L-k) (gradient_factor), an exact power-of-two rescale of
// the gradient per LOD0 voxel, so L only changes its exponent.
struct Octave {
    int k = 0;
    uint32_t seed = 0;
    Vector3i64 offset;
    float amp = 0.0f;
};

inline float gradient_factor(const Octave &p_octave, int p_lod) {
    return p_octave.amp * exact_pow2(p_lod - p_octave.k);
}

// The band rule: octave k is evaluated at LOD p_lod only if its wavelength is at least 2^BAND_SHIFT voxels of
// that LOD (8 samples per wavelength, the cascade's insertion lattice). A finer octave is replaced by its
// mean, 0: it can't be represented by the chunk's samples, and evaluating it aliases (vertices jitter).
inline constexpr int BAND_SHIFT = 3;
inline bool band_kept(int p_k, int p_lod) {
    return p_lod <= p_k - BAND_SHIFT;
}

// Bounds of amplitudes and gradient factors: no denormal in any product (see the rules above).
inline bool in_exact_range(float p_f) {
    return p_f >= exact_pow2(-16) && p_f <= exact_pow2(40);
}

struct OctaveSet {
    bool is_2d = false;
    std::vector<Octave> octaves;
};

// One octave of wavelength 2^p_k, its seed and lattice offset hashed from (p_seed, p_salt): the lattice is
// shifted by a quarter to three quarters of a cell on each axis, so the zeros of nested lattices never line up.
inline Octave make_octave(uint32_t p_seed, uint32_t p_salt, int p_k, float p_amp) {
    Octave o;
    o.k = p_k;
    o.amp = p_amp;
    const UVec3 h = pcg3d(UVec3{ p_seed, p_salt, 0x51ED270Bu });
    o.seed = h.x;
    auto axis_offset = [&](uint32_t p_a, uint32_t p_b) {
        if (o.k < 2) {
            return int64_t(0);
        }
        const uint64_t r = (uint64_t(p_a) << 32) | p_b;
        return int64_t((uint64_t(1) << (o.k - 2)) + (r & ((uint64_t(1) << (o.k - 1)) - 1)));
    };
    const UVec3 h2 = pcg3d(h);
    o.offset = Vector3i64(axis_offset(h.y, h2.x), axis_offset(h.z, h2.y), axis_offset(h2.z, h.x));
    return o;
}

// p_count octaves from wavelength 2^k_top down (lacunarity exactly 2), amplitude p_amp_top for the
// first and times p_persistence for each next one; octave i is make_octave(p_seed, i, ...).
inline OctaveSet make_fbm(bool p_2d, uint32_t p_seed, int p_k_top, int p_count, double p_amp_top, double p_persistence) {
    OctaveSet set;
    set.is_2d = p_2d;
    double amp = p_amp_top;
    for (int i = 0; i < p_count; i++) {
        const int k = p_k_top - i;
        ERR_CONTINUE_MSG(k < 0 || k > MAX_K, "ExactNoise: octave wavelength out of range.");
        const Octave o = make_octave(p_seed, uint32_t(i), k, float(amp));
        if (!in_exact_range(o.amp)) {
            ERR_PRINT(vformat("ExactNoise: octave k=%d amplitude %s outside [2^-16, 2^40]: denormals possible.", o.k, o.amp));
        }
        set.octaves.push_back(o);
        amp *= p_persistence;
    }
    return set;
}

// The two fBms FrameSpike's noise probes and the unit tests check at voxel size p_voxel (metres): a 2D one
// (wavelength 286 m, 5 octaves, 37.5 m) and a 3D one (125 m, 3 octaves).
inline OctaveSet reference_fbm2d(double p_voxel) {
    return make_fbm(true, 0xA2C4E7u, lattice_k(1.0 / 0.0035, p_voxel), 5, 50.0 / p_voxel * 0.75, 0.5);
}
inline OctaveSet reference_fbm3d(double p_voxel) {
    return make_fbm(false, 0xCA7E5u, lattice_k(125.0, p_voxel), 3, 4.0 / 3.0, 0.5);
}

// Checks the gradient factors at the LODs [0, p_lod_count) the set is evaluated at (configure time):
// an error message, empty if they are all in range.
inline String gradient_range_error(const OctaveSet &p_set, int p_lod_count) {
    for (const Octave &o : p_set.octaves) {
        for (int lod = 0; lod < p_lod_count; lod++) {
            if (!in_exact_range(gradient_factor(o, lod))) {
                return vformat("ExactNoise: octave k=%d amplitude %s at LOD %d: gradient factor %s outside [2^-16, 2^40], denormals possible.",
                        o.k, o.amp, lod, gradient_factor(o, lod));
            }
        }
    }
    return String();
}

// Value and gradient per voxel of LOD p_lod (see Octave).
inline Float4 fbm(const OctaveSet &p_set, const WorldPos &p_pos, int p_lod) {
    Float4 n;
    for (const Octave &o : p_set.octaves) {
        const Float4 v = p_set.is_2d ? perlin2(p_pos, o.k, o.seed, o.offset) : perlin3(p_pos, o.k, o.seed, o.offset);
        const float g = gradient_factor(o, p_lod);
        n.x = n.x + v.x * g;
        n.y = n.y + v.y * g;
        n.z = n.z + v.z * g;
        n.w = n.w + v.w * o.amp;
    }
    return n;
}

// GLSL literal of a float, from its bits.
inline String glsl_float(float p_f) {
    return vformat("uintBitsToFloat(0x%08Xu)", (int64_t)float_bits(p_f));
}

inline String glsl_uint(uint32_t p_u) {
    return vformat("0x%08Xu", (int64_t)p_u);
}

// GLSL low and high words of an int64 vector.
inline String glsl_lo(const Vector3i64 &p_v) {
    using namespace WorldCoord;
    return "uvec3(" + glsl_uint(lo32(p_v.x)) + ", " + glsl_uint(lo32(p_v.y)) + ", " + glsl_uint(lo32(p_v.z)) + ")";
}

inline String glsl_hi(const Vector3i64 &p_v) {
    using namespace WorldCoord;
    return "ivec3(int(" + glsl_uint(uint32_t(hi32(p_v.x))) + "), int(" + glsl_uint(uint32_t(hi32(p_v.y))) + "), int(" +
            glsl_uint(uint32_t(hi32(p_v.z))) + "))";
}

// GLSL "vec4 <name>(WorldPos p, int lod)", the GPU side of fbm(): straight-line code with literal
// constants.
inline String glsl_fbm(const String &p_name, const OctaveSet &p_set) {
    String code = "vec4 " + p_name + "(WorldPos p, int lod) {\n    precise vec4 n = vec4(0.0);\n    precise vec4 o;\n";
    for (const Octave &o : p_set.octaves) {
        code += vformat("    o = %s(p, %d, %s, %s, %s);\n", p_set.is_2d ? "perlin2" : "perlin3", o.k, glsl_uint(o.seed), glsl_lo(o.offset),
                glsl_hi(o.offset));
        code += vformat("    n.xyz = n.xyz + o.xyz * (%s * exact_pow2(lod - %d));\n", glsl_float(o.amp), o.k);
        code += "    n.w = n.w + o.w * " + glsl_float(o.amp) + ";\n";
    }
    code += "    return n;\n}\n";
    return code;
}

} // namespace ExactNoise
