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
// Exactness rules (the CPU mirror below is op for op the GLSL of GLSL_EXACT_NOISE):
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

// One 3D octave at wavelength 2^k voxels: (d/dx, d/dy, d/dz per cell, value), value in about [-1, 1]
// (std 0.27).
inline Float4 perlin3(const WorldPos &p_pos, int p_k, uint32_t p_seed, const Vector3i64 &p_offset) {
    const OctaveCell c = octave_cell(p_pos, p_k, p_offset);
    const float scale = bits_float(FADE_FLUSH_BITS); // 2^-24
    const float fx = float(c.q.x) * scale;
    const float fy = float(c.q.y) * scale;
    const float fz = float(c.q.z) * scale;
    const float ux = fade(fx);
    const float uy = fade(fy);
    const float uz = fade(fz);
    const float dux = dfade(fx);
    const float duy = dfade(fy);
    const float duz = dfade(fz);

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

    const float k0 = va;
    const float k1 = vb - va;
    const float k2 = vc - va;
    const float k3 = ve - va;
    const float k4 = va - vb - vc + vd;
    const float k5 = va - vc - ve + vg;
    const float k6 = va - vb - ve + vf;
    const float k7 = -va + vb + vc - vd + ve - vf - vg + vh;

    Float4 r;
    r.w = k0 + k1 * ux + k2 * uy + k3 * uz + k4 * ux * uy + k5 * uy * uz + k6 * uz * ux + k7 * ux * uy * uz;
    // Per component: gradient interpolation + fade derivative terms (GLSL: vector ops, same order).
    float d[3];
    const float uyzx[3] = { uy, uz, ux };
    const float uzxy[3] = { uz, ux, uy };
    const float k123[3] = { k1, k2, k3 };
    const float k456[3] = { k4, k5, k6 };
    const float k645[3] = { k6, k4, k5 };
    const float du[3] = { dux, duy, duz };
    for (int i = 0; i < 3; i++) {
        const float g0 = ga[i];
        const float g1 = gb[i] - ga[i];
        const float g2 = gc[i] - ga[i];
        const float g3 = ge[i] - ga[i];
        const float g4 = ga[i] - gb[i] - gc[i] + gd[i];
        const float g5 = ga[i] - gc[i] - ge[i] + gg[i];
        const float g6 = ga[i] - gb[i] - ge[i] + gf[i];
        const float g7 = -ga[i] + gb[i] + gc[i] - gd[i] + ge[i] - gf[i] - gg[i] + gh[i];
        const float interp = g0 + g1 * ux + g2 * uy + g3 * uz + g4 * ux * uy + g5 * uy * uz + g6 * uz * ux + g7 * ux * uy * uz;
        const float fd = k123[i] + uyzx[i] * k456[i] + uzxy[i] * k645[i] + k7 * uyzx[i] * uzxy[i];
        d[i] = interp + du[i] * fd;
    }
    r.x = d[0];
    r.y = d[1];
    r.z = d[2];
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

// Bounds of amplitudes and gradient factors: no denormal in any product (see the rules above).
inline bool in_exact_range(float p_f) {
    return p_f >= exact_pow2(-16) && p_f <= exact_pow2(40);
}

struct OctaveSet {
    bool is_2d = false;
    std::vector<Octave> octaves;
};

// p_count octaves from wavelength 2^k_top down (lacunarity exactly 2), amplitude p_amp_top for the
// first and times p_persistence for each next one. Seeds and offsets come from p_seed: each octave's
// lattice is shifted by a quarter to three quarters of a cell on each axis, so the zeros of the
// nested lattices never line up.
inline OctaveSet make_fbm(bool p_2d, uint32_t p_seed, int p_k_top, int p_count, double p_amp_top, double p_persistence) {
    OctaveSet set;
    set.is_2d = p_2d;
    double amp = p_amp_top;
    for (int i = 0; i < p_count; i++) {
        Octave o;
        o.k = p_k_top - i;
        ERR_CONTINUE_MSG(o.k < 0 || o.k > MAX_K, "ExactNoise: octave wavelength out of range.");
        const UVec3 h = pcg3d(UVec3{ p_seed, uint32_t(i), 0x51ED270Bu });
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
        o.amp = float(amp);
        if (!in_exact_range(o.amp)) {
            ERR_PRINT(vformat("ExactNoise: octave k=%d amplitude %s outside [2^-16, 2^40]: denormals possible.", o.k, o.amp));
        }
        set.octaves.push_back(o);
        amp *= p_persistence;
    }
    return set;
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

// After the frame functions (WorldPos), before the terrain SDF.
inline constexpr const char GLSL_EXACT_NOISE[] = R"glsl(

uvec3 pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

uvec3 cell_salt(ivec3 hi, uint seed) {
    return pcg3d(uvec3(hi) ^ uvec3(seed, seed ^ 0x9E3779B9u, seed ^ 0x85EBCA6Bu));
}

uint hash_cell(uvec3 lo, ivec3 hi, uint seed) {
    return pcg3d(lo ^ cell_salt(hi, seed)).x;
}

float exact_pow2(int e) {
    return uintBitsToFloat(uint(e + 127) << 23u);
}

// See ExactNoise::octave_axis. 0 <= k <= 62; a literal at every call site, so the branches fold.
void octave_axis(uint lo, int hi, float frac, int k, uint olo, int ohi, out uint clo, out int chi, out uint q) {
    uint vlo = lo + olo;
    int vhi = hi + ohi + int(vlo < lo);
    uint uhi = uint(vhi);
    if (k == 0) {
        clo = vlo;
        chi = vhi;
    } else if (k < 32) {
        clo = (vlo >> uint(k)) | (uhi << uint(32 - k));
        chi = vhi >> k;
    } else {
        clo = uint(vhi >> (k - 32));
        chi = vhi >> 31;
    }
    if (k <= 24) {
        uint rem = vlo & ((1u << uint(k)) - 1u);
        precise float scaled = frac * exact_pow2(24 - k);
        q = (rem << uint(24 - k)) | uint(floor(scaled));
    } else {
        int s = k - 24;
        uint bits = s < 32 ? ((vlo >> uint(s)) | (uhi << uint(32 - s))) : (uhi >> uint(s - 32));
        q = bits & 0xFFFFFFu;
    }
}

struct OctaveCell { uvec3 lo; ivec3 hi; uvec3 q; };

OctaveCell octave_cell(WorldPos p, int k, uvec3 olo, ivec3 ohi) {
    OctaveCell c;
    octave_axis(p.lo.x, p.hi.x, p.frac.x, k, olo.x, ohi.x, c.lo.x, c.hi.x, c.q.x);
    octave_axis(p.lo.y, p.hi.y, p.frac.y, k, olo.y, ohi.y, c.lo.y, c.hi.y, c.q.y);
    octave_axis(p.lo.z, p.hi.z, p.frac.z, k, olo.z, ohi.z, c.lo.z, c.hi.z, c.q.z);
    return c;
}

// See ExactNoise::grad3 / grad2 (selects, no table).
vec3 grad3(uint i) {
    float sa = (i & 1u) != 0u ? -1.0 : 1.0;
    float sb = (i & 2u) != 0u ? -1.0 : 1.0;
    uint plane = i >> 2u;
    vec3 g = i < 14u ? vec3(sa, sb, 0.0) : vec3(0.0, sb, sa);
    g = plane == 1u ? vec3(sa, 0.0, sb) : g;
    g = plane == 2u ? vec3(0.0, sa, sb) : g;
    return g;
}

vec2 grad2(uint i) {
    float a = (i & 1u) != 0u ? 2.0 : 1.0;
    float b = (i & 1u) != 0u ? 1.0 : 2.0;
    return vec2((i & 2u) != 0u ? -a : a, (i & 4u) != 0u ? -b : b);
}

vec3 fade3(vec3 f) {
    precise vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(u, vec3(0.0), lessThan(u, vec3(uintBitsToFloat(0x33800000u))));
}

vec3 dfade3(vec3 f) {
    precise vec3 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    return mix(du, vec3(0.0), lessThan(du, vec3(uintBitsToFloat(0x33800000u))));
}

// Corner hashes of a cell: hash_cell of each corner. The corners share the cell's high words (and so
// one salt) unless a low word carries, i.e. the cell sits on a 2^32 boundary: one branch per cell,
// not per corner.
vec3 corner_grad(uvec3 lo, uvec3 salt) {
    return grad3(pcg3d(lo ^ salt).x >> 28u);
}

vec4 perlin3(WorldPos p, int k, uint seed, uvec3 olo, ivec3 ohi) {
    OctaveCell c = octave_cell(p, k, olo, ohi);
    precise vec3 f = vec3(c.q) * uintBitsToFloat(0x33800000u);
    precise vec3 u = fade3(f);
    precise vec3 du = dfade3(f);

    uvec3 lo1 = c.lo + 1u;
    bvec3 carry = equal(lo1, uvec3(0u));
    ivec3 hi1 = c.hi + ivec3(carry);
    uvec3 s000 = cell_salt(c.hi, seed);
    uvec3 s100 = s000, s010 = s000, s110 = s000, s001 = s000, s101 = s000, s011 = s000, s111 = s000;
    if (any(carry)) {
        s100 = cell_salt(ivec3(hi1.x, c.hi.y, c.hi.z), seed);
        s010 = cell_salt(ivec3(c.hi.x, hi1.y, c.hi.z), seed);
        s110 = cell_salt(ivec3(hi1.x, hi1.y, c.hi.z), seed);
        s001 = cell_salt(ivec3(c.hi.x, c.hi.y, hi1.z), seed);
        s101 = cell_salt(ivec3(hi1.x, c.hi.y, hi1.z), seed);
        s011 = cell_salt(ivec3(c.hi.x, hi1.y, hi1.z), seed);
        s111 = cell_salt(hi1, seed);
    }
    vec3 ga = corner_grad(c.lo, s000);
    vec3 gb = corner_grad(uvec3(lo1.x, c.lo.y, c.lo.z), s100);
    vec3 gc = corner_grad(uvec3(c.lo.x, lo1.y, c.lo.z), s010);
    vec3 gd = corner_grad(uvec3(lo1.x, lo1.y, c.lo.z), s110);
    vec3 ge = corner_grad(uvec3(c.lo.x, c.lo.y, lo1.z), s001);
    vec3 gf = corner_grad(uvec3(lo1.x, c.lo.y, lo1.z), s101);
    vec3 gg = corner_grad(uvec3(c.lo.x, lo1.y, lo1.z), s011);
    vec3 gh = corner_grad(lo1, s111);

    precise vec3 f1 = f - 1.0;
    precise float va = ga.x * f.x + ga.y * f.y + ga.z * f.z;
    precise float vb = gb.x * f1.x + gb.y * f.y + gb.z * f.z;
    precise float vc = gc.x * f.x + gc.y * f1.y + gc.z * f.z;
    precise float vd = gd.x * f1.x + gd.y * f1.y + gd.z * f.z;
    precise float ve = ge.x * f.x + ge.y * f.y + ge.z * f1.z;
    precise float vf = gf.x * f1.x + gf.y * f.y + gf.z * f1.z;
    precise float vg = gg.x * f.x + gg.y * f1.y + gg.z * f1.z;
    precise float vh = gh.x * f1.x + gh.y * f1.y + gh.z * f1.z;

    precise float k0 = va;
    precise float k1 = vb - va;
    precise float k2 = vc - va;
    precise float k3 = ve - va;
    precise float k4 = va - vb - vc + vd;
    precise float k5 = va - vc - ve + vg;
    precise float k6 = va - vb - ve + vf;
    precise float k7 = -va + vb + vc - vd + ve - vf - vg + vh;

    precise float value = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    precise vec3 g1 = gb - ga;
    precise vec3 g2 = gc - ga;
    precise vec3 g3 = ge - ga;
    precise vec3 g4 = ga - gb - gc + gd;
    precise vec3 g5 = ga - gc - ge + gg;
    precise vec3 g6 = ga - gb - ge + gf;
    precise vec3 g7 = -ga + gb + gc - gd + ge - gf - gg + gh;
    precise vec3 interp = ga + g1 * u.x + g2 * u.y + g3 * u.z + g4 * u.x * u.y + g5 * u.y * u.z + g6 * u.z * u.x + g7 * u.x * u.y * u.z;
    precise vec3 fd = vec3(k1, k2, k3) + u.yzx * vec3(k4, k5, k6) + u.zxy * vec3(k6, k4, k5) + k7 * u.yzx * u.zxy;
    precise vec3 d = interp + du * fd;
    return vec4(d, value);
}

uint hash_cell_2d(uint lx, int hx, uint lz, int hz, uint seed) {
    return hash_cell(uvec3(lx, lz, 0x6A09E667u), ivec3(hx, hz, 0), seed);
}

vec4 perlin2(WorldPos p, int k, uint seed, uvec3 olo, ivec3 ohi) {
    uint cxl, czl, qx, qz;
    int cxh, czh;
    octave_axis(p.lo.x, p.hi.x, p.frac.x, k, olo.x, ohi.x, cxl, cxh, qx);
    octave_axis(p.lo.z, p.hi.z, p.frac.z, k, olo.z, ohi.z, czl, czh, qz);
    precise vec2 f = vec2(qx, qz) * uintBitsToFloat(0x33800000u);
    precise vec2 u = fade3(vec3(f, 0.0)).xy;
    precise vec2 du = dfade3(vec3(f, 0.0)).xy;

    uint cxl1 = cxl + 1u;
    int cxh1 = cxh + int(cxl1 == 0u);
    uint czl1 = czl + 1u;
    int czh1 = czh + int(czl1 == 0u);
    // Same salt sharing as perlin3.
    uvec3 s00 = cell_salt(ivec3(cxh, czh, 0), seed);
    uvec3 s10 = s00, s01 = s00, s11 = s00;
    if (cxl1 == 0u || czl1 == 0u) {
        s10 = cell_salt(ivec3(cxh1, czh, 0), seed);
        s01 = cell_salt(ivec3(cxh, czh1, 0), seed);
        s11 = cell_salt(ivec3(cxh1, czh1, 0), seed);
    }
    vec2 ga = grad2(pcg3d(uvec3(cxl, czl, 0x6A09E667u) ^ s00).x >> 29u);
    vec2 gb = grad2(pcg3d(uvec3(cxl1, czl, 0x6A09E667u) ^ s10).x >> 29u);
    vec2 gc = grad2(pcg3d(uvec3(cxl, czl1, 0x6A09E667u) ^ s01).x >> 29u);
    vec2 gd = grad2(pcg3d(uvec3(cxl1, czl1, 0x6A09E667u) ^ s11).x >> 29u);

    precise vec2 f1 = f - 1.0;
    precise float va = ga.x * f.x + ga.y * f.y;
    precise float vb = gb.x * f1.x + gb.y * f.y;
    precise float vc = gc.x * f.x + gc.y * f1.y;
    precise float vd = gd.x * f1.x + gd.y * f1.y;
    precise float k1 = vb - va;
    precise float k2 = vc - va;
    precise float k3 = va - vb - vc + vd;
    precise float value = va + k1 * u.x + k2 * u.y + k3 * u.x * u.y;
    precise vec2 g1 = gb - ga;
    precise vec2 g2 = gc - ga;
    precise vec2 g3 = ga - gb - gc + gd;
    precise vec2 interp = ga + g1 * u.x + g2 * u.y + g3 * u.x * u.y;
    precise vec2 fd = vec2(k1, k2) + k3 * u.yx;
    precise vec2 d = interp + du * fd;
    return vec4(d.x, 0.0, d.y, value);
}

float det_rsqrt(float x) {
    precise float y = uintBitsToFloat(0x5F375A86u - (floatBitsToUint(x) >> 1u));
    precise float hx = x * 0.5;
    y = y * (1.5 - hx * y * y);
    y = y * (1.5 - hx * y * y);
    y = y * (1.5 - hx * y * y);
    return y;
}

)glsl";
