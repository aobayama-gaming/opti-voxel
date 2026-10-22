#pragma once

// The GPU side of exact_noise.hpp: its GLSL text (the CPU functions and their GLSL generators stay there). Spliced
// after the frame functions (WorldPos), before the terrain SDF.

#include "exact_noise.hpp"

// GLSL_EXACT_NOISE is spliced from these fragments, so GLSL perlin3_cross (GLSL_PERLIN3_CROSS) shares
// perlin3's cell setup and value / gradient code while GLSL_EXACT_NOISE stays byte-identical (unit test).
// Raw string macros: adjacent literals concatenate at compile time.
// OV_GLSL_PCG3D is also the page shader's lattice hash (GLSL_PCG3D).
#define OV_GLSL_PCG3D R"glsl(
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
)glsl"
#define OV_GLSL_NOISE_HEAD R"glsl(
)glsl" OV_GLSL_PCG3D R"glsl(
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
)glsl"
// perlin3's cell setup: fades u, du, corner gradients ga..gh, coefficients k0..k7.
#define OV_GLSL_PERLIN3_SETUP R"glsl(    OctaveCell c = octave_cell(p, k, olo, ohi);
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

)glsl"
// perlin3's value and gradient per cell: value, g1..g7, d.
#define OV_GLSL_PERLIN3_VALUE_GRADIENT R"glsl(    precise float value = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
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
)glsl"
#define OV_GLSL_NOISE_TAIL R"glsl(    return vec4(d, value);
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

)glsl"

// After the frame functions (WorldPos), before the terrain SDF.
inline constexpr const char GLSL_PCG3D[] = OV_GLSL_PCG3D;
inline constexpr const char GLSL_EXACT_NOISE[] = OV_GLSL_NOISE_HEAD OV_GLSL_PERLIN3_SETUP OV_GLSL_PERLIN3_VALUE_GRADIENT OV_GLSL_NOISE_TAIL;

// GLSL twin of ExactNoise::perlin3_cross (after GLSL_EXACT_NOISE; only the cascade's shaders include it):
// c0 = (value, d/dx, d/dy, d/dz), c1 = (d2/dxdy, d2/dxdz, d2/dydz, d3/dxdydz) per cell. c0 has perlin3's
// bits (same code). gx[j] of the CPU is gj.x here.
inline constexpr const char GLSL_PERLIN3_CROSS[] = R"glsl(
void perlin3_cross(WorldPos p, int k, uint seed, uvec3 olo, ivec3 ohi, out vec4 c0, out vec4 c1) {
)glsl" OV_GLSL_PERLIN3_SETUP OV_GLSL_PERLIN3_VALUE_GRADIENT R"glsl(
    precise float ux = u.x;
    precise float uy = u.y;
    precise float uz = u.z;
    precise float dux = du.x;
    precise float duy = du.y;
    precise float duz = du.z;
    precise float cxy = duy * (g2.x + g4.x * ux + g5.x * uz + g7.x * ux * uz) + dux * (g1.y + g4.y * uy + g6.y * uz + g7.y * uy * uz) +
            dux * duy * (k4 + k7 * uz);
    precise float cxz = duz * (g3.x + g6.x * ux + g5.x * uy + g7.x * ux * uy) + dux * (g1.z + g4.z * uy + g6.z * uz + g7.z * uy * uz) +
            dux * duz * (k6 + k7 * uy);
    precise float cyz = duz * (g3.y + g5.y * uy + g6.y * ux + g7.y * ux * uy) + duy * (g2.z + g4.z * ux + g5.z * uz + g7.z * ux * uz) +
            duy * duz * (k5 + k7 * ux);
    precise float cxyz = duy * duz * (g5.x + g7.x * ux) + dux * duz * (g6.y + g7.y * uy) + dux * duy * (g4.z + g7.z * uz) + dux * duy * duz * k7;
    c0 = vec4(value, d);
    c1 = vec4(cxy, cxz, cyz, cxyz);
}
)glsl";

// GLSL twin of ExactNoise::cell3 (after GLSL_EXACT_NOISE: octave_cell, cell_salt, det_rsqrt; and the int64 helpers:
// i64_add, the neighbour cells' carry or borrow).
inline String glsl_cell3() {
    return String(R"glsl(
vec4 cell3(WorldPos p, int k, uint seed, uvec3 olo, ivec3 ohi) {
    OctaveCell c = octave_cell(p, k, olo, ohi);
    precise vec3 f = vec3(c.q) * uintBitsToFloat(FADE_FLUSH_BITS);
    precise float best = 16.0;
    precise vec3 eb = vec3(0.0);
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                ivec3 d = ivec3(dx, dy, dz);
                uvec3 lo = c.lo;
                ivec3 hi = c.hi;
                i64_add(lo, hi, d);
                uvec3 h = pcg3d(lo ^ cell_salt(hi, seed));
                precise vec3 e = (vec3(d) + vec3(h >> 8u) * uintBitsToFloat(FADE_FLUSH_BITS)) - f;
                precise float d2 = e.x * e.x + e.y * e.y + e.z * e.z;
                if (d2 < best) {
                    best = d2;
                    eb = e;
                }
            }
        }
    }
    precise float min_d2 = uintBitsToFloat(CELL_MIN_D2_BITS);
    precise float rs = det_rsqrt(best > min_d2 ? best : min_d2);
    precise vec3 g = -(eb * rs);
    precise float value = best * rs - CELL_MEAN;
    return vec4(g, value);
}
)glsl")
            .replace("CELL_MEAN", ExactNoise::glsl_float(ExactNoise::CELL_MEAN))
            .replace("CELL_MIN_D2_BITS", ExactNoise::glsl_uint(ExactNoise::CELL_MIN_D2_BITS))
            .replace("FADE_FLUSH_BITS", ExactNoise::glsl_uint(ExactNoise::FADE_FLUSH_BITS));
}

#undef OV_GLSL_PCG3D
#undef OV_GLSL_NOISE_HEAD
#undef OV_GLSL_PERLIN3_SETUP
#undef OV_GLSL_PERLIN3_VALUE_GRADIENT
#undef OV_GLSL_NOISE_TAIL
