#pragma once

#include "base_surface.hpp"
#include "chunk_math.hpp"
#include "exact_noise.hpp"
#include "ov/math.hpp"
#include "ov/str.hpp"
#include "ov/types.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

// The Hermite cascade's kernel: lattices, boxes and the subdivision every program level runs (cascade v2,
// docs/cascade.md; the programs themselves are level_program.hpp's).
//
// Λ_m is the lattice of points at multiples of 2^m LOD0 voxels (absolute int64 lattice indices). A node of a
// LevelProgram is computed on its level's lattice and carried down by H, the tricubic Hermite subdivision:
//     V_m = flush(H(V_{m+1})), then level m's code, on Λ_m,
// so V_m(p) is a pure function of p and m: every LOD sees bit-identical values at the points it shares with another.
// A LOD-L chunk runs down to its own lattice Λ_L and stops.
//
// Channels per lattice point, in units of the lattice's spacing (channel S times 2^(m |S|) per voxel):
// value, d/dx, d/dy, d/dz.
//
// H, one axis at a time (x, then y, then z: fixed order), parent spacing = 2 child spacings, endpoint
// values a, b and slopes A, B along the axis in parent units:
//     child on the parent lattice (even index):  a,                                slope A * 0.5
//     midpoint (odd index):                      (a + b) * 0.5 + (A - B) * 0.125,  slope (b - a) * 0.75 - (A + B) * 0.125
// to the pair (value, slope along the axis), the two transverse slopes by linear interpolation.
//
// Exactness: only + - * by the constants above and compares (correctly rounded, `precise` on the GPU);
// copies and the x0.5 slope scalings are exact, so a chunk and a point query give the same bits. Denormals:
// after each level every channel below 2^-64 is flushed to 0 (FLUSH_BITS). Stored values are then
// multiples of 2^-87, and the three passes of H keep every intermediate a multiple of 2^-96: never a
// denormal, whether the GPU flushes them or not.
namespace Cascade {

enum Channel {
    VALUE,
    DX,
    DY,
    DZ,
    CHANNELS,
};

// The derivatives a channel holds: 0 the value, 1 a slope.
constexpr int channel_order(int p_channel) {
    return p_channel == VALUE ? 0 : 1;
}

inline constexpr uint32_t FLUSH_BITS = 0x1F800000u; // 2^-64
inline constexpr int MAX_LEVEL = ExactNoise::MAX_K;

// The per-axis rule of H (see above), in the order the GPU mirror must keep.
inline void hermite_copy(float p_a, float p_sa, float &r_v, float &r_s) {
    r_v = p_a;
    r_s = p_sa * 0.5f;
}

inline void hermite_mid(float p_a, float p_b, float p_sa, float p_sb, float &r_v, float &r_s) {
    r_v = (p_a + p_b) * 0.5f + (p_sa - p_sb) * 0.125f;
    r_s = (p_b - p_a) * 0.75f - (p_sa + p_sb) * 0.125f;
}

inline float flush_value(float p_v) {
    return std::fabs(p_v) < ExactNoise::bits_float(FLUSH_BITS) ? 0.0f : p_v;
}
// flush_value's threshold as a GLSL literal (the generated code's flushes).
inline ov::str glsl_flush() {
    return ExactNoise::glsl_float(ExactNoise::bits_float(FLUSH_BITS));
}
// flush_value of the GLSL expression p_expr (a variable: it is read twice), a conditional without parentheses: the
// generators' one flush.
inline ov::str glsl_flush_of(const ov::str &p_expr) {
    return "abs(" + p_expr + ov::str(") < ") + glsl_flush() + ov::str(" ? 0.0 : ") + p_expr;
}

// Dual numbers (x, y, z: gradient, w: value), every result flushed under 2^-64: inputs 0 or >= 2^-64, so a product's
// denormal part never reaches a result. The GLSL twins are cl_flush, cl_mul, cl_sub, cl_add and cl_scale
// (Climate::shared_helpers_glsl).
inline ExactNoise::Float4 dflush(const ExactNoise::Float4 &p_a) {
    return ExactNoise::Float4{ flush_value(p_a.x), flush_value(p_a.y), flush_value(p_a.z), flush_value(p_a.w) };
}
inline ExactNoise::Float4 dmul(const ExactNoise::Float4 &p_a, const ExactNoise::Float4 &p_b) {
    return dflush(ExactNoise::Float4{ p_a.w * p_b.x + p_b.w * p_a.x, p_a.w * p_b.y + p_b.w * p_a.y, p_a.w * p_b.z + p_b.w * p_a.z, p_a.w * p_b.w });
}
inline ExactNoise::Float4 dsub(const ExactNoise::Float4 &p_a, const ExactNoise::Float4 &p_b) {
    return dflush(ExactNoise::Float4{ p_a.x - p_b.x, p_a.y - p_b.y, p_a.z - p_b.z, p_a.w - p_b.w });
}
inline ExactNoise::Float4 dadd(const ExactNoise::Float4 &p_a, const ExactNoise::Float4 &p_b) {
    return dflush(ExactNoise::Float4{ p_a.x + p_b.x, p_a.y + p_b.y, p_a.z + p_b.z, p_a.w + p_b.w });
}
// p_a times the constant p_k.
inline ExactNoise::Float4 dscale(const ExactNoise::Float4 &p_a, float p_k) {
    return dflush(ExactNoise::Float4{ p_a.x * p_k, p_a.y * p_k, p_a.z * p_k, p_a.w * p_k });
}

// A box of lattice indices, bounds included.
struct Box {
    int64_t lo[3] = { 0, 0, 0 };
    int64_t hi[3] = { 0, 0, 0 };

    int64_t size(int p_axis) const { return hi[p_axis] - lo[p_axis] + 1; }
    int64_t count() const { return size(0) * size(1) * size(2); }
    bool contains(const Vector3i64 &p_p) const {
        return p_p.x >= lo[0] && p_p.x <= hi[0] && p_p.y >= lo[1] && p_p.y <= hi[1] && p_p.z >= lo[2] && p_p.z <= hi[2];
    }
    // x-major linear index of a contained point.
    int64_t index(const Vector3i64 &p_p) const { return ((p_p.x - lo[0]) * size(1) + (p_p.y - lo[1])) * size(2) + (p_p.z - lo[2]); }
};

inline Box point_box(const Vector3i64 &p_p) {
    return Box{ { p_p.x, p_p.y, p_p.z }, { p_p.x, p_p.y, p_p.z } };
}

// Calls p_f(point, linear index) for every point of the box, in x-major order.
template <typename F>
void for_each_point(const Box &p_box, F p_f) {
    int64_t i = 0;
    for (int64_t x = p_box.lo[0]; x <= p_box.hi[0]; x++) {
        for (int64_t y = p_box.lo[1]; y <= p_box.hi[1]; y++) {
            for (int64_t z = p_box.lo[2]; z <= p_box.hi[2]; z++, i++) {
                p_f(Vector3i64(x, y, z), i);
            }
        }
    }
}

// floor(p_v / 2), for negative indices too (inner loop: a shift, not WorldCoord::floor_div).
inline int64_t half_floor(int64_t p_v) {
    return p_v >= 0 ? (p_v >> 1) : ~((~p_v) >> 1);
}

// One axis pass of H from the points of p_in onto those of p_out (p_in with axis p_axis replaced by child indices):
// p_f(child index, parent a's index, parent b's index, copy) for every child point in x-major order, both indices
// x-major in p_in. A copy (an even child index) reads parent a only.
template <typename F>
void axis_pass(const Box &p_in, const Box &p_out, int p_axis, F p_f) {
    const int64_t in_stride[3] = { p_in.size(1) * p_in.size(2), p_in.size(2), 1 };
    for_each_point(p_out, [&](const Vector3i64 &p_p, int64_t p_o) {
        const int64_t idx[3] = { p_p.x, p_p.y, p_p.z };
        const int64_t c = idx[p_axis];
        int64_t ia = 0;
        for (int a = 0; a < 3; a++) {
            ia += ((a == p_axis ? half_floor(c) : idx[a]) - p_in.lo[a]) * in_stride[a];
        }
        p_f(p_o, ia, ia + in_stride[p_axis], (c & 1) == 0);
    });
}

// The axis rule of H at one child point for one group of CHANNELS channels at slot p_base: p_get(slot, b) reads the
// parent a (b false) or b (b true), p_put(slot, v) writes the child. A copy reads parent a only.
template <typename Get, typename Put>
void axis_rule(int p_base, int p_axis, bool p_copy, Get p_get, Put p_put) {
    auto pair = [&](int p_v, int p_s) {
        float v, s;
        if (p_copy) {
            hermite_copy(p_get(p_v, false), p_get(p_s, false), v, s);
        } else {
            hermite_mid(p_get(p_v, false), p_get(p_v, true), p_get(p_s, false), p_get(p_s, true), v, s);
        }
        p_put(p_v, v);
        p_put(p_s, s);
    };
    // The transverse slopes are interpolated linearly.
    const int axis_slope = p_base + DX + p_axis;
    pair(p_base + VALUE, axis_slope);
    for (int s = p_base + DX; s <= p_base + DZ; s++) {
        if (s != axis_slope) {
            p_put(s, p_copy ? p_get(s, false) : (p_get(s, false) + p_get(s, true)) * 0.5f);
        }
    }
}

// The parent points H reads to produce p_child: floor(lo / 2) .. ceil(hi / 2).
Box parent_box(const Box &p_child);

// The boxes of levels p_lod .. p_top (index m - p_lod) for p_box on Λ_{p_lod}.
std::vector<Box> level_boxes(const Box &p_box, int p_lod, int p_top);

// Closed form of level_boxes (the GPU's): the box of level m = L + p_j for p_box on Λ_L,
// lo_m = floor(lo_L / 2^j), hi_m = ceil(hi_L / 2^j) (unit-tested equal to the iterated parent_box).
Box level_box(const Box &p_box, int p_j);

// Upper bound of a level box's size per axis, p_j levels above a box of p_size points per axis, whatever
// its position: n_0 = p_size, n_{j+1} = ceil(n_j / 2) + 1 (17, 10, 6, 4, 3, 3, ... for 32). The GPU's
// scratch strides and dispatch sizes.
int level_bound(int p_size, int p_j);

// The samples of a chunk (vertex grid, skirt included) on its own lattice Λ_L.
Box chunk_box(const ChunkId &p_chunk, const ChunkParam &p_param);
// The samples of an item anchored at p_anchor on Λ_lod (the level shader's item frame): lo = anchor >> lod - skirt,
// chunk_size points per axis. A chunk's anchor is its minimum voxel; a point item's is its point (skirt 0, size 1).
Box item_box(const VoxelPos &p_anchor, int p_lod, const ChunkParam &p_param);

// A random chunk id at LOD p_lod (odd multiple of 2^L per axis, |id| < 2^40): the checks' and tests' chunks.
ChunkId random_chunk(int p_lod, uint32_t p_seed, int p_index);

// A band is one octave of perlin3 (wavelength 2^k voxels, value * amp): a program's NOTE, at amp 1.
using Band = ExactNoise::Octave;

// |n| of a shape is smoothed: sabs(n) = sqrt(n^2 + e^2) - e, e = SMOOTH_ABS_E. A crease at n = 0 is finer than
// the 8 samples per wavelength a note is carried by: the Hermite subdivision rings along the lattice axes around it,
// and the gradient channels miss it (measured, the climate report: 16-21% gradient error on ridged bands, 2.5% plain). e = 0.1 is about how far n moves between two samples.
inline constexpr float SMOOTH_ABS_E = 0.1f;
// sabs of a dual n (gradient per lattice spacing): (n.g n / r, (n^2 + e^2) r - e), r = det_rsqrt(n^2 + e^2), from n^2
// (dmul(n, n)). Products flushed.
inline ExactNoise::Float4 smooth_abs(const ExactNoise::Float4 &p_n, const ExactNoise::Float4 &p_n2) {
    const float q = p_n2.w + SMOOTH_ABS_E * SMOOTH_ABS_E;
    const float r = ExactNoise::det_rsqrt(q);
    const float d = flush_value(p_n.w * r);
    return dflush(ExactNoise::Float4{ p_n.x * d, p_n.y * d, p_n.z * d, q * r - SMOOTH_ABS_E });
}
// E[sabs(n)] and E[n^2] of ExactNoise::perlin3 (unit test: measured on 2^20 random points within 1%).
inline constexpr float NOISE_ABS_MEAN = 0.1527f;
inline constexpr float NOISE_SQ_MEAN = 0.0728f;
// |W| bound of a warp (LOD0 voxels): its floor stays an exact float.
inline constexpr float WARP_LIMIT = 4194304.0f; // 2^22

// The numbers a generated shader reads from a buffer instead of as literals: a program's (its biomes' places, amplitudes,
// shapes, curves, constants). Its text then depends on the table's structure alone (which nodes, which coefficients are
// zero), so an edit of a value compiles nothing: the buffer is re-uploaded. Each call hands out the next slot, holding
// p_value's bits, of the uniform block of vec4s (std140) the level shader binds at CascadeGLSL::PARAM_BINDING: reads at
// constant indices, as cheap as the literals were (not a storage buffer: WebGPU caps those per stage).
struct ShaderParams {
    std::vector<float> values;
    ov::str operator()(float p_value) {
        const int64_t i = int64_t(values.size());
        values.push_back(p_value);
        return ov::format("cascade_param[%d].%c", i / 4, "xyzw"[i % 4]);
    }
    // p_values in consecutive slots: the first one's index (a generic text reads a table's slots at indices it computes:
    // at()).
    int64_t append(const std::vector<float> &p_values) {
        const int64_t i = int64_t(values.size());
        values.insert(values.end(), p_values.begin(), p_values.end());
        return i;
    }
    // The GLSL read of the slot at p_index, an int expression (CascadeGLSL::level_shader defines cl_param).
    static ov::str at(const ov::str &p_index) { return "cl_param(" + p_index + ")"; }
    // A generic text's variable-length records (the curves, the interpreter's levels): after values (level_shader
    // appends them last), so a record's length never moves the index of a value the text names. p_values' index on
    // the heap, read by heap_at (cl_heap: cl_param past the values, their count a literal of the structure).
    std::vector<float> heap;
    int64_t put(const std::vector<float> &p_values) {
        const int64_t i = int64_t(heap.size());
        heap.insert(heap.end(), p_values.begin(), p_values.end());
        return i;
    }
    static ov::str heap_at(const ov::str &p_index) { return "cl_heap(" + p_index + ")"; }
    // p_values in consecutive slots, read at the GLSL index p_index (an int expression, 0 <= p_index < p_values.size()).
    ov::str table(const std::vector<float> &p_values, const ov::str &p_index) { return at(ov::format("%d + (%s)", append(p_values), p_index)); }
    // The block's vec4s (at least one): its size in the text, a function of the structure as the rest of it.
    int64_t vec4s() const { return ov::max(int64_t(1), (int64_t(values.size()) + 3) / 4); }
};

// A note's noise at lattice point p_p of its insertion lattice Λ_p_m (perlin3, its gradient per cell of 2^k voxels),
// stretched along p_base's up if the band is (BaseSurface::stretched), read displaced by p_warp (null: none; else the
// displacement per axis, LOD0 voxels, gradients per lattice spacing of Λ_m, clamped to WARP_LIMIT), the displacement's
// gradient applied by the chain rule.
ExactNoise::Float4 note_noise(const BaseSurface &p_base, const Band &p_band, int p_m, const Vector3i64 &p_p, const ExactNoise::Float4 *p_warp);

} // namespace Cascade
