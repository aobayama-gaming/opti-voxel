#pragma once

#include "chunk_math.hpp"
#include "exact_noise.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

// Hermite cascade: the CPU reference (oracle) of how noise bands reach every LOD with one
// representation. Not used by the meshers yet (the GPU pass comes later and must match it bit for bit).
//
// Λ_m is the lattice of points at multiples of 2^m LOD0 voxels (absolute int64 lattice indices). A band
// (one octave k) is inserted once, on Λ_{k - insert_shift} (shift 2: 4 samples per wavelength, 3: 8),
// then carried down:
//     V_T = O_T on Λ_T,    V_m = flush(H(V_{m+1}) + O_m) on Λ_m,
// where O_m is the sum of the bands inserted on Λ_m and H the tricubic Hermite subdivision. A LOD-L chunk
// runs the cascade down to its own lattice Λ_L and stops: V_m(p) is a pure function of p and m, so every
// LOD sees bit-identical values at the points it shares with another, and two LODs only differ by the bands
// the coarser one doesn't insert (k - shift < L: replaced by their mean, 0).
//
// Channels per lattice point, in units of the lattice's spacing (channel S times 2^(m |S|) per voxel):
// value, d/dx, d/dy, d/dz, d2/dxdy, d2/dxdz, d2/dydz, d3/dxdydz.
//
// H, one axis at a time (x, then y, then z: fixed order), parent spacing = 2 child spacings, endpoint
// values a, b and slopes A, B along the axis in parent units:
//     child on the parent lattice (even index):  a,                                slope A * 0.5
//     midpoint (odd index):                      (a + b) * 0.5 + (A - B) * 0.125,  slope (b - a) * 0.75 - (A + B) * 0.125
// HERMITE8 applies it to the 4 (value-like, slope-like) channel pairs of the axis: the tensor-product
// cubic Hermite, C1, reproduces any tricubic polynomial; without new bands the cascade is exactly the
// piecewise tricubic Hermite interpolant of the coarse data. LINEAR4 keeps value + gradient only: the pair
// (value, slope along the axis) by the same rule, the two transverse slopes by linear interpolation (the
// 8-channel rule with each cross derivative replaced by the slope difference along the edge).
//
// Exactness: only + - * by the constants above and compares (correctly rounded, `precise` on the GPU);
// copies and the x0.5 slope scalings are exact, so a chunk and a point query give the same bits. Denormals:
// after each level every channel below 2^-64 is flushed to 0 (FLUSH_BITS). Stored values are then
// multiples of 2^-87, and the three passes of H keep every intermediate a multiple of 2^-96: never a
// denormal, whether the GPU flushes them or not.
namespace Cascade {

// Channel order of ExactNoise::Channels8 (perlin3_cross) too.
enum Channel {
    VALUE,
    DX,
    DY,
    DZ,
    DXY,
    DXZ,
    DYZ,
    DXYZ,
    CHANNELS,
};

// Derivative axes of each channel (bit 0 x, bit 1 y, bit 2 z): the one table the channel layout comes from.
inline constexpr int CHANNEL_AXES[CHANNELS] = { 0, 1, 2, 4, 3, 5, 6, 7 };

constexpr int channel_order(int p_channel) {
    return (CHANNEL_AXES[p_channel] & 1) + ((CHANNEL_AXES[p_channel] >> 1) & 1) + ((CHANNEL_AXES[p_channel] >> 2) & 1);
}

// Per axis pass of HERMITE8: the (value-like, slope-like) channel pairs, slope = value + the axis, in the
// order the passes apply them (the GPU's code is generated from this table).
inline constexpr int PAIRS[3][4][2] = {
    { { VALUE, DX }, { DY, DXY }, { DZ, DXZ }, { DYZ, DXYZ } },
    { { VALUE, DY }, { DX, DXY }, { DZ, DYZ }, { DXZ, DXYZ } },
    { { VALUE, DZ }, { DX, DXZ }, { DY, DYZ }, { DXY, DXYZ } },
};

enum class ChannelSet {
    HERMITE8,
    LINEAR4, // value + gradient (VALUE..DZ)
};

inline int channel_count(ChannelSet p_set) {
    return p_set == ChannelSet::HERMITE8 ? 8 : 4;
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

struct Level {
    int m = 0;
    Box box;
    ChannelSet set = ChannelSet::HERMITE8;
    std::vector<float> ch[CHANNELS];

    void allocate(int p_m, const Box &p_box, ChannelSet p_set);
    float at(int p_channel, const Vector3i64 &p_p) const { return ch[p_channel][box.index(p_p)]; }
    // All channels at a point, 0 for the ones the set doesn't carry.
    void channels_at(const Vector3i64 &p_p, float r_ch[CHANNELS]) const;
};

// A band is one octave of perlin3 (wavelength 2^k voxels, value * amp).
using Band = ExactNoise::Octave;

struct Spec {
    std::vector<Band> bands;
    // Band k is inserted on Λ_{k - insert_shift}: 2 = 4 samples per wavelength, 3 = 8 (the chosen one).
    int insert_shift = 3;
    ChannelSet set = ChannelSet::HERMITE8;
    // Coarsest lattice T; -1: the highest insertion lattice of the bands.
    int top = -1;

    int insert_level(const Band &p_band) const { return p_band.k - insert_shift; }
    int top_level() const;
};

// Adds the inserted channels to a level. The default source inserts the spec's bands; tests insert
// other fields (polynomials).
using Source = std::function<void(Level &)>;

// Points per lattice level (cost per chunk), from the boxes alone.
struct Stats {
    int64_t points[MAX_LEVEL + 1] = {};
    int64_t total() const;
};
void count_points(const Spec &p_spec, const Box &p_box, int p_lod, Stats &r_stats);

// Band channels on its insertion lattice at lattice point p: perlin3_cross per cell times
// amp * 2^(-insert_shift |S|).
void band_channels(const Spec &p_spec, const Band &p_band, const Vector3i64 &p_p, float r_ch[CHANNELS]);
void insert_bands(const Spec &p_spec, Level &r_level);

Level subdivide(const Level &p_parent, const Box &p_child_box);
void flush(Level &r_level);

// Runs the cascade from the top lattice down to Λ_{p_lod} over the points of p_box (on Λ_{p_lod}).
// r_levels (optional) receives every level, top first.
Level run(const Spec &p_spec, const Box &p_box, int p_lod, std::vector<Level> *r_levels = nullptr, const Source &p_source = Source());

// V_m at one lattice point of Λ_m.
void point(const Spec &p_spec, const Vector3i64 &p_p, int p_m, float r_ch[CHANNELS]);

// point() without a heap allocation (the discoverer's corners, SDFCascade::evaluate): a point's level boxes
// have at most POINT_AXIS points per axis (level_bound(1, j)), so every level and every axis pass of H fits
// in stack arrays. The same ops on the same values in the same order as run(): the same bits (unit-tested).
inline constexpr int POINT_AXIS = 2;
void point_into(const Spec &p_spec, const Vector3i64 &p_p, int p_m, float r_ch[CHANNELS]);

} // namespace Cascade
