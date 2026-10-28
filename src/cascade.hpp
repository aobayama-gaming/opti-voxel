#pragma once

#include "base_surface.hpp"
#include "chunk_math.hpp"
#include "exact_noise.hpp"
#include "ov/math.hpp"
#include "ov/str.hpp"
#include "ov/types.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
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
// the coarser one doesn't insert (k - shift < L: replaced by their mean, 0). With a climate's carry
// (Modulation::derive) a run's last level reads its modulated bands' amplitudes from the level above it: the samples
// of LOD L are a pure function of (p, L), its band L amplitude not the one a finer run's Λ_L reads.
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

// Field groups (Spec::fields, the climate): LINEAR4 channel sets carried down with the ground, each after the
// ground's CHANNELS slots: field f's channel c is Level slot field_slot(f, c). Group 0 is the ground, 1 + f field f.
// The GPU code is generated per spec (Spec::group_count), so the cap only sizes the CPU's slot arrays: a world of several
// landforms (adapter/godot/project/climate/region_test.json) carries more than 8 groups.
inline constexpr int MAX_FIELD_GROUPS = 24;
inline constexpr int MAX_GROUPS = 1 + MAX_FIELD_GROUPS;
static_assert(MAX_GROUPS < 32, "group masks (Modulation::modulated, cell_groups, (1 << groups) - 1) are 32-bit");
// A field group's channels (LINEAR4: value and gradient).
inline constexpr int FIELD_CHANNELS = 4;
// Carry groups (Modulation::derive): LINEAR4 channel sets after the field groups, what a level computes from its
// channels for the level below it (carry_slot(c, channel)).
inline constexpr int MAX_CARRY = 24;
inline constexpr int SLOTS = CHANNELS + FIELD_CHANNELS * (MAX_FIELD_GROUPS + MAX_CARRY);
constexpr int field_slot(int p_field, int p_channel = VALUE) {
    return CHANNELS + FIELD_CHANNELS * p_field + p_channel;
}
constexpr int carry_slot(int p_carry, int p_channel = VALUE) {
    return field_slot(MAX_FIELD_GROUPS + p_carry, p_channel);
}
// Group g's channel set (the ground's p_ground, a field's LINEAR4) and first Level slot.
inline ChannelSet group_set(ChannelSet p_ground, int p_group) {
    return p_group == 0 ? p_ground : ChannelSet::LINEAR4;
}
constexpr int group_slot(int p_group) {
    return p_group == 0 ? 0 : field_slot(p_group - 1);
}

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
// denormal part never reaches a result (the climate's exactness note, climate.hpp). The GLSL twins are cl_flush,
// cl_mul, cl_sub (climate_glsl.cpp).
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

struct Level {
    int m = 0;
    Box box;
    ChannelSet set = ChannelSet::HERMITE8;
    int fields = 0; // field groups carried (slots field_slot(0 .. fields - 1))
    int carry = 0; // carry groups carried (slots carry_slot(0 .. carry - 1))
    int copied = -1; // the carry group H copies (Modulation::copied_carry)
    std::vector<float> ch[SLOTS];

    void allocate(int p_m, const Box &p_box, ChannelSet p_set, int p_fields = 0, int p_carry = 0);
    // Carries p_carry carry groups, the new ones 0 (a derive writes them).
    void add_carry(int p_carry);
    float at(int p_channel, const Vector3i64 &p_p) const { return ch[p_channel][box.index(p_p)]; }
    // All channels at a point, 0 for the ones the set doesn't carry.
    void channels_at(const Vector3i64 &p_p, float r_ch[CHANNELS]) const;
    // Every slot of the point at box index p_i (the fields' too), 0 for the slots not carried: point_slots' layout.
    void slots_at(int64_t p_i, float r_slots[SLOTS]) const {
        for (int s = 0; s < SLOTS; s++) {
            r_slots[s] = ch[s].empty() ? 0.0f : ch[s][size_t(p_i)];
        }
    }
};

// A band is one octave of perlin3 (wavelength 2^k voxels, value * amp).
using Band = ExactNoise::Octave;

// Where the field groups and the carry are carried in a run down to LOD L. A modulated band above the run's last level
// reads the fields at its own level (floor <= m <= ceiling, the finest / coarsest modulated ground level); the last
// level reads none: its band and its samples' terms (sample: the climate's height terms, terraces, pick) read the carry
// that level L + 1 derives (Modulation::derive). So the fields at m >= lowest(L) (monotone in m: a level carrying them
// always finds them in its parent level), the carry at L + 1 (written) and L (read) where the last level reads it.
// The ground is the same bits whatever else is carried. Empty (no modulation): nothing anywhere. The CPU's ground
// queries and the GPU's passes both follow it; a query for the fields themselves (Climate's fields_at) carries them
// everywhere.
struct FieldRange {
    int floor = MAX_LEVEL + 1;
    int ceiling = -1;
    bool sample = false;

    bool empty() const { return floor > ceiling && !sample; }
    // Every run's last level reads the carry, every level above it has the fields: no ground-only level.
    bool everywhere() const { return sample; }
    // Whether a run down to p_lod has a last level that reads the carry (its samples' terms, or its band modulated).
    bool reads_carry(int p_lod) const { return sample || (floor <= p_lod && p_lod <= ceiling); }
    // The finest level with the fields in a run down to p_lod (past MAX_LEVEL: none).
    int lowest(int p_lod) const {
        if (sample) {
            return p_lod + 1;
        }
        return p_lod <= ceiling ? ov::max(floor, p_lod + 1) : MAX_LEVEL + 2;
    }
    bool carried(int p_lod, int p_m) const { return p_m >= lowest(p_lod); }
    // Whether level p_m inserts the fields' own bands: above lowest only. The lowest level's bands and derive read the
    // fields before its insertions and no level below carries them, so its field octaves would reach no reader.
    bool inserted(int p_lod, int p_m) const { return p_m > lowest(p_lod); }
    bool carries(int p_lod, int p_m) const { return (p_m == p_lod || p_m == p_lod + 1) && reads_carry(p_lod); }
    // Whether a batch of LODs p_lods (min, max) carries the fields (batch) at level j above some item's LOD (the largest
    // LOD at most the ceiling is the first to need them), and whether one of its last levels reads the carry
    // (batch_carry, j = 0; level 1 writes it where it has the fields).
    bool batch(const ov::ivec2 &p_lods, int p_j) const {
        if (sample) {
            return p_j >= 1;
        }
        return p_j >= 1 && p_lods.x <= ceiling && ov::min(p_lods.y, int32_t(ceiling)) + p_j >= floor;
    }
    bool batch_carry(const ov::ivec2 &p_lods, int p_j) const { return p_j == 0 && (sample || (p_lods.x <= ceiling && p_lods.y >= floor)); }
};

// Per-point band amplitudes (the climate's biome weights, climate.hpp). At every point of Λ_m above a run's last level,
// before that level's insertions, the amplitude of each modulated group's band on Λ_m is computed from the point's
// channels: from H(V_{m+1}) only, so no band reads one inserted at its own level and the groups can modulate each other
// in any order. A run's last level Λ_L computes none: at every point of Λ_{L+1}, from the same channels its own bands
// read (before its insertions: one climate per point), derive computes the amplitudes of Λ_L's bands and the terms its
// samples read into the carry slots, and H carries them to Λ_L (A = H(derive(H(V_{L+2})))): the samples, the most points
// of a run, only interpolate. The top level derives from a
// level above it that holds nothing (Spec::top_level). A modulated band inserts value A n and slopes A dn + n dA
// (LINEAR4 groups only): n = the noise, A = (dA, A) per lattice spacing of Λ_m.
//
// Shaped bands (BandMod::features): the band also inserts A_abs (|n| - NOISE_ABS_MEAN) and A_sq (n^2 - NOISE_SQ_MEAN),
// the means taken out so a coarser LOD dropping the band (mean 0) makes no step; its noise may be read at the point
// displaced by a warp (LOD0 voxels, the warp's gradient applied to the noise's by the chain rule); and the ground's
// band may overwrite the state group's channels (state_group) with S_1 + S_n n + S_abs |n| + S_sq n^2: the state its
// finer octaves read (the multifractals' previous octave). All of it dual numbers, flushed like dmul.
struct Spec;

// BandMod::features bits.
inline constexpr uint32_t MOD_ABS = 1u;
inline constexpr uint32_t MOD_SQ = 2u;
inline constexpr uint32_t MOD_STATE = 4u;
inline constexpr uint32_t MOD_WARP = 8u;
// |n| of a shaped band is smoothed: sabs(n) = sqrt(n^2 + e^2) - e, e = SMOOTH_ABS_E. A crease at n = 0 is finer than
// the 8 samples per wavelength a band is carried by: the Hermite subdivision rings along the lattice axes around it,
// and the gradient channels miss it (measured, the climate report: 16-21% gradient error on ridged bands, 2.5% plain;
// a multifractal's state streaks). e = 0.1 is about how far n moves between two samples.
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

// One modulated group's band on Λ_m at a point. Float4 = (gradient per lattice spacing of Λ_m, value).
struct BandMod {
    uint32_t features = 0;
    ExactNoise::Float4 amp[3] = {}; // on n, |n| - NOISE_ABS_MEAN, n^2 - NOISE_SQ_MEAN
    ExactNoise::Float4 state[4] = {}; // the state group's new channels, on 1, n, |n|, n^2 (MOD_STATE, the ground only)
    ExactNoise::Float4 warp[3] = {}; // displacement per axis, LOD0 voxels (MOD_WARP)
};

struct Modulation {
    virtual ~Modulation() = default;
    // Bit g: group g's band on Λ_m is modulated (the others insert their constant amplitude).
    virtual uint32_t modulated(int p_m) const = 0;
    // The finest and coarsest level with a modulated ground band (FieldRange: only the ground reaches the mesh; a field
    // band modulated elsewhere only changes the fields where they are carried, Climate's fields_at).
    virtual FieldRange field_range() const = 0;
    // r_mod[g] for every modulated group g of Λ_m, from p_pre: the point's Level slots after subdivision, before Λ_m's
    // insertions; p_carried (Λ_m is the run's last level): from its carry slots, H of what Λ_{m+1} derived.
    virtual void modulate(int p_m, bool p_carried, const float p_pre[SLOTS], BandMod r_mod[MAX_GROUPS]) const = 0;
    // The carry groups the derive writes (at most MAX_CARRY), and the one H copies from the floor parent instead of
    // interpolating (a value that can't be blended, the climate's pick; -1: none).
    virtual int carry_groups() const { return 0; }
    virtual int copied_carry() const { return -1; }
    // The carry groups Λ_m reads as a run's last level without the fields (the first ones; with them, Climate's
    // fields_at, it reads them all): the GPU's carry (Spec::gpu_carry_groups).
    virtual int carry_used(int p_m) const {
        (void)p_m;
        return carry_groups();
    }
    // Writes the carry slots of r_slots, the slots of lattice point p_p of Λ_m before its insertions (modulate's p_pre),
    // for Λ_{m-1} as a run's last level: its bands' amplitudes and the terms its samples read.
    virtual void derive(int p_m, const Vector3i64 &p_p, float r_slots[SLOTS]) const {
        (void)p_m;
        (void)p_p;
        (void)r_slots;
    }
    // The field group the ground's MOD_STATE band overwrites, or -1 (a band-less group: nothing else writes it).
    virtual int state_group() const { return -1; }
    // Whether any band uses a BandMod feature: else the GLSL is the plain amplitudes' (the same text as before them).
    virtual bool shaped() const { return false; }
    // The features group p_group's bands use at some level (the union): its GLSL insert has the code for them only.
    virtual uint32_t group_features(int p_group) const {
        (void)p_group;
        return 0u;
    }
    // The same in GLSL, bit for bit. Not shaped: `uint cascade_amplitudes(int m, Ch pre, out vec4 amp[MAX_GROUPS])`
    // returning modulated(m); shaped: `uint cascade_mods(int m, Ch pre, out BandMod mod[MAX_GROUPS])`, the same mask
    // (CascadeGLSL's insert code). pre in p_spec's GPU layout (Spec::gpu_base, gpu_carry); p_carried: modulate's, for the
    // samples' view (p_spec without the fields). p_derive (the fields' view, p_spec with the carry): `inout Ch pre` and a
    // last `WorldPos p`, and where g_carry_n says, derive too, from the same weights, into pre's carry groups.
    virtual ov::str glsl(const Spec &p_spec, bool p_carried, bool p_derive) const = 0;
};

struct Spec {
    std::vector<Band> bands;
    // Band k is inserted on Λ_{k - insert_shift}: 2 = 4 samples per wavelength, 3 = 8 (the chosen one).
    int insert_shift = ExactNoise::BAND_SHIFT;
    ChannelSet set = ChannelSet::HERMITE8;
    // Coarsest lattice T; -1: the highest insertion lattice of the ground's bands and, where they are carried
    // (FieldRange not empty, or p_all_fields), the fields'.
    int top = -1;
    // Field groups (at most MAX_FIELD_GROUPS, LINEAR4, the same insert shift; at most one band per level each),
    // carried at slots field_slot(f): the climate. They never reach the ground's channels except through modulation.
    std::vector<std::vector<Band>> fields;
    // Null: every band at its constant amplitude. A modulated ground needs LINEAR4.
    std::shared_ptr<const Modulation> modulation;
    // Bit g: group g's bands are cell noise (ExactNoise::cell3, field groups only) instead of perlin3.
    uint32_t cell_groups = 0;
    // The base a stretched band stretches along (BaseSurface::stretched): the ground's (SDFCascade), flat at datum 0
    // otherwise.
    BaseSurface base = BaseSurface::flat(0);

    int insert_level(const Band &p_band) const { return p_band.k - insert_shift; }
    // Some band of group p_group (any group: -1) is stretched along y (Band::stretch): its GLSL inserts go through
    // the base's base_stretched.
    bool stretched(int p_group = -1) const;
    int top_level(bool p_all_fields = false) const;
    // The top lattice of a run down to p_lod: top_level, at least p_lod, and above p_lod where its last level reads the
    // carry (the level that derives it, empty above the highest band). The GPU's: max(top, lod + 1) in a carry view.
    int run_top(int p_lod, bool p_all_fields = false) const;
    int group_count() const { return 1 + int(fields.size()); }
    const std::vector<Band> &group_bands(int p_group) const { return p_group == 0 ? bands : fields[size_t(p_group - 1)]; }
    ChannelSet group_set(int p_group) const { return Cascade::group_set(set, p_group); }
    FieldRange field_range() const { return modulation ? modulation->field_range() : FieldRange(); }
    int carry_groups() const { return modulation ? modulation->carry_groups() : 0; }
    // The carry groups some run's last level reads without the fields (the GPU's: its runs carry no field to it).
    int gpu_carry_groups() const;
    int copied_carry() const { return modulation ? modulation->copied_carry() : -1; }
    // The ground alone: no field group, no modulation.
    Spec ground_only() const {
        Spec s = *this;
        s.fields.clear();
        s.modulation.reset();
        return s;
    }
    // The ground and the carry, no field group: the GPU's view of a run's last level.
    Spec without_fields() const {
        Spec s = *this;
        s.fields.clear();
        return s;
    }
    // The GPU's layout (CascadeGLSL's Ch, scratch and debug channels): the ground's channel_count(set), then 4 per
    // field group, then 4 per carry group; slot_of_gpu maps a GPU channel to its Level slot.
    int gpu_channels() const { return gpu_carry(gpu_carry_groups()); }
    int gpu_base(int p_group) const { return p_group == 0 ? 0 : channel_count(set) + FIELD_CHANNELS * (p_group - 1); }
    int gpu_carry(int p_carry) const { return gpu_base(group_count()) + FIELD_CHANNELS * p_carry; }
    int slot_of_gpu(int p_channel) const {
        const int n0 = channel_count(set);
        if (p_channel < n0) {
            return p_channel;
        }
        const int f = (p_channel - n0) / FIELD_CHANNELS;
        const int c = (p_channel - n0) % FIELD_CHANNELS;
        return f < int(fields.size()) ? field_slot(f, c) : carry_slot(f - int(fields.size()), c);
    }
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
// Adds the bands of the ground and of p_fields field groups (-1: every one r_level carries) on r_level.m (modulated where
// the spec says) to its channels. Without its field groups' bands a level writes no state group either.
// A level holding carry is its run's last (the level above gets its carry after its insertions, derived from its channels
// before them): its modulated bands
// read it (Modulation::modulate).
void insert_bands(const Spec &p_spec, Level &r_level, int p_fields = -1);
// The same at one lattice point p_p of Λ_m: r_slots holds every slot after subdivision (H(V_{m+1})), the level's
// insertions are added (a modulated band: value n.w A, slopes n.s (A 2^-shift) + n.w dA, the GPU's order).
void insert_at(const Spec &p_spec, int p_m, const Vector3i64 &p_p, float r_slots[SLOTS]);

// H of p_parent onto p_child_box, carrying the ground, p_fields field groups and p_carry carry groups (-1: the
// parent's; at most them). The copied carry group (Level::copied) takes its floor parent's channels.
Level subdivide(const Level &p_parent, const Box &p_child_box, int p_fields = -1, int p_carry = -1);
void flush(Level &r_level);

// Runs the cascade from the top lattice down to Λ_{p_lod} over the points of p_box (on Λ_{p_lod}).
// r_levels (optional) receives every level, top first.
// The fields are carried where the spec's FieldRange says, everywhere with p_all_fields.
Level run(const Spec &p_spec, const Box &p_box, int p_lod, std::vector<Level> *r_levels = nullptr, const Source &p_source = Source(),
        bool p_all_fields = false);

// V_m at one lattice point of Λ_m (the ground's channels).
void point(const Spec &p_spec, const Vector3i64 &p_p, int p_m, float r_ch[CHANNELS]);

// point() without a heap allocation (the discoverer's corners, SDFCascade::evaluate): a point's level boxes
// have at most POINT_AXIS points per axis (level_bound(1, j)), so every level and every axis pass of H fits
// in stack arrays. The same ops on the same values in the same order as run(): the same bits (unit-tested).
inline constexpr int POINT_AXIS = 2;
void point_into(const Spec &p_spec, const Vector3i64 &p_p, int p_m, float r_ch[CHANNELS]);
// point_into with every slot (the fields' too, field_slot), 0 for the slots not carried (FieldRange; p_all_fields:
// the fields everywhere, Climate's fields_at).
void point_slots(const Spec &p_spec, const Vector3i64 &p_p, int p_m, float r_slots[SLOTS], bool p_all_fields = false);

} // namespace Cascade
