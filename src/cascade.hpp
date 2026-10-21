#pragma once

#include "chunk_math.hpp"
#include "exact_noise.hpp"

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

// Field groups (Spec::fields, the climate): LINEAR4 channel sets carried down with the ground, each after the
// ground's CHANNELS slots: field f's channel c is Level slot field_slot(f, c). Group 0 is the ground, 1 + f field f.
inline constexpr int MAX_FIELD_GROUPS = 8;
// Bands one group may insert on one lattice (they add up in list order); a modulated group has one.
inline constexpr int MAX_LEVEL_BANDS = 4;
inline constexpr int MAX_GROUPS = 1 + MAX_FIELD_GROUPS;
inline constexpr int SLOTS = CHANNELS + 4 * MAX_FIELD_GROUPS;
constexpr int field_slot(int p_field, int p_channel = VALUE) {
    return CHANNELS + 4 * p_field + p_channel;
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
    int fields = 0; // field groups carried (slots field_slot(0 .. fields - 1))
    std::vector<float> ch[SLOTS];

    void allocate(int p_m, const Box &p_box, ChannelSet p_set, int p_fields = 0);
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

// Where the field groups are carried: in a run down to LOD L, at level m iff floor <= m and L <= ceiling, floor /
// ceiling the finest / coarsest modulated level. Only the modulation reads the fields, so nothing below the floor nor
// in a run coarser than every modulated level needs them, and the ground is the same bits whether they are carried or
// not. Empty (no modulation): carried nowhere. The CPU's ground queries and the GPU's passes both follow it; a query
// for the fields themselves (Climate's fields_at) carries them everywhere.
struct FieldRange {
    int floor = MAX_LEVEL + 1;
    int ceiling = -1;

    bool empty() const { return floor > ceiling; }
    // Carried at every level of every run (a pick reads the fields at every chunk's level 0): no ground-only level.
    bool everywhere() const { return floor <= 0 && ceiling >= MAX_LEVEL; }
    bool carried(int p_lod, int p_m) const { return p_m >= floor && p_lod <= ceiling; }
    // Whether a batch of LODs p_lods (min, max) carries them at level j above each item's LOD: monotone in j, so a
    // level carrying them always finds them in its parent level.
    bool batch(const Vector2i &p_lods, int p_j) const { return p_lods.x <= ceiling && MIN(p_lods.y, ceiling) + p_j >= floor; }
};

// Per-point band amplitudes (the climate's biome weights, climate.hpp). At every point of Λ_m, before that level's
// insertions, the amplitude of each modulated group's band on Λ_m is computed from the point's channels: from
// H(V_{m+1}) only, so no band reads one inserted at its own level and the groups can modulate each other in any order.
// A modulated band inserts value A n and slopes A dn + n dA (LINEAR4 groups only): n = perlin3, A = (dA, A) per
// lattice spacing of Λ_m.
struct Spec;

struct Modulation {
    virtual ~Modulation() = default;
    // Bit g: group g's band on Λ_m is modulated (the others insert their constant amplitude).
    virtual uint32_t modulated(int p_m) const = 0;
    // The finest and coarsest level with a modulated ground band (FieldRange: only the ground reaches the mesh; a field
    // band modulated elsewhere only changes the fields where they are carried, Climate's fields_at).
    virtual FieldRange field_range() const = 0;
    // r_amp[g] for every modulated group g of Λ_m, from p_pre: the point's Level slots after subdivision, before Λ_m's
    // insertions. Float4 = (gradient per lattice spacing of Λ_m, value).
    virtual void amplitudes(int p_m, const float p_pre[SLOTS], ExactNoise::Float4 r_amp[MAX_GROUPS]) const = 0;
    // The same in GLSL, bit for bit: `uint cascade_amplitudes(int m, Ch pre, out vec4 amp[MAX_GROUPS])` returning
    // modulated(m), pre in p_spec's GPU layout (Spec::gpu_base).
    virtual String glsl(const Spec &p_spec) const = 0;
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

    int insert_level(const Band &p_band) const { return p_band.k - insert_shift; }
    int top_level(bool p_all_fields = false) const;
    int group_count() const { return 1 + int(fields.size()); }
    const std::vector<Band> &group_bands(int p_group) const { return p_group == 0 ? bands : fields[size_t(p_group - 1)]; }
    ChannelSet group_set(int p_group) const { return Cascade::group_set(set, p_group); }
    int group_slot(int p_group) const { return Cascade::group_slot(p_group); }
    FieldRange field_range() const { return modulation ? modulation->field_range() : FieldRange(); }
    // The GPU's layout (CascadeGLSL's Ch, scratch and debug channels): the ground's channel_count(set), then 4 per
    // field group; slot_of_gpu maps a GPU channel to its Level slot.
    int gpu_channels() const { return channel_count(set) + 4 * int(fields.size()); }
    int gpu_base(int p_group) const { return p_group == 0 ? 0 : channel_count(set) + 4 * (p_group - 1); }
    int slot_of_gpu(int p_channel) const {
        const int n0 = channel_count(set);
        return p_channel < n0 ? p_channel : field_slot((p_channel - n0) / 4, (p_channel - n0) % 4);
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
// Adds the bands of every group r_level carries on r_level.m (modulated where the spec says) to its channels.
void insert_bands(const Spec &p_spec, Level &r_level);
// The same at one lattice point p_p of Λ_m: r_slots holds every slot after subdivision (H(V_{m+1})), the level's
// insertions are added (a modulated band: value n.w A, slopes n.s (A 2^-shift) + n.w dA, the GPU's order).
void insert_at(const Spec &p_spec, int p_m, const Vector3i64 &p_p, float r_slots[SLOTS]);

// H of p_parent onto p_child_box, carrying the ground and p_fields field groups (-1: the parent's; at most them).
Level subdivide(const Level &p_parent, const Box &p_child_box, int p_fields = -1);
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
