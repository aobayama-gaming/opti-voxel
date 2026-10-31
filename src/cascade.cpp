#include "cascade.hpp"

#include <algorithm>
#include <utility>

namespace Cascade {

static_assert(sizeof(ExactNoise::Channels8::c) / sizeof(float) == CHANNELS, "perlin3_cross returns the cascade's channels");

namespace {

// floor(p_v / 2), for negative indices too (inner loop: a shift, not WorldCoord::floor_div).
int64_t half_floor(int64_t p_v) {
    return p_v >= 0 ? (p_v >> 1) : ~((~p_v) >> 1);
}

constexpr bool pairs_match_axes() {
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < 4; i++) {
            const int v = CHANNEL_AXES[PAIRS[a][i][0]];
            if ((v & (1 << a)) != 0 || CHANNEL_AXES[PAIRS[a][i][1]] != (v | (1 << a))) {
                return false;
            }
        }
    }
    return true;
}
static_assert(pairs_match_axes(), "PAIRS follows CHANNEL_AXES");

// The axis rule of H at one child point for one group (channel set p_set at slot p_base): p_get(slot, b) reads the
// parent a (b false) or b (b true), p_put(slot, v) writes the child. A copy reads parent a only.
template <typename Get, typename Put>
void axis_rule(ChannelSet p_set, int p_base, int p_axis, bool p_copy, Get p_get, Put p_put) {
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
    if (p_set == ChannelSet::HERMITE8) {
        for (const auto &pr : PAIRS[p_axis]) {
            pair(p_base + pr[0], p_base + pr[1]);
        }
        return;
    }
    // LINEAR4: the transverse slopes are interpolated linearly.
    const int axis_slope = p_base + DX + p_axis;
    pair(p_base + VALUE, axis_slope);
    for (int s = p_base + DX; s <= p_base + DZ; s++) {
        if (s != axis_slope) {
            p_put(s, p_copy ? p_get(s, false) : (p_get(s, false) + p_get(s, true)) * 0.5f);
        }
    }
}

// The axis rule for every group a level carries: the ground (set p_set), p_fields field groups and p_carry carry
// groups, the copied one (p_copied) from parent a whether the child is a copy or a midpoint.
template <typename Get, typename Put>
void axis_rules(ChannelSet p_set, int p_fields, int p_carry, int p_copied, int p_axis, bool p_copy, Get p_get, Put p_put) {
    for (int g = 0; g < 1 + p_fields; g++) {
        axis_rule(group_set(p_set, g), group_slot(g), p_axis, p_copy, p_get, p_put);
    }
    for (int c = 0; c < p_carry; c++) {
        if (c != p_copied) {
            axis_rule(ChannelSet::LINEAR4, carry_slot(c), p_axis, p_copy, p_get, p_put);
            continue;
        }
        for (int s = carry_slot(c); s < carry_slot(c + 1); s++) {
            p_put(s, p_get(s, false));
        }
    }
}

// One axis of H: p_in -> the same box with axis p_axis replaced by [p_lo, p_hi] (child indices), the ground, p_fields
// field groups and p_carry carry groups (<= p_in's).
Level subdivide_axis(const Level &p_in, int p_fields, int p_carry, int p_axis, int64_t p_lo, int64_t p_hi) {
    Box box = p_in.box;
    box.lo[p_axis] = p_lo;
    box.hi[p_axis] = p_hi;
    Level out;
    out.allocate(p_in.m, box, p_in.set, p_fields, p_carry);
    out.copied = p_in.copied;
    const int64_t in_stride[3] = { p_in.box.size(1) * p_in.box.size(2), p_in.box.size(2), 1 };
    for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_o) {
        const int64_t idx[3] = { p_p.x, p_p.y, p_p.z };
        const int64_t c = idx[p_axis];
        int64_t ia = 0;
        for (int a = 0; a < 3; a++) {
            ia += ((a == p_axis ? half_floor(c) : idx[a]) - p_in.box.lo[a]) * in_stride[a];
        }
        const int64_t ib = ia + in_stride[p_axis];
        axis_rules(
                p_in.set, p_fields, p_carry, p_in.copied, p_axis, (c & 1) == 0, [&](int p_s, bool p_b) { return p_in.ch[p_s][p_b ? ib : ia]; },
                [&](int p_s, float p_v) { out.ch[p_s][p_o] = p_v; });
    });
    return out;
}

// Channel scale of a band per derivative order: amp * 2^(-shift |S|), exact.
struct BandFactors {
    float f[4] = {};
    BandFactors() = default;
    BandFactors(const Spec &p_spec, const Band &p_band) {
        for (int o = 0; o < 4; o++) {
            f[o] = p_band.amp * ExactNoise::exact_pow2(-p_spec.insert_shift * o);
        }
    }
};

WorldPos band_pos(const Spec &p_spec, const Band &p_band, const Vector3i64 &p_p) {
    const int64_t step = int64_t(1) << p_spec.insert_level(p_band);
    WorldPos pos;
    pos.voxel = Vector3i64(p_p.x * step, p_p.y * step, p_p.z * step);
    return pos;
}

// A band's noise at p_pos (perlin3). A stretched band reads it
// stretched along the base's up (BaseSurface::stretched, the base sample at the band's insertion lattice), its gradient
// back through the stretch (per cell of 2^k voxels).
ExactNoise::Float4 band_noise(const Spec &p_spec, const Band &p_band, const WorldPos &p_pos) {
    const bool stretch = p_band.stretch > 0;
    const ExactNoise::Float4 b = stretch ? p_spec.base.stretch_base(p_pos.voxel, p_spec.insert_level(p_band)) : ExactNoise::Float4{};
    const WorldPos pos = stretch ? p_spec.base.stretched(p_pos, p_band.stretch, b, p_spec.insert_level(p_band)) : p_pos;
    const ExactNoise::Float4 n = ExactNoise::perlin3(pos, p_band.k, p_band.seed, p_band.offset);
    return stretch ? p_spec.base.stretch_gradient(n, p_band.stretch, b) : n;
}

void band_channels(const Spec &p_spec, int p_group, const Band &p_band, const BandFactors &p_factors, const Vector3i64 &p_p, float r_ch[CHANNELS]) {
    const WorldPos pos = band_pos(p_spec, p_band, p_p);
    const float *factor = p_factors.f;
    if (p_spec.group_set(p_group) == ChannelSet::HERMITE8) {
        if (p_band.stretch != 0) {
            std::fill(r_ch, r_ch + CHANNELS, 0.0f);
            OV_ERR_FAIL_MSG("Cascade: a stretched band needs LINEAR4 (perlin3_cross has no stretch).");
        }
        const ExactNoise::Channels8 n = ExactNoise::perlin3_cross(pos, p_band.k, p_band.seed, p_band.offset);
        for (int c = 0; c < CHANNELS; c++) {
            r_ch[c] = n.c[c] * factor[channel_order(c)];
        }
        return;
    }
    // The same bits as perlin3_cross's first four channels, without the cross derivatives.
    const ExactNoise::Float4 n = band_noise(p_spec, p_band, pos);
    r_ch[VALUE] = n.w * factor[0];
    r_ch[DX] = n.x * factor[1];
    r_ch[DY] = n.y * factor[1];
    r_ch[DZ] = n.z * factor[1];
}

// p_pos displaced by the warp's values (LOD0 voxels, clamped to WARP_LIMIT): the floor to the voxel, the rest to the
// fraction (below 1: a rest that rounds to 1 carries).
WorldPos warped(const WorldPos &p_pos, const ExactNoise::Float4 p_warp[3]) {
    WorldPos pos = p_pos;
    float *frac[3] = { &pos.frac.x, &pos.frac.y, &pos.frac.z };
    for (int a = 0; a < 3; a++) {
        const float w = std::clamp(p_warp[a].w, -WARP_LIMIT, WARP_LIMIT);
        float fl = std::floor(w);
        float rest = w - fl;
        if (rest >= 1.0f) {
            fl = fl + 1.0f;
            rest = 0.0f;
        }
        pos.voxel[a] += int64_t(fl);
        *frac[a] = rest;
    }
    return pos;
}

// A modulated band's 4 channels (BandMod p_mod, Modulation), p_slope = 2^-shift. The plain term in the order of the
// GPU's insert_band_mod: value n.w A, slope n.s (A p_slope) + n.w dA; the shaped terms (insert_band_shaped) added to it
// as flushed dual products, and r_state (MOD_STATE) the state's new channels.
void modulated_channels(const Spec &p_spec, const Band &p_band, const BandMod &p_mod, float p_slope, const Vector3i64 &p_p, float r_o[4],
        ExactNoise::Float4 *r_state) {
    using ExactNoise::Float4;
    WorldPos pos = band_pos(p_spec, p_band, p_p);
    const bool warp = (p_mod.features & MOD_WARP) != 0u;
    if (warp) {
        pos = warped(pos, p_mod.warp);
    }
    Float4 n = band_noise(p_spec, p_band, pos);
    if (warp) {
        // The chain rule through the displacement: dn' = (I + J)^T dn, J_ab = dW_a / dx_b per voxel (the warp's gradient
        // per lattice spacing of Λ_m times 2^-m), each product flushed.
        const float to_voxel = ExactNoise::exact_pow2(-p_spec.insert_level(p_band));
        const float d[3] = { n.x, n.y, n.z };
        float out[3];
        for (int b = 0; b < 3; b++) {
            float s = 0.0f;
            for (int a = 0; a < 3; a++) {
                const float grad[3] = { p_mod.warp[a].x, p_mod.warp[a].y, p_mod.warp[a].z };
                s = s + flush_value(flush_value(grad[b] * to_voxel) * d[a]);
            }
            out[b] = d[b] + flush_value(s);
        }
        n = Float4{ out[0], out[1], out[2], n.w };
    }
    const Float4 &amp = p_mod.amp[0];
    const float aws = amp.w * p_slope;
    r_o[VALUE] = n.w * amp.w;
    r_o[DX] = n.x * aws + n.w * amp.x;
    r_o[DY] = n.y * aws + n.w * amp.y;
    r_o[DZ] = n.z * aws + n.w * amp.z;
    if ((p_mod.features & (MOD_ABS | MOD_SQ | MOD_STATE)) == 0u) {
        return;
    }
    // n, |n| and n^2 as duals, gradients per lattice spacing.
    const Float4 nd = dflush(Float4{ n.x * p_slope, n.y * p_slope, n.z * p_slope, n.w });
    const Float4 n2 = dmul(nd, nd);
    const Float4 an = (p_mod.features & (MOD_ABS | MOD_STATE)) ? smooth_abs(nd, n2) : Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    Float4 shaped{ 0.0f, 0.0f, 0.0f, 0.0f };
    if (p_mod.features & MOD_ABS) {
        shaped = dadd(shaped, dmul(p_mod.amp[1], dsub(an, Float4{ 0.0f, 0.0f, 0.0f, NOISE_ABS_MEAN })));
    }
    if (p_mod.features & MOD_SQ) {
        shaped = dadd(shaped, dmul(p_mod.amp[2], dsub(n2, Float4{ 0.0f, 0.0f, 0.0f, NOISE_SQ_MEAN })));
    }
    r_o[VALUE] = r_o[VALUE] + shaped.w;
    r_o[DX] = r_o[DX] + shaped.x;
    r_o[DY] = r_o[DY] + shaped.y;
    r_o[DZ] = r_o[DZ] + shaped.z;
    if ((p_mod.features & MOD_STATE) && r_state) {
        Float4 s = p_mod.state[0];
        s = dadd(s, dmul(p_mod.state[1], nd));
        s = dadd(s, dmul(p_mod.state[2], an));
        s = dadd(s, dmul(p_mod.state[3], n2));
        *r_state = s;
    }
}

// The band of Λ_m of each of the first p_groups groups (at most one: one octave per group and lattice, Climate::validate
// refuses a second), its constant factors and the modulated groups: set up once per level.
struct LevelBands {
    int groups = 0;
    const Band *band[MAX_GROUPS] = {}; // null: the group inserts nothing on Λ_m
    BandFactors factors[MAX_GROUPS];
    uint32_t modulated = 0;
    float slope = 1.0f; // 2^-shift, a modulated band's slope factor
    bool any = false;
    bool carried = false; // the run's last level, its carry carried: the modulation reads it (Modulation::modulate)

    LevelBands(const Spec &p_spec, int p_m, int p_groups, bool p_carried = false) :
            groups(p_groups), slope(ExactNoise::exact_pow2(-p_spec.insert_shift)), carried(p_carried) {
        for (int g = 0; g < groups; g++) {
            for (const Band &b : p_spec.group_bands(g)) {
                if (p_spec.insert_level(b) != p_m) {
                    continue;
                }
                OV_ERR_CONTINUE_MSG(band[g] != nullptr, ov::format("Cascade: two bands of one group on lattice %d.", p_m));
                band[g] = &b;
                factors[g] = BandFactors(p_spec, b);
                any = true;
            }
        }
        // A level inserting fewer groups than the spec has no modulated band among the others (FieldRange::carried and
        // inserted cover them).
        modulated = p_spec.modulation ? p_spec.modulation->modulated(p_m) & ((1u << groups) - 1u) : 0u;
    }
};

// Adds each of p_lb's constant-amplitude bands to a level's channels, band by band (groups' slots are disjoint: the same
// ops as point by point). p_at(slot, i) is the channel at point i.
template <typename At>
void insert_constant_bands(const Spec &p_spec, const LevelBands &p_lb, const Box &p_box, At p_at) {
    for (int g = 0; g < p_lb.groups; g++) {
        if (p_lb.band[g] == nullptr) {
            continue;
        }
        const ChannelSet set = p_spec.group_set(g);
        const int base = group_slot(g);
        for_each_point(p_box, [&](const Vector3i64 &p_p, int64_t p_i) {
            float o[CHANNELS];
            band_channels(p_spec, g, *p_lb.band[g], p_lb.factors[g], p_p, o);
            for (int c = 0; c < channel_count(set); c++) {
                float &v = p_at(base + c, p_i);
                v = v + o[c];
            }
        });
    }
}

// Adds Λ_m's bands at lattice point p_p to its slots r_ch (before: H(V_{m+1}), what the modulation reads).
void insert_point(const Spec &p_spec, const LevelBands &p_lb, int p_m, const Vector3i64 &p_p, float r_ch[SLOTS]) {
    BandMod mod[MAX_GROUPS];
    if (p_lb.modulated != 0u) {
        p_spec.modulation->modulate(p_m, p_lb.carried, r_ch, mod);
    }
    // Every band's channels from the channels before the level's insertions, then the adds.
    float o[MAX_GROUPS][CHANNELS];
    ExactNoise::Float4 state{ 0.0f, 0.0f, 0.0f, 0.0f };
    for (int g = 0; g < p_lb.groups; g++) {
        if (p_lb.band[g] == nullptr) {
            continue;
        }
        if ((p_lb.modulated >> g) & 1u) {
            modulated_channels(p_spec, *p_lb.band[g], mod[g], p_lb.slope, p_p, o[g], g == 0 ? &state : nullptr);
        } else {
            band_channels(p_spec, g, *p_lb.band[g], p_lb.factors[g], p_p, o[g]);
        }
    }
    for (int g = 0; g < p_lb.groups; g++) {
        if (p_lb.band[g] == nullptr) {
            continue;
        }
        const int base = group_slot(g);
        for (int c = 0; c < channel_count(p_spec.group_set(g)); c++) {
            r_ch[base + c] = r_ch[base + c] + o[g][c];
        }
    }
    // The state group is overwritten, not added to (a band-less group: nothing else writes it).
    const int sg = p_spec.modulation ? p_spec.modulation->state_group() : -1;
    if ((p_lb.modulated & 1u) && (mod[0].features & MOD_STATE) && sg > 0 && sg < p_lb.groups) {
        const int base = group_slot(sg);
        r_ch[base + VALUE] = state.w;
        r_ch[base + DX] = state.x;
        r_ch[base + DY] = state.y;
        r_ch[base + DZ] = state.z;
    }
}

// Calls p_f(slot) for the slots a level of ground set p_set, p_fields field groups and p_carry carry groups carries.
template <typename F>
void for_each_slot(ChannelSet p_set, int p_fields, int p_carry, F p_f) {
    for (int s = 0; s < channel_count(p_set); s++) {
        p_f(s);
    }
    for (int s = CHANNELS; s < field_slot(p_fields); s++) {
        p_f(s);
    }
    for (int s = carry_slot(0); s < carry_slot(p_carry); s++) {
        p_f(s);
    }
}

// Λ_m's insertions on the points of p_box carrying ground set p_set, p_fields field groups and p_carry carry groups (a
// Level of run() or a point's SmallLevel): the constant bands band by band, or, with a modulated group, point by point
// (insert_point reads the point's slots before the insertions). p_at(slot, i) is the channel at point i.
template <typename At>
void insert_level(const Spec &p_spec, const LevelBands &p_lb, ChannelSet p_set, int p_fields, int p_carry, int p_m, const Box &p_box, At p_at) {
    if (!p_lb.any) {
        return;
    }
    if (p_lb.modulated == 0u) {
        insert_constant_bands(p_spec, p_lb, p_box, p_at);
        return;
    }
    for_each_point(p_box, [&](const Vector3i64 &p_p, int64_t p_i) {
        float ch[SLOTS] = {};
        for_each_slot(p_set, p_fields, p_carry, [&](int p_s) { ch[p_s] = p_at(p_s, p_i); });
        insert_point(p_spec, p_lb, p_m, p_p, ch);
        for_each_slot(p_set, p_fields, p_carry, [&](int p_s) { p_at(p_s, p_i) = ch[p_s]; });
    });
}

// The derive of Λ_m's points for Λ_{m-1}, the run's last level, from the slots p_at reads before Λ_m's insertions (the
// climate its own bands read): Modulation::derive's carry slots, flushed (the GPU writes them before the level's
// flush), given to p_put(slot, i, value).
template <typename At, typename Put>
void derive_level(const Spec &p_spec, int p_m, ChannelSet p_set, int p_fields, int p_carry, const Box &p_box, At p_at, Put p_put) {
    for_each_point(p_box, [&](const Vector3i64 &p_p, int64_t p_i) {
        float ch[SLOTS] = {};
        for_each_slot(p_set, p_fields, 0, [&](int p_s) { ch[p_s] = p_at(p_s, p_i); });
        p_spec.modulation->derive(p_m, p_p, ch);
        for (int s = carry_slot(0); s < carry_slot(p_carry); s++) {
            p_put(s, p_i, flush_value(ch[s]));
        }
    });
}

// The field groups a run down to p_lod carries at level p_m (FieldRange, or all of them).
int carried_fields(const Spec &p_spec, const FieldRange &p_range, int p_lod, int p_m, bool p_all) {
    return p_all || p_range.carried(p_lod, p_m) ? int(p_spec.fields.size()) : 0;
}

// Of those, the field groups whose bands level p_m inserts (FieldRange::inserted, or all of them).
int inserted_fields(const Spec &p_spec, const FieldRange &p_range, int p_lod, int p_m, bool p_all) {
    return p_all || p_range.inserted(p_lod, p_m) ? int(p_spec.fields.size()) : 0;
}

// The carry groups of a run down to p_lod: derived on Λ_{lod+1} (from its channels before its insertions, added after) and carried to
// Λ_lod by H, where its last level reads them (FieldRange::reads_carry; always with p_all: a pick query).
int run_carry(const Spec &p_spec, const FieldRange &p_range, int p_lod, bool p_all) {
    if (!p_spec.modulation) {
        return 0;
    }
    // Without the fields the last level reads only the first carry_used (the fields' parts: fields_at's).
    return p_all ? p_spec.carry_groups() : (p_range.reads_carry(p_lod) ? p_spec.modulation->carry_used(p_lod) : 0);
}

} // namespace

Box parent_box(const Box &p_child) {
    Box b;
    for (int a = 0; a < 3; a++) {
        b.lo[a] = half_floor(p_child.lo[a]);
        b.hi[a] = half_floor(p_child.hi[a] + 1);
    }
    return b;
}

std::vector<Box> level_boxes(const Box &p_box, int p_lod, int p_top) {
    std::vector<Box> boxes = { p_box };
    for (int m = p_lod + 1; m <= p_top; m++) {
        boxes.push_back(parent_box(boxes.back()));
    }
    return boxes;
}

Box level_box(const Box &p_box, int p_j) {
    Box b;
    const uint64_t mask = (uint64_t(1) << p_j) - 1;
    for (int a = 0; a < 3; a++) {
        b.lo[a] = p_box.lo[a] >> p_j;
        b.hi[a] = (p_box.hi[a] >> p_j) + ((uint64_t(p_box.hi[a]) & mask) != 0 ? 1 : 0);
    }
    return b;
}

int level_bound(int p_size, int p_j) {
    int n = p_size;
    for (int j = 0; j < p_j; j++) {
        n = (n + 1) / 2 + 1;
    }
    return n;
}

Box chunk_box(const ChunkId &p_chunk, const ChunkParam &p_param) {
    // The anchor is a multiple of 28 * 2^L: the shift is exact.
    return item_box(ChunkMath::chunk_min_voxel(p_chunk, p_param), ChunkMath::get_lod(p_chunk), p_param);
}

Box item_box(const VoxelPos &p_anchor, int p_lod, const ChunkParam &p_param) {
    const int64_t a[3] = { p_anchor.x >> p_lod, p_anchor.y >> p_lod, p_anchor.z >> p_lod };
    Box b;
    for (int i = 0; i < 3; i++) {
        b.lo[i] = a[i] - p_param.skirt_size;
        b.hi[i] = b.lo[i] + p_param.chunk_size - 1;
    }
    return b;
}

void Level::add_carry(int p_carry) {
    for (int s = carry_slot(carry); s < carry_slot(p_carry); s++) {
        ch[s].assign(size_t(box.count()), 0.0f);
    }
    carry = ov::max(carry, p_carry);
}

void Level::allocate(int p_m, const Box &p_box, ChannelSet p_set, int p_fields, int p_carry) {
    m = p_m;
    box = p_box;
    set = p_set;
    fields = p_fields;
    carry = p_carry;
    for (std::vector<float> &c : ch) {
        c.clear();
    }
    for_each_slot(p_set, p_fields, p_carry, [&](int p_s) { ch[p_s].assign(size_t(p_box.count()), 0.0f); });
}

void Level::channels_at(const Vector3i64 &p_p, float r_ch[CHANNELS]) const {
    const int64_t i = box.index(p_p);
    for (int c = 0; c < CHANNELS; c++) {
        r_ch[c] = c < channel_count(set) ? ch[c][i] : 0.0f;
    }
}

bool Spec::stretched(int p_group) const {
    for (int g = 0; g < group_count(); g++) {
        if (p_group >= 0 && g != p_group) {
            continue;
        }
        for (const Band &b : group_bands(g)) {
            if (b.stretch != 0) {
                return true;
            }
        }
    }
    return false;
}

int Spec::top_level(bool p_all_fields) const {
    if (top >= 0) {
        return top;
    }
    const bool fields = p_all_fields || !field_range().empty();
    int t = -1;
    for (int g = 0; g < (fields ? group_count() : 1); g++) {
        for (const Band &b : group_bands(g)) {
            t = std::max(t, insert_level(b));
        }
    }
    return t;
}

int Spec::run_top(int p_lod, bool p_all_fields) const {
    // A last level reading the carry has a level above it that derives it (empty above the highest band).
    const bool carry = carry_groups() > 0 && (p_all_fields || field_range().reads_carry(p_lod));
    return std::max(top_level(p_all_fields), p_lod + (carry ? 1 : 0));
}

int Spec::gpu_carry_groups() const {
    int n = 0;
    for (int m = 0; modulation && m <= MAX_LEVEL; m++) {
        n = std::max(n, modulation->carry_used(m));
    }
    return n;
}

int64_t Stats::total() const {
    int64_t t = 0;
    for (const int64_t p : points) {
        t += p;
    }
    return t;
}

namespace {

// The top lattice of a run down to p_lod, or -1 (with an error) if the levels are out of range, the spec's explicit
// top would drop bands, or the field groups / modulation are malformed. p_all_fields: the fields carried everywhere.
int checked_top(const Spec &p_spec, int p_lod, bool p_all_fields = false) {
    const FieldRange range = p_spec.field_range();
    const int top = p_spec.run_top(p_lod, p_all_fields);
    OV_ERR_FAIL_COND_V_MSG(p_lod < 0 || top > MAX_LEVEL, -1, "Cascade: lattice level out of range.");
    OV_ERR_FAIL_COND_V_MSG(int(p_spec.fields.size()) > MAX_FIELD_GROUPS, -1, "Cascade: too many field groups.");
    OV_ERR_FAIL_COND_V_MSG(range.floor <= range.ceiling && p_spec.set != ChannelSet::LINEAR4, -1, "Cascade: modulation needs a LINEAR4 ground.");
    OV_ERR_FAIL_COND_V_MSG(p_spec.carry_groups() > MAX_CARRY, -1, "Cascade: too many carry groups.");
    const bool fields = p_all_fields || !range.empty();
    for (int g = 0; g < (fields ? p_spec.group_count() : 1); g++) {
        for (const Band &b : p_spec.group_bands(g)) {
            OV_ERR_FAIL_COND_V_MSG(p_spec.insert_level(b) > top, -1, ov::format("Cascade: band k=%d is inserted above the top lattice %d.", b.k, top));
        }
    }
    return top;
}

} // namespace

void count_points(const Spec &p_spec, const Box &p_box, int p_lod, Stats &r_stats) {
    const int top = checked_top(p_spec, p_lod);
    if (top < 0) {
        return;
    }
    const std::vector<Box> boxes = level_boxes(p_box, p_lod, top);
    for (size_t i = 0; i < boxes.size(); i++) {
        r_stats.points[p_lod + int(i)] += boxes[i].count();
    }
}

void band_channels(const Spec &p_spec, const Band &p_band, const Vector3i64 &p_p, float r_ch[CHANNELS]) {
    band_channels(p_spec, 0, p_band, BandFactors(p_spec, p_band), p_p, r_ch);
}

void insert_bands(const Spec &p_spec, Level &r_level, int p_fields) {
    const LevelBands lb(p_spec, r_level.m, 1 + (p_fields < 0 ? r_level.fields : p_fields), r_level.carry > 0);
    insert_level(p_spec, lb, r_level.set, r_level.fields, r_level.carry, r_level.m, r_level.box,
            [&](int p_s, int64_t p_i) -> float & { return r_level.ch[p_s][p_i]; });
}

void insert_at(const Spec &p_spec, int p_m, const Vector3i64 &p_p, float r_slots[SLOTS]) {
    insert_point(p_spec, LevelBands(p_spec, p_m, p_spec.group_count()), p_m, p_p, r_slots);
}

Level subdivide(const Level &p_parent, const Box &p_child_box, int p_fields, int p_carry) {
    const int fields = p_fields < 0 ? p_parent.fields : p_fields;
    const int carry = p_carry < 0 ? p_parent.carry : p_carry;
    OV_ERR_FAIL_COND_V_MSG(fields > p_parent.fields || carry > p_parent.carry, Level(), "Cascade::subdivide: more groups than the parent carries.");
    Level l = subdivide_axis(p_parent, fields, carry, 0, p_child_box.lo[0], p_child_box.hi[0]);
    l = subdivide_axis(l, fields, carry, 1, p_child_box.lo[1], p_child_box.hi[1]);
    l = subdivide_axis(l, fields, carry, 2, p_child_box.lo[2], p_child_box.hi[2]);
    l.m = p_parent.m - 1;
    return l;
}

void flush(Level &r_level) {
    for (std::vector<float> &c : r_level.ch) {
        for (float &v : c) {
            v = flush_value(v);
        }
    }
}

Level run(const Spec &p_spec, const Box &p_box, int p_lod, std::vector<Level> *r_levels, const Source &p_source, bool p_all_fields) {
    const int top = checked_top(p_spec, p_lod, p_all_fields);
    if (top < 0) {
        return Level();
    }
    const FieldRange range = p_spec.field_range();
    const std::vector<Box> boxes = level_boxes(p_box, p_lod, top);
    const int carry = run_carry(p_spec, range, p_lod, p_all_fields);
    Level cur;
    for (int m = top; m >= p_lod; m--) {
        const Box &box = boxes[size_t(m - p_lod)];
        // Monotone in m (FieldRange): a level carries at most its parent's fields; the carry only from Λ_{lod+1} down.
        const int fields = carried_fields(p_spec, range, p_lod, m, p_all_fields);
        if (m == top) {
            cur.allocate(m, box, p_spec.set, fields, m == p_lod ? carry : 0);
            cur.copied = p_spec.copied_carry();
        } else {
            // With r_levels the previous level was moved there.
            cur = subdivide(r_levels ? r_levels->back() : cur, box, fields, m == p_lod ? carry : 0);
        }
        const bool derives = m == p_lod + 1 && carry > 0;
        std::vector<float> derived[SLOTS];
        if (derives) {
            for (int s = carry_slot(0); s < carry_slot(carry); s++) {
                derived[s].resize(size_t(box.count()));
            }
            derive_level(
                    p_spec, m, cur.set, cur.fields, carry, cur.box, [&](int p_s, int64_t p_i) { return cur.ch[p_s][p_i]; },
                    [&](int p_s, int64_t p_i, float p_v) { derived[p_s][p_i] = p_v; });
        }
        if (p_source) {
            p_source(cur);
        } else {
            // The carry: only the last level holds it yet.
            insert_bands(p_spec, cur, inserted_fields(p_spec, range, p_lod, m, p_all_fields));
        }
        flush(cur);
        if (derives) {
            cur.add_carry(carry);
            for (int s = carry_slot(0); s < carry_slot(carry); s++) {
                cur.ch[s] = std::move(derived[s]);
            }
        }
        if (r_levels) {
            r_levels->push_back(std::move(cur));
        }
    }
    if (r_levels) {
        return r_levels->back();
    }
    return cur;
}

void point(const Spec &p_spec, const Vector3i64 &p_p, int p_m, float r_ch[CHANNELS]) {
    run(p_spec, point_box(p_p), p_m).channels_at(p_p, r_ch);
}

namespace {

constexpr int POINT_CAP = POINT_AXIS * POINT_AXIS * POINT_AXIS;

// A Level of at most POINT_AXIS^3 points, on the stack (x-major like Level).
struct SmallLevel {
    Box box;
    float ch[SLOTS][POINT_CAP];

    int64_t stride(int p_axis) const { return p_axis == 0 ? box.size(1) * box.size(2) : (p_axis == 1 ? box.size(2) : 1); }
};

// subdivide_axis on SmallLevels: the same index arithmetic and the same per-point ops (axis_rules), the ground of set
// p_set, p_fields field groups and p_carry carry groups (p_copied copied).
void subdivide_axis_small(const SmallLevel &p_in, ChannelSet p_set, int p_fields, int p_carry, int p_copied, int p_axis, int64_t p_lo, int64_t p_hi,
        SmallLevel &r_out) {
    r_out.box = p_in.box;
    r_out.box.lo[p_axis] = p_lo;
    r_out.box.hi[p_axis] = p_hi;
    const int64_t in_stride[3] = { p_in.stride(0), p_in.stride(1), p_in.stride(2) };
    for_each_point(r_out.box, [&](const Vector3i64 &p_p, int64_t p_o) {
        const int64_t idx[3] = { p_p.x, p_p.y, p_p.z };
        const int64_t c = idx[p_axis];
        int64_t ia = 0;
        for (int a = 0; a < 3; a++) {
            ia += ((a == p_axis ? half_floor(c) : idx[a]) - p_in.box.lo[a]) * in_stride[a];
        }
        const int64_t ib = ia + in_stride[p_axis];
        axis_rules(
                p_set, p_fields, p_carry, p_copied, p_axis, (c & 1) == 0, [&](int p_s, bool p_b) { return p_in.ch[p_s][p_b ? ib : ia]; },
                [&](int p_s, float p_v) { r_out.ch[p_s][p_o] = p_v; });
    });
}

} // namespace

void point_slots(const Spec &p_spec, const Vector3i64 &p_p, int p_m, float r_slots[SLOTS], bool p_all_fields) {
    for (int c = 0; c < SLOTS; c++) {
        r_slots[c] = 0.0f;
    }
    const int top = checked_top(p_spec, p_m, p_all_fields);
    if (top < 0) {
        return;
    }
    const FieldRange range = p_spec.field_range();
    const Box base = point_box(p_p);
    // Ping-pong: cur holds level l, tmp the axis passes towards level l - 1.
    SmallLevel a;
    SmallLevel b;
    SmallLevel *cur = &a;
    SmallLevel *tmp = &b;
    const int copied = p_spec.copied_carry();
    const int run = run_carry(p_spec, range, p_m, p_all_fields);
    int fields = 0;
    int carry = 0;
    for (int l = top; l >= p_m; l--) {
        // level_box == the iterated parent_box of run() (unit-tested); a point's boxes fit POINT_AXIS.
        const Box box = level_box(base, l - p_m);
        OV_ERR_FAIL_COND_MSG(box.size(0) > POINT_AXIS || box.size(1) > POINT_AXIS || box.size(2) > POINT_AXIS, "Cascade::point_into: level box over POINT_AXIS.");
        fields = carried_fields(p_spec, range, p_m, l, p_all_fields);
        carry = l == p_m ? run : 0;
        if (l == top) {
            cur->box = box;
            for_each_slot(p_spec.set, fields, carry, [&](int p_s) {
                for (float &v : cur->ch[p_s]) {
                    v = 0.0f;
                }
            });
        } else {
            // subdivide(): x, then y, then z.
            subdivide_axis_small(*cur, p_spec.set, fields, carry, copied, 0, box.lo[0], box.hi[0], *tmp);
            subdivide_axis_small(*tmp, p_spec.set, fields, carry, copied, 1, box.lo[1], box.hi[1], *cur);
            subdivide_axis_small(*cur, p_spec.set, fields, carry, copied, 2, box.lo[2], box.hi[2], *tmp);
            std::swap(cur, tmp);
        }
        // The derive (above the last level, into the carry slots: the level's insertions and flush leave them), then
        // insert_bands(), flush().
        const bool derives = l == p_m + 1 && run > 0;
        const int64_t count = cur->box.count();
        const auto at = [&](int p_s, int64_t p_i) -> float & { return cur->ch[p_s][p_i]; };
        if (derives) {
            derive_level(p_spec, l, p_spec.set, fields, run, cur->box, at, [&](int p_s, int64_t p_i, float p_v) { cur->ch[p_s][p_i] = p_v; });
        }
        const LevelBands lb(p_spec, l, 1 + inserted_fields(p_spec, range, p_m, l, p_all_fields), carry > 0);
        insert_level(p_spec, lb, p_spec.set, fields, carry, l, cur->box, at);
        for_each_slot(p_spec.set, fields, carry, [&](int p_s) {
            for (int64_t i = 0; i < count; i++) {
                cur->ch[p_s][i] = flush_value(cur->ch[p_s][i]);
            }
        });
        if (derives) {
            carry = run;
        }
    }
    for_each_slot(p_spec.set, fields, carry, [&](int p_s) { r_slots[p_s] = cur->ch[p_s][0]; });
}

void point_into(const Spec &p_spec, const Vector3i64 &p_p, int p_m, float r_ch[CHANNELS]) {
    float slots[SLOTS];
    point_slots(p_spec, p_p, p_m, slots);
    const int n = channel_count(p_spec.set);
    for (int c = 0; c < CHANNELS; c++) {
        r_ch[c] = c < n ? slots[c] : 0.0f;
    }
}

} // namespace Cascade
