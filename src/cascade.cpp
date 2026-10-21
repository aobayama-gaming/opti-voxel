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

// One axis of H: p_in -> the same box with axis p_axis replaced by [p_lo, p_hi] (child indices), the ground and
// p_fields field groups (<= p_in's).
Level subdivide_axis(const Level &p_in, int p_fields, int p_axis, int64_t p_lo, int64_t p_hi) {
    Box box = p_in.box;
    box.lo[p_axis] = p_lo;
    box.hi[p_axis] = p_hi;
    Level out;
    out.allocate(p_in.m, box, p_in.set, p_fields);
    const int64_t in_stride[3] = { p_in.box.size(1) * p_in.box.size(2), p_in.box.size(2), 1 };
    for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_o) {
        const int64_t idx[3] = { p_p.x, p_p.y, p_p.z };
        const int64_t c = idx[p_axis];
        int64_t ia = 0;
        for (int a = 0; a < 3; a++) {
            ia += ((a == p_axis ? half_floor(c) : idx[a]) - p_in.box.lo[a]) * in_stride[a];
        }
        const int64_t ib = ia + in_stride[p_axis];
        for (int g = 0; g < 1 + p_fields; g++) {
            axis_rule(
                    group_set(p_in.set, g), group_slot(g), p_axis, (c & 1) == 0, [&](int p_s, bool p_b) { return p_in.ch[p_s][p_b ? ib : ia]; },
                    [&](int p_s, float p_v) { out.ch[p_s][p_o] = p_v; });
        }
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

void band_channels(const Spec &p_spec, ChannelSet p_set, const Band &p_band, const BandFactors &p_factors, const Vector3i64 &p_p, float r_ch[CHANNELS]) {
    const WorldPos pos = band_pos(p_spec, p_band, p_p);
    const float *factor = p_factors.f;
    if (p_set == ChannelSet::HERMITE8) {
        const ExactNoise::Channels8 n = ExactNoise::perlin3_cross(pos, p_band.k, p_band.seed, p_band.offset);
        for (int c = 0; c < CHANNELS; c++) {
            r_ch[c] = n.c[c] * factor[channel_order(c)];
        }
        return;
    }
    // The same bits as perlin3_cross's first four channels, without the cross derivatives.
    const ExactNoise::Float4 n = ExactNoise::perlin3(pos, p_band.k, p_band.seed, p_band.offset);
    r_ch[VALUE] = n.w * factor[0];
    r_ch[DX] = n.x * factor[1];
    r_ch[DY] = n.y * factor[1];
    r_ch[DZ] = n.z * factor[1];
}

// A modulated band's 4 channels (amplitude p_amp, Modulation): value n.w A, slope n.s (A p_slope) + n.w dA, p_slope =
// 2^-shift, in that order (the GPU's insert_band_mod).
void modulated_channels(const Spec &p_spec, const Band &p_band, const ExactNoise::Float4 &p_amp, float p_slope, const Vector3i64 &p_p, float r_o[4]) {
    const ExactNoise::Float4 n = ExactNoise::perlin3(band_pos(p_spec, p_band, p_p), p_band.k, p_band.seed, p_band.offset);
    const float aws = p_amp.w * p_slope;
    r_o[VALUE] = n.w * p_amp.w;
    r_o[DX] = n.x * aws + n.w * p_amp.x;
    r_o[DY] = n.y * aws + n.w * p_amp.y;
    r_o[DZ] = n.z * aws + n.w * p_amp.z;
}

// The bands of Λ_m of the first p_groups groups (in list order), their constant factors and the modulated groups: set
// up once per level.
struct LevelBands {
    int groups = 0;
    const Band *band[MAX_GROUPS][MAX_LEVEL_BANDS] = {};
    BandFactors factors[MAX_GROUPS][MAX_LEVEL_BANDS];
    int count[MAX_GROUPS] = {};
    uint32_t modulated = 0;
    float slope = 1.0f; // 2^-shift, a modulated band's slope factor
    bool any = false;

    LevelBands(const Spec &p_spec, int p_m, int p_groups) :
            groups(p_groups), slope(ExactNoise::exact_pow2(-p_spec.insert_shift)) {
        for (int g = 0; g < groups; g++) {
            for (const Band &b : p_spec.group_bands(g)) {
                if (p_spec.insert_level(b) != p_m) {
                    continue;
                }
                ERR_CONTINUE_MSG(count[g] == MAX_LEVEL_BANDS, vformat("Cascade: more than %d bands of one group on lattice %d.", MAX_LEVEL_BANDS, p_m));
                band[g][count[g]] = &b;
                factors[g][count[g]] = BandFactors(p_spec, b);
                count[g]++;
                any = true;
            }
        }
        // A level carrying fewer groups than the spec has no modulated band (FieldRange::carried covers them).
        modulated = p_spec.modulation ? p_spec.modulation->modulated(p_m) & ((1u << groups) - 1u) : 0u;
    }
};

// Adds each of p_lb's constant-amplitude bands to a level's channels, band by band (groups' slots are disjoint, a
// group's bands add up in list order: the same ops as point by point). p_at(slot, i) is the channel at point i.
template <typename At>
void insert_constant_bands(const Spec &p_spec, const LevelBands &p_lb, const Box &p_box, At p_at) {
    for (int g = 0; g < p_lb.groups; g++) {
        const ChannelSet set = p_spec.group_set(g);
        const int base = group_slot(g);
        for (int i = 0; i < p_lb.count[g]; i++) {
            for_each_point(p_box, [&](const Vector3i64 &p_p, int64_t p_i) {
                float o[CHANNELS];
                band_channels(p_spec, set, *p_lb.band[g][i], p_lb.factors[g][i], p_p, o);
                for (int c = 0; c < channel_count(set); c++) {
                    float &v = p_at(base + c, p_i);
                    v = v + o[c];
                }
            });
        }
    }
}

// Adds Λ_m's bands at lattice point p_p to its slots r_ch (before: H(V_{m+1}), what the modulation reads).
void insert_point(const Spec &p_spec, const LevelBands &p_lb, int p_m, const Vector3i64 &p_p, float r_ch[SLOTS]) {
    ExactNoise::Float4 amp[MAX_GROUPS];
    if (p_lb.modulated != 0u) {
        p_spec.modulation->amplitudes(p_m, r_ch, amp);
    }
    // Every band's channels from the channels before the level's insertions, then the adds (a modulated group has one).
    float o[MAX_GROUPS][MAX_LEVEL_BANDS][CHANNELS];
    for (int g = 0; g < p_lb.groups; g++) {
        for (int i = 0; i < p_lb.count[g]; i++) {
            if ((p_lb.modulated >> g) & 1u) {
                modulated_channels(p_spec, *p_lb.band[g][i], amp[g], p_lb.slope, p_p, o[g][i]);
            } else {
                band_channels(p_spec, p_spec.group_set(g), *p_lb.band[g][i], p_lb.factors[g][i], p_p, o[g][i]);
            }
        }
    }
    for (int g = 0; g < p_lb.groups; g++) {
        const int base = group_slot(g);
        for (int i = 0; i < p_lb.count[g]; i++) {
            for (int c = 0; c < channel_count(p_spec.group_set(g)); c++) {
                r_ch[base + c] = r_ch[base + c] + o[g][i][c];
            }
        }
    }
}

// Calls p_f(slot) for the slots a level of ground set p_set and p_fields field groups carries.
template <typename F>
void for_each_slot(ChannelSet p_set, int p_fields, F p_f) {
    for (int s = 0; s < channel_count(p_set); s++) {
        p_f(s);
    }
    for (int s = CHANNELS; s < field_slot(p_fields); s++) {
        p_f(s);
    }
}

// The field groups a run down to p_lod carries at level p_m (FieldRange, or all of them).
int carried_fields(const Spec &p_spec, const FieldRange &p_range, int p_lod, int p_m, bool p_all) {
    return p_all || p_range.carried(p_lod, p_m) ? int(p_spec.fields.size()) : 0;
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
    const int lod = ChunkMath::get_lod(p_chunk);
    const VoxelPos anchor = ChunkMath::chunk_min_voxel(p_chunk, p_param);
    // The anchor is a multiple of 28 * 2^L: the division is exact.
    const int64_t step = int64_t(1) << lod;
    const int64_t a[3] = { anchor.x / step, anchor.y / step, anchor.z / step };
    Box b;
    for (int i = 0; i < 3; i++) {
        b.lo[i] = a[i] - p_param.skirt_size;
        b.hi[i] = b.lo[i] + p_param.chunk_size - 1;
    }
    return b;
}

void Level::allocate(int p_m, const Box &p_box, ChannelSet p_set, int p_fields) {
    m = p_m;
    box = p_box;
    set = p_set;
    fields = p_fields;
    for (std::vector<float> &c : ch) {
        c.clear();
    }
    for_each_slot(p_set, p_fields, [&](int p_s) { ch[p_s].assign(size_t(p_box.count()), 0.0f); });
}

void Level::channels_at(const Vector3i64 &p_p, float r_ch[CHANNELS]) const {
    const int64_t i = box.index(p_p);
    for (int c = 0; c < CHANNELS; c++) {
        r_ch[c] = c < channel_count(set) ? ch[c][i] : 0.0f;
    }
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
    const int top = std::max(p_spec.top_level(p_all_fields), p_lod);
    ERR_FAIL_COND_V_MSG(p_lod < 0 || top > MAX_LEVEL, -1, "Cascade: lattice level out of range.");
    ERR_FAIL_COND_V_MSG(int(p_spec.fields.size()) > MAX_FIELD_GROUPS, -1, "Cascade: too many field groups.");
    ERR_FAIL_COND_V_MSG(p_spec.modulation && p_spec.set != ChannelSet::LINEAR4, -1, "Cascade: modulation needs a LINEAR4 ground.");
    const bool fields = p_all_fields || !p_spec.field_range().empty();
    for (int g = 0; g < (fields ? p_spec.group_count() : 1); g++) {
        for (const Band &b : p_spec.group_bands(g)) {
            ERR_FAIL_COND_V_MSG(p_spec.insert_level(b) > top, -1, vformat("Cascade: band k=%d is inserted above the top lattice %d.", b.k, top));
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
    band_channels(p_spec, p_spec.set, p_band, BandFactors(p_spec, p_band), p_p, r_ch);
}

void insert_bands(const Spec &p_spec, Level &r_level) {
    const LevelBands lb(p_spec, r_level.m, 1 + r_level.fields);
    if (!lb.any) {
        return;
    }
    if (lb.modulated == 0u) {
        insert_constant_bands(p_spec, lb, r_level.box, [&](int p_s, int64_t p_i) -> float & { return r_level.ch[p_s][p_i]; });
        return;
    }
    for_each_point(r_level.box, [&](const Vector3i64 &p_p, int64_t p_i) {
        float ch[SLOTS] = {};
        for_each_slot(r_level.set, r_level.fields, [&](int p_s) { ch[p_s] = r_level.ch[p_s][p_i]; });
        insert_point(p_spec, lb, r_level.m, p_p, ch);
        for_each_slot(r_level.set, r_level.fields, [&](int p_s) { r_level.ch[p_s][p_i] = ch[p_s]; });
    });
}

void insert_at(const Spec &p_spec, int p_m, const Vector3i64 &p_p, float r_slots[SLOTS]) {
    insert_point(p_spec, LevelBands(p_spec, p_m, p_spec.group_count()), p_m, p_p, r_slots);
}

Level subdivide(const Level &p_parent, const Box &p_child_box, int p_fields) {
    const int fields = p_fields < 0 ? p_parent.fields : p_fields;
    ERR_FAIL_COND_V_MSG(fields > p_parent.fields, Level(), "Cascade::subdivide: more field groups than the parent carries.");
    Level l = subdivide_axis(p_parent, fields, 0, p_child_box.lo[0], p_child_box.hi[0]);
    l = subdivide_axis(l, fields, 1, p_child_box.lo[1], p_child_box.hi[1]);
    l = subdivide_axis(l, fields, 2, p_child_box.lo[2], p_child_box.hi[2]);
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
    Level cur;
    for (int m = top; m >= p_lod; m--) {
        const Box &box = boxes[size_t(m - p_lod)];
        // Monotone in m (FieldRange::carried): a level carries at most its parent's fields.
        const int fields = carried_fields(p_spec, range, p_lod, m, p_all_fields);
        if (m == top) {
            cur.allocate(m, box, p_spec.set, fields);
        } else {
            // With r_levels the previous level was moved there.
            cur = subdivide(r_levels ? r_levels->back() : cur, box, fields);
        }
        if (p_source) {
            p_source(cur);
        } else {
            insert_bands(p_spec, cur);
        }
        flush(cur);
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

// subdivide_axis on SmallLevels: the same index arithmetic and the same per-point ops (axis_rule), the ground of set
// p_set and p_fields field groups.
void subdivide_axis_small(const SmallLevel &p_in, ChannelSet p_set, int p_fields, int p_axis, int64_t p_lo, int64_t p_hi, SmallLevel &r_out) {
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
        for (int g = 0; g < 1 + p_fields; g++) {
            axis_rule(
                    group_set(p_set, g), group_slot(g), p_axis, (c & 1) == 0, [&](int p_s, bool p_b) { return p_in.ch[p_s][p_b ? ib : ia]; },
                    [&](int p_s, float p_v) { r_out.ch[p_s][p_o] = p_v; });
        }
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
    int fields = 0;
    for (int l = top; l >= p_m; l--) {
        // level_box == the iterated parent_box of run() (unit-tested); a point's boxes fit POINT_AXIS.
        const Box box = level_box(base, l - p_m);
        ERR_FAIL_COND_MSG(box.size(0) > POINT_AXIS || box.size(1) > POINT_AXIS || box.size(2) > POINT_AXIS, "Cascade::point_into: level box over POINT_AXIS.");
        fields = carried_fields(p_spec, range, p_m, l, p_all_fields);
        if (l == top) {
            cur->box = box;
            for_each_slot(p_spec.set, fields, [&](int p_s) {
                for (float &v : cur->ch[p_s]) {
                    v = 0.0f;
                }
            });
        } else {
            // subdivide(): x, then y, then z.
            subdivide_axis_small(*cur, p_spec.set, fields, 0, box.lo[0], box.hi[0], *tmp);
            subdivide_axis_small(*tmp, p_spec.set, fields, 1, box.lo[1], box.hi[1], *cur);
            subdivide_axis_small(*cur, p_spec.set, fields, 2, box.lo[2], box.hi[2], *tmp);
            std::swap(cur, tmp);
        }
        // insert_bands(), then flush().
        const LevelBands lb(p_spec, l, 1 + fields);
        const int64_t count = cur->box.count();
        if (lb.any && lb.modulated == 0u) {
            insert_constant_bands(p_spec, lb, cur->box, [&](int p_s, int64_t p_i) -> float & { return cur->ch[p_s][p_i]; });
        } else if (lb.any) {
            for_each_point(cur->box, [&](const Vector3i64 &p_q, int64_t p_i) {
                float ch[SLOTS] = {};
                for_each_slot(p_spec.set, fields, [&](int p_s) { ch[p_s] = cur->ch[p_s][p_i]; });
                insert_point(p_spec, lb, l, p_q, ch);
                for_each_slot(p_spec.set, fields, [&](int p_s) { cur->ch[p_s][p_i] = ch[p_s]; });
            });
        }
        for_each_slot(p_spec.set, fields, [&](int p_s) {
            for (int64_t i = 0; i < count; i++) {
                cur->ch[p_s][i] = flush_value(cur->ch[p_s][i]);
            }
        });
    }
    for_each_slot(p_spec.set, fields, [&](int p_s) { r_slots[p_s] = cur->ch[p_s][0]; });
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
