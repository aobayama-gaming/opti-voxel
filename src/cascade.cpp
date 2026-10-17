#include "cascade.hpp"

#include <algorithm>

namespace Cascade {

static_assert(sizeof(ExactNoise::Channels8::c) / sizeof(float) == CHANNELS, "perlin3_cross returns the cascade's channels");

namespace {

// floor(p_v / 2), for negative indices too (inner loop: a shift, not WorldCoord::floor_div).
int64_t half_floor(int64_t p_v) {
    return p_v >= 0 ? (p_v >> 1) : ~((~p_v) >> 1);
}

// Per axis pass of HERMITE8: the (value-like, slope-like) channel pairs, slope = value + the axis.
constexpr int PAIRS[3][4][2] = {
    { { VALUE, DX }, { DY, DXY }, { DZ, DXZ }, { DYZ, DXYZ } },
    { { VALUE, DY }, { DX, DXY }, { DZ, DYZ }, { DXZ, DXYZ } },
    { { VALUE, DZ }, { DX, DXZ }, { DY, DYZ }, { DXY, DXYZ } },
};

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

// One axis of H: p_in -> the same box with axis p_axis replaced by [p_lo, p_hi] (child indices).
Level subdivide_axis(const Level &p_in, int p_axis, int64_t p_lo, int64_t p_hi) {
    Box box = p_in.box;
    box.lo[p_axis] = p_lo;
    box.hi[p_axis] = p_hi;
    Level out;
    out.allocate(p_in.m, box, p_in.set);
    const int64_t in_stride[3] = { p_in.box.size(1) * p_in.box.size(2), p_in.box.size(2), 1 };
    const int axis_slope = DX + p_axis;
    for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_o) {
        const int64_t idx[3] = { p_p.x, p_p.y, p_p.z };
        const int64_t c = idx[p_axis];
        int64_t ia = 0;
        for (int a = 0; a < 3; a++) {
            ia += ((a == p_axis ? half_floor(c) : idx[a]) - p_in.box.lo[a]) * in_stride[a];
        }
        const int64_t ib = ia + in_stride[p_axis];
        const bool copy = (c & 1) == 0;
        auto pair = [&](int p_v, int p_s) {
            if (copy) {
                hermite_copy(p_in.ch[p_v][ia], p_in.ch[p_s][ia], out.ch[p_v][p_o], out.ch[p_s][p_o]);
            } else {
                hermite_mid(p_in.ch[p_v][ia], p_in.ch[p_v][ib], p_in.ch[p_s][ia], p_in.ch[p_s][ib], out.ch[p_v][p_o], out.ch[p_s][p_o]);
            }
        };
        if (p_in.set == ChannelSet::HERMITE8) {
            for (const auto &pr : PAIRS[p_axis]) {
                pair(pr[0], pr[1]);
            }
            return;
        }
        // LINEAR4: the transverse slopes are interpolated linearly.
        pair(VALUE, axis_slope);
        for (int s = DX; s <= DZ; s++) {
            if (s != axis_slope) {
                out.ch[s][p_o] = copy ? p_in.ch[s][ia] : (p_in.ch[s][ia] + p_in.ch[s][ib]) * 0.5f;
            }
        }
    });
    return out;
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

void Level::allocate(int p_m, const Box &p_box, ChannelSet p_set) {
    m = p_m;
    box = p_box;
    set = p_set;
    const int n = channel_count(p_set);
    for (int c = 0; c < CHANNELS; c++) {
        ch[c].assign(c < n ? size_t(p_box.count()) : 0, 0.0f);
    }
}

void Level::channels_at(const Vector3i64 &p_p, float r_ch[CHANNELS]) const {
    const int64_t i = box.index(p_p);
    for (int c = 0; c < CHANNELS; c++) {
        r_ch[c] = c < channel_count(set) ? ch[c][i] : 0.0f;
    }
}

int Spec::top_level() const {
    if (top >= 0) {
        return top;
    }
    int t = -1;
    for (const Band &b : bands) {
        t = std::max(t, insert_level(b));
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

// The top lattice of a run down to p_lod, or -1 (with an error) if the levels are out of range or the
// spec's explicit top would drop bands.
int checked_top(const Spec &p_spec, int p_lod) {
    const int top = std::max(p_spec.top_level(), p_lod);
    ERR_FAIL_COND_V_MSG(p_lod < 0 || top > MAX_LEVEL, -1, "Cascade: lattice level out of range.");
    for (const Band &b : p_spec.bands) {
        ERR_FAIL_COND_V_MSG(p_spec.insert_level(b) > top, -1, vformat("Cascade: band k=%d is inserted above the top lattice %d.", b.k, top));
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

namespace {

// Channel scale of a band per derivative order: amp * 2^(-shift |S|), exact.
struct BandFactors {
    float f[4];
    BandFactors(const Spec &p_spec, const Band &p_band) {
        for (int o = 0; o < 4; o++) {
            f[o] = p_band.amp * ExactNoise::exact_pow2(-p_spec.insert_shift * o);
        }
    }
};

void band_channels(const Spec &p_spec, const Band &p_band, const BandFactors &p_factors, const Vector3i64 &p_p, float r_ch[CHANNELS]) {
    const int64_t step = int64_t(1) << p_spec.insert_level(p_band);
    WorldPos pos;
    pos.voxel = Vector3i64(p_p.x * step, p_p.y * step, p_p.z * step);
    const float *factor = p_factors.f;
    if (p_spec.set == ChannelSet::HERMITE8) {
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

} // namespace

void band_channels(const Spec &p_spec, const Band &p_band, const Vector3i64 &p_p, float r_ch[CHANNELS]) {
    band_channels(p_spec, p_band, BandFactors(p_spec, p_band), p_p, r_ch);
}

void insert_bands(const Spec &p_spec, Level &r_level) {
    const int n = channel_count(r_level.set);
    for (const Band &band : p_spec.bands) {
        if (p_spec.insert_level(band) != r_level.m) {
            continue;
        }
        const BandFactors factors(p_spec, band);
        for_each_point(r_level.box, [&](const Vector3i64 &p_p, int64_t p_i) {
            float o[CHANNELS];
            band_channels(p_spec, band, factors, p_p, o);
            for (int c = 0; c < n; c++) {
                r_level.ch[c][p_i] = r_level.ch[c][p_i] + o[c];
            }
        });
    }
}

Level subdivide(const Level &p_parent, const Box &p_child_box) {
    Level l = subdivide_axis(p_parent, 0, p_child_box.lo[0], p_child_box.hi[0]);
    l = subdivide_axis(l, 1, p_child_box.lo[1], p_child_box.hi[1]);
    l = subdivide_axis(l, 2, p_child_box.lo[2], p_child_box.hi[2]);
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

Level run(const Spec &p_spec, const Box &p_box, int p_lod, std::vector<Level> *r_levels, const Source &p_source) {
    const int top = checked_top(p_spec, p_lod);
    if (top < 0) {
        return Level();
    }
    const std::vector<Box> boxes = level_boxes(p_box, p_lod, top);
    Level cur;
    for (int m = top; m >= p_lod; m--) {
        const Box &box = boxes[size_t(m - p_lod)];
        if (m == top) {
            cur.allocate(m, box, p_spec.set);
        } else {
            // With r_levels the previous level was moved there.
            cur = subdivide(r_levels ? r_levels->back() : cur, box);
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

} // namespace Cascade
