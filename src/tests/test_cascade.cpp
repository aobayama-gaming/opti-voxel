#include "test_common.hpp"

#include "cascade.hpp"
#include "cascade_measure.hpp"
#include "cascade_terrain.hpp"

#include <cmath>

#if defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
#define CASCADE_TEST_MXCSR 1
constexpr unsigned int MXCSR_FTZ_DAZ = 0x8040u;
#endif

using namespace Cascade;
using CascadeTerrain::earth;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

constexpr ChannelSet SETS[] = { ChannelSet::HERMITE8, ChannelSet::LINEAR4 };

bool same_bits(const float *p_a, const float *p_b, int p_n) {
    for (int c = 0; c < p_n; c++) {
        if (float_bits(p_a[c]) != float_bits(p_b[c])) {
            return false;
        }
    }
    return true;
}

const Level *find_level(const std::vector<Level> &p_levels, int p_m) {
    for (const Level &l : p_levels) {
        if (l.m == p_m) {
            return &l;
        }
    }
    return nullptr;
}

int64_t subnormals(const std::vector<Level> &p_levels) {
    int64_t n = 0;
    for (const Level &l : p_levels) {
        for (const std::vector<float> &c : l.ch) {
            for (const float f : c) {
                n += std::fpclassify(f) == FP_SUBNORMAL;
            }
        }
    }
    return n;
}

int64_t differing_bits(const std::vector<Level> &p_a, const std::vector<Level> &p_b) {
    int64_t n = 0;
    for (size_t i = 0; i < p_a.size(); i++) {
        for (int c = 0; c < CHANNELS; c++) {
            for (size_t j = 0; j < p_a[i].ch[c].size(); j++) {
                n += float_bits(p_a[i].ch[c][j]) != float_bits(p_b[i].ch[c][j]);
            }
        }
    }
    return n;
}

// P(X, Y, Z), a cubic with integer coefficients over LOD0 voxels, and its channels on Λ_m (exact
// integers, small enough for float32).
struct Cubic {
    struct Term {
        int c, i, j, l; // c * x^i y^j z^l
    };
    std::vector<Term> terms = { { 1, 3, 0, 0 }, { -2, 0, 3, 0 }, { 1, 0, 0, 3 }, { 3, 1, 1, 1 }, { -1, 2, 1, 0 }, { 2, 0, 1, 2 }, { 1, 1, 0, 2 },
        { 5, 2, 0, 0 }, { -3, 0, 2, 0 }, { 2, 1, 1, 0 }, { 7, 1, 0, 0 }, { -4, 0, 0, 1 }, { 11, 0, 0, 0 } };

    // d^p_d/dx^p_d of x^e (p_d in {0, 1}).
    static int64_t dpow(int64_t p_x, int p_e, int p_d) {
        int64_t r = p_d == 0 ? 1 : p_e;
        for (int i = 0; i < p_e - p_d; i++) {
            r *= p_x;
        }
        return p_d == 1 && p_e == 0 ? 0 : r;
    }
    void channels(int p_m, const Vector3i64 &p_p, float r_ch[CHANNELS]) const {
        const int64_t s = int64_t(1) << p_m;
        for (int c = 0; c < CHANNELS; c++) {
            const int axes = CHANNEL_AXES[c];
            int64_t sum = 0;
            for (const Term &t : terms) {
                sum += t.c * dpow(p_p.x * s, t.i, axes & 1) * dpow(p_p.y * s, t.j, (axes >> 1) & 1) * dpow(p_p.z * s, t.l, (axes >> 2) & 1);
            }
            r_ch[c] = float(sum * (int64_t(1) << (p_m * channel_order(c))));
        }
    }
};

} // namespace

TEST_CASE("Cascade: parent boxes cover the Hermite stencil, chunk boxes are the vertex grid") {
    const Box b{ { -3, 4, -1 }, { 5, 9, -1 } };
    const Box p = parent_box(b);
    CHECK(p.lo[0] == -2);
    CHECK(p.hi[0] == 3);
    CHECK(p.lo[1] == 2);
    CHECK(p.hi[1] == 5);
    CHECK(p.lo[2] == -1);
    CHECK(p.hi[2] == 0);
    const ChunkParam param;
    const ChunkId id(int64_t(5) << 3, -(int64_t(7) << 3), int64_t(1) << 3);
    const Box c = chunk_box(id, param);
    const VoxelPos anchor = ChunkMath::chunk_min_voxel(id, param);
    CHECK(c.size(0) == param.chunk_size);
    CHECK(c.lo[0] == anchor.x / 8 - param.skirt_size);
    CHECK(c.lo[1] == anchor.y / 8 - param.skirt_size);
}

// Criterion 1: V_m(p) is a pure function of p and m.
TEST_CASE("Cascade: the same lattice point gives the same bits from chunks of every LOD and from a point query") {
    const ChunkParam param;
    // Near 0, around the voxel low-word wrap, the noise-cell low-word wrap of k = 20 (2^52) and +-2^60.
    const int64_t bases[] = { 0, (int64_t(1) << 32) - 40, -(int64_t(1) << 32) + 3, (int64_t(1) << 52) - 1000, (int64_t(1) << 60) - 4096,
        -(int64_t(1) << 60) + 4096 };
    const int ms[] = { 0, 1, 3, 7, 16 };
    for (const ChannelSet set : SETS) {
        const Spec spec = earth(set);
        int compared = 0;
        int differing = 0;
        for (const int64_t base : bases) {
            for (const int m : ms) {
                const int64_t step = int64_t(1) << m;
                const int64_t b = WorldCoord::floor_div(base, step);
                const Vector3i64 lattice(b, b + 3, -b - 5);
                float expect[CHANNELS];
                point(spec, lattice, m, expect);
                int previous = -1;
                for (const int lod : { 0, m / 2, std::max(m - 1, 0), m }) {
                    if (lod == previous) {
                        continue;
                    }
                    previous = lod;
                    std::vector<Level> levels;
                    run(spec, chunk_box(ChunkMath::chunk_at_voxel(lattice * step, lod, param), param), lod, &levels);
                    const Level *l = find_level(levels, m);
                    const bool found = l != nullptr && l->box.contains(lattice);
                    CHECK(found);
                    if (!found) {
                        continue;
                    }
                    float got[CHANNELS];
                    l->channels_at(lattice, got);
                    compared++;
                    differing += !same_bits(got, expect, CHANNELS);
                }
            }
        }
        CAPTURE(int(set));
        CHECK(compared == 6 * (1 + 2 + 4 + 4 + 4)); // distinct LODs for m = 0, 1, 3, 7, 16
        CHECK(differing == 0);
    }
}

// Criterion 2: V_L = flush(V_{L+1} with slopes x 2^-order + O_L) at the points L and L+1 share.
TEST_CASE("Cascade: a LOD differs from the next only by the bands it inserts, bit for bit") {
    const ChunkParam param;
    for (const ChannelSet set : SETS) {
        const Spec spec = earth(set);
        const int n = channel_count(set);
        for (const int lod : { 0, 4, 20 }) {
            std::vector<Level> levels;
            run(spec, chunk_box(CascadeMeasure::random_chunk(lod, 77, lod), param), lod, &levels);
            const Level *fine = find_level(levels, lod);
            const Level *coarse = find_level(levels, lod + 1);
            CHECK((fine != nullptr && coarse != nullptr));
            if (fine == nullptr || coarse == nullptr) {
                continue;
            }
            int checked = 0;
            int differing = 0;
            for_each_point(fine->box, [&](const Vector3i64 &p_p, int64_t) {
                if (((p_p.x | p_p.y | p_p.z) & 1) != 0) {
                    return;
                }
                float expect[CHANNELS] = {};
                coarse->channels_at(Vector3i64(p_p.x / 2, p_p.y / 2, p_p.z / 2), expect);
                for (int c = 0; c < n; c++) {
                    expect[c] = expect[c] * exact_pow2(-channel_order(c));
                }
                for (const Band &band : spec.bands) {
                    if (spec.insert_level(band) == lod) {
                        float o[CHANNELS];
                        band_channels(spec, band, p_p, o);
                        for (int c = 0; c < n; c++) {
                            expect[c] = expect[c] + o[c];
                        }
                    }
                }
                for (int c = 0; c < n; c++) {
                    expect[c] = flush_value(expect[c]);
                }
                float got[CHANNELS];
                fine->channels_at(p_p, got);
                checked++;
                differing += !same_bits(got, expect, CHANNELS);
            });
            CAPTURE(lod);
            CHECK(checked == 16 * 16 * 16);
            CHECK(differing == 0);
        }
    }
}

// Criterion 3.
TEST_CASE("Cascade: a cubic polynomial inserted at the top is reproduced exactly at every level") {
    const Cubic cubic;
    Spec spec;
    spec.top = 4;
    const Source source = [&](Level &r_level) {
        if (r_level.m != spec.top) {
            return;
        }
        for_each_point(r_level.box, [&](const Vector3i64 &p_p, int64_t p_i) {
            float ch[CHANNELS];
            cubic.channels(r_level.m, p_p, ch);
            for (int c = 0; c < CHANNELS; c++) {
                r_level.ch[c][p_i] = ch[c];
            }
        });
    };
    std::vector<Level> levels;
    run(spec, Box{ { -40, 18, -30 }, { 30, 46, 30 } }, 0, &levels, source);
    CHECK(levels.size() == 5);
    int64_t checked = 0;
    int64_t differing = 0;
    for (const Level &l : levels) {
        for_each_point(l.box, [&](const Vector3i64 &p_p, int64_t) {
            float expect[CHANNELS];
            float got[CHANNELS];
            cubic.channels(l.m, p_p, expect);
            l.channels_at(p_p, got);
            checked++;
            differing += !same_bits(got, expect, CHANNELS);
        });
    }
    MESSAGE("cubic: ", checked, " lattice points over 5 levels");
    CHECK(differing == 0);
}

// Criterion 4: below the finest inserted band the cascade is the Hermite interpolant; its carried slopes
// are the derivatives of its values.
TEST_CASE("Cascade: carried slopes agree with finite differences of the carried values") {
    for (const ChannelSet set : SETS) {
        Spec spec = CascadeTerrain::make_spec(0x51D3u, 12, 0.125, set, 3);
        // Keep the bands of wavelength >= 256 voxels: Λ_0 to Λ_4 only carry.
        std::vector<Band> coarse;
        for (const Band &b : spec.bands) {
            if (b.k >= 8) {
                coarse.push_back(b);
            }
        }
        spec.bands = coarse;
        const Box box{ { 1000, -300, 5000 }, { 1039, -261, 5039 } };
        const Box inner{ { 1001, -299, 5001 }, { 1038, -262, 5038 } };
        const Level v = run(spec, box, 0);
        double worst = 0.0;
        double scale = 0.0;
        const Vector3i64 e[3] = { Vector3i64(1, 0, 0), Vector3i64(0, 1, 0), Vector3i64(0, 0, 1) };
        for_each_point(inner, [&](const Vector3i64 &p_p, int64_t) {
            for (int a = 0; a < 3; a++) {
                const double fd = (double(v.at(VALUE, p_p + e[a])) - double(v.at(VALUE, p_p - e[a]))) * 0.5;
                worst = std::max(worst, std::fabs(fd - v.at(DX + a, p_p)));
                scale = std::max(scale, std::fabs(double(v.at(DX + a, p_p))));
            }
        });
        CAPTURE(int(set));
        MESSAGE("channel set ", int(set), ": max |FD - slope| = ", worst, " (max |slope| ", scale, ")");
        // Tolerance: 1% of the largest slope (Hermite FD error over 1/32 of a cell plus float rounding).
        CHECK(worst <= 0.01 * scale);
    }
}

// Criterion 5 (the report tool measures the same on 64x more points).
TEST_CASE("Cascade: fidelity of a cascaded octave, 8 samples per wavelength, 8 channels") {
    Spec spec;
    spec.insert_shift = 3;
    const CascadeMeasure::Fidelity f = CascadeMeasure::measure_fidelity(spec, 7, 4, 48, 0xF1u);
    MESSAGE("points ", f.points, " value max ", f.value_max, " rms ", f.value_rms, " slope max ", f.slope_max, " normal max (r = 1/8) ",
            f.normal_max_deg[CascadeMeasure::REF_STEEPNESS], " deg");
    CHECK(f.passes());
}

// Criterion 6.
TEST_CASE("Cascade: points per chunk confirm the cost table (~+1.3% at LOD 0 vs LOD 24)") {
    const Spec spec = earth(ChannelSet::HERMITE8);
    CHECK(spec.top_level() == 26);
    const CascadeMeasure::Cost top = CascadeMeasure::measure_cost(spec, 24, 50, 1, false);
    const CascadeMeasure::Cost bottom = CascadeMeasure::measure_cost(spec, 0, 50, 1, false);
    CHECK(bottom.first_levels.size() == 27);
    if (bottom.first_levels.size() != 27) {
        return;
    }
    CHECK(bottom.first_levels[0] == 32768);
    CHECK(bottom.first_levels[1] == 4913);
    CHECK(bottom.first_levels[2] == 1000);
    CHECK(bottom.first_levels[3] == 216);
    CHECK(top.points_mean == 38681.0);
    const double ratio = bottom.points_mean / top.points_mean;
    MESSAGE("points per chunk: LOD 24 ", top.points_mean, ", LOD 0 ", bottom.points_mean, " (max ", bottom.points_max, "), ratio ", ratio);
    CHECK(ratio > 1.010);
    CHECK(ratio < 1.016);
}

// Criterion 7.
TEST_CASE("Cascade: no denormal at any level, and flushing denormals (FTZ/DAZ) changes no bit") {
    const ChunkParam param;
#ifdef CASCADE_TEST_MXCSR
    // The reference runs in IEEE mode (denormals kept), whatever the engine set.
    const unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr & ~MXCSR_FTZ_DAZ);
#endif
    // One band carried 60 levels down: its slopes shrink by 2^-|S| per level.
    Spec deep;
    deep.insert_shift = 2;
    deep.bands = { Band{ 62, 0xDEEBu, Vector3i64(int64_t(3) << 59, int64_t(1) << 60, int64_t(5) << 58), 1.0f } };
    const Box box = chunk_box(CascadeMeasure::random_chunk(0, 5, 0), param);
    {
        // run() without the per-level flush: this case does produce denormals, the flush is what prevents them.
        const std::vector<Box> boxes = level_boxes(box, 0, deep.top_level());
        std::vector<Level> unflushed(1);
        unflushed[0].allocate(deep.top_level(), boxes.back(), deep.set);
        insert_bands(deep, unflushed[0]);
        for (int m = deep.top_level() - 1; m >= 0; m--) {
            unflushed.push_back(subdivide(unflushed.back(), boxes[size_t(m)]));
            insert_bands(deep, unflushed.back());
        }
        const int64_t count = subnormals(unflushed);
        MESSAGE("denormal channels without the flush: ", count);
        CHECK(count > 0);
    }
    const Spec specs[] = { deep, earth(ChannelSet::HERMITE8), earth(ChannelSet::LINEAR4) };
    for (const Spec &spec : specs) {
        std::vector<Level> ieee;
        run(spec, box, 0, &ieee);
        CHECK(subnormals(ieee) == 0);
#ifdef CASCADE_TEST_MXCSR
        _mm_setcsr(csr | MXCSR_FTZ_DAZ);
        std::vector<Level> ftz;
        run(spec, box, 0, &ftz);
        _mm_setcsr(csr & ~MXCSR_FTZ_DAZ);
        CHECK(ftz.size() == ieee.size());
        if (ftz.size() == ieee.size()) {
            CHECK(differing_bits(ieee, ftz) == 0);
        }
#else
        MESSAGE("FTZ/DAZ comparison skipped: not an SSE target");
#endif
    }
#ifdef CASCADE_TEST_MXCSR
    _mm_setcsr(csr);
#endif
}

// GPU discoverer step 2: the allocation-free point query is point() bit for bit, every LOD, near 0, across
// 2^32-voxel boundaries and the k = 20 noise-cell wrap, and at +-2^60.
TEST_CASE("Cascade: point_into equals point, bit for bit (10,240 points per LOD 0-24, both channel sets)") {
    const int64_t P32 = int64_t(1) << 32;
    const int64_t bases[] = { 0, P32, -P32, 3 * P32, (int64_t(1) << 52) - 1000, (int64_t(1) << 60) - 4096, -(int64_t(1) << 60) + 4096, -5 * P32 };
    constexpr int PER_BASE = 1280;
    for (const ChannelSet set : SETS) {
        const Spec spec = earth(set);
        int64_t compared = 0;
        int64_t differing = 0;
        for (int m = 0; m <= 24; m++) {
            const int64_t step = int64_t(1) << m;
            uint32_t n = 0;
            for (const int64_t base : bases) {
                const int64_t b = WorldCoord::floor_div(base, step);
                for (int i = 0; i < PER_BASE; i++, n++) {
                    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(m), n, 0x9017u });
                    const Vector3i64 p(b + int64_t(h.x % 129u) - 64, b + int64_t(h.y % 129u) - 64, -b + int64_t(h.z % 129u) - 64);
                    float want[CHANNELS];
                    float got[CHANNELS];
                    point(spec, p, m, want);
                    point_into(spec, p, m, got);
                    compared++;
                    differing += same_bits(want, got, CHANNELS) ? 0 : 1;
                }
            }
        }
        CHECK(compared == 25 * 8 * PER_BASE);
        CHECK(differing == 0);
    }
}

// Where the climate is read: a modulated band above a run's last level reads the fields at its own level, the last
// level the carry its level above derives (Modulation::derive). A batch's variant choice (batch, batch_carry) is the
// union of its items' (carried, carries at the last level), and the fields are monotone in the level: a level with them
// always finds them in its parent level.
TEST_CASE("Cascade: FieldRange carries the fields to every reader above the last level, the carry to the last") {
    std::vector<FieldRange> ranges;
    for (const int floor : { 0, 3, 9 }) {
        for (const int ceiling : { floor, floor + 4, 20 }) {
            ranges.push_back(FieldRange{ floor, ceiling, false });
        }
    }
    ranges.push_back(FieldRange{ MAX_LEVEL + 1, -1, true }); // the samples' terms alone (a pick, ground terms)
    ranges.push_back(FieldRange{ 2, 6, true });
    int64_t checked = 0, bad = 0;
    for (const FieldRange &r : ranges) {
        for (int lo = 0; lo <= 24; lo += 3) {
            for (int hi = lo; hi <= MIN(lo + 8, 24); hi += 2) {
                for (int j = 0; j <= 26; j++) {
                    bool fields = false, carry = false;
                    for (int lod = lo; lod <= hi; lod++) {
                        const int m = lod + j;
                        fields = fields || r.carried(lod, m);
                        carry = carry || (j == 0 && r.carries(lod, m));
                        const bool modulated = r.floor <= m && m <= r.ceiling;
                        // The fields never at a run's last level, at every modulated level above it.
                        bad += r.carried(lod, lod);
                        bad += j > 0 && modulated && !r.carried(lod, m);
                        // The carry on the last level where it reads it (a modulated band, the samples' terms), and on the
                        // level above, which derives it from its fields; nowhere else.
                        const bool reads = (j == 0 && modulated) || r.sample;
                        bad += reads && !(r.carries(lod, lod) && r.carries(lod, lod + 1) && r.carried(lod, lod + 1));
                        bad += r.carries(lod, m) && j > 1;
                        // Monotone: the fields at m + 1 wherever m has them.
                        bad += r.carried(lod, m) && !r.carried(lod, m + 1);
                    }
                    bad += fields != r.batch(ov::ivec2(lo, hi), j);
                    bad += carry != r.batch_carry(ov::ivec2(lo, hi), j);
                    checked++;
                }
            }
        }
    }
    CHECK(checked > 1000);
    CHECK(bad == 0);
    CHECK(FieldRange().empty());
    CHECK_FALSE((FieldRange{ MAX_LEVEL + 1, -1, true }).empty());
}
