#include "test_common.hpp"

#include "exact_noise.hpp"
#include "sdf_dummy.hpp"

#include <cmath>

using namespace ExactNoise;

namespace {

WorldPos at(int64_t p_x, int64_t p_y, int64_t p_z, Float3 p_frac = Float3{}) {
    WorldPos p;
    p.voxel = VoxelPos(p_x, p_y, p_z);
    p.frac = p_frac;
    return p;
}

bool same_bits(const Float4 &p_a, const Float4 &p_b) {
    return float_bits(p_a.x) == float_bits(p_b.x) && float_bits(p_a.y) == float_bits(p_b.y) && float_bits(p_a.z) == float_bits(p_b.z) &&
            float_bits(p_a.w) == float_bits(p_b.w);
}

// LODs above 0 checked against LOD 0 (up to 24, past the default 15 LODs).
constexpr int SCALED_LODS[] = { 1, 2, 5, 9, 14, 20, 24 };

// Deterministic pseudo-random points: near 0, around the 2^32 low-word wrap and far out (2^45, 2^60),
// with fractions in [0, 1) as multiples of 2^-18 (the vertex grid's resolution). y in [p_y0, p_y0 + p_dy).
std::vector<WorldPos> random_points(int p_count, int64_t p_y0, int64_t p_dy) {
    const int64_t bases[] = { 0, int64_t(1) << 32, -(int64_t(1) << 32), int64_t(1) << 45, -(int64_t(1) << 60) };
    std::vector<WorldPos> out;
    for (int i = 0; i < p_count; i++) {
        const UVec3 h = pcg3d(UVec3{ uint32_t(i), 0x5EEDu, 0x1234u });
        const UVec3 f = pcg3d(h);
        const int64_t base = bases[i % 5];
        auto frac = [](uint32_t p_u) { return float(p_u >> 14) * exact_pow2(-18); };
        out.push_back(at(base + int64_t(h.x % 200000u) - 100000, p_y0 + int64_t(h.y % uint32_t(p_dy)), base - int64_t(h.z % 200000u) + 100000,
                Float3{ frac(f.x), frac(f.y), frac(f.z) }));
    }
    return out;
}

} // namespace

TEST_CASE("ExactNoise: grad3 is Perlin's 12 cube edges, padded to 16") {
    const float table[16][3] = { { 1, 1, 0 }, { -1, 1, 0 }, { 1, -1, 0 }, { -1, -1, 0 }, { 1, 0, 1 }, { -1, 0, 1 }, { 1, 0, -1 },
        { -1, 0, -1 }, { 0, 1, 1 }, { 0, -1, 1 }, { 0, 1, -1 }, { 0, -1, -1 }, { 1, 1, 0 }, { -1, 1, 0 }, { 0, -1, 1 }, { 0, -1, -1 } };
    for (uint32_t i = 0; i < 16; i++) {
        CAPTURE(i);
        const Grad3 g = grad3(i);
        for (int a = 0; a < 3; a++) {
            CHECK(float_bits(g[a]) == float_bits(table[i][a])); // +0, never -0
        }
    }
}

TEST_CASE("ExactNoise: grad2 gives 8 distinct gradients of length sqrt(5)") {
    for (uint32_t i = 0; i < 8; i++) {
        CAPTURE(i);
        const Grad2 g = grad2(i);
        CHECK(g[0] * g[0] + g[1] * g[1] == 5.0f);
        for (uint32_t j = 0; j < i; j++) {
            CHECK((grad2(j)[0] != g[0] || grad2(j)[1] != g[1]));
        }
    }
}

TEST_CASE("ExactNoise: hash_cell folds the high words and the seed in") {
    const int64_t w = int64_t(1) << 32;
    const Vector3i64 cells[] = { Vector3i64(0, 0, 0), Vector3i64(-1, 5, 7), Vector3i64(w, 0, 0), Vector3i64(0, w, 0),
        Vector3i64(0, 0, w), Vector3i64(-w, 0, 0), Vector3i64(int64_t(1) << 60, 3, -(int64_t(1) << 60)) };
    for (const Vector3i64 &c : cells) {
        CAPTURE(c);
        // The same as the GPU's word form.
        CHECK(hash_cell(c, 42) == hash_words(WorldCoord::lo32(c.x), WorldCoord::lo32(c.y), WorldCoord::lo32(c.z), WorldCoord::hi32(c.x), WorldCoord::hi32(c.y), WorldCoord::hi32(c.z), 42));
        CHECK(hash_cell(c, 42) != hash_cell(c, 43));
    }
    // Cells 2^32 apart share their low words but not their hash.
    CHECK(hash_cell(Vector3i64(3, 4, 5), 1) != hash_cell(Vector3i64(3 + w, 4, 5), 1));
    CHECK(hash_cell(Vector3i64(3, 4, 5), 1) != hash_cell(Vector3i64(3, 4 - w, 5), 1));
    CHECK(hash_cell(Vector3i64(3, 4, 5), 1) != hash_cell(Vector3i64(3, 4, 5 + w), 1));
    CHECK(hash_cell_2d(3, 5, 1) == hash_cell(Vector3i64(3, 5, PLANE_2D), 1));
}

TEST_CASE("ExactNoise: octave_axis splits voxel + offset into cell and 24-bit position") {
    int64_t cell;
    uint32_t q;
    octave_axis(10, 0.5f, 0, 0, cell, q);
    CHECK(cell == 10);
    CHECK(q == (1u << 23));
    octave_axis(-1, 0.0f, 3, 0, cell, q); // floor(-1 / 8) = -1, 7 voxels into the cell
    CHECK(cell == -1);
    CHECK(q == (7u << 21));
    octave_axis(-9, 0.25f, 3, 1, cell, q); // -8 + 0.25 -> cell -1, position 0.25 / 8
    CHECK(cell == -1);
    CHECK(q == (1u << 19));
    octave_axis(int64_t(3) << 40, 0.75f, 40, 0, cell, q); // k > 24: frac dropped
    CHECK(cell == 3);
    CHECK(q == 0u);
    octave_axis((int64_t(3) << 40) + (int64_t(1) << 39), 0.0f, 40, 0, cell, q);
    CHECK(cell == 3);
    CHECK(q == (1u << 23));
    octave_axis(-1, 0.0f, MAX_K, 0, cell, q);
    CHECK(cell == -1);
    CHECK(q == 0xFFFFFFu);
}

TEST_CASE("ExactNoise: perlin is zero on its lattice and its derivatives match finite differences") {
    const int k = 5;
    const int64_t n = int64_t(1) << k;
    const Vector3i64 offset(9, 13, 17);
    for (int i = 0; i < 8; i++) {
        const int64_t x = (i - 3) * n - offset.x;
        const int64_t y = (i * 7) * n - offset.y;
        const int64_t z = -(i * 11) * n - offset.z;
        CHECK(perlin3(at(x, y, z), k, 77, offset).w == 0.0f);
        CHECK(perlin2(at(x, 0, z), k, 77, offset).w == 0.0f);
    }
    // r.xyz are per cell: a centred difference over one voxel is (r / 2^k) per voxel.
    for (int i = 0; i < 16; i++) {
        const int64_t x = 1000 + i * 37;
        const int64_t y = -500 + i * 53;
        const int64_t z = 77 - i * 29;
        CAPTURE(i);
        const Float4 r3 = perlin3(at(x, y, z), k, 5, offset);
        const float dx = (perlin3(at(x + 1, y, z), k, 5, offset).w - perlin3(at(x - 1, y, z), k, 5, offset).w) * 0.5f;
        const float dy = (perlin3(at(x, y + 1, z), k, 5, offset).w - perlin3(at(x, y - 1, z), k, 5, offset).w) * 0.5f;
        const float dz = (perlin3(at(x, y, z + 1), k, 5, offset).w - perlin3(at(x, y, z - 1), k, 5, offset).w) * 0.5f;
        CHECK(std::fabs(dx * float(n) - r3.x) < 0.05f);
        CHECK(std::fabs(dy * float(n) - r3.y) < 0.05f);
        CHECK(std::fabs(dz * float(n) - r3.z) < 0.05f);
        const Float4 r2 = perlin2(at(x, 0, z), k, 5, offset);
        const float ex = (perlin2(at(x + 1, 0, z), k, 5, offset).w - perlin2(at(x - 1, 0, z), k, 5, offset).w) * 0.5f;
        CHECK(std::fabs(ex * float(n) - r2.x) < 0.1f);
        CHECK(r2.y == 0.0f);
    }
}

TEST_CASE("ExactNoise: make_fbm halves wavelength and amplitude, offsets a quarter to three quarters of a cell") {
    const OctaveSet set = make_fbm(false, 1234, 20, 5, 8.0, 0.5);
    CHECK(set.octaves.size() == 5);
    if (set.octaves.size() != 5) {
        return;
    }
    for (int i = 0; i < 5; i++) {
        const Octave &o = set.octaves[i];
        CAPTURE(i);
        CHECK(o.k == 20 - i);
        CHECK(o.amp == float(std::ldexp(8.0, -i)));
        CHECK(gradient_factor(o, 0) == float(std::ldexp(double(o.amp), -o.k)));
        CHECK(gradient_factor(o, 7) == float(std::ldexp(double(o.amp), 7 - o.k)));
        const int64_t cell = int64_t(1) << o.k;
        for (const int64_t v : { o.offset.x, o.offset.y, o.offset.z }) {
            CHECK(v >= cell / 4);
            CHECK(v < 3 * cell / 4);
        }
        for (int j = 0; j < i; j++) {
            CHECK(set.octaves[j].seed != o.seed);
        }
    }
}

TEST_CASE("ExactNoise: det_rsqrt is accurate to about 1e-7") {
    for (const float x : { 1e-12f, 3e-7f, 0.5f, 1.0f, 2.0f, 3.0f, 1234.5f, 6.5e9f }) {
        CAPTURE(x);
        const double y = double(det_rsqrt(x));
        CHECK(std::fabs(y * std::sqrt(double(x)) - 1.0) < 5e-7);
    }
}

TEST_CASE("WorldCoord: sat_diff is the exact difference, saturated") {
    CHECK(WorldCoord::sat_diff(10, 3, 100) == 7);
    CHECK(WorldCoord::sat_diff(3, 10, 100) == -7);
    CHECK(WorldCoord::sat_diff(int64_t(1) << 60, 0, 1 << 24) == (1 << 24));
    CHECK(WorldCoord::sat_diff(-(int64_t(1) << 60), int64_t(1) << 60, 1 << 24) == -(1 << 24));
    CHECK(WorldCoord::sat_diff((int64_t(1) << 40) + 5, int64_t(1) << 40, 16) == 5);
}

TEST_CASE("SDFDummy: skipping the caves where they can't show changes no bit") {
    ChunkParam param;
    SDFDummy skip;
    SDFDummy full;
    full.cave_skip = false;
    skip.configure(param, 15);
    full.configure(param, 15);
    const int64_t datum = std::llround(SDFDummy::DATUM_M / param.voxel_size);
    // Columns from 400 m below the datum to 100 m above: below the cave band, in it, in its top metres
    // and in the air.
    int differing = 0;
    for (int c = 0; c < 24; c++) {
        const int64_t x = c * 977 - 5000;
        const int64_t z = 3000 - c * 613;
        for (int64_t y = datum - 800; y <= datum + 200; y++) {
            const Float4 a = skip.sdf(at(x, y, z, Float3{ 0.0f, 0.25f, 0.0f }), 0);
            const Float4 b = full.sdf(at(x, y, z, Float3{ 0.0f, 0.25f, 0.0f }), 0);
            differing += float_bits(a.x) != float_bits(b.x) || float_bits(a.y) != float_bits(b.y) ||
                    float_bits(a.z) != float_bits(b.z) || float_bits(a.w) != float_bits(b.w);
        }
        CHECK(skip.sdf(at(x, datum - 800, z), 0).w < 0.0f); // rock below the band
        CHECK(skip.sdf(at(x, datum + 200, z), 0).w > 0.0f); // air
    }
    CHECK(differing == 0);
}

TEST_CASE("ExactNoise: the fBm gradient at LOD L is exactly 2^L times the one at LOD 0, the value unchanged") {
    ChunkParam param;
    SDFDummy sdf;
    sdf.configure(param, 15);
    const OctaveSet wide = make_fbm(false, 99, 20, 6, 1000.0, 0.5); // factors 2^-10 at LOD 0
    int differing = 0;
    int checked = 0;
    for (const OctaveSet *set : { &sdf.get_height(), &sdf.get_caves(), &wide }) {
        for (const WorldPos &p : random_points(400, -100000, 200000)) {
            const Float4 n0 = fbm(*set, p, 0);
            for (const int lod : SCALED_LODS) {
                const float s = exact_pow2(lod);
                const Float4 n = fbm(*set, p, lod);
                differing += !same_bits(n, Float4{ n0.x * s, n0.y * s, n0.z * s, n0.w });
                checked++;
            }
        }
    }
    CHECK(checked > 0);
    CHECK(differing == 0);
}

TEST_CASE("SDFDummy: the distance in metres and the normalised gradient are bit-identical at every LOD") {
    for (const float voxel_size : { 0.5f, 0.05f }) {
        CAPTURE(voxel_size);
        ChunkParam param;
        param.voxel_size = voxel_size;
        SDFDummy plain;
        SDFDummy full; // caves everywhere
        full.cave_skip = false;
        SDFDummy pillar;
        pillar.zero_pillar = 1;
        SDFDummy rig;
        rig.test_rig = true;
        for (SDFDummy *s : { &plain, &full, &pillar, &rig }) {
            s->configure(param, 15);
        }
        const double vs = voxel_size;
        const int64_t datum = std::llround(SDFDummy::DATUM_M / vs);
        // Columns through the air, the ground, the cave band and the rock below, far out included, and
        // a box around the zero pillar's top and the test rig (placed relative to the origin, 0 here).
        std::vector<WorldPos> points = random_points(3000, datum - std::llround(400.0 / vs), std::llround(500.0 / vs));
        const int64_t px = std::llround(SDFDummy::ZERO_PILLAR_X / vs);
        const int64_t pz = std::llround(SDFDummy::ZERO_PILLAR_Z / vs);
        const int64_t py = std::llround(SDFDummy::ZERO_PILLAR_TOP_M / vs);
        const int64_t r = std::llround(10.0 / vs);
        for (int64_t i = 0; i < 4000; i++) {
            const UVec3 h = pcg3d(UVec3{ uint32_t(i), 0xB0Bu, 0u });
            points.push_back(at(px + int64_t(h.x % uint32_t(2 * r)) - r, py + int64_t(h.y % uint32_t(4 * r)) - 3 * r,
                    pz + int64_t(h.z % uint32_t(2 * r)) - r, Float3{ 0.5f, 0.0f, 0.25f }));
        }
        int differing = 0;
        int pillar_wins = 0;
        for (const WorldPos &p : points) {
            const Float4 a0 = plain.sdf(p, 0);
            const Float4 b0 = full.sdf(p, 0);
            const Float4 c0 = pillar.sdf(p, 0);
            const float r0 = rig.evaluate(p, 0);
            pillar_wins += !same_bits(c0, a0);
            for (const int lod : SCALED_LODS) {
                differing += !same_bits(plain.sdf(p, lod), a0);
                differing += !same_bits(full.sdf(p, lod), b0);
                differing += !same_bits(pillar.sdf(p, lod), c0);
                differing += float_bits(rig.evaluate(p, lod)) != float_bits(r0);
            }
        }
        CHECK(pillar_wins > 0);
        CHECK(differing == 0);
    }
}

TEST_CASE("ExactNoise: gradient_range_error flags factors outside [2^-16, 2^40] at the evaluated LODs") {
    ChunkParam param;
    SDFDummy sdf;
    sdf.configure(param, 15);
    // Today's terrain, up to LOD 24.
    CHECK(gradient_range_error(sdf.get_height(), 25).is_empty());
    CHECK(gradient_range_error(sdf.get_caves(), 25).is_empty());
    // amp 2^30 at k = 0: 2^40 at LOD 10, 2^41 at LOD 11.
    const OctaveSet steep = make_fbm(false, 1, 0, 1, std::ldexp(1.0, 30), 0.5);
    CHECK(gradient_range_error(steep, 11).is_empty());
    CHECK_FALSE(gradient_range_error(steep, 12).is_empty());
    // amp 1 at k = 20: 2^-20 at LOD 0; amp 16: 2^-16 at LOD 0.
    OctaveSet gentle = make_fbm(true, 1, 20, 1, 1.0, 0.5);
    CHECK_FALSE(gradient_range_error(gentle, 1).is_empty());
    gentle.octaves[0].amp = exact_pow2(4);
    CHECK(gradient_range_error(gentle, 15).is_empty());
}

TEST_CASE("SDFDummy: relief octaves only above 15 LODs, in range at 25 LODs") {
    for (float voxel : { 0.5f, 0.05f }) {
        ChunkParam param;
        param.voxel_size = voxel;
        SDFDummy sdf;
        sdf.configure(param, 15);
        CHECK(sdf.get_relief().octaves.empty());
        sdf.configure(param, 25);
        CHECK(sdf.get_relief().octaves.size() == 20);
        for (const OctaveSet *set : { &sdf.get_height(), &sdf.get_caves(), &sdf.get_relief() }) {
            CHECK(gradient_range_error(*set, 25).is_empty());
        }
        // Back to the default: the same terrain as before (no relief left over).
        sdf.configure(param, 15);
        CHECK(sdf.get_relief().octaves.empty());
    }
}

// Regression guard for the refactors of perlin3 (the cascade's perlin3_cross shares its setup): a
// digest of its output bits at 4096 points (fractions included), k 0..62, recorded on dev.
TEST_CASE("ExactNoise: perlin3 output bits are unchanged (digest)") {
    const std::vector<WorldPos> points = random_points(4096, -(int64_t(1) << 40), int64_t(1) << 30);
    uint32_t digest = 0x811C9DC5u;
    int i = 0;
    for (const WorldPos &p : points) {
        const int k = i % (MAX_K + 1);
        const Float4 r = perlin3(p, k, 0x9A11u + uint32_t(i), Vector3i64(i * 7, -i * 13, i * 29));
        for (const float f : { r.x, r.y, r.z, r.w }) {
            digest = (digest ^ float_bits(f)) * 0x01000193u;
        }
        i++;
    }
    CHECK(digest == 837132468u);
}

TEST_CASE("ExactNoise: perlin3_cross extends perlin3 with cross derivatives matching finite differences") {
    const int k = 6;
    const Vector3i64 offset(21, 37, 45);
    const float n = float(1 << k);
    double worst = 0.0;
    // Fractions included, near 0, the 2^32 wrap and far out.
    for (const WorldPos &p : random_points(64, -5000, 10000)) {
        const Channels8 c = perlin3_cross(p, k, 99, offset);
        const Float4 g = perlin3(p, k, 99, offset);
        CHECK(same_bits(Float4{ c.c[1], c.c[2], c.c[3], c.c[0] }, g));
        auto at = [&](int p_dy, int p_dz) {
            WorldPos q = p;
            q.voxel += Vector3i64(0, p_dy, p_dz);
            return perlin3_cross(q, k, 99, offset);
        };
        const Channels8 yp = at(1, 0);
        const Channels8 ym = at(-1, 0);
        const Channels8 zp = at(0, 1);
        const Channels8 zm = at(0, -1);
        // Per cell: a centred difference over one voxel (1/64 cell) times 64. Channels: 1 x, 2 y, 4 xy, 5 xz, 6 yz, 7 xyz.
        const double errors[] = { std::fabs((yp.c[1] - ym.c[1]) * 0.5f * n - c.c[4]), std::fabs((zp.c[1] - zm.c[1]) * 0.5f * n - c.c[5]),
            std::fabs((zp.c[2] - zm.c[2]) * 0.5f * n - c.c[6]), std::fabs((zp.c[4] - zm.c[4]) * 0.5f * n - c.c[7]) };
        for (const double e : errors) {
            worst = MAX(worst, e);
        }
    }
    MESSAGE("perlin3_cross vs finite differences (per cell, h = 1/64 cell): max error ", worst);
    CHECK(worst < 0.05); // |cross| up to ~8, |dxyz| up to ~18 per cell; FD error O(h^2)
}
