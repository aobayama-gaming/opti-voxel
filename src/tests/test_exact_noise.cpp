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
        CHECK(o.damp == float(std::ldexp(double(o.amp), -o.k)));
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
    skip.configure(param);
    full.configure(param);
    const int64_t datum = std::llround(SDFDummy::DATUM_M / param.voxel_size);
    // Columns from 400 m below the datum to 100 m above: below the cave band, in it, in its top metres
    // and in the air.
    int differing = 0;
    for (int c = 0; c < 24; c++) {
        const int64_t x = c * 977 - 5000;
        const int64_t z = 3000 - c * 613;
        for (int64_t y = datum - 800; y <= datum + 200; y++) {
            const Float4 a = skip.sdf(at(x, y, z, Float3{ 0.0f, 0.25f, 0.0f }));
            const Float4 b = full.sdf(at(x, y, z, Float3{ 0.0f, 0.25f, 0.0f }));
            differing += float_bits(a.x) != float_bits(b.x) || float_bits(a.y) != float_bits(b.y) ||
                    float_bits(a.z) != float_bits(b.z) || float_bits(a.w) != float_bits(b.w);
        }
        CHECK(skip.sdf(at(x, datum - 800, z)).w < 0.0f); // rock below the band
        CHECK(skip.sdf(at(x, datum + 200, z)).w > 0.0f); // air
    }
    CHECK(differing == 0);
}
