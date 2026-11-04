#include "test_climate_common.hpp"

#include "cascade.hpp"
#include "graph_compile.hpp"
#include "level_program.hpp"

#include <cmath>

#if defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
#define CASCADE_TEST_MXCSR 1
constexpr unsigned int MXCSR_FTZ_DAZ = 0x8040u;
#endif

// The Hermite kernel (cascade.hpp) through the programs that run it: the subdivision's boxes, its carry rule, its
// slopes and fidelity against analytic perlin3, and its denormal-free arithmetic. A program's own rules (purity, box ==
// point, the LOD step) are test_level_program.cpp's.

using namespace Cascade;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

// 5 cm voxels, octaves up to 62 (the deepest carry), no cap below the exact range.
Climate::Ground ground() {
    Climate::Ground g;
    g.voxel_size = 0.05;
    g.k_max = ExactNoise::MAX_K;
    g.amp_cap = exact_pow2(40);
    return g;
}

// One note per octave p_lo .. p_hi (seed p_seed), each of amplitude 2^p_steep_log2 of its wavelength, summed.
ov::Value fbm(int p_lo, int p_hi, int p_seed, int p_steep_log2) {
    ov::str nodes, rows;
    for (int k = p_lo; k <= p_hi; k++) {
        nodes += ov::format(R"({"id": "n%d", "kind": "note", "wavelength_m": %s, "seed": %d, "amp_m": %s}, )", int64_t(k), ov::num(std::ldexp(0.05, k)),
                int64_t(p_seed), ov::num(std::ldexp(0.05, k + p_steep_log2)));
        rows += ov::format(R"(%s{"in": "n%d"})", rows.empty() ? "" : ", ", int64_t(k));
    }
    return ClimateTest::json(R"({"biomes": [{"name": "b", "graph": {"nodes": [)" + nodes + R"({"id": "m", "kind": "mix", "rows": [)" + rows +
            R"(]}, {"id": "o", "kind": "output", "height": "m"}]}}]})");
}

bool compiled(const ov::Value &p_table, Terrain::LevelProgram &r_prog) {
    Climate::Table table;
    const ov::str err = Terrain::compile(p_table, ground(), Terrain::Mode::SPECIALISED, table, r_prog);
    CHECK(err == ov::str());
    return err.empty();
}

const Terrain::Values *find_level(const std::vector<Terrain::Values> &p_levels, int p_m) {
    for (const Terrain::Values &l : p_levels) {
        if (l.m == p_m) {
            return &l;
        }
    }
    return nullptr;
}

int64_t subnormals(const std::vector<Terrain::Values> &p_levels) {
    int64_t n = 0;
    for (const Terrain::Values &l : p_levels) {
        for (const std::vector<float> &c : l.ch) {
            for (const float f : c) {
                n += std::fpclassify(f) == FP_SUBNORMAL;
            }
        }
    }
    return n;
}

int64_t differing_bits(const std::vector<Terrain::Values> &p_a, const std::vector<Terrain::Values> &p_b) {
    int64_t n = 0;
    for (size_t i = 0; i < p_a.size(); i++) {
        for (size_t c = 0; c < p_a[i].ch.size(); c++) {
            for (size_t j = 0; j < p_a[i].ch[c].size(); j++) {
                n += float_bits(p_a[i].ch[c][j]) != float_bits(p_b[i].ch[c][j]);
            }
        }
    }
    return n;
}

double angle_deg(double p_ax, double p_ay, double p_az, double p_bx, double p_by, double p_bz) {
    const double cx = p_ay * p_bz - p_az * p_by, cy = p_az * p_bx - p_ax * p_bz, cz = p_ax * p_by - p_ay * p_bx;
    return ov::rad_to_deg(std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), p_ax * p_bx + p_ay * p_by + p_az * p_bz));
}

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

TEST_CASE("Cascade: random chunks are odd multiples of 2^L under 2^40, the same for the same seed") {
    int64_t bad = 0;
    for (const int lod : { 0, 5, 20 }) {
        for (int i = 0; i < 64; i++) {
            const ChunkId id = random_chunk(lod, 3, i);
            for (const int64_t a : { id.x, id.y, id.z }) {
                bad += (a >> lod) % 2 == 0 || a % (int64_t(1) << lod) != 0 || std::llabs(a) >= (int64_t(1) << 40);
            }
            bad += !(random_chunk(lod, 3, i) == id);
        }
    }
    CHECK(bad == 0);
}

// Below its finest note a program is carried by H alone: at the points a level shares with the one above, its channels
// are the parent's copied, the slopes halved (the units of the finer spacing), bit for bit.
TEST_CASE("Cascade: a level below the notes is its parent's Hermite copy at their shared points, bit for bit") {
    Terrain::LevelProgram prog;
    CHECK_OR_RETURN(compiled(fbm(8, 14, 21, -3), prog)); // levels 5 .. 11
    const ChunkParam param;
    std::vector<Terrain::Values> levels;
    Terrain::run(prog, chunk_box(random_chunk(0, 77, 0), param), 0, &levels);
    int64_t checked = 0, differing = 0;
    for (int m = 0; m < 5; m++) {
        const Terrain::Values *fine = find_level(levels, m);
        const Terrain::Values *coarse = find_level(levels, m + 1);
        CHECK_OR_RETURN((fine != nullptr && coarse != nullptr && fine->has(prog.output) && coarse->has(prog.output)));
        for_each_point(fine->box, [&](const Vector3i64 &p_p, int64_t p_i) {
            if (((p_p.x | p_p.y | p_p.z) & 1) != 0) {
                return;
            }
            float got[CHANNELS], parent[CHANNELS];
            fine->channels(prog.output, p_i, got);
            coarse->channels(prog.output, coarse->box.index(Vector3i64(p_p.x / 2, p_p.y / 2, p_p.z / 2)), parent);
            for (int c = 0; c < CHANNELS; c++) {
                differing += float_bits(got[c]) != float_bits(flush_value(parent[c] * exact_pow2(-channel_order(c))));
            }
            checked++;
        });
    }
    CHECK(checked > 16 * 16 * 16);
    CHECK(differing == 0);
}

// Below the finest note the cascade is the Hermite interpolant; its carried slopes are the derivatives of its values.
TEST_CASE("Cascade: carried slopes agree with finite differences of the carried values") {
    Terrain::LevelProgram prog;
    CHECK_OR_RETURN(compiled(fbm(8, 12, 0x51D3, -3), prog)); // wavelengths >= 256 voxels: Λ_0 to Λ_4 only carry
    const Box box{ { 1000, -300, 5000 }, { 1039, -261, 5039 } };
    const Box inner{ { 1001, -299, 5001 }, { 1038, -262, 5038 } };
    const Terrain::Values v = Terrain::run(prog, box, 0);
    const auto at = [&](int p_c, const Vector3i64 &p_p) {
        float ch[CHANNELS];
        v.channels(prog.output, box.index(p_p), ch);
        return double(ch[p_c]);
    };
    double worst = 0.0;
    double scale = 0.0;
    const Vector3i64 e[3] = { Vector3i64(1, 0, 0), Vector3i64(0, 1, 0), Vector3i64(0, 0, 1) };
    for_each_point(inner, [&](const Vector3i64 &p_p, int64_t) {
        for (int a = 0; a < 3; a++) {
            const double fd = (at(VALUE, p_p + e[a]) - at(VALUE, p_p - e[a])) * 0.5;
            worst = std::max(worst, std::fabs(fd - at(DX + a, p_p)));
            scale = std::max(scale, std::fabs(at(DX + a, p_p)));
        }
    });
    MESSAGE("max |FD - slope| = ", worst, " (max |slope| ", scale, ")");
    // Tolerance: 1% of the largest slope (Hermite FD error over 1/32 of a cell plus float rounding).
    CHECK(worst <= 0.01 * scale);
}

// One octave of amplitude 1 voxel inserted 8 samples per wavelength (k = 7 on Λ_4), carried to Λ_0, against perlin3
// there: values within 1% of the amplitude, the normals of the same octave at steepness 1/8 within 1 degree (fixed
// before measuring).
TEST_CASE("Cascade: fidelity of a cascaded octave, 8 samples per wavelength") {
    constexpr int K = 7;
    double value_max = 0.0, normal_max = 0.0;
    int64_t points = 0;
    for (int b = 0; b < 4; b++) {
        Terrain::LevelProgram prog;
        CHECK_OR_RETURN(compiled(fbm(K, K, 0xF1 + b, -K), prog));
        const Terrain::Node *note = nullptr;
        for (const Terrain::Node &n : prog.nodes) {
            note = n.op == Terrain::Op::NOTE ? &n : note;
        }
        CHECK_OR_RETURN(note != nullptr);
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ 0xF1u, uint32_t(b), 0xF1DEu });
        const int64_t base = b == 3 ? (int64_t(1) << 50) : 0;
        Box box;
        const uint32_t r[3] = { h.x, h.y, h.z };
        for (int a = 0; a < 3; a++) {
            box.lo[a] = base + int64_t(r[a] % (1u << 30)) - (int64_t(1) << 29);
            box.hi[a] = box.lo[a] + 47;
        }
        const Terrain::Values v = Terrain::run(prog, box, 0);
        const double steep = exact_pow2(K - 3); // the octave at amplitude 1/8 of its wavelength
        for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
            WorldPos pos;
            pos.voxel = p_p;
            const ExactNoise::Float4 a = ExactNoise::perlin3(pos, K, note->band.seed, note->band.offset);
            float c[CHANNELS];
            v.channels(prog.output, p_i, c);
            value_max = std::max(value_max, std::fabs(double(c[VALUE]) - double(a.w)));
            // The height field's normals (y - h): slopes per voxel, perlin3's per cell of 2^K voxels.
            const double to_voxel = exact_pow2(-K);
            normal_max = std::max(normal_max,
                    angle_deg(-steep * c[DX], 1.0 - steep * c[DY], -steep * c[DZ], -steep * a.x * to_voxel, 1.0 - steep * a.y * to_voxel, -steep * a.z * to_voxel));
            points++;
        });
    }
    MESSAGE("points ", points, " value max ", value_max, " normal max (r = 1/8) ", normal_max, " deg");
    CHECK(points == 4 * 48 * 48 * 48);
    CHECK(value_max <= 0.01);
    CHECK(normal_max <= 1.0);
}

TEST_CASE("Cascade: no denormal at any level, and flushing denormals (FTZ/DAZ) changes no bit") {
    const ChunkParam param;
#ifdef CASCADE_TEST_MXCSR
    // The reference runs in IEEE mode (denormals kept), whatever the engine set.
    const unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr & ~MXCSR_FTZ_DAZ);
#endif
    // One note carried 59 levels down (its slopes shrink by 2^-1 per level), and the template's octaves.
    const Box box = chunk_box(random_chunk(0, 5, 0), param);
    const ov::Value tables[] = { fbm(62, 62, 0xDEEB, -60), fbm(3, 19, 0xEA27, -3) };
    for (const ov::Value &t : tables) {
        Terrain::LevelProgram prog;
        CHECK_OR_RETURN(compiled(t, prog));
        std::vector<Terrain::Values> ieee;
        Terrain::run(prog, box, 0, &ieee);
        CHECK(subnormals(ieee) == 0);
#ifdef CASCADE_TEST_MXCSR
        _mm_setcsr(csr | MXCSR_FTZ_DAZ);
        std::vector<Terrain::Values> ftz;
        Terrain::run(prog, box, 0, &ftz);
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
