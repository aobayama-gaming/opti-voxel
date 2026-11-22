#include "test_climate_common.hpp"

#include "godot_conv.hpp"

#include "cascade.hpp"
#include "cascade_glsl.hpp"
#include "cascade_terrain.hpp"
#include "gpu_frame.hpp"
#include "sdf_cascade.hpp"

using namespace Cascade;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

// Lattice boxes around 0, the 2^32 wrap and +-2^60, both parities, sizes 1 to 32.
std::vector<Box> test_boxes() {
    const int64_t bases[] = { 0, -1, 7, (int64_t(1) << 32) - 9, -(int64_t(1) << 32) + 3, (int64_t(1) << 60) - 77, -(int64_t(1) << 60) + 5 };
    std::vector<Box> boxes;
    int i = 0;
    for (const int64_t b : bases) {
        for (const int n : { 1, 2, 3, 17, 32 }) {
            Box box;
            for (int a = 0; a < 3; a++) {
                box.lo[a] = b + (a * 3 + i) % 5 - 2;
                box.hi[a] = box.lo[a] + n - 1;
            }
            boxes.push_back(box);
            i++;
        }
    }
    return boxes;
}

} // namespace

TEST_CASE("Cascade: closed-form level boxes equal the iterated parent boxes, within the per-level bound") {
    int checked = 0;
    for (const Box &box : test_boxes()) {
        const std::vector<Box> iterated = level_boxes(box, 0, 40);
        for (int j = 0; j <= 40; j++) {
            const Box closed = level_box(box, j);
            for (int a = 0; a < 3; a++) {
                CHECK(closed.lo[a] == iterated[size_t(j)].lo[a]);
                CHECK(closed.hi[a] == iterated[size_t(j)].hi[a]);
                CHECK(closed.size(a) <= level_bound(int(box.size(a)), j));
            }
            checked++;
        }
    }
    CHECK(checked == 35 * 41);
    // Chunk boxes: the bound is reached (both parities occur), 32 -> 17 -> 10 -> 6 -> 4 -> 3 -> 3.
    const int expect[] = { 32, 17, 10, 6, 4, 3, 3, 3 };
    for (int j = 0; j < 8; j++) {
        CHECK(level_bound(32, j) == expect[j]);
    }
    const ChunkParam param;
    int reached[8] = {};
    for (int i = 0; i < 64; i++) {
        const Box c = chunk_box(random_chunk(i % 7, 9, i), param);
        for (int j = 0; j < 8; j++) {
            const Box l = level_box(c, j);
            for (int a = 0; a < 3; a++) {
                reached[j] = MAX(reached[j], int(l.size(a)));
            }
        }
    }
    for (int j = 0; j < 8; j++) {
        CAPTURE(j);
        CHECK(reached[j] == expect[j]);
    }
}

// SDFCascade::evaluate is Terrain::point + height_sdf over its base + metres, bit for bit, on lattice points of every LOD.
TEST_CASE("SDFCascade: evaluate equals Terrain::point and height_sdf, bit for bit") {
    for (const int lods : { 15, 25 }) {
        ChunkParam param;
        GroundSDF sdf;
        sdf.configure(param, lods);
        const Terrain::LevelProgram &prog = sdf.program();
        // The template ground at 0.5 m (ground.biomes.json, written at 5 cm): its 26.2 km note on k = 16 (32.8 km), whatever
        // the LOD count past it; its notes of 0.4 .. 1.6 m are finer than 8 voxels and play nothing: k = 3 .. 16.
        const int k_top = 16;
        CHECK(prog.top == k_top - 3);
        CHECK(ClimateTest::note_count(prog) == k_top - 2);
        CHECK(ClimateTest::note_top_k(prog) == k_top);
        // Its top note: 3.3 km, 6553.6 voxels at 0.5 m.
        CHECK(ClimateTest::note_amp(prog) == float(3276.8 / 0.5));
        int compared = 0;
        int differing = 0;
        const int64_t bases[] = { 0, 600, (int64_t(1) << 32) - 3, -(int64_t(1) << 60) + 1000, (int64_t(1) << 60) - 4000 };
        for (const int64_t base : bases) {
            for (int lod = 0; lod < lods; lod += 3) {
                const int64_t step = int64_t(1) << lod;
                const Vector3i64 lattice(WorldCoord::floor_div(base, step) + 1, WorldCoord::floor_div(600, step) - 2, -WorldCoord::floor_div(base, step) - 5);
                WorldPos pos;
                pos.voxel = lattice * step;
                const float got = sdf.evaluate(pos, lod);
                const ExactNoise::Float4 g = Terrain::point(prog, lattice, lod).ground;
                const float ch[CHANNELS] = { g.w, g.x, g.y, g.z };
                const ExactNoise::Float4 h = CascadeTerrain::height_sdf(ch, lod, sdf.get_base().at(pos.voxel, lod));
                const float want = h.w * (param.voxel_size * exact_pow2(lod));
                compared++;
                differing += float_bits(got) != float_bits(want);
            }
        }
        CHECK(compared == 5 * ((lods + 2) / 3));
        CHECK(differing == 0);
    }
}

TEST_CASE("SDFCascade: the shader stub and get_program") {
    ChunkParam param;
    GroundSDF sdf;
    sdf.configure(param, 25);
    CHECK(sdf.get_program() == &sdf.program());
    Cascade::ShaderParams params;
    const String glsl = godot_str(sdf.get_glsl_code(params));
    CHECK(glsl.contains("vec4 cascade_terrain_sdf(vec4 c0, WorldPos p, int lod)"));
    CHECK(!glsl.contains("DATUM_LO"));
}

// The shapes gallery's ground (worlds/shapes.world.json, its edits left out): a constant node, the plane of the datum.
TEST_CASE("SDFCascade: a constant ground is the plane y = datum at every LOD") {
    WorldFile world;
    CHECK(WorldFile::load("res://worlds/shapes.world.json", world) == ov::str());
    world.edits.clear();
    ChunkParam param;
    SDFCascade sdf;
    CHECK(sdf.configure_world(world, param) == ov::str());
    CHECK(sdf.current()->empty());
    const int64_t ground = std::llround(BaseSurface::DATUM_M / double(param.voxel_size));
    CHECK(sdf.ground_at(12345, -6789).y == ground);
    for (int lod = 0; lod < 15; lod++) {
        const int64_t step = int64_t(1) << lod;
        for (int i = -4; i <= 4; i++) {
            WorldPos p;
            p.voxel = VoxelPos(step * (1000 + 37 * i), WorldCoord::floor_div(ground, step) * step + step * i, step * (-300 + 11 * i));
            const double want = double(p.voxel.y - ground) * double(param.voxel_size);
            CHECK(double(sdf.evaluate(p, lod)) == doctest::Approx(want).epsilon(1e-6));
        }
    }
}

// The template ground stops at its first note (26.2 km): past 15 LODs at 5 cm the ground is the same octaves, so the
// same bits. One file at every voxel size: at 0.5 m its notes finer than 8 voxels play nothing.
TEST_CASE("SDFCascade: the template ground stops at its top, the same at any LOD count past it, one file at any voxel") {
    // The file's first note is its top: 2^19 voxels at 5 cm.
    const ov::Value graph = ClimateTest::template_graph();
    CHECK(graph["nodes"][size_t(0)]["wavelength_m"].as_double() == 26214.4);
    ChunkParam coarse;
    coarse.voxel_size = 0.5f;
    GroundSDF s50;
    s50.configure(coarse, 15);
    CHECK(s50.get_world_error() == ov::str());
    CHECK(ClimateTest::note_top_k(s50.program()) == 16);
    CHECK(ClimateTest::note_count(s50.program()) == 14);
    ChunkParam param;
    param.voxel_size = 0.05f;
    auto top = [&](const SDFCascade &p_sdf) { return ClimateTest::note_top_k(p_sdf.program()); };
    GroundSDF s15, s25;
    s15.configure(param, 15);
    s25.configure(param, 25);
    CHECK(top(s15) == 19);
    CHECK(top(s25) == 19);
    CHECK(s25.climate_ground().k_max == 29);
    for (int lod = 0; lod < 15; lod += 2) {
        const int64_t step = int64_t(1) << lod;
        for (int i = -3; i <= 3; i++) {
            const VoxelPos v = VoxelPos(int64_t(123457) + 9001 * i, WorldCoord::floor_div(s15.get_datum_y(), step) + 3 * i, -77 * i) * step;
            float a[CHANNELS], b[CHANNELS];
            s15.channels_at(v, lod, a);
            s25.channels_at(v, lod, b);
            for (int c = 0; c < CHANNELS; c++) {
                CHECK(float_bits(a[c]) == float_bits(b[c]));
            }
        }
    }
}

// A world names its ground: without biomes there is no program, the error says how to give it one (before, the
// template ground stood in).
TEST_CASE("SDFCascade: a world without biomes is refused") {
    ChunkParam param;
    SDFCascade sdf;
    sdf.configure(param, 15);
    CHECK(ov::contains(sdf.get_world_error(), "a world names its ground"));
    CHECK(ov::contains(sdf.get_world_error(), WorldFile::GROUND_BIOMES));
    CHECK(sdf.program().nodes.empty());
    sdf.climate_data = ClimateTest::json(R"({"fields": [{"name": "t"}]})");
    sdf.configure(param, 15);
    CHECK(ov::contains(sdf.get_world_error(), "a world names its ground"));
}

TEST_CASE("Terrain: the band table holds every note in note order") {
    ChunkParam param;
    GroundSDF sdf;
    sdf.configure(param, 25);
    const Terrain::LevelProgram &prog = sdf.program();
    const PackedByteArray table = godot_bytes(Terrain::band_table(prog));
    CHECK(table.size() == ClimateTest::note_count(prog) * int64_t(CascadeGLSL::BAND_BYTES));
    const std::vector<int> order = Terrain::note_order(prog);
    CHECK(int64_t(order.size()) == ClimateTest::note_count(prog));
    int64_t base = 0;
    for (const int i : order) {
        const Terrain::Node &n = prog.nodes[size_t(i)];
        CHECK(n.op == Terrain::Op::NOTE);
        CHECK(table.decode_u32(base + 48) == 1u);
        CHECK(table.decode_u32(base + 52) == n.band.seed);
        CHECK(table.decode_u32(base + 56) == uint32_t(n.band.k));
        CHECK(table.decode_u32(base + 16) == WorldCoord::lo32(n.band.offset.x));
        CHECK(table.decode_s32(base + 40) == WorldCoord::hi32(n.band.offset.z));
        base += CascadeGLSL::BAND_BYTES;
    }
}

// The threaded batch (the CPU discoverer's corners) is evaluate() point by point, bit for bit.
TEST_CASE("SDFCascade: evaluate_batch equals evaluate, bit for bit") {
    ChunkParam param;
    GroundSDF sdf;
    sdf.configure(param, 25);
    for (const int lod : { 0, 5, 13, 24 }) {
        const int64_t corner = int64_t(param.half_chunk_voxels() * 2) << lod;
        std::vector<VoxelPos> points;
        for (int i = 0; i < 4000; i++) {
            const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(lod), uint32_t(i), 0xBA7Cu });
            points.push_back(VoxelPos(int64_t(h.x % 61u) - 30, int64_t(h.y % 7u) - 3, int64_t(h.z % 61u) - 30) * corner);
        }
        std::vector<float> batch(points.size());
        sdf.evaluate_batch(points.data(), int(points.size()), lod, batch.data());
        int differing = 0;
        for (size_t i = 0; i < points.size(); i++) {
            WorldPos pos;
            pos.voxel = points[i];
            differing += float_bits(sdf.evaluate(pos, lod)) != float_bits(batch[i]) ? 1 : 0;
        }
        CAPTURE(lod);
        CHECK(differing == 0);
    }
}

// A point item carries its lattice point's words and its LOD as they are: any point of Λ_L, odd indices included (the
// site walk's readings and the discoverer's corners).
TEST_CASE("CascadeGLSL: point items carry any lattice point and its LOD") {
    uint8_t slot[GPUFrame::CHUNK_INPUT_BYTES];
    const int64_t bases[] = { 0, -(int64_t(1) << 32) - 1, (int64_t(1) << 60) - 99999, -(int64_t(1) << 60) + 777 };
    for (const int lod : { 0, 1, 5, 24 }) {
        for (const int64_t base : bases) {
            const VoxelPos p = SDFBase::ground_lattice(VoxelPos(base, base + 3, -base), lod) + VoxelPos(int64_t(1) << lod, 0, 0);
            CHECK(GPUFrame::encode_point_item(slot, p, lod));
            CHECK(ov::get_u32(slot + GPUFrame::POINT_LOD_AT) == uint32_t(lod));
            for (int a = 0; a < 3; a++) {
                CAPTURE(lod);
                CAPTURE(a);
                CHECK(ov::get_u32(slot + 4 * a) == WorldCoord::lo32(p[a]));
                CHECK(ov::get_s32(slot + 16 + 4 * a) == WorldCoord::hi32(p[a]));
            }
        }
    }
    // Off its lattice, or a LOD the GPU can't read: refused (the GPU would read another point).
    CHECK(!GPUFrame::encode_point_item(slot, VoxelPos(32, 33, 32), 5));
    CHECK(!GPUFrame::encode_point_item(slot, VoxelPos(-1, 0, 0), 1));
    CHECK(!GPUFrame::encode_point_item(slot, VoxelPos(0, 0, 0), -1));
    CHECK(!GPUFrame::encode_point_item(slot, VoxelPos(0, 0, 0), ChunkMath::MAX_LOD_NUMBERS));
    CHECK(GPUFrame::encode_point_item(slot, VoxelPos(-32, 32, 0), 5));
}

// ground_field is its three steps (the GPU's point items read the middle one): the nearest lattice point (ties up), the
// unit register there, the extension along its gradient; and evaluate_cull is that register in metres.
TEST_CASE("SDFCascade: ground_field and evaluate_cull are made of ground_unit") {
    GroundSDF sdf;
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 25);
    CHECK(SDFBase::ground_lattice(VoxelPos(15, 16, 17), 5) == VoxelPos(0, 32, 32));
    CHECK(SDFBase::ground_lattice(VoxelPos(-17, -16, -15), 5) == VoxelPos(-32, 0, 0));
    for (const int lod : { 0, 3, 5, 8 }) {
        for (int i = 0; i < 64; i++) {
            const VoxelPos v(ExactNoise::random_offset(uint32_t(lod), uint32_t(i), 1, 1 << 20), 6000 + ExactNoise::random_offset(uint32_t(lod), uint32_t(i), 2, 256),
                    ExactNoise::random_offset(uint32_t(lod), uint32_t(i), 3, 1 << 20));
            const VoxelPos at = SDFBase::ground_lattice(v, lod);
            const ExactNoise::Float4 u = sdf.ground_unit(at, lod);
            CHECK(ExactNoise::same_bits(sdf.ground_field(v, lod), SDFBase::extend_field(u, at, v, lod)));
            CHECK(ExactNoise::float_bits(sdf.evaluate_cull(WorldPos{ at, Float3() }, lod)) == ExactNoise::float_bits(SDFCascade::to_metres(u, lod, param.voxel_size).w));
        }
    }
}

// The point pass: the level template with the point fill (frame from the item, the unit register out); the chunk pass's
// text is unchanged (pinned in test_climate_shape.cpp).
TEST_CASE("CascadeGLSL: the point pass writes the unit register from the item's own frame") {
    GroundSDF sdf;
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 25);
    const String code = godot_str(CascadeGLSL::level_shader(&sdf, false, true));
    CHECK(code.contains("point_value[sample_index] = r;"));
    CHECK(code.contains("cascade_terrain_unit("));
    CHECK(code.contains("ChunkFrame f = point_frame(input_chunk[pc.chunk_base + c]);"));
    CHECK(code.contains("ChunkFrame point_frame(ChunkInput c)"));
    CHECK(!code.contains("/*ITEM_FRAME*/"));
    const String chunks = godot_str(CascadeGLSL::level_shader(&sdf, false, false));
    CHECK(chunks.contains("ChunkFrame f = chunk_frame(input_chunk[pc.chunk_base + c], chunk_param);"));
    CHECK(!chunks.contains("point_frame"));
}

// The default terrain's bits, pinned: every register component of the bare cascade (sdf, through the ground's base
// surface) and evaluate() at lattice points near the ground, at 0.5 m and 5 cm, near 0 and +-2^60.
// The hash was taken on dev before the base surface (PR "base surface"): a change of formula that moves one bit fails.
TEST_CASE("SDFCascade: the default terrain's bits are pinned") {
    uint64_t hash = 0xCBF29CE484222325ull;
    auto mix = [&hash](float p_v) {
        hash = (hash ^ float_bits(p_v)) * 0x100000001B3ull;
    };
    int samples = 0;
    for (const float voxel : { 0.5f, 0.05f }) {
        ChunkParam param;
        param.voxel_size = voxel;
        GroundSDF sdf;
        sdf.configure(param, 25);
        const int64_t datum = sdf.get_datum_y();
        for (const int64_t base : { int64_t(0), int64_t(12345), (int64_t(1) << 60) - 4000, -(int64_t(1) << 60) + 77 }) {
            for (int lod = 0; lod < 25; lod += 2) {
                const int64_t step = int64_t(1) << lod;
                for (int i = -3; i <= 3; i++) {
                    WorldPos pos;
                    pos.voxel = VoxelPos(WorldCoord::floor_div(base, step) + i, WorldCoord::floor_div(datum, step) + 5 * i, 2 - i) * step;
                    float ch[CHANNELS];
                    sdf.channels_at(pos.voxel, lod, ch);
                    const ExactNoise::Float4 r = sdf.sdf(ch, pos.voxel, lod);
                    mix(r.x);
                    mix(r.y);
                    mix(r.z);
                    mix(r.w);
                    mix(sdf.evaluate(pos, lod));
                    samples++;
                }
            }
        }
    }
    CHECK(samples == 2 * 4 * 13 * 7);
    // Re-pinned when the 0.5 m ground became the template file's notes (engine cleanup 2/5: k = 3 .. 16, no longer the
    // 50 cm file's 0.8x amplitudes) and the flat preset's half left (4/5: the gallery's ground is a world file's).
    CHECK(hash == 5750492863826811759ull);
}
