#include "test_climate_common.hpp"

#include "cascade_glsl.hpp"
#include "climate.hpp"
#include "sdf_cascade.hpp"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

using namespace Climate;
using ClimateTest::ClimateSDF;
using ExactNoise::float_bits;

namespace {

uint64_t fnv_text(const ov::str &p_s) {
    uint64_t h = ExactNoise::FNV64_BASIS;
    for (const char c : p_s) {
        h = (h ^ uint8_t(c)) * 1099511628211ull;
    }
    return h;
}

} // namespace

// The climate before shaped bands (adapter/godot/project/climate/plain.json, a copy of the example as it was) keeps its bits: every
// slot of every point query, the SDF of the ground, the skirts, and the level shaders' text. A table without the new
// keys is the old code path, op for op. Re-pinned when a run's last level began reading its climate from the level
// above's derive (Cascade::Modulation).
TEST_CASE("Climate shape: a table without shapes keeps the bits it had before them") {
    ClimateSDF c("res://climate/plain.json", 15);
    CHECK(c.sdf.get_climate() != nullptr);
    const Cascade::Spec &spec = c.sdf.get_spec();
    uint64_t h = ExactNoise::FNV64_BASIS;
    for (const int lod : { 0, 3, 8, 13 }) {
        for (int i = 0; i < 48; i++) {
            const Vector3i64 p(int64_t(i) * 7919 - 50000, (c.sdf.get_datum_y() >> lod) + 3 - i, int64_t(i) * -104729 + (int64_t(1) << 33));
            float slots[Cascade::SLOTS];
            Cascade::point_slots(spec, p, lod, slots, true);
            // The carried slots only: the pin doesn't depend on the group caps.
            for (int s = 0; s < Cascade::field_slot(int(spec.fields.size())); s++) {
                h = ExactNoise::fnv64(h, float_bits(slots[s]));
            }
            for (int s = Cascade::carry_slot(0); s < Cascade::carry_slot(spec.carry_groups()); s++) {
                h = ExactNoise::fnv64(h, float_bits(slots[s]));
            }
            const WorldPos pos{ VoxelPos(p.x << lod, p.y << lod, p.z << lod), Float3() };
            h = ExactNoise::fnv64(h, float_bits(c.sdf.evaluate(pos, lod)));
        }
    }
    for (int l = 0; l < 15; l++) {
        h = ExactNoise::fnv64(h, float_bits(c.sdf.skirt_depth_m(l)));
    }
    const uint64_t chunk_text = fnv_text(CascadeGLSL::level_shader(&c.sdf, false, false, false, true, CascadeGLSL::View::SAMPLES));
    const uint64_t point_text = fnv_text(CascadeGLSL::level_shader(&c.sdf, false, true));
    MESSAGE("plain climate: bits ", h, ", chunk shader ", chunk_text, ", point shader ", point_text);
    CHECK(h == 13659167188471627091ull); // re-pinned when the ground's octave k got the seed make_octave(seed, k)
    // The shader texts (the modulation's arrays sized by the spec's group count, the ground over its base surface). Both
    // re-pinned when the climate's numbers moved to the shader params (Cascade::ShaderParams, a uniform block: each value put
    // back as its literal gave the old texts' hashes), when the skirt depths joined them (they follow the amplitudes), and when
    // the plain amplitudes' text folded into the shaped one (cascade_mods, insert_band_shaped: the bits kept). Both again
    // when the per-LOD switches became param tables (level_table) and a level's extra notes data (its ground entry), and
    // when the multifractal state left BandMod (a parent's field instead).
    CHECK(chunk_text == 8193419427396817385ull);
    // The point shader re-pinned when point items became any lattice point with its unit register (point_frame, 16 B),
    // and (all three) when the derive moved before the level's insertions (one climate per point).
    // The bits and the point shader re-pinned when the biomes became a centre and span per field (Climate::weights).
    CHECK(point_text == 9069992480234452530ull);
}

// Free graphs (graph_test.json) as the relaxer plays them: chained's warp after its shape moves before it (one chain per
// octave) and its hills' second shape reads their sum at the samples (a sub-mix); shaped_mix shapes its two notes' mix
// at the samples (on the ground); caved's mix takes a shaped sub-mix of two notes at volume 0.5 (a sum at the samples).
// The sub-mixes share one field (no octave in common); every SDF path gives the point query's bits, within the programs'
// bounds. A multifractal past a mix, a warp past a shape at the samples: why not.
TEST_CASE("Climate shape: the relaxer plays free graphs: chains per octave, sub-mixes and shapes at the samples") {
    ClimateSDF c("res://climate/graph_test.json", 15);
    CHECK(c.sdf.get_climate_error() == ov::str());
    CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
    const Table &t = *c.sdf.get_climate();
    CLIMATE_REQUIRE(t.biomes.size() == 3);
    CLIMATE_REQUIRE(t.fields.size() == 20); // t, the warp's 6 octaves' x, y, z, the sub-mix
    const int sub = 19;
    CHECK(std::all_of(t.fields.begin() + 1, t.fields.begin() + sub, [](const Field &p_f) { return p_f.warp && p_f.bands.size() == 1; }));
    CHECK(t.fields[sub].channel);
    CHECK(t.fields[sub].bands.size() == 3); // the hills' octave, the caves' two
    auto ops = [](const SampleProgram &p_p) {
        std::vector<SampleOp> o;
        for (const SampleInstr &in : p_p.code) {
            o.push_back(in.op);
        }
        return o;
    };
    using O = SampleOp;
    CHECK(ops(t.biomes[0].program) == std::vector<O>{ O::FIELD, O::SHAPE, O::GROUND, O::CONST, O::MUL, O::ADD });
    CHECK(ops(t.biomes[1].program) == std::vector<O>{ O::GROUND, O::SHAPE });
    CHECK(ops(t.biomes[2].program) == std::vector<O>{ O::FIELD, O::SHAPE, O::GROUND, O::CONST, O::MUL, O::ADD });
    CHECK(t.biomes[0].program.code[0].index == sub);
    CHECK(t.biomes[2].program.code[0].index == sub);
    const std::vector<Amp> &chained = t.biomes[0].amps;
    CHECK(std::any_of(chained.begin(), chained.end(), [](const Amp &p_a) { return p_a.group == 0 && p_a.has_shape && !p_a.warps.empty() && p_a.amp > 0.0f; }));
    CHECK(std::any_of(chained.begin(), chained.end(), [&](const Amp &p_a) { return p_a.group == sub + 1 && p_a.has_shape && p_a.amp > 0.0f; }));
    const SampleCarry carry = sample_carry(t);
    CHECK(carry.field[sub] >= 0);
    double bound = 0.0;
    for (const Climate::Biome &b : t.biomes) {
        CHECK(b.program.bound > 0.0);
        bound = std::max(bound, b.program.bound);
    }
    const Cascade::Spec &spec = c.sdf.get_spec();
    int64_t points = 0, mismatches = 0, over = 0;
    for (const int lod : { 0, 4, 9 }) {
        Cascade::Box box;
        for (int a = 0; a < 3; a++) {
            box.lo[a] = (a == 1 ? (c.sdf.get_datum_y() >> lod) : (int64_t(-77777) >> lod)) - 2;
            box.hi[a] = box.lo[a] + 3;
        }
        const Cascade::Level level = Cascade::run(spec, box, lod, nullptr, true);
        Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
            float slots[Cascade::SLOTS], ch[Cascade::CHANNELS];
            level.slots_at(p_i, slots);
            c.sdf.ground_channels(slots, p_p * (int64_t(1) << lod), lod, ch);
            const VoxelPos v(p_p.x << lod, p_p.y << lod, p_p.z << lod);
            points++;
            mismatches += float_bits(c.sdf.sdf(ch, v, lod).w) != float_bits(c.sdf.evaluate(WorldPos{ v, Float3() }, lod));
            over += std::fabs(double(ch[Cascade::VALUE]) - double(slots[Cascade::VALUE])) > bound * (1.0 + 1e-6);
        });
    }
    CHECK(points == 192);
    CHECK(mismatches == 0);
    CHECK(over == 0);
    ov::Value data;
    CLIMATE_REQUIRE(load_table("res://climate/graph_test.json", data) == ov::str());
    // Refused: a multifractal past a mix (it weighs one note by its parent's height), and a warp of a node past
    // the notes (it moves where notes are read).
    for (const bool warp : { false, true }) {
        ov::Value bad = data;
        ov::Value &nodes = ClimateTest::graph_nodes(bad, 2);
        if (warp) {
            nodes.push_back(ClimateTest::json(R"({"id": "cs0", "kind": "shape", "in": "caves", "shape": "billow"})"));
            nodes.push_back(ClimateTest::json(R"({"id": "cw", "kind": "warp", "in": "cs0", "amount_m": 3})"));
        }
        for (size_t n = 0; n < nodes.size(); n++) {
            if (warp && nodes[n]["kind"].as_str() == "note") {
                nodes.at(n)["warp"] = "cw";
                break;
            }
            if (!warp && nodes[n]["id"].as_str() == "cs") {
                nodes.at(n) = ClimateTest::json(R"({"id": "cs", "kind": "multifractal", "in": "caves", "parent": "caves", "low_m": 0, "high_m": 1})");
            }
        }
        const ClimateSDF broken(bad, 15);
        CHECK(broken.sdf.get_climate() == nullptr);
        CHECK(ov::contains(broken.sdf.get_climate_error(), warp ? "not a node past them" : "a multifractal weighs one note"));
    }
    // Two sub-mixes of one biome sit on two fields even where their octaves don't meet (each read is its own sum), and
    // sub-mix fields past MAX_FIELDS are refused before anything indexes by them.
    ov::Value two = ClimateTest::json(R"({"fields": [], "biomes": [{"name": "b", "graph": {"nodes": [
        {"id": "a", "kind": "note", "wavelength_m": 12.8, "seed": 11, "amp_m": 1},
        {"id": "b", "kind": "note", "wavelength_m": 51.2, "seed": 12, "amp_m": 1},
        {"id": "m", "kind": "min", "a": "a", "b": "b"},
        {"id": "s", "kind": "add", "a": "ground", "b": "m"},
        {"id": "out", "kind": "output", "height": "s"}]}}]})");
    const ov::Value ground = ClimateTest::ground_nodes(0.05);
    for (const ov::Value &n : ground.elements()) {
        ClimateTest::graph_nodes(two, 0).push_back(n);
    }
    const ClimateSDF split(two, 15);
    CHECK(split.sdf.get_climate_error() == ov::str());
    CLIMATE_REQUIRE(split.sdf.get_climate() != nullptr);
    std::vector<int> reads;
    for (const SampleInstr &in : split.sdf.get_climate()->biomes[0].program.code) {
        if (in.op == SampleOp::FIELD) {
            reads.push_back(in.index);
        }
    }
    CHECK(reads.size() == 2);
    CHECK((reads.size() == 2 && reads[0] != reads[1]));
    for (int f = 0; f < MAX_FIELDS; f++) {
        ov::Value field = ClimateTest::json(R"({"bands": [{"from_m": 800, "to_m": 1600, "amp": 0.5}]})");
        field["name"] = ov::format("f%d", int64_t(f));
        two["fields"].push_back(field);
    }
    const ClimateSDF crowded(two, 15);
    CHECK(crowded.sdf.get_climate() == nullptr);
    CHECK(ov::contains(crowded.sdf.get_climate_error(), "sub-mix fields"));
}

// Caves as a graph (caves_test.json): caved's ground is smin(mix, up - C), C = min(N, T) under the band's tent T. Its
// bound is the tent's floor plus the smooth min's dip (affine forms in up - H); below the ground, columns open air only
// within the band (N > 0 there), the ground is never raised, and every SDF path gives the point query's bits. Without
// the tent the term has no bound: refused.
TEST_CASE("Climate shape: caves are a smooth min on the mix's output, bounded by their band") {
    ClimateSDF c("res://climate/caves_test.json", 15);
    CHECK(c.sdf.get_climate_error() == ov::str());
    CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
    const Table &t = *c.sdf.get_climate();
    CLIMATE_REQUIRE(t.biomes.size() == 2);
    const SampleProgram &p = t.biomes[1].program;
    CHECK(p.reads_up());
    CHECK(t.fields.back().channel); // the carve noise: a sub-mix
    const double vs = 0.05, k = 2.0 / vs, top = 4.0 / vs, floor = 60.0 / vs;
    CHECK(p.bound == doctest::Approx(floor + k / 4.0));
    CHECK(p.slope <= MAX_CURVE_SLOPE);
    const Cascade::Spec &spec = c.sdf.get_spec();
    const int64_t datum = c.sdf.get_datum_y();
    const int lod = 2;
    int64_t points = 0, air = 0, outside = 0, raised = 0, over = 0, mismatches = 0;
    for (int col = 0; col < 8; col++) {
        const int64_t x = (int64_t(col) * 7919 - 30000) >> lod, z = (int64_t(col) * -6007 + 11000) >> lod;
        float slots[Cascade::SLOTS], ch[Cascade::CHANNELS];
        Cascade::point_slots(spec, Vector3i64(x, datum >> lod, z), lod, slots, true);
        const int64_t surface = datum + int64_t(slots[Cascade::VALUE]);
        for (int64_t y = (surface - int64_t(floor) - 200) >> lod; y <= (surface + 100) >> lod; y++) {
            const Vector3i64 q(x, y, z);
            const VoxelPos v(q.x << lod, q.y << lod, q.z << lod);
            Cascade::point_slots(spec, q, lod, slots, true);
            c.sdf.ground_channels(slots, v, lod, ch);
            const double bare = slots[Cascade::VALUE], h = ch[Cascade::VALUE], up = double(v.y - datum), depth = bare - up;
            points++;
            raised += h > bare + 1e-6 * (1.0 + std::fabs(bare));
            over += std::fabs(h - bare) > p.bound * (1.0 + 1e-6);
            const bool opened = depth > k && up - h > 0.0;
            air += opened;
            outside += opened && (depth < top - k || depth > floor + k);
            mismatches += float_bits(c.sdf.sdf(ch, v, lod).w) != float_bits(c.sdf.evaluate(WorldPos{ v, Float3() }, lod));
        }
    }
    MESSAGE("caves: ", air, " of ", points, " samples opened below the ground");
    CHECK(points > 2000);
    CHECK(air > 0);
    CHECK(outside == 0);
    CHECK(raised == 0);
    CHECK(over == 0);
    CHECK(mismatches == 0);
    ov::Value bad;
    CLIMATE_REQUIRE(load_table("res://climate/caves_test.json", bad) == ov::str());
    ov::Value &nodes = ClimateTest::graph_nodes(bad, 1);
    for (size_t n = 0; n < nodes.size(); n++) {
        if (nodes[n]["id"].as_str() == "carve") {
            nodes.at(n)["b"] = "n"; // C = N everywhere: up - C as deep as up goes
        }
    }
    const ClimateSDF unbounded(bad, 15);
    CHECK(unbounded.sdf.get_climate() == nullptr);
    CHECK(ov::contains(unbounded.sdf.get_climate_error(), "without bound"));
}

// The shapes' centring constants are the noise's own means: E[sabs(n)] and E[n^2] of perlin3, over random points of
// random cells.
// A generic capacity: a power of two, at least the floor, at most the max, and never under the need (a generic carry
// clamped to MAX_CARRY under its need overran the cascade's slots).
TEST_CASE("Climate shape: cap_for holds the need") {
    CHECK(cap_for(5, 0, 24) == 8);
    CHECK(cap_for(5, 16, 24) == 16);
    CHECK(cap_for(0, 4, 24) == 4);
    CHECK(cap_for(20, 0, 24) == 24);
    CHECK(cap_for(30, 0, 24) == 30);
}

TEST_CASE("Climate shape: NOISE_ABS_MEAN and NOISE_SQ_MEAN are the noise's means") {
    double abs_sum = 0.0, sq_sum = 0.0;
    const int n = 1 << 20;
    for (int i = 0; i < n; i++) {
        const ExactNoise::UVec3 h = ExactNoise::hash3(uint32_t(i), 0x51EDu, 0xC0FFEEu);
        const WorldPos p{ VoxelPos(int64_t(h.x % 100000u) - 50000, int64_t(h.y % 100000u) - 50000, int64_t(h.z % 100000u) - 50000),
            Float3{ float(h.x >> 8) * 0x1p-24f, float(h.y >> 8) * 0x1p-24f, float(h.z >> 8) * 0x1p-24f } };
        const float v = ExactNoise::perlin3(p, 6, 0x1234u + uint32_t(i & 7), Vector3i64()).w;
        abs_sum += std::sqrt(double(v) * v + double(Cascade::SMOOTH_ABS_E) * Cascade::SMOOTH_ABS_E) - Cascade::SMOOTH_ABS_E;
        sq_sum += double(v) * v;
    }
    MESSAGE("E[sabs(n)] ", abs_sum / n, ", E[n^2] ", sq_sum / n);
    CHECK(std::fabs(abs_sum / n - Cascade::NOISE_ABS_MEAN) < 0.01 * Cascade::NOISE_ABS_MEAN);
    CHECK(std::fabs(sq_sum / n - Cascade::NOISE_SQ_MEAN) < 0.01 * Cascade::NOISE_SQ_MEAN);
}

namespace {

// Every operation at once (adapter/godot/project/climate/shaped_test.json, CascadeSpike's set 4s too).
ov::Value shaped_table() {
    ov::Value d;
    CHECK(load_table("res://climate/shaped_test.json", d) == ov::str());
    return d;
}

struct ShapedSDF {
    SDFCascade sdf;
    explicit ShapedSDF(int p_lods = 15, float p_skirt_cells = 1.0f) {
        sdf.climate_data = shaped_table();
        ChunkParam param;
        param.voxel_size = 0.05f;
        param.skirt_depth = p_skirt_cells;
        sdf.configure(param, p_lods);
    }
};

} // namespace

TEST_CASE("Climate shape: a shaped table parses into its groups, box runs equal point queries in every slot") {
    ShapedSDF s;
    CHECK(s.sdf.get_climate_error() == ov::str());
    CLIMATE_REQUIRE(s.sdf.get_climate() != nullptr);
    const Table &t = *s.sdf.get_climate();
    const Cascade::Spec &spec = s.sdf.get_spec();
    CHECK(t.has_programs());
    CHECK(spec.fields.size() == t.fields.size());
    CLIMATE_REQUIRE(spec.modulation != nullptr);
    CHECK(spec.modulation->group_features(0) != 0u);
    // The multifractal biome's octaves weighted by their parents' fields.
    int weighted = 0;
    for (const Climate::Biome &b : t.biomes) {
        for (const Amp &a : b.amps) {
            if (a.multifractal.field >= 0) {
                weighted++;
                CHECK(t.fields[size_t(a.multifractal.field)].warp);
                CHECK(a.multifractal.mul > 0.0f);
                CHECK(a.multifractal.add == doctest::Approx(0.5f));
            }
        }
    }
    CHECK(weighted > 8);
    int64_t bad = 0;
    for (const int lod : { 0, 3, 8 }) {
        for (const int64_t base : { int64_t(5), (int64_t(1) << 32) - 2, -(int64_t(1) << 44) }) {
            bad += ClimateTest::box_point_mismatches(spec, Vector3i64(base, (s.sdf.get_datum_y() >> lod) - 1, base / 3), 3, lod, true);
        }
    }
    CHECK(bad == 0);
}

TEST_CASE("Climate shape: ground terms reach every SDF path, the samples' and the points'") {
    ShapedSDF s;
    CLIMATE_REQUIRE(s.sdf.get_climate() != nullptr);
    const Cascade::Spec &spec = s.sdf.get_spec();
    int64_t points = 0, mismatches = 0, lifted = 0;
    for (const int lod : { 0, 4, 9 }) {
        Cascade::Box box;
        for (int a = 0; a < 3; a++) {
            box.lo[a] = (a == 1 ? (s.sdf.get_datum_y() >> lod) : (int64_t(123457) >> lod)) - 2;
            box.hi[a] = box.lo[a] + 3;
        }
        const Cascade::Level level = Cascade::run(spec, box, lod, nullptr, true);
        Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
            float slots[Cascade::SLOTS], ch[Cascade::CHANNELS];
            level.slots_at(p_i, slots);
            s.sdf.ground_channels(slots, p_p * (int64_t(1) << lod), lod, ch);
            const VoxelPos v(p_p.x << lod, p_p.y << lod, p_p.z << lod);
            const float want = s.sdf.sdf(ch, v, lod).w;
            points++;
            mismatches += float_bits(want) != float_bits(s.sdf.evaluate(WorldPos{ v, Float3() }, lod));
            lifted += float_bits(ch[Cascade::VALUE]) != float_bits(slots[Cascade::VALUE]);
        });
    }
    CHECK(points == 192);
    CHECK(mismatches == 0);
    CHECK(lifted == points);
    CHECK(CascadeGLSL::field_range(&s.sdf, true).everywhere()); // point items read the fields too
    Cascade::ShaderParams params;
    CHECK(ov::contains(s.sdf.get_ground_glsl(s.sdf.get_spec(), params), "vec4 climate_ground(Ch ch, WorldPos p, int lod)"));
    // The skirts take the curves' step: deeper than the same ground without them (their slope scales the ground's
    // step), the cap (skirt_depth cells) lifted.
    ShapedSDF deep(15, 1e6f);
    const ov::Value no_terms = ClimateTest::without_curves(shaped_table());
    SDFCascade bare;
    bare.climate_data = no_terms;
    ChunkParam param;
    param.voxel_size = 0.05f;
    param.skirt_depth = 1e6f;
    bare.configure(param, 15);
    CLIMATE_REQUIRE(bare.get_climate() != nullptr);
    CHECK(deep.sdf.skirt_depth_m(5) > bare.skirt_depth_m(5));
}

TEST_CASE("Climate shape: curves pass through their points, shapes bound their values") {
    const Curve c = make_curve({ -1.0, 0.0, 0.5, 2.0 }, { 3.0, -1.0, 0.25, 4.0 });
    const double xs[] = { -1.0, 0.0, 0.5 };
    const double ys[] = { 3.0, -1.0, 0.25 };
    for (int i = 0; i < 3; i++) {
        // a = 0 at a segment's start: its s, the point's y.
        CHECK(c.s[size_t(i)] == float(ys[i]));
        CHECK(c.x[size_t(i)] == float(xs[i]));
    }
    CHECK(c.y_last == 4.0f);
    CHECK(curve_slope_bound(c) > 0.0f);
    CHECK(shape_bound(Shape()) == 1.0f);
    Shape ridged;
    ridged.c[0] = 1.0f;
    ridged.c[1] = 0.0f;
    ridged.c[2] = -2.0f;
    ridged.c[3] = 1.0f;
    CHECK(shape_bound(ridged) > 0.5f);
    CHECK(shape_bound(ridged) < 1.0f);
}

TEST_CASE("Climate shape: malformed shapes and terms are refused with the reason") {
    SDFCascade sdf;
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    const Ground g = sdf.climate_ground();
    auto err = [&](const ov::str &p_json) {
        Table t;
        return parse(ClimateTest::json(p_json), g, t);
    };
    // One biome whose graph is note "n" (51.2 m) through p_nodes (each reading the one before), its last p_height.
    auto biome = [](const ov::str &p_nodes, const char *p_height, const ov::str &p_note = ov::str()) {
        return R"("biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 51.2, "seed": 1, "amp_m": 6.4)" + p_note + "}, " +
                p_nodes + R"({"id": "o", "kind": "output", "height": ")" + ov::str(p_height) + "\"}]}}]}";
    };
    const ov::str f1 = R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]}], )";
    CHECK(ov::contains(err(f1 + biome(R"({"id": "s", "kind": "shape", "in": "n", "shape": "spiky"}, )", "s")), "plain, billow"));
    CHECK(ov::contains(err(f1 + biome(R"({"id": "f", "kind": "field", "field": "t"}, {"id": "m", "kind": "multifractal", "in": "f", "parent": "n", "low_m": 0, "high_m": 1}, )", "m")),
            "a multifractal weighs one note"));
    // A multifractal's parent: a plain note strictly coarser than its own, low_m and high_m apart.
    auto mf = [&](const char *p_parent, double p_low, double p_high, const ov::str &p_note = ov::str()) {
        return ov::format(R"({"id": "c", "kind": "note", "wavelength_m": 409.6, "seed": 3, "amp_m": 0.5%s}, {"id": "w", "kind": "warp", "in": "c", "amount_m": 1}, )"
                          R"({"id": "m", "kind": "multifractal", "in": "n", "parent": "%s", "low_m": %s, "high_m": %s}, )",
                p_note, p_parent, ov::num(p_low, 3), ov::num(p_high, 3));
    };
    CHECK(err(f1 + biome(mf("c", -0.25, 0.25), "m")) == ov::str());
    CHECK(err(f1 + biome(mf("c", 0.25, -0.25), "m")) == ov::str());
    CHECK(err(f1 + biome(mf("c", 0, 1), "m")) == ov::str()); // add 0: weight 0 at the parent's zero height
    CHECK(ov::contains(err(f1 + biome(mf("n", -1, 1), "m")), "coarser than its note"));
    CHECK(ov::contains(err(f1 + biome(mf("w", -1, 1), "m")), "parent is a note"));
    CHECK(ov::contains(err(f1 + biome(mf("c", 1, 1), "m")), "differ"));
    CHECK(err(f1 + biome(mf("c", -1, 1, R"(, "stretch_y": 2)"), "m")) == ov::str());
    CHECK(ov::contains(err(f1 + biome(mf("c", -1, 1, R"(, "warp": "w")"), "m")), "without a warp"));
    // A warp is a note's input, its notes coarser than the note (the planner's rule) and plain: the ground's slope (the
    // swiss turbulence's) is not offered, nor its damping (climate.hpp).
    const ov::str coarse = R"({"id": "c", "kind": "note", "wavelength_m": 409.6, "seed": 3, "amp_m": 0.5}, )";
    CHECK(ov::contains(err(f1 + biome(coarse + R"({"id": "w", "kind": "warp", "in": "c", "amount_m": 1}, )", "w")), "a note's input"));
    CHECK(err(f1 + biome(coarse + R"({"id": "w", "kind": "warp", "in": "c", "amount_m": 1}, )", "n", R"(, "warp": "w")")) == ov::str());
    CHECK(ov::contains(err(f1 + biome(R"({"id": "c", "kind": "note", "wavelength_m": 51.2, "seed": 3, "amp_m": 0.5}, {"id": "w", "kind": "warp", "in": "c", "amount_m": 1}, )", "n", R"(, "warp": "w")")),
            "coarser"));
    CHECK(ov::contains(err(f1 + biome(coarse + R"({"id": "r", "kind": "shape", "in": "c", "shape": "ridged"}, {"id": "w", "kind": "warp", "in": "r", "amount_m": 1}, )", "n", R"(, "warp": "w")")),
            "no shape"));
    CHECK(ov::contains(err(f1 + biome(coarse, "n", R"(, "warp": "c")")), "is a warp node"));
    CHECK(ov::contains(err(f1 + biome("", "n", R"(, "damp": 4)")), "unknown key 'damp'"));
    // An offset is a constant term: inserted without its mean, it doesn't count on a note (ridged's scales its |n| too).
    CHECK(ov::contains(err(f1 + biome(R"({"id": "s", "kind": "shape", "in": "n", "shape": "billow", "offset": 0.5}, )", "s")), "doesn't count on a note"));
    CHECK(ov::contains(err(R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 400, "amp": 1, "offset": 0.5}]}], "biomes": [{"name": "a"}]})"),
            "unknown key 'offset'"));
    CHECK(ov::contains(err(f1 + biome(R"({"id": "s", "kind": "shape", "in": "n", "coeffs": [0.5, 1, 0, 0]}, )", "s")), "doesn't count on a note"));
    CHECK(err(f1 + biome(R"({"id": "s", "kind": "shape", "in": "n", "shape": "ridged", "offset": 0.8}, )", "s")) == ov::str());
    CHECK(ov::contains(err(f1 + biome(R"({"id": "s", "kind": "shape", "in": "n", "coeffs": [1, 2]}, )", "s")), "4 numbers"));
}

TEST_CASE("Climate shape: the example world parses at any LOD count, shaped, and box runs equal point queries") {
    for (const int lods : { 15, 25 }) {
        ClimateSDF c("res://climate/example.json", lods);
        CHECK(c.sdf.get_climate_error() == ov::str());
        CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
        const Table &t = *c.sdf.get_climate();
        CHECK(t.has_programs());
        CHECK(c.sdf.has_ground_materials());
        const Cascade::Spec &spec = c.sdf.get_spec();
        CLIMATE_REQUIRE(spec.modulation != nullptr);
        CHECK(spec.modulation->group_features(0) != 0u);
        int64_t bad = 0;
        for (const int lod : { 0, 5, 11 }) {
            bad += ClimateTest::box_point_mismatches(spec, Vector3i64(int64_t(18022 * 20) >> lod, (c.sdf.get_datum_y() >> lod) - 1, int64_t(65536 * 20) >> lod), 3, lod, true);
        }
        CHECK(bad == 0);
    }
}

// Several notes per octave (notes_test.json): octaves 11 .. 13 hold three notes (the template's and two biomes' own), all
// modulated; the band table holds the others after the groups' entries, the level shader inserts them after each level's
// first, and box runs equal point queries in every slot.
TEST_CASE("Climate shape: several notes per octave, modulated, box runs equal point queries") {
    ClimateSDF c("res://climate/notes_test.json", 15);
    CHECK(c.sdf.get_climate_error() == ov::str());
    CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
    const Cascade::Spec &spec = c.sdf.get_spec();
    CHECK(spec.level_notes(11 - 3) == 3);
    CHECK(spec.level_notes(10 - 3) == 2);
    CHECK(spec.level_notes(9 - 3) == 1);
    CHECK(spec.max_level_notes() == 3);
    CLIMATE_REQUIRE(spec.modulation != nullptr);
    CHECK((spec.modulation->modulated(11 - 3) & 1u) != 0u);
    const int extra = 2 * 3 + 1 * 2; // two more notes on octaves 11 .. 13, one more on 10 and 14
    CHECK(CascadeGLSL::band_table(spec).size() == (size_t(spec.group_count()) * CascadeGLSL::BAND_ENTRIES + extra) * CascadeGLSL::BAND_BYTES);
    const ov::str code = CascadeGLSL::level_shader(&c.sdf, false, false, false, true, CascadeGLSL::View::FIELDS);
    CHECK(ov::contains(code, "int count = int(lead.offset_hi.w);")); // the level's extra notes, listed in its ground entry
    CHECK(ov::contains(code, ov::format("BandMod mod[%d]", spec.group_count() + 2)));
    int64_t bad = 0;
    for (const int lod : { 0, 4, 8 }) {
        for (const int64_t base : { int64_t(5), (int64_t(1) << 32) - 2, -(int64_t(1) << 44) }) {
            bad += ClimateTest::box_point_mismatches(spec, Vector3i64(base, (c.sdf.get_datum_y() >> lod) - 1, base / 3), 3, lod, true);
        }
    }
    CHECK(bad == 0);
}

// Sample programs (programs_test.json): each biome's term at the samples reaches every SDF path (box runs' samples equal
// the point query), stays within its compiled bound, reads its field from the carry, and in GLSL defines what it calls.
TEST_CASE("Climate shape: sample programs reach every SDF path, within their bounds") {
    ClimateSDF c("res://climate/programs_test.json", 15);
    CHECK(c.sdf.get_climate_error() == ov::str());
    CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
    const Table &t = *c.sdf.get_climate();
    CLIMATE_REQUIRE(t.biomes.size() == 3);
    for (const Climate::Biome &b : t.biomes) {
        CHECK_FALSE(b.program.empty());
        CHECK_FALSE(b.program.curve_of_ground());
        CHECK(b.program.bound > 0.0);
    }
    const SampleCarry carry = sample_carry(t);
    CHECK(carry.field[0] < 0);
    CHECK(carry.field[1] >= 0);
    Cascade::ShaderParams params;
    const ov::str glsl = c.sdf.get_ground_glsl(c.sdf.get_spec(), params);
    for (const char *name : { "vec4 cl_sample0(", "vec4 cl_sample1(", "vec4 cl_sample2(", "vec4 cl_curve0_0(", "vec4 cl_smin(" }) {
        CHECK(ov::contains(glsl, name));
    }
    double bound = 0.0;
    for (const Climate::Biome &b : t.biomes) {
        bound = std::max(bound, b.program.bound);
    }
    const Cascade::Spec &spec = c.sdf.get_spec();
    int64_t points = 0, mismatches = 0, over = 0;
    for (const int lod : { 0, 4, 9 }) {
        Cascade::Box box;
        for (int a = 0; a < 3; a++) {
            box.lo[a] = (a == 1 ? (c.sdf.get_datum_y() >> lod) : (int64_t(-77777) >> lod)) - 2;
            box.hi[a] = box.lo[a] + 3;
        }
        const Cascade::Level level = Cascade::run(spec, box, lod, nullptr, true);
        Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
            float slots[Cascade::SLOTS], ch[Cascade::CHANNELS];
            level.slots_at(p_i, slots);
            c.sdf.ground_channels(slots, p_p * (int64_t(1) << lod), lod, ch);
            const VoxelPos v(p_p.x << lod, p_p.y << lod, p_p.z << lod);
            points++;
            mismatches += float_bits(c.sdf.sdf(ch, v, lod).w) != float_bits(c.sdf.evaluate(WorldPos{ v, Float3() }, lod));
            // The weights sum to 1: |h - H| <= the largest term's bound.
            over += std::fabs(double(ch[Cascade::VALUE]) - double(slots[Cascade::VALUE])) > bound * (1.0 + 1e-6);
        });
    }
    CHECK(points == 192);
    CHECK(mismatches == 0);
    CHECK(over == 0);
    // The default's program alone (no field: its weight is 1 exactly): the ground is the program's value, capped
    // smoothly at 40 m, floored at -60 m.
    ov::Value data;
    CHECK(load_table("res://climate/programs_test.json", data) == ov::str());
    ov::Value one = ov::Value::object();
    one["fields"] = ov::Value::array();
    one["biomes"] = ov::Value::array();
    one["biomes"].push_back(data["biomes"][size_t(2)]);
    ClimateSDF capped(one, 15);
    CLIMATE_REQUIRE(capped.sdf.get_climate() != nullptr);
    const float vs = 0.05f;
    int high = 0, low = 0;
    for (int i = 0; i < 400; i++) {
        float slots[Cascade::SLOTS], ch[Cascade::CHANNELS];
        const Vector3i64 p(int64_t(i) * 1237 - 250000, capped.sdf.get_datum_y() >> 6, int64_t(i) * -2711);
        Cascade::point_slots(capped.sdf.get_spec(), p, 6, slots, true);
        capped.sdf.ground_channels(slots, p * int64_t(64), 6, ch);
        const double hh = double(slots[Cascade::VALUE]) * vs, h = double(ch[Cascade::VALUE]) * vs;
        if (hh > 40.0 + 25.0) {
            high++;
            CHECK(std::fabs(h - 40.0) < 1e-3);
        } else if (hh < -60.0) {
            low++;
            CHECK(std::fabs(h + 60.0) < 1e-3);
        } else if (hh < 40.0 - 25.0) {
            CHECK(std::fabs(h - hh) < 1e-3);
        }
    }
    CHECK(high > 10);
    CHECK(low > 10);
}

// Every level shader variant (chunks / points, with / without the fields) defines each climate helper it calls: a shaped
// table with and without curves (the shared helpers are spliced once, up front; get_climate_helpers_glsl), and several notes per
// octave.
TEST_CASE("Climate shape: every level shader variant defines the climate helpers it calls") {
    const ov::Value with_terms = shaped_table();
    const ov::Value without = ClimateTest::without_curves(shaped_table());
    ov::Value notes;
    CHECK(load_table("res://climate/notes_test.json", notes) == ov::str());
    ov::Value programs;
    CHECK(load_table("res://climate/programs_test.json", programs) == ov::str());
    const char *const NAMES[] = { "cl_flush", "cl_mul", "cl_sub", "cl_add", "cl_scale", "cl_edge", "cl_multifractal", "cl_smooth_abs", "cl_curve0", "cl_curve1",
        "cascade_warped", "climate_ground", "climate_biome", "cascade_mods", "cl_smin", "cl_sample0", "cl_curve0_0" };
    ov::Value world, graph;
    CHECK(load_table("res://climate/eight_biomes_test.json", world) == ov::str());
    CHECK(load_table("res://climate/graph_test.json", graph) == ov::str()); // sub-mix fields: modulated groups past the ground
    int64_t variants = 0, missing = 0, past_mods = 0;
    ov::Value caves;
    CHECK(load_table("res://climate/caves_test.json", caves) == ov::str());
    for (const ov::Value &table : { with_terms, without, notes, programs, world, graph, caves }) {
        SDFCascade sdf;
        sdf.climate_data = table;
        ChunkParam param;
        param.voxel_size = 0.05f;
        sdf.configure(param, 25);
        CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
        for (const bool points : { false, true }) {
            for (const CascadeGLSL::View view : { CascadeGLSL::View::FIELDS, CascadeGLSL::View::SAMPLES }) {
                const ov::str code = CascadeGLSL::level_shader(&sdf, false, points, false, !points, view);
                variants++;
                // Every BandMod slot written is within the smallest array the shader declares.
                int slots = INT_MAX;
                for (size_t at = code.find("BandMod mod["); at != ov::str::npos; at = code.find("BandMod mod[", at + 12)) {
                    slots = std::min(slots, std::atoi(code.c_str() + at + 12));
                }
                for (size_t at = code.find("mod["); at != ov::str::npos; at = code.find("mod[", at + 4)) {
                    const int slot = std::atoi(code.c_str() + at + 4);
                    const bool word = at == 0 || !(std::isalnum(static_cast<unsigned char>(code[at - 1])) || code[at - 1] == '_');
                    const bool declared = at >= 8 && code.compare(at - 8, 8, "BandMod ") == 0;
                    if (word && !declared && std::isdigit(static_cast<unsigned char>(code[at + 4])) && slot >= slots) {
                        past_mods++;
                        MESSAGE("variant points=", points, " view=", int(view), ": mod[", slot, "] past ", slots, " slots");
                        break;
                    }
                }
                for (const char *name : NAMES) {
                    const ov::str call = ov::str(name) + "(";
                    const bool called = ov::contains(code, call);
                    const bool defined = ov::contains(code, " " + call + "vec4 ") || ov::contains(code, " " + call + "float ") || ov::contains(code, " " + call + "WorldPos ") ||
                            ov::contains(code, " " + call + "Ch ") || ov::contains(code, " " + call + "int ");
                    if (called && !defined) {
                        missing++;
                        MESSAGE("variant points=", points, " view=", int(view), ": ", name, " called, not defined");
                    }
                }
            }
        }
    }
    CHECK(variants == 28);
    CHECK(missing == 0);
    CHECK(past_mods == 0);
    // A run's last level (the SAMPLES view: no field groups) reads a level's other notes where the band table holds
    // them, as the FIELDS view does: the same `first = N` in both (the GPU caught 4n's last levels reading field bands).
    SDFCascade sdf;
    sdf.climate_data = notes;
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 25);
    CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
    auto firsts = [](const ov::str &p_code) {
        std::vector<ov::str> out;
        for (size_t at = p_code.find("first = "); at != ov::str::npos; at = p_code.find("first = ", at + 8)) {
            out.push_back(p_code.substr(at, p_code.find(';', at) - at));
        }
        return out;
    };
    const std::vector<ov::str> fields = firsts(CascadeGLSL::level_shader(&sdf, false, false, false, true, CascadeGLSL::View::FIELDS));
    const std::vector<ov::str> samples = firsts(CascadeGLSL::level_shader(&sdf, false, false, false, true, CascadeGLSL::View::SAMPLES));
    CHECK(!fields.empty());
    CHECK(fields == samples);
}

// Stretched notes (stretch_y): the octaves in range carry their stretch (at most k - 3), the ground's
// vertical slope drops against its horizontal one (the noise read at y / 4), and bad factors are refused. Box runs and
// point queries agree on the stretched table: the first test of this file runs on it.
TEST_CASE("Climate shape: stretched bands lower the ground's vertical slope") {
    ShapedSDF s;
    CLIMATE_REQUIRE(s.sdf.get_climate() != nullptr);
    const Cascade::Spec &spec = s.sdf.get_spec();
    CHECK(spec.stretched());
    const int k100 = ExactNoise::lattice_k(100.0, 0.05);
    int stretched = 0;
    for (const Cascade::Band &b : spec.bands) {
        CHECK(b.stretch == (b.k >= k100 ? MIN(2, b.k - ExactNoise::BAND_SHIFT) : 0));
        stretched += b.stretch != 0;
    }
    CHECK(stretched > 0);

    ov::Value flat = shaped_table();
    ClimateTest::set_stretch(flat, 1.0);
    SDFCascade iso;
    iso.climate_data = flat;
    ChunkParam param;
    param.voxel_size = 0.05f;
    iso.configure(param, 15);
    CLIMATE_REQUIRE(iso.get_climate() != nullptr);
    // Mean |dh/dy| / mean |dh/dx| at the ground surface of 24 x 24 columns of Λ_m.
    auto ratio = [](const SDFCascade &p_sdf, int m) {
        double dy = 0.0, dx = 0.0;
        for (int i = 0; i < 24; i++) {
            for (int j = 0; j < 24; j++) {
                const int64_t x = int64_t(j * 37) << m, z = int64_t(i * 41) << m;
                const int64_t y = WorldCoord::floor_div(p_sdf.ground_at(x, z).y, int64_t(1) << m) << m;
                float ch[Cascade::CHANNELS];
                p_sdf.channels_at(VoxelPos(x, y, z), m, ch);
                dy += std::fabs(double(ch[Cascade::DY]));
                dx += std::fabs(double(ch[Cascade::DX]));
            }
        }
        return dy / MAX(dx, 1e-30);
    };
    // On Λ_9 the octaves under 100 m (unstretched, their slope kept: 0.87 / 1.23 on Λ_8) are not inserted yet.
    const double r_iso = ratio(iso, 9), r_stretched = ratio(s.sdf, 9);
    MESSAGE("mean |dh/dy| / |dh/dx| on Λ_9: isotropic ", r_iso, ", stretched by 4 ", r_stretched);
    CHECK(r_stretched < 0.7 * r_iso); // measured 0.64 / 1.03 (0.46 / 1.09 under the old multifractal state's weights)

    Table t;
    for (const double bad : { 3.0, 0.5, 1024.0 }) {
        ov::Value d = shaped_table();
        ClimateTest::set_stretch(d, bad);
        CHECK(ov::contains(parse(d, s.sdf.climate_ground(), t), "power of two"));
    }
}

// A table with no field (one biome multiplying the ground's octaves, adapter/godot/project/climate/fieldless_test.json) modulates
// the ground: the GPU passes take the variant with the modulation at its levels (CascadeGLSL::field_range not empty).
// It was empty with no field group, and the GPU drew the bare ground under the CPU's flat desert (CascadeSpike 4d).
// The same without fields, with a curve (a ground term): climate_ground runs at every sample, so every level takes the
// climate variant (the early return for a table without fields skipped that rule: the GPU drew the bare ground).
TEST_CASE("Climate shape: ground terms without fields take the climate variant at every level") {
    SDFCascade sdf;
    CHECK(load_table("res://climate/fieldless_test.json", sdf.climate_data) == ov::str());
    ClimateTest::add_curve(sdf.climate_data, 0, "[[-10, -10], [10, 14]]");
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
    CHECK(sdf.has_ground_terms());
    for (const bool points : { false, true }) {
        CHECK(CascadeGLSL::field_range(&sdf, points).everywhere());
        CHECK(ov::contains(CascadeGLSL::level_shader(&sdf, false, points, false, !points, CascadeGLSL::View::SAMPLES), "climate_ground("));
    }
}

// The skirt bound (Climate::ground_step) on tables with curves (the example and shaped_test): at the samples a LOD L and a
// LOD L+1 chunk share (Λ_{L+1}) near the LOD-L surface, the field step is within the outer chunk's skirt depth, the cap
// lifted (the bound itself; the cap is a cost choice). The curves read the weights carried to each chunk's own level 0,
// so their steps come from the fields' L+2 / L+3 octaves.
TEST_CASE("Climate shape: the field step between LOD L and L+1 stays within the skirt depth with ground terms") {
    for (const auto &[path, lods] : { std::pair{ "res://climate/shaped_test.json", 15 }, std::pair{ "res://climate/example.json", 15 }, std::pair{ "res://climate/eight_biomes_test.json", 15 }, std::pair{ "res://climate/programs_test.json", 15 } }) {
        CAPTURE(path);
        ClimateSDF c(path, lods, 1e6f);
        CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
        CHECK(c.sdf.get_climate()->has_programs());
        const double vs = 0.05;
        for (const int lod : { 0, 3, 6, 9, 12 }) {
            CAPTURE(lod);
            const int64_t unit = int64_t(2) << lod; // Λ_{L+1}
            const int64_t window = 512 * unit;
            auto at = [&](const VoxelPos &p_column, int64_t p_y, int p_lod) {
                return double(c.sdf.evaluate(WorldPos{ VoxelPos(p_column.x, p_y, p_column.z), Float3() }, p_lod));
            };
            double worst = 0.0;
            int64_t points = 0;
            for (uint32_t i = 0; i < 48; i++) {
                const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(lod), 0x7E44u });
                const VoxelPos column(WorldCoord::floor_div(int64_t(h.x % uint32_t(window)) - window / 2, unit) * unit, 0,
                        WorldCoord::floor_div(int64_t(h.y % uint32_t(window)) - window / 2, unit) * unit);
                // The LOD-L surface on Λ_{L+1}, from the LOD-0 ground: scan down from a few cells above it.
                const int64_t ground = c.sdf.ground_at(column.x, column.z).y;
                int64_t y = WorldCoord::floor_div(ground, unit) * unit + 8 * unit;
                for (int s = 0; s < 64 && at(column, y - unit, lod) > 0.0; s++) {
                    y -= unit;
                }
                for (int64_t d = -2 * unit; d <= 2 * unit; d += unit) {
                    const double fine = at(column, y + d, lod);
                    if (std::fabs(fine) > vs * std::ldexp(1.0, lod)) {
                        continue; // only near the surface, where the skirt has to hide the step
                    }
                    const double step = at(column, y + d, lod + 1) - fine;
                    worst = MAX(worst, std::fabs(step) / double(c.sdf.skirt_depth_m(step > 0.0 ? lod : lod + 1)));
                    points++;
                }
            }
            MESSAGE(std::string(path), ": LOD ", lod, " | ", lod + 1, ": ", points, " shared samples near the surface, worst step ", worst, " of the skirt depth");
            CHECK(points > 20);
            CHECK(worst <= 1.0);
        }
    }
}

// p_table with every number the mixer's sliders move scaled or shifted, its structure kept: constants, smooth k, curve
// heights, note amplitudes, multifractal bounds, warp amounts, mix factors, biome centres.
static ov::Value retuned(ov::Value p_table) {
    for (size_t b = 0; b < p_table["biomes"].size(); b++) {
        ov::Value &biome = p_table["biomes"].at(b);
        const ov::Value at = std::as_const(biome)["at"]; // the default biome has none
        for (const auto &axis : at.entries()) {
            biome["at"][axis.first].at(0) = axis.second[0].as_double() + 0.03;
        }
        ov::Value &nodes = ClimateTest::graph_nodes(p_table, b);
        for (size_t n = 0; n < nodes.size(); n++) {
            ov::Value &node = nodes.at(n);
            const ov::str kind = node["kind"].as_str();
            auto scale = [&](const char *p_key, double p_by) {
                if (std::as_const(node).has(p_key)) {
                    node[p_key] = node[p_key].as_double() * p_by;
                }
            };
            if (kind == "constant") {
                node["value_m"] = node["value_m"].as_double() + 7.0;
            } else if (kind == "smin" || kind == "smax") {
                scale("k_m", 1.5);
            } else if (kind == "note") {
                scale("amp_m", 0.25);
                node["seed"] = node["seed"].as_double() + 1.0;
            } else if (kind == "multifractal") {
                scale("low_m", 0.9);
                scale("high_m", 1.2);
            } else if (kind == "warp") {
                scale("amount_m", 1.3);
            } else if (kind == "curve") {
                for (size_t i = 0; i < node["points"].size(); i++) {
                    node["points"].at(i).at(1) = node["points"][i][1].as_double() * 1.2;
                }
            } else if (kind == "mix") {
                for (size_t i = 0; i < node["rows"].size(); i++) {
                    node["rows"].at(i)["x"] = (std::as_const(node)["rows"][i].has("x") ? node["rows"][i]["x"].as_double() : 1.0) * 1.1;
                }
            }
        }
    }
    return p_table;
}

// p_table with the edits whose values the specialised compile folds (p_kind): 0 every shape on a note's chain another
// type (a shape at the samples kept: its type moves the program's bounds, not its structure), 1 the values
// at 0 (warp amounts, each biome's first mix row), 2 neutral (every row at volume 1, shapes plain, no warp). The
// structure kept.
static ov::Value folded(ov::Value p_table, int p_kind) {
    static const char *const SHAPES[] = { "ridged", "plain", "billow", "square" };
    for (size_t b = 0; b < p_table["biomes"].size(); b++) {
        ov::Value &nodes = ClimateTest::graph_nodes(p_table, b);
        for (size_t n = 0; n < nodes.size(); n++) {
            ov::Value &node = nodes.at(n);
            const ov::str kind = node["kind"].as_str();
            bool chain = false;
            for (size_t i = 0; kind == "shape" && i < nodes.size(); i++) {
                const ov::str in_kind = nodes[i]["kind"].as_str();
                chain = chain || (nodes[i]["id"].as_str() == node["in"].as_str() && in_kind == "note");
            }
            if (kind == "shape" && p_kind != 1 && (chain || p_kind == 2)) {
                node.erase("coeffs");
                node.erase("offset");
                node.erase("scale");
                node["shape"] = p_kind == 0 ? SHAPES[(b + n) % 4] : "plain";
            } else if (kind == "warp" && p_kind != 0) {
                node["amount_m"] = 0.0;
            } else if (kind == "mix" && p_kind != 0) {
                for (size_t i = 0; i < node["rows"].size(); i++) {
                    node["rows"].at(i)["x"] = p_kind == 2 ? 1.0 : (i == 0 ? 0.0 : node["rows"][i]["x"].as_double());
                }
            }
        }
    }
    return p_table;
}

// p_table with every node of p_kinds taken out of its graph: whatever read one reads its input, a note it warped reads
// no warp (a structural edit: the mixer deleting nodes).
static ov::Value bypassed(ov::Value p_table, std::initializer_list<const char *> p_kinds) {
    for (size_t b = 0; b < p_table["biomes"].size(); b++) {
        ov::Value &nodes = ClimateTest::graph_nodes(p_table, b);
        for (size_t n = nodes.size(); n-- > 0;) {
            const ov::str kind = nodes[n]["kind"].as_str();
            if (std::none_of(p_kinds.begin(), p_kinds.end(), [&](const char *p_k) { return kind == p_k; })) {
                continue;
            }
            const ov::str id = nodes[n]["id"].as_str();
            const ov::Value in = nodes[n]["in"];
            for (size_t o = 0; o < nodes.size(); o++) {
                ov::Value &node = nodes.at(o);
                for (const char *key : { "in", "height" }) {
                    if (std::as_const(node).has(key) && node[key].as_str() == id) {
                        node[key] = in;
                    }
                }
                if (std::as_const(node).has("warp") && node["warp"].as_str() == id) {
                    node.erase("warp");
                }
                for (size_t r = 0; std::as_const(node).has("rows") && r < node["rows"].size(); r++) {
                    if (node["rows"][r]["in"].as_str() == id) {
                        node["rows"].at(r)["in"] = in;
                    }
                }
            }
            nodes.remove_at(n);
        }
    }
    return p_table;
}

// A table's text follows its structure alone (Cascade::ShaderParams hold its numbers): the specialised one's through an
// edit of its values (retuned), the generic one's (Table::generic, SDFCascade::specialize false) through every edit,
// those the specialised compile folds included (a shape's type, a value at 0, notes turning constant), so the biome
// mixer's edits compile nothing; and the generic table's bits are the specialised one's, every edit.
TEST_CASE("Climate shape: a table's text follows its structure, the generic one's through every edit, with the specialised bits") {
    int64_t compared = 0, mismatches = 0;
    for (const char *path : { "res://climate/shaped_test.json", "res://climate/eight_biomes_test.json", "res://climate/programs_test.json", "res://climate/notes_test.json",
                 "res://climate/example.json", "res://climate/plain.json", "res://climate/graph_test.json", "res://climate/caves_test.json" }) {
        ov::Value data;
        CLIMATE_REQUIRE(load_table(path, data) == ov::str());
        ov::str first[2]; // the table's own texts: generic, specialised
        int64_t moved[2] = {}, params = 0;
        Climate::Caps caps; // the first compile's, kept like World::Kept keeps them through the mixer's restarts
        auto text = [&](const ClimateSDF &p_c) {
            ov::str t;
            for (const bool points : { false, true }) {
                for (const CascadeGLSL::View view : { CascadeGLSL::View::FIELDS, CascadeGLSL::View::SAMPLES }) {
                    Cascade::ShaderParams p;
                    t += CascadeGLSL::level_shader(&p_c.sdf, false, points, false, !points, view, true, &p);
                    CHECK(p.vec4s() * 16 <= 16384); // CascadePasses' uniform block (MAX_PARAM_BYTES)
                    params = std::max(params, int64_t(p.values.size()));
                }
            }
            return t;
        };
        for (int e = 0; e < 8; e++) {
            const ov::Value table = e == 0 ? data
                    : e == 1               ? retuned(data)
                    : e < 5                ? folded(data, e - 2)
                    : e == 5               ? bypassed(data, { "shape" })
                    : e == 6               ? bypassed(data, { "warp", "multifractal" })
                                           : bypassed(data, { "curve" });
            ClimateSDF generic(table, 25, false, caps), specialised(table, 25);
            caps = generic.sdf.generic_caps();
            CLIMATE_REQUIRE(generic.sdf.get_climate() != nullptr);
            CLIMATE_REQUIRE(specialised.sdf.get_climate() != nullptr);
            const ov::str texts[2] = { text(generic), text(specialised) };
            for (int g = 0; g < 2; g++) {
                if (e == 0) {
                    first[g] = texts[g];
                }
                moved[g] += texts[g] != first[g];
            }
            CHECK((e != 1 || texts[1] == first[1])); // retuned: the specialised structure kept
            for (const int lod : { 0, 2, 6, 11 }) {
                for (int i = 0; i < 40; i++) {
                    const Vector3i64 p(int64_t(i) * 7919 - 50000, (generic.sdf.get_datum_y() >> lod) + 20 - i, int64_t(i) * -104729 + (int64_t(1) << 33));
                    const WorldPos pos = ClimateTest::lattice_pos(p, lod);
                    compared++;
                    mismatches += float_bits(generic.sdf.evaluate(pos, lod)) != float_bits(specialised.sdf.evaluate(pos, lod));
                }
            }
        }
        MESSAGE(std::string(path), ": of 7 edits the generic text moved on ", moved[0], ", the specialised on ", moved[1], "; at most ", params, " params");
        CHECK(moved[0] == 0);
    }
    MESSAGE("generic vs specialised: ", compared, " samples, ", mismatches, " differ");
    CHECK(mismatches == 0);
    // No climate: the template ground, specialised either way (nothing edits it; generic, it paid cascade_mods: +11%).
    const ClimateSDF bare_generic(ov::Value(), 25, false), bare(ov::Value(), 25);
    for (const bool points : { false, true }) {
        Cascade::ShaderParams pg, ps;
        CHECK(CascadeGLSL::level_shader(&bare_generic.sdf, false, points, false, !points, CascadeGLSL::View::FIELDS, true, &pg) ==
                CascadeGLSL::level_shader(&bare.sdf, false, points, false, !points, CascadeGLSL::View::FIELDS, true, &ps));
    }
}
