#include "test_climate_common.hpp"

#include "cascade_glsl.hpp"
#include "climate.hpp"
#include "sdf_cascade.hpp"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>

#include <cmath>

using namespace Climate;
using ClimateTest::ClimateSDF;
using ExactNoise::float_bits;

namespace {

uint64_t fnv(uint64_t p_h, uint32_t p_v) {
    for (int i = 0; i < 4; i++) {
        p_h = (p_h ^ ((p_v >> (8 * i)) & 0xFFu)) * 1099511628211ull;
    }
    return p_h;
}

uint64_t fnv_text(const ov::str &p_s) {
    uint64_t h = 14695981039346656037ull;
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
    uint64_t h = 14695981039346656037ull;
    for (const int lod : { 0, 3, 8, 13 }) {
        for (int i = 0; i < 48; i++) {
            const Vector3i64 p(int64_t(i) * 7919 - 50000, (c.sdf.get_datum_y() >> lod) + 3 - i, int64_t(i) * -104729 + (int64_t(1) << 33));
            float slots[Cascade::SLOTS];
            Cascade::point_slots(spec, p, lod, slots, true);
            // The carried slots only: the pin doesn't depend on the group caps.
            for (int s = 0; s < Cascade::field_slot(int(spec.fields.size())); s++) {
                h = fnv(h, float_bits(slots[s]));
            }
            for (int s = Cascade::carry_slot(0); s < Cascade::carry_slot(spec.carry_groups()); s++) {
                h = fnv(h, float_bits(slots[s]));
            }
            const WorldPos pos{ VoxelPos(p.x << lod, p.y << lod, p.z << lod), Float3() };
            h = fnv(h, float_bits(c.sdf.evaluate(pos, lod)));
        }
    }
    for (int l = 0; l < 15; l++) {
        h = fnv(h, float_bits(c.sdf.skirt_depth_m(l)));
    }
    const uint64_t chunk_text = fnv_text(CascadeGLSL::level_shader(&c.sdf, false, false, false, true, CascadeGLSL::View::SAMPLES));
    const uint64_t point_text = fnv_text(CascadeGLSL::level_shader(&c.sdf, false, true));
    MESSAGE("plain climate: bits ", h, ", chunk shader ", chunk_text, ", point shader ", point_text);
    CHECK(h == 435047546757689742ull);
    // The shader texts (the modulation's arrays sized by the spec's group count, the ground over its base surface).
    CHECK(chunk_text == 420559920128829333ull);
    // The point shader re-pinned when point items became any lattice point with its unit register (point_frame, 16 B),
    // and (all three) when the derive moved before the level's insertions (one climate per point).
    CHECK(point_text == 5024476916576385137ull);
}

// The shapes' centring constants are the noises' own means: E[sabs(n)] and E[n^2] of perlin3, E[F1] of cell3 (its
// value plus CELL_MEAN), over random points of random cells.
TEST_CASE("Climate shape: NOISE_ABS_MEAN, NOISE_SQ_MEAN and CELL_MEAN are the noises' means") {
    double abs_sum = 0.0, sq_sum = 0.0, cell_sum = 0.0;
    const int n = 1 << 20;
    for (int i = 0; i < n; i++) {
        const ExactNoise::UVec3 h = ExactNoise::hash3(uint32_t(i), 0x51EDu, 0xC0FFEEu);
        const WorldPos p{ VoxelPos(int64_t(h.x % 100000u) - 50000, int64_t(h.y % 100000u) - 50000, int64_t(h.z % 100000u) - 50000),
            Float3{ float(h.x >> 8) * 0x1p-24f, float(h.y >> 8) * 0x1p-24f, float(h.z >> 8) * 0x1p-24f } };
        const float v = ExactNoise::perlin3(p, 6, 0x1234u + uint32_t(i & 7), Vector3i64()).w;
        abs_sum += std::sqrt(double(v) * v + double(Cascade::SMOOTH_ABS_E) * Cascade::SMOOTH_ABS_E) - Cascade::SMOOTH_ABS_E;
        sq_sum += double(v) * v;
        cell_sum += ExactNoise::cell3(p, 6, 0x1234u + uint32_t(i & 7), Vector3i64()).w + ExactNoise::CELL_MEAN;
    }
    MESSAGE("E[sabs(n)] ", abs_sum / n, ", E[n^2] ", sq_sum / n, ", E[F1] ", cell_sum / n);
    CHECK(std::fabs(abs_sum / n - Cascade::NOISE_ABS_MEAN) < 0.01 * Cascade::NOISE_ABS_MEAN);
    CHECK(std::fabs(sq_sum / n - Cascade::NOISE_SQ_MEAN) < 0.01 * Cascade::NOISE_SQ_MEAN);
    CHECK(std::fabs(cell_sum / n - ExactNoise::CELL_MEAN) < 0.01 * ExactNoise::CELL_MEAN);
}

namespace {

// Every operation at once (adapter/godot/project/climate/shaped_test.json, CascadeSpike's set 4s too).
ov::Value shaped_table() {
    ov::Value d;
    CHECK(load("res://climate/shaped_test.json", d) == ov::str());
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
    CHECK(t.has_state());
    CHECK(t.has_ground_terms());
    CHECK(spec.fields.size() == t.fields.size() + 1); // the state group, band-less
    CHECK(spec.fields.back().empty());
    CHECK(spec.cell_groups == 1u << 6); // dunes: field 5, group 6
    CLIMATE_REQUIRE(spec.modulation != nullptr);
    CHECK(spec.modulation->shaped());
    CHECK(spec.modulation->state_group() == 7);
    CHECK(t.fields[4].shapes[0].c[3] == 4.0f); // ridged, scale 2: (1 - 2|n|)^2
    int64_t bad = 0;
    for (const int lod : { 0, 3, 8 }) {
        for (const int64_t base : { int64_t(5), (int64_t(1) << 32) - 2, -(int64_t(1) << 44) }) {
            bad += ClimateTest::box_point_mismatches(spec, Vector3i64(base, (s.sdf.get_datum_y() >> lod) - 1, base / 3), 3, lod, true);
        }
    }
    CHECK(bad == 0);
    // The state is written where the multifractal biome has its octaves, a weight's complement in about [0, 1].
    int written = 0;
    for (int i = 0; i < 64; i++) {
        float slots[Cascade::SLOTS];
        Cascade::point_slots(spec, Vector3i64(int64_t(i) * 7919, 3, int64_t(i) * -1301), 2, slots, true);
        const float c = slots[Cascade::field_slot(6)];
        written += c != 0.0f;
        CHECK(std::isfinite(c));
        CHECK(c >= -0.05f);
        CHECK(c <= 1.05f);
    }
    CHECK(written > 8);
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
        const Cascade::Level level = Cascade::run(spec, box, lod, nullptr, Cascade::Source(), true);
        Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
            float slots[Cascade::SLOTS], ch[Cascade::CHANNELS];
            level.slots_at(p_i, slots);
            s.sdf.ground_channels(slots, ch);
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
    CHECK(ov::contains(s.sdf.get_ground_glsl(s.sdf.get_spec()), "vec4 climate_ground(Ch ch)"));
    // The skirts take the height terms' step: deeper than the same ground without them where a term's field has
    // octaves (dunes: 25 m, k = 9 at 5 cm, LODs 4-5), the cap (skirt_depth cells) lifted.
    ShapedSDF deep(15, 1e6f);
    ov::Value no_terms = shaped_table();
    no_terms.erase("heights");
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
    const ov::str f1 = R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]}], )";
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "shape": "spiky"}]}]})")), "plain, billow"));
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "t", "from_m": 400, "to_m": 800, "multifractal": 2}]}]})")), "ground bands only"));
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "warp_m": 1, "warp_fields": ["t"]}]}]})")),
            "3 fields"));
    // A warp reads fields: the ground's slope (the swiss turbulence's) is not offered, nor its damping (climate.hpp).
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "warp_m": 1}]}]})")), "warp reads 3 fields"));
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "damp": 4}]}]})")), "unknown key 'damp'"));
    CHECK(ov::contains(err(f1 + ov::str(R"("heights": [{"field": "t", "height_m": 10, "curve": [[0, 0], [0, 1]]}], "biomes": [{"name": "a"}]})")), "increase"));
    CHECK(ov::contains(err(f1 + ov::str(R"("heights": [{"field": "t", "height_m": 10, "curve": [[0, 0]]}], "biomes": [{"name": "a"}]})")), "2 points"));
    // A curve the field never reaches is a constant; a term over the ground's amplitude cap would outgrow the skirts' model.
    CHECK(ov::contains(err(f1 + ov::str(R"("heights": [{"field": "t", "height_m": 10, "curve": [[100, 0], [200, 1]]}], "biomes": [{"name": "a"}]})")),
            "a constant"));
    CHECK(ov::contains(err(f1 + ov::str(R"("heights": [{"field": "t", "height_m": 5000, "curve": [[-1, 0], [1, 1]]}], "biomes": [{"name": "a"}]})")),
            "over the ground's cap"));
    CHECK(err(f1 + ov::str(R"("heights": [{"field": "t", "height_m": 100, "curve": [[-1, 0], [1, 1]]}], "biomes": [{"name": "a"}]})")) == ov::str());
    // An offset is a constant term: inserted without its mean, it only counts in a multifractal's state (ridged's scales
    // its |n| too).
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "shape": "billow", "offset": 0.5}]}]})")),
            "only counts in a multifractal"));
    CHECK(ov::contains(err(R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 400, "amp": 1, "offset": 0.5}]}], "biomes": [{"name": "a"}]})"),
            "only counts in a multifractal"));
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "coeffs": [0.5, 1, 0, 0]}]}]})")),
            "only counts in a multifractal"));
    CHECK(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "multifractal": 1, "offset": 0.5}]}]})")) ==
            ov::str());
    CHECK(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "shape": "ridged", "offset": 0.8}]}]})")) ==
            ov::str());
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "terrace": {"step_m": -1}}]})")), "terrace"));
    CHECK(ov::contains(err(f1 + ov::str(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "coeffs": [1, 2]}]}]})")), "4 numbers"));
    CHECK(ov::contains(err(R"({"fields": [{"name": "d", "noise": "cell", "bands": [{"wavelength_m": 25, "amp": 1, "shape": "ridged"}]}], "biomes": [{"name": "a"}]})"),
            "cell noise"));
    CHECK(ov::contains(err(R"({"fields": [{"name": "d", "noise": "simplex", "bands": []}], "biomes": [{"name": "a"}]})"), "perlin or cell"));
    // A multifractal's state takes a field group: MAX_FIELDS fields leave it none.
    ov::str full = "{\"fields\": [";
    for (int i = 0; i < MAX_FIELDS; i++) {
        full += ov::format("%s{\"name\": \"f%d\", \"bands\": [{\"wavelength_m\": 400, \"amp\": 1}]}", i ? ", " : "", i);
    }
    full += "], \"biomes\": [{\"name\": \"a\", \"amp\": [{\"field\": \"ground\", \"from_m\": 1, \"to_m\": 100, \"multifractal\": 2}]}]}";
    CHECK(ov::contains(err(full), ov::format("at most %d fields", MAX_FIELDS - 1)));
}

TEST_CASE("Climate shape: the example world parses at any LOD count, shaped, and box runs equal point queries") {
    for (const int lods : { 15, 25 }) {
        ClimateSDF c("res://climate/example.json", lods);
        CHECK(c.sdf.get_climate_error() == ov::str());
        CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
        const Table &t = *c.sdf.get_climate();
        CHECK(t.has_state());
        CHECK(t.has_ground_terms());
        CHECK(c.sdf.has_ground_materials());
        const Cascade::Spec &spec = c.sdf.get_spec();
        CLIMATE_REQUIRE(spec.modulation != nullptr);
        CHECK(spec.modulation->shaped());
        int64_t bad = 0;
        for (const int lod : { 0, 5, 11 }) {
            bad += ClimateTest::box_point_mismatches(spec, Vector3i64(int64_t(18022 * 20) >> lod, (c.sdf.get_datum_y() >> lod) - 1, int64_t(65536 * 20) >> lod), 3, lod, true);
        }
        CHECK(bad == 0);
    }
}

// Every level shader variant (chunks / points, with / without the fields) defines each climate helper it calls: a shaped
// table with and without ground terms (the shared helpers are spliced once, up front; splices_helpers).
TEST_CASE("Climate shape: every level shader variant defines the climate helpers it calls") {
    ov::Value with_terms = shaped_table();
    ov::Value without = shaped_table();
    without.erase("heights");
    ov::Value &biomes = without["biomes"];
    for (size_t i = 0; i < biomes.size(); i++) {
        biomes.at(i).erase("terrace");
    }
    const char *const NAMES[] = { "cl_flush", "cl_mul", "cl_sub", "cl_add", "cl_scale", "cl_edge", "cl_multifractal", "cl_smooth_abs", "cl_terrace", "cascade_warped",
        "climate_ground", "climate_biome", "cascade_mods" };
    int64_t variants = 0, missing = 0;
    for (const ov::Value &table : { with_terms, without }) {
        SDFCascade sdf;
        sdf.climate_data = table;
        ChunkParam param;
        param.voxel_size = 0.05f;
        sdf.configure(param, 15);
        CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
        for (const bool points : { false, true }) {
            for (const CascadeGLSL::View view : { CascadeGLSL::View::FIELDS, CascadeGLSL::View::SAMPLES }) {
                const ov::str code = CascadeGLSL::level_shader(&sdf, false, points, false, !points, view);
                variants++;
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
    CHECK(variants == 8);
    CHECK(missing == 0);
    Table t;
    ov::Value bad = shaped_table();
    bad["biomes"].at(1)["terrace"] = ClimateTest::json(R"({"step_m": 8, "invert": "no"})");
    SDFCascade g;
    ChunkParam param;
    param.voxel_size = 0.05f;
    g.configure(param, 15);
    CHECK(ov::contains(parse(bad, g.climate_ground(), t), "true or false"));
    bad["biomes"].at(1)["terrace"] = ClimateTest::json(R"({"step_m": 0})");
    CHECK(ov::contains(parse(bad, g.climate_ground(), t), "terrace step"));
}

// Stretched bands (ground_stretch, stretch_y): the octaves in range carry their stretch (at most k - 3), the ground's
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
    CHECK(spec.fields[size_t(s.sdf.get_climate()->field_index("dunes"))].front().stretch == 2);

    ov::Value flat = shaped_table();
    flat.erase("ground_stretch");
    SDFCascade iso;
    iso.climate_data = flat;
    ChunkParam param;
    param.voxel_size = 0.05f;
    iso.configure(param, 15);
    CLIMATE_REQUIRE(iso.get_climate() != nullptr);
    // Mean |dh/dy| / mean |dh/dx| at the ground surface of 24 x 24 columns of Λ_8.
    auto ratio = [](const SDFCascade &p_sdf) {
        const int m = 8;
        double dy = 0.0, dx = 0.0;
        for (int i = 0; i < 24; i++) {
            for (int j = 0; j < 24; j++) {
                const int64_t x = int64_t(j * 37) << m, z = int64_t(i * 41) << m;
                const int64_t y = WorldCoord::floor_div(p_sdf.ground_y(x, z), int64_t(1) << m) << m;
                float ch[Cascade::CHANNELS];
                p_sdf.channels_at(VoxelPos(x, y, z), m, ch);
                dy += std::fabs(double(ch[Cascade::DY]));
                dx += std::fabs(double(ch[Cascade::DX]));
            }
        }
        return dy / MAX(dx, 1e-30);
    };
    const double r_iso = ratio(iso), r_stretched = ratio(s.sdf);
    MESSAGE("mean |dh/dy| / |dh/dx|: isotropic ", r_iso, ", stretched by 4 ", r_stretched);
    CHECK(r_stretched < 0.6 * r_iso);

    Table t;
    for (const double bad : { 3.0, 0.5, 1024.0 }) {
        ov::Value d = shaped_table();
        d["ground_stretch"].at(0)["stretch_y"] = bad;
        CHECK(ov::contains(parse(d, s.sdf.climate_ground(), t), "power of two"));
    }
}

// A table with no field (one biome multiplying the ground's octaves, adapter/godot/project/climate/fieldless_test.json) modulates
// the ground: the GPU passes take the variant with the modulation at its levels (CascadeGLSL::field_range not empty).
// It was empty with no field group, and the GPU drew the bare ground under the CPU's flat desert (CascadeSpike 4d).
// The same without fields, with a terrace (a ground term): climate_ground runs at every sample, so every level takes the
// climate variant (the early return for a table without fields skipped that rule: the GPU drew the terrace-less ground).
TEST_CASE("Climate shape: ground terms without fields take the climate variant at every level") {
    SDFCascade sdf;
    CHECK(load("res://climate/fieldless_test.json", sdf.climate_data) == ov::str());
    sdf.climate_data["biomes"].at(0)["terrace"] = ClimateTest::json(R"({"step_m": 4})");
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

// The landform tables (docs/landforms.md) load at 5 cm at 15 and 25 LODs: a tightened check that refuses one shows here,
// not at the next tools/landforms/match.py run (Mont Blanc's remap went over the height-term cap that way).
TEST_CASE("Climate shape: every landform table loads") {
    for (const char *name : { "bare", "caldera", "cliff", "desert", "dunes", "island", "mangrove", "mountain", "valley" }) {
        for (const int lods : { 15, 25 }) {
            CAPTURE(name);
            CAPTURE(lods);
            ClimateSDF c(("res://climate/landforms/" + std::string(name) + ".json").c_str(), lods);
            CHECK(c.sdf.get_climate_error() == ov::str());
            CHECK(c.sdf.get_climate() != nullptr);
        }
    }
}

// The skirt bound (Climate::ground_step) on tables with ground terms (terraces: the example and shaped_test; the cliff's
// 700 m height term): at the samples a LOD L and a LOD L+1 chunk share (Λ_{L+1}) near the LOD-L surface, the field step
// is within the outer chunk's skirt depth, the cap lifted (the bound itself; the cap is a cost choice). Height terms and
// terraces read the fields at each chunk's own level 0, so their steps come from the fields' L+2 / L+3 octaves.
TEST_CASE("Climate shape: the field step between LOD L and L+1 stays within the skirt depth with ground terms") {
    for (const char *path : { "res://climate/shaped_test.json", "res://climate/example.json", "res://climate/landforms/cliff.json", "res://climate/region_test.json" }) {
        CAPTURE(path);
        ClimateSDF c(path, 15, 1e6f);
        CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
        CHECK(c.sdf.get_climate()->has_ground_terms());
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
                const int64_t ground = c.sdf.ground_y(column.x, column.z);
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
