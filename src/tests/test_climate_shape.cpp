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

uint64_t fnv_text(const String &p_s) {
    const CharString c = p_s.utf8();
    uint64_t h = 14695981039346656037ull;
    for (int i = 0; i < c.length(); i++) {
        h = (h ^ uint8_t(c[i])) * 1099511628211ull;
    }
    return h;
}

} // namespace

// The climate before shaped bands (project/climate/plain.json, a copy of the example as it was) keeps its bits: every
// slot of every point query, the SDF of the ground, the skirts, and the level shaders' text. A table without the new
// keys is the old code path, op for op.
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
            // The carried slots only: the pin doesn't depend on the field group cap.
            for (int s = 0; s < Cascade::field_slot(int(spec.fields.size())); s++) {
                h = fnv(h, float_bits(slots[s]));
            }
            const WorldPos pos{ VoxelPos(p.x << lod, p.y << lod, p.z << lod), Float3() };
            h = fnv(h, float_bits(c.sdf.evaluate(pos, lod)));
        }
    }
    for (int l = 0; l < 15; l++) {
        h = fnv(h, float_bits(c.sdf.skirt_depth_m(l)));
    }
    const uint64_t chunk_text = fnv_text(CascadeGLSL::level_shader(&c.sdf, false, false, false, true, true));
    const uint64_t point_text = fnv_text(CascadeGLSL::level_shader(&c.sdf, false, true));
    MESSAGE("plain climate: bits ", h, ", chunk shader ", chunk_text, ", point shader ", point_text);
    CHECK(h == 16354123549857880415ull);
    // The shader texts (the modulation's arrays sized by the spec's group count, the ground over its base surface).
    CHECK(chunk_text == 585642622528421550ull);
    CHECK(point_text == 10506806479769159991ull);
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

// Every operation at once (project/climate/shaped_test.json, CascadeSpike's set 4s too).
Dictionary shaped_table() {
    Dictionary d;
    CHECK(load("res://climate/shaped_test.json", d) == String());
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
    CHECK(s.sdf.get_climate_error() == String());
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
    CHECK(s.sdf.get_ground_glsl().contains("vec4 climate_ground(Ch ch)"));
    // The skirts take the height terms' step: deeper than the same ground without them where a term's field has
    // octaves (dunes: 25 m, k = 9 at 5 cm, LODs 6-7), the cap (skirt_depth cells) lifted.
    ShapedSDF deep(15, 1e6f);
    Dictionary no_terms = shaped_table();
    no_terms.erase("heights");
    SDFCascade bare;
    bare.climate_data = no_terms;
    ChunkParam param;
    param.voxel_size = 0.05f;
    param.skirt_depth = 1e6f;
    bare.configure(param, 15);
    CLIMATE_REQUIRE(bare.get_climate() != nullptr);
    CHECK(deep.sdf.skirt_depth_m(6) > bare.skirt_depth_m(6));
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
    auto err = [&](const String &p_json) {
        Table t;
        return parse(Dictionary(JSON::parse_string(p_json)), g, t);
    };
    const String f1 = R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]}], )";
    CHECK(err(f1 + String(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "shape": "spiky"}]}]})")).contains("plain, billow"));
    CHECK(err(f1 + String(R"("biomes": [{"name": "a", "amp": [{"field": "t", "from_m": 400, "to_m": 800, "multifractal": 2}]}]})")).contains("ground bands only"));
    CHECK(err(f1 + String(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "warp_m": 1, "warp_fields": ["t"]}]}]})"))
                    .contains("3 fields"));
    // A warp reads fields: the ground's slope (the swiss turbulence's) is not offered, nor its damping (climate.hpp).
    CHECK(err(f1 + String(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "warp_m": 1}]}]})")).contains("warp reads 3 fields"));
    CHECK(err(f1 + String(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "damp": 4}]}]})")).contains("unknown key 'damp'"));
    CHECK(err(f1 + String(R"("heights": [{"field": "t", "height_m": 10, "curve": [[0, 0], [0, 1]]}], "biomes": [{"name": "a"}]})")).contains("increase"));
    CHECK(err(f1 + String(R"("heights": [{"field": "t", "height_m": 10, "curve": [[0, 0]]}], "biomes": [{"name": "a"}]})")).contains("2 points"));
    CHECK(err(f1 + String(R"("biomes": [{"name": "a", "terrace": {"step_m": -1}}]})")).contains("terrace"));
    CHECK(err(f1 + String(R"("biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 1, "to_m": 100, "coeffs": [1, 2]}]}]})")).contains("4 numbers"));
    CHECK(err(R"({"fields": [{"name": "d", "noise": "cell", "bands": [{"wavelength_m": 25, "amp": 1, "shape": "ridged"}]}], "biomes": [{"name": "a"}]})")
                    .contains("cell noise"));
    CHECK(err(R"({"fields": [{"name": "d", "noise": "simplex", "bands": []}], "biomes": [{"name": "a"}]})").contains("perlin or cell"));
    // A multifractal's state takes a field group: MAX_FIELDS fields leave it none.
    String full = "{\"fields\": [";
    for (int i = 0; i < MAX_FIELDS; i++) {
        full += vformat("%s{\"name\": \"f%d\", \"bands\": [{\"wavelength_m\": 400, \"amp\": 1}]}", i ? ", " : "", i);
    }
    full += "], \"biomes\": [{\"name\": \"a\", \"amp\": [{\"field\": \"ground\", \"from_m\": 1, \"to_m\": 100, \"multifractal\": 2}]}]}";
    CHECK(err(full).contains(vformat("at most %d fields", MAX_FIELDS - 1)));
}

TEST_CASE("Climate shape: the example world parses at any LOD count, shaped, and box runs equal point queries") {
    for (const int lods : { 15, 25 }) {
        ClimateSDF c("res://climate/example.json", lods);
        CHECK(c.sdf.get_climate_error() == String());
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
    Dictionary with_terms = shaped_table();
    Dictionary without = shaped_table();
    without.erase("heights");
    const Array biomes = without["biomes"];
    for (int64_t i = 0; i < biomes.size(); i++) {
        Dictionary(biomes[i]).erase("terrace");
    }
    const char *const NAMES[] = { "cl_flush", "cl_mul", "cl_sub", "cl_add", "cl_scale", "cl_edge", "cl_multifractal", "cl_smooth_abs", "cl_terrace", "cascade_warped",
        "climate_ground", "climate_biome", "cascade_mods" };
    int64_t variants = 0, missing = 0;
    for (const Dictionary &table : { with_terms, without }) {
        SDFCascade sdf;
        sdf.climate_data = table;
        ChunkParam param;
        param.voxel_size = 0.05f;
        sdf.configure(param, 15);
        CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
        for (const bool points : { false, true }) {
            for (const bool fields : { false, true }) {
                const String code = CascadeGLSL::level_shader(&sdf, false, points, false, !points, fields);
                variants++;
                for (const char *name : NAMES) {
                    const String call = String(name) + "(";
                    const bool called = code.contains(call);
                    const bool defined = code.contains(" " + call + "vec4 ") || code.contains(" " + call + "float ") || code.contains(" " + call + "WorldPos ") ||
                            code.contains(" " + call + "Ch ") || code.contains(" " + call + "int ");
                    if (called && !defined) {
                        missing++;
                        MESSAGE("variant points=", points, " fields=", fields, ": ", name, " called, not defined");
                    }
                }
            }
        }
    }
    CHECK(variants == 8);
    CHECK(missing == 0);
    Table t;
    Dictionary bad = shaped_table();
    Dictionary(Array(bad["biomes"])[1])["terrace"] = JSON::parse_string(R"({"step_m": 8, "invert": "no"})");
    SDFCascade g;
    ChunkParam param;
    param.voxel_size = 0.05f;
    g.configure(param, 15);
    CHECK(parse(bad, g.climate_ground(), t).contains("true or false"));
    Dictionary(Array(bad["biomes"])[1])["terrace"] = JSON::parse_string(R"({"step_m": 0})");
    CHECK(parse(bad, g.climate_ground(), t).contains("terrace step"));
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

    Dictionary flat = shaped_table();
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
        Dictionary d = shaped_table();
        Dictionary(Array(d["ground_stretch"])[0])["stretch_y"] = bad;
        CHECK(parse(d, s.sdf.climate_ground(), t).contains("power of two"));
    }
}

// A table with no field (one biome multiplying the ground's octaves, project/climate/fieldless_test.json) modulates
// the ground: the GPU passes take the variant with the modulation at its levels (CascadeGLSL::field_range not empty).
// It was empty with no field group, and the GPU drew the bare ground under the CPU's flat desert (CascadeSpike 4d).
TEST_CASE("Climate shape: a modulation without fields takes the modulated GPU variant") {
    SDFCascade sdf;
    CHECK(load("res://climate/fieldless_test.json", sdf.climate_data) == String());
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
    CHECK(sdf.get_spec().fields.empty());
    CHECK(sdf.get_spec().modulation != nullptr);
    for (const bool points : { false, true }) {
        const Cascade::FieldRange range = CascadeGLSL::field_range(&sdf, points);
        CHECK_FALSE(range.empty());
        const String code = CascadeGLSL::level_shader(&sdf, false, points, false, !points, true);
        CHECK((code.contains("cascade_mods(") || code.contains("cascade_amplitudes(")));
    }
}

// The same without fields, with a terrace (a ground term): climate_ground runs at every sample, so every level takes the
// climate variant (the early return for a table without fields skipped that rule: the GPU drew the terrace-less ground).
TEST_CASE("Climate shape: ground terms without fields take the climate variant at every level") {
    SDFCascade sdf;
    CHECK(load("res://climate/fieldless_test.json", sdf.climate_data) == String());
    Dictionary(Array(sdf.climate_data["biomes"])[0])["terrace"] = JSON::parse_string(R"({"step_m": 4})");
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
    CHECK(sdf.has_ground_terms());
    for (const bool points : { false, true }) {
        CHECK(CascadeGLSL::field_range(&sdf, points).everywhere());
        CHECK(CascadeGLSL::level_shader(&sdf, false, points, false, !points, true).contains("climate_ground("));
    }
}
