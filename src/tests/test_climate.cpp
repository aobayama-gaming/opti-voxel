#include "test_climate_common.hpp"

#include "cascade_glsl.hpp"
#include "climate.hpp"
#include "climate_report.hpp"
#include "graph_compile.hpp"
#include "level_program.hpp"
#include "sdf_cascade.hpp"

#include "ov/files.hpp"

#include <algorithm>
#include <cmath>
#include <random>

using namespace Climate;
using ExactNoise::float_bits;

namespace {

// The ground of an SDFCascade configured at p_voxel metres and p_lods LODs.
struct GroundFixture {
    SDFCascade sdf;
    Ground ground;

    GroundFixture(double p_voxel, int p_lods) {
        ChunkParam param;
        param.voxel_size = float(p_voxel);
        sdf.configure(param, p_lods);
        ground = sdf.climate_ground();
    }
};

// p_data compiled (specialised) for p_ground: empty, or the first error.
ov::str compile(const ov::Value &p_data, const Ground &p_ground, Table &r_table, Terrain::LevelProgram &r_prog) {
    return Terrain::compile(p_data, p_ground, Terrain::Mode::SPECIALISED, r_table, r_prog);
}

ov::str compile(const ov::Value &p_data, const Ground &p_ground, Table &r_table) {
    Terrain::LevelProgram prog;
    return compile(p_data, p_ground, r_table, prog);
}

ov::str parse_json(const ov::str &p_json, const Ground &p_ground, Table &r_table, Terrain::LevelProgram &r_prog) {
    const ov::Value v = ClimateTest::json(p_json);
    CHECK(v.is_object());
    return v.is_object() ? compile(v, p_ground, r_table, r_prog) : ov::str("not a JSON object");
}

ov::str parse_json(const ov::str &p_json, const Ground &p_ground, Table &r_table) {
    Terrain::LevelProgram prog;
    return parse_json(p_json, p_ground, r_table, prog);
}

bool subnormal(const Float4 &p_v) {
    return std::fpclassify(p_v.x) == FP_SUBNORMAL || std::fpclassify(p_v.y) == FP_SUBNORMAL || std::fpclassify(p_v.z) == FP_SUBNORMAL ||
            std::fpclassify(p_v.w) == FP_SUBNORMAL;
}

// A random table over p_fields fields (no bands needed: weights read field values only), 2 to MAX_BIOMES biomes.
Table random_table(std::mt19937 &p_rng, int p_fields) {
    std::uniform_real_distribution<float> centre(-1.0f, 1.0f);
    std::uniform_real_distribution<float> span(0.05f, 1.5f);
    Table t;
    for (int f = 0; f < p_fields; f++) {
        Field field;
        field.name = ov::format("f%d", f);
        t.fields.push_back(field);
    }
    const int biomes = 2 + int(p_rng() % uint32_t(MAX_BIOMES - 1));
    for (int b = 0; b < biomes; b++) {
        Climate::Biome biome;
        biome.name = ov::format("b%d", b);
        // Each field at most once, in field order (validate's rule), at least one for every biome but the default.
        if (b + 1 < biomes) {
            const int first = int(p_rng() % uint32_t(p_fields));
            for (int f = 0; f < p_fields; f++) {
                if (f == first || (p_rng() & 1u) != 0) {
                    biome.at.push_back(Axis{ f, centre(p_rng), 1.0f / span(p_rng) });
                }
            }
        }
        t.biomes.push_back(biome);
    }
    return t;
}

// The weights in double precision, values only (the reference of the gradients by finite differences): the model of
// ClimateTest::weights, k = the product over the axes of smoothstep(clamp((1 - u^2) 2)), the default's (1 - sum k)^2,
// each over their sum.
void reference_weights(const Table &p_t, const double *p_x, double *r_w) {
    const size_t n = p_t.biomes.size();
    double sum = 0.0;
    for (size_t b = 0; b + 1 < n; b++) {
        double k = 1.0;
        for (const Axis &a : p_t.biomes[b].at) {
            const double u = (p_x[a.field] - double(a.centre)) * double(a.inv_span);
            const double s = std::clamp((1.0 - u * u) * double(KERNEL_FADE), 0.0, 1.0);
            k *= s * s * (3.0 - 2.0 * s);
        }
        r_w[b] = k;
        sum += r_w[b];
    }
    r_w[n - 1] = sum < 1.0 ? (1.0 - sum) * (1.0 - sum) : 0.0;
    const double total = sum + r_w[n - 1];
    for (size_t b = 0; b < n; b++) {
        r_w[b] /= total;
    }
}

} // namespace

TEST_CASE("Climate: make_fbm's octaves are make_octave keyed by their index") {
    const ExactNoise::OctaveSet set = ExactNoise::make_fbm(false, 0xC0FFEEu, 20, 6, 300.0, 0.5);
    for (size_t i = 0; i < set.octaves.size(); i++) {
        const ExactNoise::Octave o = ExactNoise::make_octave(0xC0FFEEu, uint32_t(i), 20 - int(i), set.octaves[i].amp);
        CHECK(o.seed == set.octaves[i].seed);
        CHECK(o.offset == set.octaves[i].offset);
        CHECK(o.k == set.octaves[i].k);
    }
}

TEST_CASE("Climate: the plain preset (the example before shaped bands) parses at 5 cm at any LOD count, its ranges keeping the ground's octaves") {
    ov::Value data;
    CLIMATE_REQUIRE(load_table("res://climate/plain.json", data) == ov::str());
    GroundFixture g(0.05, 15);
    Table t;
    Terrain::LevelProgram prog;
    CHECK(compile(data, g.ground, t, prog) == ov::str());
    CLIMATE_REQUIRE(t.fields.size() == 3);
    CLIMATE_REQUIRE(t.biomes.size() == 7);
    // 3.2 km .. 26 km at 5 cm: k = 16 .. 19, coarsest first, amplitude 0.5 times 0.7 per finer octave; the
    // continental range (52 km ..) is past the ground's top octave (19) at 15 LODs.
    const Field &temp = t.fields[0];
    CLIMATE_REQUIRE(temp.bands.size() == 4);
    CHECK(temp.bands.front().k == 19);
    CHECK(temp.bands.back().k == 16);
    CHECK(temp.bands[1].amp == float(0.5 * 0.7));
    CHECK(temp.terms.empty());
    // Seeds are keyed by k: the same octave of the same field always gets the same lattice.
    CHECK(temp.bands[0].seed == ExactNoise::make_octave(temp.seed, 19u, 19, 1.0f).seed);
    CHECK(t.biomes.back().name == "plain");
    CHECK(t.biomes.back().at.empty());
    CHECK(t.biomes[0].debug_color == 0xC03818u);
    // The ground's notes, one per octave k = 3 .. 19 whatever biome mixes them (one node each in the program).
    CHECK(ClimateTest::ground_note_count(prog) == 17);
    CHECK(ClimateTest::note_top_k(prog) == 19);

    // 25 LODs: the fields' continental octaves k = 20 .. 29 join, each at the amplitude it has at any LOD count; the
    // notes stop at their ranges' top (k = 19), so the biomes play the same octaves as at 15.
    GroundFixture g25(0.05, 25);
    Table t25;
    Terrain::LevelProgram prog25;
    CHECK(compile(data, g25.ground, t25, prog25) == ov::str());
    CLIMATE_REQUIRE(t25.fields[0].bands.size() == 14);
    CHECK(t25.fields[0].bands.front().k == 29);
    CHECK(t25.fields[0].bands[9].k == 20);
    CHECK(t25.fields[0].bands[9].amp == 0.35f);
    CHECK(t25.fields[0].bands[10].amp == temp.bands[0].amp);
    CHECK(ClimateTest::ground_note_count(prog25) == 17);

    // 14 LODs: the ranges lose k = 19; a single wavelength past the top is refused.
    GroundFixture g14(0.05, 14);
    Table t14;
    Terrain::LevelProgram prog14;
    CHECK(compile(data, g14.ground, t14, prog14) == ov::str());
    CHECK(t14.fields[0].bands.size() == 3);
    CHECK(ClimateTest::ground_note_count(prog14) == 16);
    Table single;
    const char *past_top = R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 26000, "amp": 1}]}], "biomes": [{"name": "a"}]})";
    CHECK(ov::contains(parse_json(past_top, g14.ground, single), "outside the cascade's range"));
}

TEST_CASE("ClimateReport: check takes a table's text, accepts a good one, refuses bad JSON and a refused table with the reason") {
    const char *good = R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 5000, "amp": 1}]}],
        "biomes": [{"name": "a", "at": {"t": [0, 0.5]}}, {"name": "b"}]})";
    CHECK(ClimateReport::check(good, 0.05, 25) == ov::str());
    CHECK(ov::contains(ClimateReport::check("{\"fields\": [", 0.05, 25), "not a JSON object"));
    CHECK(ov::contains(ClimateReport::check("[1, 2]", 0.05, 25), "not a JSON object"));
    const char *zero_span = R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 5000, "amp": 1}]}],
        "biomes": [{"name": "a", "at": {"t": [0, 0]}}, {"name": "b"}]})";
    CHECK(ov::contains(ClimateReport::check(zero_span, 0.05, 25), "span must be > 0"));
}

// The biome mixer's maps are the engine's: a world in memory sampled by the report's own code, the same bits twice, a
// field per field, a refused world with the reason.
TEST_CASE("ClimateReport: preview gives a world's maps from its text, the same twice, or why not") {
    ov::str text;
    CLIMATE_REQUIRE(ov::read_file("res://climate/example.json", text));
    ov::str json_error;
    const ov::Value world = ov::parse_json(text, json_error);
    CLIMATE_REQUIRE(world.is_object());
    const ClimateReport::Preview a = ClimateReport::preview(world, "res://climate/", 20.0, 48, true, 0.0, 0.0);
    CHECK(a.error == ov::str());
    CLIMATE_REQUIRE(a.size == 48);
    CHECK(a.biomes.size() == size_t(48 * 48 * 3));
    CHECK(a.shaded.size() == size_t(48 * 48 * 3));
    CHECK(a.altitude_m.size() == size_t(48 * 48));
    CHECK(a.fields.size() == size_t(48 * 48) * a.field_names.size());
    CHECK(!a.biome_names.empty());
    CHECK(a.spacing_m * 48.0 > 10000.0);
    CHECK(a.spacing_m * 48.0 < 40000.0);
    const ClimateReport::Preview b = ClimateReport::preview(world, "res://climate/", 20.0, 48, true, 0.0, 0.0);
    CHECK(a.biomes == b.biomes);
    CHECK(a.relief_m == double(std::abs(*std::max_element(a.altitude_m.begin(), a.altitude_m.end(), [](float p_x, float p_y) { return std::abs(p_x) < std::abs(p_y); }))));
    CHECK(a.shaded == b.shaded);
    CHECK(a.altitude_m == b.altitude_m);
    MESSAGE("preview: ", a.us_per_point, " us per point, lattice ", a.lattice);
    // Off the surface (the tangent plane): no trace, every altitude 0.
    const ClimateReport::Preview plane = ClimateReport::preview(world, "res://climate/", 20.0, 16, false, 0.0, 0.0);
    CHECK(std::all_of(plane.altitude_m.begin(), plane.altitude_m.end(), [](float p_a) { return p_a == 0.0f; }));
    CHECK(ov::contains(ClimateReport::preview(ClimateTest::json(R"({"engine": {"voxel_m": 0.05}})"), "", 20.0, 16, true, 0.0, 0.0).error, "no fields"));
    CHECK(ov::contains(ClimateReport::preview(ClimateTest::json(R"({"trees": 1})"), "", 20.0, 16, true, 0.0, 0.0).error, "unknown key"));
    // Every mix row at 0: a ground of no notes (its top lattice -1) is drawn over the whole extent, never queried under
    // Λ_0 (the mixer's map flooded "lattice level out of range" and never drew the next edit).
    const ClimateReport::Preview silent = ClimateReport::preview(ClimateTest::json(R"({"engine": {"voxel_m": 0.05, "lods": 15},
        "climate": {"fields": [{"name": "t", "seed": 1, "bands": [{"wavelength_m": 13107.2, "amp": 1}]}]},
        "biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 1638.4, "seed": 1, "amp_m": 10},
            {"id": "mix", "kind": "mix", "rows": [{"in": "n", "x": 0}]}, {"id": "out", "kind": "output", "height": "mix"}]}}]})"),
            "", 20.0, 16, true, 0.0, 0.0);
    CHECK(silent.error == ov::str());
    CHECK(silent.lattice >= 0);
    CHECK(silent.spacing_m * 16.0 > 10000.0);
    CHECK(silent.relief_m == 0.0);
}

TEST_CASE("Climate: malformed tables are refused with the reason") {
    GroundFixture g(0.5, 15);
    struct Case {
        const char *json;
        const char *error;
    };
    const Case cases[] = {
        { R"({"fields": [], "biomes": []})", "1 to 16" },
        { R"({"biomes": [{"name": "a", "colour": 1}]})", "unknown key 'colour'" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "amp": 1}], "terms": [{"source": "h", "x": 1}]}], "biomes": [{"name": "a"}]})",
                "not an earlier field" },
        { R"({"fields": [{"name": "t"}, {"name": "h", "terms": [{"source": "h", "x": 1}]}], "biomes": [{"name": "a"}]})", "not an earlier field" },
        { R"({"fields": [{"name": "t"}, {"name": "t"}], "biomes": [{"name": "a"}]})", "used twice" },
        { R"({"fields": [{"name": "ground"}], "biomes": [{"name": "a"}]})", "reserved" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "amp": 1}, {"wavelength_m": 110, "amp": 1}]}], "biomes": [{"name": "a"}]})",
                "listed twice" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "amp": 1e13}]}], "biomes": [{"name": "a"}]})", "amplitude" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "at": {"t": [0, 0]}}, {"name": "b"}]})", "span must be > 0" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "at": {"t": {"min": 1, "max": 0}}}, {"name": "b"}]})", "span must be > 0" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "at": {"t": 1}}, {"name": "b"}]})", "[centre, span]" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "at": {"t": {"min": 0}}}, {"name": "b"}]})", "[centre, span]" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "where": [{"field": "t", "min": 0, "fade": 1}]}, {"name": "b"}]})", "became 'at'" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a"}, {"name": "b"}]})", "only the last biome" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "at": {"t": [0, 1]}}]})", "the last biome is the default" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "at": {"u": [0, 1]}}, {"name": "b"}]})", "not a field" },
        { R"({"biomes": [{"name": "a", "amp": [{"field": "ground", "wavelength_m": 4000, "x": 2}]}]})", "became the biome's 'graph'" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 1, "seed": 1, "amp_m": 1}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "outside the cascade's range" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1, "amp_m": 1e13}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "over the cap" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "amp_m": 1}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "'seed' is required" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "'amp_m' is required" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1, "amp_m": 1}]}}]})", "an output node" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "o", "kind": "output", "height": "x"}]}}]})", "no node 'x'" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1, "amp_m": 1}, {"id": "n", "kind": "note", "wavelength_m": 400, "seed": 1, "amp_m": 1}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "two nodes are 'n'" },
        { R"({"biomes": [{"name": "a", "height_m": 100}]})", "unknown key 'height_m'" },
        { R"({"biomes": [{"name": "a", "debug_color": "red"}]})", "#rrggbb" },
        { R"({"biomes": [{"name": "a", "debug_color": "#-00001"}]})", "#rrggbb" },
        { R"({"fields": [{"name": "t", "bands": [{"from_m": 1, "to_m": 1e30, "amp": 1}]}], "biomes": [{"name": "a"}]})", "outside [0, 62]" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "to_m": 500, "amp": 1}]}], "biomes": [{"name": "a"}]})", "not both" },
        { R"({"fields": [{"name": "t", "seed": 1e20}], "biomes": [{"name": "a"}]})", "whole number" },
        { R"({"fields": [{"name": "t", "seed": 7.9}], "biomes": [{"name": "a"}]})", "whole number" },
        { R"({"fields": [{"name": "temp/2"}], "biomes": [{"name": "a"}]})", "identifier" },
        { R"({"fields": [{"name": "t"}], "regions": [], "biomes": [{"name": "a"}]})", "unknown key 'regions'" },
        { R"({"fields": [{"name": "t"}], "heights": [], "biomes": [{"name": "a"}]})", "unknown key 'heights'" },
        { R"({"fields": [{"name": "t", "noise": "cell"}], "biomes": [{"name": "a"}]})", "unknown key 'noise'" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "amp": 1, "shape": "ridged"}]}], "biomes": [{"name": "a"}]})", "unknown key 'shape'" },
        { R"({"biomes": [{"name": "a", "terrace": {"step_m": 40}}]})", "unknown key 'terrace'" },
        { R"({"biomes": [{"name": "a", "curve": [[0, 0], [10, 10]]}]})", "became the biome's 'graph'" },
    };
    for (const Case &c : cases) {
        CAPTURE(c.json);
        Table t;
        const ov::str err = parse_json(c.json, g.ground, t);
        const ov::str what = ov::str(c.json) + " -> " + err;
        CHECK_MESSAGE(ov::contains(err, c.error), what);
        CHECK(t.biomes.empty());
    }
    // A curve node over the template ground.
    const Case curves[] = {
        { "[[0, 0]]", "at least 2 points" },
        { "[[0, 0], [0, 10]]", "must increase" },
        { "[[0, 0], [10]]", "[h, R(h)]" },
        { "[[0, 0], [1e12, 4e12]]", "over the cap" },
        { "[[0, 0], [0.5, 5], [1, 0]]", "has a slope up to" },
    };
    for (const Case &c : curves) {
        CAPTURE(c.json);
        ov::Value graph = ClimateTest::template_graph(0.5);
        graph["nodes"].at(graph["nodes"].size() - 1)["height"] = "c";
        graph["nodes"].push_back(ClimateTest::json(ov::str(R"({"id": "c", "kind": "curve", "in": "ground", "points": )") + c.json + "}"));
        Table t;
        const ov::str err = parse_json(R"({"biomes": [{"name": "a", "graph": )" + ov::to_json(graph) + "}]}", g.ground, t);
        const ov::str what = ov::str(c.json) + " -> " + err;
        CHECK_MESSAGE(ov::contains(err, c.error), what);
    }
    // A volume of 0 turns a note off (it is dropped), and the field count is capped.
    Table t;
    Terrain::LevelProgram prog;
    CHECK(parse_json(R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 100, "seed": 1, "amp_m": 1}, {"id": "m", "kind": "mix", "rows": [{"in": "n", "x": 0}]}, {"id": "o", "kind": "output", "height": "m"}]}}]})",
                  g.ground, t, prog) == ov::str());
    CHECK(ClimateTest::ground_note_count(prog) == 0);
    ov::str many = R"({"fields": [)";
    for (int i = 0; i <= MAX_FIELDS; i++) {
        many += ov::format("%s{\"name\": \"f%d\"}", i ? ", " : "", i);
    }
    CHECK(ov::contains(parse_json(many + R"(], "biomes": [{"name": "a"}]})", g.ground, t), ov::format("at most %d", MAX_FIELDS)));
}

TEST_CASE("Climate: the graph kind catalogue is the parser's keys, every kind parses, a shape's phi") {
    GroundFixture g(0.5, 15);
    // Each kind in a graph the parser reads (its node marked @, the note N), accepted, and refused with a key outside its
    // catalogue.
    const std::string note = R"({"id": "n", "kind": "note", "wavelength_m": 400, "seed": 1, "amp_m": 1})";
    struct Case {
        const char *kind;
        const char *nodes;
        const char *height;
    };
    const Case cases[] = {
        { "note", R"({"id": "n", "kind": "note", "wavelength_m": 400, "seed": 1, "amp_m": 1 @})", "n" },
        { "warp", R"({"id": "c", "kind": "note", "wavelength_m": 1600, "seed": 3, "amp_m": 1}, {"id": "x", "kind": "warp", "in": "c", "amount_m": 10 @}, {"id": "n", "kind": "note", "wavelength_m": 400, "seed": 1, "amp_m": 1, "warp": "x"})", "n" },
        { "shape", R"(N, {"id": "x", "kind": "shape", "in": "n", "shape": "ridged" @})", "x" },
        { "multifractal", R"(N, {"id": "c", "kind": "note", "wavelength_m": 1600, "seed": 3, "amp_m": 1}, {"id": "x", "kind": "multifractal", "in": "n", "parent": "c", "low_m": -1, "high_m": 1 @})", "x" },
        { "mix", R"(N, {"id": "x", "kind": "mix", "rows": [{"in": "n", "x": 1}] @})", "x" },
        { "constant", R"(N, {"id": "c", "kind": "constant", "value_m": 5 @}, {"id": "x", "kind": "add", "a": "n", "b": "c"})", "x" },
        { "field", R"(N, {"id": "c", "kind": "field", "field": "t" @}, {"id": "x", "kind": "multiply", "a": "n", "b": "c"})", "x" },
        { "subtract", R"(N, {"id": "x", "kind": "subtract", "a": "n", "b": "n" @})", "x" },
        { "smax", R"(N, {"id": "x", "kind": "smax", "a": "n", "b": "n", "k_m": 3 @})", "x" },
        { "curve", R"(N, {"id": "x", "kind": "curve", "in": "n", "points": [[-100, -100], [100, 120]] @})", "x" },
        { "output", "N", "n" },
    };
    for (const Case &c : cases) {
        CAPTURE(c.kind);
        CHECK(graph_kind(c.kind) != nullptr);
        for (const bool extra : { false, true }) {
            std::string nodes = c.nodes;
            const size_t n_at = nodes.find('N');
            if (n_at != std::string::npos) {
                nodes.replace(n_at, 1, note);
            }
            const size_t at = nodes.find('@');
            if (at != std::string::npos) {
                nodes.replace(at, 1, extra ? ", \"zz\": 1" : "");
            }
            const bool on_out = std::string(c.kind) == "output" && extra;
            const std::string out = std::string(R"({"id": "o", "kind": "output", "height": ")") + c.height + "\"" + (on_out ? ", \"zz\": 1" : "") + "}";
            const std::string json = R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 5000, "amp": 1}]}], "biomes": [{"name": "a", "graph": {"nodes": [)" +
                    nodes + ", " + out + "]}}]}";
            CAPTURE(json);
            ov::str json_error;
            Table t;
            const ov::str err = compile(ov::parse_json(json.c_str(), json_error), g.ground, t);
            if (extra) {
                CHECK(err.find("unknown key 'zz'") != ov::str::npos);
            } else {
                CHECK(err == "");
            }
        }
    }
    CHECK(graph_kind("carve") == nullptr);
    // Ridged at its default offset: (1 - |n|)^2, |n| smoothed (0 at n = 0); an offset on a note refused.
    ov::str json_error;
    Shape s;
    CHECK(parse_shape(ov::parse_json(R"({"shape": "ridged"})", json_error), false, s) == "");
    CHECK(s.c[0] == 1.0f);
    CHECK(s.c[2] == -2.0f);
    CHECK(s.c[3] == 1.0f);
    CHECK(shape_phi(s, 0.0) == 1.0);
    const double sabs = std::sqrt(1.0 + double(Cascade::SMOOTH_ABS_E) * Cascade::SMOOTH_ABS_E) - Cascade::SMOOTH_ABS_E;
    CHECK(shape_phi(s, -1.0) == doctest::Approx(2.0 - 2.0 * sabs));
    CHECK(parse_shape(ov::parse_json(R"({"shape": "plain", "offset": 1})", json_error), false, s).find("on a note") != ov::str::npos);
    CHECK(parse_shape(ov::parse_json(R"({"shape": "plain", "offset": 1})", json_error), true, s) == "");
    CHECK(name_hash("") == 2166136261u);
}

// The compiler's cap (a planet's relief, Ground::amp_cap): the output's bound on a note just under it compiles, just over
// it is refused, for a plain and a shaped note (the bound per metre of amplitude: the program's own at 1 m).
TEST_CASE("Climate: a biome's shaped amplitude is held to the cap") {
    const Ground ground{ 0.5, 19, 1000.0f, INFINITY };
    for (const char *shape : { "\"plain\"", "\"ridged\"" }) {
        CAPTURE(shape);
        auto table = [&](double p_amp_m) {
            return R"({"biomes": [{"name": "a", )" + ClimateTest::note_graph(ov::format(R"("wavelength_m": 4096, "seed": 1, "amp_m": %s)", p_amp_m), "\"shape\": " + ov::str(shape)) +
                    "}]}";
        };
        Table t;
        Terrain::LevelProgram prog;
        CLIMATE_REQUIRE(parse_json(table(1.0), ground, t, prog) == ov::str());
        const size_t o = size_t(prog.output);
        const double cap_m = 1000.0 / ov::max(std::fabs(prog.lo[o]), std::fabs(prog.hi[o]));
        CHECK(parse_json(table(0.99 * cap_m), ground, t) == ov::str());
        CHECK(ov::contains(parse_json(table(1.01 * cap_m), ground, t), "over the cap"));
    }
}

// The compiler keeps one note per (octave, seed, stretch) whatever biome mixes it: two biomes on the template ground
// share its notes; another seed on one of its octaves is a second note there.
TEST_CASE("Climate: the compiler shares notes between biomes") {
    GroundFixture g(0.05, 15);
    const ov::str ground = ClimateTest::ground_graph(0.05);
    const ov::str fields = R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 26000, "amp": 1}]}], "biomes": [)";
    Table t;
    Terrain::LevelProgram same, other;
    CLIMATE_REQUIRE(parse_json(fields + R"({"name": "a", "at": {"t": [0, 1]}, )" + ground + R"(}, {"name": "b", )" + ground + "}]}", g.ground, t, same) == ov::str());
    CHECK(ClimateTest::ground_note_count(same) == 17);
    const ov::str reseeded = R"({"name": "b", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 204.8, "seed": 7, "amp_m": 25.6},
            {"id": "o", "kind": "output", "height": "n"}]}})";
    CLIMATE_REQUIRE(parse_json(fields + R"({"name": "a", "at": {"t": [0, 1]}, )" + ground + "}, " + reseeded + "]}", g.ground, t, other) == ov::str());
    CHECK(ClimateTest::ground_note_count(other) == 18);
}

TEST_CASE("Climate: weights are a partition of unity, in [0, 1], without denormals, gradients by the chain rule") {
    std::mt19937 rng(0xC11A7Eu);
    std::uniform_real_distribution<float> value(-1.5f, 1.5f);
    std::uniform_real_distribution<float> slope(-2.0f, 2.0f);
    int checked = 0;
    double worst_sum = 0.0;
    double worst_grad = 0.0;
    for (int table = 0; table < 200; table++) {
        const int nf = 1 + int(rng() % uint32_t(MAX_FIELDS));
        const Table t = random_table(rng, nf);
        for (int sample = 0; sample < 200; sample++) {
            Float4 x[MAX_FIELDS] = {};
            double xd[MAX_FIELDS] = {};
            double gd[MAX_FIELDS] = {};
            for (int f = 0; f < nf; f++) {
                // Every 4th sample also exercises tiny values (flushed below 2^-64).
                const float tiny = sample % 4 == 0 ? std::ldexp(1.0f, -70 - f) : 0.0f;
                x[f] = Float4{ slope(rng), tiny, 0.0f, value(rng) };
                xd[f] = double(x[f].w);
                gd[f] = double(x[f].x);
            }
            Float4 w[MAX_BIOMES];
            ClimateTest::weights(t, x, w);
            double sum = 0.0;
            bool bad = false;
            for (size_t b = 0; b < t.biomes.size(); b++) {
                bad |= subnormal(w[b]) || !(w[b].w >= 0.0f && w[b].w <= 1.0f);
                sum += double(w[b].w);
            }
            CHECK_FALSE(bad);
            worst_sum = std::max(worst_sum, std::fabs(sum - 1.0));
            // d w / dt along x(t) = x + t g, central differences of the double reference.
            const double eps = 1e-6;
            double xp[MAX_FIELDS], xm[MAX_FIELDS], wp[MAX_BIOMES], wm[MAX_BIOMES];
            for (int f = 0; f < nf; f++) {
                xp[f] = xd[f] + eps * gd[f];
                xm[f] = xd[f] - eps * gd[f];
            }
            reference_weights(t, xp, wp);
            reference_weights(t, xm, wm);
            for (size_t b = 0; b < t.biomes.size(); b++) {
                const double fd = (wp[b] - wm[b]) / (2.0 * eps);
                worst_grad = std::max(worst_grad, std::fabs(double(w[b].x) - fd) / (1.0 + std::fabs(fd)));
            }
            checked++;
        }
    }
    CHECK(checked == 40000);
    CHECK(worst_sum < 1e-5);
    CHECK(worst_grad < 2e-3);
}

TEST_CASE("Climate: a biome is whole at its centre and gone a span away, overlaps blend, the default fills the gaps, the pick's ties") {
    Table t;
    Field f;
    f.name = "t";
    t.fields.push_back(f);
    // "hot" at 0.5 and "cold" at -0.5, both span 0.75 (they overlap around 0), then the default.
    auto biome = [](const char *p_name, float p_centre) {
        Climate::Biome b;
        b.name = p_name;
        b.at.push_back(Axis{ 0, p_centre, 1.0f / 0.75f });
        return b;
    };
    t.biomes = { biome("hot", 0.5f), biome("cold", -0.5f), Climate::Biome() };
    t.biomes[2].name = "default";
    auto at = [&](float p_x, Float4 r_w[MAX_BIOMES]) {
        Float4 x[MAX_FIELDS] = {};
        x[0] = Float4{ 1.0f, 0.0f, 0.0f, p_x };
        ClimateTest::weights(t, x, r_w);
    };
    auto pick = [](const Table &p_t, const Float4 *p_w, const Vector3i64 &p_p, int p_m) { return pick_of(int(p_t.biomes.size()), p_w, p_p, p_m); };
    Float4 w[MAX_BIOMES];
    // At a centre the biome is whole, exactly, its gradient 0 (the kernel is flat there); far from both, the default.
    const float expect[][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, 1 } };
    const float xs[] = { 0.5f, -0.5f, 1.5f, -1.25f };
    for (int i = 0; i < 4; i++) {
        at(xs[i], w);
        for (int b = 0; b < 3; b++) {
            CAPTURE(i);
            CAPTURE(b);
            CHECK(float_bits(w[b].w) == float_bits(expect[i][b]));
            CHECK(w[b].x == 0.0f);
        }
        CHECK(pick(t, w, Vector3i64(1, 2, 3), 0) == int(std::max_element(expect[i], expect[i] + 3) - expect[i]));
    }
    // Between them both kernels are partial and blend, the hot side leaning hot; their sum is over 1, the default out.
    at(0.2f, w);
    CHECK(w[0].w > w[1].w);
    CHECK(w[1].w > 0.0f);
    CHECK(w[2].w == 0.0f);
    CHECK(std::fabs(double(w[0].w) + double(w[1].w) - 1.0) < 1e-6);
    // Past hot's centre only hot is partial: the default takes what it leaves.
    at(1.1f, w);
    CHECK(w[0].w > w[2].w);
    CHECK(w[2].w > 0.0f);
    CHECK(w[1].w == 0.0f);
    CHECK(std::fabs(double(w[0].w) + double(w[2].w) - 1.0) < 1e-6);
    // At 0 the two kernels are the same bits: hot and cold tie, the hash chooses.
    at(0.0f, w);
    CHECK(float_bits(w[0].w) == float_bits(w[1].w));
    CHECK(w[0].w > w[2].w);
    // The tie goes by ExactNoise::hash_cell of the point's LOD0 voxel, high words included: the same point gives the
    // same pick from every lattice that has it.
    int seen[2] = { 0, 0 };
    for (int64_t i = 0; i < 16; i++) {
        const Vector3i64 p(i, -i, i * (int64_t(1) << 33) + 1);
        const int got = pick(t, w, p, 5);
        CHECK(got == int(ExactNoise::hash_cell(Vector3i64(p.x * 32, p.y * 32, p.z * 32), PICK_SEED) % 2u));
        CHECK(pick(t, w, Vector3i64(p.x * 2, p.y * 2, p.z * 2), 4) == got);
        CHECK(pick(t, w, Vector3i64(p.x * 32, p.y * 32, p.z * 32), 0) == got);
        seen[got]++;
    }
    CHECK(seen[0] > 0);
    CHECK(seen[1] > 0);
}

// Several notes on one lattice (two seeds of one octave) are each inserted: box runs equal point queries bit for bit,
// and the ground is the two notes' sum (within the rounding of the adds). A field holds one band per lattice: a second
// is refused.
TEST_CASE("Cascade: several ground notes on one lattice, one band per field") {
    GroundFixture g(0.05, 15);
    Table t;
    CHECK(ov::contains(parse_json(R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 400, "amp": 1}, {"wavelength_m": 400, "amp": 2}]}], "biomes": [{"name": "a"}]})",
                               g.ground, t),
            "twice"));
    const auto note = [](const char *p_id, int p_seed, double p_amp_m) {
        return ov::format(R"({"id": "%s", "kind": "note", "wavelength_m": 25.6, "seed": %d, "amp_m": %s})", p_id, int64_t(p_seed), ov::num(p_amp_m));
    };
    const auto table = [](const ov::str &p_nodes, const char *p_out) {
        return R"({"biomes": [{"name": "a", "graph": {"nodes": [)" + p_nodes + R"(, {"id": "o", "kind": "output", "height": ")" + p_out + "\"}]}}]}";
    };
    Terrain::LevelProgram one, other, two;
    CLIMATE_REQUIRE(parse_json(table(note("a", 11, 2.0), "a"), g.ground, t, one) == ov::str());
    CLIMATE_REQUIRE(parse_json(table(note("b", 12, 1.25), "b"), g.ground, t, other) == ov::str());
    CLIMATE_REQUIRE(parse_json(table(note("a", 11, 2.0) + ", " + note("b", 12, 1.25) + R"(, {"id": "m", "kind": "mix", "rows": [{"in": "a"}, {"in": "b"}]})", "m"), g.ground, t, two) ==
            ov::str());
    CHECK(ClimateTest::note_count(two) == 2);
    double worst = 0.0;
    for (int i = 0; i < 32; i++) {
        const Vector3i64 p(i * 37 - 500, i * 11, 900 - i * 23);
        const Float4 a = Terrain::point(one, p, 2).ground, b = Terrain::point(other, p, 2).ground, ab = Terrain::point(two, p, 2).ground;
        const float sa[4] = { a.x, a.y, a.z, a.w }, sb[4] = { b.x, b.y, b.z, b.w }, sab[4] = { ab.x, ab.y, ab.z, ab.w };
        for (int ch = 0; ch < 4; ch++) {
            worst = std::max(worst, std::fabs(double(sab[ch]) - double(sa[ch]) - double(sb[ch])) / (1.0 + std::fabs(double(sab[ch]))));
        }
    }
    CHECK(worst < 1e-5);
    for (const int64_t base : { int64_t(-40), (int64_t(1) << 32) - 3 }) {
        Cascade::Box box;
        for (int a = 0; a < 3; a++) {
            box.lo[a] = a == 1 ? 7 : (a == 0 ? base : -base);
            box.hi[a] = box.lo[a] + 3;
        }
        const Terrain::Values v = Terrain::run(two, box, 1);
        int64_t bad = 0;
        Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) { bad += ExactNoise::same_bits(Terrain::point(two, p_p, 1).ground, v.at(two.output, p_i)) ? 0 : 1; });
        CHECK(bad == 0);
    }
}
