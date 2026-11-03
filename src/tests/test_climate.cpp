#include "test_climate_common.hpp"

#include "cascade_glsl.hpp"
#include "climate.hpp"
#include "climate_report.hpp"
#include "sdf_cascade.hpp"

#include "ov/files.hpp"

#include <algorithm>
#include <cmath>
#include <random>

using namespace Climate;
using ClimateTest::box_point_mismatches;
using ClimateTest::differing_floats;
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

// A field's bands alone as a cascade spec (the reference of its group in the combined spec).
Cascade::Spec own_spec(const Field &p_field) {
    Cascade::Spec spec;
    spec.bands = p_field.bands;
    return spec;
}

ov::str parse_json(const ov::str &p_json, const Ground &p_ground, Table &r_table) {
    const ov::Value v = ClimateTest::json(p_json);
    CHECK(v.is_object());
    return v.is_object() ? parse(v, p_ground, r_table) : ov::str("not a JSON object");
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
// Climate::weights, k = the product over the axes of smoothstep(clamp((1 - u^2) 2)), the default's (1 - sum k)^2,
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
    CHECK(parse(data, g.ground, t) == ov::str());
    CLIMATE_REQUIRE(t.fields.size() == 3);
    CLIMATE_REQUIRE(t.biomes.size() == 7);
    // 3.2 km .. 26 km at 5 cm: k = 16 .. 19, coarsest first, amplitude 0.5 times 0.7 per finer octave; the
    // continental range (52 km ..) is past the ground's top octave (19) at 15 LODs.
    const Field &temp = t.fields[0];
    CLIMATE_REQUIRE(temp.bands.size() == 4);
    CHECK(temp.bands.front().k == 19);
    CHECK(temp.bands.back().k == 16);
    CHECK(temp.bands[1].amp == float(0.5 * 0.7));
    CLIMATE_REQUIRE(temp.terms.size() == 1);
    CHECK(temp.terms[0].source == GROUND);
    CHECK(temp.terms[0].coef == float(-0.00003 * g.ground.voxel_size));
    // Seeds are keyed by k: the same octave of the same field always gets the same lattice.
    CHECK(temp.bands[0].seed == ExactNoise::make_octave(temp.seed, 19u, 19, 1.0f).seed);
    CHECK(t.biomes.back().name == "plain");
    CHECK(t.biomes.back().at.empty());
    CHECK(t.biomes[0].debug_color == 0xC03818u);
    // The ground's notes, one per octave k = 3 .. 19 whatever biome mixes them; every biome lists each modulated one.
    CHECK(t.notes.size() == 17);
    CHECK(t.notes.front().k == 19);
    for (const Climate::Biome &b : t.biomes) {
        CHECK(b.amps.size() == t.biomes[0].amps.size());
    }
    CHECK(t.biomes[0].amps.size() == 17);

    // 25 LODs: the fields' continental octaves k = 20 .. 29 join, each at the amplitude it has at any LOD count; the
    // notes stop at their ranges' top (k = 19), so the biomes play the same octaves as at 15.
    GroundFixture g25(0.05, 25);
    Table t25;
    CHECK(parse(data, g25.ground, t25) == ov::str());
    CLIMATE_REQUIRE(t25.fields[0].bands.size() == 14);
    CHECK(t25.fields[0].bands.front().k == 29);
    CHECK(t25.fields[0].bands[9].k == 20);
    CHECK(t25.fields[0].bands[9].amp == 0.35f);
    CHECK(t25.fields[0].bands[10].amp == temp.bands[0].amp);
    CHECK(t25.notes.size() == 17);

    // 14 LODs: the ranges lose k = 19; a single wavelength past the top is refused.
    GroundFixture g14(0.05, 14);
    Table t14;
    CHECK(parse(data, g14.ground, t14) == ov::str());
    CHECK(t14.fields[0].bands.size() == 3);
    CHECK(t14.notes.size() == 16);
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
                "or an earlier field" },
        { R"({"fields": [{"name": "t"}, {"name": "h", "terms": [{"source": "h", "x": 1}]}], "biomes": [{"name": "a"}]})", "or an earlier field" },
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
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "from_m": 4000, "to_m": 9000, "seed": 1, "amp_m": 1e13}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "outside [2^-16, 2^40]" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1, "amp_m": 1}, {"id": "m", "kind": "mix", "rows": [{"in": "n", "x": -1}]}, {"id": "o", "kind": "output", "height": "m"}]}}]})",
                ">= 0" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1, "amp_m": 1}, {"id": "m", "kind": "mix", "rows": [{"in": "n"}, {"in": "n"}]}, {"id": "o", "kind": "output", "height": "m"}]}}]})",
                "mixed twice" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "amp_m": 1}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "'seed' is required" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "'steepness'" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1, "amp_m": 1}]}}]})", "an output node" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "o", "kind": "output", "height": "x"}]}}]})", "no node 'x'" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1, "amp_m": 1}, {"id": "n", "kind": "note", "wavelength_m": 400, "seed": 1, "amp_m": 1}, {"id": "o", "kind": "output", "height": "n"}]}}]})",
                "two nodes are 'n'" },
        { R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 4000, "seed": 1, "amp_m": 1}, {"id": "m", "kind": "multifractal", "in": "n", "gain": 1}, {"id": "s", "kind": "shape", "in": "m", "shape": "ridged"}, {"id": "o", "kind": "output", "height": "s"}]}}]})",
                "in that order" },
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
        { "[[0, 0], [1e12, 4e12]]", "over the ground's cap" },
        { "[[0, 0], [0.5, 5], [1, 0]]", "has a slope up to" },
        { "[[1e7, 1e7], [2e7, 2e7]]", "it would be a constant" },
    };
    for (const Case &c : curves) {
        CAPTURE(c.json);
        ov::Value graph = Climate::template_graph(0.5);
        graph["nodes"].at(1)["height"] = "c";
        graph["nodes"].push_back(ClimateTest::json(ov::str(R"({"id": "c", "kind": "curve", "in": "ground", "points": )") + c.json + "}"));
        Table t;
        const ov::str err = parse_json(R"({"biomes": [{"name": "a", "graph": )" + ov::to_json(graph) + "}]}", g.ground, t);
        const ov::str what = ov::str(c.json) + " -> " + err;
        CHECK_MESSAGE(ov::contains(err, c.error), what);
    }
    // A volume of 0 turns a note off (it is dropped), and the field count is capped.
    Table t;
    CHECK(parse_json(R"({"biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "from_m": 4, "to_m": 100, "seed": 1, "amp_m": 1}, {"id": "m", "kind": "mix", "rows": [{"in": "n", "x": 0}]}, {"id": "o", "kind": "output", "height": "m"}]}}]})",
                  g.ground, t) == ov::str());
    CHECK(t.notes.empty());
    ov::str many = R"({"fields": [)";
    for (int i = 0; i <= MAX_FIELDS; i++) {
        many += ov::format("%s{\"name\": \"f%d\"}", i ? ", " : "", i);
    }
    CHECK(ov::contains(parse_json(many + R"(], "biomes": [{"name": "a"}]})", g.ground, t), ov::format("at most %d", MAX_FIELDS)));
}

// validate's cap (a planet's relief, Ground::amp_cap): a biome's shaped amplitude on a note just under it parses, just
// over it is refused, for a plain and a shaped note.
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
        { "warp", R"(N, {"id": "x", "kind": "warp", "in": "n", "fields": ["t", "t", "t"], "amount_m": 10 @})", "x" },
        { "shape", R"(N, {"id": "x", "kind": "shape", "in": "n", "shape": "ridged" @})", "x" },
        { "multifractal", R"(N, {"id": "x", "kind": "multifractal", "in": "n", "gain": 2 @})", "x" },
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
            const ov::str err = parse(ov::parse_json(json.c_str(), json_error), g.ground, t);
            if (extra) {
                CHECK(err.find("unknown key 'zz'") != ov::str::npos);
            } else {
                CHECK(err == "");
            }
        }
    }
    CHECK(graph_kind("carve") == nullptr);
    // Ridged at its default offset: (1 - |n|)^2, |n| smoothed (0 at n = 0); an offset outside a multifractal refused.
    ov::str json_error;
    Shape s;
    CHECK(parse_shape(ov::parse_json(R"({"shape": "ridged"})", json_error), false, s) == "");
    CHECK(s.c[0] == 1.0f);
    CHECK(s.c[2] == -2.0f);
    CHECK(s.c[3] == 1.0f);
    CHECK(shape_phi(s, 0.0) == 1.0);
    const double sabs = std::sqrt(1.0 + double(Cascade::SMOOTH_ABS_E) * Cascade::SMOOTH_ABS_E) - Cascade::SMOOTH_ABS_E;
    CHECK(shape_phi(s, -1.0) == doctest::Approx(2.0 - 2.0 * sabs));
    CHECK(parse_shape(ov::parse_json(R"({"shape": "plain", "offset": 1})", json_error), false, s).find("multifractal") != ov::str::npos);
    CHECK(parse_shape(ov::parse_json(R"({"shape": "plain", "offset": 1})", json_error), true, s) == "");
    CHECK(name_hash("") == 2166136261u);
}

TEST_CASE("Climate: a biome's shaped amplitude is held to the cap") {
    const Ground ground{ 0.5, 19, 1000.0f, INFINITY };
    for (const char *shape : { "\"plain\"", "\"ridged\"" }) {
        CAPTURE(shape);
        const double bound = shape_bound(ov::str(shape) == "\"plain\"" ? Shape() : Shape{ { 1.0f, 0.0f, -2.0f, 1.0f } });
        auto table = [&](double p_amp_m) {
            return R"({"biomes": [{"name": "a", )" + ClimateTest::note_graph(ov::format(R"("wavelength_m": 4096, "seed": 1, "amp_m": %s)", p_amp_m), "\"shape\": " + ov::str(shape)) +
                    "}]}";
        };
        const double cap_m = 1000.0 * 0.5 / bound;
        Table t;
        CHECK(parse_json(table(0.99 * cap_m), ground, t) == ov::str());
        CHECK(ov::contains(parse_json(table(1.01 * cap_m), ground, t), "over the cap"));
    }
}

// The skirts' bound (SDFCascade::octave_step): a note's own amplitude without biomes, the largest biome's with them.
TEST_CASE("Climate: max_amp is the note's amplitude in a table without biomes") {
    Cascade::Band band;
    band.k = 10;
    band.amp = 3.5f;
    Table t;
    CHECK(max_amp(t, band) == 3.5f);
    t.biomes.resize(2);
    t.biomes[1].amps.push_back(Amp{});
    t.biomes[1].amps[0].k = 10;
    t.biomes[1].amps[0].amp = 7.0f;
    CHECK(max_amp(t, band) == 7.0f);
}

// The compiler keeps one note per (octave, seed, stretch) whatever biome mixes it: two biomes on the template ground
// share its notes, a note every biome plays plain at one amplitude is a constant band, and a biome's volume makes it
// modulated, the biomes that don't mix it listing it at 0.
TEST_CASE("Climate: the compiler shares notes between biomes, constant where every biome plays them alike") {
    GroundFixture g(0.05, 15);
    const ov::str ground = ClimateTest::ground_graph(0.05);
    const ov::str fields = R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 26000, "amp": 1}]}], "biomes": [)";
    Table same;
    CLIMATE_REQUIRE(parse_json(fields + R"({"name": "a", "at": {"t": [0, 1]}, )" + ground + R"(}, {"name": "b", )" + ground + "}]}", g.ground, same) == ov::str());
    CHECK(same.notes.size() == 17);
    CHECK(same.biomes[0].amps.empty());
    CHECK(same.biomes[1].amps.empty());
    CHECK(combined_spec(same, Cascade::Spec()).field_range().empty());
    // b mixes the ground's octave 12 alone, at volume 2: one note with a's (the same seed and stretch), every note
    // modulated now (b's volume on 12; the others a's alone), b listing a's other 16 at 0.
    Table two;
    const ov::str own = R"({"name": "b", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 204.8, "seed": 59943, "steepness": 0.125, "max_amp_m": 3276.8},
            {"id": "m", "kind": "mix", "rows": [{"in": "n", "x": 2}]}, {"id": "o", "kind": "output", "height": "m"}]}})";
    CLIMATE_REQUIRE(parse_json(fields + R"({"name": "a", "at": {"t": [0, 1]}, )" + ground + "}, " + own + "]}", g.ground, two) == ov::str());
    CHECK(two.notes.size() == 17);
    CLIMATE_REQUIRE(two.biomes[0].amps.size() == 17);
    CLIMATE_REQUIRE(two.biomes[1].amps.size() == 17);
    auto note_of = [](const Table &p_t, int p_k, int p_nth = 0) {
        for (const Cascade::Band &b : p_t.notes) {
            if (b.k == p_k && p_nth-- == 0) {
                return &b;
            }
        }
        return static_cast<const Cascade::Band *>(nullptr);
    };
    const Amp *a12 = find_amp(two.biomes[0], *note_of(two, 12));
    const Amp *b12 = find_amp(two.biomes[1], *note_of(two, 12));
    CLIMATE_REQUIRE((a12 && b12));
    CHECK(b12->amp == 2.0f * a12->amp);
    CHECK(find_amp(two.biomes[1], *note_of(two, 13))->amp == 0.0f);
    // Another seed on octave 12: a second note there (Cascade::MAX_LEVEL_NOTES), its own; a plays only the first, b
    // only the second, both listed in both biomes (a level is modulated as a whole).
    Table other;
    const ov::str reseeded = R"({"name": "b", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 204.8, "seed": 7, "steepness": 0.125},
            {"id": "o", "kind": "output", "height": "n"}]}})";
    CLIMATE_REQUIRE(parse_json(fields + R"({"name": "a", "at": {"t": [0, 1]}, )" + ground + "}, " + reseeded + "]}", g.ground, other) == ov::str());
    CHECK(other.notes.size() == 18);
    const Cascade::Band *first = note_of(other, 12), *second = note_of(other, 12, 1);
    CLIMATE_REQUIRE((first && second));
    CHECK(first->seed != second->seed);
    CHECK(find_amp(other.biomes[0], *second)->amp == 0.0f);
    CHECK(find_amp(other.biomes[1], *first)->amp == 0.0f);
    CHECK(find_amp(other.biomes[1], *second)->amp == float(0.125 * 4096.0));
    // Two multifractal notes on one octave: refused (the state holds one octave's signal).
    Table two_states;
    const ov::str mf = R"({"name": "b", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 204.8, "seed": 7, "steepness": 0.125},
            {"id": "m", "kind": "multifractal", "in": "n", "gain": 1}, {"id": "o", "kind": "output", "height": "m"}]}})";
    const ov::str mf_a = R"({"name": "a", "at": {"t": [0, 1]}, "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 204.8, "seed": 8, "steepness": 0.125},
            {"id": "m", "kind": "multifractal", "in": "n", "gain": 1}, {"id": "o", "kind": "output", "height": "m"}]}})";
    CHECK(ov::contains(parse_json(fields + mf_a + ", " + mf + "]}", g.ground, two_states), "one multifractal"));
    // One of them at gain 0: specialised, a plain note; generic, still a multifractal node (its text holds any gain).
    ov::str mf_a0 = mf_a;
    mf_a0.replace(mf_a0.find("\"gain\": 1"), 9, "\"gain\": 0");
    Ground generic = g.ground;
    generic.specialize = false;
    CHECK(parse_json(fields + mf_a0 + ", " + mf + "]}", g.ground, two_states) == ov::str());
    CHECK(ov::contains(parse_json(fields + mf_a0 + ", " + mf + "]}", generic, two_states), "one multifractal"));
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
        CHECK(validate(t, Ground()) == ov::str());
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
            weights(t, x, w);
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
    CLIMATE_REQUIRE(validate(t, Ground()) == ov::str());
    auto at = [&](float p_x, Float4 r_w[MAX_BIOMES]) {
        Float4 x[MAX_FIELDS] = {};
        x[0] = Float4{ 1.0f, 0.0f, 0.0f, p_x };
        weights(t, x, r_w);
    };
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

TEST_CASE("Climate: fields are pure functions of the lattice point, LOD-coherent, terms in list order") {
    GroundFixture g(0.05, 15);
    Table t;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [
            {"name": "a", "bands": [{"from_m": 800, "to_m": 26000, "amp": 1}]},
            {"name": "b", "bands": [{"wavelength_m": 3200, "amp": 0.5}], "terms": [{"source": "a", "x": -0.25}]},
            {"name": "c", "terms": [{"source": "ground", "per_m": 0.001}, {"source": "b", "x": 2}]}],
            "biomes": [{"name": "x", )" +
                                    ClimateTest::ground_graph(0.05) +
                                    "}]}",
                            g.ground, t) == ov::str());
    // The template's notes, constant: the combined spec carries the fields, unmodulated, its ground the bare one.
    const Cascade::Spec spec = combined_spec(t, g.sdf.get_spec());
    CHECK(spec.fields.size() == 3);
    CHECK(spec.field_range().empty()); // a modulation for the pick's derive only: no modulated band
    // Lattice indices: times 2^13 (the finest point at m = 12, doubled) they stay within +-2^60 voxels.
    const int64_t bases[] = { 0, 12345, (int64_t(1) << 32) - 3, -(int64_t(1) << 47) + 7, (int64_t(1) << 47) - 11 };
    int coherent = 0;
    for (const int64_t base : bases) {
        for (const int m : { 0, 3, 7, 12 }) {
            const Vector3i64 p(base, -base / 3 + 5, 7 - base);
            Float4 r[MAX_FIELDS];
            fields_at(t, spec, p, m, r);
            // Field a is its own spec; b adds -0.25 a; c is 0.001 * 0.05 ground + 2 b, in that order.
            float ch[Cascade::CHANNELS], gc[Cascade::CHANNELS];
            Cascade::point_into(own_spec(t.fields[0]), p, m, ch);
            CHECK(float_bits(r[0].w) == float_bits(ch[Cascade::VALUE]));
            CHECK(float_bits(r[0].x) == float_bits(ch[Cascade::DX]));
            Cascade::point_into(own_spec(t.fields[1]), p, m, ch);
            CHECK(float_bits(r[1].w) == float_bits(Cascade::flush_value(ch[Cascade::VALUE] + r[0].w * -0.25f)));
            Cascade::point_into(g.sdf.get_spec(), p, m, gc);
            const float c0 = Cascade::flush_value(0.0f + gc[Cascade::VALUE] * float(0.001 * g.ground.voxel_size));
            CHECK(float_bits(r[2].w) == float_bits(Cascade::flush_value(c0 + r[1].w * 2.0f)));
            for (int f = 0; f < 3; f++) {
                CHECK_FALSE(subnormal(r[f]));
            }
            CHECK(r[3].w == 0.0f);
            // a and b have no octave finer than 800 m (k = 14, inserted on Λ_11): below it nothing is inserted, so Λ_m's
            // point 2p is a copy of Λ_{m+1}'s p, the same value and half the gradient (lattice units).
            if (m < 11) {
                Float4 fine[MAX_FIELDS];
                fields_at(t, spec, Vector3i64(p.x * 2, p.y * 2, p.z * 2), m, fine);
                Float4 coarse[MAX_FIELDS];
                fields_at(t, spec, p, m + 1, coarse);
                for (int f = 0; f < 2; f++) {
                    CHECK(float_bits(fine[f].w) == float_bits(coarse[f].w));
                    CHECK(float_bits(fine[f].y) == float_bits(coarse[f].y * 0.5f));
                }
                coherent++;
            }
        }
    }
    CHECK(coherent == 15);
}

namespace {

} // namespace

TEST_CASE("Climate: carried fields and a neutral modulation leave the ground bit for bit") {
    GroundFixture g(0.05, 15);
    const Cascade::Spec &bare = g.sdf.get_spec();
    // Two biomes on the template's notes (constant): the fields carried, the ground untouched.
    const ov::str ground = ClimateTest::ground_graph(0.05);
    Table carried;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]}],
            "biomes": [{"name": "a", "at": {"t": [1, 1.2]}, )" +
                                    ground +
                                    R"(}, {"name": "b", )" + ground + "}]}",
                            g.ground, carried) == ov::str());
    // Only the default biome, the template's notes through a plain shape (modulated): A = 1 * amp exactly and dA = 0,
    // so n A and n.s (A 2^-3) + n.w dA give the constant band's bits (adding a zero product changes nothing).
    Table neutral;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]}], "biomes": [{"name": "only", )" +
                                    ClimateTest::note_graph(R"("from_m": 0.4, "to_m": 26214.4, "seed": 59943, "steepness": 0.125, "max_amp_m": 3276.8)", R"("shape": "plain")") +
                                    "}]}",
                            g.ground, neutral) == ov::str());
    const Cascade::Spec specs[] = { combined_spec(carried, bare), combined_spec(neutral, bare) };
    CHECK(specs[0].field_range().empty()); // the pick's derive only
    CHECK(specs[1].modulation != nullptr);
    int64_t compared = 0, differing = 0;
    for (const Cascade::Spec &spec : specs) {
        for (const int lod : { 0, 2, 6, 11 }) {
            for (const int64_t base : { int64_t(0), int64_t(-77), (int64_t(1) << 32) - 5, -(int64_t(1) << 45) + 3 }) {
                const Vector3i64 p(base, base / 7 + 3, -base + 11);
                float want[Cascade::CHANNELS], got[Cascade::CHANNELS];
                Cascade::point_into(bare, p, lod, want);
                Cascade::point_into(spec, p, lod, got);
                compared++;
                differing += differing_floats(want, got, 4);
            }
        }
    }
    CHECK(compared == 32);
    CHECK(differing == 0);
}

TEST_CASE("Climate: modulated cascade, box runs equal point queries in every slot, and the amplitude follows the biomes") {
    GroundFixture g(0.05, 15);
    ov::Value data;
    CLIMATE_REQUIRE(load_table("res://climate/plain.json", data) == ov::str());
    Table t;
    CLIMATE_REQUIRE(parse(data, g.ground, t) == ov::str());
    const Cascade::Spec spec = combined_spec(t, g.sdf.get_spec());
    CLIMATE_REQUIRE(spec.modulation != nullptr);
    CHECK(spec.fields.size() == 3);
    int64_t bad = 0;
    for (const int lod : { 0, 3, 8 }) {
        for (const int64_t base : { int64_t(5), (int64_t(1) << 32) - 2, -(int64_t(1) << 44) }) {
            bad += box_point_mismatches(spec, Vector3i64(base, -3, base / 3), 3, lod);
        }
    }
    CHECK(bad == 0);
    // Ground queries carry the fields only where FieldRange says; with all of them carried the ground is the same bits.
    const Cascade::FieldRange range = spec.field_range();
    CHECK(range.floor == 0); // 0.4 m at 5 cm: k = 3, inserted on Λ_0
    CHECK(range.ceiling == 16); // 26 km: k = 19
    int64_t ground_differs = 0;
    for (const int lod : { 0, 6, 7, 12, 16, 17 }) {
        for (int i = 0; i < 16; i++) {
            const Vector3i64 p(int64_t(i) * 7919 - 50000, 3 - i, int64_t(i) * -104729);
            float part[Cascade::SLOTS], all[Cascade::SLOTS];
            Cascade::point_slots(spec, p, lod, part);
            Cascade::point_slots(spec, p, lod, all, true);
            ground_differs += differing_floats(part, all, 4);
            CHECK((part[Cascade::field_slot(0)] == 0.0f) == !range.carried(lod, lod)); // the fields only where carried
        }
    }
    CHECK(ground_differs == 0);
    // The ground differs from the bare one where a biome multiplies its octaves: heights along a line at Λ_6.
    int differ = 0;
    for (int i = 0; i < 64; i++) {
        const Vector3i64 p(int64_t(i) * 997, 0, int64_t(i) * -1301);
        float a[Cascade::CHANNELS], b[Cascade::CHANNELS];
        Cascade::point_into(g.sdf.get_spec(), p, 6, a);
        Cascade::point_into(spec, p, 6, b);
        differ += float_bits(a[Cascade::VALUE]) != float_bits(b[Cascade::VALUE]);
        CHECK(std::isfinite(b[Cascade::VALUE]));
    }
    CHECK(differ > 48);
}

TEST_CASE("Climate: carried only where the ground needs it, refused where it can't apply") {
    GroundFixture g(0.05, 15);
    // A mix row is a note (a climate field isn't one: a graph reads a field past its notes, in its program).
    Table t;
    const ov::str fields = R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]},
            {"name": "h", "bands": [{"from_m": 400, "to_m": 6500, "amp": 1}]}], )";
    CHECK(ov::contains(parse_json(fields + R"("biomes": [{"name": "a", "at": {"t": [1, 1.2]}, "graph": {"nodes": [{"id": "h", "kind": "field", "field": "h"},
                       {"id": "m", "kind": "mix", "rows": [{"in": "h"}]}, {"id": "o", "kind": "output", "height": "m"}]}}, {"name": "b"}]})",
                               g.ground, t),
            "a mix row is a note"));
    // Fields with the template's notes (a pick only): carried nowhere in the ground's runs (no field scratch, no fields
    // variant), and its top lattice is the ground's.
    const ov::str ground = ClimateTest::ground_graph(0.05);
    CLIMATE_REQUIRE(parse_json(fields + R"("biomes": [{"name": "a", "at": {"t": [1, 1.2]}, )" + ground + R"(}, {"name": "b", )" + ground + "}]}", g.ground, t) ==
            ov::str());
    const Cascade::Spec spec = combined_spec(t, g.sdf.get_spec());
    CHECK(spec.modulation != nullptr); // the pick's derive, for the chunks' samples (CascadeGLSL::field_range)
    CHECK(spec.field_range().empty());
    CHECK(CascadeGLSL::vec4s_per_point(spec, spec.field_range()) == 1);
    CHECK(spec.top_level() == g.sdf.get_spec().top_level());
    // A flat ground (no band) with fields: no top lattice from them either.
    Cascade::Spec flat = spec;
    flat.bands.clear();
    CHECK(flat.top_level() == -1);
    CHECK(flat.top_level(true) >= 0);
    // A table that can't load.
    ov::Value empty;
    CHECK(ov::contains(load_table("res://climate/does_not_exist.json", empty), "cannot read"));
}

TEST_CASE("Climate: biome ground materials, the base palette, and the per-sample pick (box runs equal point queries)") {
    SDFCascade sdf;
    CLIMATE_REQUIRE(load_table("res://climate/plain.json", sdf.climate_data) == ov::str());
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
    const Table &t = *sdf.get_climate();
    const std::vector<uint32_t> base = sdf.get_gpu_table()->base_palette();
    CLIMATE_REQUIRE(base.size() == 1 + t.biomes.size());
    CHECK(base[0] == Material::GROUND);
    CHECK(CascadeGLSL::ground_materials(&sdf));
    for (size_t b = 0; b < t.biomes.size(); b++) {
        const Material::Params &m = sdf.get_materials()[base[1 + b]];
        CHECK(sdf.get_materials().name(base[1 + b]) == "biome:" + t.biomes[b].name);
        CHECK(m.pattern == Material::PATTERN_GROUND);
        CHECK(m.debug.a == 1.0f);
    }
    // "#d8b878" (desert, sRGB) as a linear albedo.
    const int desert = 2;
    CHECK(t.biomes[desert].name == "desert");
    CHECK(t.biomes[desert].surface.flat.r == doctest::Approx(ov::Color(216 / 255.0, 0, 0).srgb_to_linear().r));
    // A reconfigure keeps the materials' ids (Library::set) and the chunk passes derive the pick above every sample.
    sdf.configure(param, 15);
    CHECK(sdf.get_gpu_table()->base_palette() == base);
    const Cascade::FieldRange range = CascadeGLSL::field_range(&sdf, false);
    CHECK(range.everywhere());
    CHECK(CascadeGLSL::field_range(&sdf, true).sample == sdf.get_spec().field_range().sample); // point items: the spec's

    // The pick of a box run's level slots (what the GPU's level 0 reads) equals the point query, at several LODs and
    // origins; over a wide coarse slice the example climate shows several biomes.
    const Cascade::Spec &spec = sdf.get_spec();
    int64_t points = 0, mismatches = 0;
    for (const int lod : { 0, 3, 8, 13 }) {
        for (const int64_t origin : { int64_t(0), (int64_t(1) << 40) + 12345, -(int64_t(1) << 33) }) {
            Cascade::Box box;
            for (int a = 0; a < 3; a++) {
                box.lo[a] = (a == 1 ? (sdf.get_datum_y() >> lod) : (origin >> lod)) - 3;
                box.hi[a] = box.lo[a] + 5;
            }
            const Cascade::Level level = Cascade::run(spec, box, lod, nullptr, true);
            Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
                float slots[Cascade::SLOTS];
                level.slots_at(p_i, slots);
                const VoxelPos v(p_p.x << lod, p_p.y << lod, p_p.z << lod);
                points++;
                mismatches += sdf.ground_material(slots) != sdf.ground_material_at(v, lod);
            });
        }
    }
    CHECK(points > 0);
    CHECK(mismatches == 0);
    std::vector<int> seen(t.biomes.size() + 1, 0);
    const int lod = 12;
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 24; j++) {
            const VoxelPos v(int64_t(i - 12) << (lod + 3), sdf.get_datum_y() >> lod << lod, int64_t(j - 12) << (lod + 3));
            const uint32_t m = sdf.ground_material_at(v, lod);
            CLIMATE_REQUIRE((m >= 1 && m < base.size()));
            seen[m]++;
        }
    }
    CHECK(std::count_if(seen.begin(), seen.end(), [](int p_n) { return p_n > 0; }) >= 3);

    // Malformed surfaces.
    GroundFixture g(0.05, 15);
    Table bad;
    CHECK(ov::contains(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"flat": "green"}}]})", g.ground, bad), "#rrggbb"));
    CHECK(ov::contains(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"roughness": 2}}]})", g.ground, bad), "roughness"));
    CHECK(ov::contains(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"shiny": 1}}]})", g.ground, bad), "unknown key"));
    // A surface's steep colour: the given flat without its own, the procedural ground's rock without either.
    Table s;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"roughness": 0.5}}]})", g.ground, s) == ov::str());
    CHECK(s.biomes[0].surface.steep == Material::GROUND_STEEP);
    CHECK(s.biomes[0].surface.flat == Material::GROUND_FLAT);
    CLIMATE_REQUIRE(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"flat": "#ff0000"}}]})", g.ground, s) == ov::str());
    CHECK(s.biomes[0].surface.steep == s.biomes[0].surface.flat);
    // A climate without fields (one default biome): no pick to make, so no ground materials (CPU and GPU alike).
    SDFCascade fieldless;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [], "biomes": [{"name": "a"}]})", g.ground, s) == ov::str());
    fieldless.climate_data = ClimateTest::json(R"({"fields": [], "biomes": [{"name": "a"}]})");
    fieldless.configure(param, 15);
    CLIMATE_REQUIRE(fieldless.get_climate() != nullptr);
    CHECK_FALSE(fieldless.has_ground_materials());
    CHECK(fieldless.get_gpu_table()->base_palette() == std::vector<uint32_t>{ Material::GROUND });
    CHECK(fieldless.get_ground_material_glsl(fieldless.get_spec()).empty());
    // Without a climate: the base palette is GROUND alone, and no ground material GLSL.
    SDFCascade plain;
    plain.configure(param, 15);
    CHECK(plain.get_gpu_table()->base_palette() == std::vector<uint32_t>{ Material::GROUND });
    CHECK(plain.get_ground_material_glsl(plain.get_spec()).empty());
    CHECK(ov::contains(Climate::pick_glsl(t, spec), "uint climate_biome(Ch ch)"));
}

// Several ground notes on one lattice (the compiler's notes of one octave: Cascade::MAX_LEVEL_NOTES) are each inserted,
// the level's first then the others in spec order: box runs equal point queries bit for bit, and the ground is the two
// notes' sum (within the rounding of the adds). A field holds one band per lattice: validate refuses a second, and the
// cascade ignores it (ERR) in a spec built by hand.
TEST_CASE("Cascade: several ground notes on one lattice, one band per field") {
    GroundFixture g(0.05, 15);
    Table t;
    CHECK(ov::contains(parse_json(R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 400, "amp": 1}, {"wavelength_m": 400, "amp": 2}]}], "biomes": [{"name": "a"}]})",
                               g.ground, t),
            "twice"));
    Cascade::Spec one, other, two;
    one.bands = { ExactNoise::make_octave(11u, 9u, 9, 40.0f) };
    other.bands = { ExactNoise::make_octave(12u, 9u, 9, 25.0f) };
    two.bands = { one.bands[0], other.bands[0] }; // the same lattice Λ_6
    CHECK(two.level_notes(6) == 2);
    CHECK(two.max_level_notes() == 2);
    // The notes keep their BandMod slots in every view; the field groups follow them.
    Cascade::Spec whole = two;
    whole.fields = { { ExactNoise::make_octave(13u, 9u, 9, 1.0f) } };
    for (const Cascade::Spec &view : { whole, whole.without_fields(), whole.ground_only() }) {
        CHECK(view.max_level_notes() == 2);
        CHECK(view.note_mod(1) == 1);
        CHECK(view.group_mod(0) == 0);
    }
    CHECK(whole.group_mod(1) == 2);
    CHECK(whole.mod_count() == 3);
    int64_t mismatches = 0;
    double worst = 0.0;
    for (int i = 0; i < 32; i++) {
        const Vector3i64 p(i * 37 - 500, i * 11, 900 - i * 23);
        float a[Cascade::CHANNELS], b[Cascade::CHANNELS], ab[Cascade::CHANNELS], c[Cascade::CHANNELS];
        Cascade::point_into(one, p, 2, a);
        Cascade::point_into(other, p, 2, b);
        Cascade::point_into(two, p, 2, ab);
        Cascade::point(two, p, 2, c);
        mismatches += differing_floats(ab, c, 4);
        for (int ch = 0; ch < Cascade::CHANNELS; ch++) {
            worst = std::max(worst, std::fabs(double(ab[ch]) - double(a[ch]) - double(b[ch])) / (1.0 + std::fabs(double(ab[ch]))));
        }
    }
    CHECK(mismatches == 0);
    CHECK(worst < 1e-5);
    for (const int64_t base : { int64_t(-40), (int64_t(1) << 32) - 3 }) {
        CHECK(box_point_mismatches(two, Vector3i64(base, 7, -base), 4, 1) == 0);
    }
    // A field's second band on a lattice: ignored, its group the first's alone.
    Cascade::Spec field = one, fields = one;
    field.fields = { one.bands };
    fields.fields = { two.bands };
    int64_t field_mismatches = 0;
    for (int i = 0; i < 16; i++) {
        float a[Cascade::SLOTS], b[Cascade::SLOTS];
        Cascade::point_slots(field, Vector3i64(i * 13, -i, 5), 2, a, true);
        Cascade::point_slots(fields, Vector3i64(i * 13, -i, 5), 2, b, true);
        field_mismatches += differing_floats(a + Cascade::field_slot(0), b + Cascade::field_slot(0), 4);
    }
    CHECK(field_mismatches == 0);
}

// A biome multiplying the ground with no field: the modulated variant on the GPU too (field_range keyed on the
// modulation, not on the fields), or the GPU draws the bare ground.
TEST_CASE("Climate: a modulation without fields takes the modulated GPU variant") {
    SDFCascade sdf;
    CHECK(load_table("res://climate/fieldless_test.json", sdf.climate_data) == ov::str());
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
    CHECK(sdf.get_spec().fields.empty());
    CHECK(sdf.get_spec().modulation != nullptr);
    for (const bool points : { false, true }) {
        const Cascade::FieldRange range = CascadeGLSL::field_range(&sdf, points);
        CHECK_FALSE(range.empty());
        CHECK(ov::contains(CascadeGLSL::level_shader(&sdf, false, points, false, !points, CascadeGLSL::View::FIELDS), "cascade_mods("));
    }
}

// A run's level above its last derives its samples' climate and H carries it down (Climate's Modulator): the last level
// carries no field, only the ground and the carry, and its pick is the one its floor parent derived (copied, not
// interpolated).
TEST_CASE("Climate: a run's samples read the climate derived on the level above") {
    ClimateTest::ClimateSDF c("res://climate/example.json", 15);
    CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
    const Cascade::Spec &spec = c.sdf.get_spec();
    const SampleCarry sc = sample_carry(*c.sdf.get_climate());
    CLIMATE_REQUIRE(sc.pick >= 0);
    CHECK(spec.carry_groups() > sc.count); // the bands' parts after the samples' terms
    CHECK(spec.field_range().sample); // the example's curves
    int64_t points = 0, bad = 0;
    for (const int lod : { 0, 4, 9 }) {
        Cascade::Box box;
        for (int a = 0; a < 3; a++) {
            box.lo[a] = a == 1 ? (c.sdf.get_datum_y() >> lod) - 3 : int64_t(-5 - 1000 * lod) >> lod;
            box.hi[a] = box.lo[a] + 6;
        }
        std::vector<Cascade::Level> levels;
        Cascade::run(spec, box, lod, &levels);
        CLIMATE_REQUIRE(levels.size() >= 2);
        const Cascade::Level &last = levels.back();
        const Cascade::Level &above = levels[levels.size() - 2];
        CHECK(last.fields == 0);
        CHECK(last.carry == spec.modulation->carry_used(lod)); // the ground's groups only
        CHECK(last.carry < spec.carry_groups());
        CHECK(above.fields == int(spec.fields.size()));
        Cascade::for_each_point(last.box, [&](const Vector3i64 &p_p, int64_t p_i) {
            const Vector3i64 q(p_p.x >> 1, p_p.y >> 1, p_p.z >> 1);
            const float pick = last.ch[Cascade::carry_slot(sc.pick)][size_t(p_i)];
            points++;
            bad += float_bits(pick) != float_bits(above.ch[Cascade::carry_slot(sc.pick)][size_t(above.box.index(q))]);
        });
    }
    CHECK(points == 3 * 7 * 7 * 7);
    CHECK(bad == 0);
}
