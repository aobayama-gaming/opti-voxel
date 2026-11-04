#include "test_climate_common.hpp"

#include "cascade_glsl.hpp"
#include "climate.hpp"
#include "graph_compile.hpp"
#include "level_program.hpp"
#include "sdf_cascade.hpp"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

using namespace Climate;
using ClimateTest::ClimateSDF;
using ExactNoise::float_bits;

// The shapes' centring constants are the noise's own means: E[sabs(n)] and E[n^2] of perlin3, over random points of
// random cells.
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

TEST_CASE("Climate shape: curves pass through their points") {
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
}

TEST_CASE("Climate shape: malformed shapes and terms are refused with the reason") {
    SDFCascade sdf;
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    const Ground g = sdf.climate_ground();
    auto err = [&](const ov::str &p_json) {
        Table t;
        Terrain::LevelProgram prog;
        return Terrain::compile(ClimateTest::json(p_json), g, Terrain::Mode::SPECIALISED, t, prog);
    };
    // One biome whose graph is note "n" (51.2 m) through p_nodes (each reading the one before), its last p_height.
    auto biome = [](const ov::str &p_nodes, const char *p_height, const ov::str &p_note = ov::str()) {
        return R"("biomes": [{"name": "a", "graph": {"nodes": [{"id": "n", "kind": "note", "wavelength_m": 51.2, "seed": 1, "amp_m": 6.4)" + p_note + "}, " +
                p_nodes + R"({"id": "o", "kind": "output", "height": ")" + ov::str(p_height) + "\"}]}}]}";
    };
    const ov::str f1 = R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]}], )";
    CHECK(ov::contains(err(f1 + biome(R"({"id": "s", "kind": "shape", "in": "n", "shape": "spiky"}, )", "s")), "plain, billow"));
    // A multifractal weighs any node by a ramp of any other (a field's here), low_m and high_m apart.
    CHECK(err(f1 + biome(R"({"id": "f", "kind": "field", "field": "t"}, {"id": "m", "kind": "multifractal", "in": "f", "parent": "n", "low_m": 0, "high_m": 1}, )", "m")) == ov::str());
    auto mf = [&](const char *p_parent, double p_low, double p_high, const ov::str &p_note = ov::str()) {
        return ov::format(R"({"id": "c", "kind": "note", "wavelength_m": 409.6, "seed": 3, "amp_m": 0.5%s}, {"id": "w", "kind": "warp", "in": "c", "amount_m": 1}, )"
                          R"({"id": "m", "kind": "multifractal", "in": "n", "parent": "%s", "low_m": %s, "high_m": %s}, )",
                p_note, p_parent, ov::num(p_low, 3), ov::num(p_high, 3));
    };
    CHECK(err(f1 + biome(mf("c", -0.25, 0.25), "m")) == ov::str());
    CHECK(err(f1 + biome(mf("c", 0.25, -0.25), "m")) == ov::str());
    CHECK(err(f1 + biome(mf("c", 0, 1), "m")) == ov::str()); // add 0: weight 0 at the parent's zero height
    CHECK(err(f1 + biome(mf("n", -1, 1), "m")) == ov::str());
    CHECK(ov::contains(err(f1 + biome(mf("c", 1, 1), "m")), "differ"));
    CHECK(err(f1 + biome(mf("c", -1, 1, R"(, "stretch_y": 2)"), "m")) == ov::str());
    CHECK(ov::contains(err(f1 + biome(mf("c", -1, 1, R"(, "warp": "w")"), "m")), "the graph loops")); // c warped by its own warp
    // A warp is a note's input, strictly coarser than the note (the gate); its input any node, a shaped one too.
    const ov::str coarse = R"({"id": "c", "kind": "note", "wavelength_m": 409.6, "seed": 3, "amp_m": 0.5}, )";
    CHECK(ov::contains(err(f1 + biome(coarse + R"({"id": "w", "kind": "warp", "in": "c", "amount_m": 1}, )", "w")), "a note's input"));
    CHECK(err(f1 + biome(coarse + R"({"id": "w", "kind": "warp", "in": "c", "amount_m": 1}, )", "n", R"(, "warp": "w")")) == ov::str());
    CHECK(ov::contains(err(f1 + biome(R"({"id": "c", "kind": "note", "wavelength_m": 51.2, "seed": 3, "amp_m": 0.5}, {"id": "w", "kind": "warp", "in": "c", "amount_m": 1}, )", "n", R"(, "warp": "w")")),
            "coarser"));
    CHECK(err(f1 + biome(coarse + R"({"id": "r", "kind": "shape", "in": "c", "shape": "ridged"}, {"id": "w", "kind": "warp", "in": "r", "amount_m": 1}, )", "n", R"(, "warp": "w")")) ==
            ov::str());
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

// The skirt bound (LevelProgram::step) on tables with curves and shapes: at the samples a LOD L and a LOD L+1 chunk share
// (Λ_{L+1}) near the LOD-L surface, the step is within the outer chunk's skirt depth, the cap lifted (the bound itself;
// the cap is a cost choice).
TEST_CASE("Climate shape: the step between LOD L and L+1 stays within the skirt depth") {
    for (const auto &[path, lods] : { std::pair{ "res://climate/example.json", 15 }, std::pair{ "res://climate/eight_biomes_test.json", 15 }, std::pair{ "res://climate/programs_test.json", 15 },
                 std::pair{ "res://climate/shaped_test.json", 15 } }) {
        CAPTURE(std::string(path));
        ClimateSDF c(path, lods, 1e6f);
        CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
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

// The level shader of each test table in both modes: every template placeholder and op-table token filled, the params
// within their block, and one generic text for every table (a mixer edit only rewrites the records: no recompile).
// Compiled offline with glslang (libshaderc) when stage D landed; the GPU checks run it (CascadeCheck --generic=1).
TEST_CASE("Climate shape: the level shader's text is filled, the generic one the same for every table") {
    ov::str generic;
    for (const char *path : { "res://climate/example.json", "res://climate/eight_biomes_test.json", "res://climate/programs_test.json", "res://climate/world.json",
                 "res://climate/caves_test.json", "res://climate/graph_test.json", "res://climate/notes_test.json" }) {
        CAPTURE(std::string(path));
        for (const bool specialize : { true, false }) {
            ov::Value data;
            CHECK(load_table(path, data) == ov::str());
            ClimateSDF c(data, 15, specialize);
            CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
            for (const bool fold : { false, true }) {
                Cascade::ShaderParams params;
                const ov::str code = CascadeGLSL::level_shader(&c.sdf, true, false, fold, true, true, &params);
                CHECK(!code.empty());
                CHECK(!ov::contains(code, "/*"));
                CHECK(!ov::contains(code, "@"));
                CHECK(!ov::contains(code, "$"));
                CHECK(params.values.size() * sizeof(float) <= CascadeGLSL::MAX_PARAM_BYTES);
                if (!specialize && !fold) {
                    CHECK((generic.empty() || code == generic));
                    generic = code;
                }
            }
        }
    }
}
