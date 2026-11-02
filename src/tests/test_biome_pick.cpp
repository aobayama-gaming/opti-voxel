#include "test_climate_common.hpp"

#include "biome_pick.hpp"
#include "sdf_cascade.hpp"
#include "species.hpp"

#include <chrono>

using namespace Archetype;

namespace {

// Two biomes over the template ground on a short field (25 m .. 100 m: they alternate within a small forest): "conifers"
// where t > 0 growing p_conifers' instances, the default growing p_rest's (a JSON list, or empty for the key absent).
ov::Value two_biomes(const ov::str &p_conifers, const ov::str &p_rest, const ov::str &p_features = ov::str()) {
    const ov::str ground = ClimateTest::ground_graph(0.05);
    const ov::str rest = p_rest.empty() ? ov::str() : ", \"instances\": " + p_rest;
    return ClimateTest::json(R"({"fields": [{"name": "t", "bands": [{"from_m": 25, "to_m": 100, "amp": 1}]}], "biomes": [{"name": "conifers", "at": {"t": [0.5, 0.5]}, )" + ground +
            ", \"instances\": " + p_conifers + p_features + R"(}, {"name": "rest", )" + ground + rest + "}]}");
}

} // namespace

TEST_CASE("BiomePick: integer shares pick by the variant, n equal shares are the variant modulo n") {
    for (uint32_t v = 0; v < 1000; v++) {
        CHECK(pick_share({ 1, 1, 1 }, v) == int(v % 3u));
        CHECK(pick_share({ 1, 1 }, v) == int(v & 1u));
    }
    CHECK(pick_share({ 0, 0 }, 7) == -1);
    CHECK(pick_share({}, 7) == -1);
    int counts[2] = {};
    for (uint32_t v = 0; v < 4000; v++) {
        counts[pick_share({ 1, 3 }, v)]++;
    }
    CHECK(counts[0] == 1000);
    CHECK(counts[1] == 3000);
    CHECK(pick_share({ 0, 2 }, 5) == 1);
}

TEST_CASE("BiomePick: one biome or none is 0; a site's biome is a pure function of it, each biome about its weight's share") {
    const BiomePick none;
    CHECK(none.at(VoxelPos(123, 4, 5), 99u) == 0);
    ClimateTest::ClimateSDF c(two_biomes("[]", ""), 15);
    CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
    const BiomePick pick(c.sdf.get_climate(), &c.sdf.get_spec());
    CHECK(pick.biomes() == 2);
    CHECK(pick.level() == 9 - 3); // t's finest octave (25 m at 5 cm: k = 9)
    // A voxel where the two blend, then many seeds there: each biome's share within 2% of its weight.
    VoxelPos at;
    float w[Climate::MAX_BIOMES] = {};
    bool found = false;
    for (int i = 0; i < 4000 && !found; i++) {
        at = VoxelPos(int64_t(i) * 37, c.sdf.get_datum_y(), int64_t(i) * -11);
        pick.weights(at, w);
        found = w[0] > 0.3f && w[0] < 0.7f;
    }
    CLIMATE_REQUIRE(found);
    int counts[2] = {};
    for (uint32_t seed = 0; seed < 20000; seed++) {
        const int b = pick.at(at, ExactNoise::hash3(seed, 1u, 2u).x);
        CLIMATE_REQUIRE((b == 0 || b == 1));
        counts[b]++;
        CHECK(pick.at(at, ExactNoise::hash3(seed, 1u, 2u).x) == b);
    }
    MESSAGE("weights ", w[0], " / ", w[1], ", picked ", counts[0] / 20000.0, " / ", counts[1] / 20000.0);
    CHECK(std::fabs(double(counts[0]) / 20000.0 - double(w[0])) < 0.02);
    // The cost of a pick (the climate read once per site), for the load's budget.
    const auto t0 = std::chrono::steady_clock::now();
    int sum = 0;
    for (int i = 0; i < 2000; i++) {
        sum += pick.at(VoxelPos(int64_t(i) * 101, c.sdf.get_datum_y(), int64_t(i) * 7), uint32_t(i));
    }
    const double us = double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()) / 2000.0;
    MESSAGE("a site's biome: ", us, " us (", sum, " of 2000 sites in the default)");
    CHECK(us < 200.0);
}

TEST_CASE("Species: today's by default, a world's list parsed, refused with the reason") {
    const std::vector<Species> d = default_species();
    CLIMATE_REQUIRE(d.size() == 5);
    CHECK(d[0].name == "broadleaf");
    CHECK(d[2].name == "conifer");
    CHECK(d[2].tree.cone_tiers == tree_species()[2].cone_tiers);
    CHECK(d[3].kind == Species::ROCK);
    CHECK_FALSE(d[3].rock.pillar);
    CHECK(d[4].rock.pillar);
    std::vector<Species> s;
    CHECK(parse_species(ClimateTest::json(R"([{"name": "spruce", "kind": "tree", "height": 25, "cone_tiers": 5}, {"name": "pebble", "kind": "rock", "radius_m": [0.2, 0.5]}])"), s) ==
            ov::str());
    CLIMATE_REQUIRE(s.size() == 2);
    CHECK(s[0].tree.height == 25.0);
    CHECK(s[0].tree.cone_tiers == 5);
    CHECK(s[0].tree.crown_radius == TreeSpecies().crown_radius);
    CHECK(s[1].rock.radius_hi_m == 0.5);
    CHECK(species_key(species_of(s, Species::TREE)) != species_key(species_of(d, Species::TREE)));
    const std::pair<const char *, const char *> bad[] = {
        { R"([{"name": "a", "kind": "shrub"}])", "tree, rock or pillar" },
        { R"([{"name": "a", "kind": "tree", "leaves": 3}])", "unknown key 'leaves'" },
        { R"([{"name": "a", "kind": "tree", "limbs": 2.5}])", "whole number" },
        { R"([{"name": "a", "kind": "tree", "height": 0}])", "> 0" },
        { R"([{"name": "a", "kind": "rock", "radius_m": [3, 1]}])", "lo <= hi" },
        { R"([{"name": "a", "kind": "rock"}, {"name": "a", "kind": "pillar"}])", "named twice" },
        { R"({"name": "a"})", "a list" },
    };
    for (const auto &[json, error] : bad) {
        CAPTURE(json);
        CHECK(ov::contains(parse_species(ClimateTest::json(json), s), error));
    }
}

// The forest grows each site's species from its biome: the conifers' biome grows conifers only, the default nothing; a
// biome that lists no instances grows every species, the forest as before the biomes.
TEST_CASE("BiomePick: a forest grows each site's species from its biome") {
    auto forest = [](const ov::Value &p_climate) {
        auto sdf = std::make_unique<SDFCascade>();
        ChunkParam param;
        param.voxel_size = 0.05f;
        sdf->climate_data = p_climate;
        sdf->set_forest(160, 60.0);
        sdf->set_stream_sources(false);
        sdf->set_moment_blobs(false);
        param.skirt_depth = MAX(param.skirt_depth, sdf->min_skirt_depth(param, 16));
        sdf->configure(param, 16);
        return sdf;
    };
    const auto chosen = forest(two_biomes(R"([{"species": "conifer"}])", "[]"));
    CLIMATE_REQUIRE(chosen->get_climate() != nullptr);
    const std::vector<TreeSpan> &trees = chosen->get_trees();
    CHECK(trees.size() > 10);
    for (const TreeSpan &t : trees) {
        CHECK(t.species == 2);
    }
    // Neither lists: every species, as without a climate (variant % 3).
    const auto all = forest(two_biomes(R"([{"species": "broadleaf"}, {"species": "slender"}, {"species": "conifer"}])", ""));
    const auto bare = forest(ov::Value::object());
    int kinds[3] = {};
    for (const TreeSpan &t : all->get_trees()) {
        kinds[t.species]++;
    }
    MESSAGE("trees: conifers' biome ", trees.size(), ", both biomes ", all->get_trees().size(), ", no climate ", bare->get_trees().size());
    CHECK(kinds[0] > 0);
    CHECK(kinds[1] > 0);
    CHECK(kinds[2] > 0);
    CHECK(trees.size() < all->get_trees().size());
}
