#include "sdf_cascade.hpp"
#include "test_climate_common.hpp"
#include "world_file.hpp"

namespace {

ov::str parse(const char *p_text, WorldFile &r_world) {
    return WorldFile::parse(ClimateTest::json(p_text), ov::str(), r_world);
}

} // namespace

TEST_CASE("WorldFile: every key is optional and defaults to the engine's defaults") {
    WorldFile w;
    CHECK(parse(R"({"comment": "nothing else"})", w) == ov::str());
    CHECK(w.voxel_m == 0.5);
    CHECK(w.lods == 15);
    CHECK(w.climate_table.is_empty());
    CHECK(w.atmosphere);
    CHECK(w.clouds);
    CHECK(w.zone.side_m == 0.0);
    CHECK(w.zone.x_m == SDFTable::Zone().x_m);
    CHECK(w.instances.empty());
}

TEST_CASE("WorldFile: inline sections reach their fields") {
    WorldFile w;
    CHECK(parse(R"({"engine": {"voxel_m": 0.05, "lods": 25}, "planet": {"radius_m": 2000, "ring_m": 9000},
        "look": {"atmosphere": false, "atmosphere_scale": 2, "clouds": false}, "zone": {"side_m": 8, "centre_m": [3, -4]},
        "instances": [{"species": "conifer", "cell_m": 5}, {"species": "boulder", "cell_m": 40, "share": 3}], "features": [{"kind": "lava_lake", "cell_m": 640, "chance": 0.6, "radius_m": [18, 45]}]})",
                  w) == ov::str());
    CHECK(w.voxel_m == 0.05);
    CHECK(w.lods == 25);
    CHECK(w.planet_radius_m == 2000.0);
    CHECK(w.planet_ring_m == 9000.0);
    CHECK(!w.atmosphere);
    CHECK(w.atmosphere_scale == 2.0);
    CHECK(!w.clouds);
    CHECK(w.zone.side_m == 8.0);
    CHECK(w.zone.x_m == 3.0);
    CHECK(w.zone.z_m == -4.0);
    CLIMATE_REQUIRE(w.instances.size() == 2);
    CHECK(w.instances[0].species == "conifer");
    CHECK(w.instances[0].cell_m == 5.0);
    CHECK(w.instances[0].share == 1u);
    CHECK(w.instances[1].share == 3u);
    CLIMATE_REQUIRE(w.features.size() == 1);
    CHECK(w.features[0].kind == "lava_lake");
    CHECK(w.features[0].chance == 0.6);
    CHECK(w.features[0].radius_hi_m == 45.0);
}

TEST_CASE("WorldFile: the climate's fields merge with the world's biomes into one table") {
    WorldFile w;
    CHECK(parse(R"({"climate": {"fields": [{"name": "t"}]}, "biomes": [{"name": "a"}]})", w) == ov::str());
    CHECK(w.climate_table["fields"].size() == 1);
    CHECK(w.climate_table["biomes"].size() == 1);
    CHECK(ov::contains(parse(R"({"ground_stretch": [{"wavelength_m": 8}]})", w), "unknown key 'ground_stretch'"));
}

TEST_CASE("WorldFile: a refused file leaves the world untouched and says why") {
    WorldFile w;
    w.lods = 3;
    CHECK(ov::contains(parse(R"({"trees_count": 1})", w), "unknown key 'trees_count'"));
    CHECK(ov::contains(parse(R"({"engine": {"voxels": 1}})", w), "unknown key 'voxels'"));
    CHECK(ov::contains(parse(R"({"zone": {"cuont": 1}})", w), "unknown key 'cuont'"));
    CHECK(ov::contains(parse(R"({"zone": {"centre_m": [1]}})", w), "centre_m is [x, z]"));
    CHECK(ov::contains(parse(R"({"zone": {"side_m": -1}})", w), "side_m must be >= 0"));
    CHECK(ov::contains(parse(R"({"instances": [{"species": "conifer"}]})", w), "'cell_m' is required"));
    CHECK(ov::contains(parse(R"({"instances": {"species": "conifer"}})", w), "a list of instances"));
    CHECK(ov::contains(parse(R"({"features": [{"kind": "lava_lake", "cell_m": 640, "radius_m": [5, 2]}]})", w), "0 < lo <= hi"));
    CHECK(ov::contains(parse(R"({"features": [{"kind": "lava_lake", "cell_m": 640, "chance": 2, "radius_m": [1, 2]}]})", w), "in [0, 1]"));
    CHECK(ov::contains(parse(R"({"features": {"radius_km": 2}})", w), "a list of features"));
    CHECK(ov::contains(parse(R"({"planet": 5})", w), "not an object"));
    CHECK(ov::contains(parse(R"({"engine": 5})", w), "not a path or an object"));
    CHECK(ov::contains(parse(R"({"engine": {"voxel_m": 0}})", w), "positive"));
    CHECK(ov::contains(parse(R"({"biomes": [{"name": "a"}]})", w), "need a climate"));
    CHECK(ov::contains(parse(R"({"engine": {"lods": 32}})", w), "lods must be"));
    CHECK(ov::contains(parse(R"({"engine": {"lods": 0}})", w), "lods must be"));
    CHECK(ov::contains(parse(R"({"planet": {"radius_m": "6371000"}})", w), "not a number"));
    CHECK(ov::contains(WorldFile::load("res://worlds/does_not_exist.json", w), "cannot read"));
    CHECK(w.lods == 3);
}

TEST_CASE("WorldFile: world.json finds its engine and climate next to it") {
    WorldFile w;
    CHECK(WorldFile::load("res://climate/world.json", w) == ov::str());
    CHECK(w.voxel_m == 0.05);
    CHECK(w.lods == 25);
    CHECK(w.zone.side_m == 5000.0);
    CHECK(w.instances.empty()); // every biome lists its own
    const ov::Value &biomes = w.climate_table["biomes"];
    CHECK(biomes[size_t(0)].has("instances"));
    CHECK(w.climate_table["fields"].size() == 6);
    CHECK(w.climate_table["biomes"].size() == 8);
    SDFCascade sdf;
    ChunkParam param;
    param.voxel_size = float(w.voxel_m);
    sdf.climate_data = w.climate_table;
    sdf.configure(param, w.lods);
    CHECK(sdf.get_climate() != nullptr);
}

// Every world a preset, a scene or a tool names (their tables parse against their own engine).
TEST_CASE("WorldFile: an empty object is the default world") {
    WorldFile w;
    CHECK(parse("{}", w) == ov::str());
    CHECK(w.lods == 15);
}

TEST_CASE("WorldFile: the project's worlds load") {
    for (const char *path : { "res://worlds/default.world.json", "res://worlds/cm.world.json", "res://worlds/lods25.world.json", "res://worlds/biomes.world.json",
                 "res://worlds/forest5cm.world.json", "res://worlds/forest5cm_x10.world.json", "res://worlds/forest5cm_1m.world.json", "res://worlds/forest5cm_41m.world.json",
                 "res://worlds/earth.world.json", "res://worlds/earth25.world.json", "res://worlds/tiny.world.json", "res://climate/example.json", "res://climate/plain.json",
                 "res://climate/shaped_test.json", "res://climate/fieldless_test.json" }) {
        CAPTURE(path);
        WorldFile w;
        CHECK(WorldFile::load(path, w) == ov::str());
    }
}
