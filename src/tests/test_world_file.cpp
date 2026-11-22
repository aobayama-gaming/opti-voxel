#include "sdf_cascade.hpp"
#include "test_climate_common.hpp"
#include "world_file.hpp"

#include <cmath>
#include <cstring>

using namespace Archetype;

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
    CHECK(w.species.empty()); // a world names its species (worlds/species.json: today's)
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
    // Biomes without a climate: no fields (before, refused: "biomes need a climate").
    CHECK(parse(R"({"biomes": [{"name": "a"}]})", w) == ov::str());
    CHECK(w.climate_table["fields"].size() == 0);
    CHECK(w.climate_table["biomes"].size() == 1);
}

// "biomes" and "species" may name a file holding the list (the bare worlds' ground.biomes.json, the forests' species.json).
TEST_CASE("WorldFile: biomes and species from files") {
    WorldFile w;
    CHECK(WorldFile::load("res://worlds/forest5cm.world.json", w) == ov::str());
    ov::Value ground;
    CHECK(WorldFile::ground_table(ground) == ov::str());
    CHECK(ov::to_json(w.climate_table) == ov::to_json(ground));
    CLIMATE_REQUIRE(w.species.size() == 5);
    CHECK(w.species[0].name == "broadleaf");
    CHECK(w.species[2].name == "conifer");
    CHECK(w.species[4].name == "pillar");
    CHECK(w.species[4].rock.pillar);
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
    CHECK(ov::contains(parse(R"({"biomes": 5})", w), "not a path or a list"));
    CHECK(ov::contains(parse(R"({"species": {"name": "a"}})", w), "not a path or a list"));
    CHECK(ov::contains(parse(R"({"biomes": "res://worlds/default.world.json"})", w), "not a JSON list"));
    CHECK(ov::contains(parse(R"({"species": "res://worlds/nowhere.json"})", w), "cannot read"));
    CHECK(ov::contains(parse(R"({"engine": {"lods": 32}})", w), "lods must be"));
    CHECK(ov::contains(parse(R"({"engine": {"lods": 0}})", w), "lods must be"));
    CHECK(ov::contains(parse(R"({"planet": {"radius_m": "6371000"}})", w), "not a number"));
    CHECK(ov::contains(parse(R"({"planet": {"ring_m": 9000}})", w), "ring_m needs radius_m")); // was a flat world
    CHECK(ov::contains(parse(R"({"planet": {"radius_m": -2000}})", w), "must be finite and >= 0"));
    CHECK(ov::contains(parse(R"({"planet": {"radius_m": 1e999}})", w), "must be finite and >= 0"));
    CHECK(ov::contains(parse(R"({"planet": {"radius_m": 2000, "ring_m": 2000}})", w), "ring must be at least")); // was a flat world
    CHECK(ov::contains(parse(R"({"planet": {"radius_m": 1e20}})", w), "the radius must be in"));
    CHECK(ov::contains(WorldFile::load("res://worlds/does_not_exist.json", w), "cannot read"));
    CHECK(w.lods == 3);
}

TEST_CASE("WorldFile: world.json finds its engine and climate next to it") {
    WorldFile w;
    CHECK(WorldFile::load("res://climate/world.json", w) == ov::str());
    CHECK(w.voxel_m == 0.05);
    CHECK(w.lods == 15);
    CHECK(w.zone.side_m == 5000.0);
    CHECK(w.instances.empty()); // every biome lists its own
    const ov::Value &biomes = w.climate_table["biomes"];
    CHECK(biomes[size_t(0)].has("instances"));
    CHECK(w.climate_table["fields"].size() == 3); // temperature, humidity, wind: the warps' notes are their own
    CHECK(w.climate_table["biomes"].size() == 2); // hills, warped
    SDFCascade sdf;
    ChunkParam param;
    CHECK(sdf.configure_world(w, param) == ov::str()); // its species: worlds/species.json
    CHECK(sdf.get_climate() != nullptr);
}

// Every world a preset, a scene or a tool names (their tables parse against their own engine).
TEST_CASE("WorldFile: an empty object is the default world") {
    WorldFile w;
    CHECK(parse("{}", w) == ov::str());
    CHECK(w.lods == 15);
}

// World::start refuses a world whose planet or climate the cascade refuses (SDFCascade::get_world_error): every shipped
// world, and every CascadeCheck set's, loads and configures as itself.
TEST_CASE("WorldFile: the project's worlds load and configure as themselves, no planet or climate refused") {
    for (const char *path : { "res://worlds/default.world.json", "res://worlds/cm.world.json", "res://worlds/lods25.world.json", "res://worlds/biomes.world.json",
                 "res://worlds/forest5cm.world.json", "res://worlds/forest5cm_x10.world.json", "res://worlds/forest5cm_1m.world.json", "res://worlds/forest5cm_41m.world.json",
                 "res://worlds/earth.world.json", "res://worlds/earth25.world.json", "res://worlds/tiny.world.json", "res://worlds/tiny_biomes.world.json",
                 "res://worlds/shapes.world.json",
                 "res://climate/world.json", "res://climate/example.json",
                 "res://climate/plain.json", "res://climate/shaped_test.json", "res://climate/fieldless_test.json", "res://climate/eight_biomes_test.json",
                 "res://climate/notes_test.json", "res://climate/programs_test.json", "res://climate/graph_test.json", "res://climate/caves_test.json" }) {
        CAPTURE(path);
        WorldFile w;
        CHECK(WorldFile::load(path, w) == ov::str());
        SDFCascade sdf;
        ChunkParam param;
        CHECK(sdf.configure_world(w, param) == ov::str());
    }
}

namespace {

// Three edits that test the format's corners: int64 anchors past 2^53 (JSON numbers would round them), a turned
// rotation, a trailing -0 param (kept: not +0), floats that need 9 digits, a smooth radius.
std::vector<SDFTable::Edit> corner_edits() {
    std::vector<SDFTable::Edit> edits(3);
    edits[0].type = "sphere";
    edits[0].op = Combine::SUBTRACT;
    edits[0].anchor = VoxelPos((int64_t(1) << 53) + 1, -(int64_t(1) << 59) - 3, 12345);
    edits[0].params[0] = 0.1f;
    edits[1].type = "capsule";
    edits[1].op = Combine::SMOOTH_SUBTRACT;
    edits[1].anchor = VoxelPos(-7, 600, (int64_t(1) << 60) - 1);
    const float c = float(std::cos(0.7)), s = float(std::sin(0.7));
    const float turned[9] = { c, 0.0f, -s, 0.0f, 1.0f, 0.0f, s, 0.0f, c };
    std::copy(turned, turned + 9, edits[1].rotation);
    edits[1].params[0] = 12.000001f;
    edits[1].params[1] = 3.3333333f;
    edits[1].params[2] = -0.0f;
    edits[1].k = 0.6f;
    edits[2].type = "crown";
    edits[2].op = Combine::SMOOTH_UNION;
    edits[2].params[0] = 9.0f;
    edits[2].params[1] = 1.0f;
    edits[2].k = 2.0f;
    return edits;
}

// p_text read back as an edits file.
ov::str read_edits(const ov::str &p_text, std::vector<SDFTable::Edit> &r_edits) {
    ov::Value data;
    const ov::str err = WorldFile::parse_object(p_text, "edits", data);
    return err.empty() ? WorldFile::parse_edits(data, r_edits) : err;
}

// The records past the scene's (the replayed edits and their moment blobs), as the GPU would read them.
std::vector<TableRecord> edit_records(const SDFCascade &p_sdf) {
    const uint32_t scene_end = static_cast<const GpuTable *>(p_sdf.get_gpu_table())->get_scene_end();
    const std::shared_ptr<const InstanceSnapshot> snap = p_sdf.get_snapshot();
    std::vector<TableRecord> out;
    for (uint32_t i = scene_end; i < snap->records.size(); i++) {
        out.push_back(snap->records[i]);
    }
    return out;
}

bool same_records(const TableRecord &p_a, const TableRecord &p_b) {
    return p_a.def == p_b.def && p_a.op == p_b.op && p_a.removed == p_b.removed && p_a.anchor == p_b.anchor &&
            std::memcmp(&p_a.k, &p_b.k, sizeof(float)) == 0 && std::memcmp(p_a.rotation, p_b.rotation, sizeof(p_a.rotation)) == 0 &&
            std::memcmp(p_a.params, p_b.params, sizeof(p_a.params)) == 0 && p_a.max_lod == p_b.max_lod && p_a.salt == p_b.salt;
}

} // namespace

TEST_CASE("WorldFile: an edits file round-trips bit for bit, int64 anchors as text") {
    const std::vector<SDFTable::Edit> edits = corner_edits();
    const ov::str text = WorldFile::edits_text(edits);
    CHECK(ov::contains(text, "\"9007199254740993\"")); // 2^53 + 1: a double would read 2^53
    CHECK(ov::contains(text, "\"-576460752303423491\""));
    std::vector<SDFTable::Edit> back;
    CHECK(read_edits(text, back) == ov::str());
    CLIMATE_REQUIRE(back.size() == edits.size());
    for (size_t i = 0; i < edits.size(); i++) {
        CAPTURE(i);
        CHECK(back[i] == edits[i]);
    }
    CHECK(!ov::contains(WorldFile::edits_text({ edits[0] }), "rotation")); // the identity left out
    CHECK(ov::contains(WorldFile::edits_text({ edits[0] }), "\"params\":[0.10000000149011612]")); // 0.1f's double, the trailing +0s left out
    std::vector<SDFTable::Edit> none;
    CHECK(read_edits(WorldFile::edits_text(none), none) == ov::str());
    CHECK(none.empty());
}

TEST_CASE("WorldFile: a malformed edits file is refused with the entry and the reason") {
    std::vector<SDFTable::Edit> edits = corner_edits();
    const std::vector<SDFTable::Edit> before = edits;
    const char *anchor = R"("anchor": ["1", "2", "3"])";
    const std::pair<ov::str, const char *> cases[] = {
        { R"({"version": 2, "edits": []})", "the only version" },
        { R"({"edits": []})", "the only version" },
        { R"({"version": 1, "edit": []})", "unknown key 'edit'" },
        { R"({"version": 1, "edits": {}})", "'edits' is a list" },
        { ov::str(R"({"version": 1, "edits": [{"type": "sphere", "op": "intersect", )") + anchor + "}]}", "entry 0: 'op' is union" },
        { ov::str(R"({"version": 1, "edits": [{"type": "sphere", "op": "union", )") + anchor + R"(}, {"op": "union", )" + anchor + "}]}",
                "entry 1: 'type'" },
        { R"({"version": 1, "edits": [{"type": "sphere", "op": "union", "anchor": [1, 2, 3]}]})", "LOD0 voxels as int64 strings" },
        { R"({"version": 1, "edits": [{"type": "sphere", "op": "union", "anchor": ["1", "2.5", "3"]}]})", "int64 strings" },
        { R"({"version": 1, "edits": [{"type": "sphere", "op": "union", "anchor": ["1", "2", "99999999999999999999"]}]})", "int64 strings" },
        { ov::str(R"({"version": 1, "edits": [{"type": "sphere", "op": "union", "rotation": [1, 0, 0, 0, 1, 0, 0, 0], )") + anchor + "}]}",
                "'rotation' is 9 numbers" },
        { ov::str(R"({"version": 1, "edits": [{"type": "sphere", "op": "union", "params": [1, 2, 3, 4, 5, 6, 7, 8, 9], )") + anchor + "}]}",
                "at most 8 numbers" },
        { ov::str(R"({"version": 1, "edits": [{"type": "sphere", "op": "union", "k": "1", )") + anchor + "}]}", "'k' is a number" },
        { ov::str(R"({"version": 1, "edits": [{"type": "sphere", "op": "union", "colour": 1, )") + anchor + "}]}", "unknown key 'colour'" },
    };
    for (const std::pair<ov::str, const char *> &c : cases) {
        CAPTURE(c.first);
        CHECK(ov::contains(read_edits(c.first, edits), c.second));
    }
    CLIMATE_REQUIRE(edits.size() == before.size());
    for (size_t i = 0; i < edits.size(); i++) {
        CHECK(edits[i] == before[i]); // untouched
    }
    WorldFile w;
    CHECK(ov::contains(parse(R"({"edits": 5})", w), "not a path or an object"));
    CHECK(ov::contains(parse(R"({"edits": "res://worlds/nowhere.edits.json"})", w), "cannot read"));
    CHECK(ov::contains(parse(R"({"instance_band": 0})", w), "'instance_band' is true or false"));
    CHECK(parse(R"({"instance_band": false, "edits": {"version": 1, "edits": [{"type": "die", "op": "union", "anchor": ["0", "0", "0"], "params": [4]}]}})", w) ==
            ov::str());
    CHECK(!w.instance_band);
    CLIMATE_REQUIRE(w.edits.size() == 1);
    CHECK(w.edits[0].type == "die");
    CHECK(w.edits[0].params[0] == 4.0f);
}

// The table's library names each type once (an edits file names them): the crown is no second "rock".
TEST_CASE("SDFTable: the library's type names are unique and found by name") {
    GroundSDF sdf;
    ChunkParam param;
    sdf.configure(param, 15);
    CHECK(sdf.get_world_error() == ov::str());
    const DefLibrary &lib = sdf.get_library();
    for (int i = 0; i < lib.size(); i++) {
        CAPTURE(lib[i].name);
        CHECK(lib.find(lib[i].name) == i);
    }
    for (const char *name : { "sphere", "capsule", "rock", "crown", "swiss_cube", "die", "stairs", "tunnel_cube", "moment_blob" }) {
        CAPTURE(name);
        CHECK(lib.find(name) >= 0);
    }
    CHECK(lib.find("crown") != lib.find("rock"));
    CHECK(lib.find("no such type") == -1);
}

// configure replays the edits after the scene: the same records as add_instance makes, live (moved, removed), saved and
// replayed again to the same bits; their ids past the scene's, so they fold after the streamed instances (compose_list).
TEST_CASE("SDFTable: the edits replay after the scene, live, and save to the same records") {
    GroundSDF sdf;
    ChunkParam param;
    sdf.set_zone(SDFTable::Zone{ 200.0 });
    sdf.set_instances(SDFTable::counted(default_species(), Archetype::Species::ROCK, 400, 200.0));
    const VoxelPos centre = [&] {
        GroundSDF probe; // the ground under the zone's centre, where a carving sphere goes
        probe.configure(param, 15);
        const SDFTable::Zone zone;
        return probe.ground_at(std::llround(zone.x_m / param.voxel_size), std::llround(zone.z_m / param.voxel_size));
    }();
    std::vector<SDFTable::Edit> edits = corner_edits();
    edits[0].anchor = centre;
    edits[0].params[0] = 20.0f; // 10 m
    edits[1].anchor = centre + VoxelPos(400, 0, 0);
    edits[2].anchor = centre + VoxelPos(-400, 30, 0);
    sdf.set_edits(edits);
    param.skirt_depth = ov::max(param.skirt_depth, sdf.min_skirt_depth(param, 15));
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_world_error() == ov::str());
    CHECK(sdf.get_stream() != nullptr); // the rocks stream
    const std::vector<SDFTable::Edit> live = sdf.live_edits();
    CLIMATE_REQUIRE(live.size() == edits.size());
    for (size_t i = 0; i < edits.size(); i++) {
        CHECK(live[i] == edits[i]);
    }
    const std::vector<TableRecord> replayed = edit_records(sdf);
    CLIMATE_REQUIRE(replayed.size() >= edits.size()); // plus the crown's moment blobs
    CHECK(replayed[0].def == uint8_t(sdf.get_library().find("sphere")));
    CHECK(replayed[0].op == Combine::SUBTRACT);
    CHECK(replayed[0].anchor == centre);
    // The carving sphere folds last: the air at its centre, whatever rock streams there.
    WorldPos at;
    at.voxel = centre;
    CHECK(sdf.evaluate(at, 0) > 5.0f);

    // Live: a move rewrites the entry, a removal drops it, an add appends one; the same as add_instance's record.
    const uint32_t scene_end = static_cast<const GpuTable *>(sdf.get_gpu_table())->get_scene_end();
    const int sphere_id = int(scene_end);
    CHECK(sdf.move_instance(sphere_id, centre + VoxelPos(0, 2, 0), nullptr));
    const float params[8] = { 6.0f, 2.0f };
    const int added = sdf.add_instance(sdf.get_library().find("capsule"), centre + VoxelPos(0, 0, 300), edits[1].rotation, params, int(Combine::UNION), 1.5f);
    CHECK(added > sphere_id);
    std::vector<SDFTable::Edit> now = sdf.live_edits();
    CLIMATE_REQUIRE(now.size() == 4);
    CHECK(now[0].anchor == centre + VoxelPos(0, 2, 0));
    CHECK(now[3].type == "capsule");
    CHECK(now[3].k == 1.5f);
    const int crown_id = int(scene_end) + 2; // after the sphere and the capsule (blobs come after their record)
    CHECK(sdf.get_snapshot()->records[crown_id].def == uint8_t(sdf.get_library().find("crown")));
    CHECK(sdf.remove_instance(crown_id));
    now = sdf.live_edits();
    CLIMATE_REQUIRE(now.size() == 3);
    CHECK(now[2].type == "capsule");

    // Saved, read back and replayed: the same records, bit for bit.
    std::vector<SDFTable::Edit> saved;
    CHECK(read_edits(WorldFile::edits_text(now), saved) == ov::str());
    CLIMATE_REQUIRE(saved.size() == now.size());
    GroundSDF again;
    again.set_zone(SDFTable::Zone{ 200.0 });
    again.set_instances(SDFTable::counted(default_species(), Archetype::Species::ROCK, 400, 200.0));
    again.set_edits(saved);
    again.configure(param, 15);
    CLIMATE_REQUIRE(again.get_world_error() == ov::str());
    const std::vector<TableRecord> first = edit_records(sdf), second = edit_records(again);
    std::vector<TableRecord> live_first;
    for (const TableRecord &r : first) {
        if (!r.removed && r.op != Combine::ACCUM) {
            live_first.push_back(r);
        }
    }
    std::vector<TableRecord> live_second;
    for (const TableRecord &r : second) {
        if (r.op != Combine::ACCUM) {
            live_second.push_back(r);
        }
    }
    CLIMATE_REQUIRE(live_first.size() == live_second.size());
    for (size_t i = 0; i < live_first.size(); i++) {
        CAPTURE(i);
        CHECK(same_records(live_first[i], live_second[i]));
    }
}

TEST_CASE("SDFTable: an edit naming no type, or refused, refuses the world with its entry") {
    std::vector<SDFTable::Edit> edits = corner_edits();
    edits[1].type = "cube";
    GroundSDF unknown;
    unknown.set_edits(edits);
    ChunkParam param;
    unknown.configure(param, 15);
    CHECK(ov::contains(unknown.get_world_error(), "edits: entry 1: no library type is named 'cube'"));
    CHECK(unknown.program().nodes.empty()); // no world
    edits = corner_edits();
    edits[2].rotation[0] = 2.0f;
    GroundSDF skewed;
    skewed.set_edits(edits);
    skewed.configure(param, 15);
    CHECK(ov::contains(skewed.get_world_error(), "edits: entry 2: the rotation is not orthonormal"));
    edits = corner_edits();
    edits[0].op = Combine::ACCUM;
    GroundSDF blob;
    blob.set_edits(edits);
    blob.configure(param, 15);
    CHECK(ov::contains(blob.get_world_error(), "edits: entry 0: op 4"));
}

// The GPU table holds the scene, the replayed edits and 4096 more (runtime edits).
TEST_CASE("SDFTable: the table's capacity counts the replayed edits") {
    std::vector<SDFTable::Edit> edits(13000);
    for (size_t i = 0; i < edits.size(); i++) {
        edits[i].type = "sphere";
        edits[i].op = Combine::SUBTRACT;
        edits[i].anchor = VoxelPos(int64_t(i % 100) * 8, 600, int64_t(i / 100) * 8);
        edits[i].params[0] = 2.0f;
    }
    GroundSDF sdf;
    sdf.set_edits(edits);
    ChunkParam param;
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_world_error() == ov::str());
    const GpuTable &gpu = *static_cast<const GpuTable *>(sdf.get_gpu_table());
    CHECK(gpu.get_scene_end() == 0);
    CHECK(sdf.get_snapshot()->records.size() == 13000);
    CHECK(gpu.table_capacity() == 13000 + 4096);
    CHECK(sdf.live_edits().size() == 13000);
}
