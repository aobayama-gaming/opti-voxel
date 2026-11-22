#pragma once

#include "climate.hpp"
#include "sdf_table.hpp"
#include "species.hpp"

#include "ov/str.hpp"
#include "ov/value.hpp"

#include <vector>

// A world on disk: what a run loads (World::Settings::world_path), every terrain setting in data. The format (JSON;
// "comment" keys are ignored, any other unknown key is refused, every key but "biomes" is optional and defaults to the
// field below):
//
//   {
//     "engine": "engine.json" or { "voxel_m": 0.05, "lods": 25 },   the master variables
//     "climate": "climate.json" or { "fields": [ field... ] },       the fields that place the biomes; none: no fields
//     "planet": { "radius_m": 0, "ring_m": 0 },
//     "look": { "atmosphere": true, "atmosphere_scale": 0, "clouds": true },
//     "instances": [ { "species": "conifer", "cell_m": 5, "share": 1 }... ],
//     "features": [ { "kind": "lava_lake", "cell_m": 640, "chance": 0.6, "radius_m": [18, 45] }... ],
//     "feature_region": { "side_m": 0, "centre_m": [-50, -150] },
//     "species": "species.json" or [ species... ],
//     "biomes": "ground.biomes.json" or [ biome..., default biome ],
//     "instance_band": true,
//     "edits": "<name>.edits.json" or { "version": 1, "edits": [ edit... ] }
//   }
//
// A path is relative to the file naming it, or absolute (res://, /...); "species" and "biomes" files hold the list.
// "fields" and "biomes" are the climate table of climate_parse.cpp (each biome a graph of notes, what it grows, its
// features), merged here and compiled there (at the engine's values). A world names its ground: without biomes the
// cascade refuses it (SDFCascade::attach_climate); GROUND_BIOMES is the template ground's. "species" are what the
// instances name (species.hpp; none without the key, worlds/species.json today's trees and rocks); "instances" what grows
// where no biome lists its own, each entry a species, its cell and its share (Climate::parse_instances), over the whole
// ground (streamed: the view asks for what it sees); "features" likewise what stands on a biome that lists none, each
// entry a kind, its cell, its chance and its radius (Climate::parse_features), over the "feature_region" (SDFTable::Region:
// a square of side_m metres round centre_m, x and z), which a world whose lists name features must give until the
// features stream (SDFTable::set_feature_region's TODO).
// "instance_band" false keeps every instance whole at every LOD (SDFTable::set_instance_band: no erosion, no drop, no
// moment blobs; the shapes gallery's hard-CSG creases). "edits" are the runtime edits the world starts with
// (SDFTable::set_edits), what an edits file holds:
//
//   { "version": 1, "edits": [ { "type": "sphere", "op": "subtract", "anchor": ["-104", "600", "-190"],
//                                "rotation": [9 numbers], "params": [numbers], "k": 0 }... ] }
//
// one entry per live edit in id order (a moved edit's entry rewritten, a removed one dropped: the set, not a log).
// "type" a library type's name (TableDef::name), "op" a combine operator's (Archetype::COMBINE_NAMES), "anchor" the
// LOD0 voxel as three int64 strings (JSON numbers are doubles), "rotation" the rows of a rotation (absent: identity),
// "params" up to 8 numbers (voxels; the rest 0), "k" the smooth radius (voxels, absent: 0). The floats round-trip bit for
// bit (save_edits writes the shortest text of each).
struct WorldFile {
    // The world of a host or check given none.
    static constexpr const char *DEFAULT_PATH = "res://worlds/default.world.json";
    // The template ground's biomes (one biome, no fields): the bare ground's worlds name it, a new biome starts from it.
    static constexpr const char *GROUND_BIOMES = "res://worlds/ground.biomes.json";
    // Today's species (three trees, a boulder, a pillar): the worlds that grow instances name it, the checks read it.
    static constexpr const char *SPECIES = "res://worlds/species.json";

    double voxel_m = 0.5; // LOD 0
    int lods = 15; // LODs 0 .. lods - 1 (at most ChunkMath::MAX_LOD_NUMBERS)
    ov::Value climate_table = ov::Value::object(); // Terrain::compile's input, empty for none
    double planet_radius_m = 0.0; // > 0: a planet of this radius centred on the voxel (0, 0, 0), its +y pole at y = radius; 0 flat
    double planet_ring_m = 0.0; // > 0 with a planet: a torus round y, this ring's radius, planet_radius_m its tube's
    // The volumetric atmosphere over the base (Atmosphere) and its scale (AtmosphereMedium::for_base: 0 from the base).
    bool atmosphere = true;
    double atmosphere_scale = 0.0;
    bool clouds = true; // the volumetric clouds in it (Clouds), their layer's scale as the atmosphere's
    SDFTable::Region feature_region; // SDFTable::set_feature_region
    std::vector<Climate::InstanceShare> instances; // SDFTable::set_instances
    std::vector<Climate::FeatureShare> features; // SDFTable::set_features
    std::vector<Archetype::Species> species; // SDFTable::set_species
    bool instance_band = true; // SDFTable::set_instance_band
    std::vector<SDFTable::Edit> edits; // SDFTable::set_edits

    // The file at p_path (a host path, ov::read_file) in r_world: empty, or why it was refused (r_world untouched).
    static ov::str load(const ov::str &p_path, WorldFile &r_world);
    // A file's parsed JSON, its relative paths from the directory p_dir.
    static ov::str parse(const ov::Value &p_data, const ov::str &p_dir, WorldFile &r_world);
    // A file's text (an editor's world, never written), its relative paths from p_dir.
    static ov::str parse_text(const ov::str &p_text, const ov::str &p_dir, WorldFile &r_world);
    // p_text as a JSON object in r_data: empty, or why not (p_name: the text's name in the message).
    static ov::str parse_object(const ov::str &p_text, const ov::str &p_name, ov::Value &r_data);
    // The JSON list in the file at p_path (a "species" or "biomes" file) in r_list: empty, or why not.
    static ov::str load_list(const ov::str &p_path, ov::Value &r_list);
    // The template ground's climate table ({"fields": [], "biomes": GROUND_BIOMES'}) in r_table: empty, or why not.
    static ov::str ground_table(ov::Value &r_table);
    // The species list in the file at p_path (SPECIES: today's) in r_species: empty, or why not.
    static ov::str load_species(const ov::str &p_path, std::vector<Archetype::Species> &r_species);

    // An edits file's parsed JSON in r_edits: empty, or why it was refused (r_edits untouched).
    static ov::str parse_edits(const ov::Value &p_data, std::vector<SDFTable::Edit> &r_edits);
    // The edits file at p_path in r_edits: empty, or why not.
    static ov::str load_edits(const ov::str &p_path, std::vector<SDFTable::Edit> &r_edits);
    // p_edits as an edits file's text (one entry per line).
    static ov::str edits_text(const std::vector<SDFTable::Edit> &p_edits);
    // p_edits written to p_path (a host path, ov::write_file): empty, or why not.
    static ov::str save_edits(const ov::str &p_path, const std::vector<SDFTable::Edit> &p_edits);
};
