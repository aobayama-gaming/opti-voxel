#pragma once

#include "biome_compiler.hpp"
#include "sdf_table.hpp"
#include "species.hpp"

#include "ov/str.hpp"
#include "ov/value.hpp"

#include <vector>

// A world on disk: what a run loads (World::Settings::world_path), every terrain setting in data. The format (JSON;
// "comment" keys are ignored, any other unknown key is refused, every key is optional and defaults to the field below):
//
//   {
//     "engine": "engine.json" or { "voxel_m": 0.05, "lods": 25 },   the master variables
//     "climate": "climate.json" or { "fields": [ field... ] },       the fields that place the biomes; none: no biomes
//     "planet": { "radius_m": 0, "ring_m": 0 },
//     "look": { "atmosphere": true, "atmosphere_scale": 0, "clouds": true, "relief_m": 3276.8 },
//     "zone": { "side_m": 0, "centre_m": [-50, -150] },
//     "instances": [ { "species": "conifer", "cell_m": 5, "share": 1 }... ],
//     "features": [ { "kind": "lava_lake", "cell_m": 640, "chance": 0.6, "radius_m": [18, 45] }... ],
//     "species": [ species... ],
//     "biomes": [ biome..., default biome ]
//   }
//
// A path is relative to the file naming it, or absolute (res://, /...). "fields" and "biomes" are the climate table of
// climate_parse.cpp (each biome a graph of notes, what it grows, its features), merged here and compiled there (at the
// engine's values). Without biomes, the ground is the template biome's (Climate::template_biome). "species" are what the
// instances name (species.hpp; Archetype::default_species without the key); "instances" what grows where no biome lists
// its own (and on a world without biomes), each entry a species, its cell and its share (Climate::parse_instances), over
// the "zone" (SDFTable::Zone: a square of side_m metres round centre_m, x and z; side 0: none); "features" likewise what
// stands on a biome that lists none, each entry a kind, its cell, its chance and its radius (Climate::parse_features).
struct WorldFile {
    // The world of a host or check given none.
    static constexpr const char *DEFAULT_PATH = "res://worlds/default.world.json";

    double voxel_m = 0.5; // LOD 0
    int lods = 15; // LODs 0 .. lods - 1 (at most ChunkMath::MAX_LOD_NUMBERS)
    ov::Value climate_table = ov::Value::object(); // Climate::parse's input, empty for none
    double planet_radius_m = 0.0; // > 0: a planet of this radius centred on the voxel (0, 0, 0), its +y pole at y = radius; 0 flat
    double planet_ring_m = 0.0; // > 0 with a planet: a torus round y, this ring's radius, planet_radius_m its tube's
    // The volumetric atmosphere over the base (Atmosphere) and its scale (AtmosphereMedium::for_base: 0 from the base).
    bool atmosphere = true;
    double atmosphere_scale = 0.0;
    bool clouds = true; // the volumetric clouds in it (Clouds), their layer's scale as the atmosphere's
    // The relief the sky stays above (metres; AtmosphereMedium::for_base, CloudLayer::for_base), a planet's
    // SDFCascade::PLANET_RELIEF of its radius at most (SDFCascade::relief_m): the template ground's 3.3 km cap.
    double relief_m = Climate::TEMPLATE_MAX_AMP_M;
    double zone_x_m = SDFTable::Zone().x_m; // SDFTable::set_zone
    double zone_z_m = SDFTable::Zone().z_m;
    double zone_side_m = 0.0;
    std::vector<Climate::InstanceShare> instances; // SDFTable::set_instances
    std::vector<Climate::FeatureShare> features; // SDFTable::set_features
    std::vector<Archetype::Species> species = Archetype::default_species(); // SDFTable::set_species

    // The file at p_path (a host path, ov::read_file) in r_world: empty, or why it was refused (r_world untouched).
    static ov::str load(const ov::str &p_path, WorldFile &r_world);
    // A file's parsed JSON, its relative paths from the directory p_dir.
    static ov::str parse(const ov::Value &p_data, const ov::str &p_dir, WorldFile &r_world);
};
