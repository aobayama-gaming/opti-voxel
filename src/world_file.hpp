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
//     "seed": 48211,                                                 the planet's: every note and field mixed with it
//     "shape": { "kind": "flat" | "sphere" | "torus", "radius_m": 2000, "ring_m": 9000 },
//     "sea": { "level_m": 0, "water": { "deep": [r, g, b], "shallow": [r, g, b], "clarity_m": 12 },
//              "shore": { "below_m": 2, "above_m": 3, "sand": [r, g, b] } },
//     "climate": "climate.json" or { "fields": "fields.json" or [ field... ],
//                                    "<field name>": { "mean": 0.4, "poles": -0.9 }... },
//     "atmosphere": { "density": 1, "tint": [1, 1, 1] },
//     "weather": { "on": true, "cell_m": 0, "top_m": 600, "tick_hz": 2, "time_scale": 1, "storm_m": 0,
//                  "spin_up_s": 600, "day_s": 300, "inversion_m": 300, "exchange": 15 },
//     "look": { "atmosphere": true, "atmosphere_scale": 0, "clouds": true },
//     "instances": [ { "species": "conifer", "cell_m": 5, "share": 1 }... ],
//     "features": [ { "kind": "lava_lake", "cell_m": 640, "chance": 0.6, "radius_m": [18, 45] }... ],
//     "feature_region": { "side_m": 0, "centre_m": [-50, -150] },
//     "species": "species.json" or [ species... ],
//     "biomes": "ground.biomes.json" or [ biome or "<name>.biome.json"..., default biome ],
//     "instance_band": true,
//     "edits": "<name>.edits.json" or { "version": 1, "edits": [ edit... ] }
//   }
//
// A planet file (*.planet.json) is a world file: one format, one loader. "seed" (a whole number in [0, 2^32); 0 or
// absent: the identity, the table's own noises) gives each planet its own noises from the same biomes (Climate::mix_seed;
// the instance and feature scatters keep theirs for now). "shape" is the base: flat (the datum, no radius), a sphere of
// radius_m centred on the voxel (0, 0, 0), its +y pole at y = radius, or a torus round y, ring_m its ring and radius_m its
// tube (BaseSurface::planet refuses one it can't hold); the old "planet" key is refused (renamed "shape", with a kind).
// "sea" (none without it): its level_m above the base, what a graph's "sea" node reads and under which (plus the shore's
// above_m) no tree or rock grows (SDFTable::set_site_floor_m); its water (linear colours, the depth of clear water) and
// shore (the sand band's depth below and height above the level, its colour) are the sea pass's look (Sea, sea.hpp; refused
// on a torus). "climate": its
// fields (a list, or a file holding it), and per field name a mean (added) and poles (times the latitude n_y^2: 0 at the
// equator, 1 at the poles; refused on a flat base and a torus), merged into the fields (climate_parse.cpp); a field name
// that isn't one is refused. "atmosphere": the air's density (times every scattering and absorption coefficient) and the
// tint of its Rayleigh scattering (AtmosphereMedium::for_base); "weather" the simulated weather over a sphere (on there
// by default, refused on a flat shape or a torus; docs/weather.md): its grid's cells' width in metres (0 or absent: 64 a cube face's side), the top of its levels (about cell_m apart), its ticks a
// second, the game seconds a real one simulates, a storm's radius (the Coriolis comes from
// it; 0 or absent: a quarter of the radius) and the game seconds simulated before the first frame; "look" whether there is air and clouds, and their scale.
// A path is relative to the file naming it, or absolute (res://, /...); "species" and "biomes" files hold the list, a
// string in a "biomes" list a file holding one biome. The default biome (the one without "at") is moved last; two are
// refused.
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
    // "shape": a sphere's radius (a torus's tube), a torus's ring; 0 the flat datum.
    double planet_radius_m = 0.0; // > 0: a planet of this radius centred on the voxel (0, 0, 0), its +y pole at y = radius; 0 flat
    double planet_ring_m = 0.0; // > 0 with a planet: a torus round y, this ring's radius, planet_radius_m its tube's
    uint32_t seed = 0; // "seed" (Climate::Ground::seed)
    // "sea": its level above the base and its look (the sea pass's), on false without the key.
    struct Sea {
        bool on = false;
        double level_m = 0.0;
        Float3 deep{ 0.02f, 0.09f, 0.16f };
        Float3 shallow{ 0.1f, 0.45f, 0.45f };
        double clarity_m = 12.0;
        double shore_below_m = 2.0;
        double shore_above_m = 3.0;
        Float3 sand{ 0.76f, 0.7f, 0.5f };
    };
    Sea sea;
    // "weather" (docs/weather.md): on by default over a sphere, never over a flat shape or a torus.
    struct Weather {
        bool on = false;
        double cell_m = 0.0; // the grid's cells' width (WeatherGrid::for_base; 0: DEFAULT_SIDE cells a face side)
        double top_m = 600.0; // the levels' top over the ground, about cell_m apart (WeatherFluid)
        double tick_hz = 2.0; // 2026-11-26: a tick costs ~250 ms at 70 cells a face side, 5 a second fell behind
        double time_scale = 1.0;
        double storm_m = 0.0; // a storm seed's radius; 0: 1 / 8 of the planet's
        double spin_up_s = 600.0;
        double day_s = 300.0; // the weather's day: its Coriolis (the twist: storms spin at the game's scale)
        double inversion_m = 300.0; // the trade inversion's height in the start sounding
        double exchange = 15.0; // the surface fluxes' twist (WeatherMoist::Params::exchange)
        bool convection = true; // the shallow convection's updraft per column (WeatherConvection)
        bool cloud_spread = true; // the partial cloud (WeatherMoist::Params::spread; false: all-or-nothing)
    };
    Weather weather;
    // The volumetric atmosphere over the base (Atmosphere) and its scale (AtmosphereMedium::for_base: 0 from the base).
    bool atmosphere = true;
    double atmosphere_scale = 0.0;
    // "atmosphere": the air's density and its Rayleigh tint (AtmosphereMedium::for_base; 1: Earth's air).
    double atmosphere_density = 1.0;
    Float3 atmosphere_tint{ 1.0f, 1.0f, 1.0f };
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
