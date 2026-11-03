#pragma once

#include "ov/bytes.hpp"
#include "ov/executor.hpp"
#include "ov/value.hpp"

#include <vector>

// Report of a world file's climate table (climate.hpp), pure CPU (works with --headless): the table parsed against the
// cascade ground of the world's engine, then sampled on p_size x p_size points of one lattice Λ_m over the base's tangent
// plane (BaseSurface::tangent_point), each traced down the base's up to its ground surface (p_surface:
// SDFCascade::ground_along) or left on the plane (a debug slice: the fields are 3D, so a slice kilometres off the ground
// shows other biomes), m chosen so the slice spans about p_extent_km, centred on (p_center_x, p_center_z) (tangent metres; snapped to the
// lattice). Writes <out_dir>/report.md (per-biome share of the picks and mean weight, the pick drift, where each pair of
// biomes meets nearest the centre with the player's --spawn, the ground under each biome, per-field range, CPU time per
// point),
// biomes_blend.png (debug colours weighted), biomes_pick.png (the picked biome), the ground's height and shading maps,
// and field_<name>.png (each field, grey, -max|v| .. +max|v|).
// p_lattice >= 0 sets m (the extent is then p_size * 2^m voxels). Godot: ClimateReportNode (tools/climate_report.sh).
namespace ClimateReport {

ov::Value run(const ov::str &p_out_dir, const ov::str &p_world_path, double p_extent_km, int p_size, bool p_surface,
        double p_center_x, double p_center_z, int p_lattice = -1);

// A world's maps without files, for the biome mixer (its previews are the engine's): p_world a world file's JSON (its
// relative paths from p_dir), sampled as run() on p_size x p_size points (row-major, x right, z down), no picks. biomes:
// the debug colours weighted (biomes_blend.png) in the relief's light, shaded: the relief (ground_shaded.png), both RGB;
// altitude_m: the column's surface above the base, relief_m its largest magnitude; fields: each field the world names
// (not a warp's or a sub-mix's) its value, point-major in field_names' order. error: why the world was refused (nothing else set). p_priority LOW: its loop
// behind every frame's (the mixer's 3D view: a map drawn behind it never slows the terrain's lists).
struct Preview {
    ov::str error;
    int size = 0;
    int lattice = 0;
    double spacing_m = 0.0; // between points
    ov::bytes biomes;
    ov::bytes shaded;
    std::vector<float> altitude_m;
    double relief_m = 0.0;
    std::vector<float> fields;
    std::vector<ov::str> field_names;
    std::vector<ov::str> biome_names;
    double us_per_point = 0.0;
};
Preview preview(const ov::Value &p_world, const ov::str &p_dir, double p_extent_km, int p_size, bool p_surface, double p_center_x, double p_center_z,
        ov::Priority p_priority = ov::Priority::HIGH);

// Empty if p_json (a climate table's text) parses and the cascade of a voxel size and LOD count accepts it, else why
// not (the biome mixer checks an export with it before it overwrites world.json).
ov::str check(const ov::str &p_json, double p_voxel_size, int p_lods);

} // namespace ClimateReport
