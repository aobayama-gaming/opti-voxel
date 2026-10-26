#pragma once

#include "ov/value.hpp"

// Report of a climate table (climate.hpp), pure CPU (works with --headless): the table parsed against the cascade
// ground of a voxel size and LOD count, then sampled on p_size x p_size points of one lattice Λ_m over the base's tangent
// plane (BaseSurface::tangent_point), each traced down the base's up to its ground surface (p_surface:
// SDFCascade::ground_along) or left on the plane (a debug slice: the fields are 3D, so a slice kilometres off the ground
// shows other biomes), m chosen so the slice spans about p_extent_km, centred on (p_center_x, p_center_z) (tangent metres; snapped to the
// lattice). Writes <out_dir>/report.md (per-biome share of the picks and mean weight, the pick drift, where each pair of
// biomes meets nearest the centre with the player's --spawn, the ground under each biome, per-field range, CPU time per
// point),
// biomes_blend.png (debug colours weighted), biomes_pick.png (the picked biome) and field_<name>.png (each field,
// grey, -max|v| .. +max|v|) and heights.f32 / heights.json (the surface heights, for tools/landforms/compare.py).
// p_lattice >= 0 sets m (the extent is then p_size * 2^m voxels). Godot: ClimateReportNode (tools/climate_report.sh).
namespace ClimateReport {

ov::Value run(const ov::str &p_out_dir, const ov::str &p_climate_path, double p_voxel_size, int p_lods, double p_extent_km, int p_size, bool p_surface,
        double p_center_x, double p_center_z, int p_lattice = -1);

} // namespace ClimateReport
