#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// Report of a climate table (climate.hpp), pure CPU (works with --headless): the table parsed against the cascade
// ground of a voxel size and LOD count, then sampled on p_size x p_size columns of one lattice Λ_m, each at its ground
// surface (p_surface: the cascade's ground_y) or all through the datum (a debug slice: the fields are 3D, so a datum
// slice kilometres off the ground shows other biomes), m chosen so the slice spans about p_extent_km, centred on the
// column p_center (metres, x and z; snapped to the lattice). Writes
// <out_dir>/report.md (per-biome share of the picks and mean weight, where each pair of biomes meets nearest the
// centre with the player's --spawn, per-field range, CPU time per point),
// biomes_blend.png (debug colours weighted), biomes_pick.png (the picked biome) and field_<name>.png (each field,
// grey, -max|v| .. +max|v|). Run: tools/climate_report.sh
class ClimateReport : public Node {
    GDCLASS(ClimateReport, Node)

protected:
    static void _bind_methods();

public:
    Dictionary run(const String &p_out_dir, const String &p_climate_path, double p_voxel_size, int p_lods, double p_extent_km, int p_size, bool p_surface,
            const Vector2 &p_center);
};
