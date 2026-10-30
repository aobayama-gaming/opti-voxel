#pragma once

#include "sdf_cascade.hpp"

#include "ov/on_error.hpp"
#include "ov/value.hpp"

#include <algorithm>
#include <iterator>
#include <vector>

// The terrains CascadeSpike (cascade_spike.cpp) checks and times: "8" / "4" channels, "4c" LINEAR4 with the example
// climate carried and modulating the ground, "4f" its fields without multipliers (carried only for the biome pick: the
// cost of the ground materials alone), "4p" the plain climate (the example before shaped bands: amplitudes only), "4s"
// the shaped test table (every operation: shapes, the multifractal state, warps, the biomes' curves, stretched octaves),
// "4d" adapter/godot/project/climate/fieldless_test.json (one biome multiplying the ground's octaves, no field: a modulation
// with no field group), "4w" adapter/godot/project/climate/world.json (eight biomes on three climate axes, 8 groups;
// needs 25 LODs), "4g"
// LINEAR4 on an Earth-sized sphere (SPHERE_RADIUS_M: the base's regions, BaseRegions, below its switch LOD).
namespace CascadeCheckSets {

enum class SpikeSet {
    H8,
    L4,
    L4C,
    L4F,
    L4P,
    L4S,
    L4D,
    L4W,
    L4G,
};
inline constexpr const char *SET_LABELS[] = { "8", "4", "4c", "4f", "4p", "4s", "4d", "4w", "4g" };

// The "4g" set's planet (metres), centred on the voxel (0, 0, 0).
inline constexpr double SPHERE_RADIUS_M = 6371000.0;

// The example climate is tuned for 5 cm voxels (its modulated octaves stay under the cap there).
inline constexpr float CLIMATE_VOXEL = 0.05f;

inline ov::str set_label(SpikeSet p_v) {
    return SET_LABELS[int(p_v)];
}

// False (and an error) for an unknown label.
inline bool parse_variant(const ov::str &p_label, SpikeSet &r_set) {
    for (int i = 0; i < int(std::size(SET_LABELS)); i++) {
        if (p_label == SET_LABELS[i]) {
            r_set = SpikeSet(i);
            return true;
        }
    }
    ov::str labels;
    for (const char *l : SET_LABELS) {
        labels += ov::str(labels.empty() ? "" : ", ") + l;
    }
    OV_ERR_FAIL_V_MSG(false, "CascadeSpike: unknown set '" + p_label + "' (" + labels + ").");
}

// The median of timings (0 for none).
inline double median(std::vector<double> p_v) {
    if (p_v.empty()) {
        return 0.0;
    }
    std::sort(p_v.begin(), p_v.end());
    return p_v[p_v.size() / 2];
}

// p_sdf configured for p_v (r_param: the variant's chunk parameters).
inline void configure_variant(SDFCascade &p_sdf, SpikeSet p_v, int p_lods, ChunkParam &r_param) {
    p_sdf.channel_set = p_v == SpikeSet::H8 ? Cascade::ChannelSet::HERMITE8 : Cascade::ChannelSet::LINEAR4;
    p_sdf.planet_radius_m = p_v == SpikeSet::L4G ? SPHERE_RADIUS_M : 0.0;
    if (p_v != SpikeSet::H8 && p_v != SpikeSet::L4 && p_v != SpikeSet::L4G) {
        r_param.voxel_size = CLIMATE_VOXEL;
        ov::Value data;
        const char *path = p_v == SpikeSet::L4P ? "res://climate/plain.json"
                : p_v == SpikeSet::L4S          ? "res://climate/shaped_test.json"
                : p_v == SpikeSet::L4D          ? "res://climate/fieldless_test.json"
                : p_v == SpikeSet::L4W          ? "res://climate/world.json"
                                                : "res://climate/example.json";
        const ov::str err = Climate::load(path, data);
        if (!err.empty()) {
            OV_ERR_PRINT("CascadeSpike: " + err + " (the set runs without its climate)");
        }
        if (p_v == SpikeSet::L4F) {
            ov::Value &biomes = data["biomes"];
            for (size_t i = 0; i < biomes.size(); i++) {
                biomes.at(i).erase("amp");
            }
        }
        p_sdf.climate_data = data;
    }
    p_sdf.configure(r_param, p_lods);
}

} // namespace CascadeCheckSets
