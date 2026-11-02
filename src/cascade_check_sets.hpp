#pragma once

#include "sdf_cascade.hpp"
#include "world_file.hpp"

#include "ov/on_error.hpp"
#include "ov/value.hpp"

#include <algorithm>
#include <iterator>
#include <vector>

// The terrains CascadeSpike (cascade_spike.cpp) checks and times, each a world file (voxel and LODs from its engine): "4" the
// bare ground of the check's own world, "4c" the example climate carried and modulating the ground, "4f" its fields
// without multipliers (carried only for the biome pick: the cost of the ground materials alone), "4p" the plain climate
// (the example before shaped bands: amplitudes only), "4s" the shaped test table (every operation: shapes, the multifractal
// state, warps, the biomes' curves, stretched octaves), "4d" fieldless_test.json (one biome multiplying the ground's
// octaves, no field: a modulation with no field group), "4w" the game's world.json (eight biomes on three climate axes, 8
// groups), "4g" an Earth-sized sphere (the base's regions, BaseRegions, below its switch LOD), "4n" notes_test.json (several
// notes per octave, modulated: the band table's other notes, their BandMod slots), "4e" programs_test.json (sample
// programs: smooth and hard min / max, a field read at the samples, a curve of a program).
namespace CascadeCheckSets {

enum class SpikeSet {
    L4,
    L4C,
    L4F,
    L4P,
    L4S,
    L4D,
    L4W,
    L4G,
    L4N,
    L4E,
};
inline constexpr const char *SET_LABELS[] = { "4", "4c", "4f", "4p", "4s", "4d", "4w", "4g", "4n", "4e" };

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

// The world file of set p_v; p_world for the bare set.
inline ov::str set_world(SpikeSet p_v, const ov::str &p_world) {
    switch (p_v) {
        case SpikeSet::L4:
            return p_world;
        case SpikeSet::L4G:
            return "res://worlds/earth25.world.json";
        case SpikeSet::L4P:
            return "res://climate/plain.json";
        case SpikeSet::L4S:
            return "res://climate/shaped_test.json";
        case SpikeSet::L4D:
            return "res://climate/fieldless_test.json";
        case SpikeSet::L4W:
            return "res://climate/world.json";
        case SpikeSet::L4N:
            return "res://climate/notes_test.json";
        case SpikeSet::L4E:
            return "res://climate/programs_test.json";
        default:
            return "res://climate/example.json";
    }
}

// p_sdf configured for p_v (r_param: the variant's chunk parameters); returns its LOD count.
inline int configure_variant(SDFCascade &p_sdf, SpikeSet p_v, const ov::str &p_world, ChunkParam &r_param) {
    WorldFile world;
    const ov::str err = WorldFile::load(set_world(p_v, p_world), world);
    if (!err.empty()) {
        OV_ERR_PRINT("CascadeSpike: " + err + " (the set runs on the default world)");
    }
    r_param.voxel_size = float(world.voxel_m);
    p_sdf.planet_radius_m = world.planet_radius_m;
    p_sdf.planet_ring_m = world.planet_ring_m;
    if (p_v == SpikeSet::L4F && err.empty()) {
        // Every biome on the template's notes (constant: no modulation), its curve kept (the fields carried).
        ov::Value &biomes = world.climate_table["biomes"];
        for (size_t i = 0; i < biomes.size(); i++) {
            ov::Value graph = Climate::template_graph(world.voxel_m);
            const ov::Value &nodes = biomes[i]["graph"]["nodes"];
            for (size_t n = 0; n < nodes.size(); n++) {
                if (nodes[n]["kind"].as_str() == "curve") {
                    ov::Value curve = nodes[n];
                    curve["in"] = "ground";
                    graph["nodes"].at(1)["height"] = curve["id"];
                    graph["nodes"].push_back(curve);
                }
            }
            biomes.at(i)["graph"] = graph;
        }
    }
    p_sdf.climate_data = world.climate_table;
    p_sdf.configure(r_param, world.lods);
    return world.lods;
}

} // namespace CascadeCheckSets
