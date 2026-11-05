#pragma once

#include "sdf_cascade.hpp"
#include "world_file.hpp"

#include "ov/on_error.hpp"
#include "ov/value.hpp"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

// The terrains CascadeSpike (cascade_spike.cpp) checks and times, each a world file (voxel and LODs from its engine): "4" the
// bare ground of the check's own world (the template's), "4c" the example climate, "4f" its biomes on the template's notes
// with their curves (the weights carried for the curves and the biome pick: the cost of the climate alone), "4p" the plain
// climate (amplitudes only), "4s" the shaped test table (shapes, multifractals, warps, curves, stretched octaves), "4d"
// fieldless_test.json (one biome, no field), "4w" eight_biomes_test.json (eight biomes on three climate axes), "4g" an
// Earth-sized sphere (the base's regions, BaseRegions, below its switch LOD), "4n" notes_test.json (several notes per
// octave), "4e" programs_test.json (smooth and hard min / max, a field read in a graph, a curve of a node), "4r"
// graph_test.json (free graphs: a warp before its shape, shapes past the notes, a mix row past the notes), "4v"
// caves_test.json (caves: smin(mix, up - C) under a band's tent, the base's up read). Each runs on the cascade v2
// program the world compiles to (docs/cascade-v2.md).
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
    L4R,
    L4V,
};
inline constexpr const char *SET_LABELS[] = { "4", "4c", "4f", "4p", "4s", "4d", "4w", "4g", "4n", "4e", "4r", "4v" };
// A set's generic variant (CascadeSpike's option generic): its climate compiled generic (SDFCascade::specialize false).
inline constexpr int GENERIC = 64;
inline SpikeSet base_set(SpikeSet p_v) {
    return SpikeSet(int(p_v) & ~GENERIC);
}

inline ov::str set_label(SpikeSet p_v) {
    return SET_LABELS[int(base_set(p_v))] + ov::str(int(p_v) & GENERIC ? " generic" : "");
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
    switch (base_set(p_v)) {
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
            return "res://climate/eight_biomes_test.json";
        case SpikeSet::L4N:
            return "res://climate/notes_test.json";
        case SpikeSet::L4E:
            return "res://climate/programs_test.json";
        case SpikeSet::L4R:
            return "res://climate/graph_test.json";
        case SpikeSet::L4V:
            return "res://climate/caves_test.json";
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
    ov::Value ground;
    if (base_set(p_v) == SpikeSet::L4F && err.empty() && SDFCascade::template_table(world.voxel_m, ground).empty()) {
        // Every biome on the template's notes (every biome alike: blend factoring takes them out of the weights), its
        // curve kept (the fields carried).
        ov::Value &biomes = world.climate_table["biomes"];
        for (size_t i = 0; i < biomes.size(); i++) {
            ov::Value graph = std::as_const(ground)["biomes"][size_t(0)]["graph"];
            const ov::Value &nodes = biomes[i]["graph"]["nodes"];
            for (size_t n = 0; n < nodes.size(); n++) {
                if (nodes[n]["kind"].as_str() == "curve") {
                    ov::Value curve = nodes[n];
                    curve["in"] = "ground";
                    for (size_t o = 0; o < graph["nodes"].size(); o++) {
                        if (graph["nodes"][o]["kind"].as_str() == "output") {
                            graph["nodes"].at(o)["height"] = curve["id"];
                        }
                    }
                    graph["nodes"].push_back(curve);
                }
            }
            biomes.at(i)["graph"] = graph;
        }
    }
    p_sdf.climate_data = world.climate_table;
    p_sdf.specialize = (int(p_v) & GENERIC) == 0;
    p_sdf.configure(r_param, world.lods);
    return world.lods;
}

} // namespace CascadeCheckSets
