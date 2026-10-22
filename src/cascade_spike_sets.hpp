#pragma once

#include "sdf_cascade.hpp"

#include <iterator>

// The terrains CascadeSpike (cascade_spike.cpp) checks and times: "8" / "4" channels, "4c" LINEAR4 with the example
// climate carried and modulating the ground, "4f" its fields without multipliers (no modulation: nothing carries them
// until a biome pick reads them, so the bare ground's cost), "4d" project/climate/fieldless_test.json (one biome multiplying the ground's octaves, no field: a modulation with no
// field group).
namespace CascadeSpikeSets {

enum class SpikeSet {
    H8,
    L4,
    L4C,
    L4F,
    L4D,
};
inline constexpr const char *SET_LABELS[] = { "8", "4", "4c", "4f", "4d" };

// The example climate is tuned for 5 cm voxels (its modulated octaves stay under the cap there).
inline constexpr float CLIMATE_VOXEL = 0.05f;

inline String set_label(SpikeSet p_v) {
    return SET_LABELS[int(p_v)];
}

// False (and an error) for an unknown label.
inline bool parse_variant(const String &p_label, SpikeSet &r_set) {
    for (int i = 0; i < int(std::size(SET_LABELS)); i++) {
        if (p_label == SET_LABELS[i]) {
            r_set = SpikeSet(i);
            return true;
        }
    }
    ERR_FAIL_V_MSG(false, "CascadeSpike: unknown set '" + p_label + String("' (8, 4, 4c, 4f, 4d)."));
}

// p_sdf configured for p_v (r_param: the variant's chunk parameters).
inline void configure_variant(SDFCascade &p_sdf, SpikeSet p_v, int p_lods, ChunkParam &r_param) {
    p_sdf.channel_set = p_v == SpikeSet::H8 ? Cascade::ChannelSet::HERMITE8 : Cascade::ChannelSet::LINEAR4;
    if (p_v != SpikeSet::H8 && p_v != SpikeSet::L4) {
        r_param.voxel_size = CLIMATE_VOXEL;
        Dictionary data;
        const char *path = p_v == SpikeSet::L4D ? "res://climate/fieldless_test.json" : "res://climate/example.json";
        const String err = Climate::load(path, data);
        if (!err.is_empty()) {
            ERR_PRINT("CascadeSpike: " + err + String(" (the set runs without its climate)"));
        }
        if (p_v == SpikeSet::L4F) {
            const Array biomes = data["biomes"];
            for (int64_t i = 0; i < biomes.size(); i++) {
                Dictionary(biomes[i]).erase("amp");
            }
        }
        p_sdf.climate_data = data;
    }
    p_sdf.configure(r_param, p_lods);
}

} // namespace CascadeSpikeSets
