#pragma once

#include "sdf_cascade.hpp"

#include <iterator>

// The terrains CascadeSpike (cascade_spike.cpp) checks and times: "8" / "4" channels, "4c" LINEAR4 with the example
// climate carried and modulating the ground, "4f" its fields without multipliers (carried nowhere on the GPU: unused
// fields must cost nothing).
namespace CascadeSpikeSets {

enum class SpikeSet {
    H8,
    L4,
    L4C,
    L4F,
};
inline constexpr const char *SET_LABELS[] = { "8", "4", "4c", "4f" };

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
    ERR_FAIL_V_MSG(false, "CascadeSpike: unknown set '" + p_label + String("' (8, 4, 4c, 4f)."));
}

// p_sdf configured for p_v (r_param: the variant's chunk parameters).
inline void configure_variant(SDFCascade &p_sdf, SpikeSet p_v, int p_lods, ChunkParam &r_param) {
    p_sdf.channel_set = p_v == SpikeSet::H8 ? Cascade::ChannelSet::HERMITE8 : Cascade::ChannelSet::LINEAR4;
    if (p_v == SpikeSet::L4C || p_v == SpikeSet::L4F) {
        r_param.voxel_size = CLIMATE_VOXEL;
        Dictionary data;
        const String err = Climate::load("res://climate/example.json", data);
        ERR_FAIL_COND_MSG(!err.is_empty(), "CascadeSpike: " + err);
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
