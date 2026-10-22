#pragma once

#include "sdf_cascade.hpp"

#include <iterator>

// The terrains CascadeSpike (cascade_spike.cpp) checks and times: "8" / "4" channels, "4c" LINEAR4 with the example
// climate carried and modulating the ground, "4f" its fields without multipliers (carried only for the biome pick: the
// cost of the ground materials alone), "4p" the plain climate (the example before shaped bands: amplitudes only), "4s"
// the shaped test table (every shaped-band operation: shapes, the multifractal state, warps, cell noise, height terms,
// terraces).
namespace CascadeSpikeSets {

enum class SpikeSet {
    H8,
    L4,
    L4C,
    L4F,
    L4P,
    L4S,
};
inline constexpr const char *SET_LABELS[] = { "8", "4", "4c", "4f", "4p", "4s" };

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
    ERR_FAIL_V_MSG(false, "CascadeSpike: unknown set '" + p_label + String("' (8, 4, 4c, 4f, 4p, 4s)."));
}

// p_sdf configured for p_v (r_param: the variant's chunk parameters).
inline void configure_variant(SDFCascade &p_sdf, SpikeSet p_v, int p_lods, ChunkParam &r_param) {
    p_sdf.channel_set = p_v == SpikeSet::H8 ? Cascade::ChannelSet::HERMITE8 : Cascade::ChannelSet::LINEAR4;
    if (p_v == SpikeSet::L4C || p_v == SpikeSet::L4F || p_v == SpikeSet::L4P || p_v == SpikeSet::L4S) {
        r_param.voxel_size = CLIMATE_VOXEL;
        Dictionary data;
        const char *path = p_v == SpikeSet::L4P ? "res://climate/plain.json" : (p_v == SpikeSet::L4S ? "res://climate/shaped_test.json" : "res://climate/example.json");
        const String err = Climate::load(path, data);
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
