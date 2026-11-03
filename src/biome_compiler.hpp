#pragma once

#include "climate.hpp"

#include "ov/str.hpp"
#include "ov/value.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

// The biome compiler: every biome's graph (climate_parse.cpp's format) into the one ground the cascade carries. A biome
// lists exactly the notes it mixes, with any seed, amplitude and stretch; the compiler keeps each note once whatever
// biome mixes it (identity: octave, seed, stretch), so a biome more costs the notes it adds only, and turns each biome's
// mix rows into its amplitude, shape, multifractal gain and warp on each note (Amp): the cascade's modulation inserts
// sum_b w_b (the biome's term) per note, one noise read whatever the number of biomes. A note every biome plays plain at
// one amplitude is a constant band (no modulation). docs/climate.md's "Biome graphs" part is this.
namespace Climate {

// A note node: the octaves lo .. hi (lo == hi for one wavelength), its seed, its amplitude rule (in LOD0 voxels) and its
// stretch along the base's up.
struct NoteNode {
    int lo = 0;
    int hi = 0;
    bool range = false; // from_m / to_m: the octaves the LOD count has (a single wavelength past them is refused)
    uint32_t seed = 0;
    double steepness = 0.0; // > 0: octave k's amplitude is steepness * 2^k
    double amp = 0.0; // else the coarsest octave's, times persistence per finer one
    double persistence = 0.5;
    double max_amp = INFINITY; // every octave's at most this
    int stretch = 0; // log2 of stretch_y (Band::stretch: octave k takes at most k - BAND_SHIFT of it)
};

// One mix row of a biome's graph: a note read through a warp, a shape and a multifractal (each optional), at volume x.
struct GraphRow {
    NoteNode note;
    float x = 1.0f;
    bool has_shape = false;
    Shape shape;
    float gain = 0.0f;
    float warp = 0.0f; // LOD0 voxels per unit of the warp fields
    int warp_fields[3] = { -1, -1, -1 };
    int channel = 0; // 0 the ground; c >= 1 the biome's sub-mix c (its program reads it as FIELD -c until compile)
};

// A biome's graph as the compiler reads it: its rows, the ground's and its sub-mixes' (the relaxer's, climate_parse.cpp;
// its program and look are the Biome's own).
struct BiomeGraph {
    std::vector<GraphRow> rows;
};

// The octaves a note covers at p_ground (a range trimmed to [BAND_SHIFT, k_max]), coarsest first.
std::vector<int> note_octaves(const NoteNode &p_note, const Ground &p_ground);
// Octave p_k's amplitude of p_note (LOD0 voxels), under the note's max and p_ground's note_cap.
float note_amp(const NoteNode &p_note, int p_k, const Ground &p_ground);

// Compiles p_graphs (p_graphs[b] for r_table.biomes[b]) into r_table.notes and each biome's amps: empty, or the first
// error (a note mixed twice by a biome, two notes on one octave: the cascade's GPU table holds one per level).
ov::str compile(const std::vector<BiomeGraph> &p_graphs, const Ground &p_ground, Table &r_table);

// The template biome named p_name at p_voxel_m metres: the ground every new biome starts from (the biome mixer's) and
// the ground of a world without biomes (SDFCascade). One note range of seed TEMPLATE_SEED from 8 voxels (k = BAND_SHIFT,
// inserted on Λ_0) up to the first octave whose amplitude TEMPLATE_STEEPNESS * wavelength reaches TEMPLATE_MAX_AMP_M
// (k = 19 at 5 cm: 26.2 km, 3.3 km), every octave at that steepness, capped there. Its "graph" value and the biome.
inline constexpr uint32_t TEMPLATE_SEED = 0xEA27u;
inline constexpr double TEMPLATE_STEEPNESS = 0.125;
inline constexpr double TEMPLATE_MAX_AMP_M = 65536.0 * 0.05;
ov::Value template_graph(double p_voxel_m);
// The template's coarsest octave: the first whose amplitude TEMPLATE_STEEPNESS * wavelength reaches TEMPLATE_MAX_AMP_M.
int template_top(double p_voxel_m);
ov::Value template_biome(double p_voxel_m, const ov::str &p_name = "ground");

} // namespace Climate
