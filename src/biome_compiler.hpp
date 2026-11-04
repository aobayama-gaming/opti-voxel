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
// mix rows into its amplitude, shape, multifractal and warp on each note (Amp): the cascade's modulation inserts
// sum_b w_b (the biome's term) per note, one noise read whatever the number of biomes. A note every biome plays plain at
// one amplitude is a constant band (no modulation). docs/climate.md's "Biome graphs" part is this.
namespace Climate {

// A note node: one octave k, its seed, its amplitude (metres) and its stretch along the base's up.
struct NoteNode {
    int k = 0;
    uint32_t seed = 0;
    double amp_m = 0.0;
    int stretch = 0; // log2 of stretch_y (Band::stretch: octave k takes at most k - BAND_SHIFT of it)
};

// One mix row of a biome's graph: a note read through its warp, a shape and a multifractal (each optional), at volume x.
struct GraphRow {
    NoteNode note;
    float x = 1.0f;
    bool has_shape = false;
    Shape shape;
    Multifractal multifractal;
    std::vector<WarpTerm> warps;
    int channel = 0; // 0 the ground; c >= 1 the biome's sub-mix c (its program reads it as FIELD -c until compile)
};

// A biome's graph as the compiler reads it: its rows, the ground's and its sub-mixes' (the relaxer's, climate_parse.cpp;
// its program and look are the Biome's own).
struct BiomeGraph {
    std::vector<GraphRow> rows;
};

// The format's readers for the cascade v2 compiler (graph_compile.cpp), climate_parse.cpp's own. parse_layout: a table's
// fields and biomes (places, looks, contents) as parse() reads and validates them, each biome's graph left as its JSON
// nodes (r_graphs[b], an empty list without a graph); a field's "ground" term is refused. The others read one graph node
// (p_where names it in the errors): its keys (its kind's, graph_kinds), a note's octave, seed, amplitude and stretch, a
// number parameter (its kind's default if absent), a curve's points. Each returns empty or the error.
ov::str parse_layout(const ov::Value &p_data, const Ground &p_ground, Table &r_table, std::vector<ov::Value> &r_graphs);
ov::str parse_node_keys(const ov::Value &p_node, const ov::str &p_where);
ov::str parse_note(const ov::Value &p_node, const Ground &p_ground, const ov::str &p_where, NoteNode &r_note);
ov::str parse_number(const ov::Value &p_node, const char *p_key, const ov::str &p_where, double &r_value);
ov::str parse_curve(const ov::Value &p_node, const Ground &p_ground, const ov::str &p_where, Curve &r_curve);

// Whether p_ground has p_note's octave ([BAND_SHIFT, k_max]: a note the LOD count doesn't have plays nothing, so a
// world reads at any LOD count).
bool note_plays(const NoteNode &p_note, const Ground &p_ground);
// p_note's stretch on octave p_k (Band::stretch: octave k takes at most k - BAND_SHIFT of it).
int note_stretch(const NoteNode &p_note, int p_k);
// p_note's amplitude (LOD0 voxels), under p_ground's note_cap.
float note_amp(const NoteNode &p_note, const Ground &p_ground);

// Compiles p_graphs (p_graphs[b] for r_table.biomes[b]) into r_table.notes and each biome's amps: empty, or the first
// error (a note mixed twice by a biome, two notes on one octave: the cascade's GPU table holds one per level).
ov::str compile(const std::vector<BiomeGraph> &p_graphs, const Ground &p_ground, Table &r_table);

// The template biome named p_name at p_voxel_m metres: the ground every new biome starts from (the biome mixer's) and
// the ground of a world without biomes (SDFCascade). A mix "ground" of one note of seed TEMPLATE_SEED per octave from 8
// voxels (k = BAND_SHIFT, inserted on Λ_0) up to the first whose amplitude TEMPLATE_STEEPNESS * wavelength reaches
// TEMPLATE_MAX_AMP_M (k = 19 at 5 cm: 26.2 km, 3.3 km), every one at that steepness, capped there. With p_ground, the
// octaves stop before their amplitudes (under its note_cap) sum past its amp_cap: a small planet's template fits its
// relief (the program's output bound, Terrain::bound_program). Its "graph" value and the biome.
inline constexpr uint32_t TEMPLATE_SEED = 0xEA27u;
inline constexpr double TEMPLATE_STEEPNESS = 0.125;
inline constexpr double TEMPLATE_MAX_AMP_M = 65536.0 * 0.05;
ov::Value template_graph(double p_voxel_m, const Ground *p_ground = nullptr);
// The template's coarsest octave: the first whose amplitude TEMPLATE_STEEPNESS * wavelength reaches TEMPLATE_MAX_AMP_M,
// or the last under p_ground's amp_cap.
int template_top(double p_voxel_m, const Ground *p_ground = nullptr);
ov::Value template_biome(double p_voxel_m, const ov::str &p_name = "ground", const Ground *p_ground = nullptr);

} // namespace Climate
