#pragma once

#include "climate.hpp"
#include "level_program.hpp"

#include "ov/str.hpp"
#include "ov/value.hpp"

// The cascade v2 compiler (docs/cascade-v2.md, Programs): a world's climate table (climate_parse.cpp's format) into one
// LevelProgram. Passes:
//   parse:    the fields and biomes as Climate::parse_layout reads them, each biome's graph read node by node (the
//             format's kinds, graph_kinds), the sugar expanded on the way: a note is amp_m times its unit noise (a NOTE
//             in a LIN), a shape on a note shapes its noise without its mean (so a dropped note makes no step), a curve
//             node is its input plus the curve's offset, a multifractal is multiply(in, ramp(parent)) with the ramp a
//             two-point clamped curve, a warp reads its input three times (seeds + 0, 1, 2) or its x, y, z ports, times
//             amount_m. A field is the LIN of its bands and terms; a biome's weight its kernels' (ClimateTest::weights, op
//             for op); the ground sum_b w_b out_b; the pick the largest weight. The acyclic check and the gate (a warp's
//             input strictly coarser than the note) refuse with the node's name.
//   identity: nodes are hash-consed as they are made: a note is one node whatever reads it (octave, seed, stretch,
//             warp), and so is any node.
//   levels:   a node's level as it is made (level_program.hpp).
//   fold:     constants (a node of constants is a constant, computed by its own op); dead nodes; a LIN read once by
//             another LIN is merged into it; a LIN whose inputs sit on several levels becomes a chain of partial sums,
//             one channel crossing each level (a plain fBm is one channel). Specialised only (value-dependent): zero
//             terms and products by 0 or 1 dropped, and blend factoring (a term every biome plays alike, same node
//             and coefficient, comes out of sum_b w_b).
//   lower:    each node into its level's code, liveness into each level's out, the caps (MAX_CHANNELS,
//             MAX_LEVEL_INSTRS).
//   bounds:   each node's value interval and each LOD's step (LevelProgram::lo, hi, step), and the validation: the
//             output within amp_cap, a curve's slope within MAX_CURVE_SLOPE, a note's amplitude in the noise's exact
//             range, a warp within WARP_LIMIT.
//   per LOD:  the constant relaxation (fold_lods): what each node is on a run whose finer notes are absent.
namespace Terrain {

enum class Mode {
    SPECIALISED, // the values folded: the program the game runs
    GENERIC, // the graph's structure kept, so an edit of a value never changes the program's shape (the biome mixer)
};

// Compiles p_data (a world's climate table: fields and biomes) for p_ground into r_table (its fields and biomes:
// places, looks, contents; no notes, amps or programs) and r_program (its base the flat datum 0: the caller sets its
// own). Empty, or the first error (r_program then empty).
ov::str compile(const ov::Value &p_data, const Climate::Ground &p_ground, Mode p_mode, Climate::Table &r_table, LevelProgram &r_program);

// The program's bounds and the LOD steps of its output (LevelProgram::lo, hi, step), then the validation against
// p_ground: empty, or why the program is refused.
ov::str bound_program(const Climate::Ground &p_ground, LevelProgram &r_program);

} // namespace Terrain
