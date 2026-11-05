#pragma once

#include "cascade.hpp"
#include "level_program.hpp"

#include "ov/bytes.hpp"
#include "ov/str.hpp"

#include <vector>

// The GPU forms of a LevelProgram (docs/cascade-v2.md, Backends) for the cascade's level shader (cascade_glsl.hpp), both
// built from the op table's GLSL forms (op_def), op for op with its CPU forms:
//   specialised: each level's code straight, a switch on the level, the constants in the ShaderParams block;
//   generic (LevelProgram::generic, the biome mixer's): a fixed interpreter over the program's records in the
//     ShaderParams block, its text a function of the op set and the caps only (MAX_CHANNELS, MAX_REGISTERS), so an edit
//     compiles nothing.
//
// The level shader subdivides the channels of the level above (Ch: CascadeGLSL's Hermite passes, the pick's channel
// copied from the floor parent), flushes them, then calls `Ch level_run(Ch ch, WorldPos p, int m, bool last)`: the
// level's channels loaded into registers, its code run (on a run's last level, the code of every level from m down to 0,
// as LevelProgram::last_code), and the channels it carries stored back in Ch for the level below; on the last level,
// the output (channel 0) and the pick (channel 1's value) instead. Each statement (each generic record) is guarded by
// its node's fold (Node::const_lod, live_lod: fold_lods), as the CPU run is.
//
// Registers: one per node, allocated over the stream of every level's code top to 0 (a node live from its first event
// to its last: computed, read, loaded on a level, stored for the level below), so a level alone and a run's last level
// (a suffix of the stream) both find their values where they left them.
namespace Terrain {

// The generic text's registers (the specialised text has its program's own count).
inline constexpr int MAX_REGISTERS = 96;

struct GpuLayout {
    std::vector<std::vector<int>> chan; // chan[m]: the nodes level m carries to the level below (its out), the pick first
    std::vector<int> reg; // per node, its register (-1: never computed)
    int regs = 1;
    int channels = 2; // Ch's channels: the most a level carries, at least the last level's two (generic: MAX_CHANNELS)
    int pick_node = -1; // the PICK node (-1: one biome, a constant pick)
    // Whether level m's inputs (chan[m + 1]) start with the pick: its channel is copied, not interpolated.
    bool pick_in(int p_m) const;
    int inputs(int p_m) const;
};
// p_prog's layout; r_error: why the generic text can't hold it (empty if it can, or for a specialised program).
GpuLayout gpu_layout(const LevelProgram &p_prog, ov::str &r_error);

// The NOTE nodes in band table order: level by level from the top, each level's looped notes first (the specialised
// text's note loop reads them as one range), then its other notes, then any note no level computes.
std::vector<int> note_order(const LevelProgram &p_prog);
// std430 CascadeBand per NOTE node, in note_order (CascadeGLSL::BAND_BYTES each: factor unused, offset low words, high
// words, info (1, seed, k, stretch)), at least one entry: the level shader's `bands` (binding 5).
ov::bytes band_table(const LevelProgram &p_prog);

// The globals the Hermite and storage functions read, set per invocation by `void level_flags(int m)`: g_in (the
// channels level m receives), g_pick (its first one is the pick), g_out (the channels it stores).
ov::str level_flags_glsl(const LevelProgram &p_prog, const GpuLayout &p_layout, Cascade::ShaderParams &r_params);
// level_run (after Ch, glsl_helpers and the climate's helpers); its values and records go to r_params.
ov::str level_glsl(const LevelProgram &p_prog, const GpuLayout &p_layout, Cascade::ShaderParams &r_params);

} // namespace Terrain
