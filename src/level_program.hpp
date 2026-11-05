#pragma once

#include "base_surface.hpp"
#include "cascade.hpp"
#include "climate.hpp"
#include "exact_noise.hpp"

#include "ov/str.hpp"

#include <cstdint>
#include <vector>

// Cascade v2 (docs/cascade-v2.md): the ground as one graph computed coarse to fine. The compiler (graph_compile.hpp)
// turns a world into a LevelProgram; this file holds the program, its op table and its CPU evaluator.
//
// Every node has a level: a note of octave k sits at level k - BAND_SHIFT (inserted on that lattice), any other node at
// the finest level of its inputs; a constant and `up` have none (exact at every lattice point: computed where they are
// read). A node is computed on its level's lattice from its inputs' values there and Hermite-subdivided below
// (Cascade::axis_rule, x then y then z), so every LOD and origin gets the same bits where they share points. A run of
// LOD L stops at Λ_L: its last level computes the nodes of every level <= L, the notes finer than L reading 0 (what is
// then a constant folded to it: fold_lods).
//
// Values are dual numbers (Float4: gradient per lattice spacing of the level, value), every op's result flushed under
// 2^-64 (Cascade::dflush), each op's CPU form and GLSL form side by side in one table entry (op_table): the same ops in
// the same order on both sides, no fma, no division / sqrt / mix / dot (docs/gotchas.md).
namespace Terrain {

using ExactNoise::Float4;

// The ops. Inputs `in`, constants `k`:
//   NOTE   perlin3 of `band` at the point (amplitude 1, gradient per lattice spacing), read displaced by in[0..2] (LOD0
//          voxels per axis) if it has them; 0 on a level coarser than its own (a run of a coarser LOD: the note absent).
//   CONST  k[0]. UP: the point's height above the base (LOD0 voxels, Climate::up_value).
//   LIN    sum of k[i] in[i], in order (a k of 1 multiplies nothing).
//   MUL    in[0] in[1]. MIN, MAX: the input with the smaller / larger value (in[1] if it wins strictly).
//   SMIN, SMAX: Climate::smooth_min of in[0], in[1] over k[0] voxels (k[1] = 0.5 / k[0]); SMAX = -smin(-a, -b).
//   CURVE  curves[curve] of in[0] (Climate::eval_curve: the curve's own value, not plus its input).
//   SHAPE  k[0] + k[1] a + k[2] sabs(a) + k[3] a^2 (Cascade::smooth_abs; each term where its k isn't 0).
//   AXIS   Climate::axis_factor of in[0], centre k[0], 1 / span k[1]. FILL: Climate::fill. RCP: Climate::blend_rcp.
//   NORM   Climate::blend_weight(in[0], in[1]): a kernel over the total, capped at 1.
//   PICK   the index of the largest of its inputs (the biome weights; ties by hash_cell, Climate::pick_of): a discrete
//          value, carried by copy (the floor parent's), never interpolated.
enum class Op : uint8_t {
    NOTE,
    CONST,
    UP,
    LIN,
    MUL,
    MIN,
    MAX,
    SMIN,
    SMAX,
    CURVE,
    SHAPE,
    AXIS,
    FILL,
    RCP,
    NORM,
    PICK,
    COUNT,
};

// The level of a constant, of `up` and of a node of them only: exact everywhere, computed at each level that reads it.
inline constexpr int NO_LEVEL = 1 << 20;
// Live channels a level carries to the next (docs/cascade-v2.md, Caps) and instructions a level runs.
inline constexpr int MAX_CHANNELS = 48;
inline constexpr int MAX_LEVEL_INSTRS = 256;

struct Node {
    Op op = Op::CONST;
    std::vector<int> in; // input nodes, all before this one
    std::vector<float> k;
    int curve = -1;
    Cascade::Band band; // NOTE: octave, seed, offset, stretch (amp 1)
    int level = NO_LEVEL;
    // The fold per LOD (fold_lods): on every level of a run of LOD >= const_lod the node is the constant `folded` (its
    // notes there absent, or a product by such a 0), and a run's last level of LOD > live_lod doesn't compute it (no
    // node it computes reads it). NO_LEVEL: never constant, always read.
    int const_lod = NO_LEVEL;
    int live_lod = NO_LEVEL;
    float folded = 0.0f;
    // The biome pruning (guard_biomes): the biome whose weight gates the node (-1: none). The node is 0 where that
    // weight is 0: on a level below the weight's, in a run whose weight is 0 on the whole box there (the run's mask);
    // on a run's last level, at a point where the weight is 0.
    int guard = -1;
    // The bits this node's value sets in a run's mask above its last level (0 for most): biome b's if the node is
    // weights[b] and some node is guarded by b (several when biomes share one weight node: the graph hash-conses). The
    // CPU run and both GPU texts mark from it.
    uint32_t mark = 0;
};

// One level's program: the nodes it computes (in order, after the inputs carried from the level above: the level
// above's `out`, subdivided) and the nodes it carries to the level below (liveness: from the level that computes a node
// to the last level that reads it).
struct Level {
    std::vector<int> code;
    std::vector<int> out;
};

struct LevelProgram {
    std::vector<Node> nodes; // topological: a node's inputs come before it
    std::vector<Climate::Curve> curves;
    std::vector<Level> levels; // levels[m], m = 0 .. top
    int top = 0;
    int output = -1; // the ground height (LOD0 voxels): what height_sdf reads at a run's last level
    int pick = -1; // the biome (PICK, or a constant)
    int biomes = 1;
    // The climate's nodes the CPU reads (BiomePick, CloudClimate, ClimateReport): each field's value, each biome's weight.
    std::vector<int> fields, weights;
    bool generic = false; // compiled Mode::GENERIC: the GPU runs it through the interpreter (level_glsl.hpp)
    // What a stretched note stretches along and what `up` measures from (SDFCascade's base).
    BaseSurface base = BaseSurface::flat(0);
    // Bounds (graph_compile.cpp): each node's value within [lo, hi]; step[L], the output's largest change between LOD L
    // and L + 1 at their shared points (the notes inserted on Λ_L dropped), LOD0 voxels: the skirts' depth.
    std::vector<double> lo, hi;
    std::vector<double> step;

    // The inputs level p_m receives: the level above's `out` (none at the top).
    const std::vector<int> &inputs(int p_m) const;
    // The code a run down to p_lod runs on its last level: that of levels min(p_lod, top) .. 0, coarse to fine.
    std::vector<int> last_code(int p_lod) const;
    // The finest level that code computes a node on (folded: fold_lods): below it every node is a constant or unread, so
    // a run's last level stops there (its output and pick, if constants, read as such).
    int last_stop(int p_lod) const;
    int channels(int p_m) const { return int(levels[size_t(p_m)].out.size()); }
};

// A point an op runs at: lattice point p of Λ_m.
struct OpPoint {
    Vector3i64 p;
    int m = 0;
};

// The op table: each op's name, input count (-1: any), CPU form and GLSL form. The GLSL form is an expression of the
// inputs `$0`, `$1`, ... and the node's constants `#0`, `#1`, ... (literals or ShaderParams reads, the emitter's
// choice), the point `@p` (WorldPos) and its level `@m` (int), a NOTE's own level `@level` and band `@band`
// (CascadeBand) and a CURVE's record `@curve` (its index in the params' heap, Climate::curve_at_glsl's layout). Variadic
// ops: LIN's form is one term, of input `$i` and its constant `#i`, the terms summed in order by cl_add (the first one
// alone); PICK's is a call on the array `$all` of its inputs (`#n` of them). The functions it calls are glsl_helpers()'s and
// the climate's shared ones (cl_add, cl_mul, cl_axis, ...: Climate::shared_helpers_glsl, smooth_min_glsl,
// curve_at_glsl) and the cascade's (perlin3, base_at, base_stretched, cascade_warped, cl_smooth_abs).
struct OpDef {
    Op op;
    const char *name;
    int inputs;
    Float4 (*cpu)(const LevelProgram &p_prog, const Node &p_node, const Float4 *p_in, const OpPoint &p_at);
    const char *glsl;
};
const OpDef &op_def(Op p_op);
// The GLSL functions the forms call beyond the shared ones (cl_smooth_abs, cascade_warped, then tp_note, tp_term,
// tp_shape, tp_axis, tp_norm, tp_pick), after the climate's shared helpers, smooth min, curve and the base's stretch.
ov::str glsl_helpers();
// Node p_node's value from its inputs' values p_in at p_at (op_def's CPU form).
Float4 eval_node(const LevelProgram &p_prog, const Node &p_node, const Float4 *p_in, const OpPoint &p_at);

// The constant relaxation (docs/cascade.md, Folding): a run of LOD L reads its notes finer than L as 0, so a node of
// such notes and constants only is a constant there, computed once by its own op (and a product by such a 0 is 0),
// whatever the point: each node's const_lod and folded. The last level then computes only what its non-constant nodes
// read from the output and the pick (live_lod). The same bits as the unfolded program (every op's result flushed: no
// -0, a product by +0 is +0), so the CPU and both GPU texts fold alike. Called by compile after lower.
void fold_lods(LevelProgram &r_prog);

// The biome pruning (docs/cascade.md, Pruning): biome b's ground enters the blend as products by its weight w_b; a
// node read only by those products (and by other such nodes) is b's, and gated on levels finer than w_b's. Where w_b
// is 0 (value and gradient) on every point of a run's box on its level, it is 0 on every finer point too (Hermite of
// zeros), so the products are 0 whatever b's nodes are: they are skipped, 0. The same bits for the output and the pick
// (a product by +0 is +0); the skipped channels are 0 on the CPU and the GPU alike. Called by compile after fold_lods.
void guard_biomes(LevelProgram &r_prog);
// The run's mask's bit of biome b (Values::mask).
inline uint32_t biome_bit(int p_b) {
    return uint32_t(1) << uint32_t(p_b);
}

// The channels of one level of a run: the nodes it holds (slot[node], -1: none), 4 planar channels per slot in
// Cascade::Channel order (VALUE, DX, DY, DZ) over the box's points (x-major).
struct Values {
    int m = 0;
    Cascade::Box box;
    std::vector<int> slot;
    std::vector<std::vector<float>> ch;
    // The run's biome mask (guard_biomes): bit b set once biome b's weight isn't 0 somewhere on its level's box (computed
    // there, a level above the last); read by the levels below.
    uint32_t mask = 0;

    bool has(int p_node) const { return p_node >= 0 && slot[size_t(p_node)] >= 0; }
    Float4 at(int p_node, int64_t p_i) const;
    // Cascade::Channel order (what height_sdf reads).
    void channels(int p_node, int64_t p_i, float r_ch[Cascade::CHANNELS]) const;
};

// Runs p_prog from its top level down to Λ_p_lod over the points of p_box (on Λ_p_lod), as the cascade's CPU box run
// does. r_levels (optional) receives every level, top first. The result's output and pick slots hold the samples.
Values run(const LevelProgram &p_prog, const Cascade::Box &p_box, int p_lod, std::vector<Values> *r_levels = nullptr);

// A point query: run() on the one-point box (the same ops in the same order: the same bits), on a flat per-thread
// workspace (no allocation once warm), computing only the nodes the targets depend on.
struct Sample {
    Float4 ground;
    int pick = 0;
};
Sample point(const LevelProgram &p_prog, const Vector3i64 &p_p, int p_lod);
// The values of nodes p_targets[0 .. p_count) at point p_p of Λ_p_lod (a run down to p_lod: its last level's), a
// target the program doesn't carry to Λ_p_lod (a field, a constant weight) subdivided or computed there all the same.
void point_nodes(const LevelProgram &p_prog, const Vector3i64 &p_p, int p_lod, const int *p_targets, int p_count, Float4 *r_values);

} // namespace Terrain
