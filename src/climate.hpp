#pragma once

#include "cascade.hpp"
#include "exact_noise.hpp"
#include "material_library.hpp"

#include "ov/color.hpp"
#include "ov/str.hpp"
#include "ov/value.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

// Climate (biomes): a climate is data, read at configure: a few scalar fields, each a set of perlin3 bands, and biomes
// as parts of the space of those fields, each with its own graph of notes. The cascade v2 compiler (graph_compile.hpp)
// turns the whole into one LevelProgram: the fields, the weights, each biome's graph and the ground sum_b w_b out_b are
// its nodes (docs/cascade-v2.md, docs/climate.md). This file holds the table the parser reads (climate_parse.cpp), the
// math the weights and the graphs' nodes are made of (climate.cpp: the op table's CPU forms call it, its GLSL twins are
// shared_helpers_glsl, smooth_min_glsl and curve_at_glsl) and the biomes' looks and contents.
//
// Weights: biome b is a place in the space of the fields, a centre and a span on each field it names (its axes). Its
// kernel k_b is the product over its axes of the smoothstep of s = clamp((1 - u^2) * KERNEL_FADE, 0, 1),
// u = (f - centre) / span: on each field whole within 0.71 of its span (u^2 <= 1/2), fading to 0 at the span with a zero
// slope at both ends, each field's span its own (a field more doesn't shrink the others), so a biome costs nothing
// where it is absent. The last biome (the default, no axis) fills what the others leave, k_d = (1 - sum k)^2 for
// sum k < 1, else 0. The weights are the kernels over their sum S = sum k + k_d >= 3/4 (min of x + (1 - x)^2): any
// overlap blends by itself, no border to match and no priority. 1 / S is det_rsqrt(S)^2 and one Newton step (S >= 3/4,
// a normal float; 1 / 1 is exactly 1), its gradient -S' (1/S)^2; a weight's value is capped at 1 (rounding). Dual
// numbers (forward-mode gradient), one fixed op order, the axes sorted by field. Exactness: + - * , compares and
// det_rsqrt only; after every product each component under 2^-64 is flushed to 0 (Cascade::flush_value). Inputs are
// then 0 or >= 2^-64, so a product's denormal part (below 2^-126) either vanishes in a sum with a term >= 2^-64 (under
// half its ulp) or ends below 2^-64 and is flushed: the GPU flushing denormals or not gives the same bits.
namespace Climate {

using ExactNoise::Float4;

inline constexpr int MAX_FIELDS = 24;
inline constexpr int MAX_BIOMES = 16;
// "stretch_y" at most 2^MAX_STRETCH (Band::stretch).
inline constexpr int MAX_STRETCH = 8;
// Seed of the pick's tie-break hash (ExactNoise::hash_cell of the lattice point).
inline constexpr uint32_t PICK_SEED = 0xB10E5u;

// coef * the value of an earlier field (`source`, its index).
struct Term {
    int source = 0;
    float coef = 0.0f;
};

// A shape: phi(n) = c[0] + c[1] n + c[2] |n| + c[3] n^2 (the gallery's: plain n, billow 2|n| - 1, ridged (1 - |n|)^2,
// square n^2). On a note it shapes the noise without its mean (Cascade's NOISE_*_MEAN), so c[0] doesn't count there.
struct Shape {
    float c[4] = { 0.0f, 1.0f, 0.0f, 0.0f };
};
// phi(n), |n| smoothed as a note's shape reads it (Cascade::smooth_abs's value): what the biome mixer plots of a shape.
double shape_phi(const Shape &p_shape, double p_n);

// A field: its perlin3 bands (octave k on Λ_{k - BAND_SHIFT}, coarsest first, at most one per octave) and its terms on
// earlier fields.
struct Field {
    ov::str name;
    uint32_t seed = 0;
    std::vector<Cascade::Band> bands;
    std::vector<Term> terms;
};

// Where a biome sits on one field: u = (x - centre) * inv_span, its kernel's coordinate (the format's [centre, span],
// or min / max: centre (min + max) / 2, span (max - min) / 2).
struct Axis {
    int field = 0;
    float centre = 0.0f;
    float inv_span = 1.0f;
};

// A piecewise cubic through control points (libnoise's Curve: each segment the cubic through its two points and their
// neighbours, constant past the ends): segment i on [x_i, x_{i+1}] is ((p a + q) a + r) a + s, a = (x - x_i) inv_i.
struct Curve {
    std::vector<float> x;
    std::vector<float> inv;
    std::vector<float> p, q, r, s;
    float y_last = 0.0f;
    bool empty() const { return x.empty(); } // no curve
};
// The curve through points (p_x strictly increasing, at least 2) and their p_y.
Curve make_curve(const std::vector<double> &p_x, const std::vector<double> &p_y);
// The curve at p_x (its value and gradient).
Float4 eval_curve(const Curve &p_c, const Float4 &p_x);
// A height curve node's curve through its points (h metres increasing, at least 2; R(h) metres): C = R - h over h, both
// in voxels of p_voxel (what the node adds to its input). Empty, or why not.
ov::str height_curve(const std::vector<double> &p_h_m, const std::vector<double> &p_r_m, double p_voxel, Curve &r_c);
// max |curve'| and max |curve| (its cubic segments overshoot their points): the program's bounds.
float curve_slope_bound(const Curve &p_curve);
float curve_bound(const Curve &p_curve);
// eval_curve in GLSL over a curve's record in the params' heap (n, y_last, x[n], then inv, p, q, r, s per segment):
// `vec4 cl_curve_at(int cb, vec4 x)`, after the shared helpers.
ov::str curve_at_glsl();
// Quilez's polynomial smooth min of p_a and p_b over p_k voxels (p_inv = 0.5 / p_k), flushed like the duals; its GLSL
// `vec4 cl_smin(vec4 a, vec4 b, float k, float inv)`, after the shared helpers.
Float4 smooth_min(const Float4 &p_a, const Float4 &p_b, float p_k, float p_inv);
ov::str smooth_min_glsl();
// The steepest offset a curve may have (the compiler refuses past it): R' within [-3, 5], so the ground's gradient grows
// at most 5x where the biome is whole. libnoise's cubic overshoots its points' secants.
constexpr double MAX_CURVE_SLOPE = 4.0;
// A kernel's fade: 1 / (1 - 1/2), over the outer 0.29 of a span (u^2 from 1/2 to 1; exact).
constexpr float KERNEL_FADE = 2.0f;
// A kernel moves by at most KERNEL_SLOPE times the sum of its |du| (a product of factors in [0, 1]: the sum of theirs,
// each |d(u^2)| <= 2 |du| times |dk/d(u^2)| <= 1.5 KERNEL_FADE): the weights' Lipschitz bound (graph_bounds.cpp).
constexpr double KERNEL_SLOPE = 6.0;

// A biome's ground look (its material, biome_material): albedo on gentle slopes and on steep ones (linear colours),
// roughness. The default is the page shader's procedural ground.
struct Surface {
    ov::Color flat = Material::GROUND_FLAT;
    ov::Color steep = Material::GROUND_STEEP;
    float roughness = 1.0f;
};

// The clouds over a biome ("clouds"; CloudClimate): the sky's cover there and the cloud type (Schneider's: stratus 0,
// stratocumulus 0.5, cumulus 1), both in [0, 1]. Unset: the sky's cover and the weather noise's type.
struct BiomeClouds {
    bool set = false;
    float cover = 0.0f;
    float type = 0.0f;
};

// A species a biome grows (its "instances", or the world's for a biome that lists none): a name of the world's species,
// one site per cell of cell_m metres (a kind's entries of one cell are one source: SDFTable), and an integer share among
// the biome's entries of that source (BiomePick).
struct InstanceShare {
    ov::str species;
    double cell_m = 0.0;
    uint32_t share = 1;
};

// A feature a biome has (its "features", or the world's for a biome that lists none): a kind's name (biome_features.hpp:
// lava_lake, floating_island; SDFTable checks it), its cells (cell_m, rounded to a power of two voxels), the chance a
// cell holds one and its radius range in metres. A kind's entries of one cell size share their cells (SDFTable).
// The largest cell an instance or a feature entry may have (metres): its cell in LOD0 voxels stays far inside int64 and
// a shift of one (Biome::cell_shift) at any voxel size the engine takes.
inline constexpr double MAX_CELL_M = 1e9;

struct FeatureShare {
    ov::str kind;
    double cell_m = 0.0;
    double chance = 1.0;
    double radius_lo_m = 0.0;
    double radius_hi_m = 0.0;
};

// A biome: where it is (its axes in the fields' space), what grows on it, what stands on it and its look. Its graph is
// the compiler's (graph_compile.cpp), not kept here.
struct Biome {
    ov::str name;
    std::vector<Axis> at; // sorted by field; empty for the last (default) biome only
    // What grows on it and what features stand on it ("instances", "features"). Not listed: the world's lists
    // (SDFTable::set_instances, set_features).
    std::vector<InstanceShare> instances;
    bool instances_listed = false;
    std::vector<FeatureShare> features;
    bool features_listed = false;
    Surface surface;
    BiomeClouds clouds;
    uint32_t debug_color = 0; // 0xRRGGBB, reports and debug views
};

struct Table {
    std::vector<Field> fields;
    std::vector<Biome> biomes;

    // The field named p_name, -1 if none.
    int field_index(const ov::str &p_name) const;
};

// What a table is compiled for (SDFCascade::climate_ground): the voxel, the LOD count's octaves and the base's caps.
struct Ground {
    double voxel_size = 0.0; // metres: converts the format's metres to voxels and octaves
    // The coarsest octave a note or a field may have (the LOD count's: lod_count + 4, the cascade's top).
    int k_max = -1;
    // The program's output at most this (LOD0 voxels): a planet's relief (SDFCascade::PLANET_RELIEF of its radius), the
    // exact range's top on the flat world.
    float amp_cap = 0.0f;
    // A note's own amplitude at most this (LOD0 voxels): a planet's relief over SDFCascade::PLANET_HEADROOM, so a biome's
    // volume up to it still fits under amp_cap; infinite on the flat world.
    double note_cap = INFINITY;
};

// A note node as the format reads it: one octave k, its seed, its amplitude (metres) and its stretch along the base's up.
struct NoteNode {
    int k = 0;
    uint32_t seed = 0;
    double amp_m = 0.0;
    int stretch = 0; // log2 of stretch_y (Band::stretch: octave k takes at most k - BAND_SHIFT of it)
};

// The format's readers (climate_parse.cpp; the format is described above graph_kinds there), each returning empty or the
// first error, p_where naming the node in the errors:
//   parse_layout: a table's fields and biomes (places, looks, contents), validated (counts, names, bands, terms on
//     earlier fields, axes), each biome's graph left as its JSON nodes (r_graphs[b], an empty list without a graph) for
//     the compiler;
//   parse_node_keys: a graph node's keys (its kind's, graph_kinds); parse_note: a note's octave, seed, amplitude and
//     stretch; parse_number: a number parameter (its kind's default if absent); parse_curve: a curve's points;
//   parse_shape: a shape node ({ "shape", "scale", "offset" } or { "coeffs" }), p_constant whether its constant term
//     counts (on a note it doesn't);
//   parse_instances, parse_features: a list of instances ([{ "species", "cell_m", "share" }]) or features ([{ "kind",
//     "cell_m", "chance", "radius_m" }]), a biome's or the world's (r_out untouched on an error).
ov::str parse_layout(const ov::Value &p_data, const Ground &p_ground, Table &r_table, std::vector<ov::Value> &r_graphs);
ov::str parse_node_keys(const ov::Value &p_node, const ov::str &p_where);
ov::str parse_note(const ov::Value &p_node, const Ground &p_ground, const ov::str &p_where, NoteNode &r_note);
ov::str parse_number(const ov::Value &p_node, const char *p_key, const ov::str &p_where, double &r_value);
ov::str parse_curve(const ov::Value &p_node, const Ground &p_ground, const ov::str &p_where, Curve &r_curve);
ov::str parse_shape(const ov::Value &p_node, bool p_constant, Shape &r_shape);
ov::str parse_instances(const ov::Value &p_list, const ov::str &p_where, std::vector<InstanceShare> &r_out);
ov::str parse_features(const ov::Value &p_list, const ov::str &p_where, std::vector<FeatureShare> &r_out);
// FNV-1a of a name: a field's seed without one, a biome's debug colour without one (& 0xFFFFFF).
uint32_t name_hash(const ov::str &p_name);

// A biome graph's node kind as the parser reads it (graph_kinds, in the format's order): the keys naming its inputs
// (another node's id; a mix's rows name theirs, "rows") and its parameters. A parameter's type: "number", "seed" (a whole
// number), "pow2" (a power of two), "field" (a field's name), "shape" (one of options), "coeffs" (4
// numbers), "points" ([h, R] pairs), "rows" ([{ "in", "x" }]); required, else its default (NAN: none, the node's rule
// decides). Every node also has "id", "kind" and "pos" (the mixer's, ignored). The biome mixer builds its nodes from it.
struct GraphParam {
    const char *name;
    const char *type;
    bool required;
    double fallback;
    const char *options;
};
struct GraphKind {
    const char *kind;
    std::vector<const char *> inputs;
    std::vector<GraphParam> params;
};
const std::vector<GraphKind> &graph_kinds();
// The kind named p_kind, null if there is none.
const GraphKind *graph_kind(const ov::str &p_kind);

// The weights' steps (the op table's AXIS, FILL, RCP and NORM, level_program.cpp): one axis's factor, the smoothstep of
// clamp((1 - u^2) KERNEL_FADE, 0, 1), u = (x - centre) / span; the default's (1 - x)^2 for x < 1, else 0 (x the kernels'
// sum); 1 / s of the total s (det_rsqrt(s)^2 and one Newton step); a kernel over the total, capped at 1.
Float4 axis_factor(const Axis &p_axis, const Float4 &p_x);
Float4 fill(const Float4 &p_x);
Float4 blend_rcp(const Float4 &p_s);
Float4 blend_weight(const Float4 &p_k, const Float4 &p_rcp);
// The index of the largest of the p_count weights p_w (at most MAX_BIOMES) at lattice point p_p of Λ_m; among equal
// weights (same bits), the (hash % count)-th in list order, hash = ExactNoise::hash_cell(the point's LOD0 voxel,
// PICK_SEED): the same from every LOD that has the point (the op table's PICK).
int pick_of(int p_count, const Float4 *p_w, const Vector3i64 &p_p, int p_m);
// The op table's UP at a sample of LOD p_lod from the base's register there (BaseSurface::at: in voxels of the LOD), in
// LOD0 voxels: exact (a power of two).
Float4 up_value(const Float4 &p_base, int p_lod);
// The dual-number helpers in GLSL (cl_flush, cl_mul, cl_sub, cl_add, cl_scale: Cascade's dflush, dmul, dsub, dadd,
// dscale; cl_fill, cl_axis, cl_rcp: fill, axis_factor, blend_rcp).
ov::str shared_helpers_glsl();

// 0xRRGGBB (sRGB) as a linear colour, alpha 1.
ov::Color srgb_to_linear(uint32_t p_rgb);
// Biome p_biome's ground material ("biome:<name>" in the material library, SDFCascade's base palette): the procedural
// ground (PATTERN_GROUND) in its surface's colours, its debug colour (--view=12).
Material::Params biome_material(const Biome &p_biome);

} // namespace Climate
