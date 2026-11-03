#pragma once

#include "cascade.hpp"
#include "exact_noise.hpp"
#include "material_library.hpp"

#include "ov/color.hpp"
#include "ov/str.hpp"
#include "ov/value.hpp"

#include <cstdint>
#include <vector>

// Climate (biomes): a climate is data, read at configure: a few scalar fields, each a set of perlin3 bands carried
// down the Hermite cascade with the ground (combined_spec: the cascade's field groups), and biomes as parts of the
// space of those fields. A biome plays the ground's own octaves (its notes): it scales, shapes, weights (multifractal)
// and warps each one it names (Cascade::Modulation, climate_modulator.cpp, on the CPU and in generated GLSL, bit for
// bit), may remap the ground's height at each sample through its curve (climate_shape.cpp, ground_value), and its pick
// at each ground sample chooses its material (biome_material, SDFCascade::ground_material). docs/climate.md's Model
// part is this as math.
//
// Field f at lattice point p of Λ_m (fields_at): its channels in V_m(p) of the combined spec (Λ_m holds the octaves
// k >= m + 3), then its linear terms in list order, coef * (the ground's or an earlier field's channels at the same
// point). Each term is itself a pure function of (p, m), so a field is too: every LOD reads the same bits where it
// shares a point. Channels as Float4 (x, y, z: gradient in units of the lattice spacing of Λ_m, w: value), like the
// cascade's.
//
// Weights (weights): biome b is a place in the space of the fields, a centre and a span on each field it names (its
// axes). Its kernel k_b is the product over its axes of the smoothstep of s = clamp((1 - u^2) * KERNEL_FADE, 0, 1),
// u = (f - centre) / span: on each field whole within 0.71 of its span (u^2 <= 1/2), fading to 0 at the span with a zero
// slope at both ends, each field's span its own (a field more doesn't shrink the others), so a biome costs nothing
// where it is absent. The last biome (the default, no axis) fills what the others
// leave, k_d = (1 - sum k)^2 for sum k < 1, else 0. The weights are
// the kernels over their sum S = sum k + k_d >= 3/4 (min of x + (1 - x)^2): any overlap blends by itself, no border to
// match and no priority. 1 / S is det_rsqrt(S)^2 and one Newton step (S >= 3/4, a normal float; 1 / 1 is exactly
// 1), its gradient -S' (1/S)^2; a weight's value is capped at 1 (rounding). Dual numbers
// (forward-mode gradient), one fixed op order: the GPU mirror (Modulator::glsl) walks the same order, the axes sorted
// by field. Exactness: + - * , compares and det_rsqrt only; after every product each component under
// 2^-64 is flushed to 0 (Cascade::flush_value). Inputs are then 0 or >= 2^-64, so a product's denormal part (below
// 2^-126) either vanishes in a sum with a term >= 2^-64 (under half its ulp) or ends below 2^-64 and is flushed: the
// GPU flushing denormals or not gives the same bits.
namespace Climate {

using ExactNoise::Float4;

// Cascade channels (VALUE, DX, DY, DZ) as a Float4 (gradient, value).
inline Float4 to_float4(const float p_ch[Cascade::CHANNELS]) {
    return Float4{ p_ch[Cascade::DX], p_ch[Cascade::DY], p_ch[Cascade::DZ], p_ch[Cascade::VALUE] };
}

inline constexpr int MAX_FIELDS = Cascade::MAX_FIELD_GROUPS;
inline constexpr int MAX_BIOMES = 16;
// "stretch_y" at most 2^MAX_STRETCH (Band::stretch).
inline constexpr int MAX_STRETCH = 8;
// A term's or an amplitude's source that is not a field: the ground cascade.
inline constexpr int GROUND = -1;
// Seed of the pick's tie-break hash (ExactNoise::hash_cell of the lattice point).
inline constexpr uint32_t PICK_SEED = 0xB10E5u;

// coef * (the channels of `source`): GROUND or an earlier field.
struct Term {
    int source = GROUND;
    float coef = 0.0f;
};

// A band's shape: phi(n) = c[0] + c[1] n + c[2] |n| + c[3] n^2 of its noise n (the gallery's: plain n, billow 2|n| - 1,
// ridged (1 - |n|)^2, square n^2). A band inserts its amplitude times phi without its mean (Cascade's NOISE_*_MEAN):
// c[0] only counts in the multifractal state (Amp::gain).
struct Shape {
    float c[4] = { 0.0f, 1.0f, 0.0f, 0.0f };
};
// max |phi(n) - its mean| over n in [-1, 1] (perlin3's range): 1 for the plain shape. Skirts follow it (max_amp).
float shape_bound(const Shape &p_shape);
// phi(n), |n| smoothed as the bands insert it (Cascade::smooth_abs's value): what the biome mixer plots of a shape node.
double shape_phi(const Shape &p_shape, double p_n);

// A field: its plain perlin3 bands (a cascade group, inserted like the ground's, octave k on
// Λ_{k - BAND_SHIFT}) and terms.
struct Field {
    ov::str name;
    uint32_t seed = 0;
    std::vector<Cascade::Band> bands; // coarsest first, at most one per octave
    std::vector<Term> terms;
};

// Where a biome sits on one field: u = (x - centre) * inv_span, its kernel's coordinate (the format's [centre, span],
// or min / max: centre (min + max) / 2, span (max - min) / 2).
struct Axis {
    int field = 0;
    float centre = 0.0f;
    float inv_span = 1.0f;
};

// How a biome plays the ground's note of octave k: its amplitude (LOD0 voxels: its graph's mix row volume times the
// note's amplitude, biome_compiler.cpp), the modulated cascade's (combined_spec), and how the biome shapes it. gain > 0
// weights it by the multifractal state, clamp(gain (1 - c), 0, 1), c the
// complement of the previous multifractal octave's weighted signal (Musgrave: ridged gain 2, hybrid 1); warp != 0 reads
// its noise displaced by warp LOD0 voxels times the values of the fields warp_fields (libnoise's turbulence), the
// displacement's gradient applied by the chain rule.
// Left out: weights or warps read from the ground's slope (Quilez's and de Carpentier's slope damping, the swiss
// turbulence's warp). Their own gradient needs the ground's second derivatives, which the cascade doesn't carry, and the
// Hermite subdivision needs a band's gradient to be its value's: without that term the finer levels' values streak along
// the lattice axes (measured, the climate report's gradient error: 6% and visible streaks, 2.5% for the plain ground).
struct Amp {
    // The note's identity (its band's octave, seed and stretch: several notes may share an octave).
    int k = 0;
    uint32_t seed = 0;
    int stretch = 0;
    float amp = 0.0f;
    bool has_shape = false; // else plain
    Shape shape;
    float gain = 0.0f;
    float warp = 0.0f;
    int warp_fields[3] = { -1, -1, -1 }; // x, y, z (with a warp)
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
// The curve at p_x (its value and gradient, the samples' bits).
Float4 eval_curve(const Curve &p_c, const Float4 &p_x);
// A height curve node's curve through its points (h metres increasing, at least 2; R(h) metres): C = R - h over h, both
// in voxels of p_voxel (what a sample adds to the height). Empty, or why not.
ov::str height_curve(const std::vector<double> &p_h_m, const std::vector<double> &p_r_m, double p_voxel, Curve &r_c);
// max |curve'| (skirts).
float curve_slope_bound(const Curve &p_curve);
// The steepest offset a biome curve may have (validate): R' within [-3, 5], so the ground's gradient grows at most 5x
// where the biome is whole. libnoise's cubic overshoots its points' secants: shaped_test's hills, secants under 1,
// reach 3.
constexpr double MAX_CURVE_SLOPE = 4.0;
// A kernel's fade: 1 / (1 - 1/2), over the outer 0.29 of a span (u^2 from 1/2 to 1; exact).
constexpr float KERNEL_FADE = 2.0f;
// ground_step's weight bounds: a kernel moves by at most KERNEL_SLOPE times the sum of its |du| (a product of factors
// in [0, 1]: the sum of theirs, each |d(u^2)| <= 2 |du| times |dk/d(u^2)| <= 1.5 KERNEL_FADE), a normalised weight by at most BLEND_SLOPE times the kernels' moves summed (k / S with
// S >= 3/4: (4/3) (|dk| + |dS|), |dS| <= 3 sum, the default's |dk_d| <= 2 sum: 20/3).
constexpr double KERNEL_SLOPE = 6.0;
constexpr double BLEND_SLOPE = 6.7;
// max |curve| (its cubic segments overshoot their points; validate's bound).
float curve_bound(const Curve &p_curve);

// A biome's sample program (the biome compiler's, biome_compiler.cpp): what its graph does past its notes' mix, at each
// sample, on the blended ground H (the cascade's ground channels: every biome's notes, sum_b w_b of their terms) and the
// fields. Registers in order, dual numbers (gradient, value) with the climate's flushed ops (Cascade::dadd, dmul, ...;
// GLSL cl_*), op for op on the GPU (ground_glsl). Its term at a sample is D = f(H) - H, the ground
// h = H + sum_b w_b D_b (ground_value); a curve of H alone is D = C(H) itself (eval_curve: the curves' bits).
//   GROUND: H. CONST: value (LOD0 voxels). FIELD: field `index`'s value (carried to the samples, SampleCarry::field).
//   ADD, SUB, MUL: a op b. MIN, MAX: the whole register with the smaller / larger value (b < a ? b : a).
//   SMIN, SMAX: Quilez's polynomial smooth min / max of a and b over `value` voxels (k > 0).
//   CURVE: a + curves[index](a), the curve's R(a).
enum class SampleOp : uint8_t {
    GROUND,
    CONST,
    FIELD,
    ADD,
    SUB,
    MUL,
    MIN,
    MAX,
    SMIN,
    SMAX,
    CURVE,
};
struct SampleInstr {
    SampleOp op = SampleOp::GROUND;
    int a = -1;
    int b = -1;
    float value = 0.0f;
    int index = 0;
};
// A sample program's registers at most (validate: run_program's stack).
inline constexpr int MAX_SAMPLE_INSTRS = 64;
struct SampleProgram {
    std::vector<SampleInstr> code; // empty: no term (D = 0); the last register is f
    std::vector<Curve> curves;
    // Bounds of its term (validate, skirts; biome_compiler.cpp): max |D| (LOD0 voxels), max |dD/dH|, max |dD/df| per field.
    double bound = 0.0;
    double slope = 0.0;
    double field_slope[MAX_FIELDS] = {};
    bool empty() const { return code.empty(); }
    // A curve of H alone: D is eval_curve(curves[0], H) itself.
    bool curve_of_ground() const { return code.size() == 2 && code[0].op == SampleOp::GROUND && code[1].op == SampleOp::CURVE && code[1].a == 0; }
    bool reads_field(int p_field) const;
};

// A biome's ground look (its material, biome_material): albedo on gentle slopes and on steep ones (linear colours),
// roughness. The default is the page shader's procedural ground.
struct Surface {
    ov::Color flat = Material::GROUND_FLAT;
    ov::Color steep = Material::GROUND_STEEP;
    float roughness = 1.0f;
};

// A biome: where it is (its axes in the fields' space), how it plays the ground's notes (amps: what its graph mixes,
// biome_compiler.cpp), what it does past them at each sample (its program: curves through the format's [h, R(h)] as
// C = R - h, LOD0 voxels, constant past the ends; min / max, products, fields) and its look.
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

struct Biome {
    ov::str name;
    std::vector<Axis> at; // sorted by field; empty for the last (default) biome only
    std::vector<Amp> amps;
    SampleProgram program; // empty: none
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
    // The ground: every note some biome's graph mixes, once (biome_compiler.cpp), coarsest first. A note every biome
    // plays plain at one amplitude is a constant band at it, which no biome lists; the others are modulated, every biome
    // listing them (amplitude 0 where its graph doesn't mix it).
    std::vector<Cascade::Band> notes;

    // The field named p_name, -1 if none.
    int field_index(const ov::str &p_name) const;
    // Whether a biome weights a band by the multifractal state: the combined spec carries one more field group for it.
    bool has_state() const;
    // Whether a biome has a sample program: the ground at a sample is then more than the cascade's (ground_value).
    bool has_programs() const;
    // Whether a ground note is stretched along the base's up.
    bool has_stretch() const;
};

// What a table is compiled for (SDFCascade::climate_ground): the voxel, the LOD count's octaves and the base's caps.
struct Ground {
    double voxel_size = 0.0; // metres: converts the format's metres to voxels and octaves
    // The coarsest octave a note or a field may have (the LOD count's: lod_count + 4, the cascade's top).
    int k_max = -1;
    // A biome's shaped amplitude on a note and its curve's offset at most this (LOD0 voxels): a planet's relief
    // (SDFCascade::PLANET_RELIEF of its radius), the exact range's top on the flat world.
    float amp_cap = 0.0f;
    // A note's own amplitude at most this (LOD0 voxels): a planet's relief over SDFCascade::PLANET_HEADROOM, so a biome's
    // volume up to it still fits under amp_cap; infinite on the flat world.
    double note_cap = INFINITY;
};

// Empty if p_table is well formed, else the first error: field and biome counts, unique names, bands (k in
// [BAND_SHIFT, k_max], no duplicate, amplitude in ExactNoise's exact range), terms (GROUND or an earlier field,
// coefficient within [2^-40, 2^40]), axes (an existing field once, centre 0 or within [2^-40, 2^40], 1 / span within
// it), the last biome alone without axes, amplitudes (an existing note, >= 0, each shaped term in the exact range or 0,
// within the cap), curves (their offset under the cap, their slope within MAX_CURVE_SLOPE, their points within the
// heights the ground reaches). No per-LOD gradient check (ExactNoise::gradient_range_error): the cascade flushes what
// falls under 2^-64.
ov::str validate(const Table &p_table, const Ground &p_ground);

// The bands a term's source names: the ground's notes (GROUND), a field's, or null.
const std::vector<Cascade::Band> *bands_of(const Table &p_table, int p_source);

// The table of a JSON object (units in metres; the format is described above parse() in climate_parse.cpp),
// validated; on error r_table is left empty and the error returned.
ov::str parse(const ov::Value &p_data, const Ground &p_ground, Table &r_table);
// A list of instances ([{ "species": name, "cell_m": 5, "share": 1 }]: a biome's, the world's) in r_out: empty, or the
// first error, p_where naming the list (r_out untouched).
ov::str parse_instances(const ov::Value &p_list, const ov::str &p_where, std::vector<InstanceShare> &r_out);
// A list of features ([{ "kind": "lava_lake", "cell_m": 640, "chance": 0.6, "radius_m": [18, 45] }]: a biome's, the
// world's) in r_out, as parse_instances.
ov::str parse_features(const ov::Value &p_list, const ov::str &p_where, std::vector<FeatureShare> &r_out);
// A shape node ({ "shape", "scale", "offset" } or { "coeffs" }, the format's) as r_shape, p_multifractal whether a
// multifractal reads it (its constant term counts): empty, or why not.
ov::str parse_shape(const ov::Value &p_node, bool p_multifractal, Shape &r_shape);
// FNV-1a of a name: a field's seed without one, a biome's debug colour without one (& 0xFFFFFF).
uint32_t name_hash(const ov::str &p_name);

// A biome graph's node kind as the parser reads it (graph_kinds, in the format's order): the keys naming its inputs
// (another node's id; a mix's rows name theirs, "rows") and its parameters. A parameter's type: "number", "seed" (a whole
// number), "pow2" (a power of two), "field" (a field's name), "fields" (3 of them), "shape" (one of options), "coeffs" (4
// numbers), "points" ([h, R] pairs), "rows" ([{ "in", "x" }]); required, else its default (NAN: none, the node's rule
// decides: a note's wavelength_m or from_m and to_m, its steepness or amp_m). Every node also has "id", "kind" and "pos"
// (the mixer's, ignored). The biome mixer builds its nodes from it.
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

// The cascade p_ground (its base and insert shift) with the table's notes as its bands, its fields as its field groups
// and, if a biome lists a note, a Modulation: every listed note gets A = sum_b w_b amp_b at each point, from the weights
// of the climate before that level's insertions.
Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground);

// The amplitude biome p_biome gives the note p_band: its entry's, or the note's own (a constant note) without one. The
// modulation's per-biome amplitudes.
float biome_amp(const Biome &p_biome, const Cascade::Band &p_band);
// The largest biome_amp of any biome, times its shape's bound: bounds that follow amplitudes (skirts) take it.
float max_amp(const Table &p_table, const Cascade::Band &p_band);
// The ground's largest |H| (LOD0 voxels): each note's max_amp summed (validate's curves, the compiler's program bounds).
double ground_reach(const Table &p_table);
// A note's identity (several notes may share an octave): its octave, its band's seed and its stretch.
struct NoteId {
    int k = 0;
    uint32_t seed = 0;
    int stretch = 0;
    bool operator==(const NoteId &p_o) const { return k == p_o.k && seed == p_o.seed && stretch == p_o.stretch; }
};
inline NoteId note_id(const Cascade::Band &p_band) {
    return NoteId{ p_band.k, p_band.seed, p_band.stretch };
}
inline NoteId note_id(const Amp &p_amp) {
    return NoteId{ p_amp.k, p_amp.seed, p_amp.stretch };
}
// Biome p_biome's entry of the note p_band (its octave, seed and stretch), or null.
const Amp *find_amp(const Biome &p_biome, const Cascade::Band &p_band);
// Whether p_amp is the entry of the note p_band.
bool plays(const Amp &p_amp, const Cascade::Band &p_band);
// The shape biome p_biome gives the note p_band: its entry's, else plain. What the modulation inserts and max_amp
// bounds.
Shape band_shape(const Biome &p_biome, const Cascade::Band &p_band);
// The skirt's band step at LOD p_lod (LOD0 voxels) with the biomes' curves, from the cascade's p_ground_amp: the ground
// h + sum_b w_b C_b(h) moves by (1 + max |C_b'|) times h's step, plus each weight's change times |C_b|'s bound: a
// weight moves by at most BLEND_SLOPE times the kernels' sum of KERNEL_SLOPE |1 / span| times their fields' steps.
double ground_step(const Table &p_table, int p_lod, double p_ground_amp);

// The field values of the channels p_slots (a point's Cascade slots, combined_spec's layout): each field's own
// channels, then its terms in list order; r_fields past the table's fields are left as they are (Cascade::point_slots, or a level's channels before its insertions).
void apply_terms(const Table &p_table, const float p_slots[Cascade::SLOTS], Float4 r_fields[MAX_FIELDS]);

// Channels of every field at lattice point p_p of Λ_m (p_p = voxel >> m), r_fields[i] for field i: V_m of
// p_combined (combined_spec) at the point, the fields carried at every level (the derive too), then the terms.
void fields_at(const Table &p_table, const Cascade::Spec &p_combined, const Vector3i64 &p_p, int p_m, Float4 r_fields[MAX_FIELDS]);

// Biome weights (value and gradient) of the field channels p_fields, r_w[b] for biome b.
void weights(const Table &p_table, const Float4 p_fields[MAX_FIELDS], Float4 r_w[MAX_BIOMES]);

// The climate at a point, evaluated once per point of each level above a run's last (Modulator: a band's amplitude from
// the channels before the level's insertions; the level above the last derives the last level's amplitudes and its
// samples' terms from the same climate) and by the report: the field values of a point's slots
// p_slots (apply_terms), then the biome weights (weights); GLSL twin: glsl_weights. What reaches a reader is one rule,
// the LOD rule: octave k of a modulated group, inserted on Λ_m, reads H of the climate of Λ_{m+1} after its insertions
// (the climate low-passed at k, Cascade::Modulation); a run's samples read H of the climate the level above them reads
// for its own bands.
struct State {
    Float4 fields[MAX_FIELDS];
    Float4 w[MAX_BIOMES];
};
void state(const Table &p_table, const float p_slots[Cascade::SLOTS], State &r_state);

// The carry groups of a table's sample terms, the first of Modulation's carry (the bands' amplitudes follow): derived on
// a run's level above its last (Cascade::Modulation::derive) and read by its samples. weight[b]: biome b's weight if it
// has a sample program; pick: the biome (its index in the value, copied from the floor parent, not interpolated);
// field[f]: field f's value if a program reads it. -1: none.
struct SampleCarry {
    int weight[MAX_BIOMES];
    int pick = -1;
    int field[MAX_FIELDS];
    int count = 0;
};
SampleCarry sample_carry(const Table &p_table);
// Carry group p_c of `Ch <p_ch>` in p_spec's GPU layout (Spec::gpu_carry) as a (gradient, value) vec4: to_float4's twin.
ov::str carried_glsl(const Cascade::Spec &p_spec, const ov::str &p_ch, int p_c);

// The ground at a sample: the cascade's ground channels H of p_slots (a point's slots), then H + sum_b w_b D_b(H) over
// the biomes with a sample program, w_b carried (SampleCarry). Without programs, the cascade's channels, bit for bit.
Float4 ground_value(const Table &p_table, const float p_slots[Cascade::SLOTS]);
// The same with the table's sample_carry given (per sample: computed once at configure).
Float4 ground_value(const Table &p_table, const SampleCarry &p_carry, const float p_slots[Cascade::SLOTS]);
// The same in GLSL: `vec4 climate_ground(Ch ch)` returning (value, d/dx, d/dy, d/dz) like the cascade's c0, after the
// shared helpers (splices_helpers is true with ground terms), ch in p_combined's GPU layout.
ov::str ground_glsl(const Table &p_table, const Cascade::Spec &p_combined, Cascade::ShaderParams &r_params);

// The biome with the largest weight p_w at lattice point p_p of Λ_m; among equal weights (same bits), the
// (hash % count)-th in list order, hash = ExactNoise::hash_cell(the point's LOD0 voxel, PICK_SEED): the same from every
// LOD that has the point. The derive picks on the level above a run's last (SampleCarry::pick).
int pick(const Table &p_table, const Float4 p_w[MAX_BIOMES], const Vector3i64 &p_p, int p_m);
// The carried pick of a point's Cascade slots p_slots (combined_spec's layout, the last level of a run: Cascade::point_slots
// with p_all_fields, or Level::slots_at of a run's last level carrying the sample terms).
int pick_slots(const Table &p_table, const float p_slots[Cascade::SLOTS]);
int pick_slots(const Table &p_table, const SampleCarry &p_carry, const float p_slots[Cascade::SLOTS]);

// 0xRRGGBB (sRGB) as a linear colour, alpha 1.
ov::Color srgb_to_linear(uint32_t p_rgb);
// Biome p_biome's ground material ("biome:<name>" in the material library, SDFCascade's base palette): the procedural
// ground (PATTERN_GROUND) in its surface's colours, its debug colour (--view=12).
Material::Params biome_material(const Biome &p_biome);

// The carried pick in GLSL, bit for bit: `uint climate_biome(Ch ch)`, ch the channels of a run's last level in
// p_combined's GPU layout. The derive's pick statements (on its w<b>, at point p of Λ_m) are pick_statements_glsl,
// assigning `float <p_var>`.
ov::str pick_glsl(const Table &p_table, const Cascade::Spec &p_combined);
ov::str pick_statements_glsl(const Table &p_table, const ov::str &p_var);
// The pieces both generated functions share (climate_glsl.cpp): the dual-number helpers (cl_flush, cl_mul, cl_sub,
// cl_edge), and apply_terms then weights as statements defining f<i> (the fields) and w<b> (the weights) from the channels
// of `Ch pre` (p_spec's GPU layout).
ov::str glsl_helpers();
ov::str glsl_weights(const Table &p_table, const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params);
// Whether the level shader splices the helpers once, up front (shared_helpers_glsl), for every generated function: with a
// shaped modulation or ground terms. Else each generator brings glsl_helpers as before them (the plain text).
bool splices_helpers(const Table &p_table, const Cascade::Spec &p_combined);
// glsl_helpers plus cl_add and cl_scale (Cascade::dadd, dscale).
ov::str shared_helpers_glsl();

} // namespace Climate
