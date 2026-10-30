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
// Weights (weights): biome b's indicator r_b is the product of its edges' smoothsteps, s = clamp((x - anchor) *
// scale, 0, 1), t = s^2 (3 - 2 s) (a rising edge has scale > 0, a falling one < 0). Priority layering makes any
// indicators, overlapping or not, a partition of unity without a division: w_b = rest * r_b, rest -= w_b, the last
// biome (the default, no edge) takes the rest. Dual numbers (forward-mode gradient), one fixed op order: the GPU
// mirror (Modulator::glsl) walks the same order. Exactness: + - * and compares only; after every product each component under
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

// A field: its plain perlin3 bands (a LINEAR4 cascade group, inserted like the ground's, octave k on
// Λ_{k - BAND_SHIFT}) and terms.
struct Field {
    ov::str name;
    uint32_t seed = 0;
    std::vector<Cascade::Band> bands; // coarsest first, at most one per octave
    std::vector<Term> terms;
};

// One side of a biome: s = clamp((x - anchor) * scale, 0, 1) on field `field`, smoothstepped. min m with fade f is
// anchor m - f, scale 1 / f; max M with fade f is anchor M + f, scale -1 / f (1 on [m, M], 0 beyond the fades).
struct Edge {
    int field = 0;
    float anchor = 0.0f;
    float scale = 1.0f;
};

// How a biome plays the ground's octave k (a note): its multiplier, the modulated cascade's (combined_spec), and how
// the biome shapes it. gain > 0 weights it by the multifractal state, clamp(gain (1 - c), 0, 1), c the
// complement of the previous multifractal octave's weighted signal (Musgrave: ridged gain 2, hybrid 1); warp != 0 reads
// its noise displaced by warp LOD0 voxels times the values of the fields warp_fields (libnoise's turbulence), the
// displacement's gradient applied by the chain rule.
// Left out: weights or warps read from the ground's slope (Quilez's and de Carpentier's slope damping, the swiss
// turbulence's warp). Their own gradient needs the ground's second derivatives, which LINEAR4 doesn't carry, and the
// Hermite subdivision needs a band's gradient to be its value's: without that term the finer levels' values streak along
// the lattice axes (measured, the climate report's gradient error: 6% and visible streaks, 2.5% for the plain ground).
struct Amp {
    int k = 0;
    float mult = 1.0f;
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
// max |curve'| (skirts).
float curve_slope_bound(const Curve &p_curve);
// max |curve| (its cubic segments overshoot their points; validate's bound).
float curve_bound(const Curve &p_curve);

// A biome's ground look (its material, biome_material): albedo on gentle slopes and on steep ones (linear colours),
// roughness. The default is the page shader's procedural ground.
struct Surface {
    ov::Color flat = Material::GROUND_FLAT;
    ov::Color steep = Material::GROUND_STEEP;
    float roughness = 1.0f;
};

// A biome: where it is (its edges on the fields), how it plays the ground's octaves (amps), its height curve and its
// look. The curve remaps the ground height h at each sample: h + C(h), C the curve through the points (h_i, R_i - h_i)
// of the format's [h, R(h)] (LOD0 voxels, constant past the ends: R - h holds), blended by the biome's weight
// (ground_value). Without points, C = 0.
struct Biome {
    ov::str name;
    std::vector<Edge> edges; // empty for the last (default) biome only
    std::vector<Amp> amps;
    Curve curve; // empty: none
    Surface surface;
    uint32_t debug_color = 0; // 0xRRGGBB, reports and debug views
};

struct Table {
    std::vector<Field> fields;
    std::vector<Biome> biomes;
    // (k, stretch) of the ground's octaves stretched along the base's up ("ground_stretch": Band::stretch, combined_spec).
    std::vector<std::pair<int, int>> ground_stretch;

    // The field named p_name, -1 if none.
    int field_index(const ov::str &p_name) const;
    // Whether a biome weights a band by the multifractal state: the combined spec carries one more field group for it.
    bool has_state() const;
    // Whether a biome has a curve: the ground at a sample is then more than the cascade's (ground_value).
    bool has_curves() const;
    // Whether a ground octave is stretched along the base's up.
    bool has_stretch() const { return !ground_stretch.empty(); }
};

// What a table is checked against and reads: the ground cascade (SDFCascade::climate_ground). Ground() is no ground:
// no term or multiplier may read it, and fields have no octave.
struct Ground {
    const Cascade::Spec *spec = nullptr;
    double voxel_size = 0.0; // metres: converts the format's metres to voxels and octaves
    int k_max = -1; // the coarsest octave a band may have (the ground's)
    float amp_cap = 0.0f; // ground band amplitude cap, LOD0 voxels
};

// Empty if p_table is well formed, else the first error: field and biome counts, unique names, bands (k in
// [BAND_SHIFT, k_max], no duplicate, amplitude in ExactNoise's exact range), terms (GROUND or an earlier field,
// coefficient within [2^-40, 2^40]), edges (an existing field, bound 0 or within [2^-40, 2^40], scale within it), the
// last biome alone without edges, amplitudes (an existing ground octave, multiplier >= 0, a modulated amplitude in the
// exact range or 0, within the cap), curves (their offset under the ground's cap). No per-LOD gradient check
// (ExactNoise::gradient_range_error): the cascade flushes what falls under 2^-64.
ov::str validate(const Table &p_table, const Ground &p_ground);

// The bands a term's source names: the ground's (GROUND), a field's, or null.
const std::vector<Cascade::Band> *bands_of(const Table &p_table, const Ground &p_ground, int p_source);

// The table of a JSON object (units in metres; the format is described above parse() in climate_parse.cpp),
// validated; on error r_table is left empty and the error returned.
ov::str parse(const ov::Value &p_data, const Ground &p_ground, Table &r_table);

// The JSON object of a table file (a host path: res:// or absolute, ov::read_file) in r_data; empty if it is readable,
// else the error.
ov::str load(const ov::str &p_path, ov::Value &r_data);

// The ground spec p_ground (LINEAR4) with the table's fields as its field groups and, if any biome names an amplitude
// multiplier, a Modulation: every band some biome multiplies gets A = sum_b w_b amp_b at each point (amp_b = the
// biome's multiplier times the band's amplitude, the band's amplitude for a biome without one), from the weights of
// the climate before that level's insertions.
Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground);

// The amplitude biome p_biome gives the ground's band p_band: its multiplier times the band's amplitude (validate's
// product), or the band's own without one. The modulation's per-biome amplitudes.
float biome_amp(const Biome &p_biome, const Cascade::Band &p_band);
// The largest biome_amp of any biome, times its shape's bound: bounds that follow amplitudes (skirts) take it.
float max_amp(const Table &p_table, const Cascade::Band &p_band);
// Biome p_biome's entry of the ground's octave p_k, or null.
const Amp *find_amp(const Biome &p_biome, int p_k);
// The shape biome p_biome gives the ground's octave p_k: its entry's, else plain. What the modulation inserts and
// max_amp bounds.
Shape band_shape(const Biome &p_biome, int p_k);
// The largest multiplier biome p_biome may give the ground's octave p_k: its shaped amplitude under the ground's cap
// (validate refuses more). 0 without such a band, infinite for a shape bounded by 0.
double max_ground_mult(const Ground &p_ground, const Biome &p_biome, int p_k);
// The skirt's band step at LOD p_lod (LOD0 voxels) with the biomes' curves, from the cascade's p_ground_amp: the ground
// h + sum_b w_b C_b(h) moves by (1 + max |C_b'|) times h's step, plus each weight's change (its edges' smoothstep
// slopes, 1.5 |scale|, times their fields' steps) times |C_b|'s bound.
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
// a run's level above its last (Cascade::Modulation::derive) and read by its samples. curve[b]: biome b's weight if it
// has a curve; pick: the biome (its index in the value, copied from the floor parent, not interpolated). -1: none.
struct SampleCarry {
    int curve[MAX_BIOMES];
    int pick = -1;
    int count = 0;
};
SampleCarry sample_carry(const Table &p_table);
// Carry group p_c of `Ch <p_ch>` in p_spec's GPU layout (Spec::gpu_carry) as a (gradient, value) vec4: to_float4's twin.
ov::str carried_glsl(const Cascade::Spec &p_spec, const ov::str &p_ch, int p_c);

// The ground at a sample: the cascade's ground channels h of p_slots (a point's slots), then h + sum_b w_b C_b(h) over
// the biomes with a curve, w_b carried (SampleCarry). Without curves, the cascade's channels, bit for bit.
Float4 ground_value(const Table &p_table, const float p_slots[Cascade::SLOTS]);
// The same in GLSL: `vec4 climate_ground(Ch ch)` returning (value, d/dx, d/dy, d/dz) like the cascade's c0, after the
// shared helpers (splices_helpers is true with ground terms), ch in p_combined's GPU layout.
ov::str ground_glsl(const Table &p_table, const Cascade::Spec &p_combined);

// The biome with the largest weight p_w at lattice point p_p of Λ_m; among equal weights (same bits), the
// (hash % count)-th in list order, hash = ExactNoise::hash_cell(the point's LOD0 voxel, PICK_SEED): the same from every
// LOD that has the point. The derive picks on the level above a run's last (SampleCarry::pick).
int pick(const Table &p_table, const Float4 p_w[MAX_BIOMES], const Vector3i64 &p_p, int p_m);
// The carried pick of a point's Cascade slots p_slots (combined_spec's layout, the last level of a run: Cascade::point_slots
// with p_all_fields, or Level::slots_at of a run's last level carrying the sample terms).
int pick_slots(const Table &p_table, const float p_slots[Cascade::SLOTS]);

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
ov::str glsl_weights(const Table &p_table, const Cascade::Spec &p_spec);
// Whether the level shader splices the helpers once, up front (shared_helpers_glsl), for every generated function: with a
// shaped modulation or ground terms. Else each generator brings glsl_helpers as before them (the plain text).
bool splices_helpers(const Table &p_table, const Cascade::Spec &p_combined);
// glsl_helpers plus cl_add and cl_scale (Cascade::dadd, dscale).
ov::str shared_helpers_glsl();

} // namespace Climate
