#pragma once

#include "cascade.hpp"
#include "exact_noise.hpp"
#include "material_library.hpp"

#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include <cstdint>
#include <vector>

// Climate (biomes): a climate is data, read at configure: any number of scalar fields, each a set of perlin3 bands
// carried down the Hermite cascade with the ground (combined_spec: the cascade's field groups), and biomes as regions of
// the space of those fields whose weights modulate the amplitudes of any band (Cascade::Modulation, on the CPU and in
// generated GLSL, bit for bit), and whose pick at each ground sample chooses its material (biome_material, SDFCascade::ground_material:
// the cascade's level 0). Biomes also shape the bands they modulate (climate_modulator.cpp: band shapes, the multifractal
// state, warps), and the table adds height terms and terraces to the ground at each sample (climate_shape.cpp,
// ground_value): libnoise's complex planet's operations, as cascade data. Next: categorical picks per 3D cell for
// placement.
//
// Field f at lattice point p of Λ_m (fields_at): its channels in V_m(p) of the combined spec (Λ_m holds the octaves
// k >= m + 3), then its linear terms in list order, coef * (the ground's or an earlier field's channels at the same
// point). Each term is itself a pure function of (p, m), so a field is too: every LOD reads the same bits where it
// shares a point. Channels as Float4 (x, y, z: gradient in units of the lattice spacing of Λ_m, w: value), like the
// cascade's.
//
// Weights (weights): biome b's indicator r_b is the product of its edges' smoothsteps, s = clamp((x - anchor) *
// scale, 0, 1), t = s^2 (3 - 2 s) (a rising edge has scale > 0, a falling one < 0). Priority layering makes any
// regions, overlapping or not, a partition of unity without a division: w_b = rest * r_b, rest -= w_b, the last
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
// Regions (Region) a table may name (State keeps their values).
inline constexpr int MAX_REGIONS = 16;
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
    bool plain() const { return c[0] == 0.0f && c[1] == 1.0f && c[2] == 0.0f && c[3] == 0.0f; }
};
// max |phi(n) - its mean| over n in [-1, 1] (perlin3's range): 1 for the plain shape. Skirts follow it (max_amp).
float shape_bound(const Shape &p_shape);

// A field: its bands (a LINEAR4 cascade group, inserted like the ground's, octave k on Λ_{k - BAND_SHIFT}) and terms.
struct Field {
    String name;
    uint32_t seed = 0;
    bool cell = false; // the bands are cell noise (ExactNoise::cell3), plain shape only
    std::vector<Cascade::Band> bands; // coarsest first, at most one per octave
    std::vector<Shape> shapes; // per band (the same order)
    std::vector<Term> terms;
};

// One side of a region: s = clamp((x - anchor) * scale, 0, 1) on field `field`, smoothstepped. min m with fade f is
// anchor m - f, scale 1 / f; max M with fade f is anchor M + f, scale -1 / f (1 on [m, M], 0 beyond the fades).
struct Edge {
    int field = 0;
    float anchor = 0.0f;
    float scale = 1.0f;
};

// An amplitude multiplier of one band (field or GROUND, octave k): the modulated cascade's (combined_spec), and how the
// biome shapes it. A ground band only: gain > 0 weights it by the multifractal state, clamp(gain (1 - c), 0, 1), c the
// complement of the previous multifractal octave's weighted signal (Musgrave: ridged gain 2, hybrid 1); warp != 0 reads
// its noise displaced by warp LOD0 voxels times the values of the fields warp_fields (libnoise's turbulence), the
// displacement's gradient applied by the chain rule.
// Left out: weights or warps read from the ground's slope (Quilez's and de Carpentier's slope damping, the swiss
// turbulence's warp). Their own gradient needs the ground's second derivatives, which LINEAR4 doesn't carry, and the
// Hermite subdivision needs a band's gradient to be its value's: without that term the finer levels' values streak along
// the lattice axes (measured, the climate report's gradient error: 6% and visible streaks, 2.5% for the plain ground).
struct Amp {
    int field = GROUND;
    int k = 0;
    float mult = 1.0f;
    bool has_shape = false; // else the band's own (a field's Field::shapes, the ground's plain)
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
};
// The curve through points (p_x strictly increasing, at least 2) and their p_y.
Curve make_curve(const std::vector<double> &p_x, const std::vector<double> &p_y);
// max |curve'| (skirts).
float curve_slope_bound(const Curve &p_curve);
// max |curve| (its cubic segments overshoot their points; validate's height term bound).
float curve_bound(const Curve &p_curve);

// A named region of the fields' space (the product of its edges' smoothsteps, like a biome's indicator): several
// landforms in one table, each behind its region. A biome naming it takes its edges before its own; a height term naming
// it is multiplied by it, so the landform's features (a trough, a cliff, a coast) exist only there.
struct Region {
    String name;
    std::vector<Edge> edges;
};

// A height term: the ground at each sample adds coef * curve(field) (LOD0 voxels; no curve: coef * field), times its
// region's value if it has one (a curve then: ground_step bounds it). Applied after the cascade (ground_value), so every
// LOD has it: what lifts land with a field (libnoise's continents).
struct HeightTerm {
    int field = 0;
    float coef = 0.0f;
    bool has_curve = false;
    Curve curve;
    int region = -1; // Table::regions index, -1: everywhere
};

// A biome's ground look (its material, biome_material): albedo on gentle slopes and on steep ones (linear colours),
// roughness. The default is the page shader's procedural ground.
struct Surface {
    Color flat = Material::GROUND_FLAT;
    Color steep = Material::GROUND_STEEP;
    float roughness = 1.0f;
};

// A biome's terraces of the ground height h (after the height terms): libnoise's Terrace with points every step LOD0
// voxels, g = v0 + (v1 - v0) t^2, t = (h - v0) / step (inverted: from the upper point down). Slope at most 2.
struct Terrace {
    float step = 0.0f; // 0: none
    float inv_step = 0.0f;
    bool invert = false;
};

struct Biome {
    String name;
    std::vector<Edge> edges; // empty for the last (default) biome only
    std::vector<Amp> amps;
    Terrace terrace;
    Surface surface;
    uint32_t debug_color = 0; // 0xRRGGBB, reports and debug views
};

// The index of the entry named p_name in p_list (fields, regions), -1 if none.
template <typename T>
int index_by_name(const std::vector<T> &p_list, const String &p_name) {
    for (size_t i = 0; i < p_list.size(); i++) {
        if (p_list[i].name == p_name) {
            return int(i);
        }
    }
    return -1;
}

struct Table {
    std::vector<Field> fields;
    std::vector<Region> regions;
    std::vector<Biome> biomes;
    std::vector<HeightTerm> heights;
    // The regions some height term is gated by, in index order (gated_regions, set by parse, checked by validate): the
    // ones ground_value and climate_ground evaluate.
    std::vector<int> gated;
    // (k, stretch) of the ground's octaves stretched along the base's up ("ground_stretch": Band::stretch, combined_spec).
    std::vector<std::pair<int, int>> ground_stretch;

    int field_index(const String &p_name) const { return index_by_name(fields, p_name); }
    int region_index(const String &p_name) const { return index_by_name(regions, p_name); }
    // Whether a biome weights a band by the multifractal state: the combined spec carries one more field group for it.
    bool has_state() const;
    // Whether the ground at a sample is more than the cascade's (height terms or terraces: ground_value).
    bool has_ground_terms() const;
    bool has_terraces() const;
    // Whether a field band or a ground octave is stretched along the base's up.
    bool has_stretch() const;
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
// last biome alone without edges, amplitudes (an existing band, multiplier >= 0, a modulated amplitude in the exact
// range or 0, a ground one within the cap), regions (identifier names, unique, edges like a biome's; a gated height term
// has a curve; Table::gated up to date). No per-LOD gradient check
// (ExactNoise::gradient_range_error): the cascade flushes what falls under 2^-64.
String validate(const Table &p_table, const Ground &p_ground);

// The bands a term's or a multiplier's source names: the ground's (GROUND), a field's, or null.
const std::vector<Cascade::Band> *bands_of(const Table &p_table, const Ground &p_ground, int p_source);

// The table of a JSON-like Dictionary (units in metres; the format is described above parse() in climate.cpp),
// validated; on error r_table is left empty and the error returned.
String parse(const Dictionary &p_data, const Ground &p_ground, Table &r_table);

// The JSON object of a table file (res:// or absolute) in r_data; empty if it is readable, else the error.
String load(const String &p_path, Dictionary &r_data);

// The ground spec p_ground (LINEAR4) with the table's fields as its field groups and, if any biome names an amplitude
// multiplier, a Modulation: every band some biome multiplies gets A = sum_b w_b amp_b at each point (amp_b = the
// biome's multiplier times the band's amplitude, the band's amplitude for a biome without one), from the weights of
// the climate before that level's insertions.
Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground);

// The amplitude biome p_biome gives band p_band of group p_field (GROUND or a field): its multiplier times the band's
// amplitude (validate's product), or the band's own without one. The modulation's per-biome amplitudes.
float biome_amp(const Biome &p_biome, int p_field, const Cascade::Band &p_band);
// The largest biome_amp of any biome, times its shape's bound (a cell field's: ExactNoise::CELL_BOUND): bounds that follow
// amplitudes (skirts, validate's height-term cap) take it.
float max_amp(const Table &p_table, int p_field, const Cascade::Band &p_band);
// Biome p_biome's multiplier entry of octave p_k of p_field (GROUND or a field), or null.
const Amp *find_amp(const Biome &p_biome, int p_field, int p_k);
// The shape biome p_biome (null: none) gives octave p_k of p_field: its entry's, else the band's own (a field's
// Field::shapes, the ground's plain). What the modulation inserts and max_amp bounds.
Shape band_shape(const Table &p_table, const Biome *p_biome, int p_field, int p_k);
// The largest multiplier biome p_biome may give the ground's octave p_k: its shaped amplitude under the ground's cap
// (validate refuses more; tools/landforms/match.py caps its fits with it, from climate_report's heights.json). 0 without
// such a band, infinite for a shape bounded by 0.
double max_ground_mult(const Table &p_table, const Ground &p_ground, const Biome &p_biome, int p_k);
// The skirt's band step at LOD p_lod (LOD0 voxels) with the table's ground terms, from the cascade's p_ground_amp: plus
// each height term's |coef| times its curve's slope bound times its field's step (its largest L+2 / L+3 octave and its
// terms' sources' steps), a gated one's plus |coef| times its curve's bound times its region's step (each edge's
// smoothstep slope, 1.5 |scale|, times its field's step, at most 1); with terraces, all times 2 (their slope bound) plus the terrace blend's change with the
// weights (each terraced biome's weight step, from its edges' fields, times a quarter of its terrace step).
double ground_step(const Table &p_table, int p_lod, double p_ground_amp);

// The field values of the channels p_slots (a point's Cascade slots, combined_spec's layout): each field's own
// channels, then its terms in list order; r_fields past the table's fields are left as they are (Cascade::point_slots, or a level's channels before its insertions).
void apply_terms(const Table &p_table, const float p_slots[Cascade::SLOTS], Float4 r_fields[MAX_FIELDS]);

// Channels of every field at lattice point p_p of Λ_m (p_p = voxel >> m), r_fields[i] for field i: V_m of
// p_combined (combined_spec) at the point, the fields carried at every level (the derive too), then the terms.
void fields_at(const Table &p_table, const Cascade::Spec &p_combined, const Vector3i64 &p_p, int p_m, Float4 r_fields[MAX_FIELDS]);

// Biome weights (value and gradient) of the field channels p_fields, r_w[b] for biome b.
void weights(const Table &p_table, const Float4 p_fields[MAX_FIELDS], Float4 r_w[MAX_BIOMES]);
// The product of the edges' smoothsteps on the field channels p_fields (dual numbers, in list order): a biome's indicator
// r_b, a region's value. 1 without edges.
Float4 edges_value(const std::vector<Edge> &p_edges, const Float4 p_fields[MAX_FIELDS]);
// The insertion level of the highest band (any group) the biomes give different amplitudes, shapes or warps (the
// continental octaves a world turns off in every biome don't count): above it, no band's climate matters.
int weighted_ceiling(const Table &p_table, const Cascade::Spec &p_combined);
// Bit r: region r is exactly 1 (value 1, gradient 0) at every point whose climate lattice point p_p of Λ_lod reads: its
// level boxes from Λ_{lod+1} to weighted_ceiling before their insertions (the bands above the last level), and Λ_{lod+1}'s
// after them (the derive the last level reads), one replay of the point's cascade. There every weight a band or a sample
// term reads is the region's biomes' alone: a world's landform is its own table, bit for bit (docs/landforms.md).
uint32_t core_regions(const Table &p_table, const Cascade::Spec &p_combined, const Vector3i64 &p_p, int p_lod);
// The regions some height term is gated by, in index order (parse stores them in Table::gated).
std::vector<int> gated_regions(const Table &p_table);

// The climate at a point, evaluated once per point of each level above a run's last (Modulator: a band's amplitude from
// the channels before the level's insertions; the level above the last derives the last level's amplitudes and its
// samples' terms after them) and by the report: the field values of a point's slots
// p_slots (apply_terms), then the biome weights (weights); GLSL twin: glsl_weights. What reaches a reader is one rule,
// the LOD rule: octave k of a modulated group, inserted on Λ_m, reads H of the climate of Λ_{m+1} after its insertions
// (the climate low-passed at k, Cascade::Modulation); a run's samples read H of the climate of the level above them.
struct State {
    Float4 fields[MAX_FIELDS];
    Float4 w[MAX_BIOMES];
};
void state(const Table &p_table, const float p_slots[Cascade::SLOTS], State &r_state);

// The carry groups of a table's sample terms, the first of Modulation's carry (the bands' amplitudes follow): derived on
// a run's level above its last (Cascade::Modulation::derive) and read by its samples. heights: the height
// terms' sum; terrace[b]: biome b's weight if it has terraces; pick: the biome (its index in the value, copied from the
// floor parent, not interpolated). -1: none.
struct SampleCarry {
    int heights = -1;
    int terrace[MAX_BIOMES];
    int pick = -1;
    int count = 0;
};
SampleCarry sample_carry(const Table &p_table);
// Carry group p_c of `Ch <p_ch>` in p_spec's GPU layout (Spec::gpu_carry) as a (gradient, value) vec4: to_float4's twin.
String carried_glsl(const Cascade::Spec &p_spec, const String &p_ch, int p_c);

// The height terms' sum (climate_shape.cpp) of the field values p_fields and the region values p_regions (edges_value of
// the gated regions, Table::gated): coef * curve(field) per term in list order (a gated one times its region's value),
// added up from 0. Dual numbers like the fields (gradient per lattice spacing, value in LOD0 voxels).
Float4 height_sum(const Table &p_table, const Float4 p_fields[MAX_FIELDS], const Float4 p_regions[MAX_REGIONS]);
// The same in GLSL: the curves' functions (curves_glsl, before), then statements defining `vec4 <p_var>` from the
// f<i> of glsl_weights and the rg<r> of glsl_edges_value (height_sum_glsl).
String curves_glsl(const Table &p_table);
String height_sum_glsl(const Table &p_table, const String &p_var);

// The ground at a sample: the cascade's ground channels of p_slots (a point's slots), plus the carried height terms'
// sum, then h + sum_b w_b (terrace_b(h) - h) over the biomes with terraces, w_b carried (SampleCarry). Without height
// terms or terraces, the cascade's channels, bit for bit.
Float4 ground_value(const Table &p_table, const float p_slots[Cascade::SLOTS]);
// The same in GLSL: `vec4 climate_ground(Ch ch)` returning (value, d/dx, d/dy, d/dz) like the cascade's c0, after the
// shared helpers (splices_helpers is true with ground terms), ch in p_combined's GPU layout.
String ground_glsl(const Table &p_table, const Cascade::Spec &p_combined);

// The biome with the largest weight p_w at lattice point p_p of Λ_m; among equal weights (same bits), the
// (hash % count)-th in list order, hash = ExactNoise::hash_cell(the point's LOD0 voxel, PICK_SEED): the same from every
// LOD that has the point. The derive picks on the level above a run's last (SampleCarry::pick).
int pick(const Table &p_table, const Float4 p_w[MAX_BIOMES], const Vector3i64 &p_p, int p_m);
// The carried pick of a point's Cascade slots p_slots (combined_spec's layout, the last level of a run: Cascade::point_slots
// with p_all_fields, or Level::slots_at of a run's last level carrying the sample terms).
int pick_slots(const Table &p_table, const float p_slots[Cascade::SLOTS]);

// 0xRRGGBB (sRGB) as a linear colour, alpha 1.
Color srgb_to_linear(uint32_t p_rgb);
// Biome p_biome's ground material ("biome:<name>" in the material library, SDFCascade's base palette): the procedural
// ground (PATTERN_GROUND) in its surface's colours, its debug colour (--view=12).
Material::Params biome_material(const Biome &p_biome);

// The carried pick in GLSL, bit for bit: `uint climate_biome(Ch ch)`, ch the channels of a run's last level in
// p_combined's GPU layout. The derive's pick statements (on its w<b>, at point p of Λ_m) are pick_statements_glsl,
// assigning `float <p_var>`.
String pick_glsl(const Table &p_table, const Cascade::Spec &p_combined);
String pick_statements_glsl(const Table &p_table, const String &p_var);
// The pieces both generated functions share (climate_glsl.cpp): the dual-number helpers (cl_flush, cl_mul, cl_sub,
// cl_edge), and apply_terms then weights as statements defining f<i> (the fields) and w<b> (the weights) from the channels
// of `Ch pre` (p_spec's GPU layout).
String glsl_helpers();
// p_ch: the Ch variable they read ("pre" in the modulation, before a level's insertions; "ch" in the pick and the ground).
String glsl_weights(const Table &p_table, const Cascade::Spec &p_spec, const String &p_ch = "pre");
// edges_value as statements defining `vec4 <p_var>` from the f<i> of glsl_weights.
String glsl_edges_value(const String &p_var, const std::vector<Edge> &p_edges);
// Whether the level shader splices the helpers once, up front (shared_helpers_glsl), for every generated function: with a
// shaped modulation or ground terms. Else each generator brings glsl_helpers as before them (the plain text).
bool splices_helpers(const Table &p_table, const Cascade::Spec &p_combined);
// glsl_helpers plus cl_add and cl_scale (Cascade::dadd, dscale).
String shared_helpers_glsl();

} // namespace Climate
