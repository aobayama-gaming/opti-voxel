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
// the cascade's level 0). Next: categorical picks per 3D cell for placement.
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

inline constexpr int MAX_FIELDS = 8;
inline constexpr int MAX_BIOMES = 16;
// A term's or an amplitude's source that is not a field: the ground cascade.
inline constexpr int GROUND = -1;
// Seed of the pick's tie-break hash (ExactNoise::hash_cell of the lattice point).
inline constexpr uint32_t PICK_SEED = 0xB10E5u;

// coef * (the channels of `source`): GROUND or an earlier field.
struct Term {
    int source = GROUND;
    float coef = 0.0f;
};

// A field: its bands (a LINEAR4 cascade group, inserted like the ground's, octave k on Λ_{k - BAND_SHIFT}) and terms.
struct Field {
    String name;
    uint32_t seed = 0;
    std::vector<Cascade::Band> bands; // coarsest first, at most one per octave
    std::vector<Term> terms;
};

// One side of a region: s = clamp((x - anchor) * scale, 0, 1) on field `field`, smoothstepped. min m with fade f is
// anchor m - f, scale 1 / f; max M with fade f is anchor M + f, scale -1 / f (1 on [m, M], 0 beyond the fades).
struct Edge {
    int field = 0;
    float anchor = 0.0f;
    float scale = 1.0f;
};

// An amplitude multiplier of one band (field or GROUND, octave k): the modulated cascade's (combined_spec).
struct Amp {
    int field = GROUND;
    int k = 0;
    float mult = 1.0f;
};

// A biome's ground look (its material, biome_material): albedo on gentle slopes and on steep ones (linear colours),
// roughness. The default is the page shader's procedural ground.
struct Surface {
    Color flat = Material::GROUND_FLAT;
    Color steep = Material::GROUND_STEEP;
    float roughness = 1.0f;
};

struct Biome {
    String name;
    std::vector<Edge> edges; // empty for the last (default) biome only
    std::vector<Amp> amps;
    Surface surface;
    uint32_t debug_color = 0; // 0xRRGGBB, reports and debug views
};

struct Table {
    std::vector<Field> fields;
    std::vector<Biome> biomes;

    int field_index(const String &p_name) const;
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
// range or 0, a ground one within the cap). No per-LOD gradient check
// (ExactNoise::gradient_range_error): the cascade flushes what falls under 2^-64.
String validate(const Table &p_table, const Ground &p_ground);

// The table of a JSON-like Dictionary (units in metres; the format is described above parse() in climate.cpp),
// validated; on error r_table is left empty and the error returned.
String parse(const Dictionary &p_data, const Ground &p_ground, Table &r_table);

// The JSON object of a table file (res:// or absolute) in r_data; empty if it is readable, else the error.
String load(const String &p_path, Dictionary &r_data);

static_assert(MAX_FIELDS <= Cascade::MAX_FIELD_GROUPS, "every field is a cascade field group");

// The ground spec p_ground (LINEAR4) with the table's fields as its field groups and, if any biome names an amplitude
// multiplier, a Modulation: every band some biome multiplies gets A = sum_b w_b amp_b at each point (amp_b = the
// biome's multiplier times the band's amplitude, the band's amplitude for a biome without one), from the weights of
// the climate before that level's insertions.
Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground);

// The amplitude biome p_biome gives band p_band of group p_field (GROUND or a field): its multiplier times the band's
// amplitude (validate's product), or the band's own without one. The modulation's per-biome amplitudes.
float biome_amp(const Biome &p_biome, int p_field, const Cascade::Band &p_band);
// The largest biome_amp of any biome: bounds that follow amplitudes (skirts) take it.
float max_amp(const Table &p_table, int p_field, const Cascade::Band &p_band);

// The field values of the channels p_slots (a point's Cascade slots, combined_spec's layout): each field's own
// channels, then its terms in list order (Cascade::point_slots, or a level's channels before its insertions).
void apply_terms(const Table &p_table, const float p_slots[Cascade::SLOTS], Float4 r_fields[MAX_FIELDS]);

// Channels of every field at lattice point p_p of Λ_m (p_p = voxel >> m), r_fields[i] for field i: V_m of
// p_combined (combined_spec) at the point, the fields carried at every level, then the terms.
void fields_at(const Table &p_table, const Cascade::Spec &p_combined, const Vector3i64 &p_p, int p_m, Float4 r_fields[MAX_FIELDS]);

// Biome weights (value and gradient) of the field channels p_fields, r_w[b] for biome b.
void weights(const Table &p_table, const Float4 p_fields[MAX_FIELDS], Float4 r_w[MAX_BIOMES]);

// The biome with the largest weight p_w at lattice point p_p of Λ_m; among equal weights (same bits), the
// (hash % count)-th in list order, hash = ExactNoise::hash_cell(the point's LOD0 voxel, PICK_SEED): the same from every
// LOD that has the point.
int pick(const Table &p_table, const Float4 p_w[MAX_BIOMES], const Vector3i64 &p_p, int p_m);
// The pick of a point's Cascade slots p_slots (combined_spec's layout, the fields carried: Cascade::point_slots with
// p_all_fields, or Level::slots_at of a run carrying them): apply_terms, weights, pick.
int pick_slots(const Table &p_table, const float p_slots[Cascade::SLOTS], const Vector3i64 &p_p, int p_m);

// 0xRRGGBB (sRGB) as a linear colour, alpha 1.
Color srgb_to_linear(uint32_t p_rgb);
// Biome p_biome's ground material ("biome:<name>" in the material library, SDFCascade's base palette): the procedural
// ground (PATTERN_GROUND) in its surface's colours, its debug colour (--view=12).
Material::Params biome_material(const Biome &p_biome);

// pick of the weights in GLSL, bit for bit: `uint climate_biome(Ch pre, WorldPos p)`, pre the channels of a lattice
// point in p_combined's GPU layout (the fields carried: V_m, the terms applied here) and p its absolute voxel. Brings
// the dual-number helpers unless p_combined's modulation (its glsl) already defines them.
String pick_glsl(const Table &p_table, const Cascade::Spec &p_combined);
// The pieces both generated functions share (climate_glsl.cpp): the dual-number helpers (cl_flush, cl_mul, cl_sub,
// cl_edge), and apply_terms then weights as statements defining f<i> (the fields) and w<b> (the weights) from the channels
// of `Ch pre` (p_spec's GPU layout).
String glsl_helpers();
String glsl_weights(const Table &p_table, const Cascade::Spec &p_spec);

} // namespace Climate
