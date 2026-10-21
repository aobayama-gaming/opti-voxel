#pragma once

#include "cascade.hpp"
#include "exact_noise.hpp"

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include <cstdint>
#include <vector>

// Climate (biomes, PR 1: the table and its CPU evaluation; no meshing reads it yet). A climate is data, read at
// configure: any number of scalar fields, each a set of perlin3 bands carried down the Hermite cascade like the ground,
// and biomes as regions of the space of those fields. Later PRs: the GPU mirror and amplitudes modulated by the weights,
// categorical picks per 3D cell for placement, materials per biome.
//
// Field f at lattice point p of Λ_m (fields_at): V_m(p) of its own LINEAR4 spec (Cascade::point_into: bands of
// octave k inserted on Λ_{k-3}, so Λ_m holds the octaves k >= m + 3), then its linear terms in list order,
// coef * (the ground's or an earlier field's channels at the same point). Each term is itself a pure function of
// (p, m), so a field is too: every LOD reads the same bits where it shares a point. Channels as Float4 (x, y, z:
// gradient in units of the lattice spacing of Λ_m, w: value), like the cascade's.
//
// Weights (weights): biome b's indicator r_b is the product of its edges' smoothsteps, s = clamp((x - anchor) *
// scale, 0, 1), t = s^2 (3 - 2 s) (a rising edge has scale > 0, a falling one < 0). Priority layering makes any
// regions, overlapping or not, a partition of unity without a division: w_b = rest * r_b, rest -= w_b, the last
// biome (the default, no edge) takes the rest. Dual numbers (forward-mode gradient), one fixed op order: the GPU
// mirror (PR 2) walks the same order. Exactness: + - * and compares only; after every product each component under
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

// A field's spec before its bands: LINEAR4, the cascade's insert shift, its own top (validate refuses any other).
inline Cascade::Spec field_spec() {
    Cascade::Spec spec;
    spec.set = Cascade::ChannelSet::LINEAR4;
    spec.insert_shift = ExactNoise::BAND_SHIFT;
    spec.top = -1;
    return spec;
}

struct Field {
    String name;
    uint32_t seed = 0;
    Cascade::Spec spec = field_spec(); // its bands
    std::vector<Term> terms;
};

// One side of a region: s = clamp((x - anchor) * scale, 0, 1) on field `field`, smoothstepped. min m with fade f is
// anchor m - f, scale 1 / f; max M with fade f is anchor M + f, scale -1 / f (1 on [m, M], 0 beyond the fades).
struct Edge {
    int field = 0;
    float anchor = 0.0f;
    float scale = 1.0f;
};

// An amplitude multiplier of one band (field or GROUND, octave k), read by the modulated cascade (PR 2).
struct Amp {
    int field = GROUND;
    int k = 0;
    float mult = 1.0f;
};

struct Biome {
    String name;
    std::vector<Edge> edges; // empty for the last (default) biome only
    std::vector<Amp> amps;
    float height = 0.0f; // mean height offset, LOD0 voxels
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
// range or 0, a ground one within the cap) and heights within the cap. No per-LOD gradient check
// (ExactNoise::gradient_range_error): the cascade flushes what falls under 2^-64.
String validate(const Table &p_table, const Ground &p_ground);

// The table of a JSON-like Dictionary (units in metres; the format is described above parse() in climate.cpp),
// validated; on error r_table is left empty and the error returned.
String parse(const Dictionary &p_data, const Ground &p_ground, Table &r_table);

// Channels of every field at lattice point p_p of Λ_m (p_p = voxel >> m), r_fields[i] for field i.
void fields_at(const Table &p_table, const Ground &p_ground, const Vector3i64 &p_p, int p_m, Float4 r_fields[MAX_FIELDS]);

// Biome weights (value and gradient) of the field channels p_fields, r_w[b] for biome b.
void weights(const Table &p_table, const Float4 p_fields[MAX_FIELDS], Float4 r_w[MAX_BIOMES]);

// The biome with the largest weight p_w at lattice point p_p of Λ_m; among equal weights (same bits), the
// (hash % count)-th in list order, hash = ExactNoise::hash_cell(the point's LOD0 voxel, PICK_SEED): the same from every
// LOD that has the point.
int pick(const Table &p_table, const Float4 p_w[MAX_BIOMES], const Vector3i64 &p_p, int p_m);

} // namespace Climate
