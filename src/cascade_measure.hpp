#pragma once

#include "cascade.hpp"

#include <vector>

// Measurements of the cascade against analytic perlin3 and of its cost, shared by the unit tests (small
// samples, thresholds) and CascadeReport (large samples, tables).
namespace CascadeMeasure {

// Steepnesses r = amplitude / wavelength at which normal errors are measured; the pass criterion uses
// REF_STEEPNESS (r = 1/8).
inline constexpr int STEEPNESS_COUNT = 3;
inline constexpr int STEEPNESS_LOG2[STEEPNESS_COUNT] = { -4, -3, -2 };
inline constexpr int REF_STEEPNESS = 1;
// Pass criterion 5 (fixed before measuring): value error <= 1% of the octave's amplitude, normal <= 1 deg.
inline constexpr double MAX_VALUE_ERROR = 0.01;
inline constexpr double MAX_NORMAL_DEG = 1.0;

struct Fidelity {
    int64_t points = 0;
    double value_max = 0.0; // |V - perlin|, in units of the octave's amplitude
    double value_rms = 0.0;
    double slope_max = 0.0; // |grad V - grad perlin| per cell
    double normal_max_deg[STEEPNESS_COUNT] = {}; // height-field normal (CascadeTerrain::height_sdf) at each steepness

    bool passes() const { return value_max <= MAX_VALUE_ERROR && normal_max_deg[REF_STEEPNESS] <= MAX_NORMAL_DEG; }
};

// Angle between two vectors in degrees (atan2 of |a x b| and a . b: accurate for small angles).
double angle_deg(const ExactNoise::Float4 &p_a, const ExactNoise::Float4 &p_b);

// Height-field normals at steepness 2^p_steepness_log2 of a single octave k (amplitude 1 in p_level):
// cascaded (from p_level at p_p) and analytic (p_perlin = perlin3 at the same point).
void height_normals(const Cascade::Level &p_level, const Vector3i64 &p_p, const ExactNoise::Float4 &p_perlin, int p_k, int p_steepness_log2,
        ExactNoise::Float4 &r_cascaded, ExactNoise::Float4 &r_analytic);

// One octave k alone (amplitude 1), inserted by the set / shift of p_spec (its bands are ignored),
// cascaded down to Λ_0 over p_boxes random boxes of p_size^3 voxels and compared at every voxel with
// perlin3 (the dense limit of the Hermite interpolant). Seeds and offsets from make_fbm(p_seed + box);
// anchors include far ones (2^50 voxels).
Fidelity measure_fidelity(const Cascade::Spec &p_spec, int p_k, int p_boxes, int p_size, uint32_t p_seed);

struct Cost {
    double points_mean = 0.0; // lattice points per chunk, all levels
    int64_t points_max = 0;
    std::vector<int64_t> first_levels; // points on Λ_L, Λ_{L+1}, ... of the first chunk
    double ms_per_chunk = 0.0; // only with p_evaluate
};

// Points per chunk at LOD p_lod over p_chunks random chunks (ids up to 2^40). With p_evaluate the chunks
// also run through Cascade::run (CPU time per chunk); otherwise only the boxes are counted.
Cost measure_cost(const Cascade::Spec &p_spec, int p_lod, int p_chunks, uint32_t p_seed, bool p_evaluate);

// Random chunk id at LOD p_lod (odd multiple of 2^L per axis, |id| < 2^40).
ChunkId random_chunk(int p_lod, uint32_t p_seed, int p_index);

} // namespace CascadeMeasure
