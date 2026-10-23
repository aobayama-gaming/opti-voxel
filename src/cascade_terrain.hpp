#pragma once

#include "cascade.hpp"

// The cascade's terrain (SDFCascade meshes it; the tests use it directly): a 3D height field
// h(x, y, z) of perlin3 octaves, every octave with the same steepness (amplitude = steepness * wavelength,
// in LOD0 voxels), surface at y = datum + h.
namespace CascadeTerrain {

// Octaves from the finest one LOD 0 inserts (k = insert_shift, on Λ_0) up to p_k_top; seeds and offsets
// from ExactNoise::make_fbm.
inline Cascade::Spec make_spec(uint32_t p_seed, int p_k_top, double p_steepness, Cascade::ChannelSet p_set, int p_insert_shift) {
    Cascade::Spec spec;
    spec.set = p_set;
    spec.insert_shift = p_insert_shift;
    spec.bands = ExactNoise::make_fbm(false, p_seed, p_k_top, p_k_top - p_insert_shift + 1, p_steepness * std::ldexp(1.0, p_k_top), 0.5).octaves;
    return spec;
}

// The design's Earth: octaves up to k = 29, 8 samples per wavelength, top lattice Λ_26, steepness 1/8
// with amplitudes capped at EARTH_MAX_AMP. The value channel is a float32 height in LOD0 voxels: its
// resolution is 2^-23 of the largest height, so the planet's base shape stays out of the bands (a per-chunk
// bias in the integer frame, see the design) and continent bands keep km-scale amplitudes.
inline constexpr double EARTH_MAX_AMP = 65536.0; // 2^16 voxels: 3.3 km at 5 cm

inline Cascade::Spec earth(Cascade::ChannelSet p_set) {
    Cascade::Spec spec = make_spec(0xEA27u, 29, 0.125, p_set, 3);
    for (Cascade::Band &b : spec.bands) {
        b.amp = float(std::min(double(b.amp), EARTH_MAX_AMP));
    }
    return spec;
}

// SDF of the height field from cascade channels on Λ_L (value in LOD0 voxels, slopes per LOD-L voxel),
// in voxels of LOD L: t = (-dh/dx, 1 - dh/dy, -dh/dz) with h and the position both in LOD-L voxels
// (exact 2^-L scalings), distance (dy - h) / |t|. p_dy: height above the datum in LOD-L voxels.
// Returns (normalised t, distance). p_dy - h subtracts two heights in float: both stay within the bands'
// amplitudes of the datum (p_dy saturated by the caller).
inline ExactNoise::Float4 height_sdf(const float p_ch[Cascade::CHANNELS], int p_lod, float p_dy) {
    using namespace Cascade;
    const float s = ExactNoise::exact_pow2(-p_lod);
    const float h = p_ch[VALUE] * s;
    const float tx = -(p_ch[DX] * s);
    const float ty = 1.0f - p_ch[DY] * s;
    const float tz = -(p_ch[DZ] * s);
    const float tr = ExactNoise::det_rsqrt(tx * tx + ty * ty + tz * tz);
    return ExactNoise::Float4{ tx * tr, ty * tr, tz * tr, (p_dy - h) * tr };
}

} // namespace CascadeTerrain
