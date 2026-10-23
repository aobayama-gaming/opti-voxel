#pragma once

#include "cascade.hpp"

// The cascade's terrain (SDFCascade meshes it; the tests use it directly): a 3D height field
// h(x, y, z) of perlin3 octaves, every octave with the same steepness (amplitude = steepness * wavelength,
// in LOD0 voxels), surface at altitude h over the base surface (BaseSurface).
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

// SDF of the height field over the base surface from cascade channels on Λ_L (value in LOD0 voxels, slopes per
// LOD-L voxel), in voxels of LOD L. p_base: BaseSurface::at, (up n, altitude) in voxels of LOD L. t = n - grad h
// with h and the position both in LOD-L voxels (exact 2^-L scalings), distance (altitude - h) / |t|. Returns
// (normalised t, distance). altitude - h subtracts two heights in float: both stay within the bands' amplitudes of
// the base (the base saturates far from it). GLSL: GLSL_HEIGHT_SDF, op for op.
inline ExactNoise::Float4 height_sdf(const float p_ch[Cascade::CHANNELS], int p_lod, const ExactNoise::Float4 &p_base) {
    using namespace Cascade;
    const float s = ExactNoise::exact_pow2(-p_lod);
    const float h = p_ch[VALUE] * s;
    const float tx = p_base.x - p_ch[DX] * s;
    const float ty = p_base.y - p_ch[DY] * s;
    const float tz = p_base.z - p_ch[DZ] * s;
    const float tr = ExactNoise::det_rsqrt(tx * tx + ty * ty + tz * tz);
    return ExactNoise::Float4{ tx * tr, ty * tr, tz * tr, (p_base.w - h) * tr };
}

// height_sdf on the GPU. c0 = (value, d/dx, d/dy, d/dz) on Λ_lod, base = the base's vec4 (base_at).
inline constexpr const char GLSL_HEIGHT_SDF[] = R"glsl(
vec4 height_sdf(vec4 c0, vec4 base, int lod) {
    precise float s = exact_pow2(-lod);
    precise float h = c0.x * s;
    precise float tx = base.x - c0.y * s;
    precise float ty = base.y - c0.z * s;
    precise float tz = base.z - c0.w * s;
    precise float tr = det_rsqrt(tx * tx + ty * ty + tz * tz);
    return vec4(tx * tr, ty * tr, tz * tr, (base.w - h) * tr);
}
)glsl";

} // namespace CascadeTerrain
