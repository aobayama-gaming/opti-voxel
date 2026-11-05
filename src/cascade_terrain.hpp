#pragma once

#include "cascade.hpp"

// The cascade's terrain: a program's output h (value + gradient, LOD0 voxels) is a height over the base surface
// (BaseSurface), and height_sdf turns it into the distance SDFCascade meshes.
namespace CascadeTerrain {

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
