#pragma once

#include "ov/format.hpp"

// The shading frame: how a shader reads a pattern (noise, waves, contour lines, the cloud tiles) at a render-space
// point so that it is the same at every render origin. Render positions are metres from the render origin, which moves
// with the camera (World::recenter); a pattern read at the render position itself would jump at every move, and one read
// at the absolute position would lose its float precision kilometres away. Instead each pattern repeats every P LOD0
// voxels (P a power of two of at least SHADE_FEATURES of its largest features): its frequencies are snapped to whole
// cycles per period (shade_cells, shade_wave) and it is read at the render position plus the origin's voxel modulo P
// (shade_point), a float under P + the camera's distance. Two origins then read points a whole number of periods apart:
// the same value, bit for bit up to the float rounding of the sum.
//
// The host declares, before this text: `ivec3 shade_origin_voxel()` (the render origin's LOD0 voxel, low 32 bits) and
// `float shade_voxel_m()` (the LOD0 voxel size in metres). Plain GLSL that Godot's shading language also takes.
namespace ShadeFrame {

// The fewest features a period holds: a snapped frequency is within 1 / (2 SHADE_FEATURES) of the asked one.
inline constexpr int FEATURES = 64;
// The longest period, 2^MAX_SHIFT voxels: the origin's voxel modulo it stays exact as a float (below 2^24).
inline constexpr int MAX_SHIFT = 23;

// %d: FEATURES, MAX_SHIFT.
inline constexpr const char GLSL[] = R"glsl(
// The period (LOD0 voxels, a power of two) of a pattern whose largest feature is feature_m metres.
int shade_period_vox(float feature_m) {
    float e = ceil(log2(max(%d.0 * feature_m / shade_voxel_m(), 1.0)));
    return 1 << int(min(e, %d.0));
}

float shade_period_m(int period_vox) {
    return float(period_vox) * shade_voxel_m();
}

// A render position (metres) in the frame of a pattern of period_vox voxels: plus the render origin's voxel modulo it.
vec3 shade_point(vec3 p, int period_vox) {
    return p + vec3(shade_origin_voxel() & ivec3(period_vox - 1)) * shade_voxel_m();
}

// The whole number of lattice cells per period nearest per_m cells per metre (at least 1): the lattice's scale is then
// cells / period_m, and it repeats every `cells` cells.
vec3 shade_cells(vec3 per_m, float period_m) {
    return max(round(per_m * period_m), vec3(1.0));
}

// The angular wavenumbers (radians per metre, per axis) nearest k with a whole number of cycles per period.
vec3 shade_wave(vec3 k, float period_m) {
    return round(k * (period_m / 6.28318531)) * (6.28318531 / period_m);
}
)glsl";

inline ov::str glsl() {
    return ov::format(GLSL, FEATURES, MAX_SHIFT);
}

} // namespace ShadeFrame
