#pragma once

#include "ov/str.hpp"

// A planet's base for the hosts' shaders (World::planet_render): its local up, the altitude above it and where a ray
// leaves a shell above it, in render space. The host declares, before this text, `vec3 planet_centre_m` (render
// metres), `float planet_radius_m` (0: the flat world, up +y) and `float planet_k_m2` (|centre|^2 - radius^2 from
// double), set at start and on every rebase. Plain GLSL that Godot's shading language also takes.
namespace PlanetGLSL {

inline constexpr const char GLSL[] = R"glsl(
// The base's up at render position p.
vec3 planet_up(vec3 p) {
    return planet_radius_m > 0.0 ? normalize(p - planet_centre_m) : vec3(0.0, 1.0, 0.0);
}

// |p - centre|^2 - (radius + h)^2 (m^2) from K, so its low bits survive (p - centre alone loses them on a large planet).
float planet_q(vec3 p, float h) {
    return dot(p, p) - 2.0 * dot(p, planet_centre_m) + (planet_k_m2 - h * (2.0 * planet_radius_m + h));
}

// The altitude of p above the planet's base (metres).
float planet_altitude(vec3 p) {
    return planet_q(p, 0.0) / (length(p - planet_centre_m) + planet_radius_m);
}

// Where a ray from p along the unit dir leaves the shell h metres above the base: the far root of
// |p + t dir - c| = R + h, written so it keeps its bits; false when p is above the shell.
bool planet_shell_hit(vec3 p, vec3 dir, float h, out vec3 hit) {
    float q = planet_q(p, h);
    float b = dot(p - planet_centre_m, dir);
    float s = sqrt(max(b * b - q, 0.0));
    hit = p + dir * (b > 0.0 ? -q / (b + s) : s - b);
    return q < 0.0;
}
)glsl";

inline ov::str glsl() {
    return ov::str(GLSL);
}

} // namespace PlanetGLSL
