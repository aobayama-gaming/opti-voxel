#pragma once

#include "godot_cpp/variant/vector3.hpp"
#include "godot_cpp/core/math.hpp"

#include <cmath>

using namespace godot;

// Debug: centimetre-scale test shapes floating ~3 m in front of the main_scene spawn point, to
// check what a given voxel size resolves. Negative inside (project convention).
//   - sphere ladder (y +0.5): radii 2, 4, 6, 8, 12, 16, 24, 32 cm, left to right
//   - thin walls (y -0.3): 2, 5, 10, 20 cm thick, left to right
//   - bumpy plate (right, y -0.3): 8 cm slab with +-2 cm bumps, 20 cm wavelength
//   - small tree (behind, lower): 6 cm trunk, 4 / 3 / 2.5 cm branches, 30 cm crown, smooth blends
// The CPU version (chunk discovery) and the GLSL version (meshing) below must stay identical.
namespace SDFTestRig {

inline const Vector3 CENTER(-52.75f, 355.6f, -98.5f);
inline constexpr float BOUND_RADIUS = 2.6f;

inline float sd_sphere(const Vector3 &p, float r) {
    return p.length() - r;
}

inline float sd_box(const Vector3 &p, const Vector3 &b) {
    const Vector3 q = p.abs() - b;
    const Vector3 outside(MAX(q.x, 0.0f), MAX(q.y, 0.0f), MAX(q.z, 0.0f));
    return outside.length() + MIN(MAX(q.x, MAX(q.y, q.z)), 0.0f);
}

inline float sd_capsule(const Vector3 &p, const Vector3 &a, const Vector3 &b, float r) {
    const Vector3 pa = p - a;
    const Vector3 ba = b - a;
    const float h = CLAMP(pa.dot(ba) / ba.dot(ba), 0.0f, 1.0f);
    return (pa - ba * h).length() - r;
}

inline float smin(float a, float b, float k) {
    const float h = CLAMP(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return Math::lerp(b, a, h) - k * h * (1.0f - h);
}

inline float distance_local(const Vector3 &q) {
    float d = 1e9f;
    static const float radii[8] = { 0.02f, 0.04f, 0.06f, 0.08f, 0.12f, 0.16f, 0.24f, 0.32f };
    for (int i = 0; i < 8; i++) {
        d = MIN(d, sd_sphere(q - Vector3(-1.75f + 0.5f * i, 0.5f, 0.0f), radii[i]));
    }
    static const float thickness[4] = { 0.02f, 0.05f, 0.10f, 0.20f };
    for (int i = 0; i < 4; i++) {
        d = MIN(d, sd_box(q - Vector3(-1.6f + 0.55f * i, -0.3f, 0.0f), Vector3(thickness[i] * 0.5f, 0.2f, 0.2f)));
    }
    {
        const Vector3 r = q - Vector3(1.35f, -0.3f, 0.0f);
        const float bump = 0.02f * std::sin(r.x * 31.415927f) * std::sin(r.z * 31.415927f);
        const float slab = std::abs(r.y) - (0.04f + bump);
        d = MIN(d, MAX(slab, MAX(std::abs(r.x) - 0.4f, std::abs(r.z) - 0.4f)));
    }
    {
        const Vector3 r = q - Vector3(0.0f, -1.1f, -1.2f);
        float t = sd_capsule(r, Vector3(0, 0, 0), Vector3(0, 1.3f, 0), 0.06f);
        t = smin(t, sd_capsule(r, Vector3(0, 0.7f, 0), Vector3(0.35f, 1.0f, 0.1f), 0.04f), 0.04f);
        t = smin(t, sd_capsule(r, Vector3(0, 0.9f, 0), Vector3(-0.3f, 1.15f, -0.1f), 0.03f), 0.03f);
        t = smin(t, sd_capsule(r, Vector3(0, 1.05f, 0), Vector3(0.1f, 1.3f, 0.3f), 0.025f), 0.03f);
        t = smin(t, sd_sphere(r - Vector3(0, 1.45f, 0), 0.3f), 0.08f);
        d = MIN(d, t);
    }
    return d;
}

inline float evaluate(const Vector3 &p_world) {
    const Vector3 q = p_world - CENTER;
    const float bound = q.length() - BOUND_RADIUS;
    // Far away a bounding sphere is a valid (lower-bound) distance and costs nothing.
    return bound > 0.5f ? bound : distance_local(q);
}

inline constexpr const char GLSL[] = R"glsl(
const vec3 RIG_CENTER = vec3(-52.75, 355.6, -98.5);

float rig_sd_box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float rig_sd_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float rig_smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float rig_distance(vec3 q) {
    float d = 1e9;
    const float radii[8] = float[8](0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32);
    for (int i = 0; i < 8; i++) {
        d = min(d, length(q - vec3(-1.75 + 0.5 * float(i), 0.5, 0.0)) - radii[i]);
    }
    const float thickness[4] = float[4](0.02, 0.05, 0.10, 0.20);
    for (int i = 0; i < 4; i++) {
        d = min(d, rig_sd_box(q - vec3(-1.6 + 0.55 * float(i), -0.3, 0.0), vec3(thickness[i] * 0.5, 0.2, 0.2)));
    }
    {
        vec3 r = q - vec3(1.35, -0.3, 0.0);
        float bump = 0.02 * sin(r.x * 31.415927) * sin(r.z * 31.415927);
        float slab = abs(r.y) - (0.04 + bump);
        d = min(d, max(slab, max(abs(r.x) - 0.4, abs(r.z) - 0.4)));
    }
    {
        vec3 r = q - vec3(0.0, -1.1, -1.2);
        float t = rig_sd_capsule(r, vec3(0.0), vec3(0.0, 1.3, 0.0), 0.06);
        t = rig_smin(t, rig_sd_capsule(r, vec3(0.0, 0.7, 0.0), vec3(0.35, 1.0, 0.1), 0.04), 0.04);
        t = rig_smin(t, rig_sd_capsule(r, vec3(0.0, 0.9, 0.0), vec3(-0.3, 1.15, -0.1), 0.03), 0.03);
        t = rig_smin(t, rig_sd_capsule(r, vec3(0.0, 1.05, 0.0), vec3(0.1, 1.3, 0.3), 0.025), 0.03);
        t = rig_smin(t, length(r - vec3(0.0, 1.45, 0.0)) - 0.3, 0.08);
        d = min(d, t);
    }
    return d;
}

vec4 test_rig_sdf(vec3 pos) {
    vec3 q = pos - RIG_CENTER;
    float bound = length(q) - 2.6;
    if (bound > 0.5) {
        return vec4(normalize(q), bound);
    }
    // Gradient by central differences (1 mm): only evaluated near the rig.
    const float h = 0.001;
    vec3 g = vec3(
        rig_distance(q + vec3(h, 0.0, 0.0)) - rig_distance(q - vec3(h, 0.0, 0.0)),
        rig_distance(q + vec3(0.0, h, 0.0)) - rig_distance(q - vec3(0.0, h, 0.0)),
        rig_distance(q + vec3(0.0, 0.0, h)) - rig_distance(q - vec3(0.0, 0.0, h)));
    return vec4(normalize(g + vec3(0.0, 1e-9, 0.0)), rig_distance(q));
}
)glsl";

} // namespace SDFTestRig
