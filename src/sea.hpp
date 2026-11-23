#pragma once

#include "base_surface.hpp"
#include "world_file.hpp"

#include "ov/str.hpp"

// The sea (docs/look.md "Sea"): a world file's "sea" (WorldFile::Sea) drawn by a host's full-screen pass over the scene,
// and the shore it leaves on the ground. Engine-free: the hosts splice its GLSL and fill its uniforms from Render::visit.
// - The surface is the level sea.level_m above the base, met along each pixel's ray in closed form
//   (BaseSurface::glsl_render's base_level_span_k, its k from double on the CPU): SPHERE and FLAT (WorldFile refuses a
//   sea on a torus).
// - The water: Beer-Lambert per channel over the ray's length in it (a column of clarity_m metres passes `shallow`; deep
//   water takes `deep`'s colour, lit by the sun and the sky through it), a Fresnel (Schlick) reflection of the sky (the
//   atmosphere's radiance and its clouds, else the host's sky colour), a GGX sun glint, SEA_WAVES Gerstner waves in the
//   normals only (read through the shading frame, so they stay put when the render origin moves), each faded into the
//   glint's roughness where it is under WAVE_FADE_PX pixels, and foam where the water is under FOAM_M deep. The camera
//   under water sees the scene through the water (absorption as fog). The air in front of the surface is the
//   atmosphere's, its clouds' over it; the sun on it is cut by the clouds' shadow map.
// - The shore (the page shader): sand over [level - below_m, level + above_m], darker just above the waterline (wet).
class Sea {
public:
    // The waves: SEA_WAVES headings a golden angle apart, the longest WAVE_LONGEST_M, each next WAVE_RATIO times shorter,
    // their slopes (A k) summing to WAVE_SLOPE; a wave shorter than WAVE_FADE_PX pixels becomes roughness.
    static constexpr int WAVES = 5;
    static constexpr float WAVE_LONGEST_M = 24.0f;
    static constexpr float WAVE_RATIO = 0.57f;
    static constexpr float WAVE_SLOPE = 0.5f;
    static constexpr float WAVE_FADE_PX = 4.0f;
    // The glint's least GGX roughness (alpha), the water's reflectance at normal incidence.
    static constexpr float ROUGHNESS_MIN = 0.06f;
    static constexpr float F0 = 0.02f;
    // Foam where the water is under FOAM_M deep, a value noise of FOAM_SCALE_M features drifting FOAM_SPEED features a
    // second; its albedo.
    static constexpr float FOAM_M = 0.5f;
    static constexpr float FOAM_SCALE_M = 1.5f;
    static constexpr float FOAM_SPEED = 0.25f;
    static constexpr float FOAM_ALBEDO = 0.8f;
    // The shore: wet up to WET_M above the level (its colour times WET_DARKEN, its roughness WET_ROUGHNESS); the sand
    // band's edges blend over SHORE_EDGE of each side (at least SHORE_EDGE_MIN_M).
    static constexpr float WET_M = 0.6f;
    static constexpr float WET_DARKEN = 0.6f;
    static constexpr float WET_ROUGHNESS = 0.35f;
    static constexpr float SHORE_EDGE = 0.3f;
    static constexpr float SHORE_EDGE_MIN_M = 0.1f;

    // The sea in render space: what both shaders read (Sea::glsl_uniforms' names, in visit's order).
    struct Render {
        int on = 0; // 0: no sea (the pass draws nothing, the shore is the ground's)
        float k = 0.0f; // base_level_span_k's k for the level (BaseSurface::level_k): it changes with the render origin
        float level_m = 0.0f; // above the base
        Float3 deep;
        Float3 shallow;
        float clarity_m = 1.0f;
        float shore_below_m = 0.0f;
        float shore_above_m = 0.0f;
        Float3 sand;

        template <class F>
        void visit(F &&p_f) const {
            p_f("sea_on", on);
            p_f("sea_k", k);
            p_f("sea_level_m", level_m);
            p_f("sea_deep", deep);
            p_f("sea_shallow", shallow);
            p_f("sea_clarity_m", clarity_m);
            p_f("sea_shore_below_m", shore_below_m);
            p_f("sea_shore_above_m", shore_above_m);
            p_f("sea_sand", sand);
        }
    };
    // p_sea in render space with the render origin at LOD0 voxel p_origin over p_base: off without a sea (or on a torus).
    static Render render(const WorldFile::Sea &p_sea, const BaseSurface &p_base, const VoxelPos &p_origin, double p_voxel_m);
    // The height of p_pos above the sea's level in metres, negative under water: the base's altitude in double
    // (BaseSurface::altitude, the fraction along the up) minus p_level_m.
    static double height_at(const BaseSurface &p_base, const WorldPos &p_pos, double p_voxel_m, double p_level_m);

    // Render's fields as `uniform` declarations.
    static ov::str glsl_uniforms();
    // `void sea_shore(float h, inout vec3 albedo, inout float roughness)`: the shore on the ground at altitude h above the
    // base (metres); neutral without a sea. After glsl_uniforms.
    static ov::str glsl_shore();
    // `vec3 sea_shade(vec3 camera, vec3 dir, float scene_dist, vec3 scene, float pixel_rad, float time, vec3 sun_dir,
    // vec3 sun_light, vec3 sky_rgb, out bool hit)`: the colour of a pixel whose ray (render space, dir unit) reaches the
    // scene at scene_dist (1e30: the sky) where the scene's colour (its air included) is `scene`, the sea over it; hit
    // false (and `scene`) where the ray meets no water before it. pixel_rad: a pixel's angle; sun_dir: towards the sun;
    // sun_light: its light (atmo_sun_light's scale); sky_rgb: the sky's colour without the atmosphere. After glsl_uniforms,
    // the shading frame (ShadeFrame::glsl), BaseSurface::glsl_render, the atmosphere's and the clouds' lookups
    // (Atmosphere::glsl_lookup, Clouds::glsl_lookup) and the host's `bool atmo_enabled` and `bool cloud_enabled`.
    static ov::str glsl_water();
};
