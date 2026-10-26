#pragma once

// The Dawn viewer's look, after Godot's main scenes (their Environment and DirectionalLight3D): the sun's shadow map,
// the sky (SkyGLSL, the one Godot's VoxelSky draws), the fog towards it, then the ACES tonemapper and the grading LUT
// (tools/make_lut.py's). The page shader and the sky pass splice these after the Frame block (dawn_page_host.cpp).

#include "sky_glsl.hpp"

#include "ov/format.hpp"

namespace ov::dawn::LookGLSL {

// The look's textures, in descriptor set p_set: the shadow map (a depth texture compared through shadow_smp) and the
// grading cube (filtered: Godot samples its Texture3D with a linear filter).
inline ov::str declarations(int p_set) {
    return ov::format(R"glsl(
layout(set = %d, binding = 0) uniform texture2D shadow_tex;
layout(set = %d, binding = 1) uniform samplerShadow shadow_smp;
layout(set = %d, binding = 2) uniform texture3D lut_tex;
layout(set = %d, binding = 3) uniform sampler lut_smp;
)glsl",
            p_set, p_set, p_set, p_set);
}

// %s: the fog's light colour (glsl()). Needs the Frame block's sun, sun_color, shadow, shadow_proj and look (x: graded, y: the sky light, z: the sky's
// exposure), and SkyGLSL::GLSL.
inline constexpr const char GLSL[] = R"glsl(
// The Environment of Godot's main scenes (main_scene*.tscn): fog_light_color (linear), fog_sun_scatter,
// fog_aerial_perspective, fog_sky_affect.
const vec3 FOG_LIGHT_COLOR = %s;
const float FOG_SUN_SCATTER = 0.35;
const float FOG_AERIAL = 0.8;
const float FOG_SKY_AFFECT = 0.3;

vec3 sky_color(vec3 dir) {
    return physical_sky(dir, sun.xyz, sun_color.rgb, sun_color.w, sun.w, look.z);
}

// The fog's colour along dir (Godot's: its light colour, the sun scattered towards it, then the sky behind it, sky:
// sky_color(dir)).
vec3 fog_color(vec3 dir, vec3 sky) {
    vec3 fog = FOG_LIGHT_COLOR + sun_color.rgb * sun_color.w * FOG_SUN_SCATTER * pow(max(dot(dir, sun.xyz), 0.0), 8.0);
    return mix(fog, sky, FOG_AERIAL);
}

// How much of the sun reaches the world point p of normal n: 1 lit, 0 in a shadow. The shadow map is an orthographic
// view along the sun around the camera (shadow_proj: render metres to its clip space, depth 0 towards the sun); the
// point moves along its normal by shadow.y metres against acne, and four compared taps half a texel apart (shadow.z: a
// texel in uv; each tap a bilinear 2 x 2 compare) soften the edges, which fade out towards the map's border.
float sun_shadow(vec3 p, vec3 n) {
    if (shadow.x == 0.0) {
        return 1.0;
    }
    vec4 c = shadow_proj * vec4(p + n * shadow.y, 1.0);
    float edge = max(abs(c.x), abs(c.y));
    if (edge >= 1.0 || c.z <= 0.0 || c.z >= 1.0) {
        return 1.0;
    }
    vec2 uv = vec2(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    float lit = 0.0;
    for (int k = 0; k < 4; k++) {
        vec2 offset = vec2(float(k & 1) - 0.5, float(k >> 1) - 0.5) * shadow.z;
        lit += textureLod(sampler2DShadow(shadow_tex, shadow_smp), vec3(uv + offset, c.z), 0.0);
    }
    return mix(lit * 0.25, 1.0, smoothstep(0.8, 1.0, edge));
}

// Godot's ACES tonemapper (servers/rendering/renderer_rd/shaders/effects/tonemap.glsl, MIT: Stephen Hill's fit).
vec3 tonemap_aces(vec3 color, float white) {
    const float exposure_bias = 1.8;
    const float A = 0.0245786;
    const float B = 0.000090537;
    const float C = 0.983729;
    const float D = 0.432951;
    const float E = 0.238081;
    const mat3 rgb_to_rrt = mat3(vec3(0.59719 * exposure_bias, 0.35458 * exposure_bias, 0.04823 * exposure_bias),
            vec3(0.07600 * exposure_bias, 0.90834 * exposure_bias, 0.01566 * exposure_bias),
            vec3(0.02840 * exposure_bias, 0.13383 * exposure_bias, 0.83777 * exposure_bias));
    const mat3 odt_to_rgb = mat3(vec3(1.60475, -0.53108, -0.07367), vec3(-0.10208, 1.10813, -0.00605), vec3(-0.00327, -0.07276, 1.07602));
    color *= rgb_to_rrt;
    vec3 color_tonemapped = (color * (color + A) - B) / (color * (C * color + D) + E);
    color_tonemapped *= odt_to_rgb;
    white *= exposure_bias;
    float white_tonemapped = (white * (white + A) - B) / (white * (C * white + D) + E);
    return color_tonemapped / white_tonemapped;
}

vec3 linear_to_srgb(vec3 color) {
    color = clamp(color, vec3(0.0), vec3(1.0));
    return mix(1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, 12.92 * color, lessThan(color, vec3(0.0031308)));
}

// A linear colour as the target stores it: tonemapped, sRGB-encoded, graded (look.x: a LUT is bound). Godot's order.
vec3 display_color(vec3 color) {
    color = linear_to_srgb(tonemap_aces(max(color, vec3(0.0)), 1.0));
    if (look.x != 0.0) {
        color = textureLod(sampler3D(lut_tex, lut_smp), color, 0.0).rgb;
    }
    return color;
}
)glsl";

// The sky pass (after wgpu_util.hpp's FULLSCREEN_VERTEX, at the far depth: where no page is): the sky along the pixel's
// ray, into the fog's colour towards the horizon (where the farthest pages are fogged).
inline constexpr const char SKY_FRAGMENT[] = R"glsl(
layout(location = 0) in vec2 v_ndc;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 dir = normalize(forward.xyz + v_ndc.x * view_right.xyz + v_ndc.y * view_up.xyz);
    float horizon = 1.0 - smoothstep(0.0, 0.1, abs(dir.y));
    vec3 sky = sky_color(dir);
    vec3 color = mix(sky, fog_color(dir, sky), max(FOG_SKY_AFFECT, horizon));
    out_color = vec4(display_color(color), 1.0);
}
)glsl";

// The text to splice after the declarations and SkyGLSL::glsl(): the scenes' fog light (sRGB 0.72, 0.79, 0.9) linear.
inline ov::str glsl() {
    return ov::format(GLSL, SkyGLSL::linear_vec3(ov::Color(0.72f, 0.79f, 0.9f)));
}

} // namespace ov::dawn::LookGLSL
