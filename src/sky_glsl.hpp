#pragma once

// The sky's colour, shared by both adapters: Godot's sky shader (VoxelSky, which draws its cloud layer over it) and the
// Dawn viewer's sky pass and fog (dawn_page_host.cpp). Godot's PhysicalSkyMaterial (scene/resources/sky_material.cpp,
// MIT) at its default parameters, as functions of the view direction and the sun. Its colours are `source_color`
// uniforms there, sRGB that Godot turns linear: glsl() substitutes them linear. light_color is linear too (LIGHT0_COLOR).
// Plain GLSL that Godot's shading language also takes (no #define, no arrays, no PI: Godot declares it).

#include "ov/color.hpp"
#include "ov/format.hpp"

namespace SkyGLSL {

// A GLSL vec3 of an sRGB colour, made linear.
inline ov::str linear_vec3(const ov::Color &p_srgb) {
    const ov::Color c = p_srgb.srgb_to_linear();
    return ov::format("vec3(%.6f, %.6f, %.6f)", c.r, c.g, c.b);
}

// %s: the Rayleigh, Mie and ground colours (glsl()).
inline constexpr const char GLSL[] = R"glsl(
const float SKY_PI = 3.14159265358979;
const float SKY_RAYLEIGH = 2.0;
const vec3 SKY_RAYLEIGH_COLOR = %s;
const float SKY_MIE = 0.005;
const float SKY_MIE_ECCENTRICITY = 0.8;
const vec3 SKY_MIE_COLOR = %s;
const float SKY_TURBIDITY = 10.0;
const float SKY_SUN_DISK_SCALE = 1.0;
const vec3 SKY_GROUND_COLOR = %s;
// Optical length at zenith for molecules and aerosols.
const float SKY_RAYLEIGH_ZENITH_SIZE = 8.4e3;
const float SKY_MIE_ZENITH_SIZE = 1.25e3;

float sky_henyey_greenstein(float cos_theta, float g) {
    const float k = 0.0795774715459;
    return k * (1.0 - g * g) / (pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}

// 1 with the sun over the horizon, fading below it.
float sky_sun_fade(vec3 sun_dir) {
    return 1.0 - clamp(1.0 - exp(sun_dir.y), 0.0, 1.0);
}

// The scattering coefficients for a sun towards sun_dir (normalized, y up).
void sky_betas(vec3 sun_dir, out vec3 rayleigh_beta, out vec3 mie_beta) {
    float sun_fade = sky_sun_fade(sun_dir);
    float rayleigh_coefficient = SKY_RAYLEIGH - (1.0 * (1.0 - sun_fade));
    rayleigh_beta = rayleigh_coefficient * SKY_RAYLEIGH_COLOR * 0.0001;
    mie_beta = SKY_TURBIDITY * SKY_MIE * SKY_MIE_COLOR * 0.000434; // Preetham
}

// The sun's energy over the horizon: light_energy at the zenith, 0 below the horizon.
float sky_sun_energy(vec3 sun_dir, float light_energy) {
    float zenith_angle = clamp(sun_dir.y, -1.0, 1.0);
    return max(0.0, 1.0 - exp(-((SKY_PI * 0.5) - acos(zenith_angle)))) * light_energy;
}

// Light extinction along a ray at that cosine of the zenith angle, for the given coefficients.
vec3 sky_extinction(float cos_zenith, vec3 rayleigh_beta, vec3 mie_beta) {
    float zenith = acos(max(0.0, cos_zenith));
    float optical_mass = 1.0 / (cos(zenith) + 0.15 * pow(93.885 - degrees(zenith), -1.253));
    return exp(-(rayleigh_beta * SKY_RAYLEIGH_ZENITH_SIZE * optical_mass + mie_beta * SKY_MIE_ZENITH_SIZE * optical_mass));
}

// The sky's linear colour along eyedir (normalized, y up) under a sun towards sun_dir of light_color and light_energy,
// its disk light_size radians across: the scattering, the ground under the horizon and the sun's disk, times exposure
// (the material's energy_multiplier).
vec3 physical_sky(vec3 eyedir, vec3 sun_dir, vec3 light_color, float light_energy, float light_size, float exposure) {
    float zenith_angle = clamp(sun_dir.y, -1.0, 1.0);
    float sun_energy = sky_sun_energy(sun_dir, light_energy);
    float sun_fade = sky_sun_fade(sun_dir);
    vec3 rayleigh_beta;
    vec3 mie_beta;
    sky_betas(sun_dir, rayleigh_beta, mie_beta);

    vec3 extinction = sky_extinction(eyedir.y, rayleigh_beta, mie_beta);

    float cos_theta = dot(eyedir, sun_dir);
    float rayleigh_phase = (3.0 / (16.0 * SKY_PI)) * (1.0 + pow(cos_theta * 0.5 + 0.5, 2.0));
    vec3 beta_r_theta = rayleigh_beta * rayleigh_phase;
    float mie_phase = sky_henyey_greenstein(cos_theta, SKY_MIE_ECCENTRICITY);
    vec3 beta_m_theta = mie_beta * mie_phase;

    vec3 scatter = sun_energy * ((beta_r_theta + beta_m_theta) / (rayleigh_beta + mie_beta));
    vec3 lin = pow(scatter * (1.0 - extinction), vec3(1.5));
    lin *= mix(vec3(1.0), pow(scatter * extinction, vec3(0.5)), clamp(pow(1.0 - zenith_angle, 5.0), 0.0, 1.0));
    lin *= mix(SKY_GROUND_COLOR, vec3(1.0), smoothstep(-0.1, 0.1, eyedir.y));

    float sun_cos = cos(light_size * SKY_SUN_DISK_SCALE);
    float sun_cos_half = cos(light_size * SKY_SUN_DISK_SCALE * 0.5);
    float sun_disk = smoothstep(sun_cos, sun_cos_half, cos_theta);
    vec3 l0 = (sun_energy * extinction) * sun_disk * light_color;

    return pow(lin + l0, vec3(1.0 / (1.2 + (1.2 * sun_fade)))) * exposure;
}
)glsl";

// The text to splice: Godot's default colours (sRGB) substituted linear.
inline ov::str glsl() {
    return ov::format(GLSL, linear_vec3(ov::Color(0.3f, 0.405f, 0.6f)), linear_vec3(ov::Color(0.69f, 0.729f, 0.812f)),
            linear_vec3(ov::Color(0.1f, 0.07f, 0.034f)));
}

} // namespace SkyGLSL
