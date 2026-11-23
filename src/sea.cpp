#include "sea.hpp"

#include "exact_noise.hpp"
#include "glsl_uniforms.hpp"

#include <cmath>

namespace {

// The shore (page shader): sand over the band, wet just above the level.
constexpr const char GLSL_SHORE[] = R"glsl(
// The shore on the ground at altitude h above the base (metres): the sand blended in over [level - below, level + above]
// (soft edges), darker and smoother just above the waterline (wet). Neutral without a sea.
void sea_shore(float h, inout vec3 albedo, inout float roughness) {
    if (sea_on == 0) {
        return;
    }
    float a = h - sea_level_m;
    float lo = max(sea_shore_below_m * $SHORE_EDGE, $EDGE_MIN_M);
    float hi = max(sea_shore_above_m * $SHORE_EDGE, $EDGE_MIN_M);
    float sand = smoothstep(-sea_shore_below_m - lo, -sea_shore_below_m, a) * (1.0 - smoothstep(sea_shore_above_m - hi, sea_shore_above_m, a));
    float wet = 1.0 - smoothstep(0.0, $WET_M, a);
    albedo = mix(albedo, sea_sand, sand) * mix(1.0, $WET_DARKEN, wet);
    roughness = mix(roughness, $WET_ROUGH, wet);
}
)glsl";

// The water: its waves, light and the pass's colour (sea_shade).
constexpr const char GLSL_WATER[] = R"glsl(
const float SEA_PI = 3.14159265;

// The water's extinction per metre and channel: a column of sea_clarity_m metres passes sea_shallow.
vec3 sea_extinction() {
    return -log(clamp(sea_shallow, vec3(1e-3), vec3(1.0))) / sea_clarity_m;
}

float sea_fresnel(float c) {
    float m = 1.0 - c;
    return $F0 + (1.0 - $F0) * m * m * m * m * m;
}

// The waves' normal at the surface point p (render space; up: the base's up there) at time, and in r_var the slope
// variance of the waves under WAVE_FADE_PX pixels of footprint metres (the glint's roughness): Gerstner waves (GPU Gems 1,
// ch. 1: the slope along each heading, the crests sharpened) in the base's tangent frame, read in the shading frame.
vec3 sea_wave_normal(vec3 p, vec3 up, float footprint, float time, out float r_var) {
    vec3 t1;
    vec3 t2;
    up_frame(up, t1, t2);
    float period_m;
    vec3 s = shade_frame(p, $LONGEST, period_m);
    vec3 n = up;
    r_var = 0.0;
    float len = $LONGEST;
    float slope = $SLOPE / float($WAVES);
    for (int i = 0; i < $WAVES; i++) {
        float heading = 0.7 + 2.39996 * float(i);
        vec3 k = shade_wave((t1 * cos(heading) + t2 * sin(heading)) * (2.0 * SEA_PI / len), period_m);
        float kl = length(k);
        float phase = dot(k, s) - sqrt(9.81 * kl) * time;
        float keep = smoothstep(0.5 * $FADE, $FADE, len / max(footprint, 1e-6));
        n -= (k / kl) * (slope * keep * cos(phase)) + up * (0.5 * slope * keep * sin(phase));
        r_var += (1.0 - keep) * 0.5 * slope * slope;
        len *= $RATIO;
    }
    return normalize(n);
}

// GGX (Trowbridge-Reitz) with the height-correlated Smith visibility and Schlick's Fresnel, times n.l: the sun's glint
// for roughness alpha.
float sea_ggx(vec3 n, vec3 v, vec3 l, float alpha) {
    vec3 h = normalize(v + l);
    float nl = max(dot(n, l), 0.0);
    float nv = max(dot(n, v), 1e-4);
    float nh = max(dot(n, h), 0.0);
    float a2 = alpha * alpha;
    float d = nh * nh * (a2 - 1.0) + 1.0;
    float vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2) + 1e-6);
    return a2 / (SEA_PI * d * d) * vis * sea_fresnel(max(dot(v, h), 0.0)) * nl;
}

// The sky's light from dir seen from the camera: the atmosphere's radiance (no sun disk: the glint is the sun) with its
// clouds over it, else the host's sky colour (sky_rgb).
vec3 sea_sky_light(vec3 camera, vec3 dir, vec3 sun_dir, vec3 sun_light, vec3 sky_rgb) {
    if (!atmo_enabled) {
        return sky_rgb;
    }
    vec3 light = atmo_sky_radiance(camera, dir, sun_dir, 2.0) * sun_light;
    if (cloud_enabled) {
        vec4 c = cloud_at(camera, dir, 1e30);
        light = light * c.a + c.rgb * sun_light;
    }
    return light;
}

// The sun's light at p: through the air and under the clouds' shadow map.
vec3 sea_sun_at(vec3 p, vec3 sun_dir, vec3 sun_light) {
    vec3 sun = sun_light;
    if (atmo_enabled) {
        sun *= atmo_transmittance(p, sun_dir, ATMO_SUN_STEPS);
    }
    if (cloud_enabled) {
        sun *= cloud_sun_shadow(p);
    }
    return sun;
}

float sea_hash(vec3 c) {
    c = fract(c * vec3(0.1031, 0.1030, 0.0973));
    c += dot(c, c.yxz + 33.33);
    return fract((c.x + c.y) * c.z);
}

// Value noise in [0, 1] of FOAM_SCALE_M features at p drifting with time, its lattice wrapped on the shading frame's
// period (the same at every render origin).
float sea_foam_noise(vec3 p, float time) {
    float period_m;
    vec3 s = shade_frame(p, $FOAM_SCALE, period_m);
    vec3 cells = shade_cells(vec3(1.0 / $FOAM_SCALE), period_m);
    vec3 x = s * (cells / period_m) + vec3(1.0, 0.0, 0.7) * (time * $FOAM_SPEED);
    vec3 c = floor(x);
    vec3 f = x - c;
    f = f * f * (3.0 - 2.0 * f);
    vec3 a = shade_wrap(c, cells);
    vec3 b = shade_wrap_next(a, cells);
    float x00 = mix(sea_hash(a), sea_hash(vec3(b.x, a.y, a.z)), f.x);
    float x10 = mix(sea_hash(vec3(a.x, b.y, a.z)), sea_hash(vec3(b.x, b.y, a.z)), f.x);
    float x01 = mix(sea_hash(vec3(a.x, a.y, b.z)), sea_hash(vec3(b.x, a.y, b.z)), f.x);
    float x11 = mix(sea_hash(vec3(a.x, b.y, b.z)), sea_hash(b), f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// The camera under water: the water from it to dist absorbs and scatters (the fog under the sea), lit by the sun and the
// sky through the water over the camera.
vec3 sea_under(vec3 camera, float dist, vec3 scene, vec3 ext, vec3 sun_dir, vec3 sun_light, vec3 sky_rgb) {
    vec3 up = base_up(camera);
    vec3 down = exp(-ext * max(sea_level_m - base_altitude(camera), 0.0));
    vec3 light = (sea_sun_at(camera, sun_dir, sun_light) * max(dot(up, sun_dir), 0.0) / SEA_PI + sea_sky_light(camera, up, sun_dir, sun_light, sky_rgb)) * down;
    vec3 tw = exp(-ext * dist);
    return scene * tw + sea_deep * (1.0 - tw) * light;
}

// The camera above the water, the ray in it over [t.x, min(scene_dist, t.y)]: the surface's reflection and glint, the
// water's colour over the scene behind, foam where it is shallow, the air in front.
vec3 sea_above(vec3 camera, vec3 dir, vec2 t, float scene_dist, vec3 scene, vec3 ext, float pixel_rad, float time, vec3 sun_dir,
        vec3 sun_light, vec3 sky_rgb) {
    vec3 p = camera + dir * t.x;
    vec3 up = base_up(p);
    float cos_up = max(-dot(dir, up), 0.0);
    float slope_var;
    vec3 n = sea_wave_normal(p, up, t.x * pixel_rad / max(cos_up, 0.05), time, slope_var);
    vec3 v = -dir;
    n = dot(n, v) < 0.01 ? normalize(n + v * (0.01 - dot(n, v))) : n; // a wave facing away
    float f = sea_fresnel(max(dot(n, v), 0.0));
    vec3 r = reflect(dir, n);
    r = normalize(r - up * (2.0 * min(dot(r, up), 0.0))); // never below the horizon
    vec3 sun = sea_sun_at(p, sun_dir, sun_light);
    vec3 lit = sun * max(dot(up, sun_dir), 0.0) / SEA_PI + sea_sky_light(camera, up, sun_dir, sun_light, sky_rgb);
    float thick = min(scene_dist, t.y) - t.x;
    vec3 tw = exp(-ext * thick);
    float alpha = sqrt($RMIN * $RMIN + 2.0 * slope_var);
    vec3 light = f * sea_sky_light(camera, r, sun_dir, sun_light, sky_rgb) + sun * sea_ggx(n, v, sun_dir, alpha) + (1.0 - f) * sea_deep * (1.0 - tw) * lit;
    vec3 trans = (1.0 - f) * tw;
    float foam = (1.0 - smoothstep(0.0, $FOAM_DEPTH, thick * cos_up)) * smoothstep(0.45, 0.75, sea_foam_noise(p, time));
    light = mix(light, lit * $FOAM_ALBEDO, foam);
    trans *= 1.0 - foam;
    // The air (and its clouds) in front of the surface over the water's light; the scene behind already has its own.
    if (atmo_enabled) {
        vec4 air = atmo_at(camera, dir, t.x);
        if (cloud_enabled) {
            vec4 clouds = cloud_at(camera, dir, t.x);
            air = vec4(air.rgb * clouds.a + clouds.rgb, air.a * clouds.a);
        }
        light = light * air.a + air.rgb * sun_light * (1.0 - trans);
    }
    return scene * trans + light;
}

vec3 sea_shade(vec3 camera, vec3 dir, float scene_dist, vec3 scene, float pixel_rad, float time, vec3 sun_dir, vec3 sun_light, vec3 sky_rgb,
        out bool hit) {
    vec2 t;
    hit = sea_on != 0 && base_level_span_k(camera, dir, sea_k, t) && t.y > 0.0 && t.x < scene_dist;
    if (!hit) {
        return scene;
    }
    vec3 ext = sea_extinction();
    if (t.x <= 0.0) {
        return sea_under(camera, min(scene_dist, t.y), scene, ext, sun_dir, sun_light, sky_rgb);
    }
    return sea_above(camera, dir, t, scene_dist, scene, ext, pixel_rad, time, sun_dir, sun_light, sky_rgb);
}
)glsl";

ov::str glsl_f(float p_f) {
    return ExactNoise::glsl_float(p_f);
}

} // namespace

Sea::Render Sea::render(const WorldFile::Sea &p_sea, const BaseSurface &p_base, const VoxelPos &p_origin, double p_voxel_m) {
    Render r;
    if (!p_sea.on || p_base.kind() == BaseSurface::Kind::TORUS) {
        return r;
    }
    r.on = 1;
    r.k = float(p_base.level_k(p_origin, p_voxel_m, p_sea.level_m));
    r.level_m = float(p_sea.level_m);
    r.deep = p_sea.deep;
    r.shallow = p_sea.shallow;
    r.clarity_m = float(p_sea.clarity_m);
    r.shore_below_m = float(p_sea.shore_below_m);
    r.shore_above_m = float(p_sea.shore_above_m);
    r.sand = p_sea.sand;
    return r;
}

double Sea::height_at(const BaseSurface &p_base, const WorldPos &p_pos, double p_voxel_m, double p_level_m) {
    const UpAxis up = p_base.up_dir(p_pos.voxel);
    const double along = double(p_pos.frac.x) * up[0] + double(p_pos.frac.y) * up[1] + double(p_pos.frac.z) * up[2];
    return (p_base.altitude(p_pos.voxel) + along) * p_voxel_m - p_level_m;
}

ov::str Sea::glsl_uniforms() {
    ov::str out;
    Render().visit(GLSLUniforms::Declarations{ out, "uniform " });
    return out;
}

ov::str Sea::glsl_shore() {
    return ov::replace_all(ov::str(GLSL_SHORE),
            { { "$SHORE_EDGE", glsl_f(SHORE_EDGE) }, { "$EDGE_MIN_M", glsl_f(SHORE_EDGE_MIN_M) }, { "$WET_M", glsl_f(WET_M) },
                    { "$WET_DARKEN", glsl_f(WET_DARKEN) }, { "$WET_ROUGH", glsl_f(WET_ROUGHNESS) } });
}

ov::str Sea::glsl_water() {
    return ov::replace_all(ov::str(GLSL_WATER),
            { { "$WAVES", ov::itos(WAVES) }, { "$LONGEST", glsl_f(WAVE_LONGEST_M) }, { "$RATIO", glsl_f(WAVE_RATIO) },
                    { "$SLOPE", glsl_f(WAVE_SLOPE) }, { "$FADE", glsl_f(WAVE_FADE_PX) }, { "$RMIN", glsl_f(ROUGHNESS_MIN) },
                    { "$F0", glsl_f(F0) }, { "$FOAM_DEPTH", glsl_f(FOAM_M) }, { "$FOAM_SCALE", glsl_f(FOAM_SCALE_M) },
                    { "$FOAM_SPEED", glsl_f(FOAM_SPEED) }, { "$FOAM_ALBEDO", glsl_f(FOAM_ALBEDO) } });
}
