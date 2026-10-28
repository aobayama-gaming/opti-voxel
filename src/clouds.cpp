#include "clouds.hpp"

#include "glsl_uniforms.hpp"
#include "gpu_shader.hpp"
#include "shade_frame_glsl.hpp"

#include "ov/clock.hpp"
#include "ov/math.hpp"

#include <cmath>

namespace {

// The noises (Schneider's, after sebh's TileableVolumeNoise): every one repeats over the texture, so a tile wraps. The
// shape: R Perlin-Worley (a Perlin fbm remapped above a Worley fbm: billows), G B A Worley fbms at 4, 8 and 16 cells per
// tile; the detail: R G B Worley fbms at 4, 8 and 16 cells. Worley values are inverted (1 at a cell's point).
constexpr const char GLSL_PASS_NOISE[] = R"glsl(#version 450
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 0, rgba16f) uniform restrict writeonly image3D noise;
layout(push_constant, std430) uniform Push { int size; int detail; } push;

uvec3 noise_hash(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// A lattice cell's random point in [0, 1)^3, the cell wrapped to n per axis; seed separates the noises.
vec3 noise_point(ivec3 c, int n, uint seed) {
    uvec3 w = uvec3((c % n + n) % n);
    return vec3(noise_hash(w + uvec3(seed * 1013u, seed * 7919u, seed * 104729u)) >> 8u) * (1.0 / 16777216.0);
}

// Worley at n cells per tile, inverted: 1 at a cell's point, 0 a cell away.
float noise_worley(vec3 p, int n, uint seed) {
    vec3 x = p * float(n);
    ivec3 c = ivec3(floor(x));
    vec3 f = x - vec3(c);
    float d = 1.0;
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                ivec3 o = ivec3(i, j, k);
                vec3 v = vec3(o) + noise_point(c + o, n, seed) - f;
                d = min(d, dot(v, v));
            }
        }
    }
    return 1.0 - sqrt(d);
}

float noise_worley_fbm(vec3 p, int n, uint seed) {
    return noise_worley(p, n, seed) * 0.625 + noise_worley(p, 2 * n, seed + 1u) * 0.25 + noise_worley(p, 4 * n, seed + 2u) * 0.125;
}

// Gradient (Perlin) noise at n cells per tile, in about [-1, 1].
float noise_perlin(vec3 p, int n, uint seed) {
    vec3 x = p * float(n);
    ivec3 c = ivec3(floor(x));
    vec3 f = x - vec3(c);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float v[8];
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, k >> 2);
        vec3 g = normalize(noise_point(c + o, n, seed) * 2.0 - 1.0 + vec3(1e-4));
        v[k] = dot(g, f - vec3(o));
    }
    float x00 = mix(v[0], v[1], u.x);
    float x10 = mix(v[2], v[3], u.x);
    float x01 = mix(v[4], v[5], u.x);
    float x11 = mix(v[6], v[7], u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// A Perlin fbm of 5 octaves from n cells per tile, in [0, 1].
float noise_perlin_fbm(vec3 p, int n, uint seed) {
    float sum = 0.0;
    float amp = 0.5;
    for (int o = 0; o < 5; o++) {
        sum += amp * noise_perlin(p, n << o, seed + uint(o));
        amp *= 0.5;
    }
    return clamp(0.5 + 2.0 * sum, 0.0, 1.0); // sum: a standard deviation of ~0.11
}

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(id, ivec3(push.size)))) {
        return;
    }
    vec3 p = (vec3(id) + 0.5) / float(push.size);
    vec4 v;
    if (push.detail != 0) {
        v = vec4(noise_worley_fbm(p, 4, 40u), noise_worley_fbm(p, 8, 50u), noise_worley_fbm(p, 16, 60u), 1.0);
    } else {
        float worley = noise_worley_fbm(p, 4, 1u);
        float perlin = noise_perlin_fbm(p, 4, 10u);
        v = vec4(worley + perlin * (1.0 - worley), worley, noise_worley_fbm(p, 8, 20u), noise_worley_fbm(p, 16, 30u));
    }
    imageStore(noise, id, v);
}
)glsl";

// The grid's directions: plain GLSL that Godot's shading language takes too.
constexpr const char GLSL_GRID[] = R"glsl(
const float CLOUD_PI = 3.14159265358979;
const int CLOUD_GRID_W = $GRID_W;
const int CLOUD_GRID_H = $GRID_H;

// The camera-centred grid's direction at uv in [0, 1]^2, round the up (t1, up, t2): the azimuth along u, the elevation
// along v (linear, straight down at 0).
vec3 cloud_grid_dir(vec2 uv, vec3 up, vec3 t1, vec3 t2) {
    float phi = 2.0 * CLOUD_PI * uv.x;
    float theta = (uv.y - 0.5) * CLOUD_PI;
    return (t1 * cos(phi) + t2 * sin(phi)) * cos(theta) + up * sin(theta);
}

// cloud_grid_dir's inverse, v kept half a texel inside (no wrap from pole to pole).
vec2 cloud_grid_uv(vec3 dir, vec3 up, vec3 t1, vec3 t2) {
    float u = fract(atan(dot(dir, t2), dot(dir, t1)) / (2.0 * CLOUD_PI) + 1.0);
    float v = asin(clamp(dot(dir, up), -1.0, 1.0)) / CLOUD_PI + 0.5;
    return vec2(u, clamp(v, 0.5 / float(CLOUD_GRID_H), 1.0 - 0.5 / float(CLOUD_GRID_H)));
}
)glsl";

// The density and the layer's spans (the passes'). $SHAPE and $DETAIL are the noises' sampler expressions.
constexpr const char GLSL_DENSITY[] = R"glsl(

const int OCTAVES = $OCTAVES;
const float MS_SCATTERING = $MS_SCATTERING;
const float MS_EXTINCTION = $MS_EXTINCTION;
const float MS_ECCENTRICITY = $MS_ECCENTRICITY;

// The sun through an optical depth tau of cloud, with Wrenninge's octaves (Clouds::MS_*: octave o scatters
// MS_SCATTERING^o of the light through MS_EXTINCTION^o of the extinction), over the octaves' sum: the ground under a
// thick cloud keeps the light its octaves let through.
float cloud_octave_transmittance(float tau) {
    float sum = 0.0;
    float norm = 0.0;
    float a = 1.0;
    float b = 1.0;
    for (int o = 0; o < OCTAVES; o++) {
        sum += a * exp(-b * tau);
        norm += a;
        a *= MS_SCATTERING;
        b *= MS_EXTINCTION;
    }
    return sum / norm;
}

float cloud_remap(float v, float a, float b, float c, float d) {
    return c + (v - a) / max(b - a, 1e-5) * (d - c);
}

// A render point's coordinates in a noise of period_vox voxels (whole tiles per period: the texture repeats), carried by
// the wind (its offset wraps at the weather period, a multiple of the others).
vec3 cloud_coords(vec3 p, int period_vox) {
    return (shade_point(p, period_vox) + cloud_wind_offset_m) / shade_period_m(period_vox);
}

float cloud_height_fraction(float altitude) {
    return (altitude - cloud_bottom_m) / (cloud_top_m - cloud_bottom_m);
}

// Schneider's height gradients: stratus (type 0), stratocumulus (0.5), cumulus (1).
float cloud_gradient(float hf, float type) {
    vec4 stratus = vec4(0.02, 0.05, 0.09, 0.11);
    vec4 stratocumulus = vec4(0.02, 0.2, 0.48, 0.625);
    vec4 cumulus = vec4(0.01, 0.0625, 0.78, 1.0);
    vec4 g = stratus * (1.0 - clamp(type * 2.0, 0.0, 1.0)) + stratocumulus * (1.0 - abs(type - 0.5) * 2.0) + cumulus * clamp(type - 0.5, 0.0, 1.0) * 2.0;
    return smoothstep(g.x, g.y, hf) - smoothstep(g.z, g.w, hf);
}

// The weather at p: the coverage in [0, 1] (cloud_cover of the sky past its threshold) and the cloud type, from the
// shape noise read at the weather scale.
vec2 cloud_weather(vec3 p) {
    vec4 w = textureLod($SHAPE, cloud_coords(p, cloud_weather_period_vox), 0.0);
    float edge = $COVER_LOW + ($COVER_HIGH - $COVER_LOW) * (1.0 - clamp(cloud_cover, 0.0, 1.0));
    float coverage = smoothstep(edge - 0.2, edge + 0.2, w.r) * clamp(cloud_cover * 4.0, 0.0, 1.0);
    return vec2(coverage, smoothstep(0.25, 0.75, w.g));
}

// The base shape at p (height fraction hf, weather w), without the detail: Schneider's coverage remap.
float cloud_base(vec3 p, float hf, vec2 w) {
    if (hf <= 0.0 || hf >= 1.0 || w.x <= 0.0) {
        return 0.0;
    }
    vec4 n = textureLod($SHAPE, cloud_coords(p, cloud_shape_period_vox), 0.0);
    float fbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    float base = cloud_remap(n.r, fbm - 1.0, 1.0, 0.0, 1.0) * cloud_gradient(hf, w.y);
    return clamp(cloud_remap(base, 1.0 - w.x, 1.0, 0.0, 1.0) * w.x, 0.0, 1.0);
}

// The base eroded by the detail noise (billows at the bottom, wisps at the top).
float cloud_erode(vec3 p, float hf, float base) {
    vec3 d = textureLod($DETAIL, cloud_coords(p, cloud_detail_period_vox), 0.0).rgb;
    float fbm = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    float m = mix(fbm, 1.0 - fbm, clamp(hf * 10.0, 0.0, 1.0));
    return clamp(cloud_remap(base, m * 0.35, 1.0, 0.0, 1.0), 0.0, 1.0);
}

// Where the ray p + t dir (t >= 0) is in the layer: [seg.x, seg.y] and [seg.z, seg.w] (either empty), the part below its
// bottom cut out; at most reach, and not past the base where the ray meets it.
bool cloud_segments(vec3 p, vec3 dir, float reach, out vec4 seg) {
    seg = vec4(0.0);
    vec2 top;
    if (!base_level_span(p, dir, cloud_top_m, top)) {
        return false;
    }
    float lo = max(top.x, 0.0);
    float hi = min(top.y, reach);
    vec2 ground;
    if (base_level_span(p, dir, 0.0, ground) && ground.x > 0.0) {
        hi = min(hi, ground.x);
    }
    if (!(hi > lo)) {
        return false;
    }
    vec2 bottom;
    if (!base_level_span(p, dir, cloud_bottom_m, bottom)) {
        seg = vec4(lo, hi, hi, hi);
        return true;
    }
    seg = vec4(lo, min(hi, max(bottom.x, lo)), max(lo, min(bottom.y, hi)), hi);
    return seg.y > seg.x || seg.w > seg.z;
}

// The n-th of count points spread evenly (offset jitter in [0, 1)) over the two segments: its distance along the ray.
float cloud_segment_t(vec4 seg, float i, float count, float jitter) {
    float a = seg.y - seg.x;
    float s = (i + jitter) / count * (a + seg.w - seg.z);
    return s < a ? seg.x + s : seg.z + (s - a);
}


)glsl";

// The host's lookups in the grid and the shadow map (after GLSL_GRID).
constexpr const char GLSL_LOOKUP[] = R"glsl(
// From the camera to dist along dir (unit): the clouds' light (rgb, for a sun of illuminance 1, the air in front of them
// included) and transmittance (a). A cloud that starts past dist is not there; one that dist enters, partly.
vec4 cloud_at(vec3 camera, vec3 dir, float dist) {
    vec3 up = base_up(camera);
    vec3 t1;
    vec3 t2;
    up_frame(up, t1, t2);
    vec2 uv = cloud_grid_uv(dir, up, t1, t2);
    vec4 l = textureLod(cloud_light, uv, 0.0);
    vec2 d = textureLod(cloud_depth, uv, 0.0).xy * 1000.0;
    float w = clamp((dist - d.x) / max(2.0 * (d.y - d.x), 1.0), 0.0, 1.0);
    return vec4(l.rgb * w, mix(1.0, l.a, w));
}

// The sun's transmittance through the layer at p, under it: the shadow map's (none outside it, faded over its last
// tenth).
float cloud_sun_shadow(vec3 p) {
    vec3 d = p - cloud_shadow_origin_m;
    vec2 uv = vec2(dot(d, cloud_shadow_u), dot(d, cloud_shadow_v)) / cloud_shadow_extent_m;
    float edge = max(abs(uv.x), abs(uv.y));
    if (edge >= 0.5) {
        return 1.0;
    }
    float t = textureLod(cloud_shadow_map, uv + 0.5, 0.0).r;
    return mix(t, 1.0, smoothstep(0.45, 0.5, edge));
}
)glsl";

// The pass's head: the uniform block (the base's, the atmosphere's, the layer's and the frame's fields, block_bytes'
// order), the shading frame's host functions from it, the samplers.
constexpr const char GLSL_PASS_HEAD[] = R"glsl(#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std140) uniform CloudBlock {
$MEMBERS};
layout(set = 0, binding = 1, rgba16f) uniform restrict writeonly image2D out_light;
layout(set = 0, binding = 2, rgba16f) uniform restrict writeonly image2D out_depth;
layout(set = 0, binding = 3) uniform texture2D history_light;
layout(set = 0, binding = 4) uniform texture2D history_depth;
layout(set = 0, binding = 5) uniform texture3D shape;
layout(set = 0, binding = 6) uniform texture3D detail;
layout(set = 0, binding = 7) uniform texture2D atmo_volume_texture;
layout(set = 0, binding = 8) uniform texture2D atmo_multiscatter_texture;
layout(set = 0, binding = 9) uniform sampler linear;
layout(set = 0, binding = 10, rgba16f) uniform restrict writeonly image2D out_shadow;

ivec3 shade_origin_voxel() {
    return ivec3(cloud_origin_x, cloud_origin_y, cloud_origin_z);
}

float shade_voxel_m() {
    return cloud_voxel_m;
}
)glsl";

// The march: one invocation per grid texel.
constexpr const char GLSL_PASS_MARCH[] = R"glsl(
const int MAX_STEPS = $MAX_STEPS;
const int MIN_STEPS = $MIN_STEPS;
const int STEPS_PER_THICKNESS = $STEPS_PER_THICKNESS;
const int LIGHT_STEPS = $LIGHT_STEPS;
const float BLEND = $BLEND;

float cloud_hg(float c, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * CLOUD_PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// A dual-lobe Henyey-Greenstein phase (forward 0.8, back -0.3), its eccentricities times k (Wrenninge's octaves).
float cloud_phase(float c, float k) {
    return mix(cloud_hg(c, -0.3 * k), cloud_hg(c, 0.8 * k), 0.7);
}

// The optical depth towards the sun from p, in the base shape (no detail): LIGHT_STEPS steps from 1% of the layer's
// thickness, growing 1.8 times each (Frostbite's progressive shadow samples), 0.4 thicknesses in all; the weather of p.
float cloud_light_depth(vec3 p, vec3 sun, vec2 w) {
    float ds = (cloud_top_m - cloud_bottom_m) * 0.01;
    float t = 0.0;
    float depth = 0.0;
    for (int j = 0; j < LIGHT_STEPS; j++) {
        vec3 x = p + sun * (t + 0.5 * ds);
        depth += cloud_base(x, cloud_height_fraction(base_altitude(x)), w) * ds;
        t += ds;
        ds *= 1.8;
    }
    return depth * cloud_extinction;
}

// Interleaved gradient noise (Jimenez): a texel's jitter, decorrelated between neighbours.
float cloud_ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// The clouds along dir from the camera: their light (rgb, for a sun of illuminance 1, the air in front of them included)
// and transmittance (a); their front and mean distances in front_mean (kilometres; 0 where there is none: no light to
// hide, and a filtered texel at a cloud's edge keeps the cloud's distances, halved).
vec4 cloud_march(vec3 cam, vec3 dir, float jitter, out vec2 front_mean) {
    front_mean = vec2(0.0);
    float front_t = -1.0;
    vec4 seg;
    if (cloud_cover <= 0.0 || !cloud_segments(cam, dir, cloud_reach_m, seg)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    float len = (seg.y - seg.x) + (seg.w - seg.z);
    float thickness = cloud_top_m - cloud_bottom_m;
    int n = clamp(int(ceil(len * float(STEPS_PER_THICKNESS) / thickness)), MIN_STEPS, MAX_STEPS);
    float dt = len / float(n);
    float t_first = cloud_segment_t(seg, 0.0, 1.0, 0.0);
    float t_last = seg.w > seg.z ? seg.w : seg.y;
    // The sun through the air at the clouds (its ends; linear between).
    vec3 sun_a = atmo_transmittance(cam + dir * t_first, cloud_sun, ATMO_SUN_STEPS);
    vec3 sun_b = atmo_transmittance(cam + dir * t_last, cloud_sun, ATMO_SUN_STEPS);
    float cos_sun = dot(dir, cloud_sun);
    float phases[OCTAVES];
    float k = 1.0;
    for (int o = 0; o < OCTAVES; o++) {
        phases[o] = cloud_phase(cos_sun, k);
        k *= MS_ECCENTRICITY;
    }
    vec3 light = vec3(0.0);
    float trans = 1.0;
    float weighted_t = 0.0;
    for (int i = 0; i < n; i++) {
        float t = cloud_segment_t(seg, float(i), float(n), jitter);
        vec3 x = cam + dir * t;
        float h = base_altitude(x);
        float hf = cloud_height_fraction(h);
        vec2 w = cloud_weather(x);
        float base = cloud_base(x, hf, w);
        if (base <= 0.0) {
            continue;
        }
        float density = cloud_erode(x, hf, base);
        if (density <= 0.0) {
            continue;
        }
        front_t = front_t < 0.0 ? t : front_t;
        float sigma = density * cloud_extinction;
        // The sun: Wrenninge's octaves (octave o: the scattering times MS_SCATTERING^o, the light's extinction times
        // MS_EXTINCTION^o, the phase's eccentricity times MS_ECCENTRICITY^o). The sky: the atmosphere's isotropic light
        // there, dimmer lower in the layer (Frostbite's gradient).
        float tau = cloud_light_depth(x, cloud_sun, w);
        float sun_term = 0.0;
        float a = 1.0;
        float b = 1.0;
        for (int o = 0; o < OCTAVES; o++) {
            sun_term += a * phases[o] * exp(-b * tau);
            a *= MS_SCATTERING;
            b *= MS_EXTINCTION;
        }
        vec3 sun = mix(sun_a, sun_b, clamp((t - t_first) / max(t_last - t_first, 1.0), 0.0, 1.0)) * sun_term;
        vec3 sky = atmo_ms(max(h, 0.0), dot(base_up(x), cloud_sun)) * mix(0.4, 1.0, hf);
        float step_trans = exp(-sigma * dt);
        light += trans * cloud_albedo * (sun + sky) * (1.0 - step_trans);
        weighted_t += t * trans * (1.0 - step_trans);
        trans *= step_trans;
        if (trans < 0.01) {
            break;
        }
    }
    if (trans > 0.999) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    front_mean = 1e-3 * vec2(front_t, weighted_t / (1.0 - trans));
    // The air in front of the clouds (at their mean distance) dims them and adds its own light where they hide the sky.
    vec4 air = atmo_at(cam, dir, 1e3 * front_mean.y);
    return vec4(light * air.a + air.rgb * (1.0 - trans), trans);
}

void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= CLOUD_GRID_W || id.y >= CLOUD_GRID_H) {
        return;
    }
    vec2 uv = (vec2(id) + 0.5) / vec2(CLOUD_GRID_W, CLOUD_GRID_H);
    vec3 up = base_up(cloud_camera_m);
    vec3 t1;
    vec3 t2;
    up_frame(up, t1, t2);
    vec3 dir = cloud_grid_dir(uv, up, t1, t2);

    // The last frame's value at this direction: the point at its cloud's mean distance, seen from the last camera.
    vec4 old_light = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 old_depth = vec4(1.0);
    bool fresh = cloud_reset != 0;
    if (!fresh) {
        vec3 old_up = base_up(cloud_prev_camera_m);
        vec3 o1;
        vec3 o2;
        up_frame(old_up, o1, o2);
        vec2 old_uv = cloud_grid_uv(dir, old_up, o1, o2);
        float mean = textureLod(sampler2D(history_depth, linear), old_uv, 0.0).y * 1000.0;
        if (mean > 0.0) {
            old_uv = cloud_grid_uv(normalize(cloud_camera_m + dir * mean - cloud_prev_camera_m), old_up, o1, o2);
        }
        old_light = textureLod(sampler2D(history_light, linear), old_uv, 0.0);
        old_depth = textureLod(sampler2D(history_depth, linear), old_uv, 0.0);
    }

    // One texel of each 2 x 2 block per frame, in the order 0, 3, 1, 2.
    int order[4] = int[4](0, 3, 1, 2);
    if (fresh || (id.x & 1) + 2 * (id.y & 1) == order[cloud_frame & 3]) {
        float jitter = fract(cloud_ign(vec2(id)) + 0.618034 * float(cloud_frame & 63));
        vec2 front_mean;
        vec4 l = cloud_march(cloud_camera_m, dir, jitter, front_mean);
        vec4 d = vec4(front_mean, 0.0, 0.0);
        if (!fresh) {
            l = mix(old_light, l, BLEND);
            d = mix(old_depth, d, BLEND);
        }
        imageStore(out_light, id, l);
        imageStore(out_depth, id, d);
    } else {
        imageStore(out_light, id, old_light);
        imageStore(out_depth, id, old_depth);
    }
}
)glsl";

// The shadow map: one invocation per texel marches its line's column through the layer, from where the line leaves the
// base towards the sun (anywhere on the line where it misses the base).
constexpr const char GLSL_PASS_SHADOW[] = R"glsl(
const int SHADOW_SIZE = $SHADOW_SIZE;
const int SHADOW_STEPS = $SHADOW_STEPS;

void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= SHADOW_SIZE || id.y >= SHADOW_SIZE) {
        return;
    }
    vec2 f = (vec2(id) + 0.5) / float(SHADOW_SIZE);
    vec3 p = cloud_shadow_origin_m + (cloud_shadow_u * f.x + cloud_shadow_v * f.y) * cloud_shadow_extent_m;
    vec2 ground;
    if (base_level_span(p, cloud_sun, 0.0, ground) && ground.y < 1e29) {
        p += cloud_sun * ground.y;
    }
    float trans = 1.0;
    vec4 seg;
    if (cloud_cover > 0.0 && cloud_segments(p, cloud_sun, cloud_shadow_reach_m, seg)) {
        float len = (seg.y - seg.x) + (seg.w - seg.z);
        float depth = 0.0;
        for (int i = 0; i < SHADOW_STEPS; i++) {
            vec3 x = p + cloud_sun * cloud_segment_t(seg, float(i), float(SHADOW_STEPS), 0.5);
            depth += cloud_base(x, cloud_height_fraction(base_altitude(x)), cloud_weather(x));
        }
        trans = cloud_octave_transmittance(depth * cloud_extinction * len / float(SHADOW_STEPS));
    }
    imageStore(out_shadow, id, vec4(trans, 0.0, 0.0, 1.0));
}
)glsl";

// The weather's cover thresholds on the shape noise's R at the weather scale: cover 0 at COVER_HIGH, 1 at COVER_LOW
// (about its 5% and 95% quantiles, measured on a numpy mirror of the noise pass).
constexpr float COVER_LOW = 0.52f;
constexpr float COVER_HIGH = 0.94f;

ov::str substitute(const ov::str &p_glsl) {
    using C = Clouds;
    return ov::replace_all(p_glsl,
            { { "$GRID_W", ov::itos(C::GRID_W) }, { "$GRID_H", ov::itos(C::GRID_H) }, { "$MAX_STEPS", ov::itos(C::MAX_STEPS) },
                    { "$MIN_STEPS", ov::itos(C::MIN_STEPS) }, { "$STEPS_PER_THICKNESS", ov::itos(C::STEPS_PER_THICKNESS) },
                    { "$LIGHT_STEPS", ov::itos(C::LIGHT_STEPS) }, { "$SHADOW_STEPS", ov::itos(C::SHADOW_STEPS) },
                    { "$SHADOW_SIZE", ov::itos(C::SHADOW_SIZE) }, { "$OCTAVES", ov::itos(C::OCTAVES) },
                    { "$BLEND", ov::format("%.3f", double(C::BLEND)) }, { "$MS_SCATTERING", ov::format("%.3f", double(C::MS_SCATTERING)) },
                    { "$MS_EXTINCTION", ov::format("%.3f", double(C::MS_EXTINCTION)) }, { "$MS_ECCENTRICITY", ov::format("%.3f", double(C::MS_ECCENTRICITY)) }, { "$COVER_LOW", ov::format("%.3f", double(COVER_LOW)) },
                    { "$COVER_HIGH", ov::format("%.3f", double(COVER_HIGH)) } });
}

ov::str density_glsl(const ov::str &p_shape, const ov::str &p_detail) {
    return ov::replace_all(substitute(GLSL_DENSITY), { { "$SHAPE", p_shape }, { "$DETAIL", p_detail } });
}

// The block's fields: the base's, the atmosphere's (the lookup's uniforms), the layer's and the frame's.
template <class V>
void visit_block(V &&p_v, const BaseSurface::Render &p_base, const Atmosphere::HostConstants &p_atmosphere, float p_atmo_range_m,
        const CloudLayer &p_layer, const Clouds::Frame &p_frame) {
    p_base.visit(p_v);
    p_atmosphere.visit(p_v);
    p_v("atmo_range_m", p_atmo_range_m);
    p_layer.visit(p_v);
    p_frame.visit(p_v);
}

// A pass of the grid's bindings (GLSL_PASS_HEAD) with p_main after the base, the shading frame, the atmosphere's lookups,
// the grid and the density.
ov::str pass_glsl(const char *p_main) {
    ov::str members;
    visit_block(GLSLUniforms::Declarations{ members }, BaseSurface::Render(), Atmosphere::HostConstants(), 0.0f, CloudLayer(), Clouds::Frame());
    return ov::replace_all(ov::str(GLSL_PASS_HEAD), { { "$MEMBERS", members } }) + BaseSurface::glsl_render() + ShadeFrame::glsl() +
            Atmosphere::glsl_lookup("sampler2D(atmo_volume_texture, linear)", "sampler2D(atmo_multiscatter_texture, linear)") + substitute(GLSL_GRID) +
            density_glsl("sampler3D(shape, linear)", "sampler3D(detail, linear)") + substitute(p_main);
}

// The largest power of two of LOD0 voxels at most p_m metres (at least 1), within the shading frame's range.
int period_vox(double p_m, double p_voxel_m) {
    const double e = std::floor(std::log2(ov::max(p_m / p_voxel_m, 1.0)));
    return 1 << int(ov::min(e, double(ShadeFrame::MAX_SHIFT)));
}

} // namespace

CloudLayer CloudLayer::for_base(double p_radius_m, double p_relief_m, double p_voxel_m, double p_scale) {
    CloudLayer l;
    const double bottoms = RELIEF_BOTTOMS * p_relief_m / double(l.bottom_m);
    const double s = p_scale > 0.0 ? p_scale : ov::min(1.0, ov::max(p_radius_m / AtmosphereMedium::EARTH_RADIUS_M, bottoms));
    // The grid's march knows nothing of the terrain: no cloud may reach into it.
    const double bottom = ov::max(double(l.bottom_m) * s, RELIEF_BOTTOMS * p_relief_m);
    l.top_m = float(bottom + double(l.top_m - l.bottom_m) * s);
    l.bottom_m = float(bottom);
    l.extinction = float(l.extinction / s);
    l.shadow_reach_m = float(l.shadow_reach_m * s);
    l.shadow_extent_m = float(l.shadow_extent_m * s);
    l.shape_period_vox = period_vox(EARTH_SHAPE_PERIOD_M * s, p_voxel_m);
    l.detail_period_vox = period_vox(EARTH_DETAIL_PERIOD_M * s, p_voxel_m);
    l.weather_period_vox = period_vox(EARTH_WEATHER_PERIOD_M * s, p_voxel_m);
    return l;
}

ov::str Clouds::glsl_host_uniforms() {
    ov::str out;
    CloudLayer().visit(GLSLUniforms::Declarations{ out, "uniform " });
    return out;
}

ov::str Clouds::glsl_lookup() {
    return substitute(GLSL_GRID) + GLSL_LOOKUP;
}

ov::str Clouds::glsl_pass_noise() {
    return GLSL_PASS_NOISE;
}

ov::str Clouds::glsl_pass_march() {
    return pass_glsl(GLSL_PASS_MARCH);
}

ov::str Clouds::glsl_pass_shadow() {
    return pass_glsl(GLSL_PASS_SHADOW);
}

ov::bytes Clouds::block_bytes(const BaseSurface::Render &p_base, const Atmosphere::HostConstants &p_atmosphere, float p_atmo_range_m,
        const CloudLayer &p_layer, const Frame &p_frame) {
    ov::bytes out;
    GLSLUniforms::Std140 w{ out };
    visit_block(w, p_base, p_atmosphere, p_atmo_range_m, p_layer, p_frame);
    w.pad();
    return out;
}

void Clouds::start(ov::gpu::Device *p_device, const Atmosphere *p_atmosphere, const BaseSurface &p_base, double p_voxel_m, double p_relief_m,
        double p_scale, TexturesFn p_textures) {
    device = p_device;
    atmosphere = p_atmosphere;
    textures_fn = std::move(p_textures);
    layer_params = CloudLayer::for_base(p_base.curvature_radius_m(p_voxel_m), p_relief_m, p_voxel_m, p_scale);
    curvature_m = Atmosphere::ms_radius_m(p_base, p_voxel_m);
    frame_params.voxel_m = float(p_voxel_m);
    noise_dirty = true;
    device->post([this]() { _rt_start(); });
}

void Clouds::set_weather(float p_cover, const Float3 &p_wind_mps) {
    frame_params.cover = ov::clamp(p_cover, 0.0f, 1.0f);
    wind_mps = p_wind_mps;
}

Clouds::Frame Clouds::make_frame(const CloudLayer &p_layer, double p_curvature_m, const Float3 &p_camera_m, const Float3 &p_sun,
        const BaseSurface::Render &p_base) {
    Frame f;
    f.camera_m = p_camera_m;
    f.sun = p_sun;
    const double altitude = double(p_base.altitude(p_camera_m));
    f.reach_m = float(Atmosphere::horizon_m(altitude, p_curvature_m) + Atmosphere::horizon_m(double(p_layer.top_m), p_curvature_m));
    // The shadow map's axes perpendicular to the sun, its origin snapped to its texels (it doesn't crawl as the camera
    // moves), the camera at its centre.
    const Float3 axis = std::abs(p_sun.y) < 0.9f ? Float3{ 0.0f, 1.0f, 0.0f } : Float3{ 1.0f, 0.0f, 0.0f };
    const Float3 u = cross(axis, p_sun).normalized();
    const Float3 v = cross(p_sun, u);
    f.shadow_u = u;
    f.shadow_v = v;
    f.shadow_extent_m = float(ov::max(double(p_layer.shadow_extent_m), 4.0 * altitude));
    const double texel = double(f.shadow_extent_m) / SHADOW_SIZE;
    const double half = 0.5 * double(f.shadow_extent_m);
    const double cu = std::floor(double(dot(p_camera_m, u)) / texel) * texel - half;
    const double cv = std::floor(double(dot(p_camera_m, v)) / texel) * texel - half;
    const double along = double(dot(p_camera_m, p_sun));
    f.shadow_origin_m = Float3{ float(u.x * cu + v.x * cv + p_sun.x * along), float(u.y * cu + v.y * cv + p_sun.y * along),
        float(u.z * cu + v.z * cv + p_sun.z * along) };
    return f;
}

void Clouds::frame(const Float3 &p_camera_m, const Float3 &p_sun, const BaseSurface::Render &p_base, const VoxelPos &p_origin_voxel) {
    const uint64_t now = ov::now_usec();
    const double dt = last_usec != 0 ? double(now - last_usec) * 1e-6 : 0.0;
    last_usec = now;
    const double period = double(layer_params.weather_period_vox) * double(frame_params.voxel_m);
    const double wind[3] = { wind_mps.x, wind_mps.y, wind_mps.z };
    for (int a = 0; a < 3; a++) {
        wind_offset[a] = std::fmod(wind_offset[a] + wind[a] * ov::min(dt, 0.25), period);
    }
    // A jump of more than the layer's thickness (a teleport) leaves nothing to reproject.
    const Float3 moved = p_camera_m - frame_params.camera_m;
    const float thickness = layer_params.top_m - layer_params.bottom_m;
    const bool reset = !has_camera || dot(moved, moved) > thickness * thickness;
    Frame f = make_frame(layer_params, curvature_m, p_camera_m, p_sun, p_base);
    f.prev_camera_m = has_camera ? frame_params.camera_m : p_camera_m;
    f.wind_offset_m = Float3{ float(wind_offset[0]), float(wind_offset[1]), float(wind_offset[2]) };
    f.cover = frame_params.cover;
    f.frame = frame_params.frame + 1;
    f.reset = reset ? 1 : 0;
    f.voxel_m = frame_params.voxel_m;
    const ov::ivec3 lo = WorldCoord::lo32_i(p_origin_voxel);
    f.origin_x = lo.x;
    f.origin_y = lo.y;
    f.origin_z = lo.z;
    has_camera = true;
    frame_params = f;
    const ov::bytes data = block_bytes(p_base, atmosphere->host_constants(), atmosphere->range_m(), layer_params, f);
    const bool noise = noise_dirty;
    noise_dirty = false;
    device->post([this, data, noise]() { _rt_frame(data, noise); });
}

void Clouds::shift(const Float3 &p_shift_m) {
    frame_params.camera_m = frame_params.camera_m - p_shift_m;
}

void Clouds::destroy() {
    device->post([this]() { _rt_destroy(); });
}

void Clouds::_rt_start() {
    using ov::gpu::Binding;
    using ov::gpu::TextureFormat;
    const uint32_t noise_usage = ov::gpu::TEXTURE_STORAGE | ov::gpu::TEXTURE_SAMPLED;
    shape = device->create_texture_3d(TextureFormat::RGBA16_FLOAT, SHAPE_SIZE, SHAPE_SIZE, SHAPE_SIZE, noise_usage);
    detail = device->create_texture_3d(TextureFormat::RGBA16_FLOAT, DETAIL_SIZE, DETAIL_SIZE, DETAIL_SIZE, noise_usage);
    const uint32_t grid_usage = ov::gpu::TEXTURE_STORAGE | ov::gpu::TEXTURE_SAMPLED | ov::gpu::TEXTURE_COPY_SRC;
    light = device->create_texture(TextureFormat::RGBA16_FLOAT, GRID_W, GRID_H, grid_usage);
    depth = device->create_texture(TextureFormat::RGBA16_FLOAT, GRID_W, GRID_H, grid_usage);
    history_light = device->create_texture(TextureFormat::RGBA16_FLOAT, GRID_W, GRID_H, ov::gpu::TEXTURE_SAMPLED | ov::gpu::TEXTURE_COPY_DST);
    history_depth = device->create_texture(TextureFormat::RGBA16_FLOAT, GRID_W, GRID_H, ov::gpu::TEXTURE_SAMPLED | ov::gpu::TEXTURE_COPY_DST);
    shadow = device->create_texture(TextureFormat::RGBA16_FLOAT, SHADOW_SIZE, SHADOW_SIZE, ov::gpu::TEXTURE_STORAGE | ov::gpu::TEXTURE_SAMPLED);
    block = device->create_buffer(ov::gpu::BufferKind::UNIFORM,
            block_bytes(BaseSurface::Render(), Atmosphere::HostConstants(), 0.0f, CloudLayer(), Frame()).size());
    pipe_noise = ov::gpu::compile(*device, "clouds_noise", glsl_pass_noise());
    pipe_march = ov::gpu::compile(*device, "clouds_march", glsl_pass_march());
    pipe_shadow = ov::gpu::compile(*device, "clouds_shadow", glsl_pass_shadow());
    if (pipe_noise.is_valid()) {
        set_shape = device->create_bind_set(pipe_noise, { Binding::storage_image(0, shape) });
        set_detail = device->create_bind_set(pipe_noise, { Binding::storage_image(0, detail) });
    }
    // Both passes declare every binding of GLSL_PASS_HEAD.
    const std::vector<Binding> bindings = { Binding(0, ov::gpu::BufferKind::UNIFORM, block), Binding::storage_image(1, light), Binding::storage_image(2, depth),
        Binding::texture(3, history_light), Binding::texture(4, history_depth), Binding::texture(5, shape), Binding::texture(6, detail),
        Binding::texture(7, atmosphere->get_volume()), Binding::texture(8, atmosphere->get_multiscatter()), Binding::linear_sampler(9),
        Binding::storage_image(10, shadow) };
    if (pipe_march.is_valid() && pipe_shadow.is_valid()) {
        set_march = device->create_bind_set(pipe_march, bindings);
        set_shadow = device->create_bind_set(pipe_shadow, bindings);
    }
    if (textures_fn) {
        textures_fn(light, depth, shape, detail, shadow);
    }
}

void Clouds::_rt_frame(const ov::bytes &p_block, bool p_noise) {
    if (!set_march.is_valid() || !set_shadow.is_valid() || !set_shape.is_valid()) {
        return; // a pass that didn't compile (reported once by compile)
    }
    if (p_noise) {
        const int32_t shape_push[2] = { SHAPE_SIZE, 0 };
        const int32_t detail_push[2] = { DETAIL_SIZE, 1 };
        device->dispatch(pipe_noise, set_shape, shape_push, sizeof(shape_push), SHAPE_SIZE / 4, SHAPE_SIZE / 4, SHAPE_SIZE / 4);
        device->dispatch(pipe_noise, set_detail, detail_push, sizeof(detail_push), DETAIL_SIZE / 4, DETAIL_SIZE / 4, DETAIL_SIZE / 4);
        device->timestamp("clouds_noise");
    }
    device->update_buffer(block, 0, p_block.size(), p_block.data());
    device->dispatch(pipe_shadow, set_shadow, nullptr, 0, (SHADOW_SIZE + 7) / 8, (SHADOW_SIZE + 7) / 8, 1);
    device->dispatch(pipe_march, set_march, nullptr, 0, (GRID_W + 7) / 8, (GRID_H + 7) / 8, 1);
    device->copy_texture(light, history_light, 0, 0, 0, 0, GRID_W, GRID_H);
    device->copy_texture(depth, history_depth, 0, 0, 0, 0, GRID_W, GRID_H);
    device->timestamp("clouds");
}

void Clouds::_rt_destroy() {
    if (textures_fn) {
        textures_fn(ov::gpu::Texture(), ov::gpu::Texture(), ov::gpu::Texture(), ov::gpu::Texture(), ov::gpu::Texture());
    }
    const ov::gpu::Handle handles[] = { set_march, set_shadow, set_shape, set_detail, pipe_march, pipe_shadow, pipe_noise, block, light, depth, history_light,
        history_depth, shadow, shape, detail };
    for (const ov::gpu::Handle h : handles) {
        if (h.is_valid()) {
            device->free(h);
        }
    }
    delete this;
}
