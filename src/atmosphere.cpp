#include "atmosphere.hpp"

#include "glsl_uniforms.hpp"
#include "gpu_shader.hpp"
#include "shade_frame_glsl.hpp"

#include "ov/math.hpp"

#include <cmath>

namespace {

// The medium and the march: plain GLSL that Godot's shading language takes too.
constexpr const char GLSL_MARCH[] = R"glsl(
const float ATMO_PI = 3.14159265358979;
const int ATMO_SLICES = $SLICES;
const int ATMO_VOLUME_W = $VOLUME_W;
const int ATMO_VOLUME_H = $VOLUME_H;
const int ATMO_MS_SIZE = $MS_SIZE;

// Step i of n over [start, start + len], spaced quadratically from start (finest there, where a climbing ray's air is
// densest): its midpoint t and length dt.
void atmo_quad_step(int i, int n, float start, float len, out float t, out float dt) {
    float a = float(i) / float(n);
    float b = float(i + 1) / float(n);
    t = start + 0.25 * (a + b) * (a + b) * len;
    dt = (b * b - a * a) * len;
}
const int ATMO_SUN_STEPS = $SUN_STEPS;

// The medium at altitude h above the base (clamped at the base: no denser below it): Rayleigh and Mie scattering and
// the extinction, 1/m.
void atmo_medium(float h, out vec3 rayleigh, out float mie, out vec3 extinction) {
    float a = max(h, 0.0);
    rayleigh = atmo_rayleigh * exp(-a / atmo_rayleigh_height_m);
    float m = exp(-a / atmo_mie_height_m);
    mie = atmo_mie * m;
    float ozone = max(0.0, 1.0 - abs(a - atmo_ozone_centre_m) / (0.5 * atmo_ozone_width_m));
    extinction = rayleigh + vec3((atmo_mie + atmo_mie_absorption) * m) + atmo_ozone * ozone;
}

float atmo_rayleigh_phase(float c) {
    return 3.0 / (16.0 * ATMO_PI) * (1.0 + c * c);
}

// Cornette-Shanks (Hillaire's).
float atmo_mie_phase(float c) {
    float g = atmo_mie_g;
    float k = 3.0 / (8.0 * ATMO_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + c * c) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}

// The part of the ray p + t dir (t >= 0) inside the atmosphere: [seg.x, seg.y], ground when it ends on the base (a ray
// starting under the base ignores it); at most atmo_max_path_m long (a ray along a plane never leaves it). False when
// the ray misses the atmosphere.
bool atmo_segment(vec3 p, vec3 dir, out vec2 seg, out bool ground) {
    ground = false;
    vec2 top;
    if (!base_level_span(p, dir, atmo_top_m, top) || top.y <= 0.0) {
        seg = vec2(0.0);
        return false;
    }
    seg = vec2(max(top.x, 0.0), top.y);
    vec2 below;
    if (base_level_span(p, dir, 0.0, below) && below.x > seg.x && below.x < seg.y) {
        seg.y = below.x;
        ground = true;
    }
    seg.y = min(seg.y, seg.x + atmo_max_path_m);
    return seg.y > seg.x;
}

// The transmittance from p to the atmosphere's edge towards dir: 0 where the base is in the way. n samples spaced
// quadratically from p (the medium is densest there on a ray that climbs).
vec3 atmo_transmittance(vec3 p, vec3 dir, int n) {
    vec2 seg;
    bool ground;
    if (!atmo_segment(p, dir, seg, ground)) {
        return vec3(1.0);
    }
    if (ground) {
        return vec3(0.0);
    }
    vec3 depth = vec3(0.0);
    for (int i = 0; i < n; i++) {
        float t;
        float dt;
        atmo_quad_step(i, n, seg.x, seg.y - seg.x, t, dt);
        vec3 r;
        float m;
        vec3 e;
        atmo_medium(base_altitude(p + dir * t), r, m, e);
        depth += e * dt;
    }
    return exp(-depth);
}

// One step of a view ray from p along dir: the light scattered towards p over the dt metres around t, added to light
// through the transmittance so far, which it then takes (Hillaire's energy-conserving step). The sun towards sun of
// illuminance 1, at cos_sun from the ray.
void atmo_step(vec3 p, vec3 dir, vec3 sun, float cos_sun, float t, float dt, inout vec3 light, inout vec3 trans) {
    vec3 x = p + dir * t;
    float h = base_altitude(x);
    vec3 r;
    float m;
    vec3 e;
    atmo_medium(h, r, m, e);
    vec3 s = atmo_transmittance(x, sun, ATMO_SUN_STEPS) * (r * atmo_rayleigh_phase(cos_sun) + vec3(m * atmo_mie_phase(cos_sun)));
    s += atmo_ms(h, dot(base_up(x), sun)) * (r + vec3(m));
    vec3 td = exp(-e * dt);
    light += trans * (s - s * td) / max(e, vec3(1e-20));
    trans *= td;
}

// The light the base bounces at x (on it) towards a ray that ends there (Lambert, ground_albedo).
vec3 atmo_ground(vec3 x, vec3 sun) {
    vec3 up = base_up(x);
    vec3 lifted = x + up * (1e-3 * atmo_top_m);
    return atmo_transmittance(lifted, sun, ATMO_SUN_STEPS) * max(dot(up, sun), 0.0) * atmo_ground_albedo / ATMO_PI;
}

// The light scattered towards p along dir by the whole atmosphere (n steps, quadratic from the segment's start) and
// the mean transmittance through it; the base's bounce where the ray ends on it.
vec4 atmo_integrate(vec3 p, vec3 dir, vec3 sun, int n) {
    vec2 seg;
    bool ground;
    vec3 light = vec3(0.0);
    vec3 trans = vec3(1.0);
    if (atmo_segment(p, dir, seg, ground)) {
        float cos_sun = dot(dir, sun);
        for (int i = 0; i < n; i++) {
            float t;
            float dt;
            atmo_quad_step(i, n, seg.x, seg.y - seg.x, t, dt);
            atmo_step(p, dir, sun, cos_sun, t, dt, light, trans);
        }
        if (ground) {
            light += trans * atmo_ground(p + dir * seg.y, sun);
        }
    }
    return vec4(light, (trans.x + trans.y + trans.z) / 3.0);
}

// The camera volume's direction at texel coordinates uv in [0, 1]^2, round the up (t1, up, t2): the azimuth along u,
// the elevation along v, squared from the horizon (Hillaire's latitude: its texels finest at the horizon).
vec3 atmo_volume_dir(vec2 uv, vec3 up, vec3 t1, vec3 t2) {
    float phi = 2.0 * ATMO_PI * uv.x;
    float x = 2.0 * uv.y - 1.0;
    float theta = sign(x) * x * x * (0.5 * ATMO_PI);
    return (t1 * cos(phi) + t2 * sin(phi)) * cos(theta) + up * sin(theta);
}

// atmo_volume_dir's inverse.
vec2 atmo_volume_uv(vec3 dir, vec3 up, vec3 t1, vec3 t2) {
    float theta = asin(clamp(dot(dir, up), -1.0, 1.0));
    float u = fract(atan(dot(dir, t2), dot(dir, t1)) / (2.0 * ATMO_PI) + 1.0);
    float x = sign(theta) * sqrt(abs(theta) / (0.5 * ATMO_PI));
    return vec2(u, 0.5 + 0.5 * x);
}

// The end of slice s < ATMO_SLICES - 1 (metres from the camera): range * ((s + 1) / (ATMO_SLICES - 1))^2.
float atmo_slice_end(int s, float range) {
    float f = float(s + 1) / float(ATMO_SLICES - 1);
    return range * f * f;
}
)glsl";

// The host's multiple scattering, read from the table's sampler (before GLSL_MARCH).
constexpr const char GLSL_MS_SAMPLER[] = R"glsl(
vec3 atmo_ms(float h, float sun_height) {
    vec2 uv = vec2(0.5 + 0.5 * sun_height, clamp(h / atmo_top_m, 0.0, 1.0));
    return textureLod(atmo_multiscatter, uv, 0.0).rgb;
}
)glsl";

// The host's lookups in the camera volume (after GLSL_MARCH).
constexpr const char GLSL_LOOKUP[] = R"glsl(
// The steps of a sky marched per pixel (atmo_integrate): from above the atmosphere, where the volume's texels are too
// coarse for its limb.
const int ATMO_SPACE_STEPS = $SPACE_STEPS;
const float ATMO_SUN_DISK = $SUN_DISK;

// Slice s of the volume at uv: linear across texels, not across slices (a slice's rows clamped to its band).
vec4 atmo_slice(vec2 uv, int s) {
    float h = float(ATMO_VOLUME_H);
    float y = clamp(uv.y * h, 0.5, h - 0.5);
    return textureLod(atmo_volume, vec2(uv.x, (float(s) * h + y) / (h * float(ATMO_SLICES))), 0.0);
}

vec2 atmo_uv(vec3 camera, vec3 dir) {
    vec3 up = base_up(camera);
    vec3 t1;
    vec3 t2;
    up_frame(up, t1, t2);
    return atmo_volume_uv(dir, up, t1, t2);
}

// From the camera to dist along dir (unit): the in-scattered light (rgb, for a sun of illuminance 1) and the mean
// transmittance (a). Past the range: the range's.
vec4 atmo_at(vec3 camera, vec3 dir, float dist) {
    vec2 uv = atmo_uv(camera, dir);
    float f = min(sqrt(max(dist, 0.0) / atmo_range_m) * float(ATMO_SLICES - 1) - 1.0, float(ATMO_SLICES - 2));
    int s = int(floor(f));
    vec4 a = s < 0 ? vec4(0.0, 0.0, 0.0, 1.0) : atmo_slice(uv, s);
    vec4 b = atmo_slice(uv, min(s + 1, ATMO_SLICES - 2));
    return mix(a, b, f - float(s));
}

// The whole ray: the sky.
vec4 atmo_sky(vec3 camera, vec3 dir) {
    return atmo_slice(atmo_uv(camera, dir), ATMO_SLICES - 1);
}

// The sky's light towards the camera along dir for a sun of illuminance 1 towards sun: the volume's sky slice, or from
// above the atmosphere a march per pixel (the volume's texels are too coarse for the limb); plus the sun's disk (its
// radius' cosine sun_cos) through the air.
vec3 atmo_sky_radiance(vec3 camera, vec3 dir, vec3 sun, float sun_cos) {
    vec3 light = base_altitude(camera) > atmo_top_m ? atmo_integrate(camera, dir, sun, ATMO_SPACE_STEPS).rgb : atmo_sky(camera, dir).rgb;
    if (dot(dir, sun) > sun_cos) {
        light += atmo_transmittance(camera, sun, ATMO_SUN_STEPS) * ATMO_SUN_DISK;
    }
    return light;
}
)glsl";

// The passes' common head: the uniform block (the base's, the medium's and the frame's fields, block_bytes' order).
constexpr const char GLSL_PASS_HEAD[] = R"glsl(#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std140) uniform AtmosphereBlock {
$MEMBERS};
)glsl";

// The multiple-scattering table (Hillaire's: second-order light with an isotropic phase over MS_DIRS^2 directions,
// summed as the geometric series 1 / (1 - f_ms)), on a sphere of atmo_ms_radius_m: x = sun height, y = altitude.
constexpr const char GLSL_PASS_MULTISCATTER[] = R"glsl(
layout(set = 0, binding = 1, rgba16f) uniform restrict writeonly image2D table;
const int MS_STEPS = $MS_STEPS;
const int MS_DIRS = $MS_DIRS;

// The ray o + t d (o inside the sphere of radius r round the origin): its far root; and, ground, its near root when the
// ray meets the sphere ahead.
float ms_far(vec3 o, vec3 d, float r) {
    float b = dot(o, d);
    float c = dot(o, o) - r * r;
    return -b + sqrt(max(b * b - c, 0.0));
}

bool ms_near(vec3 o, vec3 d, float r, out float t) {
    float b = dot(o, d);
    float c = dot(o, o) - r * r;
    float disc = b * b - c;
    t = -b - sqrt(max(disc, 0.0));
    return disc >= 0.0 && t > 0.0;
}

vec3 ms_transmittance(vec3 x, vec3 sun) {
    float t_ground;
    if (ms_near(x, sun, atmo_ms_radius_m, t_ground)) {
        return vec3(0.0);
    }
    float len = ms_far(x, sun, atmo_ms_radius_m + atmo_top_m);
    vec3 depth = vec3(0.0);
    for (int i = 0; i < ATMO_SUN_STEPS; i++) {
        float t;
        float dt;
        atmo_quad_step(i, ATMO_SUN_STEPS, 0.0, len, t, dt);
        vec3 r;
        float m;
        vec3 e;
        atmo_medium(length(x + sun * t) - atmo_ms_radius_m, r, m, e);
        depth += e * dt;
    }
    return exp(-depth);
}

void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= ATMO_MS_SIZE || id.y >= ATMO_MS_SIZE) {
        return;
    }
    float mu = (float(id.x) + 0.5) / float(ATMO_MS_SIZE) * 2.0 - 1.0;
    float h = (float(id.y) + 0.5) / float(ATMO_MS_SIZE) * atmo_top_m;
    vec3 x = vec3(0.0, atmo_ms_radius_m + h, 0.0);
    vec3 sun = vec3(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
    vec3 l2 = vec3(0.0);
    vec3 f = vec3(0.0);
    for (int i = 0; i < MS_DIRS; i++) {
        for (int j = 0; j < MS_DIRS; j++) {
            float c = 1.0 - 2.0 * (float(j) + 0.5) / float(MS_DIRS);
            float phi = 2.0 * ATMO_PI * (float(i) + 0.5) / float(MS_DIRS);
            float s = sqrt(max(1.0 - c * c, 0.0));
            vec3 d = vec3(s * cos(phi), c, s * sin(phi));
            float t_end;
            bool ground = ms_near(x, d, atmo_ms_radius_m, t_end);
            if (!ground) {
                t_end = ms_far(x, d, atmo_ms_radius_m + atmo_top_m);
            }
            float dt = t_end / float(MS_STEPS);
            vec3 trans = vec3(1.0);
            for (int k = 0; k < MS_STEPS; k++) {
                vec3 y = x + d * ((float(k) + 0.5) * dt);
                vec3 r;
                float m;
                vec3 e;
                atmo_medium(length(y) - atmo_ms_radius_m, r, m, e);
                vec3 scat = r + vec3(m);
                vec3 td = exp(-e * dt);
                vec3 integral = (vec3(1.0) - td) / max(e, vec3(1e-20));
                l2 += trans * ms_transmittance(y, sun) * scat * (1.0 / (4.0 * ATMO_PI)) * integral;
                f += trans * scat * integral;
                trans *= td;
            }
            if (ground) {
                vec3 g = x + d * t_end;
                vec3 up = normalize(g);
                l2 += trans * ms_transmittance(g + up * (1e-3 * atmo_top_m), sun) * max(dot(up, sun), 0.0) * atmo_ground_albedo / ATMO_PI;
            }
        }
    }
    float n = float(MS_DIRS * MS_DIRS);
    l2 /= n;
    f /= n;
    imageStore(table, id, vec4(l2 / max(vec3(1.0) - f, vec3(1e-3)), 1.0));
}
)glsl";

// The camera volume: one invocation per direction marches its column out to each slice's end.
constexpr const char GLSL_PASS_VOLUME[] = R"glsl(
layout(set = 0, binding = 1, rgba16f) uniform restrict writeonly image2D volume;
layout(set = 0, binding = 2, rgba16f) uniform restrict readonly image2D table;
const int FAR_STEPS = $FAR_STEPS;
const int TABLE_SIZE = $MS_SIZE;

// The multiple-scattering table, bilinear (its texel centres at (i + 0.5) / TABLE_SIZE).
vec3 atmo_ms(float h, float sun_height) {
    vec2 f = clamp(vec2(0.5 + 0.5 * sun_height, h / atmo_top_m) * float(TABLE_SIZE) - 0.5, vec2(0.0), vec2(float(TABLE_SIZE - 1)));
    ivec2 i = ivec2(floor(f));
    ivec2 j = min(i + 1, ivec2(TABLE_SIZE - 1));
    vec2 w = f - vec2(i);
    vec3 a = mix(imageLoad(table, i).rgb, imageLoad(table, ivec2(j.x, i.y)).rgb, w.x);
    vec3 b = mix(imageLoad(table, ivec2(i.x, j.y)).rgb, imageLoad(table, j).rgb, w.x);
    return mix(a, b, w.y);
}

/*GLSL_MARCH*/

void store(ivec2 id, int s, vec3 light, vec3 trans) {
    imageStore(volume, ivec2(id.x, s * ATMO_VOLUME_H + id.y), vec4(light, (trans.x + trans.y + trans.z) / 3.0));
}

void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= ATMO_VOLUME_W || id.y >= ATMO_VOLUME_H) {
        return;
    }
    vec3 up = base_up(atmo_camera_m);
    vec3 t1;
    vec3 t2;
    up_frame(up, t1, t2);
    vec3 dir = atmo_volume_dir((vec2(id) + 0.5) / vec2(ATMO_VOLUME_W, ATMO_VOLUME_H), up, t1, t2);
    float cos_sun = dot(dir, atmo_sun);
    vec2 seg;
    bool ground;
    bool inside = atmo_segment(atmo_camera_m, dir, seg, ground);
    vec3 light = vec3(0.0);
    vec3 trans = vec3(1.0);
    float from = 0.0;
    for (int s = 0; s < ATMO_SLICES - 1; s++) {
        float to = atmo_slice_end(s, atmo_range_m);
        float a = max(from, seg.x);
        float b = min(to, seg.y);
        if (inside && b > a) {
            atmo_step(atmo_camera_m, dir, atmo_sun, cos_sun, 0.5 * (a + b), b - a, light, trans);
        }
        store(id, s, light, trans);
        from = to;
    }
    float a = max(from, seg.x);
    if (inside && seg.y > a) {
        for (int i = 0; i < FAR_STEPS; i++) {
            float t;
            float dt;
            atmo_quad_step(i, FAR_STEPS, a, seg.y - a, t, dt);
            atmo_step(atmo_camera_m, dir, atmo_sun, cos_sun, t, dt, light, trans);
        }
    }
    if (inside && ground) {
        light += trans * atmo_ground(atmo_camera_m + dir * seg.y, atmo_sun);
    }
    store(id, ATMO_SLICES - 1, light, trans);
}
)glsl";

ov::str substitute(const ov::str &p_glsl) {
    using A = Atmosphere;
    return ov::replace_all(p_glsl,
            { { "$SLICES", ov::itos(A::SLICES) }, { "$VOLUME_W", ov::itos(A::VOLUME_W) }, { "$VOLUME_H", ov::itos(A::VOLUME_H) },
                    { "$MS_SIZE", ov::itos(A::MS_SIZE) }, { "$SUN_STEPS", ov::itos(A::SUN_STEPS) }, { "$FAR_STEPS", ov::itos(A::FAR_STEPS) },
                    { "$MS_STEPS", ov::itos(A::MS_STEPS) }, { "$MS_DIRS", ov::itos(A::MS_DIRS) }, { "$SPACE_STEPS", ov::itos(A::SPACE_STEPS) },
                    { "$SUN_DISK", ov::str(ov::format("%.1f", double(A::SUN_DISK))) } });
}

ov::str block_members() {
    ov::str members;
    GLSLUniforms::Declarations decl{ members };
    BaseSurface::Render().visit(decl);
    AtmosphereMedium().visit(decl);
    Atmosphere::Frame().visit(decl);
    return members;
}

ov::str pass_head() {
    return ov::replace_all(ov::str(GLSL_PASS_HEAD), { { "$MEMBERS", block_members() } }) + BaseSurface::glsl_render() + ShadeFrame::GLSL_UP_FRAME;
}

double horizon_m(double p_altitude_m, double p_radius_m) {
    const double a = ov::max(p_altitude_m, 0.0);
    return std::sqrt(a * (2.0 * p_radius_m + a));
}

} // namespace

AtmosphereMedium AtmosphereMedium::for_base(double p_radius_m, double p_relief_m, double p_scale) {
    const double tops = RELIEF_TOPS * p_relief_m / double(AtmosphereMedium().top_m);
    const double s = p_scale > 0.0 ? p_scale : ov::min(1.0, ov::max(p_radius_m / EARTH_RADIUS_M, tops));
    AtmosphereMedium m;
    const float h = float(s);
    const float k = float(1.0 / s);
    m.rayleigh = Float3{ m.rayleigh.x * k, m.rayleigh.y * k, m.rayleigh.z * k };
    m.rayleigh_height_m *= h;
    m.mie *= k;
    m.mie_absorption *= k;
    m.mie_height_m *= h;
    m.ozone = Float3{ m.ozone.x * k, m.ozone.y * k, m.ozone.z * k };
    m.ozone_centre_m *= h;
    m.ozone_width_m *= h;
    m.top_m *= h;
    return m;
}

ov::str Atmosphere::glsl_host_uniforms() {
    ov::str out;
    HostConstants().visit(GLSLUniforms::Declarations{ out, "uniform " });
    return out;
}

ov::str Atmosphere::glsl_march() {
    return substitute(GLSL_MARCH);
}

ov::str Atmosphere::glsl_lookup() {
    return substitute(ov::str(GLSL_MS_SAMPLER) + GLSL_MARCH + GLSL_LOOKUP);
}

ov::str Atmosphere::glsl_pass_multiscatter() {
    // The table is the multiple scattering: the march it shares has none to read.
    return substitute(pass_head() + "vec3 atmo_ms(float h, float sun_height) {\n    return vec3(0.0);\n}\n" + GLSL_MARCH + GLSL_PASS_MULTISCATTER);
}

ov::str Atmosphere::glsl_pass_volume() {
    return substitute(pass_head() + ov::replace_all(ov::str(GLSL_PASS_VOLUME), { { "/*GLSL_MARCH*/", ov::str(GLSL_MARCH) } }));
}

ov::bytes Atmosphere::block_bytes(const BaseSurface::Render &p_base, const AtmosphereMedium &p_medium, const Frame &p_frame) {
    ov::bytes out;
    GLSLUniforms::Std140 w{ out };
    p_base.visit(w);
    p_medium.visit(w);
    p_frame.visit(w);
    w.pad();
    return out;
}

Atmosphere::Frame Atmosphere::make_frame(const AtmosphereMedium &p_medium, double p_ms_radius_m, const Float3 &p_camera_m, const Float3 &p_sun,
        const BaseSurface::Render &p_base) {
    Frame f;
    f.camera_m = p_camera_m;
    f.sun = p_sun;
    const double top_horizon = horizon_m(double(p_medium.top_m), p_ms_radius_m);
    f.range_m = float(horizon_m(double(p_base.altitude(p_camera_m)), p_ms_radius_m) + ov::min(top_horizon, AERIAL_RANGE_M));
    f.max_path_m = float(2.0 * top_horizon);
    f.ms_radius_m = float(p_ms_radius_m);
    return f;
}

double Atmosphere::ms_radius_m(const BaseSurface &p_base, double p_voxel_m) {
    const double radius = p_base.curvature_radius_m(p_voxel_m);
    return std::isfinite(radius) ? radius : AtmosphereMedium::EARTH_RADIUS_M;
}

void Atmosphere::start(ov::gpu::Device *p_device, const BaseSurface &p_base, double p_voxel_m, double p_relief_m, double p_scale, TexturesFn p_textures) {
    device = p_device;
    textures_fn = std::move(p_textures);
    curvature_m = ms_radius_m(p_base, p_voxel_m);
    constants.medium = AtmosphereMedium::for_base(p_base.curvature_radius_m(p_voxel_m), p_relief_m, p_scale);
    // Constant for a world: no camera term.
    constants.max_path_m = make_frame(constants.medium, curvature_m, Float3(), Float3(), BaseSurface::Render()).max_path_m;
    multiscatter_dirty = true;
    device->post([this]() { _rt_start(); });
}

void Atmosphere::frame(const Float3 &p_camera_m, const Float3 &p_sun, const BaseSurface::Render &p_base) {
    const Frame f = make_frame(constants.medium, curvature_m, p_camera_m, p_sun, p_base);
    last_range_m = f.range_m;
    const ov::bytes data = block_bytes(p_base, constants.medium, f);
    const bool ms = multiscatter_dirty;
    multiscatter_dirty = false;
    device->post([this, data, ms]() { _rt_frame(data, ms); });
}

void Atmosphere::destroy() {
    device->post([this]() { _rt_destroy(); });
}

void Atmosphere::_rt_start() {
    using ov::gpu::Binding;
    const uint32_t usage = ov::gpu::TEXTURE_STORAGE | ov::gpu::TEXTURE_SAMPLED | ov::gpu::TEXTURE_COPY_SRC; // COPY_SRC: AtmosphereCheck reads it
    volume = device->create_texture(ov::gpu::TextureFormat::RGBA16_FLOAT, VOLUME_W, VOLUME_H * SLICES, usage);
    multiscatter = device->create_texture(ov::gpu::TextureFormat::RGBA16_FLOAT, MS_SIZE, MS_SIZE, usage);
    block = device->create_buffer(ov::gpu::BufferKind::UNIFORM, block_bytes(BaseSurface::Render(), constants.medium, Frame()).size());
    pipe_multiscatter = ov::gpu::compile(*device, "atmosphere_multiscatter", glsl_pass_multiscatter());
    pipe_volume = ov::gpu::compile(*device, "atmosphere_volume", glsl_pass_volume());
    if (pipe_multiscatter.is_valid()) {
        set_multiscatter = device->create_bind_set(pipe_multiscatter, { Binding(0, ov::gpu::BufferKind::UNIFORM, block), Binding::storage_image(1, multiscatter) });
    }
    if (pipe_volume.is_valid()) {
        set_volume = device->create_bind_set(pipe_volume,
                { Binding(0, ov::gpu::BufferKind::UNIFORM, block), Binding::storage_image(1, volume), Binding::storage_image(2, multiscatter) });
    }
    if (textures_fn) {
        textures_fn(volume, multiscatter);
    }
}

void Atmosphere::_rt_frame(const ov::bytes &p_block, bool p_multiscatter) {
    if (!set_volume.is_valid() || !set_multiscatter.is_valid()) {
        return; // a pass that didn't compile (reported once by compile)
    }
    device->update_buffer(block, 0, p_block.size(), p_block.data());
    if (p_multiscatter) {
        device->dispatch(pipe_multiscatter, set_multiscatter, nullptr, 0, (MS_SIZE + 7) / 8, (MS_SIZE + 7) / 8, 1);
    }
    device->dispatch(pipe_volume, set_volume, nullptr, 0, (VOLUME_W + 7) / 8, (VOLUME_H + 7) / 8, 1);
    device->timestamp("atmosphere");
}

void Atmosphere::_rt_destroy() {
    if (textures_fn) {
        textures_fn(ov::gpu::Texture(), ov::gpu::Texture());
    }
    const ov::gpu::Handle handles[] = { set_volume, set_multiscatter, pipe_volume, pipe_multiscatter, block, volume, multiscatter };
    for (const ov::gpu::Handle h : handles) {
        if (h.is_valid()) {
            device->free(h);
        }
    }
    delete this;
}
