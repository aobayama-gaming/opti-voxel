#include "voxel_sky.hpp"

#include "voxel_terrain.hpp"

#include <godot_cpp/classes/environment.hpp>
#include <godot_cpp/classes/sky.hpp>
#include <godot_cpp/classes/world_environment.hpp>
#include <godot_cpp/core/class_db.hpp>

// Shared by the sky shader and the page shader (spliced at /*GLSL_CLOUDS*/): the cloud layer is a plane at
// cloud_height_m reading cloud_tex once per cloud_scale_m, drifting with the wind. Positions are render space
// (metres from the world origin), the same in a sky shader's POSITION and in the page shader's world point.
static constexpr const char GLSL_CLOUDS[] = R"glsl(
// The cloud layer (VoxelSky): no texture = black = no clouds and no shadows.
uniform sampler2D cloud_tex : hint_default_black, filter_linear_mipmap, repeat_enable;
uniform float cloud_height_m = 1500.0;
uniform float cloud_scale_m = 6000.0;
uniform float cloud_cover = 0.0;
uniform vec2 cloud_wind = vec2(0.0);
uniform float cloud_shadow = 0.0;

// Where a ray from p along dir meets the cloud layer, in the layer's xz (dir.y away from 0: the callers test it).
vec2 cloud_hit(vec3 p, vec3 dir) {
    return p.xz + dir.xz * ((cloud_height_m - p.y) / dir.y);
}

// The texture coordinate of a point of the layer: one tile per cloud_scale_m, carried by the wind.
vec2 cloud_uv(vec2 xz) {
    return (xz + cloud_wind * TIME) * (1.0 / cloud_scale_m);
}

// Cloud density in [0, 1] of a texel: the texture's values are uniform on [0, 1] (tools/make_clouds.py), so the
// cover is the fraction of the sky past the threshold, the ramp its soft edge.
float cloud_density(float texel) {
    float edge = 1.0 - cloud_cover;
    return smoothstep(edge - 0.1, edge + 0.2, texel);
}
)glsl";

// Godot's PhysicalSkyMaterial (scene/resources/sky_material.cpp, MIT), with the cloud layer blended over it.
static constexpr const char SKY_SHADER[] = R"gdshader(
shader_type sky;

uniform float rayleigh : hint_range(0, 64) = 2.0;
uniform vec4 rayleigh_color : source_color = vec4(0.3, 0.405, 0.6, 1.0);
uniform float mie : hint_range(0, 1) = 0.005;
uniform float mie_eccentricity : hint_range(-1, 1) = 0.8;
uniform vec4 mie_color : source_color = vec4(0.69, 0.729, 0.812, 1.0);
uniform float turbidity : hint_range(0, 1000) = 10.0;
uniform float sun_disk_scale : hint_range(0, 360) = 1.0;
uniform vec4 ground_color : source_color = vec4(0.1, 0.07, 0.034, 1.0);
uniform float exposure : hint_range(0, 128) = 1.0;
/*GLSL_CLOUDS*/

const vec3 UP = vec3(0.0, 1.0, 0.0);
// Optical length at zenith for molecules and aerosols.
const float rayleigh_zenith_size = 8.4e3;
const float mie_zenith_size = 1.25e3;

float henyey_greenstein(float cos_theta, float g) {
    const float k = 0.0795774715459;
    return k * (1.0 - g * g) / (pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}

// Light extinction along a ray at that cosine of the zenith angle, for the given coefficients.
vec3 extinction_at(float cos_zenith, vec3 rayleigh_beta, vec3 mie_beta) {
    float zenith = acos(max(0.0, cos_zenith));
    float optical_mass = 1.0 / (cos(zenith) + 0.15 * pow(93.885 - degrees(zenith), -1.253));
    return exp(-(rayleigh_beta * rayleigh_zenith_size * optical_mass + mie_beta * mie_zenith_size * optical_mass));
}

void sky() {
    if (!LIGHT0_ENABLED) { // no sun: black (a sky function has no return)
        COLOR = vec3(0.0);
    } else {
        vec3 sun_dir = normalize(LIGHT0_DIRECTION);
        float zenith_angle = clamp(dot(UP, sun_dir), -1.0, 1.0);
        float sun_energy = max(0.0, 1.0 - exp(-((PI * 0.5) - acos(zenith_angle)))) * LIGHT0_ENERGY;
        float sun_fade = 1.0 - clamp(1.0 - exp(LIGHT0_DIRECTION.y), 0.0, 1.0);

        float rayleigh_coefficient = rayleigh - (1.0 * (1.0 - sun_fade));
        vec3 rayleigh_beta = rayleigh_coefficient * rayleigh_color.rgb * 0.0001;
        vec3 mie_beta = turbidity * mie * mie_color.rgb * 0.000434; // Preetham

        vec3 extinction = extinction_at(dot(UP, EYEDIR), rayleigh_beta, mie_beta);

        float cos_theta = dot(EYEDIR, sun_dir);
        float rayleigh_phase = (3.0 / (16.0 * PI)) * (1.0 + pow(cos_theta * 0.5 + 0.5, 2.0));
        vec3 beta_r_theta = rayleigh_beta * rayleigh_phase;
        float mie_phase = henyey_greenstein(cos_theta, mie_eccentricity);
        vec3 beta_m_theta = mie_beta * mie_phase;

        vec3 scatter = sun_energy * ((beta_r_theta + beta_m_theta) / (rayleigh_beta + mie_beta));
        vec3 lin = pow(scatter * (1.0 - extinction), vec3(1.5));
        lin *= mix(vec3(1.0), pow(scatter * extinction, vec3(0.5)), clamp(pow(1.0 - zenith_angle, 5.0), 0.0, 1.0));
        lin *= mix(ground_color.rgb, vec3(1.0), smoothstep(-0.1, 0.1, dot(UP, EYEDIR)));

        float sun_cos = cos(LIGHT0_SIZE * sun_disk_scale);
        float sun_cos_half = cos(LIGHT0_SIZE * sun_disk_scale * 0.5);
        float sun_disk = smoothstep(sun_cos, sun_cos_half, cos_theta);
        vec3 l0 = (sun_energy * extinction) * sun_disk * LIGHT0_COLOR;

        vec3 color = pow(lin + l0, vec3(1.0 / (1.2 + (1.2 * sun_fade)))) * exposure;

        // The cloud layer where the view ray meets it (above the camera only). Two reads: the density here, and the
        // density towards the sun, which lights the edge of a cloud that thins that way. The sun's colour on the clouds
        // is its extinction at its own altitude (red at dusk); the shadowed side takes the sky's colour. The layer fades
        // into the sky near the horizon, where its texels stretch along the view.
        if (cloud_cover > 0.0 && EYEDIR.y > 0.002 && POSITION.y < cloud_height_m) {
            vec2 hit = cloud_hit(POSITION, EYEDIR);
            float density = cloud_density(texture(cloud_tex, cloud_uv(hit)).r);
            vec2 towards_sun = hit + sun_dir.xz * (0.02 * cloud_scale_m / max(sun_dir.y, 0.15));
            float lit = 1.0 - 0.75 * cloud_density(texture(cloud_tex, cloud_uv(towards_sun)).r);
            vec3 sun_light = LIGHT0_COLOR * extinction_at(zenith_angle, rayleigh_beta, mie_beta) * (sun_energy / max(LIGHT0_ENERGY, 1e-3));
            vec3 cloud_color = mix(color * 1.1, sun_light * exposure, 0.85 * lit);
            color = mix(color, cloud_color, density * smoothstep(0.0, 0.08, EYEDIR.y));
        }
        COLOR = color;
    }
}
)gdshader";

String VoxelSky::glsl_clouds() {
    return String(GLSL_CLOUDS);
}

String VoxelSky::sky_shader_code() {
    return String(SKY_SHADER).replace("/*GLSL_CLOUDS*/", glsl_clouds());
}

void VoxelSky::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_cloud_texture", "texture"), &VoxelSky::set_cloud_texture);
    ClassDB::bind_method(D_METHOD("get_cloud_texture"), &VoxelSky::get_cloud_texture);
    ClassDB::bind_method(D_METHOD("set_cloud_height_m", "metres"), &VoxelSky::set_cloud_height_m);
    ClassDB::bind_method(D_METHOD("get_cloud_height_m"), &VoxelSky::get_cloud_height_m);
    ClassDB::bind_method(D_METHOD("set_cloud_scale_m", "metres"), &VoxelSky::set_cloud_scale_m);
    ClassDB::bind_method(D_METHOD("get_cloud_scale_m"), &VoxelSky::get_cloud_scale_m);
    ClassDB::bind_method(D_METHOD("set_cloud_cover", "cover"), &VoxelSky::set_cloud_cover);
    ClassDB::bind_method(D_METHOD("get_cloud_cover"), &VoxelSky::get_cloud_cover);
    ClassDB::bind_method(D_METHOD("set_cloud_wind", "mps"), &VoxelSky::set_cloud_wind);
    ClassDB::bind_method(D_METHOD("get_cloud_wind"), &VoxelSky::get_cloud_wind);
    ClassDB::bind_method(D_METHOD("set_cloud_shadow", "strength"), &VoxelSky::set_cloud_shadow);
    ClassDB::bind_method(D_METHOD("get_cloud_shadow"), &VoxelSky::get_cloud_shadow);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "cloud_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_cloud_texture", "get_cloud_texture");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cloud_height_m", PROPERTY_HINT_RANGE, "100,20000,10"), "set_cloud_height_m", "get_cloud_height_m");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cloud_scale_m", PROPERTY_HINT_RANGE, "100,100000,10"), "set_cloud_scale_m", "get_cloud_scale_m");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cloud_cover", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_cloud_cover", "get_cloud_cover");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "cloud_wind"), "set_cloud_wind", "get_cloud_wind");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cloud_shadow", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_cloud_shadow", "get_cloud_shadow");
}

void VoxelSky::set_cloud_texture(const Ref<Texture2D> &p_texture) {
    cloud_texture = p_texture;
    _apply();
}

Ref<Texture2D> VoxelSky::get_cloud_texture() const {
    return cloud_texture;
}

void VoxelSky::set_cloud_height_m(float p_m) {
    cloud_height_m = p_m;
    _apply();
}

float VoxelSky::get_cloud_height_m() const {
    return cloud_height_m;
}

void VoxelSky::set_cloud_scale_m(float p_m) {
    cloud_scale_m = MAX(p_m, 1.0f);
    _apply();
}

float VoxelSky::get_cloud_scale_m() const {
    return cloud_scale_m;
}

void VoxelSky::set_cloud_cover(float p_cover) {
    cloud_cover = CLAMP(p_cover, 0.0f, 1.0f);
    _apply();
}

float VoxelSky::get_cloud_cover() const {
    return cloud_cover;
}

void VoxelSky::set_cloud_wind(const Vector2 &p_mps) {
    cloud_wind = p_mps;
    _apply();
}

Vector2 VoxelSky::get_cloud_wind() const {
    return cloud_wind;
}

void VoxelSky::set_cloud_shadow(float p_strength) {
    cloud_shadow = CLAMP(p_strength, 0.0f, 1.0f);
    _apply();
}

float VoxelSky::get_cloud_shadow() const {
    return cloud_shadow;
}

Dictionary VoxelSky::cloud_params() const {
    Dictionary params;
    params["cloud_tex"] = cloud_texture.is_valid() ? cloud_texture->get_rid() : RID();
    params["cloud_height_m"] = cloud_height_m;
    params["cloud_scale_m"] = cloud_scale_m;
    params["cloud_cover"] = cloud_cover;
    params["cloud_wind"] = cloud_wind;
    params["cloud_shadow"] = cloud_shadow;
    return params;
}

void VoxelSky::_ready() {
    shader.instantiate();
    shader->set_code(sky_shader_code());
    material.instantiate();
    material->set_shader(shader);
    Node *parent = get_parent();
    ERR_FAIL_NULL(parent);
    WorldEnvironment *world_env = nullptr;
    for (int i = 0; i < parent->get_child_count(); i++) {
        if (WorldEnvironment *w = Object::cast_to<WorldEnvironment>(parent->get_child(i))) {
            world_env = w;
        }
    }
    ERR_FAIL_NULL_MSG(world_env, "VoxelSky: no WorldEnvironment sibling to put the sky on.");
    Ref<Environment> env = world_env->get_environment();
    ERR_FAIL_COND_MSG(env.is_null(), "VoxelSky: the WorldEnvironment has no Environment.");
    Ref<Sky> sky = env->get_sky();
    if (sky.is_null()) {
        sky.instantiate();
        env->set_sky(sky);
    }
    sky->set_material(material);
    _apply();
}

// Pushes the cloud parameters to the sky material and to the terrain's page shader (the shadows).
void VoxelSky::_apply() {
    if (material.is_null()) {
        return; // before _ready
    }
    Dictionary params = cloud_params();
    Array keys = params.keys();
    for (int i = 0; i < keys.size(); i++) {
        // The sky material takes the texture itself, the page shader (RenderingServer level) its RID.
        material->set_shader_parameter(keys[i], keys[i] == Variant("cloud_tex") ? Variant(cloud_texture) : params[keys[i]]);
    }
    Node *parent = get_parent();
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        if (VoxelTerrain *terrain = Object::cast_to<VoxelTerrain>(parent->get_child(i))) {
            terrain->set_page_params(params);
        }
    }
}
