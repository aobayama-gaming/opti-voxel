#include "voxel_sky.hpp"

#include "voxel_terrain.hpp"

#include "godot_conv.hpp"
#include "shade_frame_glsl.hpp"
#include "sky_glsl.hpp"

#include <godot_cpp/classes/environment.hpp>
#include <godot_cpp/classes/sky.hpp>
#include <godot_cpp/classes/world_environment.hpp>
#include <godot_cpp/core/class_db.hpp>

// Shared by the sky shader and the page shader (spliced at /*GLSL_CLOUDS*/, after the shading frame:
// shade_frame_glsl.hpp): the cloud layer is a plane at cloud_height_m reading cloud_tex about once per cloud_scale_m,
// drifting with the wind. Positions are render space (metres from the render origin), the same in a sky shader's
// POSITION and in the page shader's world point; the tiles and the wind's waves are read through the shading frame, so
// they stay where they are when the render origin moves.
static constexpr const char GLSL_CLOUDS[] = R"glsl(
// The cloud layer (VoxelSky): no texture = black = no clouds and no shadows.
uniform sampler2D cloud_tex : hint_default_black, filter_linear_mipmap, repeat_enable;
uniform float cloud_height_m = 1500.0; // render metres (VoxelSky::cloud_params)
uniform float cloud_scale_m = 6000.0;
uniform float cloud_cover = 0.0;
uniform vec2 cloud_wind = vec2(0.0);
uniform float cloud_shadow = 0.0;
uniform float wind_sway_m = 0.0;

// Where a ray from p along dir meets the cloud layer, in the layer's xz (dir.y away from 0: the callers test it).
vec2 cloud_hit(vec3 p, vec3 dir) {
    return p.xz + dir.xz * ((cloud_height_m - p.y) / dir.y);
}

// The texture coordinate of a point of the layer: one tile per cloud_scale_m (snapped to the shading frame's period),
// carried by the wind.
vec2 cloud_uv(vec2 xz) {
    float period_m;
    vec2 s = shade_frame(vec3(xz.x, 0.0, xz.y), cloud_scale_m, period_m).xz;
    float tiles = shade_cells(vec3(1.0 / cloud_scale_m), period_m).x;
    return (s + cloud_wind * TIME) * (tiles / period_m);
}

// Cloud density in [0, 1] of a texel: the texture's values are uniform on [0, 1] (tools/make_clouds.py), so the
// cover is the fraction of the sky past the threshold, the ramp its soft edge.
float cloud_density(float texel) {
    float edge = 1.0 - cloud_cover;
    return smoothstep(edge - 0.1, edge + 0.2, texel);
}
// Wind sway: a world-space displacement, a pure function of the position and TIME, so every quad sharing a corner
// moves it alike and the mesh stays watertight. A gust wave of ~6 m travelling with cloud_wind, its strength breathing
// over ~23 m, plus a finer wobble of ~1 m; wind_sway_m is the amplitude (0: still). The page shader applies it to leaf
// corners only: bark has no per-tree pivot to bend around (the pages don't carry one). Its waves are snapped in the
// shading frame (period: the 23 m wave), so a corner sways alike from every render origin.
vec3 wind_sway(vec3 p) {
    float speed = max(length(cloud_wind), 1.0);
    vec2 dir = cloud_wind / speed;
    float period_m;
    vec3 s = shade_frame(p, 23.0, period_m);
    vec3 along = vec3(dir.x, 0.0, dir.y);
    vec3 k6 = shade_wave(along * (6.2832 / 6.0), period_m);
    vec3 k23 = shade_wave(along * (6.2832 / 23.0) + vec3(0.0, 0.3, 0.0), period_m);
    float gust = sin(dot(s, k6) - speed * TIME * (6.2832 / 6.0)) * (0.6 + 0.4 * sin(dot(s, k23) - speed * TIME * (6.2832 / 23.0)));
    vec3 wobble = vec3(sin(dot(s, shade_wave(vec3(7.1, 5.3, 6.7), period_m)) + TIME * 3.1),
            sin(dot(s, shade_wave(vec3(5.9, 7.7, 4.3), period_m)) + TIME * 2.3), sin(dot(s, shade_wave(vec3(6.3, 4.9, 7.9), period_m)) + TIME * 2.7));
    return wind_sway_m * (gust * vec3(dir.x, 0.15, dir.y) + 0.35 * wobble);
}
)glsl";

// Godot's PhysicalSkyMaterial (SkyGLSL: sky_glsl.hpp, shared with the Dawn viewer), with the cloud layer blended over it.
static constexpr const char SKY_SHADER[] = R"gdshader(
shader_type sky;

uniform float exposure : hint_range(0, 128) = 1.0;
// The shading frame's origin (shade_frame_glsl.hpp): the render origin's LOD0 voxel, low words, and the voxel size
// (VoxelSky::set_render_frame).
uniform ivec3 shade_origin_lo = ivec3(0);
uniform float shade_voxel_size = 0.5;

ivec3 shade_origin_voxel() {
    return shade_origin_lo;
}

float shade_voxel_m() {
    return shade_voxel_size;
}

/*GLSL_SHADE_FRAME*/
/*GLSL_CLOUDS*/
/*GLSL_SKY*/

void sky() {
    if (!LIGHT0_ENABLED) { // no sun: black (a sky function has no return)
        COLOR = vec3(0.0);
    } else {
        vec3 sun_dir = normalize(LIGHT0_DIRECTION);
        vec3 color = physical_sky(EYEDIR, sun_dir, LIGHT0_COLOR, LIGHT0_ENERGY, LIGHT0_SIZE, exposure);

        // The cloud layer where the view ray meets it (above the camera only). Two reads: the density here, and the
        // density towards the sun, which lights the edge of a cloud that thins that way. The sun's colour on the clouds
        // is its extinction at its own altitude (red at dusk); the shadowed side takes the sky's colour. The layer fades
        // into the sky near the horizon, where its texels stretch along the view.
        if (cloud_cover > 0.0 && EYEDIR.y > 0.002 && POSITION.y < cloud_height_m) {
            vec2 hit = cloud_hit(POSITION, EYEDIR);
            float density = cloud_density(texture(cloud_tex, cloud_uv(hit)).r);
            vec2 towards_sun = hit + sun_dir.xz * (0.02 * cloud_scale_m / max(sun_dir.y, 0.15));
            float lit = 1.0 - 0.75 * cloud_density(texture(cloud_tex, cloud_uv(towards_sun)).r);
            vec3 rayleigh_beta;
            vec3 mie_beta;
            sky_betas(sun_dir, rayleigh_beta, mie_beta);
            vec3 sun_light = LIGHT0_COLOR * sky_extinction(sun_dir.y, rayleigh_beta, mie_beta) * (sky_sun_energy(sun_dir, LIGHT0_ENERGY) / max(LIGHT0_ENERGY, 1e-3));
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
    return String(SKY_SHADER)
            .replace("/*GLSL_SHADE_FRAME*/", godot_str(ShadeFrame::glsl()))
            .replace("/*GLSL_CLOUDS*/", glsl_clouds())
            .replace("/*GLSL_SKY*/", godot_str(SkyGLSL::glsl()));
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
    ClassDB::bind_method(D_METHOD("set_wind_sway_m", "metres"), &VoxelSky::set_wind_sway_m);
    ClassDB::bind_method(D_METHOD("get_wind_sway_m"), &VoxelSky::get_wind_sway_m);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "cloud_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_cloud_texture", "get_cloud_texture");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cloud_height_m", PROPERTY_HINT_RANGE, "100,20000,10"), "set_cloud_height_m", "get_cloud_height_m");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cloud_scale_m", PROPERTY_HINT_RANGE, "100,100000,10"), "set_cloud_scale_m", "get_cloud_scale_m");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cloud_cover", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_cloud_cover", "get_cloud_cover");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "cloud_wind"), "set_cloud_wind", "get_cloud_wind");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cloud_shadow", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_cloud_shadow", "get_cloud_shadow");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wind_sway_m", PROPERTY_HINT_RANGE, "0,1,0.005"), "set_wind_sway_m", "get_wind_sway_m");
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

void VoxelSky::set_wind_sway_m(float p_metres) {
    wind_sway_m = MAX(p_metres, 0.0f);
    _apply();
}

float VoxelSky::get_wind_sway_m() const {
    return wind_sway_m;
}

void VoxelSky::set_render_frame(const Vector3i &p_origin_voxel_lo, float p_voxel_m, float p_origin_rise_m) {
    origin_voxel_lo = p_origin_voxel_lo;
    voxel_m = p_voxel_m;
    origin_rise_m = p_origin_rise_m;
    if (material.is_null()) {
        return; // before _ready: its _apply sends everything
    }
    _apply_frame();
    Dictionary height;
    height["cloud_height_m"] = cloud_height_m - origin_rise_m;
    _push(height);
}

Dictionary VoxelSky::cloud_params() const {
    Dictionary params;
    params["cloud_tex"] = cloud_texture.is_valid() ? cloud_texture->get_rid() : RID();
    params["cloud_height_m"] = cloud_height_m - origin_rise_m;
    params["cloud_scale_m"] = cloud_scale_m;
    params["cloud_cover"] = cloud_cover;
    params["cloud_wind"] = cloud_wind;
    params["cloud_shadow"] = cloud_shadow;
    params["wind_sway_m"] = wind_sway_m;
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

// Pushes the cloud parameters and the render frame to the sky material, the cloud parameters to the terrain's page
// shader (the shadows).
void VoxelSky::_apply() {
    if (material.is_null()) {
        return; // before _ready
    }
    _apply_frame();
    _push(cloud_params());
}

void VoxelSky::_apply_frame() {
    material->set_shader_parameter("shade_origin_lo", origin_voxel_lo);
    material->set_shader_parameter("shade_voxel_size", voxel_m);
}

void VoxelSky::_push(const Dictionary &p_params) {
    Array keys = p_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        // The sky material takes the texture itself, the page shader (RenderingServer level) its RID.
        material->set_shader_parameter(keys[i], keys[i] == Variant("cloud_tex") ? Variant(cloud_texture) : p_params[keys[i]]);
    }
    Node *parent = get_parent();
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        if (VoxelTerrain *terrain = Object::cast_to<VoxelTerrain>(parent->get_child(i))) {
            terrain->set_page_params(p_params);
        }
    }
}
