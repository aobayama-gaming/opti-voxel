#include "voxel_sky.hpp"

#include "voxel_terrain.hpp"

#include "atmosphere.hpp"
#include "base_surface.hpp"
#include "clouds.hpp"
#include "glsl_uniforms.hpp"
#include "godot_conv.hpp"
#include "shade_frame_glsl.hpp"
#include "sky_glsl.hpp"

#include <godot_cpp/classes/directional_light3d.hpp>
#include <godot_cpp/classes/environment.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/sky.hpp>
#include <godot_cpp/classes/world_environment.hpp>
#include <godot_cpp/core/class_db.hpp>

// Shared by the sky shader and the page shader (spliced at /*GLSL_CLOUDS*/, after the shading frame:
// shade_frame_glsl.hpp): the base, and the leaves' sway in the weather's wind. Positions are render space (metres from
// the render origin), the same in a sky shader's POSITION and in the page shader's world point; the wind's waves are read
// through the shading frame, so they stay where they are when the render origin moves.
static constexpr const char GLSL_CLOUDS[] = R"glsl(
// The base in render space (World::base_render, set again when the render origin moves: VoxelTerrain on its pages,
// VoxelSky on the sky): BaseSurface::Render's uniforms and BaseSurface::glsl_render's functions.
/*GLSL_BASE_RENDER*/

// The weather's wind at the camera (World::camera_wind: render metres per second, tangent to the base; VoxelTerrain sets
// it each frame, 0 without weather) and the leaves' sway amplitude (VoxelSky).
global uniform vec3 weather_wind;
uniform float wind_sway_m = 0.0;

// Wind sway: a world-space displacement, a pure function of the position and TIME, so every quad sharing a corner
// moves it alike and the mesh stays watertight. A gust wave of ~6 m travelling with weather_wind, its strength breathing
// over ~23 m, plus a finer wobble of ~1 m; wind_sway_m is the amplitude (0: still). The page shader applies it to leaf
// corners only: bark has no per-tree pivot to bend around (the pages don't carry one). Its waves are snapped in the
// shading frame (period: the 23 m wave), so a corner sways alike from every render origin.
vec3 wind_sway(vec3 p) {
    float speed = max(length(weather_wind), 1.0);
    vec3 along = weather_wind / speed;
    float period_m;
    vec3 s = shade_frame(p, 23.0, period_m);
    vec3 k6 = shade_wave(along * (6.2832 / 6.0), period_m);
    vec3 k23 = shade_wave(along * (6.2832 / 23.0) + vec3(0.0, 0.3, 0.0), period_m);
    float gust = sin(dot(s, k6) - speed * TIME * (6.2832 / 6.0)) * (0.6 + 0.4 * sin(dot(s, k23) - speed * TIME * (6.2832 / 23.0)));
    vec3 wobble = vec3(sin(dot(s, shade_wave(vec3(7.1, 5.3, 6.7), period_m)) + TIME * 3.1),
            sin(dot(s, shade_wave(vec3(5.9, 7.7, 4.3), period_m)) + TIME * 2.3), sin(dot(s, shade_wave(vec3(6.3, 4.9, 7.9), period_m)) + TIME * 2.7));
    return wind_sway_m * (gust * (along + 0.15 * base_up(p)) + 0.35 * wobble);
}
)glsl";

// The atmosphere's and the volumetric clouds' uniforms in both shaders (after GLSL_CLOUDS: the base), then the core's
// lookups (Atmosphere::glsl_lookup, Clouds::glsl_lookup). VoxelTerrain sets them once its world starts with an
// atmosphere (off until then): the camera volume and the multiple-scattering table, the medium, the clouds' grid, shadow map
// and layer; per frame the globals atmo_range_m and the shadow map's frame (VoxelTerrain), atmo_sun_light (VoxelSky:
// the sun's colour times its energy times atmosphere_gain) and atmo_sun_dir (VoxelSky: towards the sun; VoxelTerrain
// from its DirectionalLight3D sibling where no VoxelSky is: the sea pass's glint).
static constexpr const char GLSL_ATMOSPHERE[] = R"glsl(
uniform bool atmo_enabled = false;
uniform sampler2D atmo_volume : filter_linear, repeat_enable;
uniform sampler2D atmo_multiscatter : filter_linear;
global uniform float atmo_range_m;
global uniform vec3 atmo_sun_light;
global uniform vec3 atmo_sun_dir;
uniform bool cloud_enabled = false;
uniform sampler2D cloud_light : filter_linear, repeat_enable;
uniform sampler2D cloud_depth : filter_linear, repeat_enable;
uniform sampler2D cloud_shadow_map : filter_linear;
)glsl";

// The sky: the atmosphere (Atmosphere: the camera volume's sky slice, marched per pixel from above it), or without one
// Godot's PhysicalSkyMaterial (SkyGLSL: sky_glsl.hpp, shared with the Dawn viewer); the volumetric clouds over it.
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
/*GLSL_ATMOSPHERE*/
/*GLSL_SKY*/


void sky() {
    if (!LIGHT0_ENABLED) { // no sun: black (a sky function has no return)
        COLOR = vec3(0.0);
    } else {
        // The sky in the camera's local frame, its up on +y (the identity on the flat world): on a planet the horizon
        // and the sun's height are the ground's there, and the sun, a global direction, sets as the camera goes round.
        vec3 t1;
        vec3 t2;
        vec3 up = base_up(POSITION);
        up_frame(up, t1, t2);
        mat3 to_local = transpose(mat3(t1, up, t2));
        vec3 eye = to_local * EYEDIR;
        vec3 sun_world = normalize(LIGHT0_DIRECTION);
        vec3 sun_dir = to_local * sun_world;
        vec3 color;
        if (atmo_enabled) {
            // No `exposure`: the pages' aerial perspective is the same radiance, so the sky and the haze meet at the horizon.
            color = atmo_sky_radiance(POSITION, EYEDIR, sun_world, cos(0.5 * LIGHT0_SIZE)) * atmo_sun_light;
        } else {
            color = physical_sky(eye, sun_dir, LIGHT0_COLOR, LIGHT0_ENERGY, LIGHT0_SIZE, exposure);
        }

        // The volumetric clouds (Clouds: the grid's whole ray), over the sky and its sun.
        if (cloud_enabled) {
            vec4 clouds = cloud_at(POSITION, EYEDIR, 1e30);
            color = color * clouds.a + clouds.rgb * atmo_sun_light;
        }
        COLOR = color;
    }
}
)gdshader";

String VoxelSky::glsl_clouds() {
    return String(GLSL_CLOUDS).replace("/*GLSL_BASE_RENDER*/", godot_str(BaseSurface::glsl_render_uniforms() + BaseSurface::glsl_render()));
}

String VoxelSky::glsl_atmosphere() {
    ov::str globals;
    Clouds::visit_host_frame(Clouds::Frame(), GLSLUniforms::Declarations{ globals, "global uniform " });
    return String(GLSL_ATMOSPHERE) + godot_str(Atmosphere::glsl_host_uniforms() + Atmosphere::glsl_lookup() + globals + Clouds::glsl_lookup());
}

void VoxelSky::add_atmosphere_globals() {
    static bool added = false;
    if (added) {
        return;
    }
    added = true;
    RenderingServer *rs = RenderingServer::get_singleton();
    rs->global_shader_parameter_add("atmo_range_m", RenderingServer::GLOBAL_VAR_TYPE_FLOAT, 1.0);
    rs->global_shader_parameter_add("atmo_sun_light", RenderingServer::GLOBAL_VAR_TYPE_VEC3, Vector3(1.0f, 1.0f, 1.0f));
    rs->global_shader_parameter_add("atmo_sun_dir", RenderingServer::GLOBAL_VAR_TYPE_VEC3, DEFAULT_SUN_DIR);
    rs->global_shader_parameter_add("weather_wind", RenderingServer::GLOBAL_VAR_TYPE_VEC3, Vector3());
    // The clouds' shadow map frame (Clouds::visit_host_frame: vec3s and a float).
    struct AddGlobal {
        RenderingServer *rs;
        void operator()(const char *p_name, const Float3 &) { rs->global_shader_parameter_add(p_name, RenderingServer::GLOBAL_VAR_TYPE_VEC3, Vector3()); }
        void operator()(const char *p_name, float) { rs->global_shader_parameter_add(p_name, RenderingServer::GLOBAL_VAR_TYPE_FLOAT, 1.0); }
    };
    Clouds::visit_host_frame(Clouds::Frame(), AddGlobal{ rs });
}

String VoxelSky::sky_shader_code() {
    return String(SKY_SHADER)
            .replace("/*GLSL_SHADE_FRAME*/", godot_str(ShadeFrame::glsl()))
            .replace("/*GLSL_CLOUDS*/", glsl_clouds())
            .replace("/*GLSL_ATMOSPHERE*/", glsl_atmosphere())
            .replace("/*GLSL_SKY*/", godot_str(SkyGLSL::glsl()));
}

void VoxelSky::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_atmosphere_gain", "gain"), &VoxelSky::set_atmosphere_gain);
    ClassDB::bind_method(D_METHOD("get_atmosphere_gain"), &VoxelSky::get_atmosphere_gain);
    ClassDB::bind_method(D_METHOD("set_wind_sway_m", "metres"), &VoxelSky::set_wind_sway_m);
    ClassDB::bind_method(D_METHOD("get_wind_sway_m"), &VoxelSky::get_wind_sway_m);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wind_sway_m", PROPERTY_HINT_RANGE, "0,1,0.005"), "set_wind_sway_m", "get_wind_sway_m");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "atmosphere_gain", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_atmosphere_gain", "get_atmosphere_gain");
}

void VoxelSky::set_wind_sway_m(float p_metres) {
    wind_sway_m = MAX(p_metres, 0.0f);
    _apply();
}

float VoxelSky::get_wind_sway_m() const {
    return wind_sway_m;
}

void VoxelSky::set_render_frame(const Vector3i &p_origin_voxel_lo, float p_voxel_m, const Dictionary &p_base) {
    origin_voxel_lo = p_origin_voxel_lo;
    voxel_m = p_voxel_m;
    base = p_base;
    if (material.is_null()) {
        return; // before _ready: its _apply sends everything
    }
    _apply_frame();
}

void VoxelSky::set_atmosphere(const Dictionary &p_params) {
    Array keys = p_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        atmosphere[keys[i]] = p_params[keys[i]];
    }
    if (p_params.has("atmo_enabled") && bool(p_params["atmo_enabled"])) {
        _quiet_fog();
    }
    if (material.is_null()) {
        return; // before _ready: _apply_frame sends everything
    }
    _apply_frame();
}

void VoxelSky::set_atmosphere_gain(float p_gain) {
    atmosphere_gain = MAX(p_gain, 0.0f);
}

float VoxelSky::get_atmosphere_gain() const {
    return atmosphere_gain;
}

// The sun's light and direction for the atmosphere: the global atmo_sun_light (both shaders) and the terrains' sun
// (its camera volume), when they change. With weather the planet's day turns the light: the authored sun (or one a
// script set since) goes to the terrains, which turn it themselves; the light takes the first terrain's turn.
void VoxelSky::_process(double) {
    if (sun == nullptr) {
        return;
    }
    const Color c = sun->get_color().srgb_to_linear();
    const float e = float(sun->get_param(Light3D::PARAM_ENERGY)) * atmosphere_gain;
    const Vector3 light(c.r * e, c.g * e, c.b * e);
    if (light != sun_light_sent) {
        sun_light_sent = light;
        RenderingServer::get_singleton()->global_shader_parameter_set("atmo_sun_light", light);
    }
    Transform3D xf = sun->get_global_transform();
    if (!has_noon || !xf.basis.is_equal_approx(basis_set)) {
        noon_basis = xf.basis;
        has_noon = true;
    }
    const Vector3 noon = noon_basis.get_column(2).normalized();
    Node *parent = get_parent();
    bool turned = false;
    double angle = 0.0;
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        if (VoxelTerrain *terrain = Object::cast_to<VoxelTerrain>(parent->get_child(i))) {
            // Every frame: a terrain that starts later gets them at once.
            terrain->set_sun_direction(noon);
            angle = turned ? angle : terrain->get_day_angle();
            turned = true;
        }
    }
    basis_set = Basis(Vector3(0.0f, 1.0f, 0.0f), real_t(angle)) * noon_basis;
    if (!xf.basis.is_equal_approx(basis_set)) {
        xf.basis = basis_set;
        sun->set_global_transform(xf);
    }
    const Vector3 towards = basis_set.get_column(2).normalized();
    if (moon != nullptr) {
        // Turned half a turn about the sun's x: its z (towards it) the sun's negated.
        const Transform3D moon_xf(basis_set * Basis(Vector3(1.0f, 0.0f, 0.0f), real_t(Math_PI)), xf.origin);
        if (!moon->get_global_transform().is_equal_approx(moon_xf)) {
            moon->set_global_transform(moon_xf);
        }
        moon->set_color(sun->get_color());
        moon->set_param(Light3D::PARAM_ENERGY, sun->get_param(Light3D::PARAM_ENERGY) * Atmosphere::MOON_SHARE);
    }
    if (towards != sun_dir_sent) {
        sun_dir_sent = towards;
        RenderingServer::get_singleton()->global_shader_parameter_set("atmo_sun_dir", towards);
    }
}

Dictionary VoxelSky::cloud_params() const {
    Dictionary params;
    params["wind_sway_m"] = wind_sway_m;
    // The sea's sky without the atmosphere (the pass's reflection, Sea::glsl_water): the fog's colour, the horizon's.
    if (environment.is_valid()) {
        const Color fog = environment->get_fog_light_color().srgb_to_linear();
        params["sea_sky"] = Vector3(fog.r, fog.g, fog.b);
    }
    return params;
}

void VoxelSky::_ready() {
    add_atmosphere_globals();
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
    for (int i = 0; i < parent->get_child_count(); i++) {
        if (DirectionalLight3D *light = Object::cast_to<DirectionalLight3D>(parent->get_child(i))) {
            sun = light;
        }
    }
    if (sun != nullptr && moon == nullptr) {
        moon = memnew(DirectionalLight3D);
        moon->set_name("Moon");
        moon->set_sky_mode(DirectionalLight3D::SKY_MODE_LIGHT_ONLY);
        moon->set_shadow(false);
        add_child(moon);
    }
    Ref<Environment> env = world_env->get_environment();
    ERR_FAIL_COND_MSG(env.is_null(), "VoxelSky: the WorldEnvironment has no Environment.");
    environment = env;
    Ref<Sky> sky = env->get_sky();
    if (sky.is_null()) {
        sky.instantiate();
        env->set_sky(sky);
    }
    sky->set_material(material);
    _apply();
    if (atmosphere.has("atmo_enabled") && bool(atmosphere["atmo_enabled"])) {
        _quiet_fog(); // the terrain started first
    }
    set_process(true);
}

// The atmosphere is the haze: the environment's fog stays on (Godot drops a shader's own FOG without it) with nothing of
// its own.
void VoxelSky::_quiet_fog() {
    if (environment.is_null()) {
        return;
    }
    environment->set_fog_enabled(true);
    environment->set_fog_density(0.0f);
    environment->set_fog_height_density(0.0f);
    environment->set_fog_sky_affect(0.0f);
    environment->set_fog_aerial_perspective(0.0f);
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
    Array keys = base.keys();
    for (int i = 0; i < keys.size(); i++) {
        material->set_shader_parameter(keys[i], base[keys[i]]); // the terrain sets its own pages'
    }
    // The atmosphere's: the samplers are RenderingServer texture RIDs, set on the material's RID.
    keys = atmosphere.keys();
    for (int i = 0; i < keys.size(); i++) {
        if (atmosphere[keys[i]].get_type() == Variant::RID) {
            RenderingServer::get_singleton()->material_set_param(material->get_rid(), keys[i], atmosphere[keys[i]]);
        } else {
            material->set_shader_parameter(keys[i], atmosphere[keys[i]]);
        }
    }
}

void VoxelSky::_push(const Dictionary &p_params) {
    Array keys = p_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        material->set_shader_parameter(keys[i], p_params[keys[i]]);
    }
    Node *parent = get_parent();
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        if (VoxelTerrain *terrain = Object::cast_to<VoxelTerrain>(parent->get_child(i))) {
            terrain->set_page_params(p_params);
        }
    }
}
