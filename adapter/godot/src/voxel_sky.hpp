#pragma once

#include <godot_cpp/classes/directional_light3d.hpp>
#include <godot_cpp/classes/environment.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3i.hpp>

using namespace godot;

// The sky. A sibling of the VoxelTerrain, the WorldEnvironment and the DirectionalLight3D (the sun): on ready it
// installs a sky shader on the environment's Sky: the terrain's atmosphere (Atmosphere::glsl_lookup) when it has one,
// else Godot's PhysicalSkyMaterial (MIT), with the terrain's volumetric clouds over either (the weather's: planets
// only). It hands the terrain's page shader the leaves' sway (`glsl_clouds()`, both shaders); each frame the sun's light
// and direction for the atmosphere.
class VoxelSky : public Node {
    GDCLASS(VoxelSky, Node)

public:
    // The GLSL both shaders splice in: the base's uniforms and functions, the weather's wind and `wind_sway`.
    static String glsl_clouds();
    // The atmosphere's uniforms and the core's lookups (Atmosphere::glsl_lookup), after glsl_clouds in both shaders.
    static String glsl_atmosphere();
    // The sky shader's full code.
    static String sky_shader_code();
    // The atmosphere's per-frame uniforms, global (every page material and the sky read them): the camera volume's
    // range (VoxelTerrain) and the sun's light (VoxelSky). Before a shader that declares them compiles (once).
    static void add_atmosphere_globals();
    // atmo_sun_dir until a sun sets it (towards the sun, unit).
    static inline const Vector3 DEFAULT_SUN_DIR = Vector3(0.3f, 0.8f, 0.5f).normalized();

    void set_wind_sway_m(float p_metres);
    float get_wind_sway_m() const;

    // The cloud uniforms as the page shader takes them (VoxelTerrain::set_page_params).
    Dictionary cloud_params() const;
    // The terrain's render frame (VoxelTerrain, at start and whenever its render origin moves): the origin's LOD0 voxel
    // (low words: the shading frame's), the voxel size and the base's uniforms (GLSL_CLOUDS: World::base_render).
    void set_render_frame(const Vector3i &p_origin_voxel_lo, float p_voxel_m, const Dictionary &p_base);
    // The terrain's atmosphere (VoxelTerrain, once its world starts with one): its uniforms (GLSL_ATMOSPHERE; the
    // samplers as RenderingServer texture RIDs). atmo_enabled turns the environment's fog off: the atmosphere is the
    // haze.
    void set_atmosphere(const Dictionary &p_params);
    // The sun's light in the atmosphere: its colour times its energy times this (pi: Godot's lights carry it).
    void set_atmosphere_gain(float p_gain);
    float get_atmosphere_gain() const;

    void _ready() override;
    void _process(double p_delta) override;

protected:
    static void _bind_methods();

private:
    void _apply();
    void _apply_frame(); // the shading frame's and the base's uniforms, on the sky material
    void _push(const Dictionary &p_params); // to the sky material and the terrain siblings' page shader
    void _quiet_fog();

    float wind_sway_m = 0.03f; // the leaves' sway amplitude (the page shader's wind_sway), 0 = still

    Vector3i origin_voxel_lo; // set_render_frame's
    float voxel_m = 0.5f;
    Dictionary base;
    Dictionary atmosphere; // set_atmosphere's
    float atmosphere_gain = 3.14159265f;
    Vector3 sun_light_sent; // the last atmo_sun_light set
    Vector3 sun_dir_sent; // the last atmo_sun_dir set

    DirectionalLight3D *sun = nullptr; // the sibling the sky's sun is (LIGHT0)
    Ref<Environment> environment;

    Ref<Shader> shader;
    Ref<ShaderMaterial> material;
};
