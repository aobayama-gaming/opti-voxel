#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3i.hpp>

using namespace godot;

// The sky and its cloud layer. A sibling of the VoxelTerrain and the WorldEnvironment: on ready it installs a sky
// shader (Godot's PhysicalSkyMaterial, MIT, with a cloud layer over it) on the environment's Sky and hands the
// terrain's page shader the same cloud parameters, so the clouds' shadows on the ground lie under the clouds in the
// sky: both read `cloud_texture` through the one GLSL of `glsl_clouds()` (uv from the world point, density from the
// cover), at the same height, drifting with the same wind. The bench scene has no VoxelSky: its page shader keeps the
// uniforms' defaults (no texture, cloud_shadow 0), so its output does not change.
class VoxelSky : public Node {
    GDCLASS(VoxelSky, Node)

public:
    // The GLSL both shaders splice in: the cloud uniforms and `cloud_uv` / `cloud_density` / `cloud_hit`.
    static String glsl_clouds();
    // The sky shader's full code.
    static String sky_shader_code();

    void set_cloud_texture(const Ref<Texture2D> &p_texture);
    Ref<Texture2D> get_cloud_texture() const;
    void set_cloud_height_m(float p_m);
    float get_cloud_height_m() const;
    void set_cloud_scale_m(float p_m);
    float get_cloud_scale_m() const;
    void set_cloud_cover(float p_cover);
    float get_cloud_cover() const;
    void set_cloud_wind(const Vector2 &p_mps);
    Vector2 get_cloud_wind() const;
    void set_cloud_shadow(float p_strength);
    float get_cloud_shadow() const;
    void set_wind_sway_m(float p_metres);
    float get_wind_sway_m() const;

    // The cloud uniforms as the page shader takes them (VoxelTerrain::set_page_params).
    Dictionary cloud_params() const;
    // The terrain's render frame (VoxelTerrain, at start and whenever its render origin moves): the origin's LOD0 voxel
    // (low words: the shading frame's), the voxel size, how far the origin rose above the world origin (metres: the flat
    // world's cloud layer stays at its height above the world origin; 0 on a planet, whose shell is above its base) and
    // the planet's uniforms (GLSL_CLOUDS: planet_centre_m, planet_radius_m, planet_k_m2).
    void set_render_frame(const Vector3i &p_origin_voxel_lo, float p_voxel_m, float p_origin_rise_m, const Dictionary &p_planet);

    void _ready() override;

protected:
    static void _bind_methods();

private:
    void _apply();
    void _apply_frame(); // the shading frame's and the planet's uniforms, on the sky material
    void _push(const Dictionary &p_params); // to the sky material and the terrain siblings' page shader

    Ref<Texture2D> cloud_texture;
    float cloud_height_m = 1500.0f; // metres above the world origin (cloud_params: above the render origin), a planet's base
    float cloud_scale_m = 6000.0f; // metres per texture tile
    float cloud_cover = 0.4f; // fraction of the sky covered
    Vector2 cloud_wind = Vector2(9.0f, 3.0f); // m/s
    float cloud_shadow = 0.7f; // how much of the sun a cloud takes from the ground, 0 = no shadows
    float wind_sway_m = 0.03f; // the leaves' sway amplitude (the page shader's wind_sway), 0 = still

    Vector3i origin_voxel_lo; // set_render_frame's
    float voxel_m = 0.5f;
    float origin_rise_m = 0.0f;
    Dictionary planet;

    Ref<Shader> shader;
    Ref<ShaderMaterial> material;
};
