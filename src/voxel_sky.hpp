#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2.hpp>

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

    void _ready() override;

protected:
    static void _bind_methods();

private:
    void _apply();

    Ref<Texture2D> cloud_texture;
    float cloud_height_m = 1500.0f; // metres above the world origin
    float cloud_scale_m = 6000.0f; // metres per texture tile
    float cloud_cover = 0.4f; // fraction of the sky covered
    Vector2 cloud_wind = Vector2(9.0f, 3.0f); // m/s
    float cloud_shadow = 0.7f; // how much of the sun a cloud takes from the ground, 0 = no shadows
    float wind_sway_m = 0.03f; // the leaves' sway amplitude (the page shader's wind_sway), 0 = still

    Ref<Shader> shader;
    Ref<ShaderMaterial> material;
};
