#pragma once

#include "sdf_base.hpp"
#include <cmath>
#include "noise.hpp"

class SDFDummy : public SDFBase {
public:
    SDFDummy() = default;
    ~SDFDummy() override = default;
    
    //plane
    // float evaluate(const Vector3 &p_world_pos) const override {
    //     return p_world_pos.y - 350.0f;
    // }

    // GRAD OF 1 IS IMPORTANT !!!!
    // float evaluate(const Vector3 &p_world_pos) const override {
    //     return -sqrt(p_world_pos.y*p_world_pos.y + p_world_pos.x*p_world_pos.x + p_world_pos.z*p_world_pos.z)  + 10.0f;
    // }

    // Vector3 evaluate_normal(const Vector3 &p_world_pos) const override {
    //     const float len_sq = p_world_pos.length_squared();
    //     if (len_sq <= 1e-12f) {
    //         return Vector3(0.0f, 1.0f, 0.0f);
    //     }
    //     return p_world_pos / std::sqrt(len_sq);
    // }

    float evaluate(const Vector3 &p_world_pos) const override {
        // Mountain with sparse large caverns
        static const SimplexNoise::NoiseParam mountain_param{0.0035f, 1.0f, 2.0f, 0.5f};
        static const SimplexNoise::NoiseParam cave_blob_param{0.0080f, 1.0f, 2.0f, 0.5f};  // Low freq for big blobs

        const float x = p_world_pos.x;
        const float y = p_world_pos.y;
        const float z = p_world_pos.z;

        // Get height with noise
        const float n_height = SimplexNoise::fractal(5, mountain_param, Vector3(x, 0.0f, z)).w;
        const float mountain_height = 300.0f + n_height * 50.0f;

        // SDF = y - height (negative below surface)
        const float terrain_sdf = y - mountain_height;

        // Low-frequency cave blob selector (sparse big voids)
        const float cave_blob = SimplexNoise::fractal(3, cave_blob_param, Vector3(x, y, z)).w;

        // Depth from surface (positive = below terrain)
        const float depth = mountain_height - y;
        
        // Only carve caves in blob regions and when deep enough
        const float blob_mask = (cave_blob + 0.3f) * 0.5f;  // Shift and scale to 0-0.8 range
        const float depth_fade = depth / 40.0f;
        const float cave_threshold = 0.5f + blob_mask * 0.4f + (1.0f - depth_fade) * 0.1f;
        
        // Simple cave field: just the blob
        const float cave_sdf = cave_threshold - cave_blob;

        // Carve caves: max(terrain, -caves)
        const float carved = MAX(terrain_sdf, -cave_sdf);

        return carved;
    }

    //GLSL implementation
    String get_glsl_code() const override {
        return R"glsl(
        vec4 terrain_sdf(vec3 pos){
            NoiseParam mountain_param;
            mountain_param.frequency = 0.0035;
            mountain_param.amplitude = 1.0;
            mountain_param.lacunarity = 2.0;
            mountain_param.persistence = 0.5;

            NoiseParam cave_blob_param;
            cave_blob_param.frequency = 0.0080;
            cave_blob_param.amplitude = 1.0;
            cave_blob_param.lacunarity = 2.0;
            cave_blob_param.persistence = 0.5;

            // Get height noise with gradients
            vec4 height_noise = fractal(5, mountain_param, vec3(pos.x, 0.0, pos.z));
            
            // Height = base + noise * amplitude
            float base_height = 300.0;
            float mountain_height = base_height + height_noise.w * 50.0;
            
            // SDF = y - height(x,z)
            // Gradient: (−∂height/∂x, 1, −∂height/∂z)
            vec3 grad_height = height_noise.xyz * 50.0;
            vec3 sdf_gradient = vec3(-grad_height.x, 1.0, -grad_height.z);
            float terrain_sdf = pos.y - mountain_height;
            
            // Low-frequency cave blob selector (sparse big voids)
            vec4 cave_blob_noise = fractal(3, cave_blob_param, vec3(pos.x, pos.y, pos.z));
            
            // Depth from surface (positive = below terrain)
            float depth = mountain_height - pos.y;
            
            // Only carve caves in blob regions and when deep enough
            float blob_mask = (cave_blob_noise.w + 0.3) * 0.5;
            float depth_fade = depth / 40.0;
            float cave_threshold = 0.5 + blob_mask * 0.4 + (1.0 - depth_fade) * 0.1;
            
            // Cave SDF: negative = cave interior
            float cave_sdf = cave_threshold - cave_blob_noise.w;
            
            // Carve caves: max(terrain, -caves)
            float carved = max(terrain_sdf, -cave_sdf);
            
            return vec4(sdf_gradient, carved);
        }
        )glsl";
    }

    // Repeated cube SDF + sinusoidal terrain term.
    // Positive means inside/solid (project convention).
    // float evaluate(const Vector3 &p_world_pos) const override {
    //     const float wave_x = std::sin(p_world_pos.x * 1.5f);
    //     const float wave_z = std::sin(p_world_pos.z * 1.2f);
    //     const float terrain = 5.0f - p_world_pos.y + (wave_x + wave_z);

    //     const float period = 6.0f;
    //     const Vector3 half_size(1.45f, 1.45f, 1.45f);

    //     const auto repeat_axis = [period](float v) {
    //         // Wrap axis to [-period/2, period/2] for tiled cells.
    //         return v - period * std::floor(v / period + 0.5f);
    //     };

    //     const Vector3 q(
    //         repeat_axis(p_world_pos.x),
    //         -5.0f+p_world_pos.y,
    //         repeat_axis(p_world_pos.z)
    //     );

    //     // Standard box SDF with negative-inside convention.
    //     const Vector3 d(
    //         ABS(q.x) - half_size.x,
    //         ABS(q.y) - half_size.y,
    //         ABS(q.z) - half_size.z
    //     );

    //     const Vector3 d_max(
    //         MAX(d.x, 0.0f),
    //         MAX(d.y, 0.0f),
    //         MAX(d.z, 0.0f)
    //     );

    //     const float outside = d_max.length();
    //     const float inside = MIN(MAX(d.x, MAX(d.y, d.z)), 0.0f);
    //     const float sdf_negative_inside = outside + inside;
    //     const float repeated_cube = -sdf_negative_inside;

    //     // Union: keep either terrain or repeated cubes as solid.
    //     return MAX(terrain, repeated_cube);
    // }
    
    // Sinusoidal blob idea:
    // float evaluate(const Vector3 &p_world_pos) const override {
    //     const float r = p_world_pos.length();
    //     const float wobble = 0.35f * std::sin(p_world_pos.x * 2.0f)
    //         * std::sin(p_world_pos.y * 2.0f)
    //         * std::sin(p_world_pos.z * 2.0f);
    //     return r - (3.0f + wobble);
    // }

    // Periodic sphere SDF using modulo-like coordinate wrapping.
    
    // float evaluate(const Vector3 &p_world_pos) const override {
    //     const float period = 3.0f;
    //     const float radius = 0.5f;

    //     const auto repeat_axis = [period](float v) {
    //         // Wrap to [-period/2, period/2] to repeat the same cell in space.
    //         return v - period * std::floor(v / period + 0.5f);
    //     };

    //     const Vector3 q(
    //         repeat_axis(p_world_pos.x),
    //         repeat_axis(p_world_pos.y),
    //         repeat_axis(p_world_pos.z)
    //     );

    //     return -q.length() + radius;
    // }


    //GLSL implementation
    // String get_glsl_code() const override {
    //     return R"(
    //     vec4 terrain_sdf(vec3 pos){
    //         return vec4(0,1,0,pos.y-350.0);  // Dummy: plane
    //     }
    //     )";
    // }


// V1 Mountain 

    //float evaluate(const Vector3 &p_world_pos) const override {
    //     // 2D mountain shape (XZ), with ridged + fBm detail.
    //     static const SimplexNoise::NoiseParam mountain_param{0.0035f, 1.0f, 2.0f, 0.5f};
    //     static const SimplexNoise::NoiseParam detail_param{0.0100f, 1.0f, 2.0f, 0.5f};
    //     static const SimplexNoise::NoiseParam cave_param{0.0300f, 1.0f, 2.0f, 0.5f};

    //     const float x = p_world_pos.x;
    //     const float y = p_world_pos.y;
    //     const float z = p_world_pos.z;

    //     const float n_base = SimplexNoise::fractal(5, mountain_param, Vector3(x, 0.0f, z)).w;      // [-1, 1] approx
    //     const float n_ridge = 1.0f - Math::abs(n_base);          // [0, 1]
    //     const float n_detail = SimplexNoise::fractal(4, detail_param, Vector3(x + 137.0f, 0.0f, z - 71.0f)).w;

    //     // Mountain height profile.
    //     const float base_level = 300.0f;
    //     const float mountain_height =
    //         base_level
    //         + n_ridge * 55.0f
    //         + n_detail * 8.0f;

    //     // Terrain SDF: negative below terrain surface (solid).
    //     const float terrain_sdf = y - mountain_height;

    //     // 3D cave field.
    //     const float cave_n = SimplexNoise::fractal(4, cave_param, Vector3(x, y * 0.8f, z)).w;

    //     // Keep caves mostly underground (fade out near surface).
    //     const float depth = mountain_height - y; // >0 below terrain
    //     const float depth_mask = Math::clamp((depth - 6.0f) / 18.0f, 0.0f, 1.0f);

    //     // Cave threshold: higher threshold -> fewer caves.
    //     const float cave_threshold = 0.45f + (1.0f - depth_mask) * 0.2f;
    //     const float cave_sdf = cave_threshold - cave_n; // negative = cave interior

    //     // Subtract caves from terrain: A \ B => max(A, -B)
    //     const float carved = MAX(terrain_sdf, -cave_sdf);

    //     return carved;
    // }

    // //GLSL implementation
    // String get_glsl_code() const override {
    //     return R"glsl(
    //     vec4 terrain_sdf(vec3 pos){
    //         NoiseParam mountain_param;
    //         mountain_param.frequency = 0.0035;
    //         mountain_param.amplitude = 1.0;
    //         mountain_param.lacunarity = 2.0;
    //         mountain_param.persistence = 0.5;

    //         NoiseParam detail_param;
    //         detail_param.frequency = 0.0100;
    //         detail_param.amplitude = 1.0;
    //         detail_param.lacunarity = 2.0;
    //         detail_param.persistence = 0.5;

    //         NoiseParam cave_param;
    //         cave_param.frequency = 0.0300;
    //         cave_param.amplitude = 1.0;
    //         cave_param.lacunarity = 2.0;
    //         cave_param.persistence = 0.5;

    //         float x = pos.x;
    //         float y = pos.y;
    //         float z = pos.z;

    //         vec4 n_base = fractal(5, mountain_param, vec3(x, 0.0, z));
    //         vec4 n_ridge = sdf_constant(1.0) - sdf_abs(n_base);
    //         vec4 n_detail = fractal(4, detail_param, vec3(x + 137.0, 0.0, z - 71.0));

    //         vec4 base_level = sdf_constant(300.0);
    //         vec4 mountain_height = base_level + n_ridge * 55.0 + n_detail * 8.0;

    //         vec4 terrain_sdf = sdf_constant(y) - mountain_height;

    //         vec4 cave_n = fractal(4, cave_param, vec3(x, y * 0.8, z));

    //         vec4 depth = mountain_height - sdf_constant(y);
    //         float depth_mask = clamp((depth.w - 6.0) / 18.0, 0.0, 1.0);
    //         float cave_threshold = 0.45 + (1.0 - depth_mask) * 0.2;

    //         vec4 cave_sdf = sdf_constant(cave_threshold) - cave_n;

    //         vec4 carved = sdf_max(terrain_sdf, -cave_sdf);

    //         return carved;
    //     }
    //     )glsl";
    // }

    //V2

    // float evaluate(const Vector3 &p_world_pos) const override {
    //     // Mountain: simple height-based SDF with proper gradients
    //     static const SimplexNoise::NoiseParam mountain_param{0.0035f, 1.0f, 2.0f, 0.5f};

    //     const float x = p_world_pos.x;
    //     const float y = p_world_pos.y;
    //     const float z = p_world_pos.z;

    //     // Get height with noise
    //     const float n_height = SimplexNoise::fractal(5, mountain_param, Vector3(x, 0.0f, z)).w;
    //     const float mountain_height = 300.0f + n_height * 50.0f;

    //     // SDF = y - height (negative below surface)
    //     return y - mountain_height;
    // }

    // //GLSL implementation
    // String get_glsl_code() const override {
    //     return R"glsl(
    //     vec4 terrain_sdf(vec3 pos){
    //         NoiseParam mountain_param;
    //         mountain_param.frequency = 0.0035;
    //         mountain_param.amplitude = 1.0;
    //         mountain_param.lacunarity = 2.0;
    //         mountain_param.persistence = 0.5;

    //         // Get height noise with gradients
    //         vec4 height_noise = fractal(5, mountain_param, vec3(pos.x, 0.0, pos.z));
            
    //         // Height = base + noise * amplitude
    //         float base_height = 300.0;
    //         float mountain_height = base_height + height_noise.w * 50.0;
            
    //         // SDF = y - height(x,z)
    //         // Gradient: (−∂height/∂x, 1, −∂height/∂z)
    //         vec3 grad_height = height_noise.xyz * 50.0;
    //         vec3 sdf_gradient = vec3(-grad_height.x, 1.0, -grad_height.z);
    //         float sdf_value = pos.y - mountain_height;
            
    //         return vec4(sdf_gradient, sdf_value);
    //     }
    //     )glsl";
    // }

};
