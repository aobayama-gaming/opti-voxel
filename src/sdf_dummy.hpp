#pragma once

#include "sdf_base.hpp"
#include <cmath>
#include "SimplexNoise.h"

class SDFDummy : public SDFBase {
public:
    SDFDummy() = default;
    ~SDFDummy() override = default;
    
    //plane
    // float evaluate(const Vector3 &p_world_pos) const override {
    //     return -p_world_pos.y - 1.0f;
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
        // 2D mountain shape (XZ), with ridged + fBm detail.
        static const SimplexNoise mountain_fbm(0.0035f, 1.0f, 2.0f, 0.5f);
        static const SimplexNoise detail_fbm(0.0100f, 1.0f, 2.0f, 0.5f);
        static const SimplexNoise cave_fbm(0.0300f, 1.0f, 2.0f, 0.5f);

        const float x = p_world_pos.x;
        const float y = p_world_pos.y;
        const float z = p_world_pos.z;

        const float n_base = mountain_fbm.fractal(5, x, z);      // [-1, 1] approx
        const float n_ridge = 1.0f - Math::abs(n_base);          // [0, 1]
        const float n_detail = detail_fbm.fractal(4, x + 137.0f, z - 71.0f);

        // Mountain height profile.
        const float base_level = 300.0f;
        const float mountain_height =
            base_level
            + n_ridge * 55.0f
            + n_detail * 8.0f;

        // Terrain SDF: negative below terrain surface (solid).
        const float terrain_sdf = y - mountain_height;

        // 3D cave field.
        const float cave_n = cave_fbm.fractal(4, x, y * 0.8f, z);

        // Keep caves mostly underground (fade out near surface).
        const float depth = mountain_height - y; // >0 below terrain
        const float depth_mask = Math::clamp((depth - 6.0f) / 18.0f, 0.0f, 1.0f);

        // Cave threshold: higher threshold -> fewer caves.
        const float cave_threshold = 0.45f + (1.0f - depth_mask) * 0.2f;
        const float cave_sdf = cave_threshold - cave_n; // negative = cave interior

        // Subtract caves from terrain: A \ B => max(A, -B)
        const float carved = MAX(terrain_sdf, -cave_sdf);

        return -carved;
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

    // GLSL implementation
    String get_glsl_code() const override {
        return R"(
float evaluate_sdf(vec3 pos) {
    return 1.0;  // Dummy: always solid
}
        )";
    }
};
