#pragma once

#include "sdf_base.hpp"
#include <cmath>

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

    // Repeated cube SDF + sinusoidal terrain term.
    // Positive means inside/solid (project convention).
    float evaluate(const Vector3 &p_world_pos) const override {
        const float wave_x = std::sin(p_world_pos.x * 1.5f);
        const float wave_z = std::sin(p_world_pos.z * 1.2f);
        const float terrain = - p_world_pos.y + (wave_x + wave_z);

        const float period = 6.0f;
        const Vector3 half_size(1.45f, 1.45f, 1.45f);

        const auto repeat_axis = [period](float v) {
            // Wrap axis to [-period/2, period/2] for tiled cells.
            return v - period * std::floor(v / period + 0.5f);
        };

        const Vector3 q(
            repeat_axis(p_world_pos.x),
            p_world_pos.y,
            repeat_axis(p_world_pos.z)
        );

        // Standard box SDF with negative-inside convention.
        const Vector3 d(
            ABS(q.x) - half_size.x,
            ABS(q.y) - half_size.y,
            ABS(q.z) - half_size.z
        );

        const Vector3 d_max(
            MAX(d.x, 0.0f),
            MAX(d.y, 0.0f),
            MAX(d.z, 0.0f)
        );

        const float outside = d_max.length();
        const float inside = MIN(MAX(d.x, MAX(d.y, d.z)), 0.0f);
        const float sdf_negative_inside = outside + inside;
        const float repeated_cube = -sdf_negative_inside;

        // Union: keep either terrain or repeated cubes as solid.
        return MAX(terrain, repeated_cube);
    }
    
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
