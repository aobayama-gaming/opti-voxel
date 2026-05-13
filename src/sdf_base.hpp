#pragma once

#include "godot_cpp/variant/string.hpp"
#include "godot_cpp/variant/vector3.hpp"

using namespace godot;

class SDFBase {
public:
    virtual ~SDFBase() = default;
    
    // CPU evaluation: given a world position, return signed distance
    virtual float evaluate(const Vector3 &p_world_pos) const = 0;
    
    // Compute normal at position using central difference approximation
    // Default implementation uses alpha = VOXEL_SIZE / 10 should be suplented with mathematical formulation if possible/easy.
    virtual Vector3 evaluate_normal(const Vector3 &p_world_pos,float alpha=1e-4f) const {

        const Vector3 dx(alpha, 0.0f, 0.0f);
        const Vector3 dy(0.0f, alpha, 0.0f);
        const Vector3 dz(0.0f, 0.0f, alpha);
        
        const float grad_x = evaluate(p_world_pos + dx) - evaluate(p_world_pos - dx);
        const float grad_y = evaluate(p_world_pos + dy) - evaluate(p_world_pos - dy);
        const float grad_z = evaluate(p_world_pos + dz) - evaluate(p_world_pos - dz);
        
        return Vector3(grad_x, grad_y, grad_z).normalized();
    }
    
    // GPU code: GLSL function that evaluates SDF
    // Should return a function named "vec4 terrain_sdf(vec3 pos)"
    virtual String get_glsl_code() const = 0;
};
