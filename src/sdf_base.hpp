#pragma once

#include "godot_cpp/variant/string.hpp"
#include "godot_cpp/variant/vector3.hpp"

#include "chunk_math.hpp"

using namespace godot;

class SDFBase {
public:
    virtual ~SDFBase() = default;

    // Called once before any thread uses the SDF (VoxelTerrain::_ready).
    virtual void configure(const ChunkParam &p_param) { param = p_param; }

    // CPU evaluation at an absolute position (integer frame): signed distance, negative inside.
    virtual float evaluate(const WorldPos &p_pos) const = 0;

    // GPU code: must define "vec4 terrain_sdf(WorldPos p)" (xyz gradient, w distance). The shader
    // provides the chunk_param uniform and the frame functions of gpu_frame.hpp.
    virtual String get_glsl_code() const = 0;

protected:
    ChunkParam param;
};

// Adapter for SDFs written in metres around the world origin (the convention before the integer
// frame): they are evaluated at the origin-relative position, so their terrain follows the origin.
// Goes away with the exact noise.
class SDFMetreBridge : public SDFBase {
public:
    float evaluate(const WorldPos &p_pos) const final {
        return evaluate_metres(ChunkMath::legacy_metres(p_pos, param));
    }

    String get_glsl_code() const final {
        return get_metre_glsl_code() + String(R"glsl(
vec4 terrain_sdf(WorldPos p) {
    return legacy_terrain_sdf(legacy_sdf_metres(p, chunk_param));
}
)glsl");
    }

protected:
    virtual float evaluate_metres(const Vector3 &p_metres) const = 0;
    // Must define "vec4 legacy_terrain_sdf(vec3 pos)".
    virtual String get_metre_glsl_code() const = 0;
};
