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
    // provides the chunk_param uniform, the frame functions of gpu_frame.hpp and the exact noise of
    // exact_noise.hpp.
    virtual String get_glsl_code() const = 0;

protected:
    ChunkParam param;
};
