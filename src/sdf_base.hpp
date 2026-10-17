#pragma once

#include "godot_cpp/variant/string.hpp"
#include "godot_cpp/variant/vector3.hpp"

#include "chunk_math.hpp"

using namespace godot;

namespace Cascade {
struct Spec;
}

class SDFBase {
public:
    virtual ~SDFBase() = default;

    // Called once before any thread uses the SDF (VoxelTerrain::_ready). Chunks exist at LODs
    // [0, p_lod_count).
    virtual void configure(const ChunkParam &p_param, int p_lod_count) {
        (void)p_lod_count;
        param = p_param;
    }

    // CPU evaluation at an absolute position (integer frame), for a chunk of LOD p_lod: signed
    // distance in metres, negative inside. The LOD only sets the unit of the internal gradients (per
    // voxel of that LOD), the distance is the same at every LOD.
    virtual float evaluate(const WorldPos &p_pos, int p_lod) const = 0;

    // GPU code: must define "vec4 terrain_sdf(WorldPos p, int lod)" (xyz gradient, w distance). The
    // shader provides the chunk_param uniform, the frame functions of gpu_frame.hpp and the exact
    // noise of exact_noise.hpp.
    virtual String get_glsl_code() const = 0;

    // The Hermite cascade's bands, if this terrain is one (SDFCascade): the meshers then compute the
    // samples with the cascade passes (cascade_glsl.hpp) instead of calling terrain_sdf, and evaluate()
    // only takes lattice points (frac 0, voxel a multiple of 2^lod).
    virtual const Cascade::Spec *get_cascade() const { return nullptr; }

    // Depth (metres) of the skirt of a LOD-p_lod chunk: how far past its bounds the SDF is pushed towards
    // air (gpu_glsl.hpp skirt_sdf). By default skirt_depth voxels of the LOD; SDFCascade sizes it from its
    // octave amplitudes.
    virtual float skirt_depth_m(int p_lod) const {
        return param.skirt_depth * param.voxel_size * float(int64_t(1) << p_lod);
    }

protected:
    ChunkParam param;
};
