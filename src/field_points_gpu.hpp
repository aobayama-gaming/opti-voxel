#pragma once

#include "cascade_glsl.hpp"
#include "exact_noise.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

#include <vector>

using namespace godot;

// The cascade ground's field (SDFCascade::ground_field) at many points on a RenderingDevice: the cascade passes with
// field items (CascadePasses::create_fields), any point of Λ_lod at any LOD in one batch, bit for bit
// SDFCascade::ground_unit, extended to the asked voxel on the CPU (SDFCascade::extend_field) as ground_field does.
//
// It does not own the device: its user creates it, uses it and frees it on the device's thread (the cascade passes are
// not thread-safe). Per batch: upload the items, record every level, submit() + sync(), read back 16 B per point.
// Batches above max_points() are split; the scratch is sized for max_points().
class FieldPointsGPU {
public:
    // One evaluate call.
    struct BatchStats {
        int batches = 0; // submissions (at most max_points() points each)
        int64_t points = 0;
        double gpu_ms = 0.0; // GPU timestamps around the levels, summed over the batches; -1 if one did not come back
        int64_t readback_bytes = 0;
    };

    // p_sdf: a cascade terrain (false otherwise), configured; p_max_points: the points per batch (the scratch grows
    // with it, CascadeGLSL::field_points_for_scratch). False (nothing kept) if the passes fail to build.
    bool create(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points);
    void free();
    bool is_valid() const { return rd != nullptr && passes.is_valid(); }
    uint32_t max_points() const { return max_batch; }
    uint64_t scratch_bytes() const { return passes.scratch_bytes(); }

    // The unit registers (SDFCascade::ground_unit) at p_count lattice points, point i of Λ_{p_lods[i]} (a multiple of
    // 2^lod). False (with an error) if a point is off its lattice or the device fails: r_out is then incomplete.
    bool units(const VoxelPos *p_points, const int *p_lods, int p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats);
    // SDFCascade::ground_field at p_count LOD0 voxels, voxel i as LOD p_lods[i] sees it, bit for bit.
    bool ground_field(const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats);

private:
    RenderingDevice *rd = nullptr;
    uint32_t max_batch = 0;
    RID input;
    RID param_ubo;
    RID values;
    CascadePasses passes;
    PackedByteArray input_bytes;
    std::vector<VoxelPos> lattice;
    uint64_t batch_serial = 0;

    bool _run_batch(const VoxelPos *p_points, const int *p_lods, uint32_t p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats);
};
