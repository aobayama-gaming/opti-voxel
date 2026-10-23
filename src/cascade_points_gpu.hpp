#pragma once

#include "cascade_glsl.hpp"
#include "exact_noise.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

#include <vector>

using namespace godot;

// The bare cascade at many points on a RenderingDevice: the cascade passes with point items (CascadePasses::create_points),
// any point of Λ_lod at any LOD in one batch, bit for bit SDFCascade::ground_unit. From those registers, on the CPU:
// SDFBase::ground_field (extended to the asked voxel, the site walk's readings) and evaluate_cull (the distance in metres,
// the discoverer's corners).
//
// It does not own the device: its user creates it, uses it and frees it on the device's thread (the cascade passes are
// not thread-safe). Per batch: upload the items, record every level, submit() + sync(), read back 16 B per point.
// Batches above max_points are split; the scratch is sized for them.
class CascadePointsGPU {
public:
    // A batch's registers as buffer_get_data_async delivers them (the serial of the batch they belong to).
    struct Delivery {
        PackedByteArray bytes;
        uint64_t serial = 0;
    };

    // One call.
    struct BatchStats {
        int batches = 0; // submissions (at most max_points points each)
        double gpu_ms = 0.0; // GPU timestamps around the levels, summed over the batches; -1 if one did not come back
        int64_t readback_bytes = 0;
        // Wall clock of the batches' steps on the calling thread: encode + upload + record, submit + sync, readback.
        double record_ms = 0.0;
        double sync_ms = 0.0;
        double readback_ms = 0.0;
    };

    // p_sdf: a cascade terrain (false otherwise), configured with p_param; p_max_points: the points per batch (the
    // scratch grows with it, and with the climate's field groups). False (nothing kept) if the passes fail to build.
    bool create(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points);
    void free();
    bool is_valid() const { return rd != nullptr && passes.is_valid(); }
    uint64_t scratch_bytes() const { return passes.scratch_bytes(); }

    // The unit registers (SDFCascade::ground_unit) at p_count lattice points, point i of Λ_{p_lods[i]} (a multiple of
    // 2^lod). False (with an error) if a point is off its lattice or the device fails: r_out is then incomplete.
    bool units(const VoxelPos *p_points, const int *p_lods, int p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats);
    // SDFBase::ground_field at p_count LOD0 voxels, voxel i as LOD p_lods[i] sees it, bit for bit.
    bool ground_field(const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats);
    // SDFBase::evaluate_cull (the bare cascade, metres) at p_count points of Λ_lod, bit for bit.
    bool cull_values(const VoxelPos *p_points, int p_count, int p_lod, float *r_out, BatchStats &r_stats);

private:
    RenderingDevice *rd = nullptr;
    float voxel_size = 0.0f;
    uint32_t max_batch = 0;
    RID input;
    RID param_ubo;
    RID values;
    CascadePasses passes;
    PackedByteArray input_bytes;
    uint64_t batch_serial = 0;
    Delivery delivery;
    bool async_readback = true; // false once a readback didn't come with its sync: buffer_get_data

    bool _run_batch(const VoxelPos *p_points, const int *p_lods, uint32_t p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats);
};
