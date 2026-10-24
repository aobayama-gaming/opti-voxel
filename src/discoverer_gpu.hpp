#pragma once

#include "cascade_points_gpu.hpp"
#include "chunk_math.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

using namespace godot;

// The discoverer's corner SDF on the GPU (VoxelTerrain.discoverer_mode 1, the default): the cascade's point passes
// (CascadePointsGPU), bit for bit SDFBase::evaluate_cull at the same corners.
//
// It owns a local RenderingDevice, created and freed by the discoverer thread (ChunkDiscoverer::discovery_loop,
// before its first scan and before it exits) and touched by no other thread: no main-RD resource and no
// render-thread work. Batches above MAX_POINTS are split; the scratch is sized for MAX_POINTS.
class DiscovererGPU {
public:
    static constexpr uint32_t MAX_POINTS = 65536;

    // One evaluate() call (a discoverer wave).
    using BatchStats = CascadePointsGPU::BatchStats;

    // On the thread that will use it. False (nothing kept) if there is no local
    // device (headless, Compatibility renderer) or the passes fail to build.
    bool create(const SDFBase *p_sdf, const ChunkParam &p_param);
    void free();
    bool is_valid() const { return rd != nullptr && points.is_valid(); }

    // The SDF (metres) at p_count points of Λ_{p_lod} (ChunkDiscoverer's node corners). False (with an error) if a
    // point is off that lattice or the device fails: r_out is then incomplete and the caller must use the CPU.
    bool evaluate(const VoxelPos *p_points, int p_count, int p_lod, float *r_out, BatchStats &r_stats) {
        ERR_FAIL_COND_V(!is_valid(), false);
        return points.cull_values(p_points, p_count, p_lod, r_out, r_stats);
    }

    uint64_t scratch_bytes() const { return points.scratch_bytes(); }

private:
    RenderingDevice *rd = nullptr;
    CascadePointsGPU points;
};
