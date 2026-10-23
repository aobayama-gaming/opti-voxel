#pragma once

#include "cascade_glsl.hpp"
#include "chunk_math.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

using namespace godot;

// The discoverer's corner SDF on the GPU (VoxelTerrain.discoverer_mode 1, the default): the cascade
// passes with point items (CascadePasses::create_points), bit for bit SDFBase::evaluate at the same corners.
//
// It owns a local RenderingDevice, created and freed by the discoverer thread (ChunkDiscoverer::discovery_loop,
// before its first scan and before it exits) and touched by no other thread: no main-RD resource and no
// render-thread work. Per batch: upload the points, record every level, submit() + sync(), read back one float
// per point (buffer_get_data). Batches above MAX_POINTS are split; the scratch is sized for MAX_POINTS.
class DiscovererGPU {
public:
    static constexpr uint32_t MAX_POINTS = 65536;

    // One evaluate() call (a discoverer wave).
    struct BatchStats {
        int batches = 0; // submissions (at most MAX_POINTS points each)
        double gpu_ms = 0.0; // GPU timestamps around the levels, summed over the batches; -1 if one did not come back
        int64_t readback_bytes = 0;
    };

    // On the thread that will use it. False (nothing kept) if there is no local
    // device (headless, Compatibility renderer) or the passes fail to build.
    bool create(const SDFBase *p_sdf, const ChunkParam &p_param);
    void free();
    bool is_valid() const { return rd != nullptr && passes.is_valid(); }

    // The SDF (metres) at p_count corners of LOD p_lod's corner grid (ChunkDiscoverer's node corners). False
    // (with an error) if a point is off that grid: r_out is then incomplete and the caller must use the CPU.
    bool evaluate(const VoxelPos *p_points, int p_count, int p_lod, float *r_out, BatchStats &r_stats);

    uint64_t scratch_bytes() const { return passes.scratch_bytes(); }

private:
    RenderingDevice *rd = nullptr;
    ChunkParam point_param;
    RID input;
    RID param_ubo;
    RID values;
    CascadePasses passes;
    PackedByteArray input_bytes;
    uint64_t batch_serial = 0;

    bool _run_batch(const VoxelPos *p_points, uint32_t p_count, int p_lod, float *r_out, BatchStats &r_stats);
};
