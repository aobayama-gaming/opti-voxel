#pragma once

#include "cascade_points_gpu.hpp"
#include "chunk_math.hpp"
#include "gpu_device.hpp"
#include "local_gpu_thread.hpp"
#include "sdf_base.hpp"

#include <memory>

// The cascade's point passes (CascadePointsGPU) on a local Device: the discoverer's corners (ChunkDiscoverer,
// VoxelTerrain.discoverer_mode) and the site walks' field (FieldGPU, VoxelTerrain.walk_gpu).
//
// Either on a device of its own (create without a thread: one thread creates it, uses it and frees it, the field
// service's), or on a LocalGpuThread's (the discoverer's, kept across a world's restart): create, cull_values and free
// then run on that thread, the calling one waiting, and the passes freed park on its device for the next create.
// No main-RD resource and no render-thread work. Batches above the points it was created for are split; the scratch is
// sized for them.
class LocalPointsGPU {
public:
    // On the thread that will use it (p_thread: any). False (nothing kept) if there is no local device (headless,
    // Compatibility renderer) or the passes fail to build.
    bool create(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points, LocalGpuThread *p_thread = nullptr);
    void free();
    bool is_valid() const { return passes.is_valid(); }
    // The passes, called on the device's thread: a device of its own only.
    CascadePointsGPU &points() { return passes; }
    // CascadePointsGPU::cull_values, on the device's thread.
    bool cull_values(const VoxelPos *p_points, int p_count, int p_lod, float *r_out, CascadePointsGPU::BatchStats &r_stats);
    uint64_t scratch_bytes() const { return passes.scratch_bytes(); }
    // A readback's round trip on its device, in milliseconds: the fastest of three of a one-word buffer, after a warm-up.
    // A device of its own only.
    double readback_ms();

private:
    std::unique_ptr<ov::gpu::Device> device; // a device of its own
    LocalGpuThread *thread = nullptr; // else the passes on this one's
    CascadePointsGPU passes;
};
