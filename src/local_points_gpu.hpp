#pragma once

#include "cascade_points_gpu.hpp"
#include "chunk_math.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

using namespace godot;

// A local RenderingDevice with the cascade's point passes (CascadePointsGPU) on it: the discoverer's corners
// (ChunkDiscoverer, VoxelTerrain.discoverer_mode) and the site walks' field (FieldGPU, VoxelTerrain.walk_gpu).
//
// One thread creates it, uses it and frees it (the discoverer's, the field service's): no main-RD resource and no
// render-thread work. Batches above the points it was created for are split; the scratch is sized for them.
class LocalPointsGPU {
public:
    // On the thread that will use it. False (nothing kept) if there is no local device (headless, Compatibility
    // renderer) or the passes fail to build.
    bool create(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points);
    void free();
    bool is_valid() const { return rd != nullptr && passes.is_valid(); }
    CascadePointsGPU &points() { return passes; }
    uint64_t scratch_bytes() const { return passes.scratch_bytes(); }

private:
    RenderingDevice *rd = nullptr;
    CascadePointsGPU passes;
};
