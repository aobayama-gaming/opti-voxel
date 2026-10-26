#pragma once

#include "gpu_device.hpp"

#include "cascade_check_sets.hpp"

#include <vector>

// CascadeSpike's point items (CascadePasses::create_points through CascadePointsGPU), on its local device:
//   - D1: one mixed-LOD batch (LODs 0-8, 12, 16, 24 below p_lods), 5 places x 1024 voxels per LOD (near 0, +-2^60,
//     straddling 2^32-voxel boundaries, around lattice index 2^32), any voxel (odd lattice indices included): the GPU's
//     unit register at the nearest lattice point == SDFCascade::ground_unit, bit for bit, and from it on the CPU
//     ground_field (the site walk's readings) and the metres value == evaluate_cull (the discoverer's corners).
//   - D2 timing: GPU ms of single-LOD batches of 1k, 4k, 16k and 64k points at LOD 0 and 5 (the site walk's snap and
//     its coarse readings at 5 cm): the batch size a GPU site walk is sized from.
namespace CascadeCheckPoints {

ov::Value check_points(ov::gpu::Device *p_device, CascadeCheckSets::SpikeSet p_set, int p_lods, const ChunkParam &p_param);
ov::Value point_timing(ov::gpu::Device *p_device, const std::vector<CascadeCheckSets::SpikeSet> &p_sets, int p_lods, const ChunkParam &p_param,
        int p_runs);

} // namespace CascadeCheckPoints
