#pragma once

#include "cascade_spike_sets.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <vector>

// CascadeSpike's field points (CascadePasses::create_fields through FieldPointsGPU), on its local device:
//   - F1: one mixed-LOD batch (LODs 0-8, 12, 16, 24 below p_lods), 5 places x 1024 voxels per LOD (near 0, +-2^60,
//     straddling 2^32-voxel boundaries, around lattice index 2^32), any voxel (odd lattice indices included): the GPU's
//     unit register at the nearest lattice point == SDFCascade::ground_unit, and the extended reading ==
//     SDFCascade::ground_field, every component bit for bit.
//   - F2 timing: GPU ms of single-LOD batches of 1k, 4k, 16k and 64k points at LOD 0 and 5 (the site walk's snap and
//     its coarse readings at 5 cm), the batch size the site walk's GPU service is sized from.
namespace CascadeSpikeFields {

godot::Dictionary check_fields(godot::RenderingDevice *p_rd, CascadeSpikeSets::SpikeSet p_set, int p_lods, const ChunkParam &p_param);
godot::Dictionary field_timing(godot::RenderingDevice *p_rd, const std::vector<CascadeSpikeSets::SpikeSet> &p_sets, int p_lods, const ChunkParam &p_param,
        int p_runs);

} // namespace CascadeSpikeFields
