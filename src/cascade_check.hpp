#pragma once

#include "gpu_device.hpp"

#include "ov/value.hpp"

// The GPU cascade (cascade_glsl.hpp) against the CPU oracle (Cascade::run), with the meshers' level passes, on any
// device:
//   - G1: every channel of every level and the final sdf_sample of chunks of every LOD (>= 64 per LOD,
//     near 0, +-2^60 and straddling 2^32-voxel boundaries, in mixed-LOD batches): GPU == CPU bit for bit.
//     The passes run one level at a time and each level's scratch is read back before the next one.
//   - G3: on the GPU outputs, at the points a LOD-L chunk shares with its parent (LOD L+1):
//     V_L = flush(V_{L+1} * 2^-order + O_L), exact.
//   - G6 / G7 timing: single-LOD batches of 128 chunks at LOD 0, 4, ... 24, every level of the cascade
//     passes, both channel sets, interleaved runs (GPU timestamps when the device returns them, wall clock
//     of submit + sync otherwise).
//   - D1 / D2: point items (cascade_check_points.hpp, the discoverer's corners and the site walk's readings):
//     ground_unit / ground_field / evaluate_cull bit for bit, and their GPU ms per batch size.
// Godot: CascadeSpike (tools/cascade_test.sh, needs a window).
namespace CascadeCheck {

// p_device: a local device (synced). p_options: lods (25), voxel (0.5), check (true), table (true), biomes (true),
// points (true), timing (true), runs (10), sets ("8,4,4c").
ov::Value run(ov::gpu::Device &p_device, const ov::Value &p_options);

} // namespace CascadeCheck
