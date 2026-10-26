#pragma once

#include "gpu_device.hpp"

#include "ov/value.hpp"

// GPU vs CPU check of the integer frame (world_coord.hpp, gpu_frame.hpp) and the exact noise (exact_noise.hpp), on any
// device: builds WorldPos and evaluates the probes on the GPU (the same GLSL snippets as the meshers) for test vectors
// around far world origins, 2^32-voxel boundaries and absolute points, and checks against the CPU mirrors:
//   - CPU sample_pos == GPU WorldPos and anchor, bit for bit
//   - every probe of the probe table (frame_check.cpp: hash, octave decomposition, perlin3/2, a 2D and a 3D fBm;
//     archetype_probes.cpp: the archetype IR's shapes, the table probe and fuzzed programs, generated GLSL vs the
//     interpreter): CPU == GPU, bit for bit, at the fixed grid and 1,024 hash-random points per origin
//   - the same voxel reached from two chunks (neighbour, parent) gives bit-identical GPU results
//   - the same absolute point gives bit-identical GPU results at every world origin
//   - the noises stay continuous along lines across voxel and cell low-word wraps
// then GPUDeviceCheck on the same device. Godot: FrameSpike (tools/frame_test.sh); Dawn: ov_dawn --frame-check.
namespace FrameCheck {

// p_device: a local device (synced). The report: counters, examples, "ok".
ov::Value run(ov::gpu::Device &p_device, float p_voxel_size);

} // namespace FrameCheck
