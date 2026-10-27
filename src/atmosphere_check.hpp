#pragma once

#include "gpu_device.hpp"

#include "ov/value.hpp"

// The atmosphere's passes on any device against a CPU reference in double (the same march, sample for sample): the
// multiple-scattering table, every texel, and columns of the camera volume, every slice, for a flat base, Earth at its
// pole and a 2 km planet, the sun low. Half floats and float32 sums: within ATMOSPHERE_TOLERANCE, relative.
// Godot: FrameSpike (tools/frame_test.sh), after FrameCheck.
namespace AtmosphereCheck {

inline constexpr double TOLERANCE = 0.02;

// p_device: a local device (synced). The report: per case the worst relative errors and the checked texels, "ok".
ov::Value run(ov::gpu::Device &p_device);

} // namespace AtmosphereCheck
