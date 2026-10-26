#pragma once

#include "gpu_device.hpp"

// A backend's self-check, the same code on every adapter (Dawn: ov_dawn --check; Godot: FrameSpike, frame_test.sh):
// the device's plumbing (uploads, clear, copy, uniform, push constants, an unused binding, sync and async readbacks;
// textures: a storage image written, a region copied, an update and a clear read back),
// then the engine's exact noise on the GPU against the CPU, bit for bit, at hash-random points of far origins and
// every LOD of PROBE_LODS.
namespace GPUDeviceCheck {

struct Report {
    ov::str error; // a pipeline that did not compile, a readback that did not come
    int64_t plumbing_checked = 0;
    int64_t plumbing_mismatch = 0;
    int64_t noise_checked = 0;
    int64_t noise_mismatch = 0;
    ov::str first_mismatch;

    bool ok() const { return error.empty() && plumbing_mismatch == 0 && noise_mismatch == 0 && plumbing_checked > 0 && noise_checked > 0; }
    ov::str summary() const;
};

// p_device: a local device (the check calls sync()).
Report run(ov::gpu::Device &p_device);

} // namespace GPUDeviceCheck
