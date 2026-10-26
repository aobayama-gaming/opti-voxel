#pragma once

// The Dawn adapter's device: one instance, adapter, device and queue, headless (no surface).

#include <dawn/webgpu_cpp.h>

namespace ov::dawn {

struct DawnContext {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;

    // False (and an error logged) when no WebGPU adapter or device is available.
    bool initialize();
};

} // namespace ov::dawn
