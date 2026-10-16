#include "gpu_profiler.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

GPUProfiler::GPUProfiler() {
    mutex.instantiate();
}

void GPUProfiler::poll() {
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUProfiler::_rt_read));
}

void GPUProfiler::_rt_read() {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    if (!rd) {
        return;
    }
    const uint32_t n = rd->get_captured_timestamps_count();
    if (n < 2) {
        return;
    }
    Dictionary d;
    int64_t prev = rd->get_captured_timestamp_gpu_time(0);
    const int64_t first = prev;
    for (uint32_t i = 1; i < n; i++) {
        const int64_t t = rd->get_captured_timestamp_gpu_time(i);
        const String name = rd->get_captured_timestamp_name(i);
        // Same name can repeat (e.g. per viewport/cascade): accumulate. GPU time is in nanoseconds.
        const double ms = (t - prev) / 1e6;
        d[name] = (double)d.get(name, 0.0) + ms;
        prev = t;
    }
    d["total_ms"] = (prev - first) / 1e6;
    mutex->lock();
    breakdown = d;
    mutex->unlock();
}

Dictionary GPUProfiler::get_breakdown() const {
    mutex->lock();
    Dictionary d = breakdown.duplicate();
    mutex->unlock();
    return d;
}
