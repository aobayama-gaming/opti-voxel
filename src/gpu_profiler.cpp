#include "gpu_profiler.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

namespace {

bool is_mesher_name(const String &p_name) {
    return p_name.begins_with("vx") && p_name.contains("_");
}

String section_name(const String &p_name) {
    return is_mesher_name(p_name) ? "vx_" + p_name.get_slice("_", 1) : p_name;
}

} // namespace

std::vector<String> gpu_interval_labels(const std::vector<String> &p_names) {
    std::vector<String> out;
    for (size_t i = 0; i + 1 < p_names.size(); i++) {
        const String &start = p_names[i];
        const String &end = p_names[i + 1];
        if (is_mesher_name(end) && !end.ends_with("_begin")) {
            out.push_back(section_name(end));
        } else if (is_mesher_name(start)) {
            out.push_back("(untracked)"); // after a batch, before the next renderer section
        } else {
            out.push_back(start);
        }
    }
    return out;
}

GPUProfiler::GPUProfiler() {
    mutex.instantiate();
}

void GPUProfiler::poll() {
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUProfiler::_rt_read));
}

void GPUProfiler::set_recording(bool p_recording) {
    mutex->lock();
    recording = p_recording;
    mutex->unlock();
}

Array GPUProfiler::take_frames() {
    mutex->lock();
    Array out = frames;
    frames = Array();
    mutex->unlock();
    return out;
}

void GPUProfiler::_rt_read() {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    if (!rd) {
        return;
    }
    const uint32_t n = rd->get_captured_timestamps_count();
    const uint64_t frame = rd->get_captured_timestamps_frame();
    if (n < 2 || frame == last_frame) {
        return;
    }
    last_frame = frame;
    std::vector<String> names(n);
    for (uint32_t i = 0; i < n; i++) {
        names[i] = rd->get_captured_timestamp_name(i);
    }
    const std::vector<String> labels = gpu_interval_labels(names);
    // A name can repeat (per viewport, per batch, "Tonemap" twice): sections accumulate. GPU time is in
    // nanoseconds, CPU time in microseconds.
    Dictionary gpu;
    Dictionary cpu;
    for (uint32_t i = 0; i + 1 < n; i++) {
        const double gpu_ms = (rd->get_captured_timestamp_gpu_time(i + 1) - rd->get_captured_timestamp_gpu_time(i)) / 1e6;
        const double cpu_ms = (rd->get_captured_timestamp_cpu_time(i + 1) - rd->get_captured_timestamp_cpu_time(i)) / 1e3;
        gpu[labels[i]] = double(gpu.get(labels[i], 0.0)) + gpu_ms;
        cpu[labels[i]] = double(cpu.get(labels[i], 0.0)) + cpu_ms;
    }
    const double gpu_total = (rd->get_captured_timestamp_gpu_time(n - 1) - rd->get_captured_timestamp_gpu_time(0)) / 1e6;
    const double cpu_total = (rd->get_captured_timestamp_cpu_time(n - 1) - rd->get_captured_timestamp_cpu_time(0)) / 1e3;

    Dictionary snapshot = gpu.duplicate();
    snapshot["total_ms"] = gpu_total;
    mutex->lock();
    breakdown = snapshot;
    if (recording) {
        Dictionary f;
        f["frame"] = int64_t(frame);
        f["gpu"] = gpu;
        f["gpu_total_ms"] = gpu_total;
        f["cpu"] = cpu;
        f["cpu_total_ms"] = cpu_total;
        frames.push_back(f);
    }
    mutex->unlock();
}

Dictionary GPUProfiler::get_breakdown() const {
    mutex->lock();
    Dictionary d = breakdown.duplicate();
    mutex->unlock();
    return d;
}
