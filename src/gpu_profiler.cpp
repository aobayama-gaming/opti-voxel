#include "gpu_profiler.hpp"
#include "trace.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

namespace {

bool is_mesher_name(const String &p_name) {
    return p_name.begins_with("vx") && p_name.contains("_");
}

String section_name(const String &p_name) {
    return is_mesher_name(p_name) ? "vx_" + p_name.get_slice("_", 1) : p_name;
}

// The frame's intervals as spans on the trace's GPU track, inside a "frame" span. The GPU clock is not the CPU's:
// the frame is placed as early as it can run: every interval starting at or after the CPU time its timestamp was
// recorded at (the render thread records, the GPU runs later) and after the previous frame (r_end: the GPU runs
// frames one after the other), at their GPU durations. A mesh batch's copy carries the batch's flow
// ("vx<batch>_copy"): after the render thread's dispatch, before the readback.
void trace_frame(RenderingDevice *p_rd, const std::vector<String> &p_names, const std::vector<String> &p_labels, uint64_t &r_end) {
    Trace::name_thread("render");
    const uint32_t n = uint32_t(p_names.size());
    const uint64_t gpu0 = p_rd->get_captured_timestamp_gpu_time(0);
    uint64_t start = r_end;
    for (uint32_t i = 0; i < n; i++) {
        const uint64_t gpu = p_rd->get_captured_timestamp_gpu_time(i);
        const uint64_t offset = gpu >= gpu0 ? (gpu - gpu0) / 1000 : 0;
        const uint64_t cpu = p_rd->get_captured_timestamp_cpu_time(i);
        start = MAX(start, cpu > offset ? cpu - offset : 0);
    }
    const uint64_t gpu_end = p_rd->get_captured_timestamp_gpu_time(n - 1);
    if (gpu_end < gpu0) {
        return;
    }
    Trace::gpu_span("gpu", "frame", start, (gpu_end - gpu0) / 1000);
    r_end = start + (gpu_end - gpu0) / 1000;
    for (uint32_t i = 0; i < p_labels.size(); i++) {
        const uint64_t begin = p_rd->get_captured_timestamp_gpu_time(i);
        const uint64_t end = p_rd->get_captured_timestamp_gpu_time(i + 1);
        if (end < begin || begin < gpu0) {
            continue;
        }
        const uint64_t ts = start + (begin - gpu0) / 1000;
        const uint64_t dur = (end - begin) / 1000;
        const bool mesher = p_labels[i].begins_with("vx_");
        Trace::gpu_span(mesher ? "mesher" : "render", Trace::intern(p_labels[i].utf8().get_data()), ts, dur);
        if (mesher && p_names[i + 1].ends_with("_copy")) {
            const uint64_t batch = uint64_t(p_names[i + 1].substr(2).get_slice("_", 0).to_int());
            Trace::gpu_flow(Trace::Kind::FLOW_STEP, "batch", batch, ts + dur / 2);
        }
    }
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
    if (Trace::enabled()) {
        trace_frame(rd, names, labels, trace_gpu_end);
    }
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
