#pragma once

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <vector>

using namespace godot;

// The section each interval between two captured timestamps belongs to: interval i runs from timestamp i to
// i + 1. The renderer's names mark where a section starts ("Render Shadows": what follows is the shadow pass),
// the direct mesher's mark where one ends ("vx<batch>_p1": the edges pass just finished; "_begin" starts the
// batch). Mesher names lose their batch number ("vx_p1") so that sections add up across batches; what runs
// between a batch's end and the next renderer section is "(untracked)".
std::vector<String> gpu_interval_labels(const std::vector<String> &p_names);

// Per-pass GPU and render-thread CPU time of recent frames, from the timestamps the main RenderingDevice
// captures. Without Godot's visual profiler only the viewport's begin / end exist; with it (a debugger
// attached, `bench.sh --profile=1`) every pass has one. Reading happens on the render thread; poll() is cheap.
class GPUProfiler : public Object {
public:
    // Main thread: schedules a read on the render thread.
    void poll();
    // Main thread: latest breakdown { "<section>": ms, ... } and "total_ms".
    Dictionary get_breakdown() const;
    // Recording keeps one entry per captured frame (poll every frame): { "frame", "gpu": { section: ms },
    // "gpu_total_ms", "cpu": { section: ms }, "cpu_total_ms" }, the CPU being the render thread's.
    void set_recording(bool p_recording);
    // Main thread: the frames recorded since the last call.
    Array take_frames();

    GPUProfiler();

private:
    Ref<Mutex> mutex;
    Dictionary breakdown;
    bool recording = false;
    Array frames;
    uint64_t last_frame = UINT64_MAX; // render thread only
    uint64_t trace_gpu_end = 0; // render thread only: where the last traced GPU frame ends, usec

    void _rt_read();
};
