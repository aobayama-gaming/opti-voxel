#pragma once

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// Per-pass GPU time of a recent frame, from the timestamps the renderer captures (they exist while
// the viewport measures its render time, which the bench enables). Reading happens on the render
// thread; poll() is cheap and can be called every N frames from the main thread.
class GPUProfiler : public Object {
public:
    // Main thread: schedules a read on the render thread.
    void poll();
    // Main thread: latest breakdown { "<timestamp name>": ms since the previous timestamp, ... } and
    // "total_ms". Names are the renderer's ("Render Shadows", "Render Opaque Pass", ...).
    Dictionary get_breakdown() const;

    GPUProfiler();

private:
    Ref<Mutex> mutex;
    Dictionary breakdown;

    void _rt_read();
};
