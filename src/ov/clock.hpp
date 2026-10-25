#pragma once

// The core's microsecond clock. Default: std::chrono's steady clock. The engine layer installs its own
// (register_types: Godot's ticks), so core timestamps share an epoch with the engine's and the GPU
// timestamps the trace lines up against them.

#include <chrono>
#include <cstdint>

namespace ov {

using ClockSource = uint64_t (*)();

inline uint64_t steady_usec() {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline ClockSource &clock_source() {
    static ClockSource s_source = steady_usec;
    return s_source;
}

inline void set_clock_source(ClockSource p_source) {
    clock_source() = p_source;
}

inline uint64_t now_usec() {
    return clock_source()();
}

} // namespace ov
