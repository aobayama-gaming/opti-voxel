#pragma once

// Logging for the engine-free core: a level and a settable sink (default: stdout / stderr). The engine
// layer (register_types) points the sink at its console, so lines land where VLOG's did. Errors keep
// their own sink (on_error.hpp).

#include "format.hpp"

#include <atomic>
#include <cstdio>

namespace ov {

// 0 = errors only, 1 = info (lifecycle), 2 = verbose (per batch), 3 = trace (VoxelTerrain.log_level).
enum LogLevel : int {
    LOG_WARNING = -1, // always printed, through the warning channel
    LOG_INFO = 1,
    LOG_VERBOSE = 2,
    LOG_TRACE = 3,
};

using LogSink = void (*)(int p_level, const char *p_file, int p_line, const char *p_message);

inline void stdout_log_sink(int p_level, const char *p_file, int p_line, const char *p_message) {
    if (p_level < 0) {
        std::fprintf(stderr, "WARNING: %s:%d: %s\n", p_file, p_line, p_message);
    } else {
        std::printf("%s\n", p_message);
    }
}

inline LogSink &log_sink() {
    static LogSink s_sink = stdout_log_sink;
    return s_sink;
}

inline void set_log_sink(LogSink p_sink) {
    log_sink() = p_sink;
}

inline std::atomic<int> &log_level() {
    static std::atomic<int> s_level{ 0 };
    return s_level;
}

inline void log_line(int p_level, const char *p_file, int p_line, const str &p_message) {
    if (log_sink() != nullptr) {
        log_sink()(p_level, p_file, p_line, p_message.c_str());
    }
}

} // namespace ov

// A line at p_level when the log level reaches it; the arguments are formatted only then.
#define OV_LOG(p_level, ...)                                                      \
    do {                                                                          \
        if (ov::log_level().load() >= (p_level)) {                                \
            ov::log_line((p_level), __FILE__, __LINE__, ov::format(__VA_ARGS__)); \
        }                                                                         \
    } while (0)

#define OV_WARN_PRINT(p_msg) ov::log_line(ov::LOG_WARNING, __FILE__, __LINE__, (p_msg))
