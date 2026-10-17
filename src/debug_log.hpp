#pragma once

#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <atomic>

using namespace godot;

// Global verbosity shared by every subsystem (set from VoxelTerrain.log_level).
// 0 = errors only, 1 = info (lifecycle), 2 = verbose (per batch), 3 = trace.
namespace DebugLog {
inline std::atomic<int> level{ 0 };
// Leak accounting for chunk render resources (mesh/instance pairs).
inline std::atomic<int64_t> chunk_rids_created{ 0 };
inline std::atomic<int64_t> chunk_rids_freed{ 0 };
} //namespace DebugLog

#define VLOG(p_level, ...)                         \
    do {                                           \
        if (DebugLog::level.load() >= (p_level)) { \
            UtilityFunctions::print(__VA_ARGS__);  \
        }                                          \
    } while (0)

inline uint64_t now_usec() {
    return Time::get_singleton()->get_ticks_usec();
}

// Accumulates a timing series (usec) without storing every sample.
struct TimingStat {
    uint64_t count = 0;
    uint64_t total = 0;
    uint64_t max = 0;
    uint64_t last = 0;

    void add(uint64_t p_usec) {
        count++;
        total += p_usec;
        last = p_usec;
        if (p_usec > max) {
            max = p_usec;
        }
    }

    Dictionary to_dict() const {
        Dictionary d;
        d["count"] = (int64_t)count;
        d["total_ms"] = total / 1000.0;
        d["avg_ms"] = count ? (total / 1000.0) / count : 0.0;
        d["max_ms"] = max / 1000.0;
        d["last_ms"] = last / 1000.0;
        return d;
    }
};
