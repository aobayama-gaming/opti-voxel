#pragma once
#include <godot_shim.hpp>
#include <atomic>
namespace DebugLog {
inline std::atomic<int64_t> chunk_rids_created{ 0 };
inline std::atomic<int64_t> chunk_rids_freed{ 0 };
}
