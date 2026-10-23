#pragma once

// Engine-side conversions of the core's engine-free types: explicit, at call sites, never back into
// the core.

#include "latency_hist.hpp"
#include "ov/bytes.hpp"
#include "world_coord.hpp"

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2i.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <cstring>

inline godot::String godot_str(const ov::str &p_s) {
    return godot::String::utf8(p_s.c_str(), int64_t(p_s.size()));
}

// An int64 id or voxel as report text (WorldCoord::to_string).
inline godot::String godot_str(const Vector3i64 &p_v) {
    return godot_str(WorldCoord::to_string(p_v));
}

inline godot::Vector2i godot_v2i(const ov::ivec2 &p_v) {
    return godot::Vector2i(p_v.x, p_v.y);
}

inline godot::Vector3i godot_v3i(const ov::ivec3 &p_v) {
    return godot::Vector3i(p_v.x, p_v.y, p_v.z);
}

inline godot::Vector3 godot_v3(const Float3 &p_v) {
    return godot::Vector3(p_v.x, p_v.y, p_v.z);
}

inline Float3 ov_f3(const godot::Vector3 &p_v) {
    return Float3{ float(p_v.x), float(p_v.y), float(p_v.z) };
}

inline godot::PackedByteArray godot_bytes(const ov::bytes &p_bytes) {
    godot::PackedByteArray packed;
    packed.resize(int64_t(p_bytes.size()));
    std::memcpy(packed.ptrw(), p_bytes.data(), p_bytes.size());
    return packed;
}

// { count, p50, p95, max }, the reports' form of a LatencyHist.
inline godot::Dictionary godot_dict(const LatencyHist &p_h) {
    godot::Dictionary d;
    d["count"] = int64_t(p_h.count());
    d["p50"] = int64_t(p_h.percentile(0.5));
    d["p95"] = int64_t(p_h.percentile(0.95));
    d["max"] = int64_t(p_h.max());
    return d;
}
