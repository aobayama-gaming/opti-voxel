#pragma once

// Engine-side conversions of the core's engine-free types (the de-Godot sweep): explicit, at call
// sites, never back into the core (the spirit of WorldCoord::lo32_i: "no conversion back to an
// int32 triple").

#include "latency_hist.hpp"
#include "world_coord.hpp"

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2i.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

inline godot::String godot_str(const ov::str &p_s) {
    return godot::String(p_s.c_str());
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

// { count, p50, p95, max }, the reports' form of a LatencyHist.
inline godot::Dictionary godot_dict(const LatencyHist &p_h) {
    godot::Dictionary d;
    d["count"] = int64_t(p_h.count());
    d["p50"] = int64_t(p_h.percentile(0.5));
    d["p95"] = int64_t(p_h.percentile(0.95));
    d["max"] = int64_t(p_h.max());
    return d;
}
