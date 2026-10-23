#pragma once

// Engine-side conversions of the core's engine-free types (the de-Godot sweep): explicit, at call
// sites, never back into the core (the spirit of WorldCoord::lo32_i: "no conversion back to an
// int32 triple").

#include "world_coord.hpp"

#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

inline godot::String godot_str(const ov::str &p_s) {
    return godot::String(p_s.c_str());
}

inline godot::Vector3i godot_v3i(const ov::ivec3 &p_v) {
    return godot::Vector3i(p_v.x, p_v.y, p_v.z);
}

inline godot::Vector3 godot_v3(const Float3 &p_v) {
    return godot::Vector3(p_v.x, p_v.y, p_v.z);
}
