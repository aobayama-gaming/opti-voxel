#pragma once

// Shared by the unit tests: doctest plus readable failure messages for our and Godot's types.

#include <doctest.h>

#include "exact_noise.hpp"
#include "world_coord.hpp"

#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

namespace doctest {

template <>
struct StringMaker<Vector3i64> {
    static doctest::String convert(const Vector3i64 &p_v) { return WorldCoord::to_string(p_v).utf8().get_data(); }
};

template <>
struct StringMaker<godot::Vector3i> {
    static doctest::String convert(const godot::Vector3i &p_v) { return godot::String(p_v).utf8().get_data(); }
};

template <>
struct StringMaker<godot::Vector3> {
    static doctest::String convert(const godot::Vector3 &p_v) { return godot::String(p_v).utf8().get_data(); }
};

} // namespace doctest

// Every bit of a value and its gradient (the sign of a zero included).
inline bool same_bits(const ExactNoise::Float4 &p_a, const ExactNoise::Float4 &p_b) {
    using ExactNoise::float_bits;
    return float_bits(p_a.x) == float_bits(p_b.x) && float_bits(p_a.y) == float_bits(p_b.y) && float_bits(p_a.z) == float_bits(p_b.z) &&
            float_bits(p_a.w) == float_bits(p_b.w);
}
