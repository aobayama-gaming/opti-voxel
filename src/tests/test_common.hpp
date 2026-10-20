#pragma once

// Shared by the unit tests: doctest plus readable failure messages for our and Godot's types.

#include <doctest.h>

#include "exact_noise.hpp"
#include "world_coord.hpp"

#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <array>
#include <cmath>
#include <vector>

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

using ExactNoise::same_bits;

// A small set of row-major world -> local rotations (TableRecord::rotation): identity, a yaw, and a full rotation (0.5
// rad about (1, 2, 3)).
inline void rotations(std::vector<std::array<float, 9>> &r_out) {
    r_out.push_back({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });
    const float c = float(std::cos(0.7)), s = float(std::sin(0.7));
    r_out.push_back({ c, 0, -s, 0, 1, 0, s, 0, c });
    const double axis[3] = { 1, 2, 3 };
    const double n = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    const double x = axis[0] / n, y = axis[1] / n, z = axis[2] / n;
    const double a = 0.5, ca = std::cos(a), sa = std::sin(a), t = 1.0 - ca;
    const float m[9] = { float(t * x * x + ca), float(t * x * y - sa * z), float(t * x * z + sa * y), float(t * x * y + sa * z),
        float(t * y * y + ca), float(t * y * z - sa * x), float(t * x * z - sa * y), float(t * y * z + sa * x),
        float(t * z * z + ca) };
    r_out.push_back({ m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8] });
}
