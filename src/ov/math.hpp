#pragma once

// Numeric helpers for the engine-free core.

namespace ov {

// godot's MIN / MAX / CLAMP, operand for operand and type for type (mixed operands give their common type, like the
// ternary): a NaN or a signed zero picks the same side as godot's did, which mirrored float code (CPU twins of GPU math)
// depends on. std::min / std::max pick the other operand on ties and NaN.
template <typename T, typename T2>
constexpr auto min(T p_a, T2 p_b) {
    return p_a < p_b ? p_a : p_b;
}

template <typename T, typename T2>
constexpr auto max(T p_a, T2 p_b) {
    return p_a > p_b ? p_a : p_b;
}

template <typename T, typename T2, typename T3>
constexpr auto clamp(T p_v, T2 p_lo, T3 p_hi) {
    return p_v < p_lo ? p_lo : (p_v > p_hi ? p_hi : p_v);
}

inline constexpr double PI = 3.1415926535897932384626433833;

// godot's Math::deg_to_rad / rad_to_deg (float and double), the same operations in the same order.
constexpr double deg_to_rad(double p_y) {
    return p_y * PI / 180.0;
}
constexpr float deg_to_rad(float p_y) {
    return p_y * static_cast<float>(PI) / 180.f;
}
constexpr double rad_to_deg(double p_y) {
    return p_y * 180.0 / PI;
}
constexpr float rad_to_deg(float p_y) {
    return p_y * 180.f / static_cast<float>(PI);
}

} // namespace ov
