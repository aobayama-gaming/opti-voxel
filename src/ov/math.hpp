#pragma once

// Numeric helpers for the engine-free core.

namespace ov {

// godot's MIN / MAX / CLAMP, operand for operand: a NaN or a signed zero picks the same side as the
// macros did, which mirrored float code (CPU twins of GPU math) depends on. std::min / std::max pick
// the other operand on ties and NaN.
template <typename T>
constexpr T min(T p_a, T p_b) {
    return p_a < p_b ? p_a : p_b;
}

template <typename T>
constexpr T max(T p_a, T p_b) {
    return p_a > p_b ? p_a : p_b;
}

template <typename T>
constexpr T clamp(T p_v, T p_lo, T p_hi) {
    return p_v < p_lo ? p_lo : (p_v > p_hi ? p_hi : p_v);
}

} // namespace ov
