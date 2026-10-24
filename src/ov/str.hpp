#pragma once

// Text helpers for the engine-free core, with godot String's semantics where the core used its
// methods (the GLSL splices): replace_all and count scan left to right, matches don't overlap and
// replaced text is not rescanned, so a chain of replaces gives the same text as godot's.

#include "format.hpp"

#include <string>

namespace ov {

// Every occurrence of p_from replaced by p_to (godot's String::replace). An empty p_from changes nothing.
inline str replace_all(str p_text, const str &p_from, const str &p_to) {
    if (p_from.empty()) {
        return p_text;
    }
    size_t at = p_text.find(p_from);
    if (at == str::npos) {
        return p_text;
    }
    str out;
    out.reserve(p_text.size());
    size_t from = 0;
    while (at != str::npos) {
        out.append(p_text, from, at - from);
        out += p_to;
        from = at + p_from.size();
        at = p_text.find(p_from, from);
    }
    out.append(p_text, from, str::npos);
    return out;
}

// Non-overlapping occurrences of p_what (godot's String::count). 0 for an empty p_what.
inline int count(const str &p_text, const str &p_what) {
    if (p_what.empty()) {
        return 0;
    }
    int n = 0;
    for (size_t at = p_text.find(p_what); at != str::npos; at = p_text.find(p_what, at + p_what.size())) {
        n++;
    }
    return n;
}

inline bool contains(const str &p_text, const str &p_what) {
    return p_text.find(p_what) != str::npos;
}

inline str itos(int64_t p_v) {
    return std::to_string(p_v);
}

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
