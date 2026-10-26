#pragma once

// Text helpers for the engine-free core, with godot String's semantics where the core used its
// methods (the GLSL splices): replace_all and count scan left to right, matches don't overlap and
// replaced text is not rescanned, so a chain of replaces gives the same text as godot's.

#include "format.hpp"

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>

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

// The pairs replaced one after another, in order: a chain of godot replaces (a pair whose key is
// another's prefix goes first).
inline str replace_all(str p_text, std::initializer_list<std::pair<str, str>> p_pairs) {
    for (const std::pair<str, str> &pair : p_pairs) {
        p_text = replace_all(std::move(p_text), pair.first, pair.second);
    }
    return p_text;
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

// godot's String::is_valid_ascii_identifier: not empty, not starting with a digit, letters, digits and '_' only.
inline bool is_valid_ascii_identifier(const str &p_s) {
    if (p_s.empty() || (p_s[0] >= '0' && p_s[0] <= '9')) {
        return false;
    }
    for (const char c : p_s) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
            return false;
        }
    }
    return true;
}

// godot's String::num(p_v, p_decimals): p_decimals decimals (-1: 14, one fewer per power of ten past 10), trailing zeros
// dropped.
inline str num(double p_v, int p_decimals = -1) {
    if (std::isnan(p_v)) {
        return "nan";
    }
    if (std::isinf(p_v)) {
        return std::signbit(p_v) ? "-inf" : "inf";
    }
    int decimals = p_decimals;
    if (decimals < 0) {
        decimals = 14;
        const double abs_v = std::fabs(p_v);
        if (abs_v > 10) {
            decimals -= int(std::floor(std::log10(abs_v)));
        }
    }
    decimals = decimals > 32 ? 32 : (decimals < 0 ? 0 : decimals);
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, p_v);
    str out(buf);
    if (out.find('.') != str::npos) {
        while (!out.empty() && out.back() == '0') {
            out.pop_back();
        }
        if (!out.empty() && out.back() == '.') {
            out.pop_back();
        }
    }
    return out;
}

// Text pieces joined (godot's print(a, b, ...)): each through format's "%s".
template <typename... T>
inline str cat(const T &...p_parts) {
    str out;
    ((out += format("%s", p_parts)), ...);
    return out;
}

} // namespace ov
