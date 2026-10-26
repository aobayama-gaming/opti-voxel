#pragma once

// printf-style text formatting for the engine-free core. Godot's vformat is the printf-family
// String % Array operator, so these call sites map 1:1; the shim keeps core formatting identical
// without godot_cpp (tools/check_core_purity.sh enforces that).
//
// Supported: %d %i %u %x %X %o %f %F %e %E %g %G %s %c %% with printf's flags (# - + 0, space),
// width and precision, passed through to snprintf verbatim (so "%08Xu" is spec %08X then a literal
// u). Arguments are promoted like printf's own: integers to long long / unsigned long long, floats
// to double, and %s renders any of them as text (godot renders any Variant under %s). A malformed
// spec or an argument type mismatch reports through the error sink, never silently.

#include <cstdint>
#include <cstdio>
#include <cstdlib> // strtod (num_text round-trip check)
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>

#include "on_error.hpp"

namespace ov {

using str = std::string;

namespace detail {

// Shortest text that round-trips the value (godot's plain float text: 1, 0.25, 0.33333333333333):
// ascending significant-digit counts, first one that round-trips wins, trailing zeros dropped.
inline str num_text(double p_v) {
    char buf[64];
    for (int digits = 1; digits <= 17; digits++) {
        const int n = std::snprintf(buf, sizeof(buf), "%.*g", digits, p_v);
        if (n <= 0) {
            continue;
        }
        if (std::strtod(buf, nullptr) == p_v) {
            return str(buf, buf + n);
        }
    }
    return "nan";
}

// The text of one numeric conversion: the spec [p_spec, p_conv) (with its '%', flags, width and
// precision) then p_tail ("ll" + the conversion char + NUL) into a small stack format buffer, one
// snprintf. Trusted literals only: the specs are string literals at the call sites.
template <typename... TPrintfArgs>
inline str spec_text(const char *p_spec, const char *p_conv, const char *p_tail, TPrintfArgs... p_args) {
    char fmt[32];
    const size_t spec_len = size_t(p_conv - p_spec);
    const size_t tail_len = std::char_traits<char>::length(p_tail);
    std::memcpy(fmt, p_spec, spec_len);
    std::memcpy(fmt + spec_len, p_tail, tail_len + 1); // the length letters, the conversion char, the NUL
    const int n = std::snprintf(nullptr, 0, fmt, p_args...);
    if (n <= 0) {
        return str();
    }
    str text(size_t(n), '\0');
    std::snprintf(text.data(), size_t(n) + 1, fmt, p_args...);
    return text;
}

} // namespace detail

// Terminal case (no more arguments): the rest of the format string is literal text; a conversion
// left in it has no argument (reported, rendered verbatim).
inline str format(const char *p_fmt) {
    str out;
    const size_t len = std::char_traits<char>::length(p_fmt);
    out.reserve(len);
    bool missing = false;
    for (const char *p = p_fmt; *p; p++) {
        out += *p;
        if (*p == '%' && p[1] == '%') {
            p++; // "%%" is one literal percent
        } else if (*p == '%') {
            missing = true;
        }
    }
    if (missing) {
        report_error(__FILE__, __LINE__, "ov::format: not enough arguments for the conversions.");
    }
    return out;
}

// Renders p_head for the conversion starting at p_fmt ('%' then spec then conversion char) and sets
// p_next to where the format string continues. A malformed spec or an argument type the conversion
// cannot render reports through the error sink, renders verbatim and consumes the argument (no
// silent drift of the later arguments).
template <typename T>
inline str format_one(const char *p_fmt, const T &p_head, const char *&p_next) {
    // String literals arrive as char[N] under the const T&: classify through decay_t, never on T.
    using D = std::decay_t<T>;
    // Every argument is a number, a bool, an enum or text: anything else (an engine type) is a compile error, not an
    // empty rendering at run time.
    static_assert(std::is_arithmetic_v<D> || std::is_enum_v<D> || std::is_same_v<D, str> || std::is_same_v<D, std::string_view> ||
                    (std::is_pointer_v<D> && std::is_same_v<std::remove_cv_t<std::remove_pointer_t<D>>, char>),
            "ov::format: an argument must be a number, a bool, an enum or text");
    const char *s = p_fmt + 1; // past '%'
    while (*s != '\0' && std::strchr("-+ #0", *s) != nullptr) {
        s++; // flags
    }
    while (*s >= '0' && *s <= '9') {
        s++; // width
    }
    if (*s == '.') {
        s++;
        while (*s >= '0' && *s <= '9') {
            s++; // precision
        }
    }
    const char conv = *s;
    const char *spec_begin = p_fmt; // includes '%'
    const char tail[4] = { 'l', 'l', conv, '\0' }; // integers: the 64-bit length letters + the conversion char
    const char float_tail[2] = { conv, '\0' }; // floats: a double, no length letters (%llf is long double)

    // A spec too long for the format buffer is malformed for our call sites: report, render verbatim
    // and consume the argument.
    if (conv == '\0' || size_t(s - spec_begin) > 24) {
        report_error(__FILE__, __LINE__, "ov::format: malformed conversion spec, rendered verbatim.");
        p_next = s + (conv == '\0' ? 0 : 1);
        return str(p_fmt, s + (conv == '\0' ? 0 : 1));
    }

    if (conv == 'd' || conv == 'i') {
        if constexpr (std::is_integral_v<D> || std::is_enum_v<D>) {
            p_next = s + 1;
            return detail::spec_text(spec_begin, s, tail, static_cast<long long>(p_head));
        }
    } else if (conv == 'u' || conv == 'x' || conv == 'X' || conv == 'o') {
        if constexpr (std::is_integral_v<D>) {
            p_next = s + 1;
            if constexpr (std::is_signed_v<D>) {
                return detail::spec_text(spec_begin, s, tail, uint64_t(static_cast<long long>(p_head)));
            } else {
                return detail::spec_text(spec_begin, s, tail, uint64_t(p_head));
            }
        }
    } else if (conv == 'f' || conv == 'F' || conv == 'e' || conv == 'E' || conv == 'g' || conv == 'G') {
        if constexpr (std::is_arithmetic_v<D>) {
            p_next = s + 1;
            return detail::spec_text(spec_begin, s, float_tail, double(p_head));
        }
    } else if (conv == 'c') {
        if constexpr (std::is_integral_v<D>) {
            p_next = s + 1;
            return str(1, char(p_head));
        }
    } else if (conv == 's') {
        if (s - spec_begin > 1) {
            // Flags, width or precision: pad / cut the plain text with printf's own %s.
            const str text = format_one("%s", p_head, p_next);
            p_next = s + 1;
            return detail::spec_text(spec_begin, s, "s", text.c_str());
        }
        if constexpr (std::is_same_v<D, str>) {
            p_next = s + 1;
            return p_head;
        } else if constexpr (std::is_pointer_v<D> && std::is_same_v<std::remove_cv_t<std::remove_pointer_t<D>>, char>) {
            p_next = s + 1;
            if constexpr (std::is_array_v<T>) {
                return str(p_head); // a literal (or char array): never null
            } else {
                return p_head != nullptr ? str(p_head) : str();
            }
        } else if constexpr (std::is_same_v<D, std::string_view>) {
            p_next = s + 1;
            return str(p_head);
        } else if constexpr (std::is_same_v<D, bool>) {
            p_next = s + 1;
            return p_head ? str("True") : str("False"); // godot's bool text
        } else if constexpr (std::is_floating_point_v<D>) {
            p_next = s + 1;
            return detail::num_text(double(p_head));
        } else if constexpr (std::is_enum_v<D>) {
            p_next = s + 1;
            return std::to_string(static_cast<long long>(p_head));
        } else if constexpr (std::is_integral_v<D>) {
            p_next = s + 1;
            return std::to_string(p_head);
        }
    } else {
        // An unknown conversion char (a length modifier such as %ld, godot's %v): reported, rendered
        // verbatim; it consumes its argument like any other conversion.
        report_error(__FILE__, __LINE__, "ov::format: unsupported conversion, rendered verbatim.");
        p_next = s + 1;
        return str(p_fmt, s + 1);
    }
    // Reached on an argument type the conversion cannot render.
    report_error(__FILE__, __LINE__, "ov::format: argument type does not match the conversion, rendered empty.");
    p_next = s + 1;
    return str();
}

// p_fmt "…literal %spec…" + p_head rendered + the rest against p_rest. Literals are scanned up to
// the first conversion (a %% pair is one literal percent and consumes no argument).
template <typename T, typename... TRest>
inline str format(const char *p_fmt, const T &p_head, const TRest &...p_rest) {
    str out;
    const char *p = p_fmt;
    while (*p != '\0') {
        if (*p == '%' && p[1] != '%') {
            break;
        }
        out += *p;
        if (*p == '%') {
            p++; // the second % of "%%"
        }
        p++;
    }
    if (*p == '\0') {
        report_error(__FILE__, __LINE__, "ov::format: more arguments than conversions, the rest ignored.");
        return out;
    }
    const char *next = nullptr;
    out += format_one(p, p_head, next);
    return out + format(next, p_rest...);
}

} // namespace ov
