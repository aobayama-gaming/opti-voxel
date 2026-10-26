#pragma once

// Error reporting for the engine-free core: the godot ERR_* macros report through the engine's
// console; the core reports through a settable sink instead (default: stderr). The engine layer
// (register_types) points the sink at its own reporting, so messages land in the Godot console as
// before. tools/check_core_purity.sh keeps godot includes out of src/ov/ and the core.

#include <cstdio>
#include <string>

namespace ov {

using ErrorSink = void (*)(const char *p_file, int p_line, const char *p_message);

inline void stderr_sink(const char *p_file, int p_line, const char *p_message) {
    std::fprintf(stderr, "%s:%d: %s\n", p_file, p_line, p_message);
}

inline ErrorSink &error_sink() {
    static ErrorSink s_sink = stderr_sink;
    return s_sink;
}

inline void set_error_sink(ErrorSink p_sink) {
    error_sink() = p_sink;
}

inline void report_error(const char *p_file, int p_line, const std::string &p_message) {
    if (error_sink() != nullptr) {
        error_sink()(p_file, p_line, p_message.c_str());
    }
}

} // namespace ov

// The core's ERR_* replacements: same condition / return semantics as godot's macros, message
// through the sink. __FILE__ stays relative to src/ (compiled from the project root).
#define OV_ERR_FAIL_COND_V_MSG(p_cond, p_ret, p_msg)     \
    do {                                                 \
        if (p_cond) {                                    \
            ov::report_error(__FILE__, __LINE__, p_msg); \
            return p_ret;                                \
        }                                                \
    } while (0)

#define OV_ERR_FAIL_COND_MSG(p_cond, p_msg)              \
    do {                                                 \
        if (p_cond) {                                    \
            ov::report_error(__FILE__, __LINE__, p_msg); \
            return;                                      \
        }                                                \
    } while (0)

#define OV_ERR_FAIL_MSG(p_msg)                       \
    do {                                             \
        ov::report_error(__FILE__, __LINE__, p_msg); \
        return;                                      \
    } while (0)

#define OV_ERR_FAIL_V_MSG(p_ret, p_msg)              \
    do {                                             \
        ov::report_error(__FILE__, __LINE__, p_msg); \
        return p_ret;                                \
    } while (0)

// Inside a loop: report and skip to the next iteration.
#define OV_ERR_CONTINUE_MSG(p_cond, p_msg)           \
    if (p_cond) {                                    \
        ov::report_error(__FILE__, __LINE__, p_msg); \
        continue;                                    \
    } else                                           \
        ((void)0)

#define OV_ERR_PRINT(p_msg) ov::report_error(__FILE__, __LINE__, p_msg)

// The forms without a message: godot's text for them ("Condition "<cond>" is true.").
#define OV_ERR_FAIL_COND(p_cond) OV_ERR_FAIL_COND_MSG(p_cond, "Condition \"" #p_cond "\" is true.")
#define OV_ERR_FAIL_COND_V(p_cond, p_ret) OV_ERR_FAIL_COND_V_MSG(p_cond, p_ret, "Condition \"" #p_cond "\" is true. Returning: " #p_ret)
#define OV_ERR_CONTINUE(p_cond) OV_ERR_CONTINUE_MSG(p_cond, "Condition \"" #p_cond "\" is true. Continuing.")
#define OV_ERR_FAIL_INDEX(p_index, p_size) \
    OV_ERR_FAIL_COND_MSG((p_index) < 0 || (p_index) >= (p_size), "Index " #p_index " is out of bounds (" #p_size ").")
#define OV_ERR_FAIL_NULL_MSG(p_ptr, p_msg) OV_ERR_FAIL_COND_MSG((p_ptr) == nullptr, p_msg)

// Reported the first time this line runs only.
#define OV_ERR_PRINT_ONCE(p_msg)                         \
    do {                                                 \
        static bool s_printed = false;                   \
        if (!s_printed) {                                \
            s_printed = true;                            \
            ov::report_error(__FILE__, __LINE__, p_msg); \
        }                                                \
    } while (0)
