#pragma once

// Compiling the engine's GLSL on a Device: the shader dump (OPTI_VOXEL_SHADER_DUMP=<dir> writes every composed shader to
// <dir>/<name>_<hash>.glsl, to compare two builds byte for byte) and the error report with the numbered source.

#include "gpu_device.hpp"

#include "ov/log.hpp"
#include "ov/on_error.hpp"
#include "ov/str.hpp"

#include <cstdlib>
#include <fstream>

namespace ov::gpu {

// godot's String::hash of an ASCII text (djb2), the dump's file names.
inline uint32_t text_hash(const str &p_text) {
    uint32_t h = 5381;
    for (const char c : p_text) {
        h = (h << 5) + h + uint32_t(uint8_t(c));
    }
    return h;
}

inline void dump_shader(const str &p_name, const str &p_glsl) {
    const char *dir = std::getenv("OPTI_VOXEL_SHADER_DUMP");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }
    std::ofstream f(str(dir) + "/" + p_name + "_" + format("%x", text_hash(p_glsl)) + ".glsl", std::ios::binary);
    f << p_glsl;
}

// p_glsl's pipeline on p_device (dumped first); invalid, its error and numbered source reported, if it doesn't compile.
inline Pipeline compile(Device &p_device, const str &p_name, const str &p_glsl) {
    dump_shader(p_name, p_glsl);
    str error;
    const Pipeline pipeline = p_device.create_compute_pipeline(p_name, p_glsl, error);
    if (!pipeline.is_valid()) {
        str listing;
        int line = 1;
        size_t from = 0;
        for (size_t at = p_glsl.find('\n'); from <= p_glsl.size(); at = p_glsl.find('\n', from)) {
            listing += format("%d | %s\n", line++, p_glsl.substr(from, at == str::npos ? str::npos : at - from));
            if (at == str::npos) {
                break;
            }
            from = at + 1;
        }
        OV_ERR_PRINT("--- SHADER COMPILATION FAILED (" + p_name + ") ---\nERROR MESSAGE:\n" + error + "\n" + listing);
    }
    return pipeline;
}

} // namespace ov::gpu
