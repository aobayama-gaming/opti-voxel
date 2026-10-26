#pragma once

// The engine's GLSL (Vulkan dialect) to SPIR-V through glslang, and what a WebGPU pipeline layout needs of it.

#include <cstdint>
#include <string>
#include <vector>

namespace ov::dawn {

// A buffer of descriptor set 0 the shader declares (used or not: glslang keeps every declaration).
struct ShaderBinding {
    uint32_t binding = 0;
    bool uniform = false; // else a storage buffer
    bool read_only = false; // every member NonWritable (a `readonly` block)
};

struct CompiledShader {
    std::vector<uint32_t> spirv;
    std::vector<ShaderBinding> bindings; // by binding
    std::string error; // empty on success
};

CompiledShader compile_compute_glsl(const std::string &p_source);

} // namespace ov::dawn
