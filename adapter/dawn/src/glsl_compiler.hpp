#pragma once

// The engine's GLSL (Vulkan dialect) to SPIR-V through glslang, and what a WebGPU pipeline layout needs of it.

#include <cstdint>
#include <string>
#include <vector>

namespace ov::dawn {

// A buffer or storage image of descriptor set 0 the shader declares (used or not: glslang keeps every declaration).
struct ShaderBinding {
    uint32_t binding = 0;
    bool uniform = false; // else a storage buffer
    bool read_only = false; // every member NonWritable (a `readonly` block)
    bool image = false; // a storage image (the engine's: rgba16ui, 2D)
    bool write_only = false; // an image declared `writeonly` (NonReadable)
};

struct CompiledShader {
    std::vector<uint32_t> spirv;
    std::vector<ShaderBinding> bindings; // by binding
    std::string error; // empty on success
};

enum class ShaderStage {
    COMPUTE,
    VERTEX,
    FRAGMENT,
};

// p_push_as_uniform: the push constant block becomes a uniform block at set 1, binding 0 (a device without immediates:
// DawnDevice feeds it from a ring buffer).
CompiledShader compile_glsl(const std::string &p_source, ShaderStage p_stage = ShaderStage::COMPUTE, bool p_push_as_uniform = false);

} // namespace ov::dawn
