#pragma once

// GLSL -> SPIR-V for the Dawn adapter. The mesher and cascade shaders are generated as GLSL text by the core
// (CascadeGLSL, LegacyMesherGLSL); Dawn consumes SPIR-V, so glslang compiles the text at runtime. The core
// keeps one shader source for both adapters: the Godot RenderingDevice compiles the same GLSL itself.

#include <cstdint>
#include <string>
#include <vector>

namespace ov::dawn {

// Compiles one GLSL compute shader (#version 450 in the source). Empty on success, the compiler log on failure.
std::string CompileComputeGLSL(const std::string &p_source, std::vector<uint32_t> &r_spirv);

} // namespace ov::dawn
