#pragma once

// A shader module from SPIR-V: as is on native Dawn; a browser takes WGSL only, so the web build translates it with Tint
// (linked into the wasm) first.

#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <cstdint>

namespace ov::dawn {

// Null (an error logged) when the web translation fails; a module Dawn rejects comes back like any other (its error
// goes to the device's error callback or scope).
wgpu::ShaderModule create_spirv_module(const wgpu::Device &p_device, const uint32_t *p_words, size_t p_word_count, const char *p_label);

} // namespace ov::dawn
