#include "shader_module.hpp"

#include "ov/log.hpp"
#include "ov/on_error.hpp"

#ifdef __EMSCRIPTEN__
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/writer/writer.h"

#include <string>
#include <vector>
#endif

namespace ov::dawn {

#ifdef __EMSCRIPTEN__

wgpu::ShaderModule create_spirv_module(const wgpu::Device &p_device, const uint32_t *p_words, size_t p_word_count, const char *p_label) {
    const std::vector<uint32_t> words(p_words, p_words + p_word_count);
    auto ir = tint::spirv::reader::ReadIR(words);
    OV_ERR_FAIL_COND_V_MSG(ir != tint::Success, wgpu::ShaderModule(), ov::format("Tint: %s: SPIR-V not read: %s", p_label, ir.Failure().reason));
    auto wgsl = tint::wgsl::writer::WgslFromIR(ir.Get(), tint::wgsl::writer::Options{});
    OV_ERR_FAIL_COND_V_MSG(wgsl != tint::Success, wgpu::ShaderModule(), ov::format("Tint: %s: no WGSL: %s", p_label, wgsl.Failure().reason));
    wgpu::ShaderSourceWGSL source{};
    source.code = wgsl->wgsl.c_str();
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &source;
    desc.label = p_label;
    return p_device.CreateShaderModule(&desc);
}

#else

wgpu::ShaderModule create_spirv_module(const wgpu::Device &p_device, const uint32_t *p_words, size_t p_word_count, const char *p_label) {
    wgpu::ShaderSourceSPIRV source{};
    source.codeSize = uint32_t(p_word_count);
    source.code = p_words;
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &source;
    desc.label = p_label;
    return p_device.CreateShaderModule(&desc);
}

#endif

} // namespace ov::dawn
