#include "glsl.hpp"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <mutex>

namespace ov::dawn {

namespace {

std::once_flag g_glslang_once;

void initialize_once() {
    std::call_once(g_glslang_once, []() { glslang::InitializeProcess(); });
}

} // namespace

std::string CompileComputeGLSL(const std::string &p_source, std::vector<uint32_t> &r_spirv) {
    initialize_once();
    r_spirv.clear();

    glslang::TShader shader(EShLangCompute);
    const char *source = p_source.c_str();
    shader.setStrings(&source, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);

    const TBuiltInResource *resources = GetDefaultResources();
    if (!shader.parse(resources, 450, false, EShMsgDefault)) {
        return std::string(shader.getInfoLog()) + "\n" + shader.getInfoDebugLog();
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(EShMsgDefault)) {
        return std::string(program.getInfoLog()) + "\n" + program.getInfoDebugLog();
    }
    const glslang::TIntermediate *intermediate = program.getIntermediate(EShLangCompute);
    if (intermediate == nullptr) {
        return "glslang: no compute intermediate";
    }
    glslang::GlslangToSpv(*intermediate, r_spirv);
    if (r_spirv.empty()) {
        return "glslang: empty SPIR-V";
    }
    return {};
}

} // namespace ov::dawn
