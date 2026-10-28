#include "glsl_compiler.hpp"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <regex>

namespace ov::dawn {

namespace {

// SPIR-V opcodes, decorations and storage classes read by reflect().
constexpr uint32_t OP_TYPE_IMAGE = 25;
constexpr uint32_t OP_TYPE_SAMPLER = 26;
constexpr uint32_t IMAGE_FORMAT_RGBA16F = 2; // SPIR-V's ImageFormat Rgba16f
constexpr uint32_t DIM_3D = 2; // SPIR-V's Dim 3D
constexpr uint32_t OP_TYPE_POINTER = 32;
constexpr uint32_t OP_VARIABLE = 59;
constexpr uint32_t OP_DECORATE = 71;
constexpr uint32_t OP_MEMBER_DECORATE = 72;
constexpr uint32_t OP_TYPE_STRUCT = 30;
constexpr uint32_t DECORATION_NON_WRITABLE = 24;
constexpr uint32_t DECORATION_NON_READABLE = 25;
constexpr uint32_t DECORATION_BINDING = 33;
constexpr uint32_t DECORATION_DESCRIPTOR_SET = 34;
constexpr uint32_t STORAGE_UNIFORM_CONSTANT = 0;
constexpr uint32_t STORAGE_UNIFORM = 2;
constexpr uint32_t STORAGE_STORAGE_BUFFER = 12;

// An image or sampler variable (behind a UniformConstant pointer): an image's Sampled operand (1: a sampled texture,
// 2: a storage image), Dim and format; a sampler (OpTypeSampler) has sampled = 0.
struct ImageInfo {
    uint32_t sampled = 0;
    uint32_t dim = 0;
    uint32_t format = 0;
};

std::map<uint32_t, ImageInfo> image_variables(const std::vector<uint32_t> &p_spirv) {
    std::map<uint32_t, ImageInfo> types, pointers, out;
    for (size_t i = 5; i < p_spirv.size();) {
        const uint32_t count = p_spirv[i] >> 16;
        const uint32_t op = p_spirv[i] & 0xFFFFu;
        if (count == 0 || i + count > p_spirv.size()) {
            break;
        }
        const uint32_t *w = &p_spirv[i];
        if (op == OP_TYPE_IMAGE && count >= 9) {
            types[w[1]] = ImageInfo{ w[7], w[3], w[8] };
        } else if (op == OP_TYPE_SAMPLER && count >= 2) {
            types[w[1]] = ImageInfo{};
        } else if (op == OP_TYPE_POINTER && count >= 4 && w[2] == STORAGE_UNIFORM_CONSTANT && types.count(w[3])) {
            pointers[w[1]] = types[w[3]];
        } else if (op == OP_VARIABLE && count >= 4 && pointers.count(w[1])) {
            out[w[2]] = pointers[w[1]];
        }
        i += count;
    }
    return out;
}

// The storage images (Sampled = 2) of image_variables.
std::map<uint32_t, ImageInfo> storage_images(const std::vector<uint32_t> &p_spirv) {
    std::map<uint32_t, ImageInfo> out;
    for (const auto &[var, info] : image_variables(p_spirv)) {
        if (info.sampled == 2) {
            out[var] = info;
        }
    }
    return out;
}

// The set-0 buffers (variable -> pointer -> struct, the struct's members all NonWritable for a readonly block) and storage
// images.
std::vector<ShaderBinding> reflect(const std::vector<uint32_t> &p_spirv) {
    const std::map<uint32_t, ImageInfo> images = image_variables(p_spirv);
    std::map<uint32_t, uint32_t> binding, set, var_type, var_class, pointee, struct_members, non_writable_members;
    std::map<uint32_t, bool> var_non_writable, var_non_readable;
    for (size_t i = 5; i < p_spirv.size();) {
        const uint32_t count = p_spirv[i] >> 16;
        const uint32_t op = p_spirv[i] & 0xFFFFu;
        if (count == 0 || i + count > p_spirv.size()) {
            break;
        }
        const uint32_t *w = &p_spirv[i];
        if (op == OP_DECORATE && count >= 3) {
            if (w[2] == DECORATION_BINDING && count >= 4) {
                binding[w[1]] = w[3];
            } else if (w[2] == DECORATION_DESCRIPTOR_SET && count >= 4) {
                set[w[1]] = w[3];
            } else if (w[2] == DECORATION_NON_WRITABLE) {
                var_non_writable[w[1]] = true;
            } else if (w[2] == DECORATION_NON_READABLE) {
                var_non_readable[w[1]] = true;
            }
        } else if (op == OP_MEMBER_DECORATE && count >= 4 && w[3] == DECORATION_NON_WRITABLE) {
            non_writable_members[w[1]]++;
        } else if (op == OP_TYPE_STRUCT) {
            struct_members[w[1]] = count - 2;
        } else if (op == OP_TYPE_POINTER && count >= 4) {
            pointee[w[1]] = w[3];
        } else if (op == OP_VARIABLE && count >= 4) {
            var_type[w[2]] = w[1];
            var_class[w[2]] = w[3];
        }
        i += count;
    }
    std::vector<ShaderBinding> out;
    for (const auto &[var, b] : binding) {
        if ((set.count(var) ? set[var] : 0) != 0) {
            continue;
        }
        if (images.count(var)) {
            const ImageInfo &info = images.at(var);
            ShaderBinding sb;
            sb.binding = b;
            sb.image = info.sampled == 2;
            sb.texture = info.sampled == 1;
            sb.sampler = info.sampled == 0;
            sb.volume = info.dim == DIM_3D;
            sb.half_float = info.format == IMAGE_FORMAT_RGBA16F;
            sb.write_only = var_non_readable[var];
            sb.read_only = var_non_writable[var];
            out.push_back(sb);
            continue;
        }
        const uint32_t storage = var_class.count(var) ? var_class[var] : 0;
        if (storage != STORAGE_UNIFORM && storage != STORAGE_STORAGE_BUFFER) {
            continue;
        }
        const uint32_t block = pointee[var_type[var]];
        ShaderBinding sb;
        sb.binding = b;
        sb.uniform = storage == STORAGE_UNIFORM;
        sb.read_only = !sb.uniform && (var_non_writable[var] || (struct_members[block] > 0 && non_writable_members[block] == struct_members[block]));
        out.push_back(sb);
    }
    std::sort(out.begin(), out.end(), [](const ShaderBinding &a, const ShaderBinding &b) { return a.binding < b.binding; });
    return out;
}

// WebGPU has no write-only storage buffers: a `writeonly` block (NonReadable) becomes read-write. The bits a shader
// computes don't change, only what the validator allows. A storage image keeps it (a write-only storage texture).
void drop_non_readable(std::vector<uint32_t> &r_spirv) {
    const std::map<uint32_t, ImageInfo> images = storage_images(r_spirv);
    std::vector<uint32_t> out(r_spirv.begin(), r_spirv.begin() + std::min<size_t>(5, r_spirv.size()));
    for (size_t i = 5; i < r_spirv.size();) {
        const uint32_t count = r_spirv[i] >> 16;
        const uint32_t op = r_spirv[i] & 0xFFFFu;
        if (count == 0 || i + count > r_spirv.size()) {
            out.insert(out.end(), r_spirv.begin() + std::ptrdiff_t(i), r_spirv.end());
            break;
        }
        const bool drop = (op == OP_DECORATE && count >= 3 && r_spirv[i + 2] == DECORATION_NON_READABLE && !images.count(r_spirv[i + 1])) ||
                (op == OP_MEMBER_DECORATE && count >= 4 && r_spirv[i + 3] == DECORATION_NON_READABLE);
        if (!drop) {
            out.insert(out.end(), r_spirv.begin() + std::ptrdiff_t(i), r_spirv.begin() + std::ptrdiff_t(i + count));
        }
        i += count;
    }
    r_spirv = std::move(out);
}

} // namespace

CompiledShader compile_glsl(const std::string &p_source, ShaderStage p_stage, bool p_push_as_uniform) {
    static std::once_flag s_init;
    std::call_once(s_init, [] { glslang::InitializeProcess(); });

    const EShLanguage stage = p_stage == ShaderStage::VERTEX ? EShLangVertex : (p_stage == ShaderStage::FRAGMENT ? EShLangFragment : EShLangCompute);
    CompiledShader out;
    glslang::TShader shader(stage);
    // std140 gives scalar members the std430 offsets the push data is packed with, but moves arrays (16-byte strides):
    // a push block with one is refused rather than read wrong.
    static const std::regex s_push(R"(layout\s*\(\s*push_constant\s*(,\s*std430\s*)?\))");
    static const std::regex s_push_array(R"(layout\s*\(\s*push_constant[^{]*\{[^}]*\[)");
    if (p_push_as_uniform && std::regex_search(p_source, s_push_array)) {
        out.error = "compile_glsl: a push constant block with an array (its std140 uniform would move the members)";
        return out;
    }
    const std::string text = p_push_as_uniform ? std::regex_replace(p_source, s_push, "layout(set = 1, binding = 0, std140)") : p_source;
    const char *source = text.c_str();
    shader.setStrings(&source, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
    if (!shader.parse(GetDefaultResources(), 450, false, EShMsgDefault)) {
        out.error = std::string(shader.getInfoLog()) + shader.getInfoDebugLog();
        return out;
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(EShMsgDefault)) {
        out.error = std::string(program.getInfoLog()) + program.getInfoDebugLog();
        return out;
    }
    glslang::GlslangToSpv(*program.getIntermediate(stage), out.spirv);
    if (out.spirv.empty()) {
        out.error = "glslang: empty SPIR-V";
        return out;
    }
    drop_non_readable(out.spirv);
    out.bindings = reflect(out.spirv);
    return out;
}

} // namespace ov::dawn
