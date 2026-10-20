#include "material_library.hpp"

#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace Material {

Library::Library() {
    // Dark albedos: the scene's tonemapper brightens them (like the ground's).
    add("ground", Params());
    add("bark", Params{ Color(0.17, 0.11, 0.07), 1.0f, Color(0, 0, 0), PATTERN_NONE });
    add("leaves", Params{ Color(0.07, 0.17, 0.05), 0.9f, Color(0, 0, 0), PATTERN_NONE });
    add("lava", Params{ Color(0.03, 0.015, 0.01), 1.0f, Color(1.1, 0.24, 0.02), PATTERN_LAVA });
    add("scorched_rock", Params{ Color(0.09, 0.075, 0.065), 1.0f, Color(0, 0, 0), PATTERN_NONE });
    add("island_rock", Params{ Color(0.30, 0.27, 0.22), 1.0f, Color(0, 0, 0), PATTERN_NONE });
    DEV_ASSERT(size() == BUILTIN_COUNT);
}

uint32_t Library::add(const String &p_name, const Params &p_params) {
    const uint32_t existing = find(p_name);
    ERR_FAIL_COND_V_MSG(existing != UINT32_MAX, existing, "Material::Library: a material named '" + p_name + "' already exists.");
    ERR_FAIL_COND_V_MSG(size() >= MAX_MATERIALS, GROUND, "Material::Library: over MAX_MATERIALS.");
    names.push_back(p_name);
    params.push_back(p_params);
    return size() - 1;
}

String glsl_constants() {
    return vformat("const uint MATERIAL_TEXELS = %du;\nconst uint MATERIALS_PER_ROW = %du;\nconst uint MATERIAL_GROUND = %du;\n"
                   "const uint MATERIAL_ID_BITS = %du;\nconst float MATERIAL_PATTERN_LAVA = %d.0;\n",
            TEXELS, IMAGE_MATERIALS_PER_ROW, GROUND, ID_BITS, int(PATTERN_LAVA));
}

uint32_t Library::find(const String &p_name) const {
    for (uint32_t i = 0; i < size(); i++) {
        if (names[i] == p_name) {
            return i;
        }
    }
    return UINT32_MAX;
}

Ref<Image> Library::make_image() const {
    const int width = TEXELS * IMAGE_MATERIALS_PER_ROW;
    const int height = int((size() + IMAGE_MATERIALS_PER_ROW - 1) / IMAGE_MATERIALS_PER_ROW);
    PackedFloat32Array data;
    data.resize(int64_t(width) * height * 4);
    data.fill(0.0f);
    for (uint32_t m = 0; m < size(); m++) {
        const Params &p = params[m];
        const int64_t at = (int64_t(m / IMAGE_MATERIALS_PER_ROW) * width + TEXELS * (m % IMAGE_MATERIALS_PER_ROW)) * 4;
        const float texels[8] = { p.albedo.r, p.albedo.g, p.albedo.b, p.roughness, p.emission.r, p.emission.g, p.emission.b, float(p.pattern) };
        for (int i = 0; i < 8; i++) {
            data.set(at + i, texels[i]);
        }
    }
    return Image::create_from_data(width, height, false, Image::FORMAT_RGBAF, data.to_byte_array());
}

} // namespace Material
