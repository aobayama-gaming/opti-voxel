#pragma once

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/string.hpp>

#include "material_ids.hpp"

#include <cstdint>
#include <vector>

using namespace godot;

namespace Material {

// Shading patterns the page shader knows (Params::pattern).
enum Pattern : int {
    PATTERN_NONE = 0,
    PATTERN_LAVA = 1, // a warped crust: dark seams over a drifting glow
    PATTERN_BARK = 2, // plates and grooves along world up, knots (a 3D noise, the page shader's bark_grain)
    PATTERN_GROUND = 3, // the procedural ground (slope blend, contours) in the material's colours: albedo flat, steep on slopes
    PATTERN_LEAVES = 4, // a canopy: clumps of leaves (a 3D noise at two scales), lighter where the surface faces up
};

// The page shader's procedural ground (GROUND): albedo on gentle slopes and on steep ones (linear), the default of a
// biome's surface too. The shader gets them through glsl_constants().
inline const Color GROUND_FLAT = Color(0.24, 0.36, 0.16);
inline const Color GROUND_STEEP = Color(0.36, 0.32, 0.28);

struct Params {
    Color albedo = Color(0.5, 0.5, 0.5);
    float roughness = 1.0f;
    Color emission = Color(0, 0, 0); // PATTERN_GROUND has none: its texel holds steep
    Pattern pattern = PATTERN_NONE;
    Color steep = Color(0.5, 0.5, 0.5); // PATTERN_GROUND: the albedo on steep slopes
    // The debug views' colour (a biome's, --view=12); alpha 0: none.
    Color debug = Color(0, 0, 0, 0);
};

// Texels per material in make_image(): (albedo.rgb, roughness), (emission.rgb or PATTERN_GROUND's steep.rgb, pattern),
// debug (TEXEL_DEBUG).
static constexpr int TEXELS = 3;
static constexpr int TEXEL_DEBUG = 2;
// make_image()'s width in materials; a material m is at texel (TEXELS * (m % IMAGE_MATERIALS_PER_ROW) + t, m / IMAGE_MATERIALS_PER_ROW).
// MAX_MATERIALS / IMAGE_MATERIALS_PER_ROW rows stay within a texture's 16,384.
static constexpr int IMAGE_MATERIALS_PER_ROW = 4096;
static_assert(MAX_MATERIALS / IMAGE_MATERIALS_PER_ROW <= 16384, "material_tex rows");

// The layout above as GLSL constants, for the page shader.
String glsl_constants();

class Library {
public:
    Library();

    // A new material; its id, or the existing id (ERR) when the name is taken, or GROUND (ERR) past MAX_MATERIALS.
    uint32_t add(const String &p_name, const Params &p_params);
    // The material named p_name with p_params: added, or its parameters replaced (a reconfigure). Its id.
    uint32_t set(const String &p_name, const Params &p_params);
    // The id of a named material, or UINT32_MAX.
    uint32_t find(const String &p_name) const;
    uint32_t size() const { return uint32_t(params.size()); }
    const Params &operator[](uint32_t p_id) const { return params[p_id]; }
    const String &name(uint32_t p_id) const { return names[p_id]; }

    // The parameters as an RGBAF image (layout above), for the page shader's material_tex.
    Ref<Image> make_image() const;

private:
    std::vector<String> names;
    std::vector<Params> params;
};

} // namespace Material
