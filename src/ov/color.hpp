#pragma once

// An RGBA colour of floats for the engine-free core (materials, biome surfaces), with godot's Color arithmetic where
// the core used it (hex, srgb_to_linear: the same float expressions, so the material texture keeps its bits).

#include <cmath>
#include <cstdint>

namespace ov {

struct Color {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;

    constexpr Color() = default;
    constexpr Color(float p_r, float p_g, float p_b, float p_a = 1.0f) :
            r(p_r), g(p_g), b(p_b), a(p_a) {}

    // 0xRRGGBBAA.
    static Color hex(uint32_t p_hex) {
        const float ca = float(p_hex & 0xFF) / 255.0f;
        p_hex >>= 8;
        const float cb = float(p_hex & 0xFF) / 255.0f;
        p_hex >>= 8;
        const float cg = float(p_hex & 0xFF) / 255.0f;
        p_hex >>= 8;
        const float cr = float(p_hex & 0xFF) / 255.0f;
        return Color(cr, cg, cb, ca);
    }

    Color srgb_to_linear() const {
        return Color(_to_linear(r), _to_linear(g), _to_linear(b), a);
    }

    bool operator==(const Color &p_o) const { return r == p_o.r && g == p_o.g && b == p_o.b && a == p_o.a; }
    bool operator!=(const Color &p_o) const { return !(*this == p_o); }

private:
    static float _to_linear(float p_c) {
        return p_c < 0.04045f ? p_c * (1.0f / 12.92f) : std::pow((p_c + 0.055f) * float(1.0 / (1.0 + 0.055)), 2.4f);
    }
};

} // namespace ov
