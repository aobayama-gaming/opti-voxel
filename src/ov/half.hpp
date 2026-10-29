#pragma once

#include "ov/bytes.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ov {

// An IEEE half float's value (subnormals, infinities and NaNs too): the RGBA16F readbacks of the checks.
inline float half_to_float(uint16_t p_h) {
    const uint32_t sign = uint32_t(p_h >> 15) << 31;
    const uint32_t exp = (p_h >> 10) & 31u;
    const uint32_t man = p_h & 1023u;
    uint32_t bits;
    if (exp == 0) {
        const float v = std::ldexp(float(man), -24);
        std::memcpy(&bits, &v, 4);
        bits |= sign;
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// The IEEE half nearest p_v (ties up), under the half's normals 0, over its range infinity (no NaNs): uploads of
// values in range (the clouds' climate map).
inline uint16_t float_to_half(float p_v) {
    uint32_t b;
    std::memcpy(&b, &p_v, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const int exp = int((b >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = b & 0x7FFFFFu;
    if (exp <= 0) {
        return uint16_t(sign);
    }
    if (exp >= 31) {
        return uint16_t(sign | 0x7C00u);
    }
    // A carry out of the mantissa raises the exponent, as it should (to infinity past the largest half).
    return uint16_t((sign | (uint32_t(exp) << 10) | (man >> 13)) + ((man >> 12) & 1u));
}

// Every half float of a readback, in order.
inline std::vector<float> half_floats(const bytes &p_data) {
    std::vector<float> out(p_data.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        uint16_t h;
        std::memcpy(&h, p_data.data() + i * 2, 2);
        out[i] = half_to_float(h);
    }
    return out;
}

} // namespace ov
