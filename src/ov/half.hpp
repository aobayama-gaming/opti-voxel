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
