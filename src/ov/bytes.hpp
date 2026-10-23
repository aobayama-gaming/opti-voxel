#pragma once

// The core's byte buffers and their 32-bit field writers / readers, at a slot pointer (host endianness,
// like godot's marshalls: little-endian on every platform the project runs on). The engine copies a
// whole ov::bytes into its upload type once (godot_bytes, godot_conv.hpp), or the encoders write
// straight into its buffer.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ov {

using bytes = std::vector<uint8_t>;

inline void put_u32(uint8_t *r_at, uint32_t p_v) {
    std::memcpy(r_at, &p_v, 4);
}

inline void put_s32(uint8_t *r_at, int32_t p_v) {
    std::memcpy(r_at, &p_v, 4);
}

inline void put_f32(uint8_t *r_at, float p_v) {
    std::memcpy(r_at, &p_v, 4);
}

inline uint32_t get_u32(const uint8_t *p_at) {
    uint32_t v;
    std::memcpy(&v, p_at, 4);
    return v;
}

inline int32_t get_s32(const uint8_t *p_at) {
    int32_t v;
    std::memcpy(&v, p_at, 4);
    return v;
}

inline float get_f32(const uint8_t *p_at) {
    float v;
    std::memcpy(&v, p_at, 4);
    return v;
}

} // namespace ov
