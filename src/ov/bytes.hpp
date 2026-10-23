#pragma once

// The core's byte buffers: std::vector-backed, with the offset encoders the PackedByteArray call
// sites used (host endianness, like godot's marshalls: little-endian on every platform the project
// runs on). The engine layer wraps into its GPU upload types at the boundary (rd_utils.hpp).

#include <cstring>
#include <vector>

namespace ov {

using bytes = std::vector<uint8_t>;

inline void encode_u32(bytes &r_bytes, size_t p_offset, uint32_t p_v) {
    std::memcpy(r_bytes.data() + p_offset, &p_v, 4);
}

inline void encode_s32(bytes &r_bytes, size_t p_offset, int32_t p_v) {
    std::memcpy(r_bytes.data() + p_offset, &p_v, 4);
}

inline void encode_f32(bytes &r_bytes, size_t p_offset, float p_v) {
    std::memcpy(r_bytes.data() + p_offset, &p_v, 4);
}

inline void encode_u16(bytes &r_bytes, size_t p_offset, uint16_t p_v) {
    std::memcpy(r_bytes.data() + p_offset, &p_v, 2);
}

inline void encode_s16(bytes &r_bytes, size_t p_offset, int16_t p_v) {
    std::memcpy(r_bytes.data() + p_offset, &p_v, 2);
}

} // namespace ov
