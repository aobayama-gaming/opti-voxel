#pragma once

// A PNG of 8-bit pixels for the core's reports (CascadeReport, ClimateReport slices and maps): no compression (stored
// deflate blocks), so no library; any PNG reader takes it.

#include "ov/bytes.hpp"

#include <cstdint>

namespace ov {

namespace detail {

inline uint32_t png_crc(const uint8_t *p_data, size_t p_size, uint32_t p_crc = 0xFFFFFFFFu) {
    for (size_t i = 0; i < p_size; i++) {
        p_crc ^= p_data[i];
        for (int k = 0; k < 8; k++) {
            p_crc = (p_crc >> 1) ^ (0xEDB88320u & (0u - (p_crc & 1u)));
        }
    }
    return p_crc;
}

inline void png_be32(bytes &r_out, uint32_t p_v) {
    r_out.push_back(uint8_t(p_v >> 24));
    r_out.push_back(uint8_t(p_v >> 16));
    r_out.push_back(uint8_t(p_v >> 8));
    r_out.push_back(uint8_t(p_v));
}

inline void png_chunk(bytes &r_out, const char p_type[4], const bytes &p_data) {
    png_be32(r_out, uint32_t(p_data.size()));
    const size_t start = r_out.size();
    r_out.insert(r_out.end(), p_type, p_type + 4);
    r_out.insert(r_out.end(), p_data.begin(), p_data.end());
    png_be32(r_out, png_crc(r_out.data() + start, r_out.size() - start) ^ 0xFFFFFFFFu);
}

} // namespace detail

// The PNG file of a p_width x p_height image, p_channels (1 grey, 3 RGB, 4 RGBA) bytes per pixel, rows top first.
inline bytes encode_png(uint32_t p_width, uint32_t p_height, uint32_t p_channels, const uint8_t *p_pixels) {
    bytes out = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' };
    bytes header;
    detail::png_be32(header, p_width);
    detail::png_be32(header, p_height);
    const uint8_t color_type = p_channels == 1 ? 0 : (p_channels == 3 ? 2 : 6);
    header.insert(header.end(), { 8, color_type, 0, 0, 0 });
    detail::png_chunk(out, "IHDR", header);
    // The scanlines (filter 0) in a zlib stream of stored blocks of at most 65535 bytes.
    bytes raw;
    const size_t row = size_t(p_width) * p_channels;
    for (uint32_t y = 0; y < p_height; y++) {
        raw.push_back(0);
        raw.insert(raw.end(), p_pixels + y * row, p_pixels + (y + 1) * row);
    }
    bytes z = { 0x78, 0x01 };
    for (size_t at = 0; at < raw.size() || at == 0; at += 65535) {
        const size_t n = raw.size() - at < 65535 ? raw.size() - at : 65535;
        z.push_back(at + n >= raw.size() ? 1 : 0); // BFINAL on the last block
        z.push_back(uint8_t(n));
        z.push_back(uint8_t(n >> 8));
        z.push_back(uint8_t(~n));
        z.push_back(uint8_t(~n >> 8));
        z.insert(z.end(), raw.begin() + std::ptrdiff_t(at), raw.begin() + std::ptrdiff_t(at + n));
        if (raw.empty()) {
            break;
        }
    }
    uint32_t a = 1, b = 0; // adler32
    for (const uint8_t v : raw) {
        a = (a + v) % 65521u;
        b = (b + a) % 65521u;
    }
    detail::png_be32(z, (b << 16) | a);
    detail::png_chunk(out, "IDAT", z);
    detail::png_chunk(out, "IEND", bytes());
    return out;
}

} // namespace ov
