#include "test_common.hpp"

#include "page_format.hpp"

#include <godot_cpp/variant/string.hpp>

#include <vector>

using namespace PageFormat;

namespace {

// Test mirrors of the GLSL encoders in GLSL_PAGE_FORMAT (pack_normal_oct22, encode_corner,
// store_group_header), written from the layout documented in page_format.hpp.
uint32_t pack_normal_oct22(Vector3 p_n) {
    p_n /= MAX(std::abs(p_n.x) + std::abs(p_n.y) + std::abs(p_n.z), 1e-20f);
    float ox = p_n.x;
    float oy = p_n.y;
    if (p_n.z < 0.0f) {
        ox = (1.0f - std::abs(p_n.y)) * (p_n.x >= 0.0f ? 1.0f : -1.0f);
        oy = (1.0f - std::abs(p_n.x)) * (p_n.y >= 0.0f ? 1.0f : -1.0f);
    }
    const uint32_t qx = uint32_t(std::round(CLAMP(ox * 0.5f + 0.5f, 0.0f, 1.0f) * 2047.0f));
    const uint32_t qy = uint32_t(std::round(CLAMP(oy * 0.5f + 0.5f, 0.0f, 1.0f) * 2047.0f));
    return qx | (qy << 11);
}

void encode_corner(uint16_t *r_corner, const Vector3 &p_local, uint32_t p_normal22, float p_pos_scale) {
    const float q[3] = { p_local.x, p_local.y, p_local.z };
    for (int i = 0; i < 3; i++) {
        const uint32_t v = uint32_t(CLAMP(std::round(q[i] * p_pos_scale), 0.0f, 16383.0f));
        r_corner[i] = uint16_t(v | ((p_normal22 >> (2 * i)) & 3u) << 14);
    }
    r_corner[3] = uint16_t(p_normal22 >> 6);
}

void store_group_header(uint16_t *r_row, const Vector3i &p_chunk) {
    const uint32_t c[3] = { uint32_t(p_chunk.x), uint32_t(p_chunk.y), uint32_t(p_chunk.z) };
    const uint16_t texels[8] = { uint16_t(c[0] & 65535u), uint16_t(c[0] >> 16), uint16_t(c[1] & 65535u), uint16_t(c[1] >> 16),
        uint16_t(c[2] & 65535u), uint16_t(c[2] >> 16), 0, 0 };
    for (int i = 0; i < 8; i++) {
        r_row[i] = texels[i];
    }
}

} // namespace

TEST_CASE("PageFormat: layout constants") {
    CHECK(TEX_WIDTH == 258);
    CHECK(ROW_BYTES == 258 * 8);
    CHECK(pos_frac_bits(32) == 9);
    CHECK(pos_scale(32) == 512.0f);
    CHECK(pos_frac_bits(16) == 10);
}

TEST_CASE("PageFormat: the GLSL side uses the same constants") {
    const String glsl(GLSL_PAGE_FORMAT);
    CHECK(glsl.contains(vformat("#define HEADER_TEXELS %du", HEADER_TEXELS)));
    CHECK(glsl.contains(vformat("float(1 << (%d - ", POS_BITS)));
    // GLSL page_pos_scale(32): 14 - (findMSB(31) + 1) = 9 fractional bits, like pos_frac_bits.
    CHECK(glsl.contains("findMSB(chunk_size - 1) + 1"));
}

TEST_CASE("PageFormat: corner addressing") {
    std::vector<uint16_t> page(2 * TEX_WIDTH * TEXEL_CHANNELS, 0);
    // Quad 65 is the second quad of group 1: row 1, texel HEADER_TEXELS + 1 * 4 + k.
    CHECK(corner(page.data(), 65, 2) - page.data() == (1 * TEX_WIDTH + HEADER_TEXELS + 4 + 2) * TEXEL_CHANNELS);
    CHECK(row(page.data(), 1) - page.data() == TEX_WIDTH * TEXEL_CHANNELS);
    CHECK(is_zero(corner(page.data(), 0, 0)));
}

TEST_CASE("PageFormat: group header keeps the low words of the chunk id") {
    uint16_t row_data[8];
    for (const Vector3i chunk : { Vector3i(-3, 70000, -2147483647), Vector3i(1, -1, 2147483647), Vector3i(0, 0, 0) }) {
        CAPTURE(chunk);
        store_group_header(row_data, chunk);
        CHECK(header_chunk(row_data) == chunk);
    }
}

TEST_CASE("PageFormat: positions round to the fixed-point step, normal bits don't leak in") {
    const float scale = pos_scale(32);
    uint16_t c[4];
    for (const Vector3 local : { Vector3(0, 0, 0), Vector3(3.25, 0.001, 30.998), Vector3(31, 31, 31) }) {
        CAPTURE(local);
        encode_corner(c, local, 0x3FFFFFu, scale); // every normal bit set
        const Vector3 back = decode_local(c, scale);
        CHECK(std::abs(back.x - local.x) <= 0.5f / scale);
        CHECK(std::abs(back.y - local.y) <= 0.5f / scale);
        CHECK(std::abs(back.z - local.z) <= 0.5f / scale);
    }
}

TEST_CASE("PageFormat: octahedral normals survive within 0.2 degrees") {
    const Vector3 normals[] = { Vector3(0, 0, 1), Vector3(0, 0, -1), Vector3(1, 0, 0), Vector3(0, -1, 0),
        Vector3(1, 1, 1).normalized(), Vector3(-1, 2, -3).normalized(), Vector3(0.3, -0.9, -0.1).normalized() };
    const float min_dot = std::cos(0.2f * float(Math_PI) / 180.0f);
    uint16_t c[4];
    for (const Vector3 &n : normals) {
        CAPTURE(n);
        encode_corner(c, Vector3(1, 2, 3), pack_normal_oct22(n), pos_scale(32));
        CHECK(decode_normal(c).dot(n) >= min_dot);
    }
}
