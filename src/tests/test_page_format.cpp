#include "test_common.hpp"

#include "page_format.hpp"

#include <godot_cpp/variant/string.hpp>

#include <random>
#include <vector>

using namespace PageFormat;

namespace {

// Test mirrors of the GLSL encoders in GLSL_PAGE_ENCODE (pack_normal_oct, encode_corner,
// store_group_header), written from the layout documented in page_format.hpp.
uint32_t pack_normal_oct(Vector3 p_n) {
    p_n /= MAX(std::abs(p_n.x) + std::abs(p_n.y) + std::abs(p_n.z), 1e-20f);
    float ox = p_n.x;
    float oy = p_n.y;
    if (p_n.z < 0.0f) {
        ox = (1.0f - std::abs(p_n.y)) * (p_n.x >= 0.0f ? 1.0f : -1.0f);
        oy = (1.0f - std::abs(p_n.x)) * (p_n.y >= 0.0f ? 1.0f : -1.0f);
    }
    const uint32_t qx = uint32_t(std::round(CLAMP(ox * 0.5f + 0.5f, 0.0f, 1.0f) * float(NORMAL_AXIS_MAX)));
    const uint32_t qy = uint32_t(std::round(CLAMP(oy * 0.5f + 0.5f, 0.0f, 1.0f) * float(NORMAL_AXIS_MAX)));
    return qx | (qy << NORMAL_AXIS_BITS);
}

void encode_corner(uint16_t *r_corner, const Vector3 &p_local, uint32_t p_normal, uint32_t p_palette, float p_pos_scale) {
    const float q[3] = { p_local.x, p_local.y, p_local.z };
    const uint32_t lane = (1u << PALETTE_LANE_BITS) - 1;
    for (int i = 0; i < 3; i++) {
        const uint32_t v = uint32_t(CLAMP(std::round(q[i] * p_pos_scale), 0.0f, float(POS_MASK)));
        r_corner[i] = uint16_t(v | ((p_palette >> (PALETTE_LANE_BITS * i)) & lane) << POS_BITS);
    }
    r_corner[3] = uint16_t(p_normal);
}

void store_group_header(uint16_t *r_row, const Vector3i &p_chunk) {
    const uint32_t c[3] = { uint32_t(p_chunk.x), uint32_t(p_chunk.y), uint32_t(p_chunk.z) };
    const uint16_t texels[8] = { uint16_t(c[0] & 65535u), uint16_t(c[0] >> 16), uint16_t(c[1] & 65535u), uint16_t(c[1] >> 16),
        uint16_t(c[2] & 65535u), uint16_t(c[2] >> 16), 0, 0 };
    for (int i = 0; i < 8; i++) {
        r_row[i] = texels[i];
    }
}

float angle_deg(const Vector3 &p_a, const Vector3 &p_b) {
    return std::acos(CLAMP(p_a.dot(p_b), -1.0f, 1.0f)) * 180.0f / float(Math_PI);
}

} // namespace

TEST_CASE("PageFormat: layout constants") {
    CHECK(TEX_WIDTH == 258);
    CHECK(ROW_BYTES == 258 * 8);
    CHECK(pos_frac_bits(32) == 8);
    CHECK(pos_scale(32) == 256.0f);
    CHECK(pos_frac_bits(16) == 9);
    CHECK(PALETTE_SIZE == 512);
}

TEST_CASE("PageFormat: the GLSL side uses the same constants") {
    const String encode = glsl_encode();
    const String decode = glsl_decode();
    for (const String &glsl : { encode, decode }) {
        CHECK(glsl.contains(vformat("const uint HEADER_TEXELS = %du;", HEADER_TEXELS)));
        CHECK(glsl.contains(vformat("const uint PAGE_POS_BITS = %du;", POS_BITS)));
        CHECK(glsl.contains(vformat("const uint PAGE_NORMAL_AXIS_BITS = %du;", NORMAL_AXIS_BITS)));
        CHECK(glsl.contains(vformat("const uint PAGE_PALETTE_LANE_BITS = %du;", PALETTE_LANE_BITS)));
        CHECK_FALSE(glsl.contains("#define"));
    }
    // GLSL page_pos_scale(32): POS_BITS - (findMSB(31) + 1) fractional bits, like pos_frac_bits.
    CHECK(encode.contains("int(PAGE_POS_BITS) - (findMSB(chunk_size - 1) + 1)"));
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

TEST_CASE("PageFormat: position, palette and normal fields don't leak into each other") {
    const float scale = pos_scale(32);
    uint16_t c[4];
    const Vector3 n = Vector3(-1, 2, -3).normalized();
    for (const Vector3 local : { Vector3(0, 0, 0), Vector3(3.25, 0.001, 30.998), Vector3(31, 31, 31) }) {
        for (const uint32_t palette : { 0u, 1u, 0x155u, PALETTE_SIZE - 1 }) {
            CAPTURE(local);
            CAPTURE(palette);
            encode_corner(c, local, pack_normal_oct(n), palette, scale);
            const Vector3 back = decode_local(c, scale);
            CHECK(std::abs(back.x - local.x) <= 0.5f / scale);
            CHECK(std::abs(back.y - local.y) <= 0.5f / scale);
            CHECK(std::abs(back.z - local.z) <= 0.5f / scale);
            CHECK(decode_palette(c) == palette);
            CHECK(angle_deg(decode_normal(c), n) <= NORMAL_MAX_ERROR_DEG);
        }
    }
}

TEST_CASE("PageFormat: octahedral normals survive within NORMAL_MAX_ERROR_DEG") {
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss;
    std::vector<Vector3> normals = { Vector3(0, 0, 1), Vector3(0, 0, -1), Vector3(1, 0, 0), Vector3(0, -1, 0),
        Vector3(1, 1, 1).normalized(), Vector3(0.3, -0.9, -0.1).normalized() };
    for (int i = 0; i < 20000; i++) {
        normals.push_back(Vector3(gauss(rng), gauss(rng), gauss(rng)).normalized());
    }
    uint16_t c[4];
    float worst = 0.0f;
    for (const Vector3 &n : normals) {
        encode_corner(c, Vector3(1, 2, 3), pack_normal_oct(n), 0, pos_scale(32));
        worst = MAX(worst, angle_deg(decode_normal(c), n));
    }
    CHECK(worst <= NORMAL_MAX_ERROR_DEG);
    CHECK(worst >= 0.5f * NORMAL_MAX_ERROR_DEG); // the bound is tight, not stale
}
