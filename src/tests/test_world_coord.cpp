#include "test_common.hpp"

#include <godot_cpp/templates/hash_set.hpp>

#include <climits>

using namespace WorldCoord;

TEST_CASE("WorldCoord: floor_div rounds towards minus infinity") {
    CHECK(floor_div(7, 2) == 3);
    CHECK(floor_div(-7, 2) == -4);
    CHECK(floor_div(-8, 2) == -4);
    CHECK(floor_div(0, 5) == 0);
    CHECK(floor_div(-1, 458752) == -1);
    CHECK(floor_div(458752, 458752) == 1);
    CHECK(floor_div(-458752, 458752) == -1);
    CHECK(floor_div(-458753, 458752) == -2);
}

TEST_CASE("WorldCoord: ctz64 is the index of the lowest set bit") {
    CHECK(ctz64(1) == 0);
    CHECK(ctz64(8) == 3);
    CHECK(ctz64(12) == 2);
    CHECK(ctz64(uint64_t(1) << 63) == 63);
    CHECK(ctz64(uint64_t(int64_t(-2))) == 1);
}

TEST_CASE("WorldCoord: low and high words rebuild the int64") {
    const int64_t values[] = { 0, 1, -1, 7, (int64_t(5) << 32) + 7, -(int64_t(5) << 32) - 7, int64_t(1) << 60,
        -(int64_t(1) << 60), INT64_MAX, INT64_MIN, int64_t(INT32_MAX) + 1, int64_t(INT32_MIN) - 1 };
    for (const int64_t v : values) {
        CAPTURE(v);
        CHECK(int64_t(hi32(v)) * (int64_t(1) << 32) + int64_t(lo32(v)) == v);
    }
    CHECK(lo32(-1) == 0xFFFFFFFFu);
    CHECK(hi32(-1) == -1);
    CHECK(lo32((int64_t(5) << 32) + 7) == 7u);
    CHECK(hi32((int64_t(5) << 32) + 7) == 5);
    CHECK(as_i32(0x80000000u) == INT32_MIN);
    CHECK(as_i32(0xFFFFFFFFu) == -1);
}

TEST_CASE("WorldCoord: lo32 differences wrap, exact under 2^31") {
    // What the GPU does with low words: a difference across a 2^32 boundary is still exact.
    const int64_t a = (int64_t(1) << 32) - 3;
    const int64_t b = (int64_t(1) << 32) + 5;
    CHECK(as_i32(lo32(b) - lo32(a)) == 8);
    CHECK(as_i32(lo32(a) - lo32(b)) == -8);
    CHECK(lo32_i(Vector3i64(b, a, -1)) == Vector3i(5, -3, -1));
    CHECK(hi32_i(Vector3i64(b, a, -1)) == Vector3i(1, 0, -1));
}

TEST_CASE("WorldCoord: Vector3i64 arithmetic") {
    const Vector3i64 a(1, -2, int64_t(3) << 40);
    const Vector3i64 b(10, 20, -30);
    CHECK(a + b == Vector3i64(11, 18, (int64_t(3) << 40) - 30));
    CHECK(a - b == Vector3i64(-9, -22, (int64_t(3) << 40) + 30));
    CHECK(a * 3 == Vector3i64(3, -6, int64_t(9) << 40));
    Vector3i64 c = a;
    c += b;
    CHECK(c == a + b);
    CHECK(a != b);
    CHECK(Vector3i64(Vector3i(4, -5, 6)) == Vector3i64(4, -5, 6));
    CHECK(Vector3i64(0, 0, 0).distance_to(Vector3i64(3, 4, 0)) == 5.0);
}

TEST_CASE("WorldCoord: the hasher tells apart ids that differ only in their high words") {
    const Vector3i64 a(1, 2, 3);
    const Vector3i64 b(1 + (int64_t(1) << 32), 2, 3);
    CHECK(Vector3i64Hasher::hash(a) == Vector3i64Hasher::hash(Vector3i64(1, 2, 3)));
    CHECK(Vector3i64Hasher::hash(a) != Vector3i64Hasher::hash(b));

    HashSet<Vector3i64, Vector3i64Hasher> set;
    set.insert(a);
    set.insert(b);
    CHECK(set.size() == 2);
    CHECK(set.has(Vector3i64(1, 2, 3)));
    CHECK_FALSE(set.has(Vector3i64(1 + (int64_t(1) << 33), 2, 3)));
}

TEST_CASE("WorldCoord: to_string keeps the full int64") {
    CHECK(to_string(Vector3i64(1, -2, int64_t(3000000000000))) == String("(1, -2, 3000000000000)"));
}
