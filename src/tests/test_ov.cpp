#include "test_common.hpp"

#include "ov/containers.hpp"
#include "ov/format.hpp"
#include "ov/murmur3.hpp"
#include "ov/types.hpp"

// The murmur chain being compared against godot's, explicit (test files may include godot; the core
// may not).
#include <godot_cpp/templates/hashfuncs.hpp>

// The ov support layer (src/ov/): the engine-free vocabulary the core builds on. The format shim's
// results are pinned here because core report text and GLSL splices flow through it.

namespace {

// A bare hasher for the ov containers: the same murmur chain Vector3i64Hasher uses.
struct UintHasher {
    static uint32_t hash(const uint32_t &p_v) { return ov::fmix32(p_v); }
};

} // namespace

TEST_CASE("ov::format: the printf family the core uses") {
    // Integers, promoted to 64-bit like godot's vformat: an int64 argument is not truncated.
    CHECK(ov::format("(%d, %d, %d)", int64_t(1), int64_t(-2), int64_t(3000000000000)) == "(1, -2, 3000000000000)");
    CHECK(ov::format("%d", -5) == "-5");
    CHECK(ov::format("%u", uint32_t(4294967295)) == "4294967295");
    // Flags, width and precision go through verbatim (page_format / exact_noise GLSL constants).
    CHECK(ov::format("0x%08Xu", uint64_t(0xFF)) == "0x000000FFu");
    CHECK(ov::format("%08Xu", uint64_t(0xFF)) == "000000FFu");
    CHECK(ov::format("%s->%d", "hello", 7) == "hello->7");
    // %s: any argument renders as text, godot style.
    CHECK(ov::format("%s", std::string("std")) == "std");
    CHECK(ov::format("%s", 42) == "42");
    CHECK(ov::format("%s", 0.25f) == "0.25");
    CHECK(ov::format("%s", 1.0) == "1");
    // Literals: %% is one percent, and no following argument is consumed by it.
    CHECK(ov::format("100%%") == "100%");
    CHECK(ov::format("50%%: %d", 7) == "50%: 7");
}

TEST_CASE("ov::ivec3: the int32 triple at core edges") {
    ov::ivec3 a(1, -2, 3);
    CHECK(a.x == 1);
    CHECK(a.y == -2);
    CHECK(a.z == 3);
    CHECK(a == ov::ivec3(1, -2, 3));
    CHECK(a != ov::ivec3(1, -2, 4));
    CHECK_FALSE(a == ov::ivec3(0, 1, 2));
}

TEST_CASE("ov::murmur3: the vendored chain mirrors godot's hashfuncs") {
    // The one-32 chain and the final mix, from godot's HASH_MURMUR3_SEED 0x7F07C65. Pinned so the
    // vendored copy cannot drift: Vector3i64Hasher's hashes (and every container's layout) stay
    // exactly those of the pre-de-Godot code.
    CHECK(ov::murmur3_one_32(0u) == godot::hash_murmur3_one_32(0u));
    CHECK(ov::murmur3_one_32(0x9E3779B9u, 0xF00D) == godot::hash_murmur3_one_32(0x9E3779B9u, 0xF00D));
    CHECK(ov::fmix32(0x7F07C65u) == godot::hash_fmix32(0x7F07C65u));
    CHECK(ov::fmix32(0x12345678u) == godot::hash_fmix32(0x12345678u));
}

TEST_CASE("ov::HashMap: the godot call surface") {
    ov::HashMap<int, int, UintHasher> map;
    CHECK(map.is_empty());
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(1, 11); // insert replaces an existing key, godot style
    CHECK(map.size() == 2);
    CHECK(map.has(1));
    CHECK_FALSE(map.has(3));
    CHECK(*map.getptr(2) == 20);
    CHECK(map.getptr(3) == nullptr);
    CHECK(map.erase(1));
    CHECK_FALSE(map.erase(1));
    CHECK(map.size() == 1);
    int sum = 0;
    int keys = 0;
    for (const auto &kv : map) {
        sum += kv.second;
        keys += kv.first;
    }
    CHECK(sum == 20);
    CHECK(keys == 2);
    map.clear();
    CHECK(map.is_empty());
}

TEST_CASE("ov::HashSet: the godot call surface") {
    ov::HashSet<uint32_t, UintHasher> set;
    CHECK(set.is_empty());
    set.insert(5);
    set.insert(5); // an idempotent insert, as godot's
    CHECK(set.size() == 1);
    CHECK(set.has(5));
    CHECK_FALSE(set.has(6));
    set.reserve(100);
    CHECK_FALSE(set.erase(6));
    CHECK(set.erase(5));
    CHECK(set.is_empty());
}
