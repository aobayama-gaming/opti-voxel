#include "test_common.hpp"

#include "ov/clock.hpp"
#include "ov/containers.hpp"
#include "ov/format.hpp"
#include "ov/log.hpp"
#include "ov/murmur3.hpp"
#include "ov/str.hpp"
#include "ov/types.hpp"

// The murmur chain being compared against godot's, explicit (test files may include godot; the core
// may not).
#include <godot_cpp/templates/hashfuncs.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

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
    // Floats take a double (no 64-bit length letters: %llf would read a long double).
    CHECK(ov::format("%.2f|%e|%g", 1.5, 2.0f, 0.25) == "1.50|2.000000e+00|0.25");
    // %s applies its flags, width and precision to the text.
    CHECK(ov::format("%-6s|%4s|%.2s", "ab", 7, "xyz") == "ab    |   7|xy");
}

TEST_CASE("ov::format: a spec / argument mismatch is reported, never silent") {
    static int errors = 0;
    errors = 0;
    const ov::ErrorSink previous = ov::error_sink();
    ov::set_error_sink([](const char *, int, const char *) { errors++; });
    CHECK(ov::format("%d %d", 1) == "1 %d"); // not enough arguments
    CHECK(errors == 1);
    CHECK(ov::format("abc", 7) == "abc"); // more arguments than conversions: stops at the end
    CHECK(errors == 2);
    CHECK(ov::format("%ld|%d", 5L, 6) == "%ld|6"); // unsupported conversion: verbatim, argument consumed
    CHECK(errors == 3);
    ov::set_error_sink(previous);
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

namespace {
// 8 hash values for every key: long probe runs that wrap around the index, shifts on every erase.
struct CollidingHasher {
    static uint32_t hash(const uint32_t &p_v) { return (p_v % 8) * 0x9E3779B1u; }
};
} // namespace

TEST_CASE("ov::HashMap: random inserts and erases match std::unordered_map") {
    ov::HashMap<uint32_t, uint32_t, CollidingHasher> map;
    std::unordered_map<uint32_t, uint32_t> oracle;
    uint32_t state = 1;
    for (int step = 0; step < 20000; step++) {
        state = ov::fmix32(state + uint32_t(step));
        const uint32_t key = state % 300;
        switch (state >> 30) {
            case 0:
                CHECK(map.erase(key) == (oracle.erase(key) != 0));
                break;
            case 1:
                map[key] += 1;
                oracle[key] += 1;
                break;
            default:
                map.insert(key, state);
                oracle[key] = state;
                break;
        }
        if (step % 5000 == 4999) {
            map.clear(); // keeps the index: later inserts reuse it
            oracle.clear();
        }
    }
    CHECK(map.size() == oracle.size());
    size_t seen = 0;
    for (const auto &kv : map) {
        const auto it = oracle.find(kv.first);
        CHECK((it != oracle.end() && kv.second == it->second));
        seen++;
    }
    CHECK(seen == oracle.size());
    for (uint32_t key = 0; key < 300; key++) {
        const uint32_t *v = map.getptr(key);
        CHECK((v != nullptr) == (oracle.count(key) != 0));
    }
}

TEST_CASE("ov::HashSet: remove(begin()) until empty pops every key once") {
    ov::HashSet<uint32_t, CollidingHasher> set;
    for (uint32_t key = 0; key < 1000; key++) {
        set.insert(key * 7);
    }
    std::vector<bool> popped(1000, false);
    while (!set.is_empty()) {
        const uint32_t key = *set.begin();
        set.remove(set.begin());
        CHECK(key % 7 == 0);
        CHECK_FALSE(popped[key / 7]);
        popped[key / 7] = true;
        CHECK_FALSE(set.has(key));
    }
    CHECK(std::count(popped.begin(), popped.end(), true) == 1000);
}

TEST_CASE("ov::replace_all / count: godot String's replace and count") {
    // Left to right, no overlap, replaced text not rescanned.
    CHECK(ov::replace_all("aaa", "aa", "b") == "ba");
    CHECK(ov::replace_all("abab", "ab", "abab") == "abababab");
    CHECK(ov::replace_all("x DATUM_LO NEG_DATUM_LO", "NEG_DATUM_LO", "1") == "x DATUM_LO 1");
    CHECK(ov::replace_all("none", "zz", "y") == "none");
    CHECK(ov::replace_all("keep", "", "y") == "keep");
    CHECK(ov::replace_all("", "a", "b") == "");
    CHECK(ov::count("aaaa", "aa") == 2);
    CHECK(ov::count("abcabc", "abc") == 2);
    CHECK(ov::count("abc", "") == 0);
    CHECK(ov::contains("vec4 f()", "f("));
    CHECK(!ov::contains("vec4 f()", "g("));
    CHECK(ov::itos(-3000000000000) == "-3000000000000");
    // The same results as godot's String methods on a splice-shaped text.
    const char *text = "MOMENT_MATERIAL_ISO MOMENT_ISO MOMENT_ISO";
    CHECK(ov::replace_all(ov::replace_all(text, "MOMENT_MATERIAL_ISO", "1"), "MOMENT_ISO", "2") ==
            std::string(String(text).replace("MOMENT_MATERIAL_ISO", "1").replace("MOMENT_ISO", "2").utf8().get_data()));
    CHECK(ov::count(text, "MOMENT_ISO") == String(text).count("MOMENT_ISO"));
}

TEST_CASE("ov::min / max / clamp: godot's MIN / MAX / CLAMP, operand for operand") {
    const float nan = std::nanf("");
    CHECK(ov::min(1, 2) == godot::MIN(1, 2));
    CHECK(ov::max(1, 2) == godot::MAX(1, 2));
    // NaN: the macro's comparison is false, so it picks the second operand (std::min picks the first).
    CHECK(std::isnan(ov::min(1.0f, nan)) == std::isnan(godot::MIN(1.0f, nan)));
    CHECK(std::isnan(ov::min(nan, 1.0f)) == std::isnan(godot::MIN(nan, 1.0f)));
    CHECK(std::isnan(ov::max(1.0f, nan)) == std::isnan(godot::MAX(1.0f, nan)));
    // Signed zeros: equal, so the second operand.
    CHECK(std::signbit(ov::min(0.0f, -0.0f)) == std::signbit(godot::MIN(0.0f, -0.0f)));
    CHECK(std::signbit(ov::max(-0.0f, 0.0f)) == std::signbit(godot::MAX(-0.0f, 0.0f)));
    CHECK(ov::clamp(5, 0, 3) == godot::CLAMP(5, 0, 3));
    CHECK(ov::clamp(-1, 0, 3) == godot::CLAMP(-1, 0, 3));
    CHECK(std::isnan(ov::clamp(nan, 0.0f, 1.0f)) == std::isnan(godot::CLAMP(nan, 0.0f, 1.0f)));
}

namespace {
std::vector<std::string> g_log_lines;
void capture_sink(int p_level, const char *, int, const char *p_message) {
    g_log_lines.push_back(std::to_string(p_level) + ":" + p_message);
}
} // namespace

TEST_CASE("ov::log: OV_LOG filters on the level, formats only what it prints") {
    const ov::LogSink was_sink = ov::log_sink();
    const int was_level = ov::log_level().load();
    g_log_lines.clear();
    ov::set_log_sink(capture_sink);
    ov::log_level() = 1;
    int formatted = 0;
    auto arg = [&formatted]() { formatted++; return 7; };
    OV_LOG(ov::LOG_INFO, "info %d", arg());
    OV_LOG(ov::LOG_VERBOSE, "verbose %d", arg());
    OV_WARN_PRINT("warned");
    ov::set_log_sink(was_sink);
    ov::log_level() = was_level;
    CHECK(formatted == 1);
    CHECK(g_log_lines.size() == 2);
    if (g_log_lines.size() != 2) {
        return;
    }
    CHECK(g_log_lines[0] == "1:info 7");
    CHECK(g_log_lines[1] == "-1:warned");
}

TEST_CASE("ov::now_usec: the installed clock, monotonic") {
    const uint64_t a = ov::now_usec();
    const uint64_t b = ov::now_usec();
    CHECK(b >= a);
    CHECK(ov::steady_usec() > 0);
}
