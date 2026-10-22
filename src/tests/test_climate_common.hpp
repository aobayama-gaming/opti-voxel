#pragma once

#include "test_common.hpp"

#include "cascade.hpp"

// What the climate tests share (test_climate.cpp, test_climate_shape.cpp).

// doctest's REQUIRE needs exceptions: a failed check ends the test case.
#define CLIMATE_REQUIRE(p_cond) \
    do {                        \
        CHECK(p_cond);          \
        if (!(p_cond)) {        \
            return;             \
        }                       \
    } while (0)

namespace ClimateTest {

inline int64_t differing_floats(const float *p_a, const float *p_b, int p_n) {
    int64_t n = 0;
    for (int i = 0; i < p_n; i++) {
        n += ExactNoise::float_bits(p_a[i]) != ExactNoise::float_bits(p_b[i]);
    }
    return n;
}

// Every lattice point of a box on Λ_lod vs Cascade::point_slots, every slot: the box run and the point query agree bit
// for bit (the GPU's chunks and the discoverer's points). p_all_fields: the fields carried everywhere.
inline int64_t box_point_mismatches(const Cascade::Spec &p_spec, const Vector3i64 &p_lo, int p_n, int p_lod, bool p_all_fields = false) {
    Cascade::Box box;
    box.lo[0] = p_lo.x;
    box.lo[1] = p_lo.y;
    box.lo[2] = p_lo.z;
    for (int a = 0; a < 3; a++) {
        box.hi[a] = box.lo[a] + p_n - 1;
    }
    const Cascade::Level l = Cascade::run(p_spec, box, p_lod, nullptr, Cascade::Source(), p_all_fields);
    int64_t bad = 0;
    Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
        float slots[Cascade::SLOTS];
        Cascade::point_slots(p_spec, p_p, p_lod, slots, p_all_fields);
        for (int s = 0; s < Cascade::SLOTS; s++) {
            if (!l.ch[s].empty()) {
                bad += ExactNoise::float_bits(l.ch[s][size_t(p_i)]) != ExactNoise::float_bits(slots[s]);
            }
        }
    });
    return bad;
}

} // namespace ClimateTest
