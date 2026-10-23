#pragma once

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <algorithm>
#include <cstdint>

// A fixed-size histogram of non-negative integers (latencies in frames or microseconds): exact below 2 * SUB, then
// SUB buckets per octave (1 / SUB relative width), so a percentile is within 1 / SUB of the true value and never above
// the largest value added. Cumulative: the bench's phases read it as it stands at their end.
class LatencyHist {
public:
    static constexpr int SUB_BITS = 4;
    static constexpr int SUB = 1 << SUB_BITS; // buckets per octave
    static constexpr int LINEAR = 2 * SUB; // values below are exact
    static constexpr int BUCKETS = LINEAR + (64 - SUB_BITS - 1) * SUB;

    static int bucket(uint64_t p_v) {
        if (p_v < uint64_t(LINEAR)) {
            return int(p_v);
        }
        const int msb = 63 - __builtin_clzll(p_v);
        const int shift = msb - SUB_BITS;
        return LINEAR + (msb - SUB_BITS - 1) * SUB + int((p_v >> shift) & uint64_t(SUB - 1));
    }
    // The largest value of bucket p_b.
    static uint64_t upper(int p_b) {
        if (p_b < LINEAR) {
            return uint64_t(p_b);
        }
        const int octave = (p_b - LINEAR) / SUB;
        const uint64_t sub = uint64_t((p_b - LINEAR) % SUB);
        const uint64_t lower = (uint64_t(SUB) + sub) << (octave + 1);
        return lower + (uint64_t(1) << (octave + 1)) - 1;
    }

    void add(uint64_t p_v) {
        counts[bucket(p_v)]++;
        n++;
        largest = std::max(largest, p_v);
    }
    uint64_t count() const { return n; }
    uint64_t max() const { return largest; }
    // The value of rank floor(p_q * (count - 1)) (the bench's percentile convention), rounded up to its bucket's largest
    // value, at most max(). 0 when empty.
    uint64_t percentile(double p_q) const {
        if (n == 0) {
            return 0;
        }
        const uint64_t rank = uint64_t(std::clamp(p_q, 0.0, 1.0) * double(n - 1));
        uint64_t seen = 0;
        for (int b = 0; b < BUCKETS; b++) {
            seen += counts[b];
            if (seen > rank) {
                return std::min(upper(b), largest);
            }
        }
        return largest;
    }
    void clear() { *this = LatencyHist(); }

    // { count, p50, p95, max }, the reports' form.
    godot::Dictionary to_dict() const {
        godot::Dictionary d;
        d["count"] = int64_t(n);
        d["p50"] = int64_t(percentile(0.5));
        d["p95"] = int64_t(percentile(0.95));
        d["max"] = int64_t(largest);
        return d;
    }

private:
    uint64_t counts[BUCKETS] = {};
    uint64_t n = 0;
    uint64_t largest = 0;
};
