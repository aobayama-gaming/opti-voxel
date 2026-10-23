#include "test_common.hpp"

#include "latency_hist.hpp"

#include <random>

TEST_CASE("LatencyHist: buckets are contiguous, exact below LINEAR, and hold their values") {
    for (uint64_t v = 0; v < uint64_t(LatencyHist::LINEAR); v++) {
        CHECK(LatencyHist::bucket(v) == int(v));
        CHECK(LatencyHist::upper(int(v)) == v);
    }
    for (int b = LatencyHist::LINEAR; b < LatencyHist::BUCKETS - 1; b++) {
        CHECK(LatencyHist::bucket(LatencyHist::upper(b)) == b);
        CHECK(LatencyHist::bucket(LatencyHist::upper(b) + 1) == b + 1);
    }
    CHECK(LatencyHist::bucket(UINT64_MAX) == LatencyHist::BUCKETS - 1);
    CHECK(LatencyHist::upper(LatencyHist::BUCKETS - 1) == UINT64_MAX);
}

TEST_CASE("LatencyHist: percentiles follow the bench's rank, within a bucket's width, never above the max") {
    LatencyHist h;
    CHECK(h.percentile(0.5) == 0);
    CHECK(h.count() == 0);
    for (uint64_t v = 1; v <= 20; v++) {
        h.add(v);
    }
    CHECK(h.count() == 20);
    CHECK(h.max() == 20);
    CHECK(h.percentile(0.5) == 10); // rank int(0.5 * 19) = 9: the 10th value, exact below LINEAR
    CHECK(h.percentile(0.95) == 19);
    CHECK(h.percentile(1.0) == 20);

    LatencyHist big;
    std::mt19937_64 rng(7);
    std::vector<uint64_t> values;
    for (int i = 0; i < 5000; i++) {
        values.push_back(rng() % 100000);
        big.add(values.back());
    }
    std::sort(values.begin(), values.end());
    for (const double q : { 0.5, 0.95, 0.99 }) {
        const uint64_t exact = values[size_t(q * double(values.size() - 1))];
        const uint64_t got = big.percentile(q);
        CHECK(got >= exact);
        CHECK(double(got - exact) <= double(exact) / LatencyHist::SUB + 1.0);
    }
    CHECK(big.percentile(1.0) == values.back());
    big.clear();
    CHECK(big.count() == 0);
}
