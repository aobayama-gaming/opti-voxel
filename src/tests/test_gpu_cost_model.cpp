#include "test_common.hpp"

#include "gpu_cost_model.hpp"

TEST_CASE("GPUCostModel: defaults, then the nearest measured LOD") {
    GPUCostModel m;
    m.resize(6);
    CHECK(m.estimate(3) == GPUCostModel::DEFAULT_US);
    m.observe({ 0, 0, 10, 0, 0, 0 }, 200.0); // LOD 2: 20 us per chunk
    CHECK(m.measured(2));
    CHECK(m.estimate(2) == doctest::Approx(20.0));
    CHECK(m.estimate(0) == doctest::Approx(20.0)); // nearest measured
    CHECK(m.estimate(5) == doctest::Approx(20.0));
    CHECK(m.estimate({ 1, 0, 2, 0, 0, 0 }) == doctest::Approx(60.0));
}

TEST_CASE("GPUCostModel: a measured LOD converges, outliers move it a bounded step") {
    GPUCostModel m;
    m.resize(4);
    m.observe({ 0, 4, 0, 0 }, 400.0); // 100 us
    for (int i = 0; i < 60; i++) {
        m.observe({ 0, 4, 0, 0 }, 200.0); // the true cost is 50 us
    }
    CHECK(m.estimate(1) == doctest::Approx(50.0).epsilon(0.01));
    const double before = m.estimate(1);
    m.observe({ 0, 4, 0, 0 }, 1e9); // a lost frame: clamped to MAX_SCALE, then damped by RATE
    CHECK(m.estimate(1) <= before * (1.0 + GPUCostModel::RATE * (GPUCostModel::MAX_SCALE - 1.0)) + 1e-9);
    m.observe({ 0, 4, 0, 0 }, 0.0); // no measure: ignored
    CHECK(m.measured(1));
}

TEST_CASE("GPUCostModel: a mixed batch corrects each LOD by its share") {
    GPUCostModel m;
    m.resize(3);
    m.observe({ 10, 0, 0 }, 100.0); // LOD 0: 10 us
    m.observe({ 0, 10, 0 }, 1000.0); // LOD 1: 100 us (a first measure is not clamped: 10x the borrowed cost)
    CHECK(m.estimate(1) == doctest::Approx(100.0));
    // Estimated 10 * 10 + 10 * 100 = 1100 us, measured twice that: LOD 1 holds 91% of the estimate.
    m.observe({ 10, 10, 0 }, 2200.0);
    const double up0 = m.estimate(0) / 10.0 - 1.0;
    const double up1 = m.estimate(1) / 100.0 - 1.0;
    CHECK(up0 > 0.0);
    CHECK(up1 > up0 * 5.0);
    CHECK(up1 == doctest::Approx(GPUCostModel::RATE * (1000.0 / 1100.0)));
}
