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

TEST_CASE("GPUCostModel: a batch's measure pays its list words first, the rest is its LODs' first cost") {
    GPUCostModel m;
    m.resize(3);
    CHECK(m.estimate({ 0, 10, 0 }, 5000) == doctest::Approx(10 * GPUCostModel::DEFAULT_US + 5000 * GPUCostModel::WORD_US));
    m.observe({ 0, 10, 0 }, 6000.0, 5000); // 5,000 us of words, 1,000 for 10 chunks
    CHECK(m.estimate(1) == doctest::Approx(100.0));
    // The words' cost moved by their share of the estimate (5,000 / 5,600), damped.
    CHECK(m.us_per_word == doctest::Approx(1.0 + GPUCostModel::RATE * (5000.0 / 5600.0) * (6000.0 / 5600.0 - 1.0)));
    // Words overestimated: the LOD keeps at least the chunks' share of the measure.
    GPUCostModel o;
    o.resize(1);
    o.observe({ 10 }, 1000.0, 5000); // estimated 600 + 5,000
    CHECK(o.estimate(0) == doctest::Approx(1000.0 * 600.0 / 5600.0 / 10.0));
    // Without words: the chunk-only model.
    GPUCostModel p;
    p.resize(1);
    p.observe({ 10 }, 200.0);
    CHECK(p.estimate(0) == doctest::Approx(20.0));
    CHECK(p.us_per_word == GPUCostModel::WORD_US);
}

TEST_CASE("GPUCostModel: batches with short and long lists separate the chunks' cost from the words'; an outlier moves both a bounded step") {
    GPUCostModel m;
    m.resize(2);
    const auto truth = [](uint32_t p_chunks, uint64_t p_words) { return 20.0 * p_chunks + 0.5 * double(p_words); };
    for (int i = 0; i < 400; i++) {
        const uint64_t words = i % 2 ? 1000 : 20000;
        m.observe({ 10, 0 }, truth(10, words), words);
    }
    CHECK(m.estimate(0) == doctest::Approx(20.0).epsilon(0.05));
    CHECK(m.us_per_word == doctest::Approx(0.5).epsilon(0.05));
    const double chunk = m.estimate(0), word = m.us_per_word;
    m.observe({ 10, 0 }, 1e9, 20000); // a lost frame: clamped to MAX_SCALE, then damped by RATE
    const double bound = 1.0 + GPUCostModel::RATE * (GPUCostModel::MAX_SCALE - 1.0);
    CHECK(m.estimate(0) <= chunk * bound + 1e-9);
    CHECK(m.us_per_word <= word * bound + 1e-9);
    CHECK(m.us_per_word > word);
}

TEST_CASE("GPUCostModel: a chunk's word bound counts at the share the lists reach, a chunk without one at the lists' mean") {
    GPUCostModel m;
    m.resize(1);
    CHECK(m.estimate_chunk(0, 1000) == doctest::Approx(GPUCostModel::DEFAULT_US + 1000.0)); // the bound until a list says otherwise
    for (int i = 0; i < 60; i++) {
        m.observe_list(100000, 10000, 10); // the lists hold a tenth of the bounds
    }
    CHECK(m.list_share == doctest::Approx(0.1).epsilon(0.01));
    CHECK(m.words_per_chunk == doctest::Approx(0.0).epsilon(0.01));
    CHECK(m.estimate_chunk(0, 1000) == doctest::Approx(GPUCostModel::DEFAULT_US + 100.0).epsilon(0.01));
    for (int i = 0; i < 200; i++) {
        m.observe_list(1, 1000000, 1); // a bound far below the list: the share bounded
    }
    CHECK(m.list_share <= GPUCostModel::LIST_SHARE_MAX);

    // No streamed source: every chunk's bound is 0, its words the lists' mean.
    GPUCostModel s;
    s.resize(1);
    for (int i = 0; i < 60; i++) {
        s.observe_list(0, 5000, 10);
    }
    CHECK(s.list_share == 1.0);
    CHECK(s.words_per_chunk == doctest::Approx(500.0).epsilon(0.01));
    CHECK(s.estimate_chunk(0, 0) == doctest::Approx(GPUCostModel::DEFAULT_US + 500.0).epsilon(0.01));
    s.observe_list(0, 500, 0); // no chunk: ignored
    CHECK(s.words_per_chunk == doctest::Approx(500.0).epsilon(0.01));
}
