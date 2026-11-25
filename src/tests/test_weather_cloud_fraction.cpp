#include "test_common.hpp"

#include "weather_cloud_fraction.hpp"
#include "weather_moist.hpp"

#include <cmath>

// The partial cloud (weather_cloud_fraction.hpp, docs/weather.md "Partial cloud"): Cuijpers & Bechtold's fits as
// Bechtold & Siebesma (1998) give them, the render's spread consistent with them, and the moist physics' condensation:
// Kessler's all-or-nothing with no spread, some cloud under saturation with one, the conserved variables kept.

TEST_CASE("WeatherCloud: the cloudy share and water follow Cuijpers & Bechtold") {
    CHECK(WeatherCloud::fraction(0.0) == doctest::Approx(0.5));
    CHECK(WeatherCloud::fraction(-2.0) == doctest::Approx(0.5 + 0.36 * std::atan(-3.1)));
    CHECK(WeatherCloud::fraction(-10.0) == 0.0);
    CHECK(WeatherCloud::fraction(10.0) == 1.0);
    CHECK(WeatherCloud::water(-2.0) == doctest::Approx(std::exp(-3.4)));
    // Over saturation: continuous at 0, within 1% of their quadratic over [1, 2], tending to Q1 (the uniform cell).
    CHECK(WeatherCloud::water(0.0) == doctest::Approx(std::exp(-1.0)));
    for (const double q1 : { 1.0, 1.5, 2.0 }) {
        const double theirs = std::exp(-1.0) + 0.66 * q1 + 0.086 * q1 * q1;
        CHECK(std::fabs(WeatherCloud::water(q1) / theirs - 1.0) < 0.01);
    }
    CHECK(WeatherCloud::water(10.0) == doctest::Approx(10.0).epsilon(1e-5));
}

TEST_CASE("WeatherCloud: a uniform noise through the spread saturates the cloudy share") {
    for (const double q1 : { -2.0, -1.0, 0.0, 0.7 }) {
        const int n = 20000;
        int cloudy = 0;
        for (int i = 0; i < n; i++) {
            const double u = (i + 0.5) / n;
            cloudy += q1 + WeatherCloud::spread(u) > 0.0;
        }
        CHECK(double(cloudy) / n == doctest::Approx(WeatherCloud::fraction(q1)).epsilon(1e-3));
    }
}

TEST_CASE("WeatherMoist: the condensation is Kessler's with no spread, partial with one, the water kept") {
    using M = WeatherMoist;
    const double exner = 0.99;
    const double pc = 3.8 / (std::pow(exner, 1.0 / M::KAPPA_EXNER) * M::P0_MB);
    // 94% humid air at 296 K.
    const double t = 296.0;
    const double qs = pc * std::exp(17.27 * (t - 273.0) / (t - 36.0));
    const double theta = t / exner, qv = 0.94 * qs;
    double share, excess;
    CHECK(M::condense(theta, qv, 0.0, exner, pc, 0.0, share, excess) == 0.0);
    CHECK(share == 0.0);
    CHECK(excess < 0.0);
    const double sigma = 0.5e-3;
    const double cond = M::condense(theta, qv, 0.0, exner, pc, sigma, share, excess);
    MESSAGE("94% humidity, sigma_s 0.5 g/kg: Q1 ", excess / sigma, ", share ", share, ", cloud ", 1000.0 * cond, " g/kg");
    CHECK(share > 0.01);
    CHECK(share < 0.5);
    CHECK(cond > 0.0);
    // Supersaturated with no spread: the uniform cell's excess condenses, as Kessler's adjustment (to first order).
    const double wet = 1.02 * qs;
    const double uniform = M::condense(theta, wet, 0.0, exner, pc, 0.0, share, excess);
    CHECK(share == 1.0);
    CHECK(uniform == doctest::Approx(excess));
    CHECK(uniform == doctest::Approx(0.02 * qs / (1.0 + M::LATENT / M::CP * qs * 17.27 * 237.3 / ((t - 36.0) * (t - 36.0)))).epsilon(0.05));
}
