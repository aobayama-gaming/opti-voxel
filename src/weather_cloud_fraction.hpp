#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

// The partial cloud of a cell (docs/weather.md "Partial cloud"): the air inside a 45 m cell is not uniform, so a cell
// whose mean is a little under saturation still holds some cloud where it is moister. The statistical cloud scheme of
// Sommeria & Deardorff (1977) and Mellor (1977), with the fits Cuijpers & Bechtold (1995) made to large-eddy simulations
// (trade-wind cumulus, BOMEX, to stratocumulus), as Bechtold & Siebesma (1998, JAS 55, appendix B) give them:
// - s = a (q_w - q_sl): the cell mean's saturation excess (q_w the total water, q_sl the saturation at the liquid water
//   temperature T_l, a = 1 / (1 + L / c_p dq_s/dT) their appendix A, Kessler's Clausius-Clapeyron denominator);
//   sigma_s its spread inside the cell; Q1 = s / sigma_s.
// - The cloudy share N = max(0, min(1, 0.5 + 0.36 atan(1.55 Q1))) (B1).
// - The mean cloud water q_l / sigma_s = exp(1.2 Q1 - 1) under saturation (B2), over it Q1 + exp(-1) exp(-1.2 Q1):
//   their quadratic (e^-1 + 0.66 Q1 + 0.086 Q1^2) grows as Q1^2 and never meets the uniform cell's q_l = s when the
//   spread goes to 0; this one is continuous at 0, within 1% of theirs over [1, 2] and tends to s.
// - The spread's distribution: N = P(s > 0) for s = s_mean + sigma_s n makes n's CDF F(x) = 0.5 + 0.36 atan(1.55 x) on
//   |x| <= REACH (it reaches 0 and 1 there: the density 0.36 * 1.55 / (1 + (1.55 x)^2) holds all the probability). A
//   value u of a uniform noise is the spread n = tan((u - 0.5) / 0.36) / 1.55 (spread()); the share of a cell's points
//   it saturates is N.
// - The render (clouds.cpp): a point's cloud is max(Q1 + n, 0) / mean_excess(Q1) of the cell's mean water: its points
//   average to the mean water, the cloudy ones are the share N. The physics never reads the noise.
namespace WeatherCloud {

constexpr double SHARE_SLOPE = 0.36;
constexpr double SHARE_SCALE = 1.55;
constexpr double Q1_LIMIT = 8.0; // a published Q1's range (the uniform cell's: +-Q1_LIMIT)

// The spread's reach: F(REACH) = 1.
inline double reach() {
    return std::tan(0.5 / SHARE_SLOPE) / SHARE_SCALE;
}

// The cloudy share at Q1.
inline double fraction(double p_q1) {
    const double n = 0.5 + SHARE_SLOPE * std::atan(SHARE_SCALE * p_q1);
    return n < 0.0 ? 0.0 : (n > 1.0 ? 1.0 : n);
}

// The mean cloud water over sigma_s at Q1.
inline double water(double p_q1) {
    return p_q1 < 0.0 ? std::exp(1.2 * p_q1 - 1.0) : p_q1 + std::exp(-1.0 - 1.2 * p_q1);
}

// The spread (in units of sigma_s) of the noise value p_u in [0, 1] (the inverse of the spread's CDF).
inline double spread(double p_u) {
    const double half = 0.5 / SHARE_SLOPE;
    double x = (p_u - 0.5) / SHARE_SLOPE;
    x = x < -half ? -half : (x > half ? half : x);
    return std::tan(x) / SHARE_SCALE;
}

// The mean of max(Q1 + n, 0) over the spread's distribution: SHARE_SLOPE [Q1 (atan(k X) - atan(k l)) + ln((1 + k^2 X^2)
// / (1 + k^2 l^2)) / (2 k)], k = SHARE_SCALE, X = reach(), l = max(-Q1, -X).
inline double mean_excess(double p_q1) {
    const double k = SHARE_SCALE, x = reach();
    const double l = -p_q1 > -x ? -p_q1 : -x;
    if (l >= x) {
        return 0.0;
    }
    return SHARE_SLOPE * (p_q1 * (std::atan(k * x) - std::atan(k * l)) + std::log((1.0 + k * k * x * x) / (1.0 + k * k * l * l)) / (2.0 * k));
}

// The GLSL twins of spread() and mean_excess() (float; the render's).
const char *glsl();

// p_size^3 values (x fastest) of a tileable gradient noise (p_lattice cells per side, then p_octaves - 1 octaves each
// twice as fine at half the amplitude), their ranks turned into values in (0, 1): exactly uniform, as smooth as the
// noise (the rank keeps the order). The render's u.
std::vector<float> uniform_noise(int p_size, int p_lattice, int p_octaves, uint32_t p_seed);

} // namespace WeatherCloud
