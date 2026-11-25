#pragma once

#include <cmath>

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
// - The spread's distribution: N = P(s > 0) for s = s_mean + sigma_s n makes n's CDF F(x) = 0.5 + 0.36 atan(1.55 x)
//   (clamped to [0, 1]): a value u of a uniform noise is the spread n = tan((u - 0.5) / 0.36) / 1.55 (the render's
//   remap: the share of a cell's points a noise saturates is N).
namespace WeatherCloud {

// The cloudy share at Q1.
inline double fraction(double p_q1) {
    const double n = 0.5 + 0.36 * std::atan(1.55 * p_q1);
    return n < 0.0 ? 0.0 : (n > 1.0 ? 1.0 : n);
}

// The mean cloud water over sigma_s at Q1.
inline double water(double p_q1) {
    return p_q1 < 0.0 ? std::exp(1.2 * p_q1 - 1.0) : p_q1 + std::exp(-1.0 - 1.2 * p_q1);
}

// The spread (in units of sigma_s) of the noise value p_u in [0, 1] (the inverse of the spread's CDF).
inline double spread(double p_u) {
    const double half = 0.5 / 0.36; // atan's reach for a CDF of 0 or 1
    double x = (p_u - 0.5) / 0.36;
    x = x < -half ? -half : (x > half ? half : x);
    return std::tan(x) / 1.55;
}

} // namespace WeatherCloud
