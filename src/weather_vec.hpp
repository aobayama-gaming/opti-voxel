#pragma once

#include <cmath>

// The weather's small vector helpers (double 3-vectors in the planet's axes), shared by the simulation's passes.
namespace weather_vec {

inline double dot3(const double p_a[3], const double p_b[3]) {
    return p_a[0] * p_b[0] + p_a[1] * p_b[1] + p_a[2] * p_b[2];
}

inline void cross3(const double p_a[3], const double p_b[3], double r[3]) {
    r[0] = p_a[1] * p_b[2] - p_a[2] * p_b[1];
    r[1] = p_a[2] * p_b[0] - p_a[0] * p_b[2];
    r[2] = p_a[0] * p_b[1] - p_a[1] * p_b[0];
}

// p_v minus its component along the unit p_n.
inline void tangent(const double p_n[3], double r_v[3]) {
    const double d = dot3(p_n, r_v);
    for (int k = 0; k < 3; k++) {
        r_v[k] -= d * p_n[k];
    }
}

// The fraction of the way to its target a quantity relaxing over p_tau goes in p_dt.
inline double relax(double p_dt, double p_tau) {
    return 1.0 - std::exp(-p_dt / p_tau);
}

// The Coriolis turn of a wind tangent to the unit up p_up: rotated about it by -p_f p_dt (exact, the speed kept;
// f = 2 Omega n_y: to the right in the north).
inline void coriolis(const double p_up[3], double p_f, double p_dt, double r_wind[3]) {
    const double a = -p_f * p_dt;
    double nu[3];
    cross3(p_up, r_wind, nu);
    const double c = std::cos(a);
    const double s = std::sin(a);
    for (int k = 0; k < 3; k++) {
        r_wind[k] = r_wind[k] * c + nu[k] * s;
    }
}

} // namespace weather_vec
