// Mass properties of the archetype instances (instance_moments.hpp).
#include "instance_moments.hpp"

#include <algorithm>
#include <cmath>

namespace Archetype {

namespace {

// S as a full 3x3 (row major) and back.
void to_matrix(const double *p_s, double r_m[9]) {
    r_m[0] = p_s[0];
    r_m[4] = p_s[1];
    r_m[8] = p_s[2];
    r_m[1] = r_m[3] = p_s[3];
    r_m[2] = r_m[6] = p_s[4];
    r_m[5] = r_m[7] = p_s[5];
}

void from_matrix(const double *p_m, double r_s[6]) {
    r_s[0] = p_m[0];
    r_s[1] = p_m[4];
    r_s[2] = p_m[8];
    r_s[3] = 0.5 * (p_m[1] + p_m[3]);
    r_s[4] = 0.5 * (p_m[2] + p_m[6]);
    r_s[5] = 0.5 * (p_m[5] + p_m[7]);
}

// r_s += p_m * d d^T.
void add_outer(double r_s[6], double p_m, const double d[3]) {
    r_s[0] += p_m * d[0] * d[0];
    r_s[1] += p_m * d[1] * d[1];
    r_s[2] += p_m * d[2] * d[2];
    r_s[3] += p_m * d[0] * d[1];
    r_s[4] += p_m * d[0] * d[2];
    r_s[5] += p_m * d[1] * d[2];
}

} // namespace

void Moments::add(const Moments &p_other, const double p_offset[3]) {
    if (p_other.empty()) {
        return;
    }
    double oc[3];
    for (int a = 0; a < 3; a++) {
        oc[a] = p_other.c[a] + p_offset[a];
    }
    if (empty()) {
        *this = p_other;
        std::copy(oc, oc + 3, c);
        return;
    }
    const double m = m0 + p_other.m0;
    double nc[3], d0[3], d1[3];
    for (int a = 0; a < 3; a++) {
        nc[a] = (m0 * c[a] + p_other.m0 * oc[a]) / m;
        d0[a] = c[a] - nc[a];
        d1[a] = oc[a] - nc[a];
    }
    for (int i = 0; i < 6; i++) {
        s[i] += p_other.s[i];
    }
    add_outer(s, m0, d0);
    add_outer(s, p_other.m0, d1);
    m0 = m;
    std::copy(nc, nc + 3, c);
}

Moments Moments::scaled(double p_weight) const {
    Moments r = *this;
    r.m0 *= p_weight;
    for (double &v : r.s) {
        v *= p_weight;
    }
    return r;
}

Moments Moments::rotated(const float *p_r) const {
    Moments out = *this;
    local_to_world(p_r, c, out.c);
    double m[9], t[9], w[9];
    to_matrix(s, m);
    // t = S R, w = R^T t.
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            t[3 * i + j] = m[3 * i] * double(p_r[j]) + m[3 * i + 1] * double(p_r[3 + j]) + m[3 * i + 2] * double(p_r[6 + j]);
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            w[3 * i + j] = double(p_r[i]) * t[j] + double(p_r[3 + i]) * t[3 + j] + double(p_r[6 + i]) * t[6 + j];
        }
    }
    from_matrix(w, out.s);
    return out;
}

void Moments::principal(double r_lambda[3], double *r_axes) const {
    // Cyclic Jacobi on the covariance: a handful of sweeps converge to double precision for a 3x3.
    double a[9], v[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    to_matrix(s, a);
    const double inv = empty() ? 0.0 : 1.0 / m0;
    for (double &x : a) {
        x *= inv;
    }
    for (int sweep = 0; sweep < 32; sweep++) {
        const double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
        if (off <= 1e-30 * (a[0] * a[0] + a[4] * a[4] + a[8] * a[8])) {
            break;
        }
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                const double apq = a[3 * p + q];
                if (apq == 0.0) {
                    continue;
                }
                const double theta = (a[3 * q + q] - a[3 * p + p]) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double cs = 1.0 / std::sqrt(t * t + 1.0), sn = t * cs;
                for (int k = 0; k < 3; k++) { // columns p, q
                    const double akp = a[3 * k + p], akq = a[3 * k + q];
                    a[3 * k + p] = cs * akp - sn * akq;
                    a[3 * k + q] = sn * akp + cs * akq;
                }
                for (int k = 0; k < 3; k++) { // rows p, q
                    const double apk = a[3 * p + k], aqk = a[3 * q + k];
                    a[3 * p + k] = cs * apk - sn * aqk;
                    a[3 * q + k] = sn * apk + cs * aqk;
                }
                for (int k = 0; k < 3; k++) {
                    const double vkp = v[3 * k + p], vkq = v[3 * k + q];
                    v[3 * k + p] = cs * vkp - sn * vkq;
                    v[3 * k + q] = sn * vkp + cs * vkq;
                }
            }
        }
    }
    int order[3] = { 0, 1, 2 };
    std::sort(order, order + 3, [&a](int p_i, int p_j) { return a[4 * p_i] < a[4 * p_j]; });
    for (int i = 0; i < 3; i++) {
        r_lambda[i] = std::max(a[4 * order[i]], 0.0);
        if (r_axes) {
            for (int k = 0; k < 3; k++) {
                r_axes[3 * i + k] = v[3 * k + order[i]];
            }
        }
    }
}

void Moments::semi_axes(double r_axes[3]) const {
    double lambda[3];
    principal(lambda);
    for (int i = 0; i < 3; i++) {
        r_axes[i] = std::sqrt(5.0 * lambda[i]);
    }
}

double Moments::half_thickness() const {
    double axes[3];
    semi_axes(axes);
    return axes[0];
}

void MomentSums::add(double p_volume, const double p_at[3]) {
    v += p_volume;
    for (int a = 0; a < 3; a++) {
        x[a] += p_volume * p_at[a];
    }
    add_outer(xx, p_volume, p_at);
}

Moments MomentSums::moments() const {
    Moments m;
    if (!(v > 0.0)) {
        return m;
    }
    m.m0 = v;
    for (int a = 0; a < 3; a++) {
        m.c[a] = x[a] / v;
    }
    for (int i = 0; i < 6; i++) {
        m.s[i] = xx[i];
    }
    add_outer(m.s, -v, m.c);
    return m;
}

Moments box_moments(double p_hx, double p_hy, double p_hz) {
    Moments m;
    m.m0 = 8.0 * p_hx * p_hy * p_hz;
    m.s[0] = m.m0 * p_hx * p_hx / 3.0;
    m.s[1] = m.m0 * p_hy * p_hy / 3.0;
    m.s[2] = m.m0 * p_hz * p_hz / 3.0;
    return m;
}

Moments sphere_moments(double p_r) {
    Moments m;
    m.m0 = 4.0 / 3.0 * M_PI * p_r * p_r * p_r;
    m.s[0] = m.s[1] = m.s[2] = m.m0 * p_r * p_r / 5.0;
    return m;
}

Moments cylinder_moments(double p_half_height, double p_r) {
    return revolution_moments(p_r * p_r, 0.0, 0.0, -p_half_height, p_half_height);
}

Moments revolution_moments(double p_a, double p_b, double p_c, double p_y0, double p_y1) {
    // The integrands r^2, r^2 y, r^2 y^2 and r^4 are polynomials of degree <= 4: 5 points are exact to degree 9.
    static const double node[5] = { 0.0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640 };
    static const double weight[5] = { 0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891 };
    if (!(p_y1 > p_y0)) {
        return Moments();
    }
    const double half = 0.5 * (p_y1 - p_y0), mid = 0.5 * (p_y1 + p_y0);
    double v = 0.0, vy = 0.0, vyy = 0.0, vxx = 0.0;
    for (int i = 0; i < 5; i++) {
        const double y = mid + half * node[i];
        const double r2 = std::max(p_a + y * (p_b + y * p_c), 0.0);
        const double w = weight[i] * half;
        v += w * r2;
        vy += w * r2 * y;
        vyy += w * r2 * y * y;
        vxx += w * r2 * r2;
    }
    Moments m;
    m.m0 = M_PI * v;
    if (m.empty()) {
        return Moments();
    }
    m.c[1] = M_PI * vy / m.m0;
    m.s[1] = M_PI * vyy - m.m0 * m.c[1] * m.c[1];
    m.s[0] = m.s[2] = 0.25 * M_PI * vxx; // a disc of radius r: the integral of x^2 is pi r^4 / 4
    return m;
}

Moments tapered_capsule_moments(double p_h, double p_r0, double p_r1) {
    const double slope = p_h > 0.0 ? (p_r1 - p_r0) / p_h : 0.0;
    // Ball r0 at 0 below y = 0, the frustum (r0 + slope y)^2 while its radius is > 0, ball r1 at p_h above y = p_h:
    // r^2 = r1^2 - (y - h)^2. A radius that reaches 0 (a cone's tip) ends the solid there: no end ball.
    const double r0 = std::max(p_r0, 0.0), r1 = std::max(p_r1, 0.0);
    const double end = slope < 0.0 ? std::min(p_h, -r0 / slope) : p_h;
    Moments m = revolution_moments(r0 * r0, 0.0, -1.0, -r0, 0.0);
    m.add(revolution_moments(r0 * r0, 2.0 * r0 * slope, slope * slope, 0.0, end));
    m.add(revolution_moments(r1 * r1 - p_h * p_h, 2.0 * p_h, -1.0, p_h, p_h + r1));
    return m;
}

void MaterialMoments::add(uint32_t p_material, const Moments &p_moments, const double p_offset[3]) {
    for (std::pair<uint32_t, Moments> &part : parts) {
        if (part.first == p_material) {
            part.second.add(p_moments, p_offset);
            return;
        }
    }
    parts.push_back({ p_material, Moments() });
    parts.back().second.add(p_moments, p_offset);
}

const Moments *MaterialMoments::find(uint32_t p_material) const {
    for (const std::pair<uint32_t, Moments> &part : parts) {
        if (part.first == p_material) {
            return &part.second;
        }
    }
    return nullptr;
}

} // namespace Archetype
