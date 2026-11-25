#include "weather_moist.hpp"

#include "parallel_for.hpp"

#include "ov/math.hpp"
#include "weather_cloud_fraction.hpp"

#include <cmath>

namespace {

double dim(double p_a, double p_b) {
    return p_a > p_b ? p_a - p_b : 0.0;
}

} // namespace

WeatherMoist::Sounding WeatherMoist::sounding(const WeatherFluid &p_fluid, const std::vector<double> &p_theta, const std::vector<double> &p_qv) {
    Sounding s;
    s.theta = p_theta;
    s.qv = p_qv;
    const int levels = p_fluid.levels();
    s.theta.resize(size_t(levels), p_theta.empty() ? 300.0 : p_theta.back());
    s.qv.resize(size_t(levels), p_qv.empty() ? 0.0 : p_qv.back());
    // Hydrostatic: d exner / dz = -g / (c_p theta), from 1 at the ground (P0).
    double exner = 1.0;
    double z = 0.0;
    for (int k = 0; k < levels; k++) {
        const double zk = p_fluid.level_m(k);
        const double theta_mid = k == 0 ? s.theta[0] : 0.5 * (s.theta[size_t(k)] + s.theta[size_t(k - 1)]);
        exner -= GRAVITY / (CP * theta_mid) * (zk - z);
        z = zk;
        const double p_pa = P0_MB * 100.0 * std::pow(exner, 1.0 / KAPPA_EXNER);
        s.exner.push_back(exner);
        s.rho.push_back(p_pa / (R_D * s.theta[size_t(k)] * exner));
        s.z.push_back(zk);
    }
    return s;
}

WeatherMoist::WeatherMoist(const WeatherFluid &p_fluid, const Sounding &p_sounding, const Params &p_params) :
        sound(p_sounding), params(p_params) {
    if (p_fluid.scalars() < SCALARS) {
        OV_ERR_PRINT(ov::format("WeatherMoist: the fluid carries %d scalars, the physics needs %d.", p_fluid.scalars(), int(SCALARS)));
    }
    if (int(sound.exner.size()) != p_fluid.levels()) {
        OV_ERR_PRINT("WeatherMoist: the sounding is not the fluid's (WeatherMoist::sounding).");
        sound = sounding(p_fluid, sound.theta, sound.qv);
    }
    rain_rate.assign(p_fluid.grid().cells(), 0.0f);
    spread_s.assign(p_fluid.cell_count(), 0.0f);
    cloud_share.assign(p_fluid.cell_count(), 0.0f);
    surface_flux.assign(p_fluid.grid().cells(), Surface());
}

double WeatherMoist::saturation(double p_t, double p_p_mb) {
    return 3.8 / p_p_mb * std::exp(17.27 * (p_t - 273.0) / (p_t - 36.0));
}

double WeatherMoist::neutral_exchange(double p_z_m, double p_z0_m) {
    const double c = VON_KARMAN / std::log(p_z_m / p_z0_m);
    return c * c;
}

// kessler.f90 (DCMIP2016, Ullrich after Klemp), step for step; levels from the ground up.
void WeatherMoist::saturation_terms(double p_theta_l, double p_exner, double p_pc, double &r_a, double &r_b, double &r_qsl) {
    const double f2x = 17.27;
    const double f5 = 237.3 * f2x * LATENT / CP;
    const double tl = p_exner * p_theta_l;
    r_qsl = p_pc * std::exp(f2x * (tl - 273.0) / (tl - 36.0));
    const double gamma = r_qsl * f5 / ((tl - 36.0) * (tl - 36.0)); // L / c_p dq_s/dT
    r_a = 1.0 / (1.0 + gamma);
    r_b = r_a * p_exner * gamma * CP / LATENT; // a (T / theta) dq_s/dT
}

double WeatherMoist::condense(double p_theta, double p_qv, double p_qc, double p_exner, double p_pc, double p_sigma, double &r_share, double &r_excess) {
    const double theta_l = p_theta - LATENT / (CP * p_exner) * p_qc;
    const double qw = p_qv + p_qc;
    double a, b, qsl;
    saturation_terms(theta_l, p_exner, p_pc, a, b, qsl);
    r_excess = a * (qw - qsl);
    double ql;
    if (p_sigma > 1e-9) {
        const double q1 = r_excess / p_sigma;
        r_share = WeatherCloud::fraction(q1);
        ql = ov::min(p_sigma * WeatherCloud::water(q1), qw);
    } else {
        r_share = r_excess > 0.0 ? 1.0 : 0.0;
        ql = ov::max(r_excess, 0.0);
    }
    return ql - p_qc;
}

double WeatherMoist::kessler(double *r_theta, double *r_qv, double *r_qc, double *r_qr, const double *p_rho, const double *p_exner,
        const double *p_z, int p_levels, double p_dt, const double *p_sigma, double *r_share) {
    constexpr int MAX = WeatherFluid::MAX_LEVELS;
    const double f2x = 17.27;
    const double psl = P0_MB;
    const double rhoqr = 1000.0;
    double r[MAX], rhalf[MAX], velqr[MAX], sed[MAX], pc[MAX], share[MAX];
    for (int k = 0; k < p_levels; k++) {
        r[k] = 0.001 * p_rho[k];
        rhalf[k] = std::sqrt(p_rho[0] / p_rho[k]);
        pc[k] = 3.8 / (std::pow(p_exner[k], 1.0 / KAPPA_EXNER) * psl);
        velqr[k] = 36.34 * std::pow(r_qr[k] * r[k], 0.1364) * rhalf[k];
        double excess;
        condense(r_theta[k], r_qv[k], r_qc[k], p_exner[k], pc[k], p_sigma ? p_sigma[k] : 0.0, share[k], excess);
    }
    double dt_max = p_dt;
    for (int k = 0; k + 1 < p_levels; k++) {
        if (velqr[k] != 0.0) {
            dt_max = ov::min(dt_max, 0.8 * (p_z[k + 1] - p_z[k]) / velqr[k]);
        }
    }
    const int split = int(std::ceil(p_dt / dt_max));
    const double dt0 = p_dt / split;
    double precl = 0.0;
    for (int nt = 0; nt < split; nt++) {
        precl += p_rho[0] * r_qr[0] * velqr[0] / rhoqr;
        for (int k = 0; k + 1 < p_levels; k++) {
            sed[k] = dt0 * (r[k + 1] * r_qr[k + 1] * velqr[k + 1] - r[k] * r_qr[k] * velqr[k]) / (r[k] * (p_z[k + 1] - p_z[k]));
        }
        const int top = p_levels - 1;
        sed[top] = -dt0 * r_qr[top] * velqr[top] / (0.5 * (p_z[top] - p_z[top - 1]));
        for (int k = 0; k < p_levels; k++) {
            // Autoconversion (KW eq. 2.13a, from the cloudy part's water: N k1 (q_c / N - a)) and accretion (2.13b).
            const double qrprod = r_qc[k] - (r_qc[k] - dt0 * ov::max(0.001 * (r_qc[k] - 0.001 * share[k]), 0.0)) / (1.0 + dt0 * 2.2 * std::pow(r_qr[k], 0.875));
            r_qc[k] = ov::max(r_qc[k] - qrprod, 0.0);
            r_qr[k] = ov::max(r_qr[k] + qrprod + sed[k], 0.0);
            // The saturation (KW eq. 2.11) and the adjustment towards the partial cloud (eq. 3.10 when uniform).
            const double t = p_exner[k] * r_theta[k];
            const double qvs = pc[k] * std::exp(f2x * (t - 273.0) / (t - 36.0));
            double excess;
            const double cond = condense(r_theta[k], r_qv[k], r_qc[k], p_exner[k], pc[k], p_sigma ? p_sigma[k] : 0.0, share[k], excess);
            // The rain's evaporation (KW eq. 2.14a,b).
            const double rq = r[k] * r_qr[k];
            const double ern = ov::min(dt0 * (((1.6 + 124.9 * std::pow(rq, 0.2046)) * std::pow(rq, 0.525)) / (2550000.0 * pc[k] / (3.8 * qvs) + 540000.0)) *
                            (dim(qvs, r_qv[k]) / (r[k] * qvs)),
                    ov::min(ov::max(-excess, 0.0), r_qr[k]));
            r_theta[k] += LATENT / (CP * p_exner[k]) * (cond - ern);
            r_qv[k] = ov::max(r_qv[k] - cond + ern, 0.0);
            r_qc[k] += cond;
            r_qr[k] -= ern;
        }
        if (nt + 1 == split && r_share) {
            for (int k = 0; k < p_levels; k++) {
                r_share[k] = share[k];
            }
        }
        if (nt + 1 != split) {
            for (int k = 0; k < p_levels; k++) {
                velqr[k] = 36.34 * std::pow(r_qr[k] * r[k], 0.1364) * rhalf[k];
            }
        }
    }
    return precl / split;
}

void WeatherMoist::eddy_diffusivity(const double *p_theta, const double *p_qv, const double *p_qc, const double (*p_wind)[3], double p_dz,
        double p_delta_m, int p_levels, double *r_k) {
    const double l2 = (C_S * p_delta_m) * (C_S * p_delta_m);
    for (int k = 0; k + 1 < p_levels; k++) {
        const double tv0 = p_theta[k] * (1.0 + 0.61 * p_qv[k] - p_qc[k]);
        const double tv1 = p_theta[k + 1] * (1.0 + 0.61 * p_qv[k + 1] - p_qc[k + 1]);
        const double n2 = GRAVITY * (tv1 - tv0) / (0.5 * (tv0 + tv1) * p_dz);
        double s2 = 0.0;
        for (int a = 0; a < 3; a++) {
            const double d = (p_wind[k + 1][a] - p_wind[k][a]) / p_dz;
            s2 += d * d;
        }
        r_k[k] = l2 * std::sqrt(ov::max(s2 - n2 / PRANDTL, 0.0));
    }
}

void WeatherMoist::diffuse(double *r_x, const double *p_k, double p_dz, double p_dt, int p_levels) {
    constexpr int MAX = WeatherFluid::MAX_LEVELS;
    // Thomas: -a_k x_(k-1) + (1 + a_k + b_k) x_k - b_k x_(k+1) = x_k, a and b the interfaces below and above.
    double c[MAX], d[MAX];
    const double s = p_dt / (p_dz * p_dz);
    for (int k = 0; k < p_levels; k++) {
        const double below = k > 0 ? p_k[k - 1] * s : 0.0;
        const double above = k + 1 < p_levels ? p_k[k] * s : 0.0;
        const double diag = 1.0 + below + above - (k > 0 ? below * c[k - 1] : 0.0);
        c[k] = above / diag;
        d[k] = (r_x[k] + (k > 0 ? below * d[k - 1] : 0.0)) / diag;
    }
    for (int k = p_levels - 1; k >= 0; k--) {
        r_x[k] = d[k] + (k + 1 < p_levels ? c[k] * r_x[k + 1] : 0.0);
    }
}

void WeatherMoist::longwave(const double *p_qc, const double *p_rho, const double *p_exner, double p_dz, int p_levels, double *r_heating) {
    constexpr int MAX = WeatherFluid::MAX_LEVELS;
    // The optical depth of each level, then the flux at each interface (0 the ground, levels the top).
    double q[MAX];
    double total = 0.0;
    for (int k = 0; k < p_levels; k++) {
        q[k] = LW_KAPPA * p_rho[k] * p_qc[k] * p_dz;
        total += q[k];
    }
    double flux[MAX + 1];
    double below = 0.0;
    for (int i = 0; i <= p_levels; i++) {
        flux[i] = LW_F0 * std::exp(-(total - below)) + LW_F1 * std::exp(-below);
        if (i < p_levels) {
            below += q[i];
        }
    }
    for (int k = 0; k < p_levels; k++) {
        r_heating[k] = -(flux[k + 1] - flux[k]) / (p_rho[k] * CP * p_exner[k] * p_dz);
    }
}

void WeatherMoist::spread(size_t p_cell, const double (*p_col)[WeatherFluid::MAX_LEVELS], const double *p_eddy, const Convective &p_convective,
        const WeatherFluid &p_fluid, double *r_sigma) const {
    constexpr int MAX = WeatherFluid::MAX_LEVELS;
    const int levels = p_fluid.levels();
    const double dz = p_fluid.dz_m();
    double theta_l[MAX], qw[MAX], a[MAX], b[MAX];
    for (int k = 0; k < levels; k++) {
        const double exner = sound.exner[size_t(k)];
        theta_l[k] = p_col[THETA][k] - LATENT / (CP * exner) * p_col[QC][k];
        qw[k] = p_col[QV][k] + p_col[QC][k];
        double qsl;
        saturation_terms(theta_l[k], exner, 3.8 / (std::pow(exner, 1.0 / KAPPA_EXNER) * P0_MB), a[k], b[k], qsl);
    }
    // tau: the boundary layer's turnover z_i / w* (w* from the surface's buoyancy flux; at least u* and 0.1 m/s).
    const double top = p_convective.top_m ? double(p_convective.top_m[p_cell]) : 0.0;
    const double zi = ov::max(top, 2.0 * dz);
    const Surface &sf = surface_flux[p_cell];
    const double buoyancy_flux = GRAVITY / p_col[THETA][0] * (double(sf.theta_flux) + 0.61 * p_col[THETA][0] * double(sf.qv_flux));
    const double wstar = std::cbrt(ov::max(buoyancy_flux, 0.0) * zi);
    const double tau = zi / ov::max(ov::max(wstar, double(sf.ustar)), 0.1);
    for (int k = 0; k < levels; k++) {
        const int k0 = ov::max(k - 1, 0), k1 = ov::min(k + 1, levels - 1);
        const double span = double(k1 - k0) * dz;
        const double ds = (a[k] * (qw[k1] - qw[k0]) - b[k] * (theta_l[k1] - theta_l[k0])) / span;
        const double eddy = 0.5 * ((k > 0 ? p_eddy[k - 1] : 0.0) + (k + 1 < levels ? p_eddy[k] : 0.0));
        double production = eddy * ds * ds; // -w's' ds/dz, w's' = -K ds/dz
        if (p_convective.qt_flux) {
            const size_t at = p_fluid.cell_index(p_cell, k);
            const double flux = a[k] * double(p_convective.qt_flux[at]) - b[k] * double(p_convective.theta_l_flux[at]);
            production += std::fabs(flux * ds);
        }
        r_sigma[k] = std::sqrt(2.0 * tau * production);
    }
}

void WeatherMoist::apply(WeatherFluid &p_fluid, const std::vector<Ground> &p_ground, double p_dt, float *r_buoyancy, const Convective &p_convective) {
    const WeatherGrid &g = p_fluid.grid();
    const int levels = p_fluid.levels();
    const int st = p_fluid.stride();
    const double dz = p_fluid.dz_m();
    const double clear = CLEAR_COOLING / 86400.0;
    const double sink = params.subsidence_per_s;
    const double delta = std::cbrt(g.cell_m() * g.cell_m() * dz);
    std::vector<float> &state = p_fluid.state();
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherMoist::apply", [&](uint32_t p_row) {
        constexpr int MAX = WeatherFluid::MAX_LEVELS;
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            float *here = &state[g.index(face, i, j) * size_t(st)];
            double col[SCALARS][MAX];
            for (int k = 0; k < levels; k++) {
                for (int s = 0; s < SCALARS; s++) {
                    col[s][k] = here[p_fluid.scalar_offset(k, s)];
                }
            }
            // Subsidence: w = -D z, upwind (from above; the top level's air from its own level).
            for (int k = 0; k + 1 < levels; k++) {
                const double a = sink * sound.z[size_t(k)] * p_dt / dz;
                for (int s = 0; s < SCALARS; s++) {
                    col[s][k] += a * (col[s][k + 1] - col[s][k]);
                }
            }
            // The ground: drag, heat and evaporation into the lowest level (implicit, so any dt is stable).
            const Ground &gr = in < p_ground.size() ? p_ground[in] : Ground();
            double up[3];
            p_fluid.column_up(in, up);
            float *u = here + p_fluid.wind_offset(0);
            const double w = u[0] * up[0] + u[1] * up[1] + u[2] * up[2];
            double h[3] = { u[0] - w * up[0], u[1] - w * up[1], u[2] - w * up[2] };
            const double speed = ov::max(std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]), GUSTINESS);
            const double c = neutral_exchange(0.5 * dz, ov::max(double(gr.z0_m), 1e-6)) * params.exchange;
            const double a = c * speed * p_dt / dz;
            for (int k = 0; k < 3; k++) {
                u[k] = float(h[k] / (1.0 + a) + w * up[k]);
            }
            const double d_theta = (double(gr.temp_k) - col[THETA][0]) * a / (1.0 + a); // the ground's exner is 1 (P0)
            col[THETA][0] += d_theta;
            const double qs = saturation(gr.temp_k, P0_MB);
            const double ae = qs > col[QV][0] ? a * double(gr.wetness) : a;
            const double d_qv = (qs - col[QV][0]) * ae / (1.0 + ae);
            col[QV][0] += d_qv;
            surface_flux[in] = Surface{ float(d_theta * dz / p_dt), float(d_qv * dz / p_dt), float(std::sqrt(c) * speed) };
            // The subgrid turbulence: theta, vapour, cloud and the horizontal wind.
            double wind[MAX][3], vert[MAX], eddy[MAX];
            for (int k = 0; k < levels; k++) {
                const float *uk = here + p_fluid.wind_offset(k);
                vert[k] = uk[0] * up[0] + uk[1] * up[1] + uk[2] * up[2];
                for (int a = 0; a < 3; a++) {
                    wind[k][a] = uk[a] - vert[k] * up[a];
                }
            }
            eddy_diffusivity(col[THETA], col[QV], col[QC], wind, dz, delta, levels, eddy);
            for (const int s : { int(THETA), int(QV), int(QC) }) {
                diffuse(col[s], eddy, dz, p_dt, levels);
            }
            for (int a = 0; a < 3; a++) {
                double x[MAX];
                for (int k = 0; k < levels; k++) {
                    x[k] = wind[k][a];
                }
                diffuse(x, eddy, dz, p_dt, levels);
                for (int k = 0; k < levels; k++) {
                    here[p_fluid.wind_offset(k) + a] = float(x[k] + vert[k] * up[a]);
                }
            }
            // Radiation: the cloud's longwave and the clear air's cooling.
            double heating[MAX];
            longwave(col[QC], sound.rho.data(), sound.exner.data(), dz, levels, heating);
            for (int k = 0; k < levels; k++) {
                col[THETA][k] += (heating[k] - clear) * p_dt;
            }
            double sigma[MAX], share[MAX];
            spread(in, col, eddy, p_convective, p_fluid, sigma);
            rain_rate[in] = float(kessler(col[THETA], col[QV], col[QC], col[QR], sound.rho.data(), sound.exner.data(), sound.z.data(), levels, p_dt, sigma, share));
            for (int k = 0; k < levels; k++) {
                for (int s = 0; s < SCALARS; s++) {
                    here[p_fluid.scalar_offset(k, s)] = float(col[s][k]);
                }
                // The reference in the state's precision: air at it is exactly neutral.
                const double theta_ref = double(float(sound.theta[size_t(k)]));
                const double qv_ref = double(float(sound.qv[size_t(k)]));
                const double th = here[p_fluid.scalar_offset(k, THETA)];
                const double qv = here[p_fluid.scalar_offset(k, QV)];
                const double qc = here[p_fluid.scalar_offset(k, QC)];
                const double qr = here[p_fluid.scalar_offset(k, QR)];
                r_buoyancy[p_fluid.cell_index(in, k)] = float(GRAVITY * ((th - theta_ref) / theta_ref + 0.61 * (qv - qv_ref) - qc - qr));
                spread_s[p_fluid.cell_index(in, k)] = float(sigma[k]);
                cloud_share[p_fluid.cell_index(in, k)] = float(share[k]);
            }
        } }, ov::Priority::LOW);
}
