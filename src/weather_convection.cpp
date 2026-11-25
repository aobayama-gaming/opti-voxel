#include "weather_convection.hpp"

#include "parallel_for.hpp"

#include "ov/math.hpp"

#include <cmath>

using M = WeatherMoist;

WeatherConvection::WeatherConvection(const WeatherFluid &p_fluid, const WeatherMoist::Sounding &p_ref, double p_top_m) : ref(p_ref) {
    top.assign(p_fluid.grid().cells(), float(p_top_m));
    cloud_water.assign(p_fluid.grid().cells() * size_t(p_fluid.levels()), 0.0f);
}

double WeatherConvection::sigma_w(double p_z, double p_top_m, double p_wstar, double p_ustar) {
    if (!(p_wstar > 0.0) || p_z >= p_top_m) {
        return 0.0;
    }
    const double r = p_ustar / p_wstar;
    return 1.3 * p_wstar * std::cbrt(r * r * r + 0.6 * p_z / p_top_m) * std::sqrt(1.0 - p_z / p_top_m);
}

void WeatherConvection::adjust(double p_theta_l, double p_qt, double p_exner, double &r_theta, double &r_qc) {
    const double p_mb = M::P0_MB * std::pow(p_exner, 1.0 / M::KAPPA_EXNER);
    const double tl = p_theta_l * p_exner; // the liquid water temperature
    if (p_qt <= M::saturation(tl, p_mb)) {
        r_theta = p_theta_l;
        r_qc = 0.0;
        return;
    }
    // Newton on T - T_l - L / c_p (q_t - q_s(T)) = 0, q_s Tetens': d q_s / dT = q_s 17.27 (273 - 36) / (T - 36)^2.
    double t = tl;
    for (int i = 0; i < 5; i++) {
        const double qs = M::saturation(t, p_mb);
        const double f = t - tl - M::LATENT / M::CP * (p_qt - qs);
        const double df = 1.0 + M::LATENT / M::CP * qs * 17.27 * 237.0 / ((t - 36.0) * (t - 36.0));
        t -= f / df;
    }
    r_qc = ov::max(p_qt - M::saturation(t, p_mb), 0.0);
    r_theta = t / p_exner;
}

void WeatherConvection::plume(const double *p_theta, const double *p_qv, const double *p_qc, const double *p_exner, double p_dz, int p_levels,
        const WeatherMoist::Surface &p_surface, double p_top_m, Plume &r_plume) {
    constexpr int MAX = WeatherFluid::MAX_LEVELS;
    r_plume.levels = 0;
    r_plume.top_m = 0.0;
    double tl[MAX], qt[MAX], tv[MAX];
    for (int k = 0; k < p_levels; k++) {
        tl[k] = p_theta[k] - M::LATENT / (M::CP * p_exner[k]) * p_qc[k];
        qt[k] = p_qv[k] + p_qc[k];
        tv[k] = p_theta[k] * (1.0 + 0.61 * p_qv[k] - p_qc[k]);
    }
    const double tf = p_surface.theta_flux, qf = p_surface.qv_flux;
    const double buoyancy_flux = tf * (1.0 + 0.61 * p_qv[0]) + 0.61 * p_theta[0] * qf;
    if (!(buoyancy_flux > 0.0)) {
        return; // the ground does not heat the air: no thermals
    }
    const double zs = ov::max(p_top_m, 2.0 * p_dz);
    const double wstar = std::cbrt(M::GRAVITY / tv[0] * buoyancy_flux * zs);
    const double z1 = 0.5 * p_dz;
    const double s1 = sigma_w(z1, zs, wstar, p_surface.ustar);
    if (!(s1 > 1e-6)) {
        return;
    }
    double tlu = tl[0] + ALPHA * tf / s1;
    double qtu = qt[0] + ALPHA * qf / s1;
    double w2 = s1 * s1 / MU;
    auto record = [&](int p_k, double p_z) {
        double theta;
        adjust(tlu, qtu, p_exner[p_k], theta, r_plume.qc[p_k]);
        r_plume.theta_l[p_k] = tlu;
        r_plume.qt[p_k] = qtu;
        r_plume.w[p_k] = std::sqrt(ov::max(w2, 0.0));
        r_plume.mass[p_k] = C_M * sigma_w(p_z, zs, wstar, p_surface.ustar);
        r_plume.levels = p_k + 1;
    };
    record(0, z1);
    const double h = p_dz / SUBSTEPS;
    double z = z1;
    for (int k = 0; k + 1 < p_levels; k++) {
        for (int s = 0; s < SUBSTEPS; s++) {
            const double zm = z + 0.5 * h;
            const double f = (zm - (k + 0.5) * p_dz) / p_dz; // between levels k and k + 1
            const double tle = tl[k] + f * (tl[k + 1] - tl[k]);
            const double qte = qt[k] + f * (qt[k + 1] - qt[k]);
            const double tve = tv[k] + f * (tv[k + 1] - tv[k]);
            const double exner = p_exner[k] + f * (p_exner[k + 1] - p_exner[k]);
            const double eps = C_EPS * (1.0 / ov::max(zm, p_dz) + 1.0 / ov::max(zs - zm, p_dz));
            double theta, qc;
            adjust(tlu, qtu, exner, theta, qc);
            const double tvu = theta * (1.0 + 0.61 * (qtu - qc) - qc);
            const double b = M::GRAVITY * (tvu - tve) / tve;
            tlu -= eps * (tlu - tle) * h;
            qtu -= eps * (qtu - qte) * h;
            const double w2_next = w2 + h * 2.0 / (1.0 - 2.0 * MU) * (b - B_W * eps * w2);
            if (w2_next <= 0.0) {
                r_plume.top_m = z + h * w2 / (w2 - w2_next);
                return;
            }
            w2 = w2_next;
            z += h;
        }
        record(k + 1, (k + 1.5) * p_dz);
    }
    r_plume.top_m = z;
}

void WeatherConvection::apply(WeatherFluid &p_fluid, const std::vector<WeatherMoist::Surface> &p_surface, double p_dt) {
    const WeatherGrid &g = p_fluid.grid();
    const int levels = p_fluid.levels();
    const int st = p_fluid.stride();
    const double dz = p_fluid.dz_m();
    std::vector<float> &state = p_fluid.state();
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherConvection::apply", [&](uint32_t p_row) {
        constexpr int MAX = WeatherFluid::MAX_LEVELS;
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            float *here = &state[g.index(face, i, j) * size_t(st)];
            double theta[MAX], qv[MAX], qc[MAX];
            for (int k = 0; k < levels; k++) {
                theta[k] = here[p_fluid.scalar_offset(k, M::THETA)];
                qv[k] = here[p_fluid.scalar_offset(k, M::QV)];
                qc[k] = here[p_fluid.scalar_offset(k, M::QC)];
            }
            Plume u;
            plume(theta, qv, qc, ref.exner.data(), dz, levels, in < p_surface.size() ? p_surface[in] : M::Surface(), top[in], u);
            top[in] = float(u.top_m);
            // The fluxes M (phi_u - phi) through the interface over each level the updraft reaches; none at the
            // ground (the surface's own are the moist physics') nor over its top.
            double f_tl[MAX + 1] = {}, f_qt[MAX + 1] = {};
            for (int k = 0; k < u.levels && k + 1 < levels; k++) {
                const double tl = theta[k] - M::LATENT / (M::CP * ref.exner[size_t(k)]) * qc[k];
                f_tl[k + 1] = u.mass[k] * (u.theta_l[k] - tl);
                f_qt[k + 1] = u.mass[k] * (u.qt[k] - (qv[k] + qc[k]));
            }
            for (int k = 0; k < levels; k++) {
                // theta_l and q_t change, the cloud water kept: theta by theta_l's change, the vapour by q_t's.
                here[p_fluid.scalar_offset(k, M::THETA)] = float(theta[k] - p_dt * (f_tl[k + 1] - f_tl[k]) / dz);
                here[p_fluid.scalar_offset(k, M::QV)] = float(ov::max(qv[k] - p_dt * (f_qt[k + 1] - f_qt[k]) / dz, 0.0));
                const double area = k < u.levels ? ov::min(u.mass[k] / ov::max(u.w[k], 1e-3), MAX_AREA) : 0.0;
                cloud_water[p_fluid.cell_index(in, k)] = float(area * (k < u.levels ? u.qc[k] : 0.0));
            }
        } });
}
