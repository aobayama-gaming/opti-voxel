#pragma once

#include "weather_moist.hpp"

#include <vector>

// The shallow convection (docs/weather.md "Convection"): the mass-flux half of EDMF (Siebesma, Soares & Teixeira 2007,
// "A combined eddy-diffusivity mass-flux approach for the convective boundary layer", JAS 64: equation numbers below
// are theirs), its eddy diffusivity the moist physics' Smagorinsky turbulence. The cells (3-4 levels in the mixed
// layer) cannot resolve the thermals: one updraft per column stands for them, carrying the ground's heat and vapour up
// to its cloud. Engine-free.
// - The updraft starts at the lowest level with the mean plus an excess of the surface flux (17): phi_u = phi + ALPHA
//   w'phi'_s / sigma_w(z_1), its speed w_u = sigma_w / sqrt(MU) (MU = (sigma_w / w_u)^2, section 3); only while the
//   ground heats the air (a buoyancy flux over 0: by day over land, day and night over a sea warmer than its air).
// - It rises entraining the air round it, d phi_u / dz = -eps (phi_u - phi) (10), eps = C_EPS (1 / z + 1 / (z* - z))
//   (16; both distances at least a level, z* the column's last top), its speed from
//   1/2 (1 - 2 MU) dw_u^2 / dz = -B_W eps w_u^2 + B (15), B its virtual buoyancy over the column's. It carries the
//   moist conserved theta_l and q_t, saturation-adjusted on the way (the cloudy extension of Soares et al. 2004): its
//   cloud base is where it saturates, its top z* where w_u reaches 0.
// - Its mass flux M = C_M sigma_w (21), sigma_w = 1.3 w* ((u*/w*)^3 + 0.6 z/z*)^(1/3) (1 - z/z*)^(1/2) (7), w* the
//   convective velocity over z*; the column's theta_l and q_t change by -d/dz (M (phi_u - phi)) (22), explicit
//   (M dt / dz is under 0.1), the column's totals kept; the grid's saturation adjustment (Kessler) condenses what the
//   updraft brought.
// - The updraft's cloud: its water qc_u over its area M / w_u (MAX_AREA at most) is the column's convective cloud,
//   which the renderer and the ground's shade add to the grid's (a cumulus narrower than a cell). It does not rain.
class WeatherConvection {
public:
    static constexpr double C_EPS = 0.4; // (16)
    static constexpr double MU = 0.15; // (sigma_w / w_u)^2, section 3
    static constexpr double B_W = 0.5; // (14)
    static constexpr double ALPHA = 1.0; // (17)
    static constexpr double C_M = 0.3; // (21)
    static constexpr double MAX_AREA = 0.3;
    static constexpr int SUBSTEPS = 8; // the updraft's steps per level

    // One updraft over a column, per level k (its centre (k + 1/2) dz).
    struct Plume {
        double theta_l[WeatherFluid::MAX_LEVELS] = {};
        double qt[WeatherFluid::MAX_LEVELS] = {};
        double qc[WeatherFluid::MAX_LEVELS] = {};
        double w[WeatherFluid::MAX_LEVELS] = {};
        double mass[WeatherFluid::MAX_LEVELS] = {}; // M (m/s)
        int levels = 0; // the levels it reaches (0: none)
        double top_m = 0.0; // z*, where w_u reaches 0
    };

    // Over p_fluid's columns, each one's first top p_top_m; p_ref the sounding (its exner per level).
    WeatherConvection(const WeatherFluid &p_fluid, const WeatherMoist::Sounding &p_ref, double p_top_m);

    // The updrafts over p_dt from the ground's last fluxes (WeatherMoist::surface, per interior cell).
    void apply(WeatherFluid &p_fluid, const std::vector<WeatherMoist::Surface> &p_surface, double p_dt);
    // The columns' convective cloud water (kg/kg) per interior cell and level (WeatherFluid::cell_index).
    const std::vector<float> &cloud() const { return cloud_water; }
    // The columns' updraft tops (m; 0 where none rose).
    const std::vector<float> &tops() const { return top; }
    // Its fluxes and tops for the moist physics' spread (WeatherMoist::Convective).
    WeatherMoist::Convective fluxes() const { return { flux_theta_l.data(), flux_qt.data(), top.data() }; }

    // A column's updraft: theta, qv, qc per level, the sounding's exner, the surface fluxes, the last top.
    static void plume(const double *p_theta, const double *p_qv, const double *p_qc, const double *p_exner, double p_dz, int p_levels,
            const WeatherMoist::Surface &p_surface, double p_top_m, Plume &r_plume);
    // theta and qc from the conserved theta_l and q_t at exner p_exner (the saturation adjustment, Tetens').
    static void adjust(double p_theta_l, double p_qt, double p_exner, double &r_theta, double &r_qc);
    // sigma_w (7) at p_z under a top p_top_m, for w* and u*.
    static double sigma_w(double p_z, double p_top_m, double p_wstar, double p_ustar);

private:
    WeatherMoist::Sounding ref;
    std::vector<float> top;
    std::vector<float> cloud_water;
    std::vector<float> flux_theta_l, flux_qt; // M (phi_u - phi) at the levels' centres, per interior cell and level
};
