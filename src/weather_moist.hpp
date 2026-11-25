#pragma once

#include "weather_fluid.hpp"

#include <vector>

// The weather's moist physics (docs/weather.md "Moist physics"): what turns WeatherFluid's air into weather, per column
// between its advect() and its force(). Engine-free; the reference schemes of the cloud-resolving models:
// - Warm microphysics: Kessler (1969) as in Klemp & Wilhelmson (1978) and the DCMIP2016 reference code (kessler.f90,
//   Ullrich; ported step for step: kessler()): vapour, cloud water and rain; the saturation adjustment (KW eq. 3.10,
//   with its Clausius-Clapeyron denominator), autoconversion past 1 g/kg, accretion, rain falling at its terminal speed
//   (sub-stepped under its CFL) and evaporating under the cloud; latent heat into the potential temperature. The
//   saturation adjustment is partial (weather_cloud_fraction.hpp): a cell's cloud water is the statistical scheme's
//   mean for its saturation excess and spread sigma_s, its cloudy share N; autoconversion from the cloudy part's water
//   (q_c / N). With no spread it is Kessler's all-or-nothing adjustment.
// - The spread sigma_s per cell: the variance budget of s in balance (Stull 1988, 4.3: production -2 w's' ds/dz,
//   dissipation sigma^2 / tau), sigma^2 = 2 tau |w's' ds/dz|, the flux w's' the subgrid turbulence's (-K ds/dz) and the
//   shallow convection's (its updraft's M (s_u - s); the convection sets the humidity's variance, as Lenderink &
//   Siebesma 2000 couple them), tau the boundary layer's turnover z_i / w* (Deardorff; z_i the updraft's top).
// - Bulk surface fluxes over each column's ground (Monin-Obukhov neutral): momentum C_D U u, sensible heat C_H U
//   (theta_s - theta), evaporation beta C_E U (q_s(T_s) - q_v) (dew at full rate), C_D = (k / ln(z_1 / z_0))^2 at the
//   lowest level's centre, C_H = C_E = C_D; U at least GUSTINESS (free convection at no wind).
// - Radiation: the clear air cools CLEAR_COOLING K/day; the cloud's longwave flux is DYCOMS-II RF01's (Stevens et al.
//   2005): F(z) = F0 exp(-Q(z, top)) + F1 exp(-Q(0, z)), Q = KAPPA int rho q_c dz, so a cloud top cools hard and drives
//   the stratocumulus' turnover.
// - The subgrid turbulence in the vertical (Smagorinsky-Lilly, Lilly 1962): at each interface of a column K = (C_S D)^2
//   sqrt(max(S^2 - N^2 / PRANDTL, 0)), D the cell's (width^2 dz)^(1/3), S the wind's vertical shear, N^2 the
//   buoyancy frequency of the virtual potential temperature: mixing where the air shears or is unstable, none in the
//   inversion; implicit on the scalars (rain falls on its own) and the horizontal wind. The cells resolve the thermals,
//   not the surface layer under them: without it the lowest level takes the ground's own state (a sea's saturated
//   air, fog over a warm sea).
// - Large-scale subsidence w = -D z on the scalars (upwind): it holds the inversion against the boundary layer's growth.
// - The twist, Params::exchange: the surface fluxes' coefficients (all three, their ratio kept) times exchange: a
//   storm's intensity, set by the ratio, is kept, its spin-up shortened. Only these: radiation and subsidence sped up
//   as well (tried, one time compression of the slow processes) outrun the boundary layer's turbulence, which runs at
//   its own speed: the humid layer cools to fog, a cloud top overturns its fog into a deck.
// - The buoyancy the fluid takes: b = g ((theta - theta_ref) / theta_ref + 0.61 (q_v - q_v,ref) - q_c - q_r) (vapour
//   lightens the air, the water weighs it).
// The reference column (Sounding): the hydrostatic Exner function and density of the sounding the world gives, from
// P0 at the ground.
class WeatherMoist {
public:
    enum Scalar {
        THETA, // potential temperature (K)
        QV, // vapour (kg/kg)
        QC, // cloud water
        QR, // rain
        SCALARS,
    };

    static constexpr double GRAVITY = WeatherFluid::GRAVITY;
    static constexpr double LATENT = 2.5e6; // J/kg
    static constexpr double CP = 1003.0; // J/(kg K), kessler.f90's
    static constexpr double KAPPA_EXNER = 0.2875; // R_d / c_p, kessler.f90's
    static constexpr double R_D = 287.0;
    static constexpr double P0_MB = 1000.0; // the ground's pressure
    static constexpr double VON_KARMAN = 0.4;
    static constexpr double GUSTINESS = 1.0; // m/s
    static constexpr double CLEAR_COOLING = 2.0; // K/day
    static constexpr double LW_F0 = 70.0; // W/m^2, DYCOMS-II RF01
    static constexpr double LW_F1 = 22.0;
    static constexpr double LW_KAPPA = 85.0; // m^2/kg
    static constexpr double C_S = 0.18; // Smagorinsky's (Lilly)
    static constexpr double PRANDTL = WeatherFluid::PRANDTL;

    // The sounding: the reference potential temperature and vapour per level (the buoyancy's reference; the world's
    // air at rest), and their hydrostatic Exner function, density and pressure.
    struct Sounding {
        std::vector<double> theta;
        std::vector<double> qv;
        std::vector<double> exner;
        std::vector<double> rho; // kg/m^3
        std::vector<double> z; // the levels' centres (m)
    };
    // A sounding of p_theta and p_qv per level of p_fluid.
    static Sounding sounding(const WeatherFluid &p_fluid, const std::vector<double> &p_theta, const std::vector<double> &p_qv);

    // A column's ground (the 2D forcing): its temperature, how wet it is (beta: 1 a sea, 0 a desert) and its
    // roughness length.
    struct Ground {
        float temp_k = 300.0f;
        float wetness = 1.0f;
        float z0_m = 1.5e-4f;
    };

    // A column's surface fluxes in the last apply(), kinematic: theta (K m/s), vapour (kg/kg m/s), and the friction
    // velocity (m/s). The shallow convection's closure (WeatherConvection).
    struct Surface {
        float theta_flux = 0.0f;
        float qv_flux = 0.0f;
        float ustar = 0.0f;
    };

    // The shallow convection's fluxes per interior cell and level (WeatherFluid::cell_index), kinematic, at the levels'
    // centres: theta_l (K m/s) and q_t (kg/kg m/s); its updraft's top per interior cell (m, 0 where none rose). {}: none.
    struct Convective {
        const float *theta_l_flux;
        const float *qt_flux;
        const float *top_m;
    };

    struct Params {
        double exchange = 15.0; // the surface fluxes' twist
        double subsidence_per_s = 1.5e-5; // D: w = -D z
    };

    WeatherMoist(const WeatherFluid &p_fluid, const Sounding &p_sounding, const Params &p_params);

    const Sounding &reference() const { return sound; }
    // Everything above on p_fluid's state for p_dt (p_ground per interior cell), then the buoyancy (per fluid cell).
    void apply(WeatherFluid &p_fluid, const std::vector<Ground> &p_ground, double p_dt, float *r_buoyancy, const Convective &p_convective = {});
    // Each cell's spread sigma_s (kg/kg) and cloudy share in the last apply(), per interior cell and level.
    const std::vector<float> &spread() const { return spread_s; }
    const std::vector<float> &share() const { return cloud_share; }
    // Each cell's Q1 (its mean saturation excess over its spread, within +-WeatherCloud::Q1_LIMIT; a uniform cell's the
    // limit of its sign), per interior cell and level: the render's.
    const std::vector<float> &q1() const { return cloud_q1; }
    // The rain reaching each column's ground in the last apply() (m of water per second), per interior cell.
    const std::vector<float> &rain() const { return rain_rate; }
    // The surface fluxes of the last apply(), per interior cell.
    const std::vector<Surface> &surface() const { return surface_flux; }

    // The saturation mixing ratio at temperature p_t (K) and pressure p_p_mb (Tetens as in KW eq. 2.11).
    static double saturation(double p_t, double p_p_mb);
    // The bulk exchange coefficient at height p_z_m over a roughness p_z0_m (neutral).
    static double neutral_exchange(double p_z_m, double p_z0_m);
    // The Kessler step on one column (kessler.f90): theta, qv, qc, qr per level, the column's rho, exner and z; the
    // precipitation rate (m of water per second) returned.
    // p_sigma (or null: none) the cells' spread per level; r_share (or null) their cloudy share.
    static double kessler(double *r_theta, double *r_qv, double *r_qc, double *r_qr, const double *p_rho, const double *p_exner,
            const double *p_z, int p_levels, double p_dt, const double *p_sigma = nullptr, double *r_share = nullptr, double *r_q1 = nullptr);
    // The condensation (kg/kg, negative: evaporation) that takes a cell's theta, qv, qc to the partial cloud of spread
    // p_sigma (0: uniform), p_pc Kessler's saturation factor at its pressure; r_share its cloudy share, r_excess its mean
    // saturation excess s.
    static double condense(double p_theta, double p_qv, double p_qc, double p_exner, double p_pc, double p_sigma, double &r_share, double &r_excess);
    // A level's saturation coefficients at the liquid water temperature: a and b of s = a q_w - b theta_l (Bechtold &
    // Siebesma 1998, A1), q_sl.
    static void saturation_terms(double p_theta_l, double p_exner, double p_pc, double &r_a, double &r_b, double &r_qsl);
    // The eddy diffusivity (m^2/s) at each interface k (between levels k and k + 1) of a column: theta, qv, qc per level
    // and its horizontal wind (3 per level), p_delta_m the Smagorinsky length.
    static void eddy_diffusivity(const double *p_theta, const double *p_qv, const double *p_qc, const double (*p_wind)[3], double p_dz,
            double p_delta_m, int p_levels, double *r_k);
    // r_x per level diffused for p_dt by r_k per interface (implicit, the ground and the top closed).
    static void diffuse(double *r_x, const double *p_k, double p_dz, double p_dt, int p_levels);
    // The heating rate (K/s) of the cloud's longwave flux per level of one column (qc per level).
    static void longwave(const double *p_qc, const double *p_rho, const double *p_exner, double p_dz, int p_levels, double *r_heating);

private:
    // The spread sigma_s per level of interior cell p_cell's column p_col (the scalars per level, after the turbulence of
    // diffusivity p_eddy per interface).
    void spread(size_t p_cell, const double (*p_col)[WeatherFluid::MAX_LEVELS], const double *p_eddy, const Convective &p_convective,
            const WeatherFluid &p_fluid, double *r_sigma) const;

    Sounding sound;
    Params params;
    std::vector<float> rain_rate;
    std::vector<float> spread_s, cloud_share, cloud_q1;
    std::vector<Surface> surface_flux;
};
