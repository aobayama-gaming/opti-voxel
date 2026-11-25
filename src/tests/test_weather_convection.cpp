#include "test_common.hpp"

#include "weather_convection.hpp"
#include "weather_forcing.hpp"

#include <cmath>
#include <vector>

// The shallow convection (WeatherConvection, docs/weather.md "Convection"): EDMF's updraft (Siebesma, Soares & Teixeira
// 2007) over a column.

namespace {

using C = WeatherConvection;
using M = WeatherMoist;

struct Column {
    std::shared_ptr<CubeSphereGrid> grid;
    std::unique_ptr<WeatherFluid> fluid;
    M::Sounding sounding;
};

// The trade-wind start (WeatherForcing's sounding) over a sea of p_ground_k: 15 levels to 600 m, the inversion at 300 m.
Column trade_column(float p_ground_k) {
    Column c;
    c.grid = std::make_shared<CubeSphereGrid>(8, 1000.0);
    WeatherFluid::Params fp;
    fp.levels = 15;
    fp.top_m = 600.0;
    fp.scalars = M::SCALARS;
    c.fluid = std::make_unique<WeatherFluid>(c.grid, fp);
    std::vector<Climate::BiomeWeather> climate(c.grid->cells());
    for (Climate::BiomeWeather &b : climate) {
        b.ground_k = p_ground_k;
    }
    WeatherForcing drive(*c.fluid, WeatherForcing::Params(), climate);
    c.sounding = drive.sounding(*c.fluid);
    drive.start(*c.fluid, c.sounding);
    return c;
}

// BOMEX's surface fluxes (Siebesma et al. 2003): 8e-3 K m/s, 5.2e-5 m/s of vapour, u* 0.28 m/s.
const M::Surface BOMEX{ 8e-3f, 5.2e-5f, 0.28f };

void read_column(const WeatherFluid &p_fluid, int p_scalar, double *r_x) {
    const float *here = &p_fluid.state()[p_fluid.grid().index(0, 0, 0) * size_t(p_fluid.stride())];
    for (int k = 0; k < p_fluid.levels(); k++) {
        r_x[k] = here[p_fluid.scalar_offset(k, p_scalar)];
    }
}

} // namespace

TEST_CASE("WeatherConvection::adjust: dry air as it is, saturated air keeps its theta_l and q_t") {
    double theta, qc;
    C::adjust(300.0, 0.01, 0.99, theta, qc);
    CHECK(theta == 300.0);
    CHECK(qc == 0.0);
    const double exner = 0.98;
    C::adjust(296.0, 0.022, exner, theta, qc);
    CHECK(qc > 0.0);
    CHECK(theta - M::LATENT / (M::CP * exner) * qc == doctest::Approx(296.0).epsilon(1e-9));
    const double p_mb = M::P0_MB * std::pow(exner, 1.0 / M::KAPPA_EXNER);
    CHECK(qc + M::saturation(theta * exner, p_mb) == doctest::Approx(0.022).epsilon(1e-6));
}

TEST_CASE("WeatherConvection::sigma_w: Holtslag and Moeng's profile, 0.6 w* at its peak, none at the top") {
    double peak = 0.0, at = 0.0;
    for (int i = 1; i < 100; i++) {
        const double z = i * 3.0;
        const double s = C::sigma_w(z, 300.0, 1.0, 0.0);
        if (s > peak) {
            peak = s;
            at = z;
        }
    }
    CHECK(peak == doctest::Approx(0.61).epsilon(0.03));
    CHECK(at / 300.0 > 0.25);
    CHECK(at / 300.0 < 0.45);
    CHECK(C::sigma_w(300.0, 300.0, 1.0, 0.0) == 0.0);
    CHECK(C::sigma_w(100.0, 300.0, 0.0, 0.3) == 0.0);
}

TEST_CASE("WeatherConvection: over a trade-wind sea the updraft saturates under the inversion, a cumulus") {
    Column c = trade_column(300.5f);
    const WeatherFluid &f = *c.fluid;
    double theta[WeatherFluid::MAX_LEVELS], qv[WeatherFluid::MAX_LEVELS], qc[WeatherFluid::MAX_LEVELS];
    read_column(f, M::THETA, theta);
    read_column(f, M::QV, qv);
    read_column(f, M::QC, qc);
    C::Plume u;
    C::plume(theta, qv, qc, c.sounding.exner.data(), f.dz_m(), f.levels(), BOMEX, 300.0, u);
    CHECK_OR_RETURN(u.levels > 0);
    int base = -1;
    for (int k = 0; k < u.levels; k++) {
        if (base < 0 && u.qc[k] > 0.0) {
            base = k;
        }
    }
    MESSAGE(ov::format("base level %d (%.0f m), top %.0f m, M %.3f m/s at the ground", base, (base + 0.5) * f.dz_m(), u.top_m, u.mass[0]).c_str());
    CHECK(base >= 1);
    CHECK((base + 0.5) * f.dz_m() < 300.0); // under the inversion
    CHECK(u.top_m > (base + 0.5) * f.dz_m());
    CHECK(u.top_m < 450.0); // the inversion caps it
    // Over a ground no warmer than its air: no updraft.
    C::Plume none;
    C::plume(theta, qv, qc, c.sounding.exner.data(), f.dz_m(), f.levels(), M::Surface{ -1e-3f, 0.0f, 0.2f }, 300.0, none);
    CHECK(none.levels == 0);
}

TEST_CASE("WeatherConvection::apply: the column's theta_l and q_t kept, the vapour carried up, a convective cloud") {
    Column c = trade_column(300.5f);
    WeatherFluid &f = *c.fluid;
    const int levels = f.levels();
    std::vector<M::Surface> surface(c.grid->cells(), BOMEX);
    auto totals = [&](double &r_tl, double &r_qt, double &r_low_qv) {
        double theta[WeatherFluid::MAX_LEVELS], qv[WeatherFluid::MAX_LEVELS], qc[WeatherFluid::MAX_LEVELS];
        read_column(f, M::THETA, theta);
        read_column(f, M::QV, qv);
        read_column(f, M::QC, qc);
        r_tl = r_qt = 0.0;
        for (int k = 0; k < levels; k++) {
            r_tl += theta[k] - M::LATENT / (M::CP * c.sounding.exner[size_t(k)]) * qc[k];
            r_qt += qv[k] + qc[k];
        }
        r_low_qv = qv[0];
    };
    double tl0, qt0, low0;
    totals(tl0, qt0, low0);
    C convection(f, c.sounding, 300.0);
    for (int step = 0; step < 6; step++) {
        convection.apply(f, surface, 5.0);
    }
    double tl1, qt1, low1;
    totals(tl1, qt1, low1);
    CHECK(tl1 == doctest::Approx(tl0).epsilon(1e-7)); // the state's float32
    CHECK(qt1 == doctest::Approx(qt0).epsilon(1e-7));
    CHECK(low1 < low0); // the updraft takes the surface's vapour up
    double cloud = 0.0;
    for (int k = 0; k < levels; k++) {
        cloud = std::max(cloud, double(convection.cloud()[f.cell_index(0, k)]));
    }
    CHECK(cloud > 1e-5);
    CHECK(convection.tops()[0] > 0.0f);
}
