#pragma once

#include "weather_grid.hpp"

#include <cstdint>
#include <memory>
#include <vector>

// The weather's air as a 3D fluid (docs/weather.md "Fluid"): the non-hydrostatic Boussinesq equations over a
// WeatherGrid's columns (Harris et al. 2003, Hadrich et al. 2020 "Stormscapes"; SAM, Khairoutdinov & Randall 2003, as
// the reference physics):
//   du/dt + (u . grad) u = -grad p + b up - f up x u,   div u = 0
// where p is the kinematic pressure perturbation (over the reference density), b the buoyancy the physics gives (m/s^2)
// and f = 2 Omega n_y. Engine-free, in float state and double geometry.
// - Cells: each column of the grid cut into `levels` levels of dz = top / levels (the shallow-atmosphere approximation:
//   a level's cells share their column's surface metric). A padded cell's record (stride() floats) holds per level the
//   wind (a 3D vector in the planet's axes: its tangent part and its vertical part w along the up) and `scalars` scalars
//   the physics owns (WeatherPhysics: the potential temperature first).
// - A step (the caller's, at most max_dt() long): advect(dt), then the caller's physics on state() (it may change the
//   scalars), then force(dt, buoyancy) (the vorticity confinement, the buoyancy along the up, the Coriolis turn about
//   it, the sponge under the top), then project(dt) (the pressure that takes the wind's divergence out).
// - The advection: semi-Lagrangian (each field read where the air comes from, bilinear in the face's cells and linear
//   between the levels). Its interpolation smears eddies under ~4-8 cells. Tried (2026-11-25): MacCormack's second-order
//   correction (Selle et al. 2008, with their limiter); on every field the trade cumulus became a deck (0.83 of a level)
//   and a fog rose to 180 m, on the wind alone the resolved convection nearly stopped (0.07% of a level, a bubble's air
//   5% unbalanced): the moist physics are tuned to this trace's mixing, so it stays.
// - The vorticity confinement (Fedkiw, Stam & Jensen 2001, "Visual simulation of smoke"; Harris 2003's clouds): the
//   curl w of the wind (central differences, the neighbours' winds as they are: the sphere's turn over a cell is 2% of a
//   radian), N = grad |w| / |grad |w||, the force confinement D (N x w), D the cell's (width^2 dz)^(1/3): it puts back
//   the small swirls the grid's dissipation takes out.
// - The projection is the fluxes through each cell's 6 faces (4 sides, a top, a bottom; ground and top rigid): the
//   pressure solves sum_f a_f (p_f - p) = (sum_f flux_f) / dt with a_f = face area over the centres' distance,
//   red-black Gauss-Seidel (over-relaxed) warm-started from the last step's pressure; the wind at the centres then
//   loses dt times the pressure's central gradient (an approximate projection on a collocated grid: the divergence left
//   is small, not zero, and no checkerboard). Across a seam a neighbour is the halo's (WeatherGrid::fill_halo).
// - The departure point is found in the face's own cells (the cell's dual vectors, no locate): it stays within the
//   halo while the wind crosses at most half a cell in dt (max_dt; across a seam the halo's bilinear values are the
//   neighbour face's).
// Every interior cell of a pass is written from the previous state only (red-black: one colour from the other), so
// parallel_for over rows gives the same bits on any thread count.
class WeatherFluid {
public:
    static constexpr int WIND = 3; // the wind's floats, first in a level's record
    static constexpr int MAX_LEVELS = 64;
    static constexpr int MAX_SCALARS = 8;
    // The largest fraction of a cell (or a level) the wind may cross in a step (max_dt).
    static constexpr double MAX_COURANT = 0.5;
    // The pressure solve's over-relaxation.
    static constexpr double SOR = 1.6;

    struct Params {
        int levels = 15; // in [2, MAX_LEVELS]
        double top_m = 600.0;
        int scalars = 1; // in [0, MAX_SCALARS]
        double omega_rad_s = 0.0; // the Coriolis's Omega
        int pressure_sweeps = 20;
        // The layer under the top whose vertical wind is damped (gravity waves leave it unreflected; 0: a fifth of the
        // top), at a rate rising as the square of the depth into it to 1 / sponge_tau_s at the top.
        double sponge_m = 0.0;
        double sponge_tau_s = 30.0;
        double confinement = 0.0; // epsilon, the vorticity confinement's strength (0: none)
    };

    WeatherFluid(std::shared_ptr<const WeatherGrid> p_grid, const Params &p_params);

    const WeatherGrid &grid() const { return *cells; }
    int levels() const { return params.levels; }
    double dz_m() const { return dz; }
    double top_m() const { return params.top_m; }
    // A level's centre's altitude (metres).
    double level_m(int p_level) const { return (p_level + 0.5) * dz; }
    int scalars() const { return params.scalars; }
    int per_level() const { return WIND + params.scalars; }
    int stride() const { return params.levels * per_level(); }
    int wind_offset(int p_level) const { return p_level * per_level(); }
    int scalar_offset(int p_level, int p_scalar) const { return p_level * per_level() + WIND + p_scalar; }
    // A per-cell table's index (buoyancy, the physics' tables): an interior cell (WeatherGrid::interior) and a level.
    size_t cell_index(size_t p_interior, int p_level) const { return p_interior * size_t(params.levels) + size_t(p_level); }
    size_t cell_count() const { return cells->cells() * size_t(params.levels); }

    // The state (stride() floats per padded cell): the caller sets its scalars and wind, then fill_halo().
    std::vector<float> &state() { return cur; }
    const std::vector<float> &state() const { return cur; }
    void fill_halo();
    // A column's unit up and its surface point (metres from the base's centre).
    void column_up(size_t p_interior, double r_up[3]) const;
    void column_pos(size_t p_interior, double r_pos[3]) const;
    double column_area_m2(size_t p_interior) const { return column[p_interior].area; }
    // The kinematic pressure (m^2/s^2) per padded cell and level (the halo filled).
    const std::vector<float> &pressure() const { return p; }

    // The longest step the wind allows now (MAX_COURANT of a cell or a level; 1e30 at rest).
    double max_dt() const;
    void advect(double p_dt);
    // p_buoyancy: m/s^2 along the up per cell (cell_index).
    void force(double p_dt, const float *p_buoyancy);
    void project(double p_dt);

    // The buoyancy g (theta - p_theta_ref[level]) / THETA_0 of scalar p_scalar (a potential temperature, K): dry air's.
    static constexpr double GRAVITY = 9.81;
    static constexpr double THETA_0 = 300.0; // the Boussinesq reference
    void dry_buoyancy(int p_scalar, const std::vector<double> &p_theta_ref, float *r_buoyancy) const;
    // The flux divergence (1/s) of the wind now at an interior cell and level (the projection's measure).
    double divergence(size_t p_interior, int p_level) const;
    double max_divergence() const;

private:
    // Per interior cell: its surface point and up; the dual vectors of its central differences (grad = ax d_i + ay d_j,
    // d_i the difference across the two neighbours, 1/m); per side (+x, -x, +y, -y) the unit outward normal in the
    // tangent plane, the face's length over the distance between the centres and the face's length (metres); the
    // column's area (m^2) and f.
    struct Column {
        double pos[3];
        float up[3];
        float ax[3];
        float ay[3];
        float normal[4][3];
        float length[4];
        float distance[4];
        float area;
        float f;
    };
    static constexpr int SIDES = 4;

    std::shared_ptr<const WeatherGrid> cells;
    Params params;
    double dz = 40.0;
    std::vector<Column> column;
    std::vector<float> cur;
    std::vector<float> next;
    std::vector<float> p; // per padded cell and level
    std::vector<float> rhs; // per interior cell and level
    std::vector<float> vort, vort_mag; // the confinement's: the curl (3 per padded cell and level), its length
    double min_spacing_m = 1.0;

    // The padded indices of an interior cell's 4 side neighbours (+x, -x, +y, -y).
    void neighbours(int p_face, int p_i, int p_j, size_t r_nb[SIDES]) const;
    void make_columns();
    void sweep(int p_colour);
    // The vorticity confinement's push over p_dt (the halo filled first).
    void confine(double p_dt);
    // The 3D gradient at an interior cell of a per-level field p_f (p_stride floats per padded cell, p_at its offset in a
    // level, p_per its floats per level), from the neighbours p_nb and the levels round p_level.
    void gradient(const Column &p_c, const float *p_f, size_t p_here, const size_t p_nb[SIDES], int p_stride, int p_per, int p_at, int p_level,
            double r_g[3]) const;
    // The wind r_wind brought from the departure point p_c.pos + p_d turned into the column's frame: the rotation
    // carrying the departure's up onto the column's (parallel transport over the sphere; its vertical speed kept).
    static void transport(const Column &p_c, const double p_d[3], float *r_wind);
};
