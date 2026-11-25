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
//   scalars), then force(dt, buoyancy) (the buoyancy along the up, the Coriolis turn about it, the sponge under the
//   top), then project(dt) (the pressure that takes the wind's divergence out).
// - The advection: one finite-volume scheme for every field, the wind's 3 components, the scalars and the interfaces'
//   w alike (the interfaces' volumes span a level's centre to the next's): a field changes only by its fluxes through
//   its volume's 6 faces, so what one cell loses its neighbour gains (conserved to the floats' rounding). A face's value
//   is the upwind volume's, linear from its centre along its gradient, the gradient limited so no face value leaves the
//   range of the volume and its 6 neighbours (Barth & Jespersen 1989: no new extreme, water never negative); two stages
//   (Heun, SSP-RK2, Shu & Osher 1988). The limiter's dissipation is the subgrid model: an implicit large-eddy simulation
//   (Margolin, Smolarkiewicz & Sorbjan 1999 ran the convective boundary layer this way, with MPDATA, a flux-form scheme).
//   The wind's components are the planet's axes; centre_vertical then sets each centre's w from its interfaces, which
//   also takes out the radial part a wind gains following the sphere.
// - Faces: each cell's 4 sides and its top and bottom. A side's neighbour is the true cell across it, across a seam the
//   neighbour face's (Column::across; the faces' cells meet one to one along their edges), and a face belongs to its
//   lower-index cell: its length, normal, speed and flux are computed once, from that side, so both cells see the same
//   bits. The sides' speeds are the two centres' mean normal wind, then corrected by the projection.
// - The projection: the pressure solves sum_f a_f (p_f - p) = (sum_f flux_f) / dt over the same faces, a_f the face's
//   area over the centres' distance (ground and top rigid), red-black Gauss-Seidel over-relaxed (a seam's neighbour read
//   at the half sweep's start), warm-started from the last step's pressure. Each side's speed then loses dt times the
//   difference of the two pressures across it, the solve's own (the MAC projection of a cell-centred grid: Bell,
//   Colella & Glaz 1989; Almgren et al. 1998), so the fluxes that move the fields keep only the solver's residual
//   divergence; the vertical wind lives on the levels' interfaces (staggered: SAM's C grid in the vertical) and loses
//   the difference of the two levels' pressures round it. The centres' tangent wind loses the pressure's central
//   gradient (the halo's). Collocated in the vertical (until 2026-11-25), a w alternating level by level had zero face
//   means: the projection never saw it.
// - Measured (2026-11-25) on the dry convective boundary layer (test_weather_cbl.cpp; Moeng & Sullivan 1994's peak w
//   rms 0.63 w* near 0.35 h, entrainment flux ~-0.2 Q): 0.625 w* at 0.44 h, -0.26 Q. MacCormack (a semi-Lagrangian
//   step, corrected) gave the same dry profile but created water at the clouds' edges (+0.25 g/kg in 5 minutes of
//   BOMEX, the extra rain evaporating into a 5 K/h cooling); this scheme conserves it.
// Every interior cell of a pass is written from the previous state only (red-black: one colour from the other), so
// parallel_for over rows gives the same bits on any thread count.
class WeatherFluid {
public:
    static constexpr int WIND = 3; // the wind's floats, first in a level's record
    static constexpr int MAX_LEVELS = 64;
    static constexpr int MAX_SCALARS = 8;
    // The largest sum of a volume's Courant numbers over the directions in a step (max_dt).
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

    // The state (stride() floats per padded cell): the caller sets its scalars and wind, then fill_halo(). The wind's
    // vertical part at a centre is the mean of the interfaces' round it (vertical_w): a caller that sets it calls
    // vertical_from_centres() first, else what it wrote there is replaced at the next step.
    std::vector<float> &state() { return cur; }
    const std::vector<float> &state() const { return cur; }
    void fill_halo();
    // The interfaces' vertical wind from the centres' (each interface the mean of the two levels round it; the ground's
    // and the top's 0).
    void vertical_from_centres();
    // The vertical wind (m/s) at interface p_m of a padded cell: p_m in [0, levels], the bottom of level p_m (0 the ground,
    // levels the top: both 0).
    float vertical_w(size_t p_padded, int p_m) const { return wf[p_padded * size_t(params.levels + 1) + size_t(p_m)]; }
    // A column's unit up and its surface point (metres from the base's centre).
    void column_up(size_t p_interior, double r_up[3]) const;
    void column_pos(size_t p_interior, double r_pos[3]) const;
    double column_area_m2(size_t p_interior) const { return column[p_interior].area; }
    // The kinematic pressure (m^2/s^2) per padded cell and level (the halo filled).
    const std::vector<float> &pressure() const { return p; }

    // The longest step the wind allows now (MAX_COURANT summed over the directions; 1e30 at rest).
    double max_dt() const;
    void advect(double p_dt);
    // p_buoyancy: m/s^2 along the up per cell (cell_index).
    void force(double p_dt, const float *p_buoyancy);
    void project(double p_dt);

    // The buoyancy g (theta - p_theta_ref[level]) / THETA_0 of scalar p_scalar (a potential temperature, K): dry air's.
    static constexpr double GRAVITY = 9.81;
    static constexpr double THETA_0 = 300.0; // the Boussinesq reference
    static constexpr double PRANDTL = 1.0 / 3.0; // the turbulent Prandtl number (the subgrid closures' stability limit)
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
        // The cell across each side (interior and padded index: across a seam the neighbour face's, not a halo cell)
        // and which of its sides faces back. A face is its lower-index cell's (owns): its speed, length and flux.
        uint32_t across[4];
        uint32_t across_pad[4];
        uint8_t back[4];
    };
    static constexpr int SIDES = 4;

    std::shared_ptr<const WeatherGrid> cells;
    Params params;
    double dz = 40.0;
    std::vector<Column> column;
    std::vector<float> cur;
    std::vector<float> next;
    // The vertical wind on the levels' interfaces (levels + 1 per padded cell, the ground's and the top's 0): the
    // prognostic one (a vertically staggered grid, Lorenz / SAM's C grid): the projection's vertical difference and
    // divergence are exact on it, so no level-by-level checkerboard escapes the pressure.
    std::vector<float> wf;
    std::vector<float> wf_next;
    std::vector<float> corr, wf_corr; // the second stage's results
    std::vector<float> wf_from; // the interfaces' w at the step's start (the stages' speeds)
    std::vector<float> side_speed; // per interior cell, level and side: the owned faces' outward speeds
    std::vector<float> p_from; // the pressure at a half sweep's start
    std::vector<float> grad_c, grad_w; // the limited gradients (3 per field and volume per padded cell)
    std::vector<float> p; // per padded cell and level
    std::vector<float> rhs; // per interior cell and level
    double min_spacing_m = 1.0;

    // The padded indices of an interior cell's 4 side neighbours (+x, -x, +y, -y).
    void neighbours(int p_face, int p_i, int p_j, size_t r_nb[SIDES]) const;
    // Both halos (the state's and the interfaces').
    void halo();
    // Each interior centre's vertical wind set to the mean of its interfaces', then the halos.
    void centre_vertical();
    float *face_w(size_t p_padded) { return &wf[p_padded * size_t(params.levels + 1)]; }
    const float *face_w(size_t p_padded) const { return &wf[p_padded * size_t(params.levels + 1)]; }
    void make_columns();
    void make_faces();
    bool owns(size_t p_interior, int p_side) const { return p_interior < column[p_interior].across[p_side]; }
    // A face's geometry (its owner's) and outward speed from either side.
    double face_length(size_t p_interior, int p_side) const {
        const Column &c = column[p_interior];
        return owns(p_interior, p_side) ? c.length[p_side] : column[c.across[p_side]].length[c.back[p_side]];
    }
    double face_distance(size_t p_interior, int p_side) const {
        const Column &c = column[p_interior];
        return owns(p_interior, p_side) ? c.distance[p_side] : column[c.across[p_side]].distance[c.back[p_side]];
    }
    double outward(size_t p_interior, int p_level, int p_side) const;
    size_t padded(size_t p_interior) const {
        const size_t per_face = size_t(cells->width()) * size_t(cells->height());
        return cells->index(int(p_interior / per_face), int(p_interior % size_t(cells->width())), int((p_interior % per_face) / size_t(cells->width())));
    }
    // The normal wind through each owned face of each interior cell's levels from the centres alone (their mean), into
    // side_speed (project() then corrects them).
    void face_speeds();
    // One forward-Euler stage of the flux-form advection (advect) from p_in to r_out (halo filled): p_count volumes per
    // padded cell of p_per fields each, those from p_first to p_last moved; p_vertical(padded, v) the speed up through
    // the face between volumes v - 1 and v, p_side(interior, v, s) the speed out through side s.
    template <typename Vertical, typename Side>
    void flux_stage(const std::vector<float> &p_in, std::vector<float> &r_out, int p_count, int p_per, int p_first, int p_last, double p_dt,
            Vertical p_vertical, Side p_side);
    void sweep(int p_colour);
};
