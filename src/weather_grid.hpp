#pragma once

#include "base_surface.hpp"

#include "ov/str.hpp"

#include <cstdint>
#include <memory>
#include <vector>

// The weather's grid (docs/weather.md "Grid"): a planet's surface as faces of width x height cells, each face padded with
// a halo of HALO cells read from its neighbours, so a stencil or a bilinear read crosses a seam like any other cell. The
// interface the simulation (WeatherSim), the worker and the clouds see; a base's shape is one implementation of it
// (CubeSphereGrid for a sphere). Engine-free and in double; positions are metres from the base's centre.
// - The padded layout: x fastest, then y, then the face (index()); x in [-HALO, width + HALO), y likewise. A field has
//   `comps` floats per padded cell, interleaved. The layout is also the renderer's atlas: a 2D texture of stride_x()
//   texels by faces() * stride_y() rows, row-major (a level of the weather is one such slice).
// - The halo of each face is a bilinear read of the interior at the halo cell's surface point (fill_halo, from a table
//   made once, at construction): the neighbour's for a seam, the face's own across a wrap; a corner's halo cell, which no
//   face may hold, is linear in its own face's chart from its neighbours. A smooth field stays smooth across a seam to
//   second order, at the corners too.
// - The grid is the surface only: the simulation's levels stand over it along the up (the shallow-atmosphere
//   approximation: a level's cells share their column's surface metric).
class WeatherGrid {
public:
    static constexpr int HALO = 1;
    // The cells a face side: at least 4 (bilinear reads stay inside one neighbour), at most 1024.
    static constexpr int MIN_N = 4;
    static constexpr int MAX_N = 1024;

    // A surface point on the grid: its face and continuous cell coordinates (cell (i, j) spans [i, i + 1) x [j, j + 1)).
    struct Point {
        int face = 0;
        double x = 0.0;
        double y = 0.0;
    };

    virtual ~WeatherGrid() = default;

    // The cells a face side of a sphere's grid when no width is asked for.
    static constexpr int DEFAULT_SIDE = 64;
    // The cells' width asked for (p_cell_m > 0) or the default grid's over a sphere of p_radius_m (p_cell_m 0).
    static double cell_width(double p_radius_m, double p_cell_m);

    // The grid of cells about p_cell_m wide (cell_width) over p_base (LOD0 voxels of p_voxel_m metres): a CubeSphereGrid on a sphere,
    // none on another base (r_error says why, and on a sphere whose side would leave [MIN_N, MAX_N]).
    static std::unique_ptr<WeatherGrid> for_base(const BaseSurface &p_base, double p_voxel_m, double p_cell_m, ov::str *r_error = nullptr);

    int faces() const { return face_count; }
    int width() const { return nx; }
    int height() const { return ny; }
    int stride_x() const { return nx + 2 * HALO; }
    int stride_y() const { return ny + 2 * HALO; }
    // The interior cells and the padded ones.
    size_t cells() const { return size_t(face_count) * size_t(nx) * size_t(ny); }
    size_t padded_cells() const { return size_t(face_count) * size_t(stride_x()) * size_t(stride_y()); }
    // The padded index of cell (p_i, p_j) of p_face, p_i in [-HALO, width + HALO), p_j in [-HALO, height + HALO).
    size_t index(int p_face, int p_i, int p_j) const {
        return (size_t(p_face) * size_t(stride_y()) + size_t(p_j + HALO)) * size_t(stride_x()) + size_t(p_i + HALO);
    }
    // The interior index of an interior cell (face-major, then y, then x): a per-cell table's.
    size_t interior(int p_face, int p_i, int p_j) const { return (size_t(p_face) * size_t(ny) + size_t(p_j)) * size_t(nx) + size_t(p_i); }
    // The grid's typical cell width (metres): its scale for a stabiliser.
    double cell_m() const { return cell_width_m; }

    // The surface point (metres from the base's centre) and its unit up at continuous cell coordinates of a face; the
    // coordinates may leave the face (the halo's points).
    virtual void surface(int p_face, double p_x, double p_y, double r_pos[3], double r_up[3]) const = 0;
    void cell_surface(int p_face, int p_i, int p_j, double r_pos[3], double r_up[3]) const { surface(p_face, p_i + 0.5, p_j + 0.5, r_pos, r_up); }
    // The surface point under (or over) a position in metres from the base's centre.
    virtual Point locate(const double p_pos[3]) const = 0;

    // The halo of a field of p_comps floats per padded cell, from the interior (which it leaves untouched).
    void fill_halo(float *p_field, int p_comps) const;
    // The field (halo filled) under a position, bilinear between the 4 cell centres round it: r_value gets p_comps floats
    // read from p_field + p_stride * cell (p_stride >= p_comps: a level of a wider record).
    void sample(const float *p_field, int p_stride, int p_comps, const double p_pos[3], float *r_value) const;
    void sample(const float *p_field, int p_comps, const double p_pos[3], float *r_value) const { sample(p_field, p_comps, p_comps, p_pos, r_value); }

    // `vec2 weather_uv(vec3 p)`: the atlas coordinates (over [0, 1]^2) of the column under a render-space point p, from
    // the base's render functions (BaseSurface::glsl_render); a face's coordinates kept within its padded block, so no
    // face bleeds into the next under bilinear filtering. In float: not bit-exact with sample() (the weather is a look,
    // never a terrain input).
    virtual ov::str glsl_lookup() const = 0;

protected:
    WeatherGrid(int p_faces, int p_nx, int p_ny, double p_cell_m);
    // Builds the halo's table from surface() and locate(): the subclass's constructor calls it last.
    void make_halo();

private:
    int face_count = 1;
    int nx = MIN_N;
    int ny = MIN_N;
    double cell_width_m = 1.0;
    // Each halo cell's 4 sources and their weights (fill_halo): the edges' read interiors, then the corners' read edges.
    struct HaloRead {
        uint32_t dst;
        uint32_t src[4];
        float w[4];
    };
    std::vector<HaloRead> halo;
};

// A sphere as an equiangular cube: 6 faces of N x N cells. A face is a cube axis (MAJOR) with two tangent axes (U, V;
// U x V = MAJOR). A direction d belongs to the face of its largest |component| (x before y before z on a tie), at the
// angles a = atan(d.U / d.MAJOR), b = atan(d.V / d.MAJOR) in [-pi/4, pi/4]; its continuous cell coordinates are
// x = (a / (pi/2) + 1/2) N, y likewise from b. Equal angles keep the cells' areas within a factor ~1.4 of each other (a
// gnomonic grid's: ~5). A position's cell is its direction's: the grid reads directions only.
class CubeSphereGrid : public WeatherGrid {
public:
    static constexpr int FACES = 6;
    struct Face {
        int major[3];
        int u[3];
        int v[3];
    };
    static const Face FACE[FACES];

    // The cells a face side for cells about p_cell_m wide over a sphere of p_radius_m (may leave [MIN_N, MAX_N]).
    static double side_for(double p_radius_m, double p_cell_m);
    // p_n cells a face side over a sphere of p_radius_m: p_n is clamped to [MIN_N, MAX_N] (ERR when it was outside).
    CubeSphereGrid(int p_n, double p_radius_m);

    int size() const { return width(); }
    double radius_m() const { return radius; }
    // The unit direction at continuous cell coordinates of a face; x and y may leave [0, N] (the halo's centres).
    void direction(int p_face, double p_x, double p_y, double r_dir[3]) const;
    void cell_direction(int p_face, int p_i, int p_j, double r_dir[3]) const { direction(p_face, p_i + 0.5, p_j + 0.5, r_dir); }

    void surface(int p_face, double p_x, double p_y, double r_pos[3], double r_up[3]) const override;
    Point locate(const double p_pos[3]) const override;
    ov::str glsl_lookup() const override;

private:
    double radius = 1.0;
};
