#pragma once

#include "base_surface.hpp"

#include "ov/str.hpp"

#include <cstdint>
#include <vector>

// The weather's grid (docs/weather.md "Grid"): the whole planet as an equiangular cube sphere, 6 faces of N x N cells, each
// face padded with a halo of HALO cells read from its neighbours, so a stencil or a bilinear read crosses a seam like any
// other cell. Engine-free and in double: the simulation (WeatherSim) steps fields laid out on it, the renderer reads them
// through glsl_lookup (float, the same layout as a 3D texture).
// - A face is a cube axis (MAJOR) with two tangent axes (U, V; U x V = MAJOR). A direction d belongs to the face of its
//   largest |component| (x before y before z on a tie), at the angles a = atan(d.U / d.MAJOR), b = atan(d.V / d.MAJOR) in
//   [-pi/4, pi/4]; its continuous cell coordinates are x = (a / (pi/2) + 1/2) N, y likewise from b. Equal angles keep the
//   cells' areas within a factor ~1.4 of each other (a gnomonic grid's: ~5).
// - The padded layout: x fastest, then y, then the face (index()); x, y in [-HALO, N + HALO). A field has `comps` floats
//   per padded cell, interleaved.
// - The halo of each face is a bilinear read of its neighbour's interior at the halo cell's centre extended past the edge
//   (fill_halo, from a table made once); a cube corner's halo cell, which no face holds, is linear in its own face's chart
//   from its neighbours. A smooth field stays smooth across a seam to second order, at the corners too.
// The grid is the planet's directions only: a cell's position is its direction at an altitude over a SPHERE (cell_voxel).
class WeatherGrid {
public:
    static constexpr int FACES = 6;
    static constexpr int HALO = 1;
    // The cells a face side: at least 4 (bilinear reads stay inside one neighbour), at most 1024 (6 M cells).
    static constexpr int MIN_N = 4;
    static constexpr int MAX_N = 1024;

    struct Face {
        int major[3];
        int u[3];
        int v[3];
    };
    static const Face FACE[FACES];

    // A direction on the grid: its face and continuous cell coordinates (cell (i, j) spans [i, i + 1) x [j, j + 1)).
    struct Point {
        int face = 0;
        double x = 0.0;
        double y = 0.0;
    };

    // A grid of p_n cells a face side: p_n is clamped to [MIN_N, MAX_N] (ERR when it was outside).
    explicit WeatherGrid(int p_n);

    int size() const { return n; }
    int stride() const { return n + 2 * HALO; }
    // The interior cells (6 N^2) and the padded ones (6 (N + 2 HALO)^2).
    size_t cells() const { return size_t(FACES) * size_t(n) * size_t(n); }
    size_t padded_cells() const { return size_t(FACES) * size_t(stride()) * size_t(stride()); }
    // The padded index of cell (p_i, p_j) of p_face, p_i and p_j in [-HALO, N + HALO).
    size_t index(int p_face, int p_i, int p_j) const {
        return (size_t(p_face) * size_t(stride()) + size_t(p_j + HALO)) * size_t(stride()) + size_t(p_i + HALO);
    }

    // The face and cell coordinates of a direction (any length but 0).
    Point locate(const double p_dir[3]) const;
    // The unit direction at continuous cell coordinates of a face; x and y may leave [0, N] (the halo's centres).
    void direction(int p_face, double p_x, double p_y, double r_dir[3]) const;
    // The unit direction through the centre of cell (p_i, p_j).
    void cell_direction(int p_face, int p_i, int p_j, double r_dir[3]) const { direction(p_face, p_i + 0.5, p_j + 0.5, r_dir); }

    // The halo of a field of p_comps floats per padded cell, from the interior (which it leaves untouched).
    void fill_halo(float *p_field, int p_comps) const;
    // The field (halo filled) at a direction, bilinear between the 4 cell centres round it: r_value gets p_comps floats.
    void sample(const float *p_field, int p_comps, const double p_dir[3], float *r_value) const;

    // The LOD0 voxel at p_alt altitude (LOD0 voxels) over a SPHERE along a unit direction: its centre plus the direction
    // times radius + altitude, rounded. ERR and the centre on another base.
    static VoxelPos cell_voxel(const BaseSurface &p_base, const double p_dir[3], double p_alt);

    // `vec3 weather_uvw(vec3 dir, float n)`: locate in float for a 3D texture of the padded layout (N + 2 HALO texels
    // wide and high, FACES deep): x and y the texel coordinates over [0, 1] (the cell centres at (i + HALO + 0.5) / (N + 2
    // HALO)), z the face's index (an integer: the caller maps it onto its texture's depth). A smooth field read through it
    // is not bit-exact with sample() (atan in float32, hardware filtering): the weather is a look, never a terrain input.
    static ov::str glsl_lookup();

private:
    int n = MIN_N;
    // Each halo cell's 4 sources and their weights (fill_halo): the edges' read interiors, then the corners' read edges.
    struct HaloRead {
        uint32_t dst;
        uint32_t src[4];
        float w[4];
    };
    std::vector<HaloRead> halo;

    void make_halo();
};
