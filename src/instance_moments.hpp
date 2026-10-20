#pragma once

#include <cstdint>
#include <utility>
#include <vector>

// Mass properties of the archetype instances: what a part weighs and how its weight spreads, the moment model of the
// instances' LOD (resolution-driven erosion: a part is drawn whole while its cells resolve its thinnest axis;
// aggregates: the parts too thin for a LOD hand their moments to coarse cells).
//
// A Moments is the zeroth, first and central second moments of a solid in voxels: its volume m0 (voxels^3, times a
// weight), its centroid c relative to an anchor (an int64 LOD0 voxel kept by the owner) and S = the integral of
// (x - c)(x - c)^T over it (voxels^5). S / m0 is its covariance; the uniform ellipsoid with the same covariance has
// semi-axes sqrt(5 lambda_i), lambda_i the covariance's eigenvalues. Everything is CPU double math done when a record
// is placed: nothing here reaches the GPU or needs to be bit-exact. A record's moments: Archetype::record_moments
// (archetype_table.hpp).
namespace Archetype {

// A point of a record's local frame in world axes: R^T p_q, p_r the world -> local rotation (rows,
// TableRecord::rotation).
inline void local_to_world(const float *p_r, const double p_q[3], double r_w[3]) {
    for (int j = 0; j < 3; j++) {
        r_w[j] = double(p_r[j]) * p_q[0] + double(p_r[3 + j]) * p_q[1] + double(p_r[6 + j]) * p_q[2];
    }
}

struct Moments {
    double m0 = 0.0;
    double c[3] = {}; // centroid (voxels), relative to the owner's anchor
    double s[6] = {}; // central second moment: xx, yy, zz, xy, xz, yz (voxels^5)

    bool empty() const { return !(m0 > 0.0); }
    // Adds p_other, whose anchor is p_offset voxels from this one's (the parallel axis theorem).
    void add(const Moments &p_other, const double p_offset[3]);
    void add(const Moments &p_other) {
        const double zero[3] = { 0.0, 0.0, 0.0 };
        add(p_other, zero);
    }
    // Weight times: m0 and S scale, the centroid stays.
    Moments scaled(double p_weight) const;
    // The same solid seen through p_r, a world -> local rotation: from the local frame to the world axes, c' = R^T c,
    // S' = R^T S R.
    Moments rotated(const float *p_r) const;
    // The covariance's eigenvalues, ascending (voxels^2), and optionally its unit eigenvectors (row i for lambda_i).
    void principal(double r_lambda[3], double *r_axes = nullptr) const;
    // The equal-covariance ellipsoid's semi-axes sqrt(5 lambda_i), ascending (voxels).
    void semi_axes(double r_axes[3]) const;
    // Its smallest semi-axis: a sphere's radius, 1.12 times a long cylinder's, 1.29 times a box's half side.
    double half_thickness() const;
};

// Point samples (a volume each, at an offset from the anchor) summed raw, centred once at the end.
struct MomentSums {
    double v = 0.0;
    double x[3] = {};
    double xx[6] = {}; // xx, yy, zz, xy, xz, yz

    void add(double p_volume, const double p_at[3]);
    Moments moments() const;
};

// Moments of the basic solids in their own frame, relative to their origin.
Moments box_moments(double p_hx, double p_hy, double p_hz); // centred
Moments sphere_moments(double p_r); // centred
Moments cylinder_moments(double p_half_height, double p_r); // centred, axis y
// A solid of revolution about y between p_y0 and p_y1 whose disc at height y has radius^2 a + b y + c y^2 (clamped at
// 0): a cylinder, a cone or frustum, a sphere's slice. Exact (5-point Gauss-Legendre).
Moments revolution_moments(double p_a, double p_b, double p_c, double p_y0, double p_y1);
// A capsule along y from 0 to p_h, radius p_r0 at y = 0 going linearly to p_r1 at y = p_h, a ball of that radius at
// each end (make_branch_type's shape; p_r0 == p_r1: make_capsule_type's).
Moments tapered_capsule_moments(double p_h, double p_r0, double p_r1);

// Moments per material (global ids, material_library.hpp), each relative to one anchor: an instance made of several
// materials (a tree: its bark and its leaves).
struct MaterialMoments {
    std::vector<std::pair<uint32_t, Moments>> parts; // in order of first appearance

    void add(uint32_t p_material, const Moments &p_moments, const double p_offset[3]);
    const Moments *find(uint32_t p_material) const;
};

} // namespace Archetype
