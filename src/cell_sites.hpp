#pragma once

#include "exact_noise.hpp"
#include "world_coord.hpp"

#include <array>
#include <functional>
#include <vector>

// Procedural sites on the ground: at most one per 3D cell of an absolute grid, where the ground crosses the cell. A
// site is a pure function of its cell, the ground's field and a salt (one per source: the forest's trees, the
// scatter's rocks), so any box gives the same sites for the cells it shares with another, from any origin. No column,
// no y up: the cells are 3D and a site is moved along the field's gradient (planets come).
namespace Archetype {

// Where a source places something: an absolute LOD0 voxel on the ground, a seed and a variant (hashes of the cell: the
// source picks its kind from the variant, e.g. a tree's species).
struct Site {
    VoxelPos base;
    uint32_t seed = 0;
    uint32_t variant = 0;
};

// The ground's field at a LOD0 voxel as LOD p_lod sees it: the gradient (x, y, z) and the distance (w), both in LOD0
// voxels. LOD 0 is the field itself; a coarser LOD extends its nearest lattice sample along its gradient (one cheap
// evaluation, the octaves the LOD drops left out). Thread-safe.
using FieldFn = std::function<ExactNoise::Float4(const VoxelPos &, int)>;
// A FieldFn read at many points at once: r_out[i] = field(p_points[i], p_lods[i]), bit for bit (the walk asks for a
// whole round of readings in one call: a GPU's batch, CascadePointsGPU::ground_field). Thread-safe.
using FieldBatchFn = std::function<void(const VoxelPos *p_points, const int *p_lods, int p_count, ExactNoise::Float4 *r_out)>;
// p_field point by point, over the worker pool (the pool rule of surface_sites).
FieldBatchFn batch_of(const FieldFn &p_field);

// The order of cells (and their sites): z, then y, then x.
inline bool cell_order(const VoxelPos &p_a, const VoxelPos &p_b) {
    return p_a.z != p_b.z ? p_a.z < p_b.z : (p_a.y != p_b.y ? p_a.y < p_b.y : p_a.x < p_b.x);
}

// The side (LOD0 voxels) of the cells for about p_count sites on p_area_m^2 of flat ground: one site per cell crossed by
// the surface. 0 for none.
int64_t forest_cell_side(int p_count, double p_area_m, double p_voxel);

// Sites stand in the middle part of their cell: this share of the side is left out on each side of the two axes across
// the ground's normal, so two sites of cells side by side along the ground are 2 * SITE_INNER cells apart without
// looking at a neighbour. The axis closest to the normal keeps the whole side (a margin there would leave bare stripes
// along the contours, where the ground runs near a cell face).
inline constexpr double SITE_INNER = 0.15;
// Newton steps on the LOD-0 field that snap a site onto the ground (cell_site).
inline constexpr int SNAP_STEPS = 2;
// The coarse field's lattice: at most a third of the cell side, and no coarser than LOD SITE_LOD_MAX.
inline constexpr int SITE_LOD_MAX = 8;
int site_lod(int64_t p_side);

// The site of the 3D cell p_cell (cells of p_side LOD0 voxels on the absolute grid) for the source salted p_salt.
// Whether it exists is decided on the coarse field (site_lod) at the cell's centre: the cell's hashed voxel, in its
// middle 70%, projected on that field's plane must land in the cell's middle part (SITE_INNER). Then it is snapped onto
// the LOD-0 ground (SNAP_STEPS Newton steps on whole voxels) and must end within a voxel of it and inside the cell (a
// surface point has one owner). Seed and variant from the cell's hash.
bool cell_site(const VoxelPos &p_cell, int64_t p_side, uint32_t p_salt, const FieldFn &p_field, Site &r_site);

// cell_site of every cell meeting [p_lo, p_hi] (LOD0 voxels, inclusive) whose site lies in it, with its cell, in cell
// order. The cells tried come from an absolute octree walk (nodes of p_side * 2^k voxels aligned on their own
// multiples, from the coarsest level whose node holds the box) that reads the coarse field once at each node's centre
// and keeps the node while the plane it gives crosses the node, give or take SITE_SLACK cells plus SITE_BEND node sides
// (the ground bends away from the plane, and the cascade's field changes faster than its gradient over tens of metres);
// a kept cell's reading is the one cell_site uses. An empty box costs one evaluation. Without p_snap, the sites stay on
// the coarse plane (no LOD-0 evaluation: far proxies, which only weigh where the sites are). The walk starts at level
// p_min_level at least: two boxes walked from the same level (each node at it holding them) test the same nodes over every
// cell they share, so they keep the same sites there, by construction (a box's own level depends on its extent, and a
// coarser ancestor may prune a cell a finer start keeps). On the worker pool (not from inside a pool task, unless it holds
// an InlineParallelFor).
struct CellSite {
    VoxelPos cell;
    Site site;
};
void surface_sites(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, uint32_t p_salt, const FieldFn &p_field, std::vector<CellSite> &r_out,
        bool p_snap = true, int p_min_level = 0);
// The nodes of p_side << p_level voxels (in node units, cell order) meeting [p_lo, p_hi] that surface_sites' walk from
// p_min_level keeps down to that level: a node it drops holds no site of a box walked from p_min_level (a node's verdict
// is its centre's reading, whatever the box). Every node meeting the box when the walk would start at p_level or below.
// Same pool rule as surface_sites.
void surface_nodes(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, const FieldBatchFn &p_field, int p_level, int p_min_level,
        std::vector<VoxelPos> &r_out);
// surface_sites of many boxes at once, each walk its own (box, cells, salt, snap, start level): the same sites, appended to
// each walk's out in cell order. The walks advance in rounds (a level, then each snap step) and a round reads the field
// once for all of them (p_field, one call; a reading two walks share, an ancestor node of neighbouring tiles, is read
// once): ~levels + SNAP_STEPS + 1 calls whatever the number of boxes, so a batched field (a GPU) pays its latency per
// round, not per box. Pool rule of surface_sites (p_field's).
struct SiteWalk {
    VoxelBox box;
    int64_t side = 0;
    uint32_t salt = 0;
    bool snap = true;
    int min_level = 0;
    std::vector<CellSite> *out = nullptr;
};
struct WalkStats {
    int rounds = 0; // field calls
    int64_t readings = 0; // asked by the walks
    int64_t unique = 0; // read (the distinct (voxel, LOD) of each round)
};
void surface_sites_many(const SiteWalk *p_walks, size_t p_count, const FieldBatchFn &p_field, WalkStats *r_stats = nullptr);

// The level surface_sites starts a box at: the coarsest whose nodes (p_side << level voxels) hold its extent.
int walk_level(const VoxelBox &p_box, int64_t p_side);
inline constexpr double SITE_SLACK = 0.25;
inline constexpr double SITE_BEND = 0.25;

} // namespace Archetype
