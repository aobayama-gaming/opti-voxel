#pragma once

#include "instance_moments.hpp"
#include "world_coord.hpp"

#include <vector>

// The far field of the archetype instances: what the parts a LOD drops still weigh. For each LOD L, the parts dropped
// at L or before (TableRecord::max_lod <= L) hand their moments (record_moments) to cells of 2^(L + cell_shift) LOD0
// voxels on an absolute grid, per material; each cell becomes a blob, a smooth occupancy kernel with the cell's mass and
// spread, which the LOD L chunks fold into a coarse surface (a forest's canopy seen from far above; the fold is the
// next step).
//
// A part is splatted onto the 2 x 2 x 2 cells around its centroid with tent weights (each cell's share falls linearly
// to 0 one cell away): the blobs of a uniformly covered area sum to a near-constant occupancy, with no seam where the
// cell edges fall (nearest-cell assignment split a canopy lying across a cell boundary into random halves and cut
// holes along the edges). Each cell's spread is that of the parts' actual positions, so a thin canopy still makes
// thin blobs: it spreads only along the axes the parts are spread along. Each cell's moments are summed in item order,
// so the same items give the same bits.
//
// A blob's kernel is a max(1 - q, 0)^2 with q = sum_i (l_i / s_i)^2, l the point in the blob's principal frame: a
// compact polynomial (mul / add only: the GPU and the CPU agree bit for bit), whose spread per axis is s_i^2 / 9, so
// s_i = 3 sigma_i with sigma_i^2 the cell's variance along that axis plus the LOD's filter (filter * 2^L)^2, the box
// filter of a cell of the LOD (sigma = 1 / sqrt(12) cells by default). a = m0 * 105 / (32 pi s_x s_y s_z): its
// integral is the cell's mass, so a uniformly filled region reads its fill fraction.
namespace Archetype {

// One part, as the layers see it.
struct MomentItem {
    VoxelPos anchor; // the moments' anchor
    Moments moments; // weight included
    uint32_t material = 0;
    int drop_lod = 0; // the first LOD that no longer lists it
};

struct MomentLayerParams {
    int cell_shift = 2; // cells of 2^(L + cell_shift) voxels at LOD L: 4 cells of the LOD
    double filter = 0.28867513459481287; // the kernel's added sigma, cells of the LOD (1 / sqrt(12): a box filter)
    // A blob whose occupancy, with its neighbours', stays under this is pruned, and one whose own peak is under 1/16 of
    // it (a sliver of a part's tent: it only spreads its neighbours' surface).
    double min_occupancy = 1.0 / 32.0;
    int lod_min = 0; // the layers built
    int lod_max = 14;
};

// A blob of layer lod: a record-to-be (the fold's next step places it in the table).
struct MomentBlob {
    int lod = 0;
    uint32_t material = 0;
    VoxelPos anchor; // the cell's centroid, rounded to the voxel
    float rotation[9] = {}; // world -> local: the principal axes (rows), orthonormal
    float inv_s2[3] = {}; // 1 / s_i^2 (voxels^-2)
    float amplitude = 0.0f; // a: the kernel's peak occupancy
    double s[3] = {}; // the support's semi-axes (voxels)
    double mass = 0.0; // the cell's m0
    double peak_bound = 0.0; // a bound of the occupancy summed over the layer's blobs, anywhere in its support

    // The kernel at a point p_offset voxels from the anchor (double: tests and reports; the fold has its own GLSL / IR).
    double occupancy(const double p_offset[3]) const;
};

// The blobs of layers lod_min .. lod_max, in (lod, cell, material) order; the pruned ones left out.
void build_moment_blobs(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params, std::vector<MomentBlob> &r_out);

// A blob of one cell's moments at layer p_lod (the kernel's frame, size and amplitude); p_anchor: the moments' anchor.
MomentBlob make_blob(const Moments &p_cell, const VoxelPos &p_anchor, int p_lod, uint32_t p_material, double p_filter_voxels);

} // namespace Archetype
