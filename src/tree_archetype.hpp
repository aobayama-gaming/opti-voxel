#pragma once

#include "archetype_table.hpp"
#include "cell_sites.hpp"

#include <array>
#include <vector>

// Trees as archetype-table instances (archetype step 5, v1). A tree is a contiguous run of ordinary
// TableRecords, smooth-unioned into the terrain in id order:
// - branch segments (the `branch` type, a tapered capsule) from one recursion that makes both the topology and
//   the bends: a stem's polyline is its straight axis split at midpoints moved by a hash of their path, its
//   children fork off at hashed points with radii from the pipe model (r^2 of the parent >= the sum of the
//   children's r^2 plus the tip's);
// - a crown hull (1-3 spheres along the upper trunk) and a clump at the tip of every leaf stem (the `crown`
//   type: a rock with bumps of about CROWN_WAVELENGTH_M), or for a conifer a few stacked cones with the same
//   bumps (the `cone` type).
// Nothing in it depends on the LOD: the band rule (set_band) is the LOD rule. Each record stays whole while the
// grid resolves it, erodes in the LODs just before its drop and is dropped once gone, so the thin parts leave first
// and the hull last, and no part appears with the LOD: the records listed at LOD L+1 are a subset of those at L. The
// generation is CPU work done once (double math, hashes of the tree's seed and each part's path); the GPU reads
// the records as data. Branch segments start and end on whole voxels, so a segment's end is the next one's anchor.
namespace Archetype {

// One species (metres, radians). Each tree scales it by a hashed factor.
struct TreeSpecies {
    double height = 14.0; // the trunk, base to top
    double trunk_radius = 0.3; // at the base
    double lean = 0.1; // the trunk's largest lean from vertical
    double crown_base = 0.4; // limbs fork off the trunk above this fraction of it
    double crown_radius = 3.5; // the limbs' reach
    double hull_ratio = 0.55; // hull sphere radius / crown_radius
    int hull_spheres = 2; // along the trunk, between the crown base and the top
    double hull_taper = 1.0; // the top hull sphere's (or tier's) radius / the bottom one's
    int cone_tiers = 0; // > 0: the crown is that many stacked cones (a conifer) instead of hull spheres
    int limbs = 6; // order-1 stems on the trunk
    int twigs = 2; // order-2 stems on each limb
    double limb_angle = 0.9; // a child's angle from its parent's axis
    double child_ratio = 0.45; // child radius / parent radius at the fork
    double end_ratio = 0.5; // a stem's tip radius is at most this of its base radius
    double tip_ratio = 0.2; // and at least this (the pipe model's floor)
    double twig_length = 0.5; // order-2 length / parent length
    double tortuosity = 0.15; // a midpoint moves by up to this times its segment's length
    double trunk_tortuosity = 0.05; // the same for the trunk
    double segment_ratio = 10.0; // a segment is split while longer than this many radii
    double clump_ratio = 0.38; // clump radius / crown_radius
    double flare = 2.0; // the trunk's smooth union with the ground, times its radius
    double fork_smooth = 0.5; // a child's smooth union with its parent, times its radius
    double crown_smooth = 0.5; // the hull's and clumps' smooth union, times their radius: one foliage mass
    int budget = 96; // records per tree: the thinnest stems are cut first
};

// The species the forest draws from.
const std::vector<TreeSpecies> &tree_species();

// The crown type: make_rock_type(crown_k(voxel), CROWN_SEED). Its bump amplitude is set so that the bumps add
// CROWN_BUMP to its Lipschitz bound (the band depth, the locality margin and the list lengths grow with it).
inline constexpr uint32_t CROWN_SEED = 0xC40Du;
inline constexpr double CROWN_WAVELENGTH_M = 0.8;
inline constexpr double CROWN_BUMP = 1.5;
int crown_k(double p_voxel);

// The library types trees use and the crown type's numbers, read from its bounds (crown_types).
struct TreeTypes {
    int branch = -1; // make_branch_type()
    int crown = -1; // the crown type
    int cone = -1; // make_cone_type(crown_k(voxel), CROWN_SEED): its bumps are the crown's
    float crown_amp = 0.0f; // its amp param: the bumps add CROWN_BUMP to L
    float crown_dent = 0.0f; // how far the bumps can reach inside its radius (voxels)
};
TreeTypes crown_types(int p_branch, int p_crown, const TableDef &p_crown_def, int p_cone = -1);

// One stem as grown (tests): its order, base and tip radius and the radii of the children forking off it (voxels).
struct TreeStem {
    int order = 0;
    double r0 = 0.0;
    double r_end = 0.0;
    std::vector<double> children;
};

// A unit direction in absolute axes (SDFBase::up_at).
using UpAxis = std::array<double, 3>;
inline constexpr UpAxis UP_Y{ 0.0, 1.0, 0.0 };

// Appends one tree's records to r_out (op, k, anchor, rotation, params and salt set; finish_record sets the
// boxes and the band). p_base: the absolute LOD0 voxel at the bottom of the trunk. p_up: the unit direction the
// trunk grows along, and the cone tiers' axis; the tree is grown in absolute axes, joints on whole voxels (UP_Y: the
// records from before trees had an up axis, bit for bit). Deterministic in (species, up, seed, voxel); p_base only
// translates it. The stems, in the order they were grown, go to r_stems if given.
void grow_tree(const TreeSpecies &p_species, const VoxelPos &p_base, const UpAxis &p_up, uint32_t p_seed, double p_voxel, const TreeTypes &p_types,
        std::vector<TableRecord> &r_out, std::vector<TreeStem> *r_stems = nullptr);

// A tree in the table: its records are the ids [first, first + count), grown from (species, seed).
struct TreeSpan {
    uint32_t first = 0;
    uint32_t count = 0;
    int species = 0;
    uint32_t seed = 0;
};

// The salt of the forest's cell sites (cell_sites.hpp); a site's species is its variant modulo the species count.
inline constexpr uint32_t TREE_SITE_SALT = 0xF0E57u;
int tree_species_of(uint32_t p_variant);

} // namespace Archetype
