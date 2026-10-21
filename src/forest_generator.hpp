#pragma once

#include "tree_archetype.hpp"

#include <vector>

// The procedural forest: its trees are a pure function of the 3D cells of its zone (forest_sites), so any part of it
// can be made alone, anywhere, at any time, the same: the table's forest at configure, streamed by cell next. Far away
// a tree is its moments, and those come without growing it (TreeClasses).
namespace Archetype {

// A tree's moments far away, per class: its species and its size bucket. Every tree is unique (grown from its seed),
// but every length of it is its species' times tree_scale(seed), so its moments are close to its class's mean at scale
// 1 resized by its scale and turned to its up. Not exactly: the per-tree record budget cuts more of a big tree's thin
// stems (and their leaf clumps), hence the size buckets (one class per species left the broadleaf's leaves 5% light in
// sum). A class's mean is over SAMPLES grown trees of that bucket, each brought to scale 1 (m0 / s^3, c / s, S / s^5),
// upright; their leans and azimuths average out, their spread of centroids counts (a mixture's covariance).
struct TreeSamples;
class TreeClasses {
public:
    static constexpr int SAMPLES = 32;
    static constexpr int SIZE_BUCKETS = 8; // over tree_scale's [0.8, 1.2)

    TreeClasses() = default;
    TreeClasses(const DefLibrary &p_library, const TreeSamples &p_samples);
    TreeClasses(const DefLibrary &p_library, const TreeTypes &p_types, double p_voxel);

    // The moments of p_site's tree grown along p_up, per material, relative to its base (voxels).
    MaterialMoments tree(const TreeSite &p_site, const UpAxis &p_up) const;
    // A class's mean at scale 1, upright, relative to the base (empty and ERR if there is no such class).
    const MaterialMoments &mean(int p_species, int p_bucket) const;
    static int size_bucket(double p_scale);

private:
    std::vector<MaterialMoments> classes; // species-major
};

// The trees the classes (and a forest's profile, forest_field.hpp) are read from: per species, SAMPLES hashed seeds per
// size bucket (species-major, bucket by bucket in hash order), each grown upright at the origin, its records not
// finished. As a whole, a fair draw of the forest's trees: its species are uniform, its scales uniform over the buckets.
struct TreeSamples {
    std::vector<int> species;
    std::vector<uint32_t> seeds;
    std::vector<std::vector<TableRecord>> parts;

    TreeSamples(const TreeTypes &p_types, double p_voxel);
    size_t size() const { return seeds.size(); }
};

// A forest: the trees of the 3D cells of `side` voxels whose base lies in `zone`. A tree's key is its cell.
struct ForestGenerator {
    VoxelBox zone;
    int64_t side = 0;

    // The trees whose base lies in p_query and the zone, in cell order (forest_sites: a tree's neighbours, for the
    // spacing, are looked up past the query, so a tree is the same from any query; only the zone's trees count).
    void trees(const VoxelBox &p_query, const FieldFn &p_ground, std::vector<CellSite> &r_out) const;
};

} // namespace Archetype
