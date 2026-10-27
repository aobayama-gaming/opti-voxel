#pragma once

#include "archetype_table.hpp"
#include "base_surface.hpp"

#include <functional>
#include <vector>

// Biome features: lava lakes and floating islands, placed on the ground through the instance table like the trees
// (plain records, no change to the IR, the fold or the meshers). Where they stand comes from biome cells: each feature
// kind tiles space with cubic cells of 2^k LOD0 voxels (its cell size in metres rounded to a power of two voxels), and a
// cell holds at most one site: a hashed point of the cell projected onto the base surface (BaseSurface::on_base), kept
// by the cell's hash with the kind's chance and when the projection stays in the cell (so each point of the base has one
// cell; on the flat base, the cells the datum crosses: one per column). A site depends only on its cell, the base and
// the voxel size: the same from any box, any origin (int64 cells, ExactNoise::hash_cell). A feature is built in its
// site's tangent frame (up_rotation of the base's up): on a planet as on the flat world.
namespace Biome {

enum Feature {
    LAVA_LAKE,
    FLOATING_ISLAND,
    FEATURE_COUNT,
};

// A kind's cells and sizes (metres).
struct Kind {
    const char *name;
    double cell_m;
    double chance; // a cell holds a site with this probability
    double radius_lo_m; // the feature's radius, uniform in [lo, hi)
    double radius_hi_m;
    uint32_t seed;
};
const Kind &kind(Feature p_feature);

struct Site {
    Feature feature = LAVA_LAKE;
    VoxelPos at; // on the base surface, LOD0 voxels
    double radius = 0.0; // voxels
    uint32_t seed = 0;
};

// log2 of the kind's cell side in voxels (at least 4).
int cell_shift(Feature p_feature, double p_voxel);
// The sites of p_feature in the box [p_lo, p_hi] (LOD0 voxels, inclusive), cell by cell (z, y, then x). A site's hashed
// point stays in the middle half of its cell: on the flat base two sites of a kind are at least half a cell apart.
void sites(Feature p_feature, const VoxelPos &p_lo, const VoxelPos &p_hi, const BaseSurface &p_base, double p_voxel, std::vector<Site> &r_out);
// Another spot for p_site in the middle half of its cell (hashed from its seed and p_attempt > 0; 0 keeps it) projected
// onto the base, the rest the same: place() tries LAKE_ATTEMPTS of them for a lake before giving up on a slope.
Site moved(const Site &p_site, int p_attempt, const BaseSurface &p_base, double p_voxel);
inline constexpr int LAKE_ATTEMPTS = 4;

// The library types the features are made of.
struct Types {
    int basin = -1;
    int cylinder = -1;
    int island = -1;
};
inline constexpr uint32_t BASIN_SEED = 0xBA51u;
inline constexpr uint32_t ISLAND_SEED = 0x15A1u;
// Appends them to r_library (SDFTable::build_table, the tests); their bumps are on a 16 m lattice.
Types add_types(Archetype::DefLibrary &r_library, double p_voxel);

inline constexpr double LAKE_KEEP_OUT = 1.5;
// A lake's lava surface: the cylinder of the basin's radius `radius` (voxels) along `up` through `top` (the slab's top
// face's centre, LOD0 voxels), the lava at and below it.
struct LavaPool {
    VoxelPos top;
    UpAxis up = UP_Y;
    double radius = 0.0;
    // Whether p_voxel is within LAKE_KEEP_OUT radii of the cylinder's axis: no tree or rock there.
    bool near(const VoxelPos &p_voxel) const;
};

// The ground a feature stands on (SDFTable: the cascade's): its base (the up, the tangent frame) and the top solid voxel
// on the line through a point along the base's up there (SDFTable::ground_under). Thread-safe.
struct Ground {
    const BaseSurface *base = nullptr;
    std::function<VoxelPos(const VoxelPos &)> under;
};

// A placed feature: its records, where it ended up and its top (the lava's centre, or the island's rim plane's).
struct Placed {
    std::vector<Archetype::TableRecord> records;
    Site at;
    VoxelPos top;
    bool lava = false; // pool is set
    LavaPool pool;
};

// A lava lake: two records, the basin subtracted from the ground, then the lava slab unioned into it. The lava is
// flat, 4 % of the radius (at least 3 voxels) below the lowest ground on the slab's rim; the basin clears the ground
// from below the lava up to above the rim's highest ground. It tries LAKE_ATTEMPTS spots of the site's cell and is
// false where the rim's ground spans more than half the radius at every one (a slope: the basin would cut a cliff).
// A floating island: one record, its underside's apex a hashed 25-70 m above the highest ground under it.
// Heights are along the site's up, the records turned by its up_rotation (the identity on the flat base).
bool place(const Site &p_site, const Ground &p_ground, const Archetype::DefLibrary &p_library, const Types &p_types, double p_voxel, Placed &r_out);

} // namespace Biome
