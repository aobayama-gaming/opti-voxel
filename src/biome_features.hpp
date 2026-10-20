#pragma once

#include "archetype_table.hpp"

#include <vector>

// Biome features: lava lakes and floating islands, placed on the ground through the instance table like the trees
// (plain records, no change to the IR, the fold or the meshers). Where they stand comes from biome cells: each feature
// kind tiles the plane with square cells of 2^k LOD0 voxels (its cell size in metres rounded to a power of two
// voxels), and a cell holds at most one site, kept by the cell's hash with the kind's chance. A site depends only on its
// cell and the voxel size: the same from any square, any origin (int64 cells, ExactNoise::hash_cell).
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
    int64_t x = 0; // LOD0 voxels
    int64_t z = 0;
    double radius = 0.0; // voxels
    uint32_t seed = 0;
};

// log2 of the kind's cell side in voxels (at least 4).
int cell_shift(Feature p_feature, double p_voxel);
// The sites of p_feature whose centre is in the square [p_x0, p_x1] x [p_z0, p_z1] (LOD0 voxels), cell by cell (z then
// x). A site stays in the middle half of its cell: two sites of a kind are at least half a cell apart.
void sites(Feature p_feature, int64_t p_x0, int64_t p_z0, int64_t p_x1, int64_t p_z1, double p_voxel, std::vector<Site> &r_out);
// Another spot for p_site in the middle half of its cell (hashed from its seed and p_attempt > 0; 0 keeps it), the
// rest the same: place() tries LAKE_ATTEMPTS of them for a lake before giving up on a slope.
Site moved(const Site &p_site, int p_attempt, double p_voxel);
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

// A lake's lava surface, for the page shader's colour: the cylinder of the basin's radius `radius` (voxels) around
// (x, z), the lava at and below y (LOD0 voxels, the slab's top). Trees and rocks keep LAKE_KEEP_OUT radii away.
struct LavaPool {
    VoxelPos top;
    double radius = 0.0;
};
inline constexpr double LAKE_KEEP_OUT = 1.5;

// A placed feature: its records, where it ended up and its top (the lava, or the island's rim plane).
struct Placed {
    std::vector<Archetype::TableRecord> records;
    Site at;
    int64_t top_y = 0;
    bool lava = false; // pool is set
    LavaPool pool;
};

// A lava lake: two records, the basin subtracted from the ground, then the lava slab unioned into it. The lava is
// flat, 4 % of the radius (at least 3 voxels) below the lowest ground on the slab's rim; the basin clears the ground
// from below the lava up to above the rim's highest ground. It tries LAKE_ATTEMPTS spots of the site's cell and is
// false where the rim's ground spans more than half the radius at every one (a slope: the basin would cut a cliff).
// A floating island: one record, its underside's apex a hashed 25-70 m above the highest ground under it.
bool place(const Site &p_site, const Archetype::GroundFn &p_ground_y, const Archetype::DefLibrary &p_library, const Types &p_types, double p_voxel,
        Placed &r_out);

} // namespace Biome
