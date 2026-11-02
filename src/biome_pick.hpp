#pragma once

#include "cascade.hpp"
#include "climate.hpp"
#include "world_coord.hpp"

#include <cstdint>
#include <vector>

// Which biome a site belongs to (the instances' species, the features): the climate's biome weights at the site, read
// once on the CPU (Climate::fields_at, then weights) on the coarsest lattice its placement fields resolve, and a hash of
// the site against them, so where biomes blend each takes about its weight's share of the sites. A pure function of
// the site's voxel and seed: the same from any box, chunk or thread.
struct BiomePick {
    BiomePick() = default;
    // p_table's biomes over p_spec (the combined cascade, Climate::combined_spec); null, or one biome: always 0.
    BiomePick(const Climate::Table *p_table, const Cascade::Spec *p_spec);

    // The biome of a site at p_voxel of seed p_seed.
    int at(const VoxelPos &p_voxel, uint32_t p_seed) const;
    // The biome weights' values there (r_w[b], 0 past the table's biomes).
    void weights(const VoxelPos &p_voxel, float r_w[Climate::MAX_BIOMES]) const;
    int biomes() const { return table ? int(table->biomes.size()) : 1; }
    // The lattice the weights are read on (Λ_level: the finest insertion level of the fields the biomes' places name).
    int level() const { return lattice; }

private:
    const Climate::Table *table = nullptr;
    const Cascade::Spec *spec = nullptr;
    int lattice = 0;
};

// The entry p_variant picks among integer shares (p_variant mod their sum, the entries' runs in order: n equal shares
// are p_variant % n), -1 if they sum to 0.
int pick_share(const std::vector<uint32_t> &p_shares, uint32_t p_variant);
