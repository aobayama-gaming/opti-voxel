#include "forest_generator.hpp"

#include "parallel_for.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <cmath>

namespace Archetype {

int TreeClasses::size_bucket(double p_scale) {
    return CLAMP(int(std::floor((p_scale - TREE_SCALE_MIN) / TREE_SCALE_SPAN * double(SIZE_BUCKETS))), 0, SIZE_BUCKETS - 1);
}

TreeClasses::TreeClasses(const DefLibrary &p_library, const TreeTypes &p_types, double p_voxel) {
    const int species = int(tree_species().size());
    // Hashed seeds filed by size bucket until every bucket has SAMPLES (the scale alone, no tree grown yet).
    std::vector<uint32_t> seeds;
    for (int sp = 0; sp < species; sp++) {
        std::vector<int> count(SIZE_BUCKETS, 0);
        int filled = 0;
        for (uint32_t i = 0; filled < SIZE_BUCKETS && i < (1u << 20); i++) {
            const uint32_t seed = ExactNoise::hash3(uint32_t(sp), i, 0xC1A5u).x;
            const int b = size_bucket(tree_scale(seed));
            if (count[b] < SAMPLES) {
                seeds.push_back(seed);
                filled += ++count[b] == SAMPLES;
            }
        }
        ERR_FAIL_COND_MSG(filled < SIZE_BUCKETS, "TreeClasses: a size bucket that tree_scale never reaches.");
    }
    // The samples grown on the pool, summed in seed order (the same bits).
    std::vector<MaterialMoments> grown(seeds.size());
    parallel_for(uint32_t(seeds.size()), "TreeClasses", [&](uint32_t p_i) {
        const int sp = int(p_i / uint32_t(SIZE_BUCKETS * SAMPLES));
        std::vector<TableRecord> parts;
        grow_tree(tree_species()[sp], VoxelPos(), UP_Y, seeds[p_i], p_voxel, p_types, parts);
        grown[p_i] = span_moments(p_library, parts.data(), uint32_t(parts.size()), VoxelPos());
    });
    classes.assign(size_t(species * SIZE_BUCKETS), MaterialMoments());
    for (size_t i = 0; i < seeds.size(); i++) {
        const int sp = int(i / size_t(SIZE_BUCKETS * SAMPLES));
        const double scale = tree_scale(seeds[i]);
        MaterialMoments &mean = classes[sp * SIZE_BUCKETS + size_bucket(scale)];
        for (const std::pair<uint32_t, Moments> &m : grown[i].parts) {
            mean.add(m.first, m.second.resized(1.0 / scale).scaled(1.0 / double(SAMPLES)));
        }
    }
}

const MaterialMoments &TreeClasses::mean(int p_species, int p_bucket) const {
    static const MaterialMoments none;
    ERR_FAIL_COND_V_MSG(p_species < 0 || p_bucket < 0 || p_bucket >= SIZE_BUCKETS || size_t(p_species * SIZE_BUCKETS + p_bucket) >= classes.size(), none,
            "TreeClasses: no such class (not built, or an unknown species).");
    return classes[p_species * SIZE_BUCKETS + p_bucket];
}

MaterialMoments TreeClasses::tree(const TreeSite &p_site, const UpAxis &p_up) const {
    float rows[9];
    up_rotation(p_up, rows);
    const double scale = tree_scale(p_site.seed);
    MaterialMoments out;
    for (const std::pair<uint32_t, Moments> &m : mean(p_site.species, size_bucket(scale)).parts) {
        out.add(m.first, m.second.resized(scale).rotated(rows));
    }
    return out;
}

void ForestGenerator::trees(const VoxelBox &p_query, const FieldFn &p_ground, std::vector<CellSite> &r_out) const {
    const VoxelBox box = p_query.clipped(zone);
    if (box.valid) {
        forest_sites(box.lo, box.hi, side, p_ground, r_out, &zone);
    }
}

} // namespace Archetype
