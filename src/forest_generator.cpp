#include "forest_generator.hpp"

#include <algorithm>

namespace Archetype {

int TreeClasses::size_bucket(double p_scale) {
    return CLAMP(int(std::floor((p_scale - 0.8) / 0.4 * double(SIZE_BUCKETS))), 0, SIZE_BUCKETS - 1);
}

TreeClasses::TreeClasses(const DefLibrary &p_library, const TreeTypes &p_types, double p_voxel) {
    const double zero[3] = { 0.0, 0.0, 0.0 };
    for (int sp = 0; sp < int(tree_species().size()); sp++) {
        std::vector<MaterialMoments> means(SIZE_BUCKETS);
        std::vector<int> count(SIZE_BUCKETS, 0);
        int filled = 0;
        // Hashed seeds, each filed in its size bucket, until every bucket has SAMPLES.
        for (uint32_t i = 0; filled < SIZE_BUCKETS; i++) {
            const uint32_t seed = ExactNoise::hash3(uint32_t(sp), i, 0xC1A5u).x;
            const double scale = tree_scale(seed);
            const int b = size_bucket(scale);
            if (count[b] == SAMPLES) {
                continue;
            }
            filled += ++count[b] == SAMPLES;
            std::vector<TableRecord> parts;
            grow_tree(tree_species()[sp], VoxelPos(), UP_Y, seed, p_voxel, p_types, parts);
            const MaterialMoments mm = span_moments(p_library, parts.data(), uint32_t(parts.size()), VoxelPos());
            for (const std::pair<uint32_t, Moments> &m : mm.parts) {
                means[b].add(m.first, m.second.resized(1.0 / scale).scaled(1.0 / double(SAMPLES)), zero);
            }
        }
        classes.insert(classes.end(), means.begin(), means.end());
    }
}

MaterialMoments TreeClasses::tree(const TreeSite &p_site, const UpAxis &p_up) const {
    float rows[9];
    up_rotation(p_up, rows);
    const double scale = tree_scale(p_site.seed);
    const double zero[3] = { 0.0, 0.0, 0.0 };
    MaterialMoments out;
    for (const std::pair<uint32_t, Moments> &m : mean(p_site.species, size_bucket(scale)).parts) {
        out.add(m.first, m.second.resized(scale).rotated(rows), zero);
    }
    return out;
}

void ForestGenerator::trees(const VoxelBox &p_query, const FieldFn &p_ground, std::vector<CellSite> &r_out) const {
    if (side <= 0 || !p_query.intersects(zone)) {
        return;
    }
    VoxelBox box;
    box.valid = true;
    for (int a = 0; a < 3; a++) {
        box.lo[a] = MAX(p_query.lo[a], zone.lo[a]);
        box.hi[a] = MIN(p_query.hi[a], zone.hi[a]);
    }
    forest_sites(box.lo, box.hi, side, p_ground, r_out);
}

} // namespace Archetype
