#include "biome_pick.hpp"

#include "exact_noise.hpp"
#include "ov/math.hpp"

#include <algorithm>

BiomePick::BiomePick(const Climate::Table *p_table, const Cascade::Spec *p_spec) :
        table(p_table && p_spec && p_table->biomes.size() > 1 ? p_table : nullptr), spec(p_spec) {
    if (table == nullptr) {
        return;
    }
    // The finest band of a field some biome's place names: coarser lattices read the same weights (no finer octave of
    // theirs), finer ones would cost levels for nothing.
    lattice = Cascade::MAX_LEVEL;
    for (const Climate::Biome &b : table->biomes) {
        for (const Climate::Axis &a : b.at) {
            for (const Cascade::Band &band : table->fields[size_t(a.field)].bands) {
                lattice = ov::min(lattice, spec->insert_level(band));
            }
        }
    }
    lattice = ov::max(lattice, 0);
}

void BiomePick::weights(const VoxelPos &p_voxel, float r_w[Climate::MAX_BIOMES]) const {
    std::fill(r_w, r_w + Climate::MAX_BIOMES, 0.0f);
    if (table == nullptr) {
        r_w[0] = 1.0f;
        return;
    }
    const Vector3i64 p = WorldCoord::floor_div(p_voxel, int64_t(1) << lattice);
    ExactNoise::Float4 fields[Climate::MAX_FIELDS], w[Climate::MAX_BIOMES];
    Climate::fields_at(*table, *spec, p, lattice, fields);
    Climate::weights(*table, fields, w);
    for (int b = 0; b < Climate::MAX_BIOMES; b++) {
        r_w[b] = w[b].w;
    }
}

int BiomePick::at(const VoxelPos &p_voxel, uint32_t p_seed) const {
    if (table == nullptr) {
        return 0;
    }
    float w[Climate::MAX_BIOMES];
    weights(p_voxel, w);
    const double u = ExactNoise::unit_of(ExactNoise::hash3(p_seed, 0xB10Eu, 0x5173u).x);
    double sum = 0.0;
    const int n = biomes();
    for (int b = 0; b < n; b++) {
        sum += double(w[b]);
    }
    double run = 0.0;
    for (int b = 0; b + 1 < n; b++) {
        run += double(w[b]);
        if (u * sum < run) {
            return b;
        }
    }
    return n - 1;
}

int pick_share(const std::vector<uint32_t> &p_shares, uint32_t p_variant) {
    uint64_t total = 0;
    for (const uint32_t s : p_shares) {
        total += s;
    }
    if (total == 0) {
        return -1;
    }
    uint64_t at = uint64_t(p_variant) % total;
    for (size_t i = 0; i < p_shares.size(); i++) {
        if (at < p_shares[i]) {
            return int(i);
        }
        at -= p_shares[i];
    }
    return int(p_shares.size()) - 1;
}
