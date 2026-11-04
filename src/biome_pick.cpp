#include "biome_pick.hpp"

#include "exact_noise.hpp"
#include "ov/math.hpp"

#include <algorithm>

BiomePick::BiomePick(const Climate::Table *p_table, const Terrain::LevelProgram *p_prog) :
        table(p_table && p_prog && p_table->biomes.size() > 1 ? p_table : nullptr), prog(p_prog) {
    if (table == nullptr) {
        return;
    }
    lattice = Cascade::MAX_LEVEL;
    for (const int w : prog->weights) {
        lattice = ov::min(lattice, prog->nodes[size_t(w)].level);
    }
    lattice = ov::clamp(lattice, 0, Cascade::MAX_LEVEL);
}

void BiomePick::weights(const VoxelPos &p_voxel, float r_w[Climate::MAX_BIOMES]) const {
    std::fill(r_w, r_w + Climate::MAX_BIOMES, 0.0f);
    if (table == nullptr) {
        r_w[0] = 1.0f;
        return;
    }
    const Vector3i64 p = WorldCoord::floor_div(p_voxel, int64_t(1) << lattice);
    ExactNoise::Float4 w[Climate::MAX_BIOMES];
    Terrain::point_nodes(*prog, p, lattice, prog->weights.data(), int(prog->weights.size()), w);
    for (size_t b = 0; b < prog->weights.size(); b++) {
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
