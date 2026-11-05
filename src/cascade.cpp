#include "cascade.hpp"

#include <algorithm>

namespace Cascade {

namespace {

// p_pos displaced by the warp's values (LOD0 voxels, clamped to WARP_LIMIT): the floor to the voxel, the rest to the
// fraction (below 1: a rest that rounds to 1 carries).
WorldPos warped(const WorldPos &p_pos, const ExactNoise::Float4 p_warp[3]) {
    WorldPos pos = p_pos;
    float *frac[3] = { &pos.frac.x, &pos.frac.y, &pos.frac.z };
    for (int a = 0; a < 3; a++) {
        const float w = std::clamp(p_warp[a].w, -WARP_LIMIT, WARP_LIMIT);
        float fl = std::floor(w);
        float rest = w - fl;
        if (rest >= 1.0f) {
            fl = fl + 1.0f;
            rest = 0.0f;
        }
        pos.voxel[a] += int64_t(fl);
        *frac[a] = rest;
    }
    return pos;
}

} // namespace

ExactNoise::Float4 note_noise(const BaseSurface &p_base, const Band &p_band, int p_m, const Vector3i64 &p_p, const ExactNoise::Float4 *p_warp) {
    using ExactNoise::Float4;
    const int64_t step = int64_t(1) << p_m;
    WorldPos pos;
    pos.voxel = Vector3i64(p_p.x * step, p_p.y * step, p_p.z * step);
    if (p_warp != nullptr) {
        pos = warped(pos, p_warp);
    }
    // A stretched band reads its noise stretched along the base's up (the base sample at its insertion lattice), its
    // gradient back through the stretch (per cell of 2^k voxels).
    const bool stretch = p_band.stretch > 0;
    const Float4 b = stretch ? p_base.stretch_base(pos.voxel, p_m) : Float4{};
    const WorldPos at = stretch ? p_base.stretched(pos, p_band.stretch, b, p_m) : pos;
    Float4 n = ExactNoise::perlin3(at, p_band.k, p_band.seed, p_band.offset);
    n = stretch ? p_base.stretch_gradient(n, p_band.stretch, b) : n;
    if (p_warp == nullptr) {
        return n;
    }
    // The chain rule through the displacement: dn' = (I + J)^T dn, J_ab = dW_a / dx_b per voxel (the warp's gradient per
    // lattice spacing of Λ_m times 2^-m), each product flushed.
    const float to_voxel = ExactNoise::exact_pow2(-p_m);
    const float d[3] = { n.x, n.y, n.z };
    float out[3];
    for (int i = 0; i < 3; i++) {
        float s = 0.0f;
        for (int a = 0; a < 3; a++) {
            const float grad[3] = { p_warp[a].x, p_warp[a].y, p_warp[a].z };
            s = s + flush_value(flush_value(grad[i] * to_voxel) * d[a]);
        }
        out[i] = d[i] + flush_value(s);
    }
    return Float4{ out[0], out[1], out[2], n.w };
}

Box parent_box(const Box &p_child) {
    Box b;
    for (int a = 0; a < 3; a++) {
        b.lo[a] = half_floor(p_child.lo[a]);
        b.hi[a] = half_floor(p_child.hi[a] + 1);
    }
    return b;
}

std::vector<Box> level_boxes(const Box &p_box, int p_lod, int p_top) {
    std::vector<Box> boxes = { p_box };
    for (int m = p_lod + 1; m <= p_top; m++) {
        boxes.push_back(parent_box(boxes.back()));
    }
    return boxes;
}

Box level_box(const Box &p_box, int p_j) {
    Box b;
    const uint64_t mask = (uint64_t(1) << p_j) - 1;
    for (int a = 0; a < 3; a++) {
        b.lo[a] = p_box.lo[a] >> p_j;
        b.hi[a] = (p_box.hi[a] >> p_j) + ((uint64_t(p_box.hi[a]) & mask) != 0 ? 1 : 0);
    }
    return b;
}

int level_bound(int p_size, int p_j) {
    int n = p_size;
    for (int j = 0; j < p_j; j++) {
        n = (n + 1) / 2 + 1;
    }
    return n;
}

Box chunk_box(const ChunkId &p_chunk, const ChunkParam &p_param) {
    // The anchor is a multiple of 28 * 2^L: the shift is exact.
    return item_box(ChunkMath::chunk_min_voxel(p_chunk, p_param), ChunkMath::get_lod(p_chunk), p_param);
}

Box item_box(const VoxelPos &p_anchor, int p_lod, const ChunkParam &p_param) {
    const int64_t a[3] = { p_anchor.x >> p_lod, p_anchor.y >> p_lod, p_anchor.z >> p_lod };
    Box b;
    for (int i = 0; i < 3; i++) {
        b.lo[i] = a[i] - p_param.skirt_size;
        b.hi[i] = b.lo[i] + p_param.chunk_size - 1;
    }
    return b;
}

ChunkId random_chunk(int p_lod, uint32_t p_seed, int p_index) {
    using ExactNoise::pcg3d;
    using ExactNoise::UVec3;
    const UVec3 h = pcg3d(UVec3{ p_seed, uint32_t(p_index), uint32_t(p_lod) });
    const UVec3 h2 = pcg3d(h);
    const int64_t range = int64_t(1) << (39 - p_lod);
    auto axis = [range, p_lod](uint32_t p_a, uint32_t p_b) {
        const int64_t n = int64_t(((uint64_t(p_a) << 32) | p_b) % uint64_t(range)) - range / 2;
        return (n * 2 + 1) * (int64_t(1) << p_lod);
    };
    return ChunkId(axis(h.x, h2.x), axis(h.y, h2.y), axis(h.z, h2.z));
}

} // namespace Cascade