#include "test_common.hpp"

#include "sdf_cascade.hpp"

using namespace Archetype;
using ExactNoise::float_bits;

namespace {

// A configured cascade terrain with the --instances scatter (rocks and pillars around the demo spot).
struct ScatteredCascade {
    ChunkParam param;
    SDFCascade sdf;
    explicit ScatteredCascade(int p_count, double p_area = 200.0, int p_lods = 15) {
        sdf.set_scatter(p_count, p_area);
        param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, p_lods));
        sdf.configure(param, p_lods);
    }
};

// sample_sdf at a LOD-p_lod lattice point with a given list.
Float4 sample_with(const SDFCascade &p_sdf, const InstanceSnapshot &p_snap, const std::vector<uint32_t> &p_list, const VoxelPos &p_voxel, int p_lod) {
    float ch[Cascade::CHANNELS];
    p_sdf.channels_at(p_voxel, p_lod, ch);
    WorldPos p;
    p.voxel = p_voxel;
    return p_sdf.sample_sdf(p_snap, p_list.data(), uint32_t(p_list.size()), ch, p, p_lod);
}

// The bare cascade at a LOD-p_lod lattice point (metres).
float bare_sdf(const SDFCascade &p_sdf, const VoxelPos &p_voxel, int p_lod) {
    float ch[Cascade::CHANNELS];
    p_sdf.channels_at(p_voxel, p_lod, ch);
    return p_sdf.sdf(ch, p_voxel.y, p_lod).w;
}

using ChunkMath::chunk_at_voxel;

} // namespace

TEST_CASE("SDFCascade table: ground_y is the bare cascade's top solid voxel") {
    ScatteredCascade t(0);
    for (int i = 0; i < 48; i++) {
        const int64_t x = int64_t(i) * 977 - 20000;
        const int64_t z = int64_t(i) * -613 + 5000;
        const int64_t y = t.sdf.ground_y(x, z);
        CHECK(!(bare_sdf(t.sdf, VoxelPos(x, y, z), 0) > 0.0f));
        CHECK(bare_sdf(t.sdf, VoxelPos(x, y + 1, z), 0) > 0.0f);
        CHECK(t.sdf.ground_y(x, z) == y); // deterministic
    }
}

TEST_CASE("SDFCascade table: the scatter stands on the ground, folded at lattice points; far from it the bare cascade, bit for bit") {
    ScatteredCascade t(200);
    const auto snap = t.sdf.get_snapshot();
    CHECK(snap->records.size() == 200);
    std::vector<uint32_t> list;
    int64_t pillars = 0;
    for (uint32_t i = 0; i < snap->records.size(); i++) {
        const TableRecord &rec = snap->records[i];
        if (rec.def != 1) {
            continue; // the rocks are half sunk: solid either way
        }
        // Halfway up a pillar (a capsule on its column's ground): air in the bare cascade, solid with the table.
        WorldPos p;
        p.voxel = rec.anchor + VoxelPos(0, int64_t(0.5f * rec.params[0]), 0);
        CHECK(bare_sdf(t.sdf, p.voxel, 0) > 0.0f);
        CHECK(t.sdf.evaluate(p, 0) < 0.0f);
        snap->point_candidates(p.voxel, 0, list);
        CHECK(std::find(list.begin(), list.end(), i) != list.end());
        pillars++;
    }
    CHECK(pillars == 100);
    // 4 km away, at LODs 0-6: no candidate, the bare sdf().
    for (int lod = 0; lod <= 6; lod++) {
        for (int i = 0; i < 16; i++) {
            WorldPos p;
            const int64_t step = int64_t(1) << lod;
            p.voxel = VoxelPos((8000 + i * 37) * step, t.sdf.ground_y(8000 * step, 0) / step * step, (i * 53 - 400) * step);
            snap->point_candidates(p.voxel, lod, list);
            CHECK(list.empty());
            CHECK(float_bits(t.sdf.evaluate(p, lod)) == float_bits(bare_sdf(t.sdf, p.voxel, lod)));
        }
    }
}

TEST_CASE("SDFCascade table: two neighbour chunks give the same bits at the samples they share, whatever their lists") {
    // The fold only finishes (renormalises) where an instance's box test passes: a pure function of the point, so a
    // seam sample is the same from either chunk even when one of them lists an instance the other doesn't.
    ScatteredCascade t(400, 120.0);
    const auto snap = t.sdf.get_snapshot();
    std::vector<uint32_t> list_a, list_b;
    int64_t compared = 0, different_lists = 0, folded = 0;
    for (int lod = 0; lod <= 4; lod++) {
        const int64_t step = int64_t(1) << lod;
        for (uint32_t r = 0; r < snap->records.size(); r += 9) {
            const TableRecord &rec = snap->records[r];
            const ChunkId a = chunk_at_voxel(rec.anchor, lod, t.param);
            for (int axis = 0; axis < 3; axis++) {
                ChunkId b = a;
                b[axis] += int64_t(2) << lod;
                snap->chunk_list(a, list_a);
                snap->chunk_list(b, list_b);
                different_lists += list_a != list_b ? 1 : 0;
                VoxelPos lo_a, hi_a, lo_b, hi_b;
                chunk_sample_box(a, t.param, lo_a, hi_a);
                chunk_sample_box(b, t.param, lo_b, hi_b);
                VoxelPos lo, hi;
                for (int c = 0; c < 3; c++) {
                    lo[c] = WorldCoord::floor_div(MAX(lo_a[c], lo_b[c]) + step - 1, step) * step;
                    hi[c] = MIN(hi_a[c], hi_b[c]);
                }
                // Every shared lattice point on the seam slab, a coarse grid across it.
                const int64_t grid = 7 * step;
                for (int64_t x = lo.x; x <= hi.x; x += axis == 0 ? step : grid) {
                    for (int64_t y = lo.y; y <= hi.y; y += axis == 1 ? step : grid) {
                        for (int64_t z = lo.z; z <= hi.z; z += axis == 2 ? step : grid) {
                            const VoxelPos v(x, y, z);
                            const Float4 sa = sample_with(t.sdf, *snap, list_a, v, lod);
                            const Float4 sb = sample_with(t.sdf, *snap, list_b, v, lod);
                            CHECK(same_bits(sa, sb));
                            folded += float_bits(sa.w) != float_bits(bare_sdf(t.sdf, v, lod)) ? 1 : 0;
                            compared++;
                        }
                    }
                }
            }
        }
    }
    MESSAGE("seams: ", compared, " shared samples, ", folded, " changed by an instance, ", different_lists, " chunk pairs with different lists");
    CHECK(different_lists > 0);
    CHECK(folded > 0);
}

TEST_CASE("SDFCascade table: runtime edits on the cascade") {
    ScatteredCascade t(0);
    const int64_t x = 1000, z = -3000;
    const VoxelPos anchor(x, t.sdf.ground_y(x, z), z);
    const float params[8] = { 20.0f, 3.0f, 0, 0, 0, 0, 0, 0 };
    WorldPos p;
    p.voxel = anchor + VoxelPos(0, 10, 0);
    CHECK(t.sdf.evaluate(p, 0) > 0.0f);
    const int id = t.sdf.add_instance(1, anchor, nullptr, params, int(Combine::UNION), 0.0f);
    CHECK(id == 0);
    if (id != 0) {
        return;
    }
    CHECK(t.sdf.evaluate(p, 0) < 0.0f);
    CHECK(t.sdf.last_edit_box().valid);
    CHECK(t.sdf.edit_affects_chunk(chunk_at_voxel(p.voxel, 0, t.param)));
    CHECK(!t.sdf.edit_affects_chunk(chunk_at_voxel(p.voxel + VoxelPos(5000, 0, 0), 0, t.param)));
    CHECK(t.sdf.cull_exempt(chunk_at_voxel(p.voxel, 2, t.param), 2));
    CHECK(t.sdf.remove_instance(id));
    CHECK(t.sdf.evaluate(p, 0) > 0.0f);
}
