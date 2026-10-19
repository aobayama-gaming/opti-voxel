#include "test_common.hpp"

#include "occluder_set.hpp"

#include <vector>

// The merged boxes are disjoint and cover exactly the solid blocks, on random grids of several densities.
TEST_CASE("OccluderSet: merge_blocks covers the solid blocks exactly") {
    uint64_t state = 0x5EEDu;
    const auto next = [&]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return uint32_t(state >> 33);
    };
    for (int n : { 1, 4, 7, 14 }) {
        for (int density : { 0, 30, 70, 95, 100 }) {
            std::vector<uint8_t> solid(size_t(n) * n * n);
            for (uint8_t &s : solid) {
                s = int(next() % 100) < density ? 1 : 0;
            }
            std::vector<OccluderSet::BlockBox> boxes;
            OccluderSet::merge_blocks(solid, n, boxes);
            std::vector<int> cover(solid.size(), 0);
            for (const OccluderSet::BlockBox &b : boxes) {
                for (int z = b.z0; z < b.z1; z++) {
                    for (int y = b.y0; y < b.y1; y++) {
                        for (int x = b.x0; x < b.x1; x++) {
                            cover[(size_t(z) * n + y) * n + x]++;
                        }
                    }
                }
            }
            int wrong = 0;
            for (size_t i = 0; i < solid.size(); i++) {
                wrong += cover[i] != int(solid[i]) ? 1 : 0;
            }
            CAPTURE(n);
            CAPTURE(density);
            CHECK(wrong == 0);
            if (density == 100) {
                CHECK(boxes.size() == 1);
            }
        }
    }
}

// A wall in front of the eye hides a chunk behind it; shrunk by a margin wider than it, it is gone.
TEST_CASE("OccluderSet: build") {
    OccluderSet set;
    const VoxelPos base(1000, 2000, 3000);
    set.set(ChunkId(1, 1, 1), { { base + VoxelPos(40, -50, -50), base + VoxelPos(44, 50, 50) } });
    set.set(ChunkId(3, 1, 1), { { base + VoxelPos(-2, -2, -2), base + VoxelPos(2, 2, 2) } }); // holds the eye
    CHECK(set.box_count() == 2);
    OcclusionMap map;
    OccluderSet::BuildStats s = set.build(map, base, { 0.5, 0.5, 0.5 }, 0.0);
    CHECK(s.boxes == 2);
    CHECK(s.holding_eye == 1);
    CHECK(s.written == 1);
    CHECK(map.box_hidden({ 100, -5, -5 }, { 110, 5, 5 }));
    CHECK_FALSE(map.box_hidden({ -110, -5, -5 }, { -100, 5, 5 }));

    s = set.build(map, base, { 0.5, 0.5, 0.5 }, 3.0);
    CHECK(s.vanished == 2); // both are 4 voxels thick
    CHECK_FALSE(map.box_hidden({ 100, -5, -5 }, { 110, 5, 5 }));

    set.erase(ChunkId(1, 1, 1));
    set.set(ChunkId(3, 1, 1), {});
    CHECK(set.box_count() == 0);
}
