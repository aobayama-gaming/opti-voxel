#include "test_common.hpp"

#include "chunk_discoverer.hpp"
#include "sdf_archetype.hpp"
#include "sdf_cascade.hpp"

#include <cmath>
#include <memory>

namespace {

// The discoverer's walk before the wave walk (recursive, depth first, one SDF call per corner with an early
// exit on the first near corner), kept as the reference the wave walk must reproduce.
struct ReferenceWalk {
    const ChunkParam &param;
    const SDFBase &sdf;
    const PackedFloat32Array &lod_distances;
    int64_t nodes_visited = 0;
    int64_t culled = 0;

    bool outside(const ChunkId &p_chunk) const {
        const float chunk_outer_radius = ChunkMath::world_chunk_size(p_chunk, param) * 2;
        const VoxelPos center = ChunkMath::chunk_center_voxel(p_chunk, param);
        const int64_t half_voxels = ChunkMath::get_lod_size(p_chunk) * param.half_chunk_voxels();
        const int lod = ChunkMath::get_lod(p_chunk);
        for (int corner = 0; corner < 8; ++corner) {
            WorldPos corner_pos;
            corner_pos.voxel = center + VoxelPos((corner & 1) ? half_voxels : -half_voxels, (corner & 2) ? half_voxels : -half_voxels, (corner & 4) ? half_voxels : -half_voxels);
            if (std::abs(sdf.evaluate(corner_pos, lod)) <= chunk_outer_radius) {
                return false;
            }
        }
        return true;
    }

    void walk(const ChunkId &p_camera, const ChunkId &p_node, LODHashSet &r_set) {
        const int lod = ChunkMath::get_lod(p_node);
        nodes_visited++;
        if (outside(p_node)) {
            culled++;
            return;
        }
        if (lod == 0) {
            r_set.insert(p_node);
            return;
        }
        const double distance = p_camera.distance_to(p_node);
        const float radius = std::sqrt(3.0f) * static_cast<float>(int64_t(1) << lod);
        if (distance - radius < lod_distances[lod]) {
            const int64_t child = int64_t(1) << (lod - 1);
            for (int i = 0; i < 8; ++i) {
                walk(p_camera, p_node + ChunkId((i & 1) ? child : -child, (i & 2) ? child : -child, (i & 4) ? child : -child), r_set);
            }
        } else {
            r_set.insert(p_node);
        }
    }

    void scan(const ChunkDiscoverer &p_discoverer, const ChunkId &p_camera, LODHashSet &r_set) {
        int64_t step;
        int64_t count;
        p_discoverer.get_scan_grid(step, count);
        const ChunkId ancestor = ChunkMath::get_parent_from_child_until(p_camera, p_discoverer.get_max_lod());
        const ChunkId span = ChunkId(count, count, count) * step;
        for (int64_t x = ancestor.x - span.x; x <= ancestor.x + span.x; x += step) {
            for (int64_t y = ancestor.y - span.y; y <= ancestor.y + span.y; y += step) {
                for (int64_t z = ancestor.z - span.z; z <= ancestor.z + span.z; z += step) {
                    walk(p_camera, ChunkId(x, y, z), r_set);
                }
            }
        }
    }
};

int64_t set_difference(const LODHashSet &p_a, const LODHashSet &p_b) {
    int64_t n = 0;
    for (const ChunkId &id : p_a) {
        n += p_b.has(id) ? 0 : 1;
    }
    return n;
}

// Both walks around three camera poses (near the dummy's ground, the 15- and 25-LOD cascade grounds at
// ~3 and ~6 km), same sets and the same node counts; the wave walk again with a cache (cold, then warm).
void compare_walks(SDFBase &p_sdf, int p_lods) {
    ChunkParam param;
    p_sdf.configure(param, p_lods);
    ChunkDiscoverer *discoverer = memnew(ChunkDiscoverer);
    discoverer->set_config(param, &p_sdf);
    // The bench's 1280x720 view at lod_error 4 (a coarser tree than the default 2, to keep the test short).
    discoverer->setup_lod(p_lods, 4.0f, 720.0f, 75.0f * (3.14159265f / 180.0f));
    const Vector3 poses[] = { Vector3(-52.76f, 330.0f, -95.64f), Vector3(3000.0f, 3200.0f, 2000.0f), Vector3(-800.0f, 6500.0f, 400.0f) };
    for (const Vector3 &pose : poses) {
        CAPTURE(pose);
        const ChunkId camera = ChunkMath::chunk_at_render(pose, param);
        ReferenceWalk reference{ param, p_sdf, discoverer->get_lod_distances() };
        LODHashSet want;
        reference.scan(*discoverer, camera, want);

        LODHashSet got;
        ChunkDiscoverer::ScanCounters counters;
        discoverer->scan(camera, ChunkDiscoverer::CULL_LEGACY, got, counters);
        CHECK(want.size() > 0);
        CHECK(got.size() == want.size());
        CHECK(set_difference(got, want) == 0);
        CHECK(set_difference(want, got) == 0);
        CHECK(counters.nodes_visited == reference.nodes_visited);
        CHECK(counters.culled == reference.culled);
        // One wave per LOD from the top, down to the last LOD a node was split to.
        CHECK(!counters.waves.empty());
        CHECK(counters.waves.size() <= size_t(discoverer->get_max_lod() + 1));
        for (size_t w = 0; w < counters.waves.size(); w++) {
            CHECK(counters.waves[w].lod == discoverer->get_max_lod() - int(w));
        }
        CHECK(counters.sdf_calls < counters.corners); // shared corners are evaluated once

        ChunkDiscoverer::CullCache cache;
        cache.verify = true;
        for (int pass = 0; pass < 2; pass++) {
            cache.scan_index++;
            LODHashSet cached;
            ChunkDiscoverer::ScanCounters c;
            discoverer->scan(camera, ChunkDiscoverer::CULL_LEGACY, cached, c, &cache);
            CHECK(set_difference(cached, want) == 0);
            CHECK(cached.size() == want.size());
            CHECK(c.verify_mismatches == 0);
            CHECK(c.cache_hits == (pass == 0 ? 0 : c.nodes_visited));
        }
    }
    memdelete(discoverer);
}

} // namespace

TEST_CASE("ChunkDiscoverer: the wave walk gives the recursive walk's set (SDFArchetype, 15 and 25 LODs)") {
    for (const int lods : { 15, 25 }) {
        CAPTURE(lods);
        SDFArchetype sdf;
        compare_walks(sdf, lods);
    }
}

TEST_CASE("ChunkDiscoverer: the wave walk gives the recursive walk's set (SDFCascade, 15 and 25 LODs)") {
    for (const int lods : { 15, 25 }) {
        CAPTURE(lods);
        SDFCascade sdf;
        compare_walks(sdf, lods);
    }
}
