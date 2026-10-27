#include "test_common.hpp"

#include "chunk_gate.hpp"
#include "sdf_cascade.hpp"

#include <cmath>
#include <vector>

namespace {

// A mesher's queue as ChunkGate::next pops it: first meshes of load queue 0, popped from the back.
struct Queue {
    std::vector<ChunkId> chunks;
    std::vector<ChunkId> requeued;
    bool pop(ChunkId &r_chunk, ChunkGate::Kind &r_kind, int &r_queue) {
        if (chunks.empty()) {
            return false;
        }
        r_chunk = chunks.back();
        chunks.pop_back();
        r_kind = ChunkGate::CHUNK;
        r_queue = 0;
        return true;
    }
};

// Pops through the gate until it says no: the chunks answered READY.
std::vector<ChunkId> drain_gate(ChunkGate &r_gate, Queue &r_queue) {
    std::vector<ChunkId> ready;
    ChunkId chunk;
    ChunkGate::Kind kind = ChunkGate::CHUNK;
    const auto pop = [&r_queue](ChunkId &r_c, ChunkGate::Kind &r_k, int &r_q) { return r_queue.pop(r_c, r_k, r_q); };
    const auto requeue = [&r_queue](const ChunkId &p_c, ChunkGate::Kind) { r_queue.requeued.push_back(p_c); };
    while (r_gate.next(chunk, kind, pop, requeue)) {
        ready.push_back(chunk);
    }
    return ready;
}

// LOD-0 chunks along x through p_centre, p_step voxels apart.
std::vector<ChunkId> row(const VoxelPos &p_centre, int p_count, int64_t p_step, const ChunkParam &p_param) {
    std::vector<ChunkId> out;
    for (int i = 0; i < p_count; i++) {
        out.push_back(ChunkMath::chunk_at_voxel(p_centre + VoxelPos((i - p_count / 2) * p_step, 0, 0), 0, p_param));
    }
    return out;
}

} // namespace

TEST_CASE("ChunkGate: the admit cap counts the chunks that ask for keys, never the READY ones") {
    ChunkParam param;
    SDFCascade sdf; // no streamed source: every chunk READY
    sdf.configure(param, 15);
    ChunkGate gate;
    gate.bind(&sdf, &gate);
    Queue queue;
    queue.chunks = row(VoxelPos(), 16, 28, param);
    gate.begin_frame(2);
    CHECK(drain_gate(gate, queue).size() == 16);
    CHECK(gate.capped_frames == 0);
}

TEST_CASE("ChunkGate: a frame's popping stops at the cap of parked chunks; next_scan keeps the parked chunks the scan holds") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.set_forest(30, 40.0);
    param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 18));
    sdf.configure(param, 18);
    CHECK(sdf.get_stream() != nullptr);
    const int64_t x = std::llround(SDFTable::SCENE_X / 0.05), z = std::llround(SDFTable::SCENE_Z / 0.05);
    const VoxelPos centre(x, sdf.ground_at(x, z).y, z);
    ChunkGate gate;
    gate.bind(&sdf, &gate);
    Queue queue;
    queue.chunks = row(centre, 8, 56, param); // inside the forest's zone: each lists keys not computed yet

    gate.begin_frame(3);
    CHECK(drain_gate(gate, queue).empty());
    CHECK(gate.capped_frames == 1);
    CHECK(gate.parked == 3);
    CHECK(queue.chunks.size() == 5); // the rest waits for the next frame
    CHECK_FALSE(gate.idle(ChunkGate::CHUNK));

    // A new scan holding one of them: it stays parked (not to queue again), the other two are forgotten.
    LODHashSet scan;
    const ChunkId still = row(centre, 8, 56, param)[7];
    scan.insert(still);
    gate.next_scan(scan, [](const ChunkId &) { return 0; });
    CHECK(gate.keeps(still));
    CHECK(sdf.parked_count(uint64_t(uintptr_t(&gate)) + ChunkGate::CHUNK) == 1);
}
