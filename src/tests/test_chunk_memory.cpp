#include "test_common.hpp"

#include "chunk_memory.hpp"

#include <vector>

namespace {

constexpr int MAX_LOD = 5;

const ChunkId PARENT(2, 2, 2); // LOD 1

std::vector<ChunkId> children_of(const ChunkId &p_parent) {
    const int64_t h = ChunkMath::get_lod_size(p_parent) / 2;
    std::vector<ChunkId> out;
    for (int i = 0; i < 8; i++) {
        out.push_back(p_parent + ChunkId((i & 1) ? h : -h, (i & 2) ? h : -h, (i & 4) ? h : -h));
    }
    return out;
}

void scan(VoxelState &r_state, const std::vector<ChunkId> &p_ids) {
    std::unique_ptr<LODHashSet> scanned = std::make_unique<LODHashSet>();
    for (const ChunkId &id : p_ids) {
        scanned->insert(id);
    }
    r_state.add_scan_chunk(std::move(scanned));
}

// A chunk as a mesher hands it back. No RIDs and no owner, so free() has nothing to release.
void load(VoxelState &r_state, const std::vector<ChunkId> &p_ids) {
    for (const ChunkId &id : p_ids) {
        ChunkOutput out;
        out.chunk_id = id;
        r_state.add_new_loaded(out);
    }
}

VoxelState make_state() {
    VoxelState state;
    state.set_max_lod(MAX_LOD);
    return state;
}

} // namespace

TEST_CASE("VoxelState: a scan marks the chunks not loaded yet as dirty") {
    VoxelState state = make_state();
    load(state, { ChunkId(1, 1, 1) });
    scan(state, { ChunkId(1, 1, 1), ChunkId(3, 1, 1) });
    CHECK(state.dirty_chunk.size() == 1);
    CHECK(state.dirty_chunk.has(ChunkId(3, 1, 1)));
}

TEST_CASE("VoxelState: a parent stays until all its children are loaded (split)") {
    VoxelState state = make_state();
    load(state, { PARENT });
    const std::vector<ChunkId> children = children_of(PARENT);
    scan(state, children);

    state.update_destroy_from_scan();
    CHECK_FALSE(state.to_destroy_chunk.has(PARENT));

    load(state, std::vector<ChunkId>(children.begin(), children.end() - 1));
    state.update_destroy_from_scan();
    CHECK_FALSE(state.to_destroy_chunk.has(PARENT)); // one child still missing

    load(state, { children.back() });
    state.update_destroy_from_scan();
    CHECK(state.to_destroy_chunk.has(PARENT));
    CHECK(state.to_destroy_chunk.size() == 1);
}

TEST_CASE("VoxelState: children stay until their parent is loaded (merge)") {
    VoxelState state = make_state();
    const std::vector<ChunkId> children = children_of(PARENT);
    load(state, children);
    scan(state, { PARENT });

    state.update_destroy_from_scan();
    CHECK(state.to_destroy_chunk.size() == 0);

    load(state, { PARENT });
    state.update_destroy_from_scan();
    CHECK(state.to_destroy_chunk.size() == 8);
    for (const ChunkId &child : children) {
        CHECK(state.to_destroy_chunk.has(child));
    }
}

TEST_CASE("VoxelState: an unrelated chunk outside the scan is destroyed at once") {
    VoxelState state = make_state();
    load(state, { ChunkId(1, 1, 1), ChunkId(-31, 1, 1) });
    scan(state, { ChunkId(1, 1, 1) });
    state.update_destroy_from_scan();
    CHECK(state.to_destroy_chunk.size() == 1);
    CHECK(state.to_destroy_chunk.has(ChunkId(-31, 1, 1)));
}
