#include "test_common.hpp"

#include "lod_structures.hpp"

namespace {

void add(LODHashSet &r_set, const ChunkId &p_id) {
    r_set.insert(p_id);
}

void add(ChunkHashMap &r_map, const ChunkId &p_id) {
    ChunkOutput out;
    out.chunk_id = p_id;
    r_map.insert(p_id, out);
}

} // namespace

// VoxelState uses both containers to find the nearest loaded / pending ancestor of a chunk.
TEST_CASE_TEMPLATE("LOD sets: get_max_lod finds the nearest ancestor", T, LODHashSet, ChunkHashMap) {
    T set;
    set.max_lod = 5;
    add(set, ChunkId(4, 4, 4)); // LOD 2
    add(set, ChunkId(8, 8, 8)); // LOD 3, further up the same branch

    ChunkId id(1, 1, 1);
    CHECK(set.get_max_lod(id) == 2);
    CHECK(id == ChunkId(4, 4, 4)); // rewritten to the ancestor

    ChunkId lod1(2, 2, 2);
    CHECK(set.get_max_lod(lod1) == 2);
    CHECK(lod1 == ChunkId(4, 4, 4));
}

TEST_CASE_TEMPLATE("LOD sets: get_max_lod ignores the chunk itself and other branches", T, LODHashSet, ChunkHashMap) {
    T set;
    set.max_lod = 5;
    add(set, ChunkId(1, 1, 1));
    add(set, ChunkId(-4, 4, 4)); // LOD 2, not an ancestor of (1, 1, 1)

    ChunkId id(1, 1, 1);
    CHECK(set.get_max_lod(id) == -1);
    CHECK(id == ChunkId(1, 1, 1)); // untouched when nothing is found
}

TEST_CASE_TEMPLATE("LOD sets: get_max_lod searches every LOD below max_lod", T, LODHashSet, ChunkHashMap) {
    T set;
    set.max_lod = 4; // chunks exist at LOD 0..3
    add(set, ChunkId(8, 8, 8)); // LOD 3

    ChunkId id(1, 1, 1);
    CHECK(set.get_max_lod(id) == 3);
    CHECK(id == ChunkId(8, 8, 8));
}
