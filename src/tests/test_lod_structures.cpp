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
TEST_CASE_TEMPLATE("LOD sets: find_ancestor finds the nearest ancestor", T, LODHashSet, ChunkHashMap) {
    T set;
    set.max_lod = 5;
    add(set, ChunkId(4, 4, 4)); // LOD 2
    add(set, ChunkId(8, 8, 8)); // LOD 3, further up the same branch

    ChunkId id(1, 1, 1);
    CHECK(set.find_ancestor(id) == 2);
    CHECK(id == ChunkId(4, 4, 4)); // rewritten to the ancestor

    ChunkId lod1(2, 2, 2);
    CHECK(set.find_ancestor(lod1) == 2);
    CHECK(lod1 == ChunkId(4, 4, 4));
}

TEST_CASE_TEMPLATE("LOD sets: find_ancestor ignores the chunk itself and other branches", T, LODHashSet, ChunkHashMap) {
    T set;
    set.max_lod = 5;
    add(set, ChunkId(1, 1, 1));
    add(set, ChunkId(-4, 4, 4)); // LOD 2, not an ancestor of (1, 1, 1)

    ChunkId id(1, 1, 1);
    CHECK(set.find_ancestor(id) == -1);
    CHECK(id == ChunkId(1, 1, 1)); // untouched when nothing is found
}

TEST_CASE_TEMPLATE("LOD sets: find_ancestor searches every LOD below max_lod, and only those", T, LODHashSet, ChunkHashMap) {
    T set;
    set.max_lod = 4; // chunks exist at LOD 0..3
    add(set, ChunkId(8, 8, 8)); // LOD 3
    add(set, ChunkId(16, 16, 16)); // LOD 4: no chunk lives there

    ChunkId id(1, 1, 1);
    CHECK(set.find_ancestor(id) == 3);
    CHECK(id == ChunkId(8, 8, 8));

    ChunkId root(8, 8, 8);
    CHECK(set.find_ancestor(root) == -1);
}

TEST_CASE_TEMPLATE("LOD sets: find_ancestor stops at the top of the int64 octree", T, LODHashSet, ChunkHashMap) {
    // get_parent_from_child returns the coordinate itself past bit 62; the search must still end.
    T set;
    set.max_lod = 64;
    const int64_t top = int64_t(1) << 62;
    add(set, ChunkId(top, top, top)); // its own "parent": must not count as an ancestor
    ChunkId id(top, top, top);
    CHECK(set.find_ancestor(id) == -1);
    CHECK(id == ChunkId(top, top, top));
}
