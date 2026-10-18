#include "test_common.hpp"

#include "lod_structures.hpp"

#include <random>
#include <vector>

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

namespace {

LODHashSet set_of(std::initializer_list<ChunkId> p_ids) {
    LODHashSet s;
    for (const ChunkId &id : p_ids) {
        s.insert(id);
    }
    return s;
}

bool same_set(const LODHashSet &p_a, const LODHashSet &p_b) {
    if (p_a.size() != p_b.size()) {
        return false;
    }
    for (const ChunkId &id : p_a) {
        if (!p_b.has(id)) {
            return false;
        }
    }
    return true;
}

// a -> b, then b -> c merged into it.
ScanDelta merged(const LODHashSet &p_a, const LODHashSet &p_b, const LODHashSet &p_c) {
    ScanDelta d;
    ScanDelta::between(p_a, p_b, d);
    ScanDelta next;
    ScanDelta::between(p_b, p_c, next);
    d.merge(next);
    return d;
}

} // namespace

TEST_CASE("ScanDelta: between two scans") {
    const ChunkId a(1, 1, 1), b(3, 1, 1), c(5, 1, 1);
    ScanDelta d;
    ScanDelta::between(set_of({ a, b }), set_of({ b, c }), d);
    CHECK(same_set(d.entered, set_of({ c })));
    CHECK(same_set(d.left, set_of({ a })));
}

TEST_CASE("ScanDelta: merging across a skipped scan cancels what came and went") {
    const ChunkId a(1, 1, 1), b(3, 1, 1), c(5, 1, 1);
    const LODHashSet empty;

    // Entered then left: nothing.
    ScanDelta d = merged(empty, set_of({ a }), empty);
    CHECK(d.entered.is_empty());
    CHECK(d.left.is_empty());

    // Left then entered again: nothing.
    d = merged(set_of({ a }), empty, set_of({ a }));
    CHECK(d.entered.is_empty());
    CHECK(d.left.is_empty());

    // Entered twice in a row is impossible, but an id kept across both steps stays out of the delta.
    d = merged(set_of({ a }), set_of({ a, b }), set_of({ a, b, c }));
    CHECK(same_set(d.entered, set_of({ b, c })));
    CHECK(d.left.is_empty());

    // Repeats: enter, leave, enter.
    ScanDelta step;
    d = merged(empty, set_of({ a }), empty);
    ScanDelta::between(empty, set_of({ a }), step);
    d.merge(step);
    CHECK(same_set(d.entered, set_of({ a })));
    CHECK(d.left.is_empty());
}

TEST_CASE("ScanDelta: merged deltas of random scan sequences equal the direct delta") {
    std::mt19937_64 rng(7);
    int64_t merges = 0;
    for (int run = 0; run < 200; run++) {
        // Ids drawn from a small pool so they come and go often.
        auto random_scan = [&rng]() {
            LODHashSet s;
            for (int i = 0; i < 40; i++) {
                if (rng() % 2) {
                    s.insert(ChunkId(2 * i + 1, 1, 1));
                }
            }
            return s;
        };
        const LODHashSet first = random_scan();
        LODHashSet previous = first;
        LODHashSet applied = first;
        ScanDelta acc;
        const int steps = 1 + int(rng() % 8);
        for (int i = 0; i < steps; i++) {
            const LODHashSet next = random_scan();
            ScanDelta step;
            ScanDelta::between(previous, next, step);
            acc.merge(step);
            merges++;
            previous = next;
        }
        ScanDelta direct;
        ScanDelta::between(first, previous, direct);
        CHECK(same_set(acc.entered, direct.entered));
        CHECK(same_set(acc.left, direct.left));
        acc.apply(applied);
        CHECK(same_set(applied, previous));
    }
    MESSAGE(merges, " merges over 200 random scan sequences");
}
