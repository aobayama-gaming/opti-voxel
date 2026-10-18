#pragma once

#include "chunk_math.hpp"
#include "chunk_output.hpp"
#include "godot_cpp/templates/hash_set.hpp"

using namespace godot;

// Nearest strict ancestor of r_chunk that p_set holds, searching the LODs below p_max_lod (the LODs
// chunks exist at). Returns its LOD and sets r_chunk to it, or returns -1 and leaves r_chunk alone.
template <typename T>
int find_ancestor_in(const T &p_set, ChunkId &r_chunk, int p_max_lod) {
    ChunkId parent = r_chunk;
    for (int lod = ChunkMath::get_lod(r_chunk) + 1; lod < p_max_lod; lod++) {
        const ChunkId next = ChunkMath::get_parent_from_child(parent);
        if (next == parent) {
            return -1; // past the top of the int64 octree
        }
        parent = next;
        if (p_set.has(parent)) {
            r_chunk = parent;
            return lod;
        }
    }
    return -1;
}

// Chunk ids (chunk centres, see chunk_math.hpp).
class LODHashSet : public HashSet<ChunkId, Vector3i64Hasher> {
public:
    int max_lod = 0; // chunks exist at LODs [0, max_lod)

    int find_ancestor(ChunkId &r_chunk) const { return find_ancestor_in(*this, r_chunk, max_lod); }
};

class ChunkHashMap : public HashMap<ChunkId, ChunkOutput, Vector3i64Hasher> {
public:
    int max_lod = 0; // chunks exist at LODs [0, max_lod)

    int find_ancestor(ChunkId &r_chunk) const { return find_ancestor_in(*this, r_chunk, max_lod); }
};

// What changed between two scans. The discoverer publishes one with each scan, merged across the scans
// the main thread skipped, so it always leads from the last scan the main thread took to the new one.
struct ScanDelta {
    LODHashSet entered; // in the new scan, not in the old one
    LODHashSet left; // in the old scan, not in the new one

    // r_delta = p_from -> p_to.
    static void between(const LODHashSet &p_from, const LODHashSet &p_to, ScanDelta &r_delta) {
        r_delta.entered.clear();
        r_delta.left.clear();
        for (const ChunkId &id : p_to) {
            if (!p_from.has(id)) {
                r_delta.entered.insert(id);
            }
        }
        for (const ChunkId &id : p_from) {
            if (!p_to.has(id)) {
                r_delta.left.insert(id);
            }
        }
    }

    // this = a -> b and p_next = b -> c: this becomes a -> c. An id that entered then left (or left then
    // entered again) cancels out.
    void merge(const ScanDelta &p_next) {
        for (const ChunkId &id : p_next.entered) {
            if (!left.erase(id)) {
                entered.insert(id);
            }
        }
        for (const ChunkId &id : p_next.left) {
            if (!entered.erase(id)) {
                left.insert(id);
            }
        }
    }

    // a -> b applied to r_scan = a makes it b.
    void apply(LODHashSet &r_scan) const {
        for (const ChunkId &id : left) {
            r_scan.erase(id);
        }
        for (const ChunkId &id : entered) {
            r_scan.insert(id);
        }
    }
};
