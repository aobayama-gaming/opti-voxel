#pragma once

#include "chunk_output.hpp"
#include "lod_structures.hpp"

#include <memory>

using StallCountMap = ov::HashMap<ChunkId, int32_t, Vector3i64Hasher>;

// What the main thread knows of the chunks: the scan, the loaded set, and which loaded chunks can go.
// The destroy rule (a chunk outside the scan goes once nothing still being generated needs it) is kept
// up to date per event (scan delta, mesher result, destruction) instead of recomputed per frame:
// update_destroy only looks at the loaded chunks outside the scan. compute_destroy_full is the
// recompute from scratch, the oracle of the unit tests and of --destroy_verify.
struct VoxelState {
    void set_max_lod(int p_max_lod) {
        dirty_chunk.max_lod = p_max_lod;
        loaded_chunk.max_lod = p_max_lod;
        to_destroy_chunk.max_lod = p_max_lod;
        leaving.max_lod = p_max_lod;
        scanned_chunk->max_lod = p_max_lod;
    };

    // Scanned, not loaded yet: always the scan minus the loaded set.
    LODHashSet dirty_chunk;
    // The latest scan, handed over by the discoverer without a copy.
    std::unique_ptr<LODHashSet> scanned_chunk = std::make_unique<LODHashSet>();
    ChunkHashMap loaded_chunk;
    ChunkHashMap to_destroy_chunk;

    // Dirty chunk -> its nearest loaded ancestor, for the dirty chunks that have one.
    ov::HashMap<ChunkId, ChunkId, Vector3i64Hasher> dirty_anchor;
    // Loaded chunk -> how many dirty chunks have it as nearest loaded ancestor (only counts > 0). Split
    // rule: those stay.
    StallCountMap stall_count;
    // Loaded and not scanned: the only chunks that can be destroyed.
    LODHashSet leaving;

    // Debug (--destroy_verify): check every scan delta against the full sets, and every update_destroy
    // against compute_destroy_full.
    bool verify = false;

    struct Stats {
        // A stale result (loaded after it left the scan) can become the nearest loaded ancestor of dirty
        // chunks below it: a pass over the dirty set. Counted, it should be rare.
        int64_t rescans = 0;
        int64_t rescan_reanchored = 0; // dirty chunks those passes moved to the new chunk
        uint64_t rescan_usec = 0;
        int64_t leaving_max = 0;
        // --destroy_verify.
        int64_t verify_checks = 0;
        int64_t verify_mismatches = 0; // to_destroy, stall counts, dirty and leaving sets vs the full recompute
        int64_t delta_mismatches = 0; // scan deltas that don't lead from the previous scan to the new one
        int64_t octree_violations = 0; // scanned chunks with a scanned ancestor (the scan must be an octree cut)
    } stats;

    // A new scan and the change from the previous one (see ScanDelta).
    void add_scan_chunk(std::unique_ptr<LODHashSet> p_scanned_chunk, const ScanDelta &p_delta);

    void add_new_loaded(ChunkOutput &new_output);

    // Fills to_destroy_chunk (copies of the outputs) from the leaving chunks.
    void update_destroy();

    // The destroy rule recomputed from scratch over every dirty and loaded chunk. r_to_destroy gets the
    // loaded chunks that can go (copies of their outputs); r_stall_count (optional) gets, for every loaded
    // chunk that a dirty chunk has as nearest loaded ancestor, how many do (split rule: those stay).
    void compute_destroy_full(ChunkHashMap &r_to_destroy, StallCountMap *r_stall_count) const;
    void update_destroy_full() { compute_destroy_full(to_destroy_chunk, nullptr); }
    // Compares to_destroy_chunk (as update_destroy left it), stall_count, dirty_chunk and leaving with
    // the full recompute; the number of differences, also added to stats.verify_mismatches.
    int64_t verify_against_full();

    // Frees the chunks of to_destroy_chunk and drops them from the loaded set.
    void destroy_marked();

    // Back to an empty state (the loaded outputs must have been freed).
    void clear();

private:
    void _anchor(const ChunkId &p_dirty);
    void _unanchor(const ChunkId &p_dirty);
    void _stall_add(const ChunkId &p_loaded, int32_t p_count);
    void _rescan_below(const ChunkId &p_loaded);
    void _verify_delta(const LODHashSet &p_new, const ScanDelta &p_delta);
};
