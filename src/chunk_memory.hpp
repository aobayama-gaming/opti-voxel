#pragma once

#include "chunk_output.hpp"
#include "latency_hist.hpp"
#include "lod_structures.hpp"
#include "ov/clock.hpp"

#include <memory>

// A stalled chunk: the dirty chunks that have it as nearest loaded ancestor (split rule: it stays), in total and
// per sub-cube they overlap (ChunkMath::subcube_bits, bit order Subcube::bit).
struct Stall {
    int32_t count = 0;
    int32_t n[64] = {};
    uint64_t waiting = 0; // the sub-cubes with n > 0
    uint64_t since = 0; // VoxelState::frame when it was first stalled (stats; not compared)

    // One dirty chunk overlapping p_bits anchored (+1) or no longer anchored (-1).
    void add(uint64_t p_bits, int32_t p_delta);
    // Coverage: the sub-cubes no dirty chunk waits on are covered (the chunks replacing it there are loaded).
    uint64_t hidden() const { return ~waiting; }
    bool operator==(const Stall &p_o) const;
};
using StallMap = ov::HashMap<ChunkId, Stall, Vector3i64Hasher>;

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
    // Loaded chunk -> its Stall, for the loaded chunks some dirty chunk has as nearest loaded ancestor.
    StallMap stalled;
    // Loaded and not scanned: the only chunks that can be destroyed.
    LODHashSet leaving;
    // Chunks whose hidden_mask changed (or that were re-meshed into new rows) since push_coverage.
    ov::HashSet<ChunkId, Vector3i64Hasher> coverage_changed;

    // The caller's frame counter (VoxelTerrain::_process), for the hand-off latency.
    uint64_t frame = 0;

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
        int64_t coverage_written = 0; // hidden masks pushed (push_coverage), in total
        int64_t coverage_written_max = 0; // in one push
        // --destroy_verify.
        int64_t verify_checks = 0;
        int64_t verify_mismatches = 0; // to_destroy, stall counts, dirty and leaving sets vs the full recompute
        int64_t delta_mismatches = 0; // scan deltas that don't lead from the previous scan to the new one
        int64_t octree_violations = 0; // scanned chunks with a scanned ancestor (the scan must be an octree cut)
        // Frames from a loaded chunk's first stalled sub-cube to its last one clearing (its replacements loaded, or no
        // longer scanned): how long a parent stays drawn over its children. Cumulative.
        LatencyHist handoff_frames;
    } stats;

    // A new scan and the change from the previous one (see ScanDelta).
    void add_scan_chunk(std::unique_ptr<LODHashSet> p_scanned_chunk, const ScanDelta &p_delta);

    void add_new_loaded(ChunkOutput &new_output);

    // Fills to_destroy_chunk (copies of the outputs) from the leaving chunks.
    void update_destroy();

    // The destroy rule recomputed from scratch over every dirty and loaded chunk. r_to_destroy gets the
    // loaded chunks that can go (copies of their outputs); r_stalled (optional) the Stall of every loaded chunk a
    // dirty chunk has as nearest loaded ancestor.
    void compute_destroy_full(ChunkHashMap &r_to_destroy, StallMap *r_stalled = nullptr) const;
    void update_destroy_full() { compute_destroy_full(to_destroy_chunk); }
    // The sub-cubes of a loaded chunk that can be hidden (PageFormat's coverage texel): the covered ones of a
    // stalled chunk, none (0) otherwise.
    uint64_t hidden_mask(const ChunkId &p_loaded) const;
    // The covered sub-cubes of all the stalled chunks (stats).
    int64_t hidden_subcubes() const;
    // Compares to_destroy_chunk (as update_destroy left it), stalled, dirty_chunk and leaving
    // with the full recompute; the number of differences, also added to stats.verify_mismatches.
    int64_t verify_against_full();

    // Frees the chunks of to_destroy_chunk and drops them from the loaded set.
    void destroy_marked();

    // Hands the hidden_mask of every loaded chunk in coverage_changed to its owner (the meshers whose chunks live in
    // pages, ChunkReleaser::set_hidden); call after compaction, so the rows are current.
    void push_coverage();

    // Back to an empty state (the loaded outputs must have been freed).
    void clear();

private:
    void _anchor(const ChunkId &p_dirty);
    void _unanchor(const ChunkId &p_dirty);
    // Dirty chunk p_dirty anchored to (+1) or no longer anchored to (-1) loaded chunk p_loaded: its Stall, and
    // coverage_changed when its hidden mask changes.
    void _stall_add(const ChunkId &p_loaded, const ChunkId &p_dirty, int32_t p_delta);
    void _rescan_below(const ChunkId &p_loaded);
    void _verify_delta(const LODHashSet &p_new, const ScanDelta &p_delta);
};
