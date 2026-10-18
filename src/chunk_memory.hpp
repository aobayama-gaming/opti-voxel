#pragma once

#include "chunk_output.hpp"
#include "godot_cpp/variant/variant.hpp"
#include "lod_structures.hpp"

#include <memory>

using namespace godot;

using StallCountMap = HashMap<ChunkId, int32_t, Vector3i64Hasher>;

struct VoxelState {
    void set_max_lod(int p_max_lod) {
        dirty_chunk.max_lod = p_max_lod;
        loaded_chunk.max_lod = p_max_lod;
        to_destroy_chunk.max_lod = p_max_lod;
    };

    LODHashSet dirty_chunk;
    // The latest scan, handed over by the discoverer without a copy.
    std::unique_ptr<LODHashSet> scanned_chunk = std::make_unique<LODHashSet>();
    ChunkHashMap loaded_chunk;
    ChunkHashMap to_destroy_chunk;

    void add_scan_chunk(std::unique_ptr<LODHashSet> p_scanned_chunk);

    void add_new_loaded(ChunkOutput &new_output);

    // The destroy rule recomputed from scratch over every dirty and loaded chunk. r_to_destroy gets the
    // loaded chunks that can go (copies of their outputs); r_stall_count (optional) gets, for every loaded
    // chunk that a dirty chunk has as nearest loaded ancestor, how many do (split rule: those stay).
    void compute_destroy_full(ChunkHashMap &r_to_destroy, StallCountMap *r_stall_count) const;
    void update_destroy_full() { compute_destroy_full(to_destroy_chunk, nullptr); }

    // Frees the chunks of to_destroy_chunk and drops them from the loaded set.
    void destroy_marked();
};
