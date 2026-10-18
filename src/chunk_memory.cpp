#include "chunk_memory.hpp"

void VoxelState::add_scan_chunk(std::unique_ptr<LODHashSet> p_scanned_chunk) {
    p_scanned_chunk->max_lod = dirty_chunk.max_lod;
    scanned_chunk = std::move(p_scanned_chunk);

    dirty_chunk.clear();

    // Any scanned chunk not already loaded must be generated.
    for (const ChunkId &pos : *scanned_chunk) {
        if (!loaded_chunk.has(pos)) {
            dirty_chunk.insert(pos);
        }
    }
}

void VoxelState::add_new_loaded(ChunkOutput &new_output) {
    if (loaded_chunk.has(new_output.chunk_id)) {
        loaded_chunk[new_output.chunk_id].free();
    }

    dirty_chunk.erase(new_output.chunk_id);
    loaded_chunk[new_output.chunk_id] = new_output;
}

void VoxelState::compute_destroy_full(ChunkHashMap &r_to_destroy, StallCountMap *r_stall_count) const {
    // A loaded chunk stays while one of its descendants is still being generated (split).
    StallCountMap local_stall;
    StallCountMap &stall_chunk = r_stall_count ? *r_stall_count : local_stall;
    stall_chunk.clear();
    for (const ChunkId &scan_pos : dirty_chunk) {
        ChunkId parent_id = scan_pos;
        if (loaded_chunk.find_ancestor(parent_id) >= 0) {
            int32_t *count = stall_chunk.getptr(parent_id);
            if (count) {
                (*count)++;
            } else {
                stall_chunk.insert(parent_id, 1);
            }
        }
    }

    r_to_destroy.clear();

    // Anything loaded but not stalled can be safely destroyed.
    for (ChunkHashMap::ConstIterator it = loaded_chunk.begin(); it != loaded_chunk.end(); ++it) {
        if (scanned_chunk->has(it->key)) {
            continue;
        }

        if (stall_chunk.has(it->key)) {
            continue;
        }

        // ... and stays while an ancestor replacing it is still being generated (merge).
        ChunkId parent_id = it->key;
        if (dirty_chunk.find_ancestor(parent_id) >= 0) {
            continue;
        }

        r_to_destroy.insert(it->key, it->value);
    }
}

void VoxelState::destroy_marked() {
    for (const KeyValue<ChunkId, ChunkOutput> &kv : to_destroy_chunk) {
        ChunkOutput output = kv.value;
        output.free();
        loaded_chunk.erase(kv.key);
    }
    to_destroy_chunk.clear();
}
