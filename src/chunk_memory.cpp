#include "chunk_memory.hpp"

void VoxelState::add_scan_chunk(LODHashSet &scanned_chunk){

    // Any loaded chunk not present in the new scan must be destroyed.
    for (ChunkHashMap::ConstIterator it = loaded_chunk.begin(); it != loaded_chunk.end(); ++it) {
        const Vector3i &pos = it->key;
        if (!scanned_chunk.has(pos)) {
            to_destroy_chunk.insert(pos, it->value);
        }
    }

    dirty_chunk.clear();

    // Any scanned chunk not already loaded must be generated.
    for (const Vector3i &pos : scanned_chunk) {
        if (!loaded_chunk.has(pos)) {
            dirty_chunk.insert(pos);
        }
    }

}

void VoxelState::add_new_loaded(ChunkOutput &new_output) {

    dirty_chunk.erase(new_output.chunk_id);
    loaded_chunk[new_output.chunk_id] = new_output;
}