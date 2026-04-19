#include "chunk_memory.hpp"

void VoxelState::add_scan_chunk(LODHashSet &p_scanned_chunk){

    // Any loaded chunk not present in the new scan must be destroyed.
    // for (ChunkHashMap::ConstIterator it = loaded_chunk.begin(); it != loaded_chunk.end(); ++it) {
    //     const Vector3i &pos = it->key;
    //     if (!scanned_chunk.has(pos)) {
    //         to_destroy_chunk.insert(pos, it->value);
    //     }
    // }

    scanned_chunk = p_scanned_chunk;

    dirty_chunk.clear();

    // Any scanned chunk not already loaded must be generated.
    for (const Vector3i &pos : p_scanned_chunk) {
        if (!loaded_chunk.has(pos)) {
            dirty_chunk.insert(pos);
        }
    }

}

void VoxelState::add_new_loaded(ChunkOutput &new_output) {

    dirty_chunk.erase(new_output.chunk_id);
    loaded_chunk[new_output.chunk_id] = new_output;
}

constexpr int CHUNK_DIFFERENCE = 4;

void VoxelState::update_destroy_from_scan() {

    LODHashSet stall_chunk;
    stall_chunk.max_lod = loaded_chunk.max_lod;
    for (const Vector3i &scan_pos : dirty_chunk){

        Vector3i parent_id = scan_pos;

        

        const int first_loaded_lod = loaded_chunk.get_max_lod(parent_id);
        if (first_loaded_lod < 0 ) { //|| first_loaded_lod-ChunkMath::get_lod(scan_pos) > CHUNK_DIFFERENCE
            continue;
        }

        stall_chunk.insert(parent_id);

    }

    to_destroy_chunk.clear();

    // Anything loaded but not stalled can be safely destroyed.
    for (ChunkHashMap::ConstIterator it = loaded_chunk.begin(); it != loaded_chunk.end(); ++it) {

        if(scanned_chunk.has(it->key)){
            continue;
        }

        if(stall_chunk.has(it->key)){
            continue;
        }

        Vector3i parent_id = it->key;

        const int parent_lod =dirty_chunk.get_max_lod(parent_id);
        if(parent_lod>=0 ){ //&& parent_lod-ChunkMath::get_lod(it->key)>CHUNK_DIFFERENCE
            continue;
        }

        to_destroy_chunk.insert(it->key, it->value);
        
    }
}