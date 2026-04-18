#pragma once

#include "godot_cpp/templates/hash_set.hpp"
#include "chunk_math.hpp"

using namespace godot;

class LODHashSet : public HashSet<Vector3i> {
// HashSet of chunk position. The position is stored as the chunk center (LOD 0 : 2 wide),
// It enables fast LOD calculation and chunk management.

public:

    int get_max_lod(const Vector3i &p_chunk_pos,int max_lod) const {
        // Get if a parent of the chunk is already in the HashSet, and return the LOD of the parent chunk. If no parent is found, return -1.
        
        int actual_lod = 0;

        while (actual_lod < max_lod) {

            Vector3i parent_pos = ChunkMath::get_parent_from_child(p_chunk_pos, actual_lod);

            if (has(parent_pos)) {
                return actual_lod;
            }
            actual_lod++;
        };

        return -1; // Not found
    };
};