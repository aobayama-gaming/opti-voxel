#pragma once

#include "chunk_math.hpp"
#include "chunk_output.hpp"
#include "godot_cpp/templates/hash_set.hpp"

using namespace godot;

class LODHashSet : public HashSet<ChunkId, Vector3i64Hasher> {
    // HashSet of chunk position. The position is stored as the chunk center (LOD 0 : 2 wide),
    // It enables fast LOD calculation and chunk management.

public:
    uint32_t max_lod;

    int get_max_lod(ChunkId &p_chunk_pos) const {
        // Get if a parent of the chunk is already in the HashSet, and return the LOD of the parent chunk. If no parent is found, return -1.

        int actual_lod = 0;
        ChunkId parent_pos = p_chunk_pos;

        while (actual_lod < max_lod) {
            parent_pos = ChunkMath::get_parent_from_child(parent_pos, actual_lod);

            if (has(parent_pos)) {
                p_chunk_pos = parent_pos;
                return actual_lod;
            }
            actual_lod++;
        };

        return -1; // Not found
    };
};

class ChunkHashMap : public HashMap<ChunkId, ChunkOutput, Vector3i64Hasher> {
public:
    uint32_t max_lod;

    int get_max_lod(ChunkId &p_chunk_pos) const {
        // Get if a parent of the chunk is already in the HashSet, and return the LOD of the parent chunk. If no parent is found, return -1.

        int actual_lod = 0;

        ChunkId parent_pos = p_chunk_pos;

        while (actual_lod < max_lod) {
            parent_pos = ChunkMath::get_parent_from_child(parent_pos, actual_lod);

            if (has(parent_pos)) {
                p_chunk_pos = parent_pos;
                return actual_lod;
            }
        };

        return -1; // Not found
    };

    int get_chunk(const ChunkId &p_chunk_pos, ChunkOutput &chunk) {
        // retrieve the chunk or its parent.

        int actual_lod = 0;

        ChunkId parent_pos = p_chunk_pos;
        while (actual_lod < max_lod) {
            parent_pos = ChunkMath::get_parent_from_child(parent_pos, actual_lod);

            if (has(parent_pos)) {
                chunk = get(parent_pos);
                return actual_lod; //actual
            }
        };

        return -1; // Not found
    };
};