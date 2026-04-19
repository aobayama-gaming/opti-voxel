#pragma once

#include "godot_cpp/variant/variant.hpp"
#include "lod_structures.hpp"
#include "chunk_output.hpp"

using namespace godot;

struct VoxelState{



    void set_max_lod(int p_max_lod){
        dirty_chunk.max_lod = p_max_lod;
        loaded_chunk.max_lod = p_max_lod;
        to_destroy_chunk.max_lod = p_max_lod;
        scanned_chunk.max_lod = p_max_lod;
    };

	LODHashSet dirty_chunk;
    LODHashSet scanned_chunk;
	ChunkHashMap loaded_chunk;
	ChunkHashMap to_destroy_chunk;

	void add_scan_chunk(LODHashSet &p_scanned_chunk);

    void add_new_loaded(ChunkOutput &new_output);

    void update_destroy_from_scan();
};