#pragma once

#include "godot_cpp/variant/variant.hpp"
#include "lod_structures.hpp"
#include "chunk_output.hpp"

#include <memory>

using namespace godot;

struct VoxelState{



    void set_max_lod(int p_max_lod){
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

    void update_destroy_from_scan();
};