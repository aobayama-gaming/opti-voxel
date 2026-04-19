#pragma once

#include "godot_cpp/variant/variant.hpp"
#include "lod_structures.hpp"
#include "chunk_output.hpp"

using namespace godot;

struct VoxelState{

	LODHashSet dirty_chunk;
	ChunkHashMap loaded_chunk;
	ChunkHashMap to_destroy_chunk;

	void add_scan_chunk(LODHashSet &scanned_chunk);

    void add_new_loaded(ChunkOutput &new_output);
};