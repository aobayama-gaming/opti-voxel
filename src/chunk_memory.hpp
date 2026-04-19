#pragma once

#include "godot_cpp/variant/variant.hpp"
#include "lod_structures.hpp"

using namespace godot;

struct ChunkOutput{
    RID mesh_rid;
};

struct VoxelState{

	LODHashSet dirty_chunk;
	ChunkHashMap loaded_chunk;
	ChunkHashMap to_destroy_chunk;

	void add_scan_chunk(LODHashSet &scanned_chunk);

    
};