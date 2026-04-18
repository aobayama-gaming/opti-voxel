#pragma once

#include "godot_cpp/classes/engine.hpp"

using namespace godot;

struct ChunkParam{
    int chunk_size = 32;
    float voxel_size = 0.5f;
    int skirt_size = 2;

    int chunk_voxel_dimension() const noexcept {
        return (chunk_size-skirt_size*2);
    }

    float chunk_world_dimension() const noexcept {
        return chunk_voxel_dimension() * voxel_size;
    }

};

namespace ChunkMath{

    // Just the signatures, no function bodies
    int32_t get_parent_from_child(int32_t coord, int &lod);
    int32_t get_parent_from_child(int32_t coord);

    Vector3i get_parent_from_child(Vector3i coord, int &lod);
    Vector3i get_parent_from_child(Vector3i coord);

    int32_t get_lod(Vector3i coord);

    Vector3i get_parent_from_child_until(Vector3i coord, int target_lod);

    Vector3 chunk_to_world(Vector3i chunk_coord,ChunkParam chunk_param);
    Vector3i world_to_chunk(Vector3 world_coord,ChunkParam chunk_param);

    Vector3 vertices_to_world(Vector3i chunk_coord, Vector3 local_coord,ChunkParam chunk_param);
    Vector3 world_to_vertices(Vector3i chunk_coord, Vector3 world_coord,ChunkParam chunk_param);
    
    float world_chunk_size(Vector3i chunk_coord,ChunkParam chunk_param);
    float world_chunk_spherical_radius(Vector3i chunk_coord,ChunkParam chunk_param);

    bool vertices_out_of_bound(Vector3 vertices_coord,ChunkParam chunk_param);
}
