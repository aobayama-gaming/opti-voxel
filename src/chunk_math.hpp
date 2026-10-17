#pragma once

#include "godot_cpp/classes/engine.hpp"

#include "world_coord.hpp"

using namespace godot;

struct ChunkParam {
    int chunk_size = 32;
    float voxel_size = 0.5f;
    int skirt_size = 2;
    // Depth of the skirt in voxels of the chunk's own LOD: the overhang past the chunk bounds is
    // pushed this far into the solid, under the neighbour's surface, to hide LOD seams.
    float skirt_depth = 1.0f;
    // Integer world origin in chunk-id units (snapped to the root chunk span, see VoxelTerrain::_ready).
    // Godot space is the integer frame shifted by it: render positions are metres relative to the
    // origin, so they stay small wherever the world is.
    ChunkId origin_chunk;

    int chunk_voxel_dimension() const noexcept {
        return (chunk_size - skirt_size * 2);
    }

    float chunk_world_dimension() const noexcept {
        return chunk_voxel_dimension() * voxel_size;
    }

    // One chunk-id unit is half a LOD0 chunk (chunk_voxel_dimension must be even).
    int64_t half_chunk_voxels() const noexcept {
        return chunk_voxel_dimension() / 2;
    }

    VoxelPos origin_voxel() const noexcept {
        return origin_chunk * half_chunk_voxels();
    }
};

// Chunk ids are int64 chunk centres in chunk-id units (half a LOD0 chunk): LOD0 chunks sit on odd
// coordinates, a LOD-L chunk's coordinates have their lowest set bit at L and it spans 2^(L+1) units.
namespace ChunkMath {

int64_t get_parent_from_child(int64_t coord, int &lod);
int64_t get_parent_from_child(int64_t coord);

ChunkId get_parent_from_child(const ChunkId &coord, int &lod);
ChunkId get_parent_from_child(const ChunkId &coord);

int get_lod(const ChunkId &coord);
int64_t get_lod_size(const ChunkId &coord);

ChunkId get_parent_from_child_until(const ChunkId &coord, int target_lod);

// Edge length of the chunk in metres (a size, not a position).
float world_chunk_size(const ChunkId &chunk_coord, const ChunkParam &chunk_param);

// LOD0 voxel of the chunk's minimum corner (vertex grid index skirt_size) and of its centre.
VoxelPos chunk_min_voxel(const ChunkId &chunk_coord, const ChunkParam &chunk_param);
VoxelPos chunk_center_voxel(const ChunkId &chunk_coord, const ChunkParam &chunk_param);

// Render space (Godot space): metres relative to the world origin, shifted by the legacy lattice
// offset like the GPU sampling and the page shader. Only these helpers produce metres.
Vector3 chunk_render_center(const ChunkId &chunk_coord, const ChunkParam &chunk_param);
// A point of the chunk's vertex grid (the grid the SDF is sampled on, skirt included).
Vector3 grid_to_render(const ChunkId &chunk_coord, const Vector3 &grid, const ChunkParam &chunk_param);
// Same from a page header, which only carries the low 32 bits of the id (mirrors the page shader).
Vector3 grid_to_render_lo32(const Vector3i &chunk_lo, const Vector3 &grid, const ChunkParam &chunk_param);
// LOD0 chunk holding a render-space position.
ChunkId chunk_at_render(const Vector3 &render_pos, const ChunkParam &chunk_param);

// Absolute position of a vertex-grid point: CPU mirror of world_pos_from_grid (gpu_frame.hpp).
WorldPos sample_pos(const ChunkId &chunk_coord, const Float3 &grid, const ChunkParam &chunk_param);
// Origin-relative voxels and metres for SDFs written in metres (SDFMetreBridge): CPU mirrors of
// legacy_rel_voxels / legacy_sdf_metres, op for op.
Float3 legacy_rel_voxels(const WorldPos &pos, const ChunkParam &chunk_param);
Vector3 legacy_metres(const WorldPos &pos, const ChunkParam &chunk_param);
} //namespace ChunkMath
