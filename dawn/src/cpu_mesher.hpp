#pragma once

// An engine-free surface-nets mesher for the Dawn host: the same cell/quad topology as the legacy GPU mesher,
// on the CPU. It is the fallback path when a Dawn device cannot compile the compute shaders, and the reference
// the GPU path is compared against. Positions are metres relative to the world origin.

#include "world_coord.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace ov::dawn {

struct MeshVertex {
    Float3 position;
    Float3 normal;
};

struct MeshData {
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;
};

// p_sample: the signed distance at a LOD0 voxel, metres. p_min: the chunk's minimum LOD0 voxel. p_cells: the
// number of cells per axis (the sample grid is p_cells + 1). p_voxel_size: metres per LOD0 voxel.
MeshData mesh_surface_nets(const std::function<float(const VoxelPos &)> &p_sample, const VoxelPos &p_min, int p_cells, float p_voxel_size, int p_lod = 0);

} // namespace ov::dawn
