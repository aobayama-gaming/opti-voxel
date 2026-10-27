#pragma once

#include "world_coord.hpp"

#include <cstdint>
#include <vector>

// A chunk's triangles in its vertex grid (cells of the chunk's LOD): the legacy mesher's output.
struct ChunkMesh {
    std::vector<Float3> points;
    std::vector<Float3> normals;
    std::vector<int32_t> indices;
    // Per point: its global material (material_library.hpp), exact as a float below Material::MAX_MATERIALS.
    std::vector<float> materials;
};

// The host's side of the legacy meshes (GPUMesher): one drawn mesh per chunk. Godot's: a mesh and an instance in a
// scenario (adapter/godot/src/godot_mesh_host.*). Thread-safe: the legacy worker creates meshes, the main thread frees
// and reads them.
class MeshHost {
public:
    virtual ~MeshHost() = default;

    // A drawn instance of p_mesh, scaled by p_cell_size with its vertex grid's origin at p_origin (render metres); 0 if
    // none was made.
    virtual uint64_t create_mesh(const ChunkMesh &p_mesh, float p_cell_size, const Float3 &p_origin) = 0;
    virtual void free_mesh(uint64_t p_mesh) = 0;
    // Moves every mesh by -p_shift metres: the render origin moved (World::rebase, ChunkMath::origin_shift_m). Never
    // while a create_mesh runs (GPUMesher::set_render_origin).
    virtual void shift_meshes(const Float3 &p_shift) = 0;
    // The mesh as the host holds it, to compare with the direct mesher's (MeshValidator); false if unknown.
    virtual bool read_mesh(uint64_t p_mesh, ChunkMesh &r_mesh) const = 0;
};
