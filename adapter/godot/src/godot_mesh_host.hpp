#pragma once

#include "mesh_host.hpp"

#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform3d.hpp>

#include <map>
#include <mutex>

// MeshHost on Godot's RenderingServer: a mesh (vertex, normal, index, the material in CUSTOM0 as an R float) and an
// instance per chunk, in a scenario.
class GodotMeshHost : public MeshHost {
public:
    // p_scenario: where the instances are drawn; invalid: a scenario of its own that nothing renders (the validator's).
    explicit GodotMeshHost(godot::RID p_scenario = godot::RID());
    ~GodotMeshHost() override;

    uint64_t create_mesh(const ChunkMesh &p_mesh, float p_cell_size, const Float3 &p_origin) override;
    void free_mesh(uint64_t p_mesh) override;
    void shift_meshes(const Float3 &p_shift) override;
    bool read_mesh(uint64_t p_mesh, ChunkMesh &r_mesh) const override;

private:
    struct Drawn {
        godot::RID mesh;
        godot::RID instance;
        godot::Transform3D transform;
    };
    godot::RID scenario;
    bool owns_scenario = false;
    mutable std::mutex mutex;
    std::map<uint64_t, Drawn> meshes;
    uint64_t next_id = 1;
};
