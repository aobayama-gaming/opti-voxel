#include "godot_mesh_host.hpp"

#include "godot_conv.hpp"

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>

using namespace godot;

namespace {

PackedVector3Array godot_points(const std::vector<Float3> &p_points) {
    PackedVector3Array out;
    out.resize(int64_t(p_points.size()));
    for (size_t i = 0; i < p_points.size(); i++) {
        out.set(int64_t(i), godot_v3(p_points[i]));
    }
    return out;
}

} // namespace

GodotMeshHost::GodotMeshHost(RID p_scenario) : scenario(p_scenario) {
    if (!scenario.is_valid()) {
        scenario = RenderingServer::get_singleton()->scenario_create();
        owns_scenario = true;
    }
}

GodotMeshHost::~GodotMeshHost() {
    RenderingServer *rs = RenderingServer::get_singleton();
    for (const std::pair<const uint64_t, Drawn> &kv : meshes) {
        rs->free_rid(kv.second.instance);
        rs->free_rid(kv.second.mesh);
    }
    if (owns_scenario) {
        rs->free_rid(scenario);
    }
}

uint64_t GodotMeshHost::create_mesh(const ChunkMesh &p_mesh, float p_cell_size, const Float3 &p_origin) {
    RenderingServer *rs = RenderingServer::get_singleton();
    PackedInt32Array indices;
    indices.resize(int64_t(p_mesh.indices.size()));
    for (size_t i = 0; i < p_mesh.indices.size(); i++) {
        indices.set(int64_t(i), p_mesh.indices[i]);
    }
    PackedFloat32Array materials;
    materials.resize(int64_t(p_mesh.materials.size()));
    for (size_t i = 0; i < p_mesh.materials.size(); i++) {
        materials.set(int64_t(i), p_mesh.materials[i]);
    }
    Array arrays;
    arrays.resize(Mesh::ARRAY_MAX);
    arrays[Mesh::ARRAY_VERTEX] = godot_points(p_mesh.points);
    arrays[Mesh::ARRAY_INDEX] = indices;
    arrays[Mesh::ARRAY_NORMAL] = godot_points(p_mesh.normals);
    arrays[Mesh::ARRAY_CUSTOM0] = materials;

    Drawn drawn;
    drawn.mesh = rs->mesh_create();
    rs->mesh_add_surface_from_arrays(drawn.mesh, RenderingServer::PRIMITIVE_TRIANGLES, arrays, Array(), Dictionary(),
            RenderingServer::ARRAY_CUSTOM_R_FLOAT << RenderingServer::ARRAY_FORMAT_CUSTOM0_SHIFT);
    drawn.instance = rs->instance_create();
    rs->instance_set_scenario(drawn.instance, scenario);
    rs->instance_set_base(drawn.instance, drawn.mesh);
    Basis scaled_basis;
    scaled_basis.scale(Vector3(p_cell_size, p_cell_size, p_cell_size));
    drawn.transform = Transform3D(scaled_basis, godot_v3(p_origin));
    rs->instance_set_transform(drawn.instance, drawn.transform);

    const std::lock_guard<std::mutex> lock(mutex);
    const uint64_t id = next_id++;
    meshes[id] = drawn;
    return id;
}

void GodotMeshHost::free_mesh(uint64_t p_mesh) {
    Drawn drawn;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        const auto it = meshes.find(p_mesh);
        if (it == meshes.end()) {
            return;
        }
        drawn = it->second;
        meshes.erase(it);
    }
    RenderingServer *rs = RenderingServer::get_singleton();
    rs->free_rid(drawn.instance);
    rs->free_rid(drawn.mesh);
}

void GodotMeshHost::place_mesh(uint64_t p_mesh, const Float3 &p_origin) {
    const std::lock_guard<std::mutex> lock(mutex);
    const auto it = meshes.find(p_mesh);
    if (it == meshes.end()) {
        return;
    }
    it->second.transform.origin = godot_v3(p_origin);
    RenderingServer::get_singleton()->instance_set_transform(it->second.instance, it->second.transform);
}

bool GodotMeshHost::read_mesh(uint64_t p_mesh, ChunkMesh &r_mesh) const {
    RID mesh;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        const auto it = meshes.find(p_mesh);
        if (it == meshes.end()) {
            return false;
        }
        mesh = it->second.mesh;
    }
    // What the renderer holds (its vertex format included), as the validator always compared it.
    const Array arrays = RenderingServer::get_singleton()->mesh_surface_get_arrays(mesh, 0);
    const PackedVector3Array v = arrays[Mesh::ARRAY_VERTEX];
    const PackedVector3Array vn = arrays[Mesh::ARRAY_NORMAL];
    const PackedInt32Array idx = arrays[Mesh::ARRAY_INDEX];
    const PackedFloat32Array vm = arrays[Mesh::ARRAY_CUSTOM0];
    r_mesh = ChunkMesh();
    for (int64_t i = 0; i < v.size(); i++) {
        r_mesh.points.push_back(ov_f3(v[i]));
    }
    for (int64_t i = 0; i < vn.size(); i++) {
        r_mesh.normals.push_back(ov_f3(vn[i]));
    }
    for (int64_t i = 0; i < idx.size(); i++) {
        r_mesh.indices.push_back(idx[i]);
    }
    for (int64_t i = 0; i < vm.size(); i++) {
        r_mesh.materials.push_back(vm[i]);
    }
    return true;
}
