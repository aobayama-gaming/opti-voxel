#pragma once

// The Dawn host's world: a quadtree of LOD chunks around the camera, meshed by the core's surface nets (CPU
// fallback) or the legacy GPU mesher, plus the carve primitives and the ray march the player's tools use.

#include "cpu_mesher.hpp"
#include "terrain_sdf.hpp"
#include "world_coord.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ov::dawn {

struct ChunkIdHash {
    size_t operator()(const ChunkId &p_id) const { return Vector3i64Hasher::hash(p_id); }
};

class Terrain {
public:
    struct Chunk {
        ChunkId id;
        VoxelPos min;
        int lod = 0;
        int cells = 16;
        MeshData mesh;
        bool dirty = true;
        uint64_t revision = 0;
    };

    void initialize(float p_voxel_size, const Float3 &p_spawn_m);
    // Rebuilds the chunk set around p_camera_m and meshes at most p_budget chunks (0: all).
    void update(const Float3 &p_camera_m, int p_budget = 4);
    void clear();

    // The chunks to draw, each with its revision (the renderer caches vertex buffers by it).
    const std::unordered_map<ChunkId, Chunk, ChunkIdHash> &chunks() const { return chunk_map; }

    // Ray march along a unit direction; false when nothing is hit within p_max_m.
    bool raycast(const Float3 &p_origin_m, const Float3 &p_direction, float p_max_m, Float3 &r_hit_m, Float3 &r_normal) const;

    // Edits (metres). Both mark the chunks they touch dirty; update() remeshes them.
    void carve_sphere(const Float3 &p_center_m, float p_radius_m, bool p_subtract, float p_smooth_m = 0.15f);
    void carve_capsule(const Float3 &p_a_m, const Float3 &p_b_m, float p_radius_m, bool p_subtract, float p_smooth_m = 0.15f);
    void clear_edits();

    const DawnSDF &sdf() const { return world_sdf; }
    DawnSDF &sdf() { return world_sdf; }
    int edit_count() const { return world_sdf.edit_count(); }
    int loaded_chunks() const { return int(chunk_map.size()); }
    int pending_chunks() const;

private:
    void mesh_chunk(Chunk &r_chunk);
    void mark_edit_dirty(const Edit &p_edit);

    DawnSDF world_sdf;
    std::unordered_map<ChunkId, Chunk, ChunkIdHash> chunk_map;
    float voxel_m = 0.25f;
    static constexpr int MAX_LOD = 2;
};

} // namespace ov::dawn
