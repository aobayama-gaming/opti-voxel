#pragma once

#include "debug_log.hpp"
#include "godot_cpp/variant/variant.hpp"
#include "palette_pool.hpp"
#include "world_coord.hpp"
#include <godot_cpp/classes/rendering_server.hpp>

using namespace godot;

struct ChunkOutput;

// Implemented by meshers whose chunks do not own their render resources (GPUDirectMesher: the
// chunk is a quad range inside a shared MultiMesh page).
struct ChunkReleaser {
    virtual ~ChunkReleaser() = default;
    virtual void release_chunk(const ChunkOutput &p_chunk) = 0;
    // The chunk's hidden sub-cubes (PageFormat's coverage texel; 0 draws all), from its current rows.
    virtual void set_hidden(const ChunkOutput &p_chunk, uint64_t p_mask) = 0;
};

struct ChunkOutput {
    ChunkId chunk_id;
    RID mesh_rid;
    RID instance_rid;
    uint32_t triangle_count = 0; // stats only: real (non-degenerate) triangles
    uint32_t drawn_triangles = 0; // stats only: triangles in the index buffer, padding included

    // GPU direct path: quad range in a pool page (owner != nullptr).
    ChunkReleaser *owner = nullptr;
    int32_t page = -1;
    uint32_t quad_offset = 0;
    uint32_t quad_count = 0;
    PalettePool::Block palette; // the chunk's material palette (its offset is in the rows' headers)

    void free() {
        if (owner) {
            owner->release_chunk(*this);
            owner = nullptr;
            return;
        }
        RenderingServer *rs = RenderingServer::get_singleton();
        // Empty chunks have no mesh/instance.
        if (instance_rid.is_valid()) {
            rs->free_rid(instance_rid);
            DebugLog::chunk_rids_freed++;
        }
        if (mesh_rid.is_valid()) {
            rs->free_rid(mesh_rid);
        }
        instance_rid = RID();
        mesh_rid = RID();
    }
};
