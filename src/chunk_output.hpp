#pragma once

#include "palette_block.hpp"
#include "world_coord.hpp"

#include <cstdint>

struct ChunkOutput;

// The owner of a chunk's render resources: the mesher that made it (GPUDirectMesher: a quad range inside a shared
// MultiMesh page; GPUMesher: its own mesh and instance, held under the chunk's handle). The core holds outputs and
// hands them back to their owner to free.
struct ChunkReleaser {
    virtual ~ChunkReleaser() = default;
    virtual void release_chunk(const ChunkOutput &p_chunk) = 0;
    // The chunk's hidden sub-cubes (PageFormat's coverage texel; 0 draws all), from its current rows.
    virtual void set_hidden(const ChunkOutput &p_chunk, uint64_t p_mask) = 0;
};

struct ChunkOutput {
    ChunkId chunk_id;
    uint32_t triangle_count = 0; // stats only: real (non-degenerate) triangles
    uint32_t drawn_triangles = 0; // stats only: triangles in the index buffer, padding included

    // Who frees it (null: an empty chunk, nothing to free) and its key there (owner-defined).
    ChunkReleaser *owner = nullptr;
    uint64_t handle = 0;
    // GPU direct path: quad range in a pool page (page >= 0).
    int32_t page = -1;
    uint32_t quad_offset = 0;
    uint32_t quad_count = 0;
    PaletteBlock palette; // the chunk's material palette (its offset is in the rows' headers)

    void free() {
        if (owner) {
            owner->release_chunk(*this);
            owner = nullptr;
        }
    }
};
