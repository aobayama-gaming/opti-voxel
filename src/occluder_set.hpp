#pragma once

#include "occlusion_map.hpp"
#include "world_coord.hpp"

#include <godot_cpp/templates/hash_map.hpp>

#include <cstdint>
#include <vector>

using namespace godot;

// The occluders of the occlusion test: solid axis-aligned boxes in LOD0 voxels, per chunk. A meshed chunk's solid
// blocks (every sample corner solid: the surface-nets mesh closes around them) are merged into a few boxes; the
// set rasterises them into an OcclusionMap around an eye, front to back. Pure (no GPU), doctested.
class OccluderSet {
public:
    struct Box {
        VoxelPos lo;
        VoxelPos hi;
    };
    // A box of whole blocks: [x0, x1) x [y0, y1) x [z0, z1) in block units.
    struct BlockBox {
        int x0, y0, z0, x1, y1, z1;
    };

    // Greedy merge of an n^3 grid of solid blocks (index (z * n + y) * n + x) into disjoint boxes covering exactly
    // the solid ones: from each unused solid block in index order, the longest run in x, then the most rows of
    // that run in y, then the most such slabs in z.
    static void merge_blocks(const std::vector<uint8_t> &p_solid, int p_n, std::vector<BlockBox> &r_boxes);

    void set(const ChunkId &p_chunk, std::vector<Box> p_boxes);
    void erase(const ChunkId &p_chunk);
    void clear() { boxes.clear(); }
    int64_t box_count() const;

    struct BuildStats {
        int64_t boxes = 0; // in the set
        int64_t vanished = 0; // shrunk to nothing by the margin
        int64_t holding_eye = 0;
        int64_t already_hidden = 0; // behind the boxes rasterised before them
        int64_t rasterised = 0;
        int64_t written = 0; // rasterised boxes that wrote a pixel
    };
    // Clears p_map and fills it with every box relative to the eye (p_eye_base + p_eye, LOD0 voxels), each shrunk
    // by p_margin voxels (hidden from the eye then means hidden from any eye within p_margin of it), nearest
    // first; a box already hidden by those before it is skipped.
    BuildStats build(OcclusionMap &r_map, const VoxelPos &p_eye_base, const OcclusionMap::Vec3 &p_eye, double p_margin) const;

private:
    HashMap<ChunkId, std::vector<Box>, Vector3i64Hasher> boxes;
};
