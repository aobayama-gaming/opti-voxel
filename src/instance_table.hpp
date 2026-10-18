#pragma once

#include "archetype_table.hpp"
#include "chunk_math.hpp"

#include <godot_cpp/templates/hash_map.hpp>

#include <memory>
#include <vector>

// The instance table: a versioned list of TableRecords with a spatial index over the chunk-id octree.
//
// An instance is stored at the finest LOD whose chunk span holds its influence box (bare box + the
// locality margin), in the 1 to 8 chunks that box touches (a loose octree). A chunk's list is the
// instances stored in its subtree (own + descendants) plus those stored at its strict ancestors,
// merged in id (insertion) order: any instance whose box contains one of the chunk's samples is in
// the list, wherever its node sits. Readers take a Snapshot (an immutable version); the SDF keeps the
// current one and hands it out.
namespace Archetype {

struct InstanceSnapshot {
    std::shared_ptr<const DefLibrary> library;
    Program ground; // the ground register (archetype_table.hpp fold)
    float out_scale = 1.0f;
    bool normalize = true;
    uint32_t version = 0;
    int max_lod = 0;
    ChunkParam param; // the chunk geometry (chunk span per LOD)
    std::vector<TableRecord> records;
    std::vector<uint32_t> unbounded; // instances with an unbounded box: listed by every chunk
    // node -> the instances stored at it
    HashMap<ChunkId, std::vector<uint32_t>, Vector3i64Hasher> stored_at;
    // node -> every instance in its subtree (own + descendants)
    HashMap<ChunkId, std::vector<uint32_t>, Vector3i64Hasher> subtree;

    // The instances a chunk's samples can see, in id order.
    void chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const;
    // The instances whose box test at p_lod can pass for p_voxel, in id order (the chunk list filtered
    // to the instances actually on the point's node path: same set, same order, cheaper to gather).
    void point_candidates(const VoxelPos &p_voxel, std::vector<uint32_t> &r_out) const;
};

// Builds the index and the subtree lists. Deterministic. p_max_lod bounds the octree depth.
std::shared_ptr<const InstanceSnapshot> build_snapshot(std::shared_ptr<const DefLibrary> p_library, const Program &p_ground,
        float p_out_scale, bool p_normalize, std::vector<TableRecord> p_records, const ChunkParam &p_param, int p_max_lod,
        uint32_t p_version);

} // namespace Archetype
