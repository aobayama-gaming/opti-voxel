#pragma once

#include "archetype_table.hpp"
#include "chunk_math.hpp"

#include <godot_cpp/templates/hash_map.hpp>

#include <memory>
#include <vector>

// The instance table: a versioned list of TableRecords with a spatial index over the chunk-id octree.
//
// An instance is stored at the finest LOD whose chunk span holds its influence box (bare box + the
// locality margin), in the 1 to 8 chunks that box touches (a loose octree). A chunk's samples reach past
// its span (the skirt overhang) and at a coarser LOD an instance's influence box outgrows the nodes it is
// stored in, so a chunk's list gathers the 27 chunks around it (each one's subtree plus its strict
// ancestors' own) and keeps the instances whose influence box at the chunk's LOD meets the chunk's sample
// box, in id (insertion) order: exactly the instances whose box test can pass at one of its samples.
// Readers take a Snapshot (an immutable version); the SDF keeps the current one and hands it out.
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

    // The instances whose box test can pass at one of the chunk's samples, in id order.
    void chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const;
    // The instances whose box test at p_lod can pass at p_voxel, in id order.
    void point_candidates(const VoxelPos &p_voxel, int p_lod, std::vector<uint32_t> &r_out) const;

private:
    // The instances whose influence box at p_lod meets [p_lo, p_hi] (LOD0 voxels, inclusive), gathered
    // from the 27 LOD-p_lod chunks around p_centre.
    void gather(const ChunkId &p_centre, int p_lod, const VoxelPos &p_lo, const VoxelPos &p_hi, std::vector<uint32_t> &r_out) const;
};

// The box (LOD0 voxels, inclusive) of a chunk's vertex-grid samples: its span plus the skirt overhang
// (ChunkMath::sample_pos of grid indices 0 .. chunk_size - 1).
void chunk_sample_box(const ChunkId &p_chunk, const ChunkParam &p_param, VoxelPos &r_lo, VoxelPos &r_hi);

// Whether p_record's influence box at p_lod meets [p_lo, p_hi]: the box form of in_influence_box.
bool influence_meets(const TableRecord &p_record, int p_lod, const VoxelPos &p_lo, const VoxelPos &p_hi);

// Builds the index and the subtree lists. Deterministic. p_max_lod bounds the octree depth.
std::shared_ptr<const InstanceSnapshot> build_snapshot(std::shared_ptr<const DefLibrary> p_library, const Program &p_ground,
        float p_out_scale, bool p_normalize, std::vector<TableRecord> p_records, const ChunkParam &p_param, int p_max_lod,
        uint32_t p_version);

} // namespace Archetype
