#pragma once

#include "archetype_table.hpp"
#include "chunk_math.hpp"
#include "ov/containers.hpp"

#include <godot_cpp/templates/hash_map.hpp>
#include <godot_cpp/templates/hash_set.hpp>

#include <memory>
#include <vector>

// The instance table: a versioned list of TableRecords with a spatial index over the chunk-id octree.
//
// An instance is stored once, at the finest LOD whose chunk span holds its influence box (bare box + the
// locality margin), in the node holding its centre (wider than a node even at the top LOD: in every node it
// touches). A chunk's samples reach past its span (the skirt overhang) and its influence box grows with the
// chunk's LOD, so a chunk's list gathers the subtrees of the 125 chunks within 2 of it and, for bigger
// instances stored coarser, the 27 nodes around its ancestor at each coarser LOD, and keeps the instances
// whose influence box at the chunk's LOD meets the chunk's sample box, in id (insertion) order: exactly the
// instances whose box test can pass at one of its samples.
// Readers take a Snapshot (an immutable version); the SDF keeps the current one and hands it out.
//
// Two index layers: the base, built once over the whole table (a million instances take seconds), and an
// overlay over the records edited since, rebuilt by every edit (milliseconds). A base entry whose record
// was edited is masked; the overlay holds its current state. Lists merge both in id order, so the fold is
// the same as from one index over the whole table. Past REBASE edited records, an edit rebuilds the base.
namespace Archetype {

struct IndexRange {
    uint32_t begin = 0;
    uint32_t count = 0;
};

// Node -> IndexRange, open addressing (linear probing) in one array: insert-only, built once per layer. A
// million instances put millions of nodes in it, and a map that allocates per entry spends seconds there.
class NodeRangeMap {
public:
    void reserve(size_t p_count);
    void insert(const ChunkId &p_node, const IndexRange &p_range); // p_node must not be there yet
    const IndexRange *getptr(const ChunkId &p_node) const;
    size_t size() const { return count; }
    template <typename F>
    void for_each(F p_f) const {
        for (const Slot &s : slots) {
            if (s.used) {
                p_f(s.node, s.range);
            }
        }
    }

private:
    struct Slot {
        ChunkId node;
        IndexRange range;
        bool used = false;
    };
    static uint64_t hash(const ChunkId &p_node) {
        return (uint64_t(p_node.x) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(p_node.y) * 0xC2B2AE3D27D4EB4Full) ^ (uint64_t(p_node.z) * 0x165667B19E3779F9ull);
    }
    std::vector<Slot> slots;
    size_t mask = 0;
    size_t count = 0;
};

// One layer of the index over a set of records (flat arrays: no allocation per node).
struct IndexLayer {
    std::vector<uint32_t> unbounded; // instances with an unbounded box: listed by every chunk
    // node -> the instances stored at it (stored_ids[begin, begin + count))
    NodeRangeMap stored_at;
    std::vector<uint32_t> stored_ids;
    // node -> every instance in its subtree (own + descendants) still listed at the node's LOD (the band rule)
    NodeRangeMap subtree;
    std::vector<uint32_t> subtree_ids;
    // Per LOD, sorted: every node whose subtree stores an instance, band or not (near_instance).
    std::vector<std::vector<ChunkId>> occupied;

    bool is_occupied(const ChunkId &p_node, int p_lod) const;
};

struct InstanceSnapshot {
    std::shared_ptr<const DefLibrary> library;
    Program ground; // the ground register (archetype_table.hpp fold)
    float out_scale = 1.0f;
    bool normalize = true;
    uint32_t version = 0;
    int max_lod = 0;
    ChunkParam param; // the chunk geometry (chunk span per LOD)
    RecordStore records;
    std::shared_ptr<const IndexLayer> base; // over the records [0, base_count) as they were when it was built
    uint32_t base_count = 0;
    std::shared_ptr<const IndexLayer> overlay; // over overlay_ids, their current state
    std::vector<uint32_t> overlay_ids; // sorted: every record edited (added, moved, removed) since the base
    ov::HashSet<uint32_t> masked; // the base ids among overlay_ids: their base entries are stale

    static constexpr uint32_t REBASE = 2048;

    // The instances whose box test can pass at one of the chunk's samples, in id order.
    void chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const;
    // The instances whose box test at p_lod can pass at p_voxel, in id order.
    void point_candidates(const VoxelPos &p_voxel, int p_lod, std::vector<uint32_t> &r_out) const;
    // The discoverer's no-cull rule: whether an instance's bare box (at any LOD it is listed at or not) may
    // meet the node's box grown by 2 chunk spans of its LOD. Conservative to a node (an occupied neighbour
    // counts whole, a masked base entry still counts).
    bool near_instance(const ChunkId &p_node, int p_lod) const;

private:
    // The instances whose influence box at p_lod meets [p_lo, p_hi] (LOD0 voxels, inclusive), gathered
    // from the 27 LOD-p_lod chunks around p_centre.
    void gather(const ChunkId &p_centre, int p_lod, const VoxelPos &p_lo, const VoxelPos &p_hi, std::vector<uint32_t> &r_out) const;
    bool near_layer(const IndexLayer &p_layer, const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod) const;
};

// The box (LOD0 voxels, inclusive) of a chunk's vertex-grid samples: its span plus the skirt overhang
// (ChunkMath::sample_pos of grid indices 0 .. chunk_size - 1).
void chunk_sample_box(const ChunkId &p_chunk, const ChunkParam &p_param, VoxelPos &r_lo, VoxelPos &r_hi);

// Whether p_record's influence box at p_lod meets [p_lo, p_hi]: the box form of in_influence_box.
bool influence_meets(const TableRecord &p_record, int p_lod, const VoxelPos &p_lo, const VoxelPos &p_hi);

// Builds the snapshot with a base index over every record. Deterministic. p_max_lod bounds the octree depth.
std::shared_ptr<const InstanceSnapshot> build_snapshot(std::shared_ptr<const DefLibrary> p_library, const Program &p_ground,
        float p_out_scale, bool p_normalize, RecordStore p_records, const ChunkParam &p_param, int p_max_lod, uint32_t p_version);
inline std::shared_ptr<const InstanceSnapshot> build_snapshot(std::shared_ptr<const DefLibrary> p_library, const Program &p_ground,
        float p_out_scale, bool p_normalize, const std::vector<TableRecord> &p_records, const ChunkParam &p_param, int p_max_lod,
        uint32_t p_version) {
    return build_snapshot(p_library, p_ground, p_out_scale, p_normalize, RecordStore(p_records, p_version), p_param, p_max_lod, p_version);
}

// The next snapshot after an edit of the records p_changed (added, moved or removed; p_records is the table
// after it): the previous base is kept and the overlay rebuilt, or past REBASE edited records, a new base.
std::shared_ptr<const InstanceSnapshot> edit_snapshot(const InstanceSnapshot &p_previous, RecordStore p_records,
        const std::vector<uint32_t> &p_changed, uint32_t p_version);

} // namespace Archetype
