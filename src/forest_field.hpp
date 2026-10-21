#pragma once

#include "forest_generator.hpp"
#include "instance_table.hpp"
#include "moment_layers.hpp"

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <vector>

// A procedural forest as a field of keys: everything it adds to the table (its trees' parts and its moment blobs) comes
// in pieces, each a pure function of its key, so any piece can be made alone, anywhere, at any time, the same. A chunk
// asks for the keys whose records may reach it at its LOD (keys()), and folds their records in key order.
//
// - A tree tile of LOD L (FieldKey::TREES): the cube of TREE_TILE_CELLS forest cells per side; its records are the parts
//   listed at L of the trees whose base lies in it (ForestGenerator: a tree is keyed by its cell), in cell order, each
//   finished like a table record (band rule included). One key per LOD: a tree is 39 parts at LOD 0, 8 at LOD 6, 1.4 at
//   LOD 7 (5 cm), and the LOD-7 chunks reach kilometres out. None from the profile's lod_end on.
// - A blob tile of layer L (the moment layers, moment_layers.hpp): the cube of blob_tile_cells(L) layer-L cells per side;
//   its records are the kept blobs of its own cells, in (cell, material) order. Each tile runs the layers on the items
//   around it (MARGIN_CELLS cells past its own, enough for its cells' sums and their pruning neighbours) with the joins
//   fixed by the profile, and keeps its own cells' blobs:
//   - part layers (a cell under PART_CELL_SIDES forest cells, while parts are listed): the items are the parts of the
//     grown trees around it (the LOD-0 tree tiles it reads, inputs()), the layers' own model: what a whole build gives,
//     bit for bit;
//   - the layers above: a cell holds several trees, then thousands, and reading every part is out of the question (a
//     LOD-7 tile would read ~20k trees, a LOD-14 one millions): the items are proxies, one per cell of 1 / PROXY_PER_CELL
//     layer cell along the surface (surface_sites), each what the profile's mean tree has lost at the layer times the
//     trees its patch of ground holds in the zone. Their cost per tile is the same at any layer and any tree count: a
//     32 km forest's far field costs what a 300 m one's does. Their mass is the trees' within a few percent, their blobs'
//     supports too from LOD 6 on (5 cm); finer, a cell holds few trees and their twigs make thin blobs proxies can't.
namespace Archetype {

// What a forest's trees are as a whole, read from TreeSamples finished like the table's records.
struct ForestProfile {
    static constexpr double REACH_SAFETY = 1.25; // the samples' reach and k times this: the forest's trees are not the samples

    double reach = 0.0; // how far any part's box reaches from its tree's base (voxels, any up; REACH_SAFETY included)
    int lod_end = 0; // no part of any tree is listed at this LOD or above
    float k_max = 0.0f; // the parts' largest smooth radius and locality margin: the trees' influence margin at a LOD
    float margin_max = 0.0f;
    double part_variance = 0.0; // the largest eigenvalue of a part's covariance (voxels^2)
    MomentJoins joins; // the forest's (moment_joins over the samples' parts)
    // What the mean tree has lost at a LOD (its parts' moments times their lost shares, MomentItem::lost_share), upright,
    // relative to its base (voxels): what a far proxy stands for, per tree.
    struct Lost {
        MaterialMoments moments;
        std::map<uint32_t, std::array<double, 6>> area; // the lost parts' summed silhouettes, per material
        double variance = 0.0; // the largest eigenvalue of a material's covariance
    };
    std::vector<Lost> lost; // per LOD

    ForestProfile() = default;
    ForestProfile(const DefLibrary &p_library, const TreeSamples &p_samples, const RecordFinish &p_finish, const MomentLayerParams &p_layers, int p_lod_count);
};

struct FieldKey {
    enum Kind : int {
        TREES,
        BLOBS,
    };
    int kind = TREES;
    int lod = 0; // the LOD its records are listed at (a blob tile: its layer)
    VoxelPos tile; // the tile's index on its kind's grid

    // Fold order: the tree tiles, then the blob tiles, each LOD by LOD, tiles in cell order.
    bool operator<(const FieldKey &p_o) const {
        return kind != p_o.kind ? kind < p_o.kind : (lod != p_o.lod ? lod < p_o.lod : cell_order(tile, p_o.tile));
    }
    bool operator==(const FieldKey &p_o) const { return kind == p_o.kind && lod == p_o.lod && tile == p_o.tile; }
};

// A key's records, finished, in fold order, and their groups (a tree's parts, a blob): a chunk tests a group's box
// before its records.
struct FieldContent {
    struct Group {
        uint32_t first = 0;
        uint32_t count = 0;
        TableRecord bounds; // the members' union box, largest k and margin, LODs listed: influence_meets on it
    };
    std::vector<TableRecord> records;
    std::vector<Group> groups;
    int64_t out_of_profile = 0; // trees with a part past the profile (reach, LODs, margins: chunks may miss it; stays 0)

    // Appends the indices (into records) of the records whose box test at p_lod can pass in [p_lo, p_hi].
    void meeting(const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod, std::vector<uint32_t> &r_out) const {
        meeting(groups, RecordView(records.data()), 0, p_lo, p_hi, p_lod, r_out);
    }
    // The same over p_groups whose records are p_records[p_first + i] (a streamed key's slots), appending p_first + i.
    static void meeting(const std::vector<Group> &p_groups, RecordView p_records, uint32_t p_first, const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod,
            std::vector<uint32_t> &r_out);
};

class ForestField {
public:
    static constexpr int64_t TREE_TILE_CELLS = 8; // forest cells per tree tile side
    static constexpr int64_t BLOB_TILE_CELLS = 4; // layer cells per blob tile side, at least (and a tree tile wide)
    static constexpr int64_t MARGIN_CELLS = 5; // the cells past its own a blob tile reads: tents 1, pruning neighbours 4
    static constexpr int64_t PROXY_PER_CELL = 2; // far proxies per layer cell side
    static constexpr int64_t PART_CELL_SIDES = 3; // a layer reads the parts while its cell is under this many forest cells

    struct Setup {
        std::shared_ptr<const DefLibrary> library;
        TreeTypes types;
        int blob_type = -1; // -1: no moment layers
        double voxel = 0.05;
        RecordFinish finish;
        int64_t max_extent = 0; // record_fit_error's: what the table refuses (SDFTable::max_instance_extent())
        ForestGenerator generator;
        FieldFn ground; // thread-safe
        std::function<UpAxis(const VoxelPos &)> up; // thread-safe
        std::function<bool(const VoxelPos &)> keep_out; // no tree with its base there (lakes); may be empty
        MomentLayerParams layers; // the table's filter, pruning and cover (the LOD range is the field's)
        int lod_count = 15;
    };

    explicit ForestField(const Setup &p_setup);

    const ForestProfile &profile() const { return prof; }
    const Setup &setup() const { return set; }
    int64_t tree_tile_side() const { return TREE_TILE_CELLS * set.generator.side; }
    int64_t layer_cell(int p_layer) const { return int64_t(1) << (p_layer + set.layers.cell_shift); }
    int64_t blob_tile_cells(int p_layer) const;
    // Whether layer p_layer has blob tiles (the joins' first LOD up to the last LOD; none without the band rule or a blob type).
    bool has_layer(int p_layer) const { return has_blobs && p_layer >= first_layer && p_layer < set.lod_count; }
    // Whether a layer's blob tiles read the grown parts (else proxies).
    bool part_layer(int p_layer) const { return p_layer < prof.lod_end && layer_cell(p_layer) < PART_CELL_SIDES * set.generator.side; }

    // The keys whose records' box test at p_lod can pass somewhere in [p_lo, p_hi] (LOD0 voxels, inclusive), in key order.
    void keys(const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod, std::vector<FieldKey> &r_out) const;
    // How far from the ground the records listed at p_lod or finer reach with their influence margins (voxels; 0: none):
    // the discoverer keeps a node of the zone whose corners are this close to the ground (SDFTable::cull_exempt).
    int64_t ground_reach(int p_lod) const;
    // keys() over a chunk's samples.
    void chunk_keys(const ChunkId &p_chunk, const ChunkParam &p_param, std::vector<FieldKey> &r_out) const;
    // The keys p_key's content reads (a coarser tree tile: its LOD-0 tile; a part layer's blob tile: the LOD-0 tree tiles
    // around it), in key order.
    void inputs(const FieldKey &p_key, std::vector<FieldKey> &r_out) const;

    // An input's content if the caller has it (null: computed here, the same).
    using Lookup = std::function<const FieldContent *(const FieldKey &)>;
    // p_key's records. Thread-safe; on the worker pool from inside a pool task, hold an InlineParallelFor.
    void content(const FieldKey &p_key, FieldContent &r_out, const Lookup &p_inputs = nullptr) const;
    // A blob tile's kept blobs (its own cells'), in (cell, material) order: what its records are made of.
    void blobs(const FieldKey &p_key, std::vector<MomentBlob> &r_out, const Lookup &p_inputs = nullptr) const;

private:
    Setup set;
    ForestProfile prof;
    bool has_blobs = false;
    int first_layer = 0;
    float blob_margin_c = 0.0f; // a blob record's locality margin (its type's, the table's skirt)
    std::vector<int64_t> ground_reaches; // ground_reach per LOD
    mutable std::atomic<bool> profile_reported{ false };

    // Appends the keys (p_kind, p_lod) of the tiles of p_side voxels that meet p_box and the zone grown by p_zone_grow,
    // in cell order.
    void append_tiles(int p_kind, int p_lod, int64_t p_side, const VoxelBox &p_box, int64_t p_zone_grow, std::vector<FieldKey> &r_out) const;
    // How far a layer's blobs can reach from their tile's box (voxels; the influence margin at the layer excluded).
    int64_t blob_reach(int p_layer) const;
    // How far a tile's records can pass a box test from its box at a LOD: a tree tile's (from its bases), a blob tile's.
    int64_t tree_grow(int p_lod) const;
    int64_t blob_grow(int p_layer) const;
    VoxelBox tile_box(const FieldKey &p_key) const;
    // A LOD-0 tree tile grows its trees; a coarser one keeps the parts of its LOD-0 tile's listed at its LOD.
    void tree_tile(const FieldKey &p_key, FieldContent &r_out, const Lookup &p_inputs) const;
    void blob_tile(const FieldKey &p_key, FieldContent &r_out, const Lookup &p_inputs) const;
    // The items of a near blob tile (the parts around it) and of a far one (the proxies around it), in key order.
    void near_items(const FieldKey &p_key, const Lookup &p_inputs, std::vector<MomentItem> &r_items) const;
    void far_items(const FieldKey &p_key, std::vector<MomentItem> &r_items) const;
};

} // namespace Archetype
