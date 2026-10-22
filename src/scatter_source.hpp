#pragma once

#include "cell_scatter.hpp"
#include "chunk_math.hpp"
#include "instance_table.hpp"

#include <memory>
#include <vector>

// A procedural source (cell_scatter.hpp) as a field of keys: everything it adds to the table (its sites' records and
// their moment blobs) comes in pieces, each a pure function of its key, so any piece can be made alone, anywhere, at any
// time, the same. A chunk asks for the keys whose records may reach it at its LOD (keys()) and folds their records in key
// order.
//
// - A part tile of LOD L (SourceKey::PARTS): a cube of part_tile_side(L) voxels. At LOD 0 it holds the records of the
//   sites whose base lies in it (CellScatter::grow, finished like the table's), in cell order, all of them (a record no
//   LOD lists still weighs in the blobs). Coarser, a pyramid: its side is PART_TILE_CELLS source cells doubled until it
//   holds two chunks of the LOD, and its records are those of the L - 1 tiles it covers (1 or 8) listed at L. A chunk
//   reads a few keys at any LOD; none from the profile's lod_end on.
// - A blob tile of layer L (SourceKey::BLOBS): a cube of blob_tile_cells(L) layer-L cells (moment_layers.hpp); its
//   records are the kept blobs of its own cells, in (cell, material) order. It runs the layers on the items within
//   MARGIN_CELLS cells of it (a cell's verdict reads the cells within PRUNE_CELLS of it, their sums the items whose tents
//   touch them) with the joins fixed by the profile, and judges only its own cells:
//   - part layers (a cell under PART_CELL_SIDES source cells, while records are listed): the items are the records of the
//     LOD-0 part tiles around it (inputs()), the layers' own model: a whole build's blobs, the sums in another order;
//   - the layers above: a cell holds several sites, then thousands, and reading every record is out of the question: the
//     items are proxies, one per cell of 1 / PROXY_PER_CELL layer cell along the surface (surface_sites, unsnapped), each
//     what the profile's mean site has lost at the layer times the sites its patch of ground holds in the zone. Their
//     cost per tile is the same at any layer and any site count.
namespace Archetype {

struct SourceKey {
    enum Kind : int {
        PARTS,
        BLOBS,
    };
    uint32_t source = 0; // its source's id (ScatterSource::Setup::id)
    int kind = PARTS;
    int lod = 0; // the LOD its records are listed at (a blob tile: its layer)
    VoxelPos tile; // the tile's index on its kind's grid at that LOD

    // Fold order: by source, the part tiles, then the blob tiles, each LOD by LOD, tiles in cell order.
    bool operator<(const SourceKey &p_o) const {
        if (source != p_o.source) {
            return source < p_o.source;
        }
        return kind != p_o.kind ? kind < p_o.kind : (lod != p_o.lod ? lod < p_o.lod : cell_order(tile, p_o.tile));
    }
    bool operator==(const SourceKey &p_o) const { return source == p_o.source && kind == p_o.kind && lod == p_o.lod && tile == p_o.tile; }
};

// A key's records, finished, in fold order, and their groups (a site's records, a blob): a chunk tests a group's box
// before its records.
struct SourceContent {
    struct Group {
        uint32_t first = 0;
        uint32_t count = 0;
        TableRecord bounds; // the members' union box, largest k and margin, LODs listed: influence_meets on it
    };
    std::vector<TableRecord> records;
    std::vector<Group> groups;
    int64_t out_of_profile = 0; // sites with a record past the profile (reach, LODs, margins: chunks may miss it; stays 0)

    // Appends the indices (into records) of the records whose box test at p_lod can pass in [p_lo, p_hi].
    void meeting(const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod, std::vector<uint32_t> &r_out) const {
        meeting(groups, RecordView(records.data()), 0, p_lo, p_hi, p_lod, r_out);
    }
    // The same over p_groups whose records are p_records[p_first + i] (a streamed key's slots), appending p_first + i.
    static void meeting(const std::vector<Group> &p_groups, RecordView p_records, uint32_t p_first, const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod,
            std::vector<uint32_t> &r_out);
};

class ScatterSource {
public:
    static constexpr int64_t PART_TILE_CELLS = 8; // source cells per LOD-0 part tile side
    static constexpr int64_t BLOB_TILE_CELLS = 8; // layer cells per blob tile side, at least
    static constexpr int64_t MARGIN_CELLS = MomentLayers::PRUNE_CELLS + 1; // the cells past its own a blob tile reads
    static constexpr int64_t PROXY_PER_CELL = 2; // far proxies per layer cell side
    static constexpr int64_t PART_CELL_SIDES = 3; // a layer reads the records while its cell is under this many source cells
    static constexpr uint32_t PROXY_SALT = 0x9A0C5u; // the proxies' sites: the source's salt xor this

    struct Setup {
        uint32_t id = 0; // its keys' source
        std::shared_ptr<const DefLibrary> library;
        CellScatter scatter;
        int blob_type = -1; // -1: no moment layers
        RecordFinish finish;
        MomentLayerParams layers; // the table's filter, pruning and cover (the LOD range is the source's)
        int64_t max_extent = 0; // record_fit_error's: what the table refuses (SDFTable::max_instance_extent())
        int64_t chunk_voxels = 28; // a LOD-0 chunk's side (ChunkParam::chunk_voxel_dimension())
        int lod_count = 15;
    };

    // Reads the profile (on the worker pool).
    explicit ScatterSource(const Setup &p_setup);

    const SourceProfile &profile() const { return prof; }
    const Setup &setup() const { return set; }
    int64_t part_tile_side(int p_lod) const { return part_sides[MIN(MAX(p_lod, 0), int(part_sides.size()) - 1)]; }
    int64_t layer_cell(int p_layer) const { return int64_t(1) << (p_layer + set.layers.cell_shift); }
    int64_t blob_tile_cells(int p_layer) const;
    // Whether layer p_layer has blob tiles (the joins' first LOD up to the last LOD; none without the band rule or a blob type).
    bool has_layer(int p_layer) const { return has_blobs && p_layer >= first_layer && p_layer < set.lod_count; }
    // Whether a layer's blob tiles read the grown records (else proxies).
    bool part_layer(int p_layer) const { return p_layer < prof.lod_end && layer_cell(p_layer) < PART_CELL_SIDES * set.scatter.side; }
    VoxelBox tile_box(const SourceKey &p_key) const;

    // The keys whose records' box test at p_lod can pass somewhere in [p_lo, p_hi] (LOD0 voxels, inclusive), in key order.
    void keys(const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod, std::vector<SourceKey> &r_out) const;
    // keys() over a chunk's samples.
    void chunk_keys(const ChunkId &p_chunk, const ChunkParam &p_param, std::vector<SourceKey> &r_out) const;
    // How far from the ground the records listed at p_lod or finer reach with their influence margins (voxels; 0: none):
    // the discoverer keeps a node of the zone whose corners are this close to the ground (SDFTable::cull_exempt).
    int64_t ground_reach(int p_lod) const;
    // The keys p_key's content reads, in key order: a coarser part tile's L - 1 tiles, a part layer's blob tile the LOD-0
    // part tiles around it; none otherwise.
    void inputs(const SourceKey &p_key, std::vector<SourceKey> &r_out) const;

    // The contents of inputs(p_key), in its order; an entry missing or null is computed here (the same).
    using Inputs = std::vector<const SourceContent *>;
    // p_key's records. Thread-safe; from inside a pool task, hold an InlineParallelFor (the site walks).
    void content(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const;
    // A blob tile's kept blobs (its own cells'), in (cell, material) order: what its records are made of.
    void blobs(const SourceKey &p_key, const Inputs &p_inputs, std::vector<MomentBlob> &r_out) const;

private:
    Setup set;
    SourceProfile prof;
    bool has_blobs = false;
    int first_layer = 0;
    float blob_margin_c = 0.0f; // a blob record's locality margin (its type's, the table's skirt)
    std::vector<int64_t> part_sides; // part_tile_side per LOD
    std::vector<int64_t> ground_reaches; // ground_reach per LOD

    // Appends the keys (p_kind, p_lod) of the tiles of p_side voxels that meet p_box and the zone grown by p_zone_grow,
    // in cell order.
    void append_tiles(int p_kind, int p_lod, int64_t p_side, const VoxelBox &p_box, int64_t p_zone_grow, std::vector<SourceKey> &r_out) const;
    // How far a layer's blobs can reach from their tile's box (voxels; the influence margin at the layer excluded).
    int64_t blob_reach(int p_layer) const;
    // How far a tile's records can pass a box test from its box at a LOD: a part tile's (from its bases), a blob tile's.
    int64_t part_grow(int p_lod) const;
    int64_t blob_grow(int p_layer) const;
    // Input p_i of p_inputs, else computed into r_local.
    const SourceContent &input(const SourceKey &p_key, const Inputs &p_inputs, size_t p_i, SourceContent &r_local) const;
    // A LOD-0 part tile grows its sites; a coarser one keeps the records of the tiles under it listed at its LOD.
    void part_tile(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const;
    void blob_tile(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const;
    // The items of a part layer's blob tile (the records around it) and of a far one (the proxies around it).
    void near_items(const SourceKey &p_key, const Inputs &p_inputs, std::vector<MomentItem> &r_items) const;
    void far_items(const SourceKey &p_key, std::vector<MomentItem> &r_items) const;
};

} // namespace Archetype
