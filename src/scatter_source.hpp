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
// - Walks, no records (the costly part: the ground evaluated cell by cell; small, kept resident), never listed by a chunk:
//   - a site tile (SourceKey::SITES, lod 0), on the LOD-0 part-tile grid: the sites whose base lies in the tile
//     (CellScatter::sites);
//   - a proxy tile (SourceKey::PROXIES) of a far layer L, on its blob-tile grid: the proxies' walk of its own box; a blob
//     tile reads the proxy tiles its region meets (far_items), so each is walked once for the blob tiles around it.
// - A part tile of LOD L (SourceKey::PARTS): a cube of part_tile_side(L) voxels, its side PART_TILE_CELLS source cells
//   doubled until it holds two chunks of the LOD. It grows the sites of the site tiles inside it and the zone (CellScatter::grow,
//   finished like the table's), recursively by its 1 or 8 L - 1 tiles (z y x) down to LOD 0, and keeps one group
//   per site: at LOD 0 every record (a record no LOD lists still weighs in the blobs), coarser the ones listed at L. A LOD
//   whose side is a finer LOD's has no key of its own: its chunks read that LOD's (part_key_lod) and filter at their LOD.
//   A chunk reads a few keys at any LOD; none from the profile's lod_end on.
// - A blob tile of layer L (SourceKey::BLOBS): a cube of blob_tile_cells(L) layer-L cells (moment_layers.hpp); its
//   records are the kept blobs of its own cells, in (cell, material) order. It runs the layers on the items within
//   MARGIN_CELLS cells of it (a cell's verdict reads the cells within PRUNE_CELLS of it, their sums the items whose tents
//   touch them) with the joins fixed by the profile, and judges only its own cells:
//   - part layers (a cell under Setup::part_cell_sides source cells, while records are listed): the items are the
//     records of the LOD-0 part tiles around it (inputs()), the layers' own model: a whole build's blobs, the sums in
//     another order;
//   - the layers above: a cell holds several sites, then thousands, and reading every record is out of the question: the
//     items are proxies, one per cell of 1 / PROXY_PER_CELL layer cell along the surface (surface_sites, unsnapped, the
//     PROXIES tiles around it), each what the profile's mean site has lost at the layer times the sites its patch of ground
//     holds in the zone. Their cost per tile is the same at any layer and any site count.
namespace Archetype {

struct SourceKey {
    enum Kind : int {
        PARTS,
        BLOBS,
        SITES,
        PROXIES,
        KIND_COUNT,
    };
    static const char *kind_name(int p_kind) {
        static constexpr const char *NAMES[KIND_COUNT] = { "parts", "blobs", "sites", "proxies" };
        return NAMES[p_kind];
    }
    // A walk (no records, never listed): a site or proxy tile.
    bool walk() const { return kind == SITES || kind == PROXIES; }
    uint32_t source = 0; // its source's id (ScatterSource::Setup::id)
    int kind = PARTS;
    int lod = 0; // the LOD its records are listed at (a blob or proxy tile: its layer; a site tile: 0)
    VoxelPos tile; // the tile's index on its kind's grid at that LOD

    // Fold order: by source, the part tiles, then the blob tiles, each LOD by LOD, tiles in cell order (the walks last:
    // never folded).
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
    std::vector<CellSite> sites; // a site or proxy tile's (no records), in cell order
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
    // A layer reads the records while its cell is under this many source cells (Setup::part_cell_sides): the whole build's
    // blobs, bit for bit; the streamed sources take proxies from the source cell on (their tiles read fewer part tiles).
    static constexpr int64_t PART_CELL_SIDES = 3;
    static constexpr int64_t STREAM_PART_CELL_SIDES = 1;
    static constexpr int64_t STREAM_PART_CELL_SIDES_MAX = 8; // VoxelTerrain.archetype_stream_part_sides at most
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
        int64_t part_cell_sides = PART_CELL_SIDES;
    };

    // Reads the profile (on the worker pool).
    explicit ScatterSource(const Setup &p_setup);

    const SourceProfile &profile() const { return prof; }
    const Setup &setup() const { return set; }
    int64_t part_tile_side(int p_lod) const { return part_sides[MIN(MAX(p_lod, 0), int(part_sides.size()) - 1)]; }
    // The LOD of the part keys a chunk of p_lod reads: the finest with its tile side.
    int part_key_lod(int p_lod) const { return part_key_lods[MIN(MAX(p_lod, 0), int(part_key_lods.size()) - 1)]; }
    int64_t layer_cell(int p_layer) const { return int64_t(1) << (p_layer + set.layers.cell_shift); }
    int64_t blob_tile_cells(int p_layer) const;
    // Whether layer p_layer has blob tiles (the joins' first LOD up to the last LOD; none without the band rule or a blob type).
    bool has_layer(int p_layer) const { return has_blobs && p_layer >= first_layer && p_layer < set.lod_count; }
    // Whether a layer's blob tiles read the grown records (else proxies).
    bool part_layer(int p_layer) const { return p_layer < prof.lod_end && layer_cell(p_layer) < set.part_cell_sides * set.scatter.side; }
    VoxelBox tile_box(const SourceKey &p_key) const;
    // The side of a layer's proxy cells (voxels).
    int64_t proxy_side(int p_layer) const { return MAX<int64_t>(1, layer_cell(p_layer) / PROXY_PER_CELL); }
    // Where a layer's proxies weigh: the zone grown by a proxy cell (outside it a proxy weighs nothing).
    VoxelBox proxy_zone(int p_layer) const { return set.scatter.zone.grown(proxy_side(p_layer)); }
    // The box a far layer's blob tile reads proxies in: its cells and MARGIN_CELLS past them, plus the records' reach, in
    // proxy_zone.
    VoxelBox proxy_region(const SourceKey &p_key) const;
    // The level a layer's proxy walks start at (surface_sites' p_min_level): an unclipped region's, so a proxy tile keeps
    // the sites a region walk would over it, by construction.
    int proxy_level(int p_layer) const;

    // The keys whose records' box test at p_lod can pass somewhere in [p_lo, p_hi] (LOD0 voxels, inclusive), in key order.
    void keys(const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod, std::vector<SourceKey> &r_out) const;
    // keys() over a chunk's samples.
    void chunk_keys(const ChunkId &p_chunk, const ChunkParam &p_param, std::vector<SourceKey> &r_out) const;
    // How far from the ground the records listed at p_lod or finer reach with their influence margins (voxels; 0: none):
    // the discoverer keeps a node of the zone whose corners are this close to the ground (SDFTable::cull_exempt).
    int64_t ground_reach(int p_lod) const;
    // The keys p_key's content reads: a part tile the site tiles inside it that meet the zone (recursively by L - 1 tiles,
    // z y x), a part layer's blob tile the LOD-0 part tiles around it (key order), a far layer's the proxy tiles its
    // region meets (proxy_region, key order); none otherwise.
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
    std::vector<int> part_key_lods; // part_key_lod per LOD
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
    // Appends the site tiles inside part tile p_tile of LOD p_lod that meet the zone: recursively its L - 1 tiles meeting
    // it, z y x.
    void append_part_sites(const VoxelPos &p_tile, int p_lod, std::vector<SourceKey> &r_out) const;
    // How far a far blob tile's region reaches past its box.
    int64_t proxy_grow(int p_layer) const;
    // A walk's sites: a site tile's, or a proxy tile's (its own box, in proxy_zone).
    void walk_tile(const SourceKey &p_key, SourceContent &r_out) const;
    // A part tile grows the sites of its inputs: every record at LOD 0, the ones listed at its LOD coarser.
    void part_tile(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const;
    // Grows one site into a group of r_out (the records listed at p_lod; all of them: p_lod < 0), unless the table would
    // refuse one of its records; counts it out of the profile if a record is past it.
    void grow_site(const CellSite &p_site, int p_lod, std::vector<TableRecord> &r_grown, SourceContent &r_out) const;
    void blob_tile(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const;
    // The items of a part layer's blob tile (the records around it) and of a far one (the proxies around it).
    void near_items(const SourceKey &p_key, const Inputs &p_inputs, std::vector<MomentItem> &r_items) const;
    void far_items(const SourceKey &p_key, const Inputs &p_inputs, std::vector<MomentItem> &r_items) const;
};

} // namespace Archetype
