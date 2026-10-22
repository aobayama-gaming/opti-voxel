#pragma once

#include "archetype_table.hpp"
#include "instance_moments.hpp"
#include "world_coord.hpp"

#include <map>
#include <set>
#include <tuple>
#include <vector>

// The far field of the archetype instances: what the parts a LOD drops still weigh. For each LOD L, the parts gone at L
// or before hand their moments (record_moments) to cells of 2^(L + cell_shift) LOD0 voxels on an absolute grid, per
// material; each cell becomes a blob, a smooth occupancy kernel with the cell's mass and spread, which the LOD L chunks
// fold into a coarse surface (a forest's canopy seen from far above: archetype_table.hpp moment_surface).
//
// A part is splatted onto the 2 x 2 x 2 cells around its centroid with tent weights (each cell's share falls linearly
// to 0 one cell away): the blobs of a uniformly covered area sum to a near-constant occupancy, with no seam where the
// cell edges fall (nearest-cell assignment split a canopy lying across a cell boundary into random halves and cut
// holes along the edges). Each cell's spread is that of the parts' actual positions, so a thin canopy still makes
// thin blobs: it spreads only along the axes the parts are spread along. A cell's moments are summed over its members
// in id order, so the same items give the same bits however they got there (a build or edits).
//
// A blob's kernel is a max(1 - q, 0)^2 with q = sum_i (l_i / s_i)^2, l the point in the blob's principal frame: a
// compact polynomial (mul / add only: the GPU and the CPU agree bit for bit), whose spread per axis is s_i^2 / 9, so
// s_i = 3 sigma_i with sigma_i^2 the cell's variance along that axis plus the LOD's filter (filter * 2^L)^2, the box
// filter of a cell of the LOD (sigma = 1 / sqrt(12) cells by default). a = m0 * 105 / (32 pi s_x s_y s_z): its
// integral is the cell's mass, so a uniformly filled region reads its fill fraction.
//
// Keeping the mass alone, the filter, which grows with the LOD, spread a thin canopy thinner every LOD (under MOMENT_ISO
// past a few: the forest vanished from far away, first at its edges, where a cell holds part of it and looks round).
// From far, a canopy's thickness doesn't show, its cover does: each part also carries an area tensor (its silhouette
// along a unit n is n^T P n: pi r^2 for a sphere, about mass over thickness for a flat part), and a cell's peak is at
// least its cover along its thinnest principal axis, 3 A / (pi s_1 s_2) (the kernel's mid-plane then averages to the
// silhouettes' share of the area), at most 1, and at most 2^(L - MomentJoin::cover_from) times the mass-kept peak
// (the cover takes over one doubling per LOD: seams step no more). A dense canopy reads its cover at every LOD, edges
// included; a lone tree fades once the filter outgrows it. Principal axes only: no up direction.
//
// The hand-off from the parts: a part eroding (the band rule) gives the layers the share of its mass the erosion took,
// its equal-covariance ellipsoid shrunk by the erosion (1 - prod max(a_i - e, 0) / a_i), and all of it from its drop on:
// what the parts draw plus what the blobs hold is the part's mass at every LOD (the forest thinned out between them).
namespace Archetype {

// One part, as the layers see it.
struct MomentItem {
    VoxelPos anchor; // the moments' anchor
    Moments moments; // weight included; empty: the part weighs nothing
    uint32_t material = 0;
    int drop_lod = 0; // the first LOD that no longer draws it
    // Its erosion (archetype_table.hpp band_erosion; erode 0: whole until its drop).
    float erode = 0.0f;
    float erode_from = 1.0f;
    // Set by the layers from the moments: the ellipsoid's semi-axes (ascending) and the area tensor (xx yy zz xy xz yz).
    float semi[3] = {};
    double area[6] = {};

    // The share of its mass the LOD no longer draws (0 .. 1).
    double lost_share(int p_lod) const;
    // The first LOD that loses any of it: its erosion's first, at the latest its drop.
    int first_lost_lod() const;
};

struct MomentLayerParams {
    int cell_shift = 3; // cells of 2^(L + cell_shift) voxels at LOD L: 8 cells of the LOD
    double filter = 0.28867513459481287; // the kernel's added sigma, cells of the LOD (1 / sqrt(12): a box filter)
    // A blob whose occupancy, with its neighbours', stays under this is pruned, and one whose own peak is under 1/16 of
    // it (a sliver of a part's tent: it only spreads its neighbours' surface).
    double min_occupancy = 1.0 / 32.0;
    // A material joins the layers from the first LOD that has lost this share of its mass (0: every part, from its first
    // loss on): before, its lost parts are lone thin parts (a tree's twigs while its crown is drawn), which a blob would
    // draw back fattened to the LOD's filter, at the cost of a blob per tent corner on the fine chunks.
    double min_mass_share = 0.25;
    // The cover ramps in (one doubling per LOD) from the first LOD that has lost this share: before, the blobs keep their
    // mass next to the parts still drawn (a seam steps no more than with the mass kept).
    double cover_mass_share = 0.5;
    // A blob's peak is at least its cell's cover (see above); false: every blob keeps its mass.
    bool keep_cover = true;
    int lod_min = 0; // the layers built
    int lod_max = 14;
};

// A blob of layer lod: a record-to-be (SDFTable places it in the table).
struct MomentBlob {
    int lod = 0;
    uint32_t material = 0;
    int64_t cell[3] = {}; // the cell's index on the layer's grid
    VoxelPos anchor; // the cell's centroid, rounded to the voxel
    float rotation[9] = {}; // world -> local: the principal axes (rows), orthonormal
    float inv_s2[3] = {}; // 1 / s_i^2 (voxels^-2)
    float amplitude = 0.0f; // a: the kernel's peak occupancy
    double s[3] = {}; // the support's semi-axes (voxels)
    double mass = 0.0; // the cell's m0
    double peak_bound = 0.0; // a bound of the occupancy summed over the layer's blobs, anywhere in its support

    // The kernel at a point p_offset voxels from the anchor (double: tests and reports; the fold has its own GLSL / IR).
    double occupancy(const double p_offset[3]) const;
};

// The semi-axes and area tensor of p_item from its moments.
void prepare_moment_item(MomentItem &r_item);

// A blob of one cell's moments and area tensor at layer p_lod (the kernel's frame, size and amplitude); p_anchor: the
// moments' anchor; the cover raises its peak to at most p_max_gain times the mass-kept one (1: the mass kept).
MomentBlob make_blob(const Moments &p_cell, const double p_area[6], const VoxelPos &p_anchor, int p_lod, uint32_t p_material, double p_filter_voxels,
        double p_max_gain);

// Per material, the first LOD that has lost min_mass_share of its mass (the layers it joins) and the first that has lost
// cover_mass_share (where its cover ramps in from).
struct MomentJoin {
    int lod = 0;
    int cover_from = 0;
};
using MomentJoins = std::map<uint32_t, MomentJoin>;
MomentJoins moment_joins(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params);

// A kept blob an update made appear, change or vanish.
struct MomentBlobChange {
    MomentBlob blob; // its lod, cell, material; the rest is the new blob when kept
    bool kept = false;
};

// The layers of a set of items, kept up to date as the items change. Each cell holds its members (the items its tent
// touches, by id), its blob and whether the pruning keeps it. A blob's pruning bound sums the amplitudes of the blobs whose
// support box meets its own, in cell order: an update re-sums the cells an item leaves or joins and recomputes the bounds
// around them, and lands on the same bits as a build of the items as they now are (with the same joins).
class MomentLayers {
public:
    // Every layer from p_items (indexed by id), with the joins they give.
    void build(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params);
    // Item p_id (the table grows as needed) becomes p_item; the kept blobs that appeared, changed or vanished are appended
    // to r_changes.
    void update(uint32_t p_id, const MomentItem &p_item, std::vector<MomentBlobChange> &r_changes);
    // The kept blobs, in (lod, cell, material) order.
    void kept(std::vector<MomentBlob> &r_out) const;
    // A build of the current items with the same joins: what the updates must match (tests).
    void rebuild(std::vector<MomentBlob> &r_out) const;

    const MomentJoins &get_joins() const { return joins; }
    size_t item_count() const { return items.size(); }
    // The box tests the pruning has run so far (builds and updates): its cost, per item flat at a fixed density.
    uint64_t pruning_tests() const { return box_tests; }

private:
    struct CellKey {
        int64_t x = 0, y = 0, z = 0;
        uint32_t material = 0;
        bool operator<(const CellKey &p_o) const { return std::tie(x, y, z, material) < std::tie(p_o.x, p_o.y, p_o.z, p_o.material); }
        bool operator==(const CellKey &p_o) const { return x == p_o.x && y == p_o.y && z == p_o.z && material == p_o.material; }
    };
    // A cell's sum is the sum, in block order, of its members' partial sums per block of 2^BLOCK_SHIFT ids (each in id
    // order): an update re-sums one block and adds the blocks up, a build does the same, so they agree bit for bit.
    static constexpr int BLOCK_SHIFT = 8;
    struct Sum {
        Moments moments;
        double area[6] = {};
    };
    struct Cell {
        std::vector<uint32_t> members; // ascending ids
        std::map<uint32_t, Sum> blocks; // id >> BLOCK_SHIFT -> its members' sum
        MomentBlob blob;
        double extent[3] = {}; // its support's world-axis half extents
        bool kept = false;
        bool placed = false; // in the bins (or the wide set): it has a blob
    };
    using Bin = std::tuple<int64_t, int64_t, int64_t>;
    using Near = std::pair<CellKey, const Cell *>; // a cell and where it lives (map nodes don't move)
    struct Layer {
        int lod = 0;
        int64_t cell = 0; // voxels
        std::map<CellKey, Cell> cells;
        std::map<Bin, std::vector<Near>> bins; // the blobs whose support box meets each bin of BIN_CELLS cells
        std::set<CellKey> wide; // the blobs whose box meets more than WIDE_BINS bins
    };
    // A blob is filed in every bin its support box meets, so two boxes that meet share a bin. A blob over more than
    // WIDE_BINS bins (none in a forest: a tall trunk's blob spans tens) is tested by every blob instead.
    static constexpr int64_t BIN_CELLS = 4;
    static constexpr double WIDE_BINS = 512.0;
    // A blob under this many times min_occupancy on its own is a sliver of a part's tent.
    static constexpr double SLIVER = 1.0 / 16.0;

    std::vector<MomentItem> items;
    MomentLayerParams params;
    MomentJoins joins;
    std::vector<Layer> layers; // lod_min .. lod_max
    uint64_t box_tests = 0;

    bool on_layer(const MomentItem &p_item, int p_lod) const;
    // The cells p_item's tent touches on p_layer with a weight > 0, and the weights.
    int tent(const MomentItem &p_item, const Layer &p_layer, CellKey r_keys[8], double r_weights[8]) const;
    // The partial sum of block p_block of the cell's members (in id order).
    void resum_block(const Layer &p_layer, const CellKey &p_key, uint32_t p_block, Cell &r_cell) const;
    // The cell's blob from its block sums (in block order), and its extents.
    void resum(const Layer &p_layer, const CellKey &p_key, Cell &r_cell) const;
    // The bins a support box at p_anchor with half extents p_extent meets (r_lo .. r_hi per axis); false when they are
    // more than WIDE_BINS.
    bool bin_range(const Layer &p_layer, const VoxelPos &p_anchor, const double p_extent[3], Bin &r_lo, Bin &r_hi) const;
    void place(Layer &r_layer, const CellKey &p_key, Cell &r_cell);
    void unplace(Layer &r_layer, const CellKey &p_key, Cell &r_cell);
    // The cells whose blob support box may meet one at p_anchor with half extents p_extent (a superset, sorted by key).
    void candidates(const Layer &p_layer, const VoxelPos &p_anchor, const double p_extent[3], std::vector<Near> &r_out) const;
    // The cell's pruning bound and verdict from the current blobs; the box tests it ran.
    uint64_t judge(const Layer &p_layer, const CellKey &p_key, Cell &r_cell) const;
    // A build, in two steps: the members (reads only the items: the layers fill theirs side by side), then the sums,
    // the bins and the pruning (each cell's on the worker pool, the bins in cell order).
    void fill_members(Layer &r_layer) const;
    void finish_layer(Layer &r_layer);
    void build_layer(Layer &r_layer);
    // update() on one layer: item p_id was p_before (items[p_id] holds its new state).
    void update_layer(Layer &layer, uint32_t p_id, const MomentItem &p_before, std::vector<MomentBlobChange> &r_changes);
};

// A record's item: its moments, material and the first LOD it is gone at (dropped, or eroded past its half thickness);
// empty moments for one that weighs nothing (removed, a cutter, a blob, never dropped).
MomentItem record_item(const DefLibrary &p_library, const TableRecord &p_record);
// A blob's record (op ACCUM, type p_blob_type, listed at its layer's LOD alone).
TableRecord blob_record(const DefLibrary &p_library, int p_blob_type, const MomentBlob &p_blob, float p_skirt_depth);

// The kept blobs of p_items' layers (a MomentLayers build), in (lod, cell, material) order.
void build_moment_blobs(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params, std::vector<MomentBlob> &r_out);

} // namespace Archetype
