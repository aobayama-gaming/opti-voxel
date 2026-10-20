#pragma once

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
namespace Archetype {

// One part, as the layers see it.
struct MomentItem {
    VoxelPos anchor; // the moments' anchor
    Moments moments; // weight included; empty: the part weighs nothing
    uint32_t material = 0;
    int drop_lod = 0; // the first LOD that no longer draws it
};

struct MomentLayerParams {
    int cell_shift = 3; // cells of 2^(L + cell_shift) voxels at LOD L: 8 cells of the LOD
    double filter = 0.28867513459481287; // the kernel's added sigma, cells of the LOD (1 / sqrt(12): a box filter)
    // A blob whose occupancy, with its neighbours', stays under this is pruned, and one whose own peak is under 1/16 of
    // it (a sliver of a part's tent: it only spreads its neighbours' surface).
    double min_occupancy = 1.0 / 32.0;
    // A material joins the layers from the first LOD that has dropped this share of its mass (0: every dropped part, from
    // its drop on): before, its dropped parts are lone thin parts (a tree's twigs while its crown is drawn), which a blob
    // would draw back fattened to the LOD's filter, at the cost of a blob per tent corner on the fine chunks.
    double min_mass_share = 0.5;
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

// A blob of one cell's moments at layer p_lod (the kernel's frame, size and amplitude); p_anchor: the moments' anchor.
MomentBlob make_blob(const Moments &p_cell, const VoxelPos &p_anchor, int p_lod, uint32_t p_material, double p_filter_voxels);

// Per material, the first LOD that has dropped min_mass_share of its mass: the layers it joins.
using MomentJoins = std::map<uint32_t, int>;
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
    struct Cell {
        std::vector<uint32_t> members; // ascending ids
        std::map<uint32_t, Moments> blocks; // id >> BLOCK_SHIFT -> its members' sum
        MomentBlob blob;
        double extent[3] = {}; // its support's world-axis half extents
        bool kept = false;
        bool placed = false; // in the bins (or the wide set): it has a blob
    };
    using Bin = std::tuple<int64_t, int64_t, int64_t>;
    struct Layer {
        int lod = 0;
        int64_t cell = 0; // voxels
        std::map<CellKey, Cell> cells;
        std::map<Bin, std::vector<CellKey>> bins; // the near blobs by bin of 2 NEAR_CELLS cells
        std::set<CellKey> wide; // the blobs reaching further than NEAR_CELLS cells
    };
    // Blobs reaching at most this many cells from their anchor (nearly all) are binned; the rest are tested by every blob.
    static constexpr int64_t NEAR_CELLS = 2;
    // A blob under this many times min_occupancy on its own is a sliver of a part's tent.
    static constexpr double SLIVER = 1.0 / 16.0;

    std::vector<MomentItem> items;
    MomentLayerParams params;
    MomentJoins joins;
    std::vector<Layer> layers; // lod_min .. lod_max

    bool on_layer(const MomentItem &p_item, int p_lod) const;
    // The cells p_item's tent touches on p_layer with a weight > 0, and the weights.
    int tent(const MomentItem &p_item, const Layer &p_layer, CellKey r_keys[8], double r_weights[8]) const;
    // The partial sum of block p_block of the cell's members (in id order).
    void resum_block(const Layer &p_layer, const CellKey &p_key, uint32_t p_block, Cell &r_cell) const;
    // The cell's blob from its block sums (in block order), and its extents.
    void resum(const Layer &p_layer, const CellKey &p_key, Cell &r_cell) const;
    Bin bin_of(const Layer &p_layer, const VoxelPos &p_v) const;
    void place(Layer &r_layer, const CellKey &p_key, Cell &r_cell);
    void unplace(Layer &r_layer, const CellKey &p_key, Cell &r_cell);
    // The cells whose blob support box may meet one at p_anchor with half extents p_extent (a superset, sorted).
    void candidates(const Layer &p_layer, const VoxelPos &p_anchor, const double p_extent[3], std::vector<CellKey> &r_out) const;
    // The cell's pruning bound and verdict from the current blobs.
    void judge(const Layer &p_layer, const CellKey &p_key, Cell &r_cell) const;
    void build_layer(Layer &r_layer);
    // update() on one layer: item p_id was p_before (items[p_id] holds its new state).
    void update_layer(Layer &layer, uint32_t p_id, const MomentItem &p_before, std::vector<MomentBlobChange> &r_changes);
};

// The kept blobs of p_items' layers (a MomentLayers build), in (lod, cell, material) order.
void build_moment_blobs(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params, std::vector<MomentBlob> &r_out);

} // namespace Archetype
