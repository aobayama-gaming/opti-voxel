#pragma once

#include "cell_sites.hpp"
#include "moment_layers.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

// A procedural instance source: what a zone of the ground holds, one site per 3D cell (cell_sites.hpp), each site
// growing its own records (a tree's parts, a rock). Everything is a pure function of the cell: any box gives the same
// sites and records for the cells it shares with another, so the source can be placed whole (the table at configure)
// or in pieces. The forest and the rock scatter are two of them.
namespace Archetype {

struct CellScatter {
    uint32_t salt = 0; // its sites' (cell_site)
    VoxelBox zone; // the sites' bases lie in it
    int64_t side = 0; // the cells (LOD0 voxels); 0: no site
    FieldFn ground; // thread-safe
    std::function<UpAxis(const VoxelPos &)> up; // thread-safe
    std::function<bool(const VoxelPos &)> accept; // whether a base may hold a site (empty: every one); thread-safe
    // A site's records grown along p_up (op, k, anchor, rotation, params, material, salt set; not finished), for the chunks
    // of LOD p_lod: every record listed there, in the full growth's order, a subset of the full growth bit for bit (the
    // caller filters once they are finished); -1: the full growth. A source that grows less for a coarse LOD (the forest's
    // hull-only trees) reads it; one that doesn't ignores it. Thread-safe.
    std::function<void(const Site &, const UpAxis &, int p_lod, std::vector<TableRecord> &)> generate;

    // The site classes the far moments are kept per: a site's class is its variant modulo classes (the forest: a tree's
    // species, tree_species_of).
    uint32_t classes = 1;
    // Sources with the same key share one SourceProfile (ScatterSource's registry): their generate gives the same records
    // for the same site whatever their zone, ground or density (the profile samples sites at the origin, upright). Empty: a
    // profile of its own.
    std::string profile_key;

    bool valid() const { return zone.valid && side > 0 && ground && up && generate; }
    uint32_t class_index(uint32_t p_variant) const { return classes > 1 ? p_variant % classes : 0u; }
    // The accepted sites whose base lies in p_query and the zone, in cell order (surface_sites; p_snap false: on the
    // coarse plane). On the worker pool (not from inside a pool task, unless it holds an InlineParallelFor).
    void sites(const VoxelBox &p_query, std::vector<CellSite> &r_out, bool p_snap = true) const;
    // A site's records along the up at its base, for LOD p_lod (generate; -1: all).
    void grow(const Site &p_site, int p_lod, std::vector<TableRecord> &r_out) const { generate(p_site, up(p_site.base), p_lod, r_out); }
};

// What a source's sites are as a whole, read from samples of its generate (SAMPLES hashed sites per class at the origin,
// upright, their records finished like the table's): how far a site's records reach and what they weigh far away. Those
// bounds let a piece of the source say which chunks it can touch without growing its neighbours. The bounds are over every
// sample; what a site has lost at a LOD is kept per class (a far proxy stands for the trees of its own species).
struct SourceProfile {
    static constexpr int SAMPLES = 256;
    static constexpr double REACH_SAFETY = 1.25; // the samples' reach, k and margin times this: the sources' sites are not the samples

    double reach = 0.0; // how far any record's box reaches from its site's base (voxels, any up; REACH_SAFETY included)
    int lod_end = 0; // no record of any site is listed at this LOD or above
    float k_max = 0.0f; // the records' largest smooth radius and locality margin: their influence margin at a LOD
    float margin_max = 0.0f;
    double records_per_site = 0.0; // the mean over the samples
    double part_variance = 0.0; // the largest eigenvalue of a record's covariance (voxels^2)
    MomentJoins joins; // the source's (moment_joins over the samples' records)
    // What the mean site of a class has lost at a LOD (its records' moments times their lost shares,
    // MomentItem::lost_share), upright, relative to its base (voxels): what a far proxy stands for, per site.
    struct Lost {
        MaterialMoments moments;
        std::map<uint32_t, std::array<double, 6>> area; // the lost records' summed silhouettes, per material
        double variance = 0.0; // the largest eigenvalue of a material's covariance
    };
    std::vector<std::vector<Lost>> lost; // per class, per LOD

    // What a site of class p_class (CellScatter::class_index) has lost at p_lod.
    const Lost &lost_of(uint32_t p_class, int p_lod) const { return lost[MIN<size_t>(p_class, lost.size() - 1)][p_lod]; }
    // The largest variance lost at p_lod over the classes.
    double lost_variance(int p_lod) const;

    SourceProfile() = default;
    // p_scatter's generate (the full growth) on SAMPLES sites per class (the rest of p_scatter unused). On the worker pool.
    SourceProfile(const DefLibrary &p_library, const CellScatter &p_scatter, const RecordFinish &p_finish, const MomentLayerParams &p_layers, int p_lod_count);
    // Whether a finished record of a site at p_base is within the profile (keys trust it: a record past it may be missed).
    bool covers(const VoxelPos &p_base, const TableRecord &p_record) const;
};

// The bits of a value as text, for a profile key (CellScatter::profile_key and the registry's): equal only for equal values.
template <typename T>
std::string key_bits(const T &p_v) {
    static_assert(sizeof(T) <= 8, "key_bits: a scalar");
    uint64_t b = 0;
    std::memcpy(&b, &p_v, sizeof(T));
    return std::to_string(b) + ",";
}

} // namespace Archetype
