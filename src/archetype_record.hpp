#pragma once

#include "archetype_ir.hpp"
#include "material_ids.hpp"
#include "world_coord.hpp"

#include <memory>
#include <utility>
#include <vector>

// The archetype table's records (archetype_table.hpp): what one instance is on the CPU, the pages they live in, and
// the locality rule's boxes. Engine-free: the table's spatial index (instance_table.hpp) is built from these alone.
namespace Archetype {

// A world box (LOD0 voxels, int64, inclusive) and a Lipschitz bound: outside the box the value is >= the distance
// to it (voxels), and the gradient's length is <= L (per LOD0 voxel).
struct Bounds {
    VoxelPos lo;
    VoxelPos hi;
    float lipschitz = 1.0f;
};

// How a record combines with the register folded so far.
enum class Combine : uint8_t {
    UNION, // min(r, c)
    SMOOTH_UNION, // smooth_min, radius k
    SUBTRACT, // max(r, -c)
    SMOOTH_SUBTRACT, // smooth_max(r, -c), radius k
    ACCUM, // a moment blob: c is an occupancy, summed into phi; the fold's epilogue draws phi (moment_surface)
};

// The operators that cut the register instead of adding to it.
inline bool is_cutter(Combine p_op) {
    return p_op == Combine::SUBTRACT || p_op == Combine::SMOOTH_SUBTRACT;
}

// One instance, as the CPU keeps it. The GPU reads the same fields from a DefRecord (encode_record).
struct TableRecord {
    uint8_t def = 0; // library index
    bool removed = false; // an edit removed it: kept in the array (stable ids), skipped by the index
    Combine op = Combine::UNION;
    float k = 0.0f; // smooth radius (voxels)
    float inv_k = 0.0f; // 1 / k, precomputed (never divided on the GPU): set both with set_k
    VoxelPos anchor;
    float rotation[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    float params[8] = {};
    uint32_t salt = 0; // noise salt (the seed the T_PERLIN ops are salted with)
    int32_t box_lo[3] = {}; // influence box, relative to the anchor, world axes
    int32_t box_hi[3] = {};
    float margin_c = 0.0f; // 2 * L + skirt_depth (voxels); M(lod) = ceil(k + margin_c * 2^lod) + 2
    float lipschitz = 1.0f; // the instance's L (metrics / tests)
    // The band rule (set_band): listed by the chunks of LOD below max_lod only, and eroded at LOD L by
    // erode * max(2^L - erode_from, 0) voxels (its distance raised: the shape shrinks uniformly once the
    // grid can no longer draw it whole; erode_from = 2^onset).
    uint8_t max_lod = NEVER_DROPPED;
    uint8_t min_lod = 0; // listed from this LOD on (a moment blob: its layer's LOD only)
    float erode = 0.0f;
    float erode_from = 1.0f;
    // Global material (material_library.hpp), by default its type's: CPU only, the GPU sees its chunk palette index
    // in the list word (GpuTable::build_lists). A sample takes the material of the record that wins the fold there
    // (record_wins).
    uint32_t material = Material::FROM_TYPE;

    static constexpr uint8_t NEVER_DROPPED = 255;
    bool listed_at(int p_lod) const { return p_lod >= min_lod && p_lod < max_lod; }
    void set_k(float p_k) {
        k = p_k;
        inv_k = p_k > 0.0f ? float(1.0 / double(p_k)) : 0.0f;
    }

    DefRecord def_record() const;
};

// The records of a table, in pages of PAGE records shared between snapshots: an edit copies the one page it
// changes (copy on write), not the whole table, and each page remembers the version it last changed in, so the
// GPU re-uploads only the pages changed since the version it has (changed_since).
class RecordStore {
public:
    static constexpr uint32_t PAGE = 4096;

    RecordStore() = default;
    explicit RecordStore(const std::vector<TableRecord> &p_records, uint32_t p_version = 0);

    uint32_t size() const { return count; }
    const TableRecord &operator[](uint32_t p_i) const { return pages[p_i / PAGE]->r[p_i % PAGE]; }
    void push_back(const TableRecord &p_record, uint32_t p_version);
    void set(uint32_t p_i, const TableRecord &p_record, uint32_t p_version);
    void clear() {
        pages.clear();
        count = 0;
    }
    // The runs of records (first, count) of the pages changed after p_version (all of them when p_version is
    // UINT32_MAX, nothing sent yet); consecutive changed pages make one run.
    void changed_since(uint32_t p_version, std::vector<std::pair<uint32_t, uint32_t>> &r_runs) const;

private:
    struct Page {
        std::vector<TableRecord> r;
        uint32_t version = 0;
    };
    // The page p_page, owned by this store alone (cloned if a snapshot shares it).
    Page &own(uint32_t p_page);
    std::vector<std::shared_ptr<Page>> pages; // shared with the snapshots; mutated only when owned alone
    uint32_t count = 0;
};

// The records fold() reads: a plain array (tests, probes) or a RecordStore.
class RecordView {
public:
    RecordView(const TableRecord *p_flat) :
            flat(p_flat) {}
    RecordView(const RecordStore &p_store) :
            store(&p_store) {}
    const TableRecord &operator[](uint32_t p_i) const { return flat ? flat[p_i] : (*store)[p_i]; }

private:
    const TableRecord *flat = nullptr;
    const RecordStore *store = nullptr;
};

// M(lod) = ceil(min(k + margin_c * 2^lod, INPUT_LIMIT)) + 2 voxels: the influence box's growth at p_lod.
// The CPU and the GPU (arch_in_box) compute the same float expression, so the same branch.
int influence_margin(const TableRecord &p_record, int p_lod);

// The locality rule: whether p_pos is inside p_record's influence box at LOD p_lod. The CPU mirror of
// arch_in_box (same sat_diff, same integer branch).
bool in_influence_box(const TableRecord &p_record, const WorldPos &p_pos, int p_lod);
// Whether the influence box at p_lod may hold a point of [p_lo, p_hi] (the CPU mirror of arch_meets_box: the same
// sat_diff, so it saturates where arch_in_box does). sat_diff keeps the order, so a point inside a box it rejects is
// rejected by in_influence_box too: a workgroup's samples skip only records none of them would fold.
bool in_influence_cube(const TableRecord &p_record, const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod);

// A finished record's bare box (anchor + box_lo .. box_hi), absolute LOD0 voxels, inclusive.
inline VoxelBox record_box(const TableRecord &p_record) {
    return VoxelBox{ p_record.anchor + VoxelPos(p_record.box_lo[0], p_record.box_lo[1], p_record.box_lo[2]),
        p_record.anchor + VoxelPos(p_record.box_hi[0], p_record.box_hi[1], p_record.box_hi[2]), true };
}

} // namespace Archetype
