#pragma once

#include "archetype_glsl.hpp"
#include "archetype_instance.hpp"
#include "exact_noise.hpp"

#include <godot_cpp/variant/packed_byte_array.hpp>

#include <functional>
#include <memory>
#include <vector>

// Step 3 of the archetype work: instances live in a table instead of being compiled into the terrain.
//
// A library type (TableDef) is a Program written once in the record's frame (T_INPUT / T_PARAM /
// T_PERLIN): its sizes are parameters, its anchor, rotation and noise lattice come from the instance's
// record. generate_library_glsl() turns it into `vec4 arch_<k>(WorldPos p, int lod, DefRecord rec)`,
// the CPU interpreter runs the same program with a DefRecord (archetype_ir.hpp). A TableRecord is the
// CPU record (the GPU reads the same fields from a DefRecord buffer: encode_record).
//
// The terrain is F(p, lod) = fold over the instances whose influence box contains p, in table order, of
// r = op_i(r, c_i(p)), starting from the ground (a compiled Program). The influence box is the
// instance's world box grown by M(lod) = ceil(k + margin_c * 2^lod) + 2 voxels (margin_c = 2 L +
// skirt_depth). Outside it the instance's value is >= M, so the fold changes F only where |F| > 2 L
// 2^lod: the mesh (and vertex normals) are the same as without the rule. The test is integer (a saturating
// sat_diff of the voxel against the anchor), so the CPU and the GPU and two chunks of the same LOD take
// the same branch and seams stay exact.
namespace Archetype {

// A library type: a program in the record's frame and a bounds function of its parameters.
struct TableDef {
    Program program;
    String name;
    int param_count = 0;
    bool bounded = true;
    // Local (unrotated) box and Lipschitz bound (voxels) for the given params.
    std::function<void(const float *p_params, Float3 &r_lo, Float3 &r_hi, float &r_lipschitz)> bounds;
};

using TableDefPtr = std::shared_ptr<const TableDef>;

// The types available to the table (a new type = a shader recompile). GLSL functions arch_<index>.
class DefLibrary {
public:
    int add(const TableDefPtr &p_def);
    int size() const { return int(list.size()); }
    const TableDef &operator[](int p_index) const { return *list[p_index]; }
    const std::vector<TableDefPtr> &all() const { return list; }

private:
    std::vector<TableDefPtr> list;
};

// One instance, as the CPU keeps it. The GPU reads the same fields from a DefRecord (encode_record).
struct TableRecord {
    uint8_t def = 0; // library index
    bool removed = false; // an edit removed it: kept in the array (stable ids), skipped by the index
    Combine op = Combine::UNION;
    float k = 0.0f; // smooth radius (voxels)
    float inv_k = 0.0f; // 1 / k, precomputed (never divided on the GPU)
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
    float erode = 0.0f;
    float erode_from = 1.0f;

    static constexpr uint8_t NEVER_DROPPED = 255;
    bool listed_at(int p_lod) const { return p_lod < max_lod; }

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

// The band rule for instances. At LOD L an instance is eroded by rate * max(2^L - 2^onset, 0) LOD0 voxels: from
// its onset LOD on it shrinks by `rate` cells per LOD, so the surfaces of a LOD L and a LOD L+1 chunk differ by at
// most `rate` cells of L, under the fine chunk's skirt (skirt_depth >= rate): no crack at a seam. It is dropped
// (no longer listed) from the first LOD whose erosion reaches its depth plus k (band_depth), where the eroded
// shape no longer reaches zero, so dropping it changes no sign: the drop itself is invisible. Its mean (nothing,
// or an aggregate for a cluster) stands in for it. p_enabled false (--instance_band=0) keeps it whole at every
// LOD: the reference F_ref. Needs the record's world box (finish_record sets it first).
// rate (--instance_erode, default ERODE_CELLS): any rate > 0 keeps both properties, a lower one keeps thin parts
// to coarser LODs (a 0.3 m trunk at 5 cm voxels: gone at LOD 3 at 1, at LOD 4 at 0.5).
// resolve (--instance_resolve, cells, 0 = onset 0: erodes from LOD 1): the onset is the coarsest LOD whose cells
// still fit `resolve` times in the part's half thickness (band_radius), so a part erodes only once the grid can
// no longer draw it whole, whatever its size: a 0.3 m trunk at 5 cm stays whole to LOD 2 at resolve 1.
inline constexpr float ERODE_CELLS = 1.0f;
struct ErodeRule {
    float rate = ERODE_CELLS;
    float resolve = 0.0f;
};
void set_band(TableRecord &r_record, const TableDef &p_def, bool p_enabled, const ErodeRule &p_rule = ErodeRule());
// The erosion at p_lod, in voxels: what both back-ends add to the record's distance.
inline float band_erosion(float p_erode, float p_erode_from, int p_lod) {
    return p_erode * MAX(ExactNoise::exact_pow2(p_lod) - p_erode_from, 0.0f);
}

// How deep inside its shape a point can be, plus k: half its box's smallest side times L. Outside the box the
// value is >= 0 and every point inside is within half that side of its boundary. The smaller of the local box
// (its type's bounds, grown by rotation_margin on each side) and the world box: the local one doesn't grow
// when the record turns (a diagonal branch's world box is far wider than the branch).
double band_depth(const TableDef &p_def, const TableRecord &p_record);
// Half the smallest side of that box (voxels): the part's half thickness, which sets its onset.
double band_radius(const TableDef &p_def, const TableRecord &p_record);

// The record's box (relative to the anchor), L and margin from its type's bounds, then its band.
void finish_record(const TableDef &p_def, TableRecord &r_record, float p_skirt_depth, bool p_band, const ErodeRule &p_rule = ErodeRule());

// The DefRecord struct (with the sdf_records / sdf_list buffer declarations): spliced with the frame
// structs, before the bindings, by the SDF's GPU data. The fold helpers (arch_in_box, arch_combine) go
// with the library, in get_glsl_code().
extern const char GLSL_DEF_RECORD_STRUCT[];

String generate_library_glsl(const DefLibrary &p_library);
// The mesher's terrain_sdf: the ground register, the fold, then the epilogue (out_scale / normalize).
String generate_table_sdf(const Program &p_ground, const DefLibrary &p_library, float p_out_scale, bool p_normalize);

// Bytes of one DefRecord in the GPU buffer (10 x vec4, all 16-byte aligned).
inline constexpr uint32_t RECORD_BYTES = 160;
void encode_record(PackedByteArray &r_bytes, uint32_t p_index, const TableRecord &p_record);

// M(lod) = ceil(min(k + margin_c * 2^lod, INPUT_LIMIT)) + 2 voxels: the influence box's growth at p_lod.
// The CPU and the GPU (arch_in_box) compute the same float expression, so the same branch.
int influence_margin(const TableRecord &p_record, int p_lod);

// The locality rule: whether p_pos is inside p_record's influence box at LOD p_lod. The CPU mirror of
// arch_in_box (same sat_diff, same integer branch).
bool in_influence_box(const TableRecord &p_record, const WorldPos &p_pos, int p_lod);

// The fold: r = ground, then each listed instance (in order, its box test), the epilogue applied.
// Returns the final register (gradient, value). p_list holds the instance indices in table order.
// p_box_test false folds every instance (the locality rule's reference, M1b).
Float4 fold(const Program &p_ground, const DefLibrary &p_library, RecordView p_records, const uint32_t *p_list,
        uint32_t p_count, const WorldPos &p_pos, int p_lod, bool p_box_test = true);
// fold() then the epilogue (archetype_ir.cpp finish()).
Float4 finish_table(const Float4 &p_out, int p_lod, float p_out_scale, bool p_normalize);

// Library types. Parameters (voxels): sphere {radius}; box {hx, hy, hz}; capsule {height, radius};
// branch {height, radius, slope} (a capsule tapering by slope voxels of radius per voxel of height);
// cylinder {half_height, radius}; rock {radius, amp} (an fbm-displaced sphere); cone {radius, height, cos, sin,
// amp} (a cone on y = 0, its apex at y = height, cos / sin of its half angle from the CPU, displaced like the
// rock: a conifer's tier); swiss_cube
// {hx, hy, hz, ring_radius, hole_radius} (a box with p_holes spherical holes, a composite through the
// table).
TableDefPtr make_sphere_type();
TableDefPtr make_box_type();
TableDefPtr make_capsule_type();
TableDefPtr make_branch_type();
TableDefPtr make_cylinder_type();
TableDefPtr make_rock_type(int p_k, uint32_t p_seed);
TableDefPtr make_cone_type(int p_k, uint32_t p_seed);
TableDefPtr make_swiss_cube_type(int p_holes);

// The influence box (world axes, int64, absolute) of a record: the type's local box, rotated by the
// record's rotation, rounded out, plus the record's anchor. Used by the table's spatial index.
Bounds record_world_bounds(const TableDef &p_def, const TableRecord &p_record);

// The steepest field a runtime edit may add (the library's shapes are ~1-2 voxels per voxel).
inline constexpr float MAX_RECORD_LIPSCHITZ = 1024.0f;
// Why a runtime edit's record doesn't fit the table (empty if it does): an anchor beyond the world-origin
// bound (+-2^60 voxels, so int64 boxes around it can't overflow), a non-finite type box or a Lipschitz bound
// above MAX_RECORD_LIPSCHITZ, or a box (under any rotation) plus twice the smooth radius k wider than
// p_max_extent voxels. Checked before finish_record, whose int64 / int32 conversions of the box assume it,
// and whose index places one entry per top-LOD node the box touches.
String record_fit_error(const TableDef &p_def, const TableRecord &p_record, int64_t p_max_extent);

} // namespace Archetype
