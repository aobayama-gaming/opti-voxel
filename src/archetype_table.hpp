#pragma once

#include "archetype_glsl.hpp"
#include "archetype_record.hpp"
#include "exact_noise.hpp"
#include "instance_moments.hpp"
#include "material_library.hpp"
#include "ov/bytes.hpp"
#include "ov/math.hpp"

#include <functional>
#include <memory>
#include <vector>

// The archetype table: the instances placed on the terrain (SDFTable), each chunk folding only the ones near it.
//
// A library type (TableDef) is a Program written once in the record's frame (T_INPUT / T_PARAM /
// T_PERLIN): its sizes are parameters, its anchor, rotation and noise lattice come from the instance's
// record. generate_library_glsl() turns it into `vec4 arch_<k>(WorldPos p, int lod, DefRecord rec)`,
// the CPU interpreter runs the same program with a DefRecord (archetype_ir.hpp). A TableRecord is the
// CPU record (the GPU reads the same fields from a DefRecord buffer: encode_record).
//
// The terrain is F(p, lod) = fold over the instances whose influence box contains p, in table order, of
// r = op_i(r, c_i(p)), starting from the ground's register (SDFCascade). The influence box is the
// instance's world box grown by M(lod) = ceil(k + margin_c * 2^lod) + 2 voxels (margin_c = 2 L +
// skirt_depth). Outside it the instance's value is >= M, so the fold changes F only where |F| > 2 L
// 2^lod: the mesh (and vertex normals) are the same as without the rule. The test is integer (a saturating
// sat_diff of the voxel against the anchor), so the CPU and the GPU and two chunks of the same LOD take
// the same branch and seams stay exact.
namespace Archetype {

// The y of the ground's top solid voxel in column (x, z), instances left out (SDFTable::ground_y): where the scenes stand.
using GroundFn = std::function<int64_t(int64_t, int64_t)>;

// Bounds of a perlin octave (ExactNoise): |value| and |gradient| per cell. Conservative; the unit tests
// sample them.
inline constexpr float PERLIN3_MAX = 1.5f;
inline constexpr float PERLIN3_GRAD_MAX = 7.0f;

// The far field drawn from the moment blobs (moment_layers.hpp) a chunk lists: with phi their summed occupancy (value
// and gradient) and h = 2^lod, F = min(r - h phi, (MOMENT_ISO - phi) h / max(|grad phi|, MOMENT_ISO / 64)). The first
// term lifts the surface under mass too thin to draw (phi < MOMENT_ISO: by at most that many cells), the second draws a surface of its own
// where the occupancy reaches MOMENT_ISO (a canopy), a first-order distance (64 cells where phi is flat), at most
// MOMENT_DEPTH_CELLS deep: at a near-flat maximum of phi just over the iso it read 10 cells next to a sample one cell
// away on the surface (a step no skirt hides), and no 1-Lipschitz field reads past one cell next to a crossing. MOMENT_ISO is
// an opacity more than a volume: a canopy half its crowns' volume reads opaque from afar. The samples where phi >= half
// of it take the strongest blob's material.
inline constexpr float MOMENT_ISO = 0.25f;
inline constexpr float MOMENT_MATERIAL_ISO = 0.5f * MOMENT_ISO;
inline constexpr float MOMENT_DEPTH_CELLS = 2.0f;
inline constexpr float MOMENT_GRAD_FLOOR = (MOMENT_ISO / 64.0f) * (MOMENT_ISO / 64.0f); // of |grad phi|^2, per cell

// The margin world_bounds adds on each side for the float rotation: one voxel, plus R^T's distance to R^-1
// (rows orthonormal within 1e-5, SDFTable::edit_error) times the local box's reach (its largest |x| + |y| + |z|).
double rotation_margin(Float3 p_lo, Float3 p_hi);

// The world box (LOD0 voxels, int64, inclusive) of a local box (p_lo, p_hi) placed at p_anchor with rotation p_r:
// the corners rotated back (R^T q) and rounded out, plus rotation_margin.
Bounds world_bounds(Float3 p_lo, Float3 p_hi, float p_lipschitz, const VoxelPos &p_anchor, const float *p_r);

// A library type: a program in the record's frame and a bounds function of its parameters.
struct TableDef {
    Program program;
    ov::str name;
    int param_count = 0;
    // Local (unrotated) box and Lipschitz bound (voxels) for the given params.
    std::function<void(const float *p_params, Float3 &r_lo, Float3 &r_hi, float &r_lipschitz)> bounds;
    // Its moments in its own frame, relative to the anchor (instance_moments.hpp), for the given params: the solid
    // without its bumps (zero-mean noise). Unset: record_moments integrates the program on a grid.
    std::function<Moments(const float *p_params)> moments;
};

using TableDefPtr = std::shared_ptr<const TableDef>;

// The types available to the table (a new type = a shader recompile). GLSL functions arch_<index>.
class DefLibrary {
public:
    // p_material: the type's default material (a record with Material::FROM_TYPE takes it). Generic shapes inherit.
    int add(const TableDefPtr &p_def, uint32_t p_material = Material::INHERIT);
    int size() const { return int(list.size()); }
    const TableDef &operator[](int p_index) const { return *list[p_index]; }
    const std::vector<TableDefPtr> &all() const { return list; }
    uint32_t material(int p_index) const { return materials[p_index]; }

private:
    std::vector<TableDefPtr> list;
    std::vector<uint32_t> materials;
};

// The band rule for instances: a part is drawn whole while the grid resolves it, and erodes only over the LODs just
// before the one that drops it. At LOD L it is eroded by rate * max(2^L - 2^onset, 0) LOD0 voxels: from its onset
// on it shrinks by `rate` cells per LOD, so the surfaces of a LOD L and a LOD L+1 chunk differ by at most `rate`
// cells of L, under the fine chunk's skirt (skirt_depth >= rate): no crack at a seam. It is dropped (no longer
// listed) from max_lod, where the erosion reaches its depth plus k (band_depth): the eroded shape no longer reaches
// zero, so dropping it changes no sign and the drop itself is invisible. Its mean (an aggregate for a cluster)
// stands in for it. p_enabled false (--instance_band=0) keeps it whole at every LOD: the reference F_ref. Needs the
// record's world box (finish_record sets it first).
// max_lod is the later of the first LOD whose cells no longer fit `resolve` times in the part's half thickness (its
// moments' equal-covariance ellipsoid, record_moments: where the grid stops resolving it) and the first LOD by which
// erosion at `rate` from LOD 0 can reach its depth. The onset is the latest LOD from which erosion at `rate` still
// reaches the depth at max_lod: the part shrinks only in the band that is about to disappear.
// rate (--instance_erode, default ERODE_CELLS): the skirt's depth and the fastest a part may shrink; a higher one keeps
// parts whole to later onsets, a lower one spreads the erosion over more LODs.
// resolve (--instance_resolve, cells, default 1): a part is kept while its half thickness holds that many cells (0:
// dropped as soon as erosion can make it vanish).
inline constexpr float ERODE_CELLS = 1.0f;
struct ErodeRule {
    float rate = ERODE_CELLS;
    float resolve = 1.0f;
};
void set_band(TableRecord &r_record, const TableDef &p_def, bool p_enabled, const ErodeRule &p_rule = ErodeRule());
// The erosion at p_lod, in voxels: what both back-ends add to the record's distance.
inline float band_erosion(float p_erode, float p_erode_from, int p_lod) {
    return p_erode * ov::max(ExactNoise::exact_pow2(p_lod) - p_erode_from, 0.0f);
}

// How deep inside its shape a point can be, plus k: half its box's smallest side times L. Outside the box the
// value is >= 0 and every point inside is within half that side of its boundary. The smaller of the local box
// (its type's bounds, grown by rotation_margin on each side) and the world box: the local one doesn't grow
// when the record turns (a diagonal branch's world box is far wider than the branch).
double band_depth(const TableDef &p_def, const TableRecord &p_record);

// The record's box (relative to the anchor), L and margin from its type's bounds, then its band.
void finish_record(const TableDef &p_def, TableRecord &r_record, float p_skirt_depth, bool p_band, const ErodeRule &p_rule = ErodeRule());

// The DefRecord struct (with the sdf_records / sdf_list buffer declarations): spliced with the frame
// structs, before the bindings, by the SDF's GPU data. The fold helpers (arch_in_box, arch_combine) go
// with the library, in get_glsl_code().
extern const char GLSL_DEF_RECORD_STRUCT[];

ov::str generate_library_glsl(const DefLibrary &p_library);
// The library, `vec4 arch_fold(vec4 r, WorldPos p, int lod, inout bool touched, inout uint material, inout bool moments)` (fold_onto
// over the chunk's list: g_sdf_chunk, sdf_records, sdf_list from GpuTable::glsl_buffers, a list word being the record
// id | its palette index << SDFGpuTable::LIST_ID_BITS; touched is set when an instance passes its box test, material
// to the palette index of a record that wins, unless it is ChunkPalette::INHERIT_INDEX) and `vec4 arch_finish(vec4 o, int lod)` (finish_table).
// p_group_size 0: each invocation walks the whole list. Otherwise (a multiple of 32, the shader's local size) it is
// `arch_fold(vec4 r, WorldPos p, WorldPos group_lo, WorldPos group_hi, int lod, ...)`, called by every invocation of the
// workgroup in uniform control flow, [group_lo, group_hi] holding all their p: the workgroup tests the list p_group_size
// words at a time against that box (arch_meets_box) and each invocation folds the words that pass, in list order, with
// the same loop body: the same bits.
ov::str generate_fold_glsl(const DefLibrary &p_library, float p_out_scale, bool p_normalize, uint32_t p_group_size = 0);

// Bytes of one DefRecord in the GPU buffer (10 x vec4, all 16-byte aligned).
inline constexpr uint32_t RECORD_BYTES = 160;
void encode_record(uint8_t *r_slot, const TableRecord &p_record);

// Whether record c's side wins combine4(r, c) (the sample takes the record's material): the side min / max picks,
// and for the smooth operators the side their hard min picks. The same comparisons as the operators, so the
// distances are untouched.
bool record_wins(const Float4 &p_r, const Float4 &p_c, Combine p_op);

// A record's material: p_material, or its type's (p_type_material) when Material::FROM_TYPE. Material::INHERIT only
// holds for the cutters (subtractions): a union that inherited would take whatever record won the sample before it
// in list order (a rock touching a crown turning green), so it is ground.
uint32_t record_material(uint32_t p_material, uint32_t p_type_material, Combine p_op);

// The fold onto the ground's register p_r: each listed instance (in order, its box test) combined in. Returns the
// register (gradient, value) before the epilogue. p_list holds the instance indices in table order. p_box_test
// false folds every instance (the locality rule's reference, M1b). *r_touched (if given) is set when an instance
// passed its box test; untouched, the result is p_r bit for bit. *r_material (if given, with p_palette: the list
// entries' palette indices) is the palette index of the last record that won (record_wins), else left as it was.
// *r_blobs (if given) is set when a moment blob passed its box test (its sample takes the moment skirt).
Float4 fold_onto(const Float4 &p_r, const DefLibrary &p_library, RecordView p_records, const uint32_t *p_list, uint32_t p_count,
        const WorldPos &p_pos, int p_lod, bool p_box_test = true, bool *r_touched = nullptr, const uint16_t *p_palette = nullptr,
        uint32_t *r_material = nullptr, bool *r_blobs = nullptr);
// The moment blobs' surface on register p_r (the end of fold_onto when a blob passed its box test; GLSL arch_moments):
// p_phi their summed occupancy, p_best the strongest one's palette index (ChunkPalette::INHERIT_INDEX: none), which
// *r_material takes where p_phi >= MOMENT_MATERIAL_ISO.
Float4 moment_surface(const Float4 &p_r, const Float4 &p_phi, int p_lod, uint32_t p_best, uint32_t *r_material);
// The epilogue after fold_onto (archetype_ir.cpp finish()).
Float4 finish_table(const Float4 &p_out, int p_lod, float p_out_scale, bool p_normalize);

// Library types. Parameters (voxels): sphere {radius}; box {hx, hy, hz}; capsule {height, radius};
// branch {height, radius, slope} (a capsule tapering by slope voxels of radius per voxel of height);
// cylinder {half_height, radius}; rock {radius, amp} (an fbm-displaced sphere); cone {radius, height, cos, sin,
// amp} (a cone on y = 0, its apex at y = height, cos / sin of its half angle from the CPU, displaced like the
// rock: a conifer's tier); swiss_cube {hx, hy, hz, ring_radius, hole_radius} (a box with p_holes spherical holes on
// a ring, a composite through the table; ring 0 and one hole: a cube holed through every face). The shapes
// gallery's hard-CSG composites (SDFTable::Scene::SHAPES): die {h} (a cube of half size h rounded by a sphere, its
// pips subtracted: 1 on +y, 2 on +z, 3 on +x, 4 on -x, 5 on -z); stairs {half_width, rise, half_depth} (8 steps
// rising along +z, step i a box of half height (i + 1) rise centred on y = 0); tunnel_cube {h, t} (a cube of half
// size h tunnelled through along its three axes, tunnels of half width t). Biome features (biome_features.hpp): basin
// {half_height, radius, amp} (a cylinder displaced like the rock: a lava lake's crater); island {radius, height, cos,
// sin, dome_radius, dome_drop, amp} (a spherical cap on a hanging cone, displaced like the rock).
TableDefPtr make_sphere_type();
TableDefPtr make_box_type();
TableDefPtr make_capsule_type();
TableDefPtr make_branch_type();
TableDefPtr make_cylinder_type();
TableDefPtr make_rock_type(int p_k, uint32_t p_seed);
TableDefPtr make_cone_type(int p_k, uint32_t p_seed);
TableDefPtr make_basin_type(int p_k, uint32_t p_seed);
TableDefPtr make_island_type(int p_k, uint32_t p_seed);
TableDefPtr make_swiss_cube_type(int p_holes);
TableDefPtr make_die_type();
TableDefPtr make_stairs_type();
TableDefPtr make_tunnel_cube_type();
// A moment blob {1 / s_x^2, 1 / s_y^2, 1 / s_z^2, a}: the occupancy a max(1 - q, 0)^2, q = sum_i l_i^2 / s_i^2 in the
// record's frame (its rotation: the blob's principal axes), folded with Combine::ACCUM (MomentBlob, moment_layers.hpp).
TableDefPtr make_moment_blob_type();

// The influence box (world axes, int64, absolute) of a record: the type's local box, rotated by the
// record's rotation, rounded out, plus the record's anchor. Used by the table's spatial index.
Bounds record_world_bounds(const TableDef &p_def, const TableRecord &p_record);

// A record's moments (instance_moments.hpp) in world axes, relative to its anchor: its type's moments(params) rotated,
// or, for a type without one, the CPU program at LOD 0 (solid where !(w > 0), like the meshers) sampled at the centres
// of a FALLBACK_GRID^3 grid of its local box. Weight 1: the record's volume. (archetype_moments.cpp)
inline constexpr int FALLBACK_GRID = 24;
Moments record_moments(const TableDef &p_def, const TableRecord &p_record);
// The moments of p_count records (an instance made of parts: a tree's span) per material, relative to p_anchor: each
// record weighs its volume, its material resolved (record_material). The cutters and the removed records weigh
// nothing: their overlap with the rest is unknown. Overlapping parts count twice (a tree's hull spheres and clumps
// overlap).
MaterialMoments span_moments(const DefLibrary &p_library, const TableRecord *p_records, uint32_t p_count, const VoxelPos &p_anchor);

// The steepest field a runtime edit may add (the library's shapes are ~1-2 voxels per voxel).
inline constexpr float MAX_RECORD_LIPSCHITZ = 1024.0f;
// Why a runtime edit's record doesn't fit the table (empty if it does): an anchor beyond the world-origin
// bound (+-2^60 voxels, so int64 boxes around it can't overflow), a non-finite type box or a Lipschitz bound
// above MAX_RECORD_LIPSCHITZ, or a box (under any rotation) plus twice the smooth radius k wider than
// p_max_extent voxels. Checked before finish_record, whose int64 / int32 conversions of the box assume it,
// and whose index places one entry per top-LOD node the box touches.
ov::str record_fit_error(const TableDef &p_def, const TableRecord &p_record, int64_t p_max_extent);

// How a table finishes its records: its skirt depth (voxels), the band rule on or off and its erosion (SDFTable's
// --instance_band / --instance_erode / --instance_resolve).
struct RecordFinish {
    float skirt_depth = 0.0f;
    bool band = true;
    ErodeRule rule;
};
// The record's material resolved against its type's (record_material), then finish_record: the table's records and the
// sources' alike.
void finish_table_record(const DefLibrary &p_library, const RecordFinish &p_finish, TableRecord &r_record);

} // namespace Archetype
