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

    DefRecord def_record() const;
};

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
Float4 fold(const Program &p_ground, const DefLibrary &p_library, const TableRecord *p_records, const uint32_t *p_list,
        uint32_t p_count, const WorldPos &p_pos, int p_lod, bool p_box_test = true);
// fold() then the epilogue (archetype_ir.cpp finish()).
Float4 finish_table(const Float4 &p_out, int p_lod, float p_out_scale, bool p_normalize);

// Library types. Parameters (voxels): sphere {radius}; box {hx, hy, hz}; capsule {height, radius};
// cylinder {half_height, radius}; rock {radius, amp} (an fbm-displaced sphere); swiss_cube
// {hx, hy, hz, ring_radius, hole_radius} (a box with p_holes spherical holes, a composite through the
// table).
TableDefPtr make_sphere_type();
TableDefPtr make_box_type();
TableDefPtr make_capsule_type();
TableDefPtr make_cylinder_type();
TableDefPtr make_rock_type(int p_k, uint32_t p_seed);
TableDefPtr make_swiss_cube_type(int p_holes);

// The influence box (world axes, int64, absolute) of a record: the type's local box, rotated by the
// record's rotation, rounded out, plus the record's anchor. Used by the table's spatial index.
Bounds record_world_bounds(const TableDef &p_def, const TableRecord &p_record);

} // namespace Archetype
