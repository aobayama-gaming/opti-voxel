#pragma once

// The Dawn host's terrain: the core's Hermite cascade on a flat base, plus a small list of carve primitives
// (spheres and capsules) folded onto the distance. The same list is uploaded to the mesher shader, so the CPU
// (ray march, chunk culling) and the GPU (surface nets) see one field.

#include "base_surface.hpp"
#include "cascade.hpp"
#include "cascade_source.hpp"
#include "chunk_math.hpp"
#include "ov/bytes.hpp"
#include "world_coord.hpp"

#include <cstdint>
#include <vector>

namespace ov::dawn {

using ::Float3;
using ::VoxelPos;

struct Edit {
    enum class Kind : uint32_t {
        SPHERE = 0,
        CAPSULE = 1,
    };
    Kind kind = Kind::SPHERE;
    VoxelPos center; // LOD0 voxels
    float basis[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f }; // rows: local x, y, z
    float radius = 0.0f; // LOD0 voxels
    float half_length = 0.0f; // capsule: along local y, LOD0 voxels
    float smooth = 0.0f; // metres
    bool subtract = true;
};

class DawnSDF : public CascadeSource {
public:
    static constexpr int EDIT_BINDING = 11;

    void configure(uint32_t p_seed, int p_k_top, double p_steepness, float p_voxel_size, const BaseSurface &p_base);
    const Cascade::Spec *get_cascade() const override { return &spec; }
    ov::str get_glsl_code() const override;
    ov::str get_sample_fold_glsl(uint32_t) const override { return {}; }
    ov::str get_ground_material_glsl(const Cascade::Spec &) const override { return {}; }
    ov::str get_ground_glsl(const Cascade::Spec &) const override { return {}; }
    ov::str get_climate_helpers_glsl() const override { return {}; }
    ov::str get_table_glsl(int, int) const override { return {}; }
    bool samples_carry_materials() const override { return false; }
    bool has_ground_terms() const override { return false; }

    // The signed distance at a LOD-p_lod lattice voxel, metres (positive air, negative solid). The voxel is an
    // absolute LOD0 voxel, a multiple of 2^p_lod.
    float distance_m(const VoxelPos &p_voxel, int p_lod = 0) const;
    // The normal (unit) at a LOD0 voxel, by central differences.
    Float3 normal(const VoxelPos &p_voxel) const;

    // The edits buffer: 6 vec4 per edit after a header vec4 (its x = the edit count).
    ov::bytes edit_bytes() const;
    int edit_count() const { return int(edits.size()); }
    std::vector<Edit> &edit_list() { return edits; }
    void clear_edits() { edits.clear(); }
    // The chunks an edit can change: its world box grown by the smooth radius, in LOD0 voxels.
    void edit_box(const Edit &p_edit, VoxelPos &r_lo, VoxelPos &r_hi) const;

    float voxel_size() const { return voxel_m; }
    const BaseSurface &base() const { return base_surface; }
    const ChunkParam &chunk_param() const { return param; }
    ChunkParam &chunk_param() { return param; }

private:
    Cascade::Spec spec;
    BaseSurface base_surface;
    ChunkParam param;
    std::vector<Edit> edits;
    float voxel_m = 0.25f;
};

} // namespace ov::dawn
