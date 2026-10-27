#pragma once

#include "ov/format.hpp"
#include "ov/str.hpp"

#include <cstdint>

namespace Cascade {
struct Spec;
}
class BaseSurface;

// What the cascade's level shader (cascade_glsl.hpp) reads of a terrain: its bands and the GLSL pieces it splices.
// SDFBase implements it; the shader text needs nothing else of the terrain, so it is built without the engine.
class CascadeSource {
public:
    virtual ~CascadeSource() = default;

    // The Hermite cascade's bands (never null): the meshers compute the samples with the cascade passes
    // (cascade_glsl.hpp), and evaluate() only takes lattice points (frac 0, voxel a multiple of 2^lod).
    virtual const Cascade::Spec *get_cascade() const = 0;

    // GPU code spliced into the cascade passes (SDFCascade: the cascade's register, metres and skirt
    // functions). The shader provides the chunk_param
    // uniform, the frame functions of gpu_frame.hpp and the exact noise of exact_noise.hpp.
    virtual ov::str get_glsl_code() const = 0;
    // The GLSL the cascade level passes' fold variant splices after the table buffers (cascade_glsl.hpp), defining
    // `vec4 cascade_sample_sdf(vec4 c0, WorldPos p, int lod, WorldPos group_lo, WorldPos group_hi, uint chunk, out uint
    // material, out float skirt_extra)` (material: the winning record's palette index, 0 = ground; skirt_extra: metres
    // added to the sample's skirt depth), called by every invocation of a p_group_size workgroup together (it has
    // barriers), [group_lo, group_hi] holding the workgroup's points; p_group_size 0: per sample (no barrier, the group
    // ignored).
    virtual ov::str get_sample_fold_glsl(uint32_t p_group_size) const = 0;
    // With a per-sample ground material (base_palette longer than one): the GLSL the cascade level passes' fields variant
    // splices after its channel code, defining `uint cascade_ground_material(Ch ch, WorldPos p)`, the palette index of a
    // sample no record colours (ch: its channels on Λ_L, the fields carried; p: its absolute point). Empty otherwise.
    virtual ov::str get_ground_material_glsl(const Cascade::Spec &p_view) const {
        (void)p_view;
        return ov::str();
    }
    // Whether a sample's ground is more than the cascade's ground channels (a climate's height terms or terraces): then
    // the cascade level passes carry the fields at every level, point items too, and splice get_ground_glsl after their
    // channel code, defining `vec4 climate_ground(Ch ch)`, the ground channels (value, gradient) of a sample.
    virtual bool has_ground_terms() const { return false; }
    virtual ov::str get_ground_glsl(const Cascade::Spec &p_view) const {
        (void)p_view;
        return ov::str();
    }
    // The dual-number helpers the climate's generated functions share, spliced once before them all (Climate's
    // splices_helpers: a shaped modulation or ground terms); empty when each brings its own.
    virtual ov::str get_climate_helpers_glsl() const { return ov::str(); }

    // The ground's base surface when it is a sphere: get_glsl_code's base_at then reads the batch's regions
    // (base_regions.hpp), which the cascade passes upload with each batch. Null for a flat base.
    virtual const BaseSurface *get_sphere_base() const { return nullptr; }

    // The instance table's DefRecord struct and buffer declarations at the given bindings (the fold variant; a terrain
    // with a table).
    virtual ov::str get_table_glsl(int p_record_binding, int p_list_binding) const = 0;
    // Whether every sample carries a material (the table's base palette is longer than {GROUND}: the climate's biomes).
    virtual bool samples_carry_materials() const = 0;
};
