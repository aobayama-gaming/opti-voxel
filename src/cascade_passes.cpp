#include "cascade_passes.hpp"

#include "godot_conv.hpp"
#include "rd_utils.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace Cascade;
using namespace CascadeGLSL;

bool CascadePasses::create(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_chunks, RID p_input, RID p_param_ubo,
        RID p_sdf_sample, RID p_debug, const CascadePasses *p_share_scratch, RID p_table_records, RID p_table_list, RID p_sdf_material) {
    return _create(p_rd, p_sdf, p_param.chunk_size, p_max_chunks, p_input, p_param_ubo, p_sdf_sample, p_debug, p_share_scratch, p_table_records,
            p_table_list, p_sdf_material);
}

bool CascadePasses::create_points(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_point_param, uint32_t p_max_points, RID p_input,
        RID p_param_ubo, RID p_values) {
    ERR_FAIL_COND_V_MSG(p_point_param.skirt_size != 0, false, "CascadePasses: point items need CascadeGLSL::point_param (skirt 0).");
    return _create(p_rd, p_sdf, 1, p_max_points, p_input, p_param_ubo, p_values, RID(), nullptr, RID(), RID(), RID());
}

bool CascadePasses::_create(RenderingDevice *p_rd, const SDFBase *p_sdf, int p_item_size, uint32_t p_max_items, RID p_input, RID p_param_ubo,
        RID p_output, RID p_debug, const CascadePasses *p_share_scratch, RID p_table_records, RID p_table_list, RID p_material) {
    const Spec *spec = p_sdf->get_cascade();
    rd = p_rd;
    item_size = p_item_size;
    max_chunks = p_max_items;
    const bool points = item_size == 1;
    fields = CascadeGLSL::field_range(p_sdf, points);
    top = spec->top_level(!fields.empty());

    // Variant (fields ? 2 : 0) + (fold ? 1 : 0): the bare cascade at level 0 or the instance fold (given the table's
    // buffers), with the ground alone or with the fields (where the spec's FieldRange carries them: none if empty).
    const bool table = p_table_records.is_valid() && p_table_list.is_valid();
    ERR_FAIL_COND_V_MSG((table || p_material.is_valid()) && points, false, "CascadePasses: point items don't fold or carry a material.");
    // The fold variant's workgroups are cubes of level-0 samples, given whole cubes; else it folds per sample.
    const bool cubes = item_size % int(CascadeGLSL::FOLD_CUBE) == 0;
    // Fields carried everywhere: the ground-only variants would never run.
    for (int c = fields.everywhere() ? 1 : 0; c < (fields.empty() ? 1 : 2); c++) {
        for (int f = 0; f < (table ? 2 : 1); f++) {
            const int v = 2 * c + f;
            const String code = godot_str(CascadeGLSL::level_shader(p_sdf, p_debug.is_valid(), points, f == 1, p_material.is_valid(), c == 1, cubes));
            const String name = String(points ? "cascade_points" : (p_debug.is_valid() ? "cascade_levels_debug" : "cascade_levels")) + (f == 1 ? "_fold" : "") +
                    (c == 1 ? "_fields" : "");
            if (!_compile(v, name, code) && f == 0) {
                return false; // a fold variant that fails leaves the bare one working (record_level ERRs once)
            }
        }
    }

    uint64_t bytes_a = 0, bytes_b = 0;
    CascadeGLSL::scratch_bytes(p_sdf, item_size, max_chunks, bytes_a, bytes_b);
    if (p_share_scratch != nullptr) {
        ERR_FAIL_COND_V(p_share_scratch->max_chunks < max_chunks || p_share_scratch->scratch_bytes_total < bytes_a + bytes_b, false);
        scratch_a = p_share_scratch->scratch_a;
        scratch_b = p_share_scratch->scratch_b;
        owns_scratch = false;
    } else {
        scratch_a = rd->storage_buffer_create(bytes_a);
        scratch_b = rd->storage_buffer_create(bytes_b);
        scratch_bytes_total = bytes_a + bytes_b;
    }
    const PackedByteArray band_bytes = godot_bytes(CascadeGLSL::band_table(*spec));
    bands = rd->storage_buffer_create(band_bytes.size(), band_bytes);
    RID debug = p_debug;
    if (!debug.is_valid()) {
        debug_placeholder = rd->storage_buffer_create(16);
        debug = debug_placeholder;
    }

    auto make_uniform = [](int p_binding, RID p_rid, RenderingDevice::UniformType p_type) {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(p_type);
        u->set_binding(p_binding);
        u->add_id(p_rid);
        return u;
    };
    TypedArray<RDUniform> uniforms;
    uniforms.push_back(make_uniform(0, p_input, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(1, p_param_ubo, RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER));
    uniforms.push_back(make_uniform(2, p_output, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(3, scratch_a, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(4, scratch_b, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(5, bands, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(6, debug, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    if (p_material.is_valid()) {
        uniforms.push_back(make_uniform(CascadeGLSL::MATERIAL_BINDING, p_material, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    }
    // The fold variants also bind the table's buffers.
    TypedArray<RDUniform> fold_uniforms = uniforms.duplicate();
    if (table) {
        fold_uniforms.push_back(make_uniform(CascadeGLSL::TABLE_RECORD_BINDING, p_table_records, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
        fold_uniforms.push_back(make_uniform(CascadeGLSL::TABLE_LIST_BINDING, p_table_list, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    }
    for (int v = 0; v < VARIANTS; v++) {
        if (shader[v].is_valid()) {
            uniform_set[v] = rd->uniform_set_create((v & 1) ? fold_uniforms : uniforms, shader[v], 0);
        }
    }
    return (uniform_set[0].is_valid() || fields.everywhere()) && (fields.empty() || uniform_set[2].is_valid());
}

bool CascadePasses::_compile(int p_variant, const String &p_name, const String &p_code) {
    RDUtils::dump_shader(p_name, p_code);
    Ref<RDShaderSource> source;
    source.instantiate();
    source->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, p_code);
    Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(source);
    const String error = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!error.is_empty()) {
        UtilityFunctions::printerr("--- SDF SHADER COMPILATION FAILED (CascadePasses " + p_name + ") ---");
        UtilityFunctions::printerr("ERROR MESSAGE:\n", error);
        const PackedStringArray lines = p_code.split("\n");
        for (int i = 0; i < lines.size(); i++) {
            UtilityFunctions::printerr(String::num_int64(i + 1) + " | " + lines[i]);
        }
        return false;
    }
    shader[p_variant] = rd->shader_create_from_spirv(spirv);
    pipeline[p_variant] = rd->compute_pipeline_create(shader[p_variant]);
    return true;
}

void CascadePasses::free() {
    if (rd == nullptr) {
        return;
    }
    for (int v = 0; v < VARIANTS; v++) {
        for (const RID &rid : { uniform_set[v], pipeline[v], shader[v] }) {
            if (rid.is_valid()) {
                rd->free_rid(rid);
            }
        }
    }
    for (const RID &rid : { bands, debug_placeholder }) {
        if (rid.is_valid()) {
            rd->free_rid(rid);
        }
    }
    if (owns_scratch) {
        for (const RID &rid : { scratch_a, scratch_b }) {
            if (rid.is_valid()) {
                rd->free_rid(rid);
            }
        }
    }
    for (int v = 0; v < VARIANTS; v++) {
        uniform_set[v] = RID();
        pipeline[v] = RID();
        shader[v] = RID();
    }
    scratch_a = RID();
    scratch_b = RID();
    bands = RID();
    debug_placeholder = RID();
    rd = nullptr;
}

int CascadePasses::level_count(int p_min_lod) const {
    return MAX(top - p_min_lod, 0) + 1;
}

void CascadePasses::record_level(int p_j, uint32_t p_base, uint32_t p_count, const Vector2i &p_lods, bool p_fold) {
    ERR_FAIL_COND(p_count == 0 || p_count > max_chunks);
    const uint32_t n = uint32_t(level_bound(item_size, p_j));
    PackedByteArray push;
    push.resize(CascadeGLSL::PUSH_BYTES);
    push.fill(0);
    push.encode_u32(0, p_base);
    push.encode_u32(4, p_count);
    push.encode_s32(8, p_j);
    push.encode_s32(12, top);
    push.encode_u32(16, n);
    push.encode_u32(20, uint32_t(level_bound(item_size, p_j + 1)));
    // The fields where a chunk of the batch carries them (monotone in j: a level with the fields always finds them in
    // its parent level).
    const int with_fields = !fields.empty() && fields.batch(ov::ivec2(p_lods.x, p_lods.y), p_j) ? 2 : 0;
    bool fold = p_j == 0 && p_fold;
    if (fold && !uniform_set[with_fields + 1].is_valid()) {
        ERR_PRINT_ONCE("CascadePasses: a fold without its variant (no table buffers at create, or it failed to compile): the bare cascade instead.");
        fold = false;
    }
    const int v = with_fields + (fold ? 1 : 0);
    const int64_t list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(list, pipeline[v]);
    rd->compute_list_bind_uniform_set(list, uniform_set[v], 0);
    rd->compute_list_set_push_constant(list, push, push.size());
    if (is_points()) {
        // Items flattened over x: item c's points are threads c * n^3 ...
        rd->compute_list_dispatch(list, (p_count * n * n * n + CascadeGLSL::LOCAL_SIZE - 1) / CascadeGLSL::LOCAL_SIZE, 1, 1);
    } else {
        rd->compute_list_dispatch(list, (n * n * n + CascadeGLSL::LOCAL_SIZE - 1) / CascadeGLSL::LOCAL_SIZE, p_count, 1);
    }
    rd->compute_list_end();
}

void CascadePasses::record_all(const Vector2i &p_lods, uint32_t p_count, bool p_fold) {
    for (uint32_t base = 0; base < p_count; base += max_chunks) {
        const uint32_t count = MIN(max_chunks, p_count - base);
        for (int j = level_count(p_lods.x) - 1; j >= 0; j--) {
            record_level(j, base, count, p_lods, p_fold);
        }
    }
}
