#pragma once

#include "cascade_glsl.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/string.hpp>

// The cascade passes of one mesher on one RenderingDevice: shader, pipeline, scratch and band buffers,
// uniform set. Not thread-safe; use it on the thread that owns the device (render thread for the main one).
class CascadePasses {
public:
    // p_input (ChunkInput[]), p_param_ubo (ChunkParam std140), p_sdf_sample (vec4 per sample): the mesher's.
    // p_max_chunks: chunks per dispatch (scratch size). p_debug: valid for the debug variant, a buffer for
    // the last level's channels (vec4s_per_point vec4s per sample, CascadeSpike); invalid for the meshers' (a
    // 16 B placeholder is bound). p_share_scratch: use that object's scratch buffers instead of new ones.
    // p_table_records / p_table_list: the mesher's instance table buffers (a terrain with a table): with them, a
    // second variant whose level 0 folds the chunks' instance lists (record_level's p_fold). p_sdf_material: the
    // mesher's per-sample material buffer (uint per sample, indexed like sdf_sample), written by level 0 when given.
    bool create(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_chunks, RID p_input, RID p_param_ubo,
            RID p_sdf_sample, RID p_debug = RID(), const CascadePasses *p_share_scratch = nullptr, RID p_table_records = RID(),
            RID p_table_list = RID(), RID p_sdf_material = RID());
    // Point items (CascadePointsGPU): p_input holds GPUFrame::encode_point_item items (any LODs), p_param_ubo the std140
    // point_param, p_values one vec4 per item (SDFCascade::ground_unit's bits).
    bool create_points(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_point_param, uint32_t p_max_points, RID p_input, RID p_param_ubo,
            RID p_values);
    void free();
    bool is_valid() const { return uniform_set[0].is_valid() || uniform_set[2].is_valid() || uniform_set[4].is_valid(); }

    // Number of levels a batch whose smallest LOD is p_min_lod runs (j = count - 1 ... 0).
    int level_count(int p_min_lod) const;
    // Records level j for chunks [p_base, p_base + p_count) of the input (p_count <= max chunks: the scratch
    // holds chunk p_base + i at slot i) as one compute list. p_fold: level 0 folds the chunks' instance lists (a
    // batch with any instance listed; needs the table's buffers). Off, the bare cascade: the same bits over empty
    // lists, without the fold's cost. p_lods: the batch's smallest and largest LOD (ChunkMath::lod_range): the
    // fields' variant where Cascade::FieldRange::batch says.
    void record_level(int p_j, uint32_t p_base, uint32_t p_count, const Vector2i &p_lods, bool p_fold = false);
    // Every level, sub-batches of at most max chunks.
    void record_all(const Vector2i &p_lods, uint32_t p_count, bool p_fold = false);

    uint64_t scratch_bytes() const { return scratch_bytes_total; }
    uint32_t get_max_chunks() const { return max_chunks; }
    bool is_points() const { return item_size == 1; }
    RID get_scratch_a() const { return scratch_a; }
    RID get_scratch_b() const { return scratch_b; }

private:
    RenderingDevice *rd = nullptr;
    // [2 * view + (fold ? 1 : 0)]: the bare level-0 output or the instance fold (given the table's buffers), view the
    // CascadeGLSL::View (the fields and the samples' views where the FieldRange carries any).
    static constexpr int VARIANTS = 6;
    RID shader[VARIANTS];
    RID pipeline[VARIANTS];
    RID uniform_set[VARIANTS];
    Cascade::FieldRange fields; // empty: no fields variant
    RID scratch_a;
    RID scratch_b;
    RID bands;
    RID debug_placeholder;
    int top = 0;
    bool carry = false; // some run's last level reads carry groups (Spec::gpu_carry_groups)
    int item_size = 32; // points per axis of an item: the chunk size, or 1 for point items
    uint32_t max_chunks = 0;
    uint64_t scratch_bytes_total = 0;
    bool owns_scratch = true;

    bool _compile(int p_variant, const String &p_name, const String &p_code);
    bool _create(RenderingDevice *p_rd, const SDFBase *p_sdf, int p_item_size, uint32_t p_max_items, RID p_input, RID p_param_ubo, RID p_output,
            RID p_debug, const CascadePasses *p_share_scratch, RID p_table_records, RID p_table_list, RID p_material);
};
