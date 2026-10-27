#pragma once

#include "cascade_glsl.hpp"
#include "gpu_device.hpp"
#include "sdf_base.hpp"

#include "ov/types.hpp"

// The cascade passes of one mesher on one Device: pipelines, scratch and band buffers, bind sets. Not thread-safe; use
// it on the thread that owns the device (the render thread for Godot's main one).
class CascadePasses {
public:
    // p_input (ChunkInput[]), p_param_ubo (ChunkParam std140), p_sdf_sample (vec4 per sample): the mesher's.
    // p_max_chunks: chunks per dispatch (scratch size). p_debug: valid for the debug variant, a buffer for
    // the last level's channels (vec4s_per_point vec4s per sample, CascadeSpike); invalid for the meshers' (a
    // 16 B placeholder is bound). p_share_scratch: use that object's scratch buffers instead of new ones.
    // p_table_records / p_table_list: the mesher's instance table buffers (a terrain with a table): with them, a
    // second variant whose level 0 folds the chunks' instance lists (record_level's p_fold). p_sdf_material: the
    // mesher's per-sample material buffer (uint per sample, indexed like sdf_sample), written by level 0 when given.
    // p_batch_items: the items of a batch (upload_regions), when record_all runs it in several sub-batches; 0: p_max_chunks.
    using Buffer = ov::gpu::Buffer;
    bool create(ov::gpu::Device *p_device, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_chunks, Buffer p_input, Buffer p_param_ubo,
            Buffer p_sdf_sample, Buffer p_debug = Buffer(), const CascadePasses *p_share_scratch = nullptr, Buffer p_table_records = Buffer(),
            Buffer p_table_list = Buffer(), Buffer p_sdf_material = Buffer(), uint32_t p_batch_items = 0);
    // Point items (CascadePointsGPU): p_input holds GPUFrame::encode_point_item items (any LODs), p_param_ubo the std140
    // point_param, p_values one vec4 per item (SDFCascade::ground_unit's bits).
    bool create_points(ov::gpu::Device *p_device, const SDFBase *p_sdf, const ChunkParam &p_point_param, uint32_t p_max_points, Buffer p_input,
            Buffer p_param_ubo, Buffer p_values);
    void free();
    bool is_valid() const { return uniform_set[0].is_valid() || uniform_set[2].is_valid() || uniform_set[4].is_valid(); }

    // Number of levels a batch whose smallest LOD is p_min_lod runs (j = count - 1 ... 0).
    int level_count(int p_min_lod) const;
    // A sphere's regions for the next records (base_regions.hpp): the batch's p_count items as uploaded to the input
    // buffer, their ChunkInput bytes (point items for create_points' passes). Nothing on a flat base. Call it with every
    // batch, before its record_level / record_all.
    void upload_regions(const uint8_t *p_items, uint32_t p_count);

    // Records level j for chunks [p_base, p_base + p_count) of the input (p_count <= max chunks: the scratch
    // holds chunk p_base + i at slot i) as one compute list. p_fold: level 0 folds the chunks' instance lists (a
    // batch with any instance listed; needs the table's buffers). Off, the bare cascade: the same bits over empty
    // lists, without the fold's cost. p_lods: the batch's smallest and largest LOD (ChunkMath::lod_range): the
    // fields' variant where Cascade::FieldRange::batch says.
    void record_level(int p_j, uint32_t p_base, uint32_t p_count, const ov::ivec2 &p_lods, bool p_fold = false);
    // Every level, sub-batches of at most max chunks.
    void record_all(const ov::ivec2 &p_lods, uint32_t p_count, bool p_fold = false);

    uint64_t scratch_bytes() const { return scratch_bytes_total; }
    uint32_t get_max_chunks() const { return max_chunks; }
    bool is_points() const { return item_size == 1; }
    Buffer get_scratch_a() const { return scratch_a; }
    Buffer get_scratch_b() const { return scratch_b; }

private:
    ov::gpu::Device *device = nullptr;
    // [2 * view + (fold ? 1 : 0)]: the bare level-0 output or the instance fold (given the table's buffers), view the
    // CascadeGLSL::View (the fields and the samples' views where the FieldRange carries any).
    static constexpr int VARIANTS = 6;
    ov::gpu::Pipeline pipeline[VARIANTS];
    ov::gpu::BindSet uniform_set[VARIANTS];
    Cascade::FieldRange fields; // empty: no fields variant
    Buffer scratch_a;
    Buffer scratch_b;
    Buffer bands;
    Buffer debug_placeholder;
    // A sphere's regions (BaseRegions::Tables), sized for batch_items items; invalid on a flat base.
    const BaseSurface *sphere = nullptr;
    ChunkParam item_param;
    Buffer base_regions;
    Buffer base_region_index;
    uint32_t batch_items = 0;
    uint32_t max_regions = 0;
    int top = 0;
    bool carry = false; // some run's last level reads carry groups (Spec::gpu_carry_groups)
    int item_size = 32; // points per axis of an item: the chunk size, or 1 for point items
    uint32_t max_chunks = 0;
    uint64_t scratch_bytes_total = 0;
    bool owns_scratch = true;

    bool _create(ov::gpu::Device *p_device, const SDFBase *p_sdf, int p_item_size, uint32_t p_max_items, Buffer p_input, Buffer p_param_ubo,
            Buffer p_output, Buffer p_debug, const CascadePasses *p_share_scratch, Buffer p_table_records, Buffer p_table_list, Buffer p_material,
            uint32_t p_batch_items);
};
