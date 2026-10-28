#include "cascade_passes.hpp"

#include "gpu_frame.hpp"
#include "gpu_shader.hpp"

#include "ov/bytes.hpp"
#include "ov/math.hpp"
#include "ov/on_error.hpp"

using namespace Cascade;
using namespace CascadeGLSL;
using ov::gpu::Binding;
using ov::gpu::BufferKind;

static_assert(BaseRegions::REGION_BINDING > CascadeGLSL::MATERIAL_BINDING && BaseRegions::REGION_BINDING > CascadeGLSL::TABLE_LIST_BINDING,
        "CascadePasses: the base regions' bindings follow the level shader's own.");
static_assert(CascadeGLSL::CLIMATE_BINDING > BaseRegions::INDEX_BINDING, "CascadePasses: the climate items' binding follows the base regions'.");

bool CascadePasses::create(ov::gpu::Device *p_device, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_chunks, Buffer p_input,
        Buffer p_param_ubo, Buffer p_sdf_sample, Buffer p_debug, const CascadePasses *p_share_scratch, Buffer p_table_records, Buffer p_table_list,
        Buffer p_sdf_material, uint32_t p_batch_items) {
    item_param = p_param;
    return _create(p_device, p_sdf, p_param.chunk_size, p_max_chunks, p_input, p_param_ubo, p_sdf_sample, p_debug, p_share_scratch, p_table_records,
            p_table_list, p_sdf_material, p_batch_items);
}

bool CascadePasses::create_points(ov::gpu::Device *p_device, const SDFBase *p_sdf, const ChunkParam &p_point_param, uint32_t p_max_points,
        Buffer p_input, Buffer p_param_ubo, Buffer p_values) {
    OV_ERR_FAIL_COND_V_MSG(p_point_param.skirt_size != 0, false, "CascadePasses: point items need CascadeGLSL::point_param (skirt 0).");
    item_param = p_point_param;
    return _create(p_device, p_sdf, 1, p_max_points, p_input, p_param_ubo, p_values, Buffer(), nullptr, Buffer(), Buffer(), Buffer(), 0);
}

bool CascadePasses::_create(ov::gpu::Device *p_device, const SDFBase *p_sdf, int p_item_size, uint32_t p_max_items, Buffer p_input,
        Buffer p_param_ubo, Buffer p_output, Buffer p_debug, const CascadePasses *p_share_scratch, Buffer p_table_records, Buffer p_table_list,
        Buffer p_material, uint32_t p_batch_items) {
    const Spec *spec = p_sdf->get_cascade();
    device = p_device;
    input = p_input;
    item_size = p_item_size;
    max_chunks = p_max_items;
    const bool points = item_size == 1;
    fields = CascadeGLSL::field_range(p_sdf, points);
    top = spec->top_level(!fields.empty());
    carry = !fields.empty() && spec->gpu_carry_groups() > 0;

    // Variant 2 * view + (fold ? 1 : 0): the bare cascade at level 0 or the instance fold (given the table's buffers), with
    // the ground alone, the fields and the carry, or the carry alone (CascadeGLSL::View; the climate's views where the
    // FieldRange carries any: none if empty).
    const bool table = p_table_records.is_valid() && p_table_list.is_valid();
    OV_ERR_FAIL_COND_V_MSG((table || p_material.is_valid()) && points, false, "CascadePasses: point items don't fold or carry a material.");
    // The fold variant's workgroups are cubes of level-0 samples, given whole cubes; else it folds per sample.
    const bool cubes = item_size % int(CascadeGLSL::FOLD_CUBE) == 0;
    static const char *VIEW_NAMES[] = { "", "_fields", "_samples" };
    // A climate everywhere: the ground-only variants would never run. The fields' view is never a last level: no fold.
    for (int c = fields.everywhere() ? 1 : 0; c < (fields.empty() ? 1 : 3); c++) {
        for (int f = 0; f < (table && CascadeGLSL::View(c) != CascadeGLSL::View::FIELDS ? 2 : 1); f++) {
            const int v = 2 * c + f;
            const ov::str code =
                    CascadeGLSL::level_shader(p_sdf, p_debug.is_valid(), points, f == 1, p_material.is_valid(), CascadeGLSL::View(c), cubes);
            const ov::str name =
                    ov::str(points ? "cascade_points" : (p_debug.is_valid() ? "cascade_levels_debug" : "cascade_levels")) + (f == 1 ? "_fold" : "") + VIEW_NAMES[c];
            pipeline[v] = ov::gpu::compile(*device, name, code);
            if (!pipeline[v].is_valid() && f == 0) {
                return false; // a fold variant that fails leaves the bare one working (record_level ERRs once)
            }
        }
    }

    uint64_t bytes_a = 0, bytes_b = 0;
    CascadeGLSL::scratch_bytes(p_sdf, item_size, max_chunks, bytes_a, bytes_b);
    if (p_share_scratch != nullptr) {
        OV_ERR_FAIL_COND_V(p_share_scratch->max_chunks < max_chunks || p_share_scratch->scratch_bytes_total < bytes_a + bytes_b, false);
        scratch_a = p_share_scratch->scratch_a;
        scratch_b = p_share_scratch->scratch_b;
        owns_scratch = false;
    } else {
        scratch_a = device->create_buffer(BufferKind::STORAGE, bytes_a);
        scratch_b = device->create_buffer(BufferKind::STORAGE, bytes_b);
        scratch_bytes_total = bytes_a + bytes_b;
    }
    const ov::bytes band_bytes = CascadeGLSL::band_table(*spec);
    bands = device->create_buffer(BufferKind::STORAGE, band_bytes.size(), band_bytes.data());
    Buffer debug = p_debug;
    if (!debug.is_valid()) {
        debug_placeholder = device->create_buffer(BufferKind::STORAGE, 16);
        debug = debug_placeholder;
    }

    std::vector<Binding> bindings = {
        { 0, BufferKind::STORAGE, p_input },
        { 1, BufferKind::UNIFORM, p_param_ubo },
        { 2, BufferKind::STORAGE, p_output },
        { 3, BufferKind::STORAGE, scratch_a },
        { 4, BufferKind::STORAGE, scratch_b },
        { 5, BufferKind::STORAGE, bands },
        { 6, BufferKind::STORAGE, debug },
    };
    if (p_material.is_valid()) {
        bindings.push_back({ uint32_t(CascadeGLSL::MATERIAL_BINDING), BufferKind::STORAGE, p_material });
    }
    batch_items = ov::max(p_batch_items, max_chunks);
    if (spec->modulation && spec->modulation->region_path_count() > 0) {
        climate_items = device->create_buffer(BufferKind::STORAGE, uint64_t(batch_items) * CascadeGLSL::CLIMATE_ITEM_BYTES);
        bindings.push_back({ uint32_t(CascadeGLSL::CLIMATE_BINDING), BufferKind::STORAGE, climate_items });
    }
    sphere = p_sdf->get_sphere_base();
    if (sphere != nullptr) {
        base_regions = device->create_buffer(BufferKind::STORAGE, uint64_t(max_regions()) * BaseRegions::REGION_BYTES);
        base_region_index = device->create_buffer(BufferKind::STORAGE, uint64_t(batch_items) * BaseRegions::SLOTS * sizeof(uint32_t));
        bindings.push_back({ uint32_t(BaseRegions::REGION_BINDING), BufferKind::STORAGE, base_regions });
        bindings.push_back({ uint32_t(BaseRegions::INDEX_BINDING), BufferKind::STORAGE, base_region_index });
    }
    // The fold variants also bind the table's buffers.
    std::vector<Binding> fold_bindings = bindings;
    if (table) {
        fold_bindings.push_back({ uint32_t(CascadeGLSL::TABLE_RECORD_BINDING), BufferKind::STORAGE, p_table_records });
        fold_bindings.push_back({ uint32_t(CascadeGLSL::TABLE_LIST_BINDING), BufferKind::STORAGE, p_table_list });
    }
    for (int v = 0; v < VARIANTS; v++) {
        if (pipeline[v].is_valid()) {
            uniform_set[v] = device->create_bind_set(pipeline[v], (v & 1) ? fold_bindings : bindings);
        }
    }
    return (uniform_set[0].is_valid() || fields.everywhere()) && (fields.empty() || (uniform_set[2].is_valid() && uniform_set[4].is_valid()));
}

void CascadePasses::free() {
    if (device == nullptr) {
        return;
    }
    for (int v = 0; v < VARIANTS; v++) {
        for (const ov::gpu::Handle h : { ov::gpu::Handle(uniform_set[v]), ov::gpu::Handle(pipeline[v]) }) {
            if (h.is_valid()) {
                device->free(h);
            }
        }
    }
    for (const Buffer b : { bands, debug_placeholder, base_regions, base_region_index, climate_items }) {
        if (b.is_valid()) {
            device->free(b);
        }
    }
    if (owns_scratch) {
        for (const Buffer b : { scratch_a, scratch_b }) {
            if (b.is_valid()) {
                device->free(b);
            }
        }
    }
    for (int v = 0; v < VARIANTS; v++) {
        uniform_set[v] = ov::gpu::BindSet();
        pipeline[v] = ov::gpu::Pipeline();
    }
    scratch_a = Buffer();
    scratch_b = Buffer();
    bands = Buffer();
    debug_placeholder = Buffer();
    base_regions = Buffer();
    base_region_index = Buffer();
    climate_items = Buffer();
    input = Buffer();
    sphere = nullptr;
    device = nullptr;
}

void CascadePasses::upload_items(const uint8_t *p_items, uint32_t p_count, uint32_t p_slots, const uint32_t *p_modes) {
    const uint32_t slots = ov::max(p_slots, p_count);
    if (slots == 0) {
        return;
    }
    OV_ERR_FAIL_COND_MSG((sphere != nullptr || climate_items.is_valid()) && p_count > batch_items, "CascadePasses: more items than the passes were created for.");
    device->update_buffer(input, 0, uint64_t(slots) * GPUFrame::CHUNK_INPUT_BYTES, p_items);
    if (climate_items.is_valid() && p_count > 0) {
        ov::bytes words(size_t(p_count) * CascadeGLSL::CLIMATE_ITEM_BYTES, 0);
        for (uint32_t i = 0; i < p_count; i++) {
            ov::put_u32(words.data() + i * CascadeGLSL::CLIMATE_ITEM_BYTES, p_modes ? p_modes[i] : Cascade::CLIMATE_UNKNOWN);
            ov::put_u32(words.data() + i * CascadeGLSL::CLIMATE_ITEM_BYTES + 4, ~0u); // the report, ANDed
        }
        device->update_buffer(climate_items, 0, words.size(), words.data());
    }
    if (sphere == nullptr || p_count == 0) {
        return;
    }
    BaseRegions::build(*sphere, p_items, p_count, item_param, is_points(), regions);
    OV_ERR_FAIL_COND_MSG(regions.region_count() > max_regions(), "CascadePasses: more base regions than the buffer holds.");
    device->update_buffer(base_regions, 0, regions.regions.size(), regions.regions.data());
    device->update_buffer(base_region_index, 0, regions.index.size() * sizeof(uint32_t), regions.index.data());
}

int CascadePasses::level_count(int p_min_lod) const {
    // With the carry, a level above the batch's LOD even above the top band (its chunks derive there).
    return ov::max(top - p_min_lod, carry ? 1 : 0) + 1;
}

void CascadePasses::record_level(int p_j, uint32_t p_base, uint32_t p_count, const ov::ivec2 &p_lods, bool p_fold) {
    OV_ERR_FAIL_COND(p_count == 0 || p_count > max_chunks);
    const uint32_t n = uint32_t(level_bound(item_size, p_j));
    uint8_t push[CascadeGLSL::PUSH_BYTES] = {};
    ov::put_u32(push + 0, p_base);
    ov::put_u32(push + 4, p_count);
    ov::put_s32(push + 8, p_j);
    ov::put_s32(push + 12, top);
    ov::put_u32(push + 16, n);
    ov::put_u32(push + 20, uint32_t(level_bound(item_size, p_j + 1)));
    // The fields where a chunk of the batch derives, else the carry where one carries it (monotone in j: a level with
    // either always finds the fields in its parent level).
    CascadeGLSL::View view = CascadeGLSL::View::GROUND;
    if (!fields.empty() && fields.batch(p_lods, p_j)) {
        view = CascadeGLSL::View::FIELDS;
    } else if (!fields.empty() && fields.batch_carry(p_lods, p_j)) {
        view = CascadeGLSL::View::SAMPLES;
    }
    bool fold = p_j == 0 && p_fold;
    if (fold && !uniform_set[2 * int(view) + 1].is_valid()) {
        OV_ERR_PRINT_ONCE("CascadePasses: a fold without its variant (no table buffers at create, or it failed to compile): the bare cascade instead.");
        fold = false;
    }
    const int v = 2 * int(view) + (fold ? 1 : 0);
    if (is_points()) {
        // Items flattened over x: item c's points are threads c * n^3 ...
        device->dispatch(pipeline[v], uniform_set[v], push, sizeof(push), (p_count * n * n * n + CascadeGLSL::LOCAL_SIZE - 1) / CascadeGLSL::LOCAL_SIZE, 1, 1);
    } else {
        device->dispatch(pipeline[v], uniform_set[v], push, sizeof(push), (n * n * n + CascadeGLSL::LOCAL_SIZE - 1) / CascadeGLSL::LOCAL_SIZE, p_count, 1);
    }
}

void CascadePasses::record_all(const ov::ivec2 &p_lods, uint32_t p_count, bool p_fold) {
    for (uint32_t base = 0; base < p_count; base += max_chunks) {
        const uint32_t count = ov::min(max_chunks, p_count - base);
        for (int j = level_count(p_lods.x) - 1; j >= 0; j--) {
            record_level(j, base, count, p_lods, p_fold);
        }
    }
}
