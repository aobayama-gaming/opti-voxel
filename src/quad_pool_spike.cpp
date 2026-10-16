#include "quad_pool_spike.hpp"

#include "debug_log.hpp"
#include "page_format.hpp"
#include "rd_utils.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/world3d.hpp>

#include <cmath>

static constexpr const char SPIKE_COMPUTE[] = R"glsl(#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0, rgba16ui) uniform restrict writeonly uimage2D staging;
layout(set = 0, binding = 1, std430) restrict buffer Counter { uint counter[]; };
layout(push_constant, std430) uniform Params {
    vec4 origin_size; // xyz origin, w sheet size
    uvec4 dims;       // x = side
    ivec4 chunk;      // xyz: the chunk the corners are stored relative to
    vec4 grid;        // xyz: world position of its vertex (0, 0, 0), w: its cell size
} params;

#define QUAD_GROUP 64u

/*GLSL_PAGE_FORMAT*/

float height(vec2 xz) { return params.origin_size.y + 3.0 * sin(xz.x * 0.3) * cos(xz.y * 0.3); }
vec3 surf(vec2 xz) { return vec3(xz.x, height(xz), xz.y); }
vec3 surf_normal(vec2 xz) {
    float dx = 0.9 * cos(xz.x * 0.3) * cos(xz.y * 0.3);
    float dz = -0.9 * sin(xz.x * 0.3) * sin(xz.y * 0.3);
    return normalize(vec3(-dx, 1.0, -dz));
}

void put_corner(uint slot, uint k, vec2 xz) {
    vec3 local = (surf(xz) - params.grid.xyz) / params.grid.w;
    store_corner(slot, k, encode_corner(local, pack_normal_oct22(surf_normal(xz)), page_pos_scale(32)));
}

void main() {
    uint side = params.dims.x;
    uint i = gl_GlobalInvocationID.x;
    if (i == 0u) {
        counter[1] = (side * side + 63u) / 64u; // used groups of 64 quads
    }
    if (i >= side * side) {
        return;
    }
    uint qx = i % side;
    uint qz = i / side;
    float step = params.origin_size.w / float(side);
    vec2 base = params.origin_size.xz + vec2(qx, qz) * step;
    vec2 c1 = base + vec2(step, 0.0);
    vec2 c2 = base + vec2(0.0, step);
    if (qx >= side / 2u) { // swapped winding on the right half
        vec2 t = c1; c1 = c2; c2 = t;
    }
    uint slot = atomicAdd(counter[0], 1u);
    if (slot % QUAD_GROUP == 0u) {
        store_group_header(slot / QUAD_GROUP, params.chunk.xyz);
    }
    put_corner(slot, 0u, base);
    put_corner(slot, 1u, c1);
    put_corner(slot, 2u, c2);
    put_corner(slot, 3u, base + vec2(step, step));
}
)glsl";

void QuadPoolSpike::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_side", "side"), &QuadPoolSpike::set_side);
    ClassDB::bind_method(D_METHOD("get_side"), &QuadPoolSpike::get_side);
    ClassDB::bind_method(D_METHOD("set_extra_pages", "pages"), &QuadPoolSpike::set_extra_pages);
    ClassDB::bind_method(D_METHOD("get_extra_pages"), &QuadPoolSpike::get_extra_pages);
    ClassDB::bind_method(D_METHOD("set_page_capacity", "capacity"), &QuadPoolSpike::set_page_capacity);
    ClassDB::bind_method(D_METHOD("get_page_capacity"), &QuadPoolSpike::get_page_capacity);
    ClassDB::bind_method(D_METHOD("set_debug_view", "view"), &QuadPoolSpike::set_debug_view);
    ClassDB::bind_method(D_METHOD("get_report"), &QuadPoolSpike::get_report);
    ClassDB::bind_method(D_METHOD("get_stage"), &QuadPoolSpike::get_stage);
    ClassDB::bind_method(D_METHOD("run_kill_test"), &QuadPoolSpike::run_kill_test);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "side"), "set_side", "get_side");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "extra_pages"), "set_extra_pages", "get_extra_pages");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "page_capacity"), "set_page_capacity", "get_page_capacity");
}

void QuadPoolSpike::_ready() {
    mutex.instantiate();
    ERR_FAIL_COND_MSG(side * side > page_capacity, "QuadPoolSpike: sheet does not fit in one page.");
    ERR_FAIL_COND_MSG(KILL_B_OFFSET + KILL_B_COUNT > (uint32_t)(side * side), "QuadPoolSpike: sheet too small for the kill test.");

    // Corners are stored relative to one chunk: the smallest one whose vertex grid holds the sheet.
    const AABB sheet(sheet_origin - Vector3(0, 5, 0), Vector3(sheet_size, 10, sheet_size));
    sheet_chunk = ChunkMath::world_to_chunk(sheet.get_center(), chunk_param);
    for (int lod = 0; lod < 24; lod++) {
        const Vector3 lo = ChunkMath::vertices_to_world(sheet_chunk, Vector3(), chunk_param);
        const Vector3 hi = ChunkMath::vertices_to_world(sheet_chunk, Vector3(1, 1, 1) * (chunk_param.chunk_size - 1), chunk_param);
        if (AABB(lo, hi - lo).encloses(sheet)) {
            break;
        }
        sheet_chunk = ChunkMath::get_parent_from_child(sheet_chunk);
    }

    pool = memnew(QuadPagePool);
    pool->initialize(get_world_3d()->get_scenario(), page_capacity, 1 + extra_pages, chunk_param.voxel_size);
    page = pool->acquire_page();
    pool->set_page_aabb(page, AABB(sheet_origin - Vector3(0, 5, 0), Vector3(sheet_size, 10, sheet_size)));

    RenderingServer::get_singleton()->call_on_render_thread(
            callable_mp(this, &QuadPoolSpike::_rt_run).bind(pool->get_page_multimesh(page), page));
}

void QuadPoolSpike::_rt_run(RID p_multimesh, int p_page) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    ERR_FAIL_NULL_MSG(rd, "QuadPoolSpike: no RenderingDevice (Compatibility renderer?)");

    String code = String(SPIKE_COMPUTE).replace("/*GLSL_PAGE_FORMAT*/", GLSL_PAGE_FORMAT);
    Ref<RDShaderSource> source;
    source.instantiate();
    source->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, code);
    Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(source);
    const String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    ERR_FAIL_COND_MSG(!err.is_empty(), "QuadPoolSpike: SHADER COMPILATION FAILED: " + err);
    rt_shader = rd->shader_create_from_spirv(spirv);
    rt_pipeline = rd->compute_pipeline_create(rt_shader);

    Ref<RDTextureFormat> format;
    format.instantiate();
    format->set_format(QuadPagePool::TEX_FORMAT);
    format->set_width(QuadPagePool::TEX_WIDTH);
    format->set_height(pool->get_page_rows());
    format->set_usage_bits(RenderingDevice::TEXTURE_USAGE_STORAGE_BIT | RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT);
    Ref<RDTextureView> view;
    view.instantiate();
    rt_staging = rd->texture_create(format, view);
    rd->texture_clear(rt_staging, Color(0, 0, 0, 0), 0, 1, 0, 1);
    rt_counter = rd->storage_buffer_create(16);

    TypedArray<RDUniform> uniforms;
    Ref<RDUniform> u0;
    u0.instantiate();
    u0->set_uniform_type(RenderingDevice::UNIFORM_TYPE_IMAGE);
    u0->set_binding(0);
    u0->add_id(rt_staging);
    uniforms.push_back(u0);
    Ref<RDUniform> u1;
    u1.instantiate();
    u1->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
    u1->set_binding(1);
    u1->add_id(rt_counter);
    uniforms.push_back(u1);
    rt_uniform_set = rd->uniform_set_create(uniforms, rt_shader, 0);

    const Vector3 grid = ChunkMath::vertices_to_world(sheet_chunk, Vector3(), chunk_param);
    PackedByteArray push;
    push.resize(64);
    push.fill(0);
    push.encode_float(0, sheet_origin.x);
    push.encode_float(4, sheet_origin.y);
    push.encode_float(8, sheet_origin.z);
    push.encode_float(12, sheet_size);
    push.encode_u32(16, side);
    push.encode_s32(32, sheet_chunk.x);
    push.encode_s32(36, sheet_chunk.y);
    push.encode_s32(40, sheet_chunk.z);
    push.encode_float(48, grid.x);
    push.encode_float(52, grid.y);
    push.encode_float(56, grid.z);
    push.encode_float(60, chunk_param.voxel_size * ChunkMath::get_lod_size(sheet_chunk));

    const uint32_t quad_count = side * side;
    rd->buffer_clear(rt_counter, 0, 16);
    const int64_t list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(list, rt_pipeline);
    rd->compute_list_bind_uniform_set(list, rt_uniform_set, 0);
    rd->compute_list_set_push_constant(list, push, push.size());
    rd->compute_list_dispatch(list, (quad_count + 63) / 64, 1, 1);
    rd->compute_list_end();

    // The whole point: GPU -> GPU, the CPU never sees the quads or the count.
    pool->rt_copy_page_from(p_page, p_multimesh, rt_staging, 0);
    const QuadPagePool::PageRD page_rd = pool->rt_get_page_rd(p_page, p_multimesh);
    rd->buffer_copy(rt_counter, page_rd.command, 4, QuadPagePool::INDIRECT_INSTANCE_COUNT_OFFSET, 4);

    // Debug verification only.
    rd->texture_get_data_async(page_rd.texture, 0, callable_mp(this, &QuadPoolSpike::_rt_on_texture).bind(0));
    rd->buffer_get_data_async(page_rd.command, callable_mp(this, &QuadPoolSpike::_rt_on_command), 0, 20);
    VLOG(1, "QuadPoolSpike: dispatched ", quad_count, " quads into page ", p_page);
}

void QuadPoolSpike::_rt_kill(RID p_multimesh, int p_page) {
    pool->rt_kill_range(p_page, p_multimesh, KILL_A_OFFSET, KILL_A_COUNT);
    pool->rt_kill_range(p_page, p_multimesh, KILL_B_OFFSET, KILL_B_COUNT);
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    rd->texture_get_data_async(pool->rt_get_page_rd(p_page, p_multimesh).texture, 0,
            callable_mp(this, &QuadPoolSpike::_rt_on_texture).bind(1));
}

void QuadPoolSpike::_rt_on_texture(const PackedByteArray &p_data, int p_stage) {
    mutex->lock();
    texture_data[p_stage] = p_data;
    readback_ready = p_stage == 1 || !command_data.is_empty();
    mutex->unlock();
}

void QuadPoolSpike::_rt_on_command(const PackedByteArray &p_data) {
    mutex->lock();
    command_data = p_data;
    readback_ready = !texture_data[0].is_empty();
    mutex->unlock();
}

void QuadPoolSpike::_process(double p_delta) {
    mutex->lock();
    const bool ready = readback_ready;
    readback_ready = false;
    mutex->unlock();
    if (!ready) {
        return;
    }
    if (stage == 0) {
        Dictionary r;
        _verify_write(r);
        mutex->lock();
        report = r;
        mutex->unlock();
        stage = 1;
    } else if (stage == 2) {
        Dictionary r = report;
        _verify_kills(r);
        r["ok"] = (bool)r["write_ok"] && (bool)r["kill_ok"];
        r["pool"] = pool->get_stats();
        stage = 3;
        mutex->lock();
        report = r;
        mutex->unlock();
        VLOG(1, "QuadPoolSpike report: ", r);
    }
}

void QuadPoolSpike::_verify_write(Dictionary &r) {
    mutex->lock();
    const PackedByteArray bytes = texture_data[0];
    const PackedByteArray command = command_data;
    mutex->unlock();
    const uint16_t *data = reinterpret_cast<const uint16_t *>(bytes.ptr());
    const uint32_t rows = bytes.size() / PageFormat::ROW_BYTES;
    const float scale = PageFormat::pos_scale(chunk_param.chunk_size);
    // One fixed-point step, plus float rounding of the world <-> grid transforms.
    const double pos_tolerance = chunk_param.voxel_size * ChunkMath::get_lod_size(sheet_chunk) / scale + 1e-3;

    const uint32_t quad_count = side * side;
    r["expected_quads"] = quad_count;
    r["expected_groups"] = (quad_count + 63) / 64;
    r["indirect_index_count"] = command.decode_u32(0);
    r["indirect_group_count"] = command.decode_u32(4);
    r["texture_bytes_read"] = bytes.size();
    r["sheet_chunk"] = sheet_chunk;

    const float step = sheet_size / side;
    auto height = [&](float x, float z) { return sheet_origin.y + 3.0f * std::sin(x * 0.3f) * std::cos(z * 0.3f); };
    auto surf = [&](float x, float z) { return Vector3(x, height(x, z), z); };
    auto surf_normal = [&](float x, float z) {
        const float dx = 0.9f * std::cos(x * 0.3f) * std::cos(z * 0.3f);
        const float dz = -0.9f * std::sin(x * 0.3f) * std::sin(z * 0.3f);
        return Vector3(-dx, 1.0f, -dz).normalized();
    };

    // Quads land in arbitrary slots (atomicAdd), so match each slot to its grid cell by corner A.
    double max_pos_err = 0.0;
    double max_normal_err_deg = 0.0;
    int bad = 0;
    PackedInt32Array seen;
    seen.resize(quad_count);
    seen.fill(0);
    int bad_headers = 0;
    const uint32_t used_groups = (quad_count + PageFormat::QUAD_GROUP - 1) / PageFormat::QUAD_GROUP;
    for (uint32_t g = 0; g < used_groups && g < rows; g++) {
        bad_headers += PageFormat::header_chunk(PageFormat::row(data, g)) != sheet_chunk ? 1 : 0;
    }
    auto world_of = [&](uint32_t slot, int k) {
        const Vector3i chunk = PageFormat::header_chunk(PageFormat::row(data, slot / PageFormat::QUAD_GROUP));
        return ChunkMath::vertices_to_world(chunk, PageFormat::decode_local(PageFormat::corner(data, slot, k), scale), chunk_param);
    };
    for (uint32_t slot = 0; slot < quad_count && slot / PageFormat::QUAD_GROUP < rows; slot++) {
        if (PageFormat::is_zero(PageFormat::corner(data, slot, 0))) {
            bad++;
            continue;
        }
        const Vector3 a = world_of(slot, 0);
        const int qx = (int)std::lround((a.x - sheet_origin.x) / step);
        const int qz = (int)std::lround((a.z - sheet_origin.z) / step);
        if (qx < 0 || qz < 0 || qx >= side || qz >= side) {
            bad++;
            continue;
        }
        seen.set(qz * side + qx, seen[qz * side + qx] + 1);
        Vector2 c[4] = {
            Vector2(sheet_origin.x + qx * step, sheet_origin.z + qz * step),
            Vector2(sheet_origin.x + (qx + 1) * step, sheet_origin.z + qz * step),
            Vector2(sheet_origin.x + qx * step, sheet_origin.z + (qz + 1) * step),
            Vector2(sheet_origin.x + (qx + 1) * step, sheet_origin.z + (qz + 1) * step),
        };
        if (qx >= side / 2) {
            SWAP(c[1], c[2]);
        }
        for (int k = 0; k < 4; k++) {
            const Vector3 got = world_of(slot, k);
            max_pos_err = MAX(max_pos_err, (double)got.distance_to(surf(c[k].x, c[k].y)));
            const Vector3 n = PageFormat::decode_normal(PageFormat::corner(data, slot, k));
            const double ang = Math::rad_to_deg(std::acos(CLAMP(n.dot(surf_normal(c[k].x, c[k].y)), -1.0f, 1.0f)));
            max_normal_err_deg = MAX(max_normal_err_deg, ang);
        }
    }
    int missing = 0;
    int duplicated = 0;
    for (int i = 0; i < (int)quad_count; i++) {
        missing += seen[i] == 0 ? 1 : 0;
        duplicated += seen[i] > 1 ? 1 : 0;
    }
    // Slots past the sheet must be zero (the tail of the last group is drawn), and so must the rows
    // past it.
    int dirty_tail = 0;
    for (uint32_t slot = quad_count; slot < used_groups * PageFormat::QUAD_GROUP; slot++) {
        for (int k = 0; k < 4; k++) {
            dirty_tail += PageFormat::is_zero(PageFormat::corner(data, slot, k)) ? 0 : 1;
        }
    }
    for (uint32_t i = used_groups * PageFormat::TEX_WIDTH * PageFormat::TEXEL_CHANNELS; i < bytes.size() / 2; i++) {
        dirty_tail += data[i] != 0 ? 1 : 0;
    }
    r["max_position_error"] = max_pos_err;
    r["position_tolerance"] = pos_tolerance;
    r["bad_group_headers"] = bad_headers;
    r["max_normal_error_deg"] = max_normal_err_deg;
    r["bad_quads"] = bad;
    r["missing_cells"] = missing;
    r["duplicated_cells"] = duplicated;
    r["dirty_tail_floats"] = dirty_tail;
    r["write_ok"] = (int)command.decode_u32(4) == (int)((quad_count + 63) / 64) && (int)command.decode_u32(0) == 64 * 6 &&
            max_pos_err < pos_tolerance && max_normal_err_deg < 0.5 && bad == 0 && bad_headers == 0 && missing == 0 &&
            duplicated == 0 && dirty_tail == 0;
}

void QuadPoolSpike::_verify_kills(Dictionary &r) {
    mutex->lock();
    const PackedByteArray before_bytes = texture_data[0];
    const PackedByteArray after_bytes = texture_data[1];
    mutex->unlock();
    const uint16_t *before = reinterpret_cast<const uint16_t *>(before_bytes.ptr());
    const uint16_t *after = reinterpret_cast<const uint16_t *>(after_bytes.ptr());
    // Kills zero whole rows (groups), headers included.
    auto killed = [](uint32_t g) {
        const uint32_t q = g * PageFormat::QUAD_GROUP;
        return (q >= KILL_A_OFFSET && q < KILL_A_OFFSET + KILL_A_COUNT) || (q >= KILL_B_OFFSET && q < KILL_B_OFFSET + KILL_B_COUNT);
    };
    int not_zeroed = 0;
    int damaged = 0;
    const uint32_t row_channels = PageFormat::TEX_WIDTH * PageFormat::TEXEL_CHANNELS;
    const uint32_t rows = MIN(before_bytes.size(), after_bytes.size()) / PageFormat::ROW_BYTES;
    for (uint32_t g = 0; g < rows; g++) {
        for (uint32_t i = g * row_channels; i < (g + 1) * row_channels; i++) {
            if (killed(g)) {
                not_zeroed += after[i] != 0 ? 1 : 0;
            } else {
                damaged += after[i] != before[i] ? 1 : 0;
            }
        }
    }
    r["kill_not_zeroed_channels"] = not_zeroed;
    r["kill_damaged_channels"] = damaged;
    r["kill_ok"] = rows > 0 && not_zeroed == 0 && damaged == 0;
}

Dictionary QuadPoolSpike::get_report() {
    mutex->lock();
    Dictionary r = report.duplicate();
    mutex->unlock();
    return r;
}

void QuadPoolSpike::run_kill_test() {
    ERR_FAIL_COND_MSG(stage != 1, "QuadPoolSpike: run_kill_test() needs a verified write first.");
    stage = 2;
    RenderingServer::get_singleton()->call_on_render_thread(
            callable_mp(this, &QuadPoolSpike::_rt_kill).bind(pool->get_page_multimesh(page), page));
}

void QuadPoolSpike::set_debug_view(int p_view) {
    if (pool) {
        pool->set_debug_view(p_view);
    }
}

void QuadPoolSpike::_exit_tree() {
    if (!pool) {
        return;
    }
    // Bound by value: this node may be gone by the time the render thread runs it.
    RDUtils::free_on_render_thread({ rt_uniform_set, rt_pipeline, rt_shader, rt_staging, rt_counter });
    pool->destroy();
    pool = nullptr;
}
