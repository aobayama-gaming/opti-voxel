#include "quad_pool_spike.hpp"

#include "debug_log.hpp"
#include "gpu_glsl.hpp"
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

layout(set = 0, binding = 0, rgba32f) uniform restrict writeonly image2D staging;
layout(set = 0, binding = 1, std430) restrict buffer Counter { uint counter[]; };
layout(push_constant, std430) uniform Params {
    vec4 origin_size; // xyz origin, w sheet size
    uvec4 dims;       // x = side
} params;

#define TEX_WIDTH 1024u

/*GLSL_QUAD_PACK*/

float height(vec2 xz) { return params.origin_size.y + 3.0 * sin(xz.x * 0.3) * cos(xz.y * 0.3); }
vec3 surf(vec2 xz) { return vec3(xz.x, height(xz), xz.y); }
vec3 surf_normal(vec2 xz) {
    float dx = 0.9 * cos(xz.x * 0.3) * cos(xz.y * 0.3);
    float dz = -0.9 * sin(xz.x * 0.3) * sin(xz.y * 0.3);
    return normalize(vec3(-dx, 1.0, -dz));
}

void store_corner(uint texel, vec2 xz) {
    imageStore(staging, ivec2(texel % TEX_WIDTH, texel / TEX_WIDTH), vec4(surf(xz), pack_normal_oct24(surf_normal(xz)) + 1.0));
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
    store_corner(slot * 4u + 0u, base);
    store_corner(slot * 4u + 1u, c1);
    store_corner(slot * 4u + 2u, c2);
    store_corner(slot * 4u + 3u, base + vec2(step, step));
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

    pool = memnew(QuadPagePool);
    pool->initialize(get_world_3d()->get_scenario(), page_capacity, 1 + extra_pages, 0.5f);
    page = pool->acquire_page();
    pool->set_page_aabb(page, AABB(sheet_origin - Vector3(0, 5, 0), Vector3(sheet_size, 10, sheet_size)));

    RenderingServer::get_singleton()->call_on_render_thread(
            callable_mp(this, &QuadPoolSpike::_rt_run).bind(pool->get_page_multimesh(page), page));
}

void QuadPoolSpike::_rt_run(RID p_multimesh, int p_page) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    ERR_FAIL_NULL_MSG(rd, "QuadPoolSpike: no RenderingDevice (Compatibility renderer?)");

    String code = String(SPIKE_COMPUTE).replace("/*GLSL_QUAD_PACK*/", GLSL_QUAD_PACK);
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
    format->set_format(RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT);
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

    PackedByteArray push;
    push.resize(32);
    push.fill(0);
    push.encode_float(0, sheet_origin.x);
    push.encode_float(4, sheet_origin.y);
    push.encode_float(8, sheet_origin.z);
    push.encode_float(12, sheet_size);
    push.encode_u32(16, side);

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

static Vector3 oct_decode24(float p_packed) {
    const uint32_t p = (uint32_t)p_packed;
    const float fx = float(p & 4095u) / 4095.0f * 2.0f - 1.0f;
    const float fy = float(p >> 12u) / 4095.0f * 2.0f - 1.0f;
    Vector3 n(fx, fy, 1.0f - std::abs(fx) - std::abs(fy));
    const float t = MAX(-n.z, 0.0f);
    n.x += n.x >= 0.0f ? -t : t;
    n.y += n.y >= 0.0f ? -t : t;
    return n.normalized();
}

void QuadPoolSpike::_verify_write(Dictionary &r) {
    mutex->lock();
    const PackedFloat32Array f = texture_data[0].to_float32_array();
    const PackedByteArray command = command_data;
    mutex->unlock();

    const uint32_t quad_count = side * side;
    r["expected_quads"] = quad_count;
    r["expected_groups"] = (quad_count + 63) / 64;
    r["indirect_index_count"] = command.decode_u32(0);
    r["indirect_group_count"] = command.decode_u32(4);
    r["texture_floats_read"] = f.size();

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
    for (uint32_t slot = 0; slot < quad_count && (slot + 1) * 16 <= (uint32_t)f.size(); slot++) {
        const float *q = f.ptr() + slot * 16; // 4 corners x (x, y, z, packed normal + 1)
        const Vector3 a(q[0], q[1], q[2]);
        if (!a.is_finite() || q[3] <= 0.0f) {
            bad++;
            continue;
        }
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
            const Vector3 got(q[k * 4 + 0], q[k * 4 + 1], q[k * 4 + 2]);
            max_pos_err = MAX(max_pos_err, (double)got.distance_to(surf(c[k].x, c[k].y)));
            const Vector3 n = oct_decode24(q[k * 4 + 3] - 1.0f);
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
    // Slots past the sheet must be zero (the tail of the last group is drawn).
    int dirty_tail = 0;
    for (uint32_t i = quad_count * 16; i < (uint32_t)f.size(); i++) {
        dirty_tail += f[i] != 0.0f ? 1 : 0;
    }
    r["max_position_error"] = max_pos_err;
    r["max_normal_error_deg"] = max_normal_err_deg;
    r["bad_quads"] = bad;
    r["missing_cells"] = missing;
    r["duplicated_cells"] = duplicated;
    r["dirty_tail_floats"] = dirty_tail;
    r["write_ok"] = (int)command.decode_u32(4) == (int)((quad_count + 63) / 64) && (int)command.decode_u32(0) == 64 * 6 &&
            max_pos_err < 1e-3 && max_normal_err_deg < 0.5 && bad == 0 && missing == 0 && duplicated == 0 && dirty_tail == 0;
}

void QuadPoolSpike::_verify_kills(Dictionary &r) {
    mutex->lock();
    const PackedFloat32Array before = texture_data[0].to_float32_array();
    const PackedFloat32Array after = texture_data[1].to_float32_array();
    mutex->unlock();
    auto killed = [](uint32_t q) {
        return (q >= KILL_A_OFFSET && q < KILL_A_OFFSET + KILL_A_COUNT) || (q >= KILL_B_OFFSET && q < KILL_B_OFFSET + KILL_B_COUNT);
    };
    int not_zeroed = 0;
    int damaged = 0;
    const uint32_t quads = MIN(before.size(), after.size()) / 16;
    for (uint32_t q = 0; q < quads; q++) {
        for (int i = 0; i < 16; i++) {
            const float a = after[q * 16 + i];
            if (killed(q)) {
                not_zeroed += a != 0.0f ? 1 : 0;
            } else {
                damaged += a != before[q * 16 + i] ? 1 : 0;
            }
        }
    }
    r["kill_not_zeroed_floats"] = not_zeroed;
    r["kill_damaged_floats"] = damaged;
    r["kill_ok"] = quads > 0 && not_zeroed == 0 && damaged == 0;
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
