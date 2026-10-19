#include "GPU_direct_mesher.hpp"

#include "cascade_glsl.hpp"
#include "exact_noise.hpp"
#include "gpu_direct_glsl.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"
#include "page_format.hpp"
#include "rd_utils.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

#include <algorithm>
#include <cmath>

static constexpr const char DIRECT_TEMPLATE[] = R"glsl(#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

/*GLSL_STRUCT_DEFINITION*/

/*GLSL_DIRECT_BINDINGS*/

/*GLSL_HELPER_FUNCTION*/

/*GLSL_TERRAIN_SDF*/

vec4 evaluate_sdf(WorldPos p, int lod) {
    return terrain_sdf(p, lod);
}

/*GLSL_SKIRT_SDF*/

/*GLSL_QEF_VERTEX*/

shared uint wg_quad_count;

void main() {
    // Every pass must reference the push constant block, otherwise the compiler may strip it from
    // some pipelines and the shared push constant upload would mismatch. Dynamically uniform.
    if (batch.batch_count == 0u) {
        return;
    }
    ChunkFrame chunk = chunk_frame(input_chunk[gl_GlobalInvocationID.z / chunk_param.chunk_size], chunk_param);
    ivec3 chunk_id = chunk.id_lo;/*GLSL_SET_CHUNK*/
    ivec3 cell_chunk_coordinate = ivec3(
        gl_GlobalInvocationID.x,
        gl_GlobalInvocationID.y % chunk_param.chunk_size,
        gl_GlobalInvocationID.z % chunk_param.chunk_size);

    /*GLSL_ALGORITHM_PASS*/
}
)glsl";

static const char *PASS_SOURCES[5] = {
    GLSL_DIRECT_PASS_SDF,
    GLSL_DIRECT_PASS_EDGES,
    GLSL_DIRECT_PASS_VERTICES,
    GLSL_DIRECT_PASS_RESERVE,
    GLSL_DIRECT_PASS_EMIT,
};
static const char *PASS_NAMES[5] = { "sdf", "edges", "vertices", "reserve", "emit" };
// Cascade level groups (timestamps): j >= 3, 2, 1, 0.
static const char *CASCADE_GROUP_NAMES[4] = { "top", "j2", "j1", "j0" };

static constexpr uint32_t STATUS_HEADER_UINTS = 8; // must match STATUS_PAGE_FILL in the GLSL

static uint32_t groups_of(uint32_t p_quads) {
    return (p_quads + QuadPagePool::QUAD_GROUP - 1) / QuadPagePool::QUAD_GROUP;
}

// Morton order of a chunk id relative to the world origin, for batch and page locality only: ids
// more than 2^20 units from the origin (far LODs with more than 15 LODs, a camera far from the origin)
// are clamped onto the faces of the key space and only keep a coarse locality.
static uint64_t morton_key(const ChunkId &p_chunk, const ChunkId &p_origin) {
    auto spread = [](uint64_t v) {
        v &= 0x1fffff;
        v = (v | v << 32) & 0x1f00000000ffffULL;
        v = (v | v << 16) & 0x1f0000ff0000ffULL;
        v = (v | v << 8) & 0x100f00f00f00f00fULL;
        v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
        v = (v | v << 2) & 0x1249249249249249ULL;
        return v;
    };
    constexpr int64_t bias = int64_t(1) << 20;
    const ChunkId d = p_chunk - p_origin;
    const ChunkId rel(CLAMP(d.x, -bias, bias - 1), CLAMP(d.y, -bias, bias - 1), CLAMP(d.z, -bias, bias - 1));
    return spread(uint64_t(rel.x + bias)) | (spread(uint64_t(rel.y + bias)) << 1) | (spread(uint64_t(rel.z + bias)) << 2);
}

void GPUDirectMesher::initialize(SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers, RID p_scenario, const Settings &p_settings) {
    sdf = p_sdf;
    chunk_param = p_chunk_param;
    scenario = p_scenario;
    settings = p_settings;
    settings.max_batches_per_frame = CLAMP(settings.max_batches_per_frame, 1u, MAX_BATCHES_PER_FRAME);
    if (settings.max_in_flight == 0) {
        settings.max_in_flight = 3 * settings.max_batches_per_frame;
    }
    if (settings.spare_pages == 0) {
        settings.spare_pages = settings.max_in_flight * settings.pages_per_batch;
    }
    mutex.instantiate();

    chunk_queue.resize(p_lod_numbers);
    remesh_queue.resize(p_lod_numbers);
    cost_model.resize(p_lod_numbers);

    pool = memnew(QuadPagePool);
    pool->initialize(scenario, settings.page_capacity, settings.initial_pages, chunk_param.voxel_size);
    // The page shader rebuilds world positions from the group header's chunk id.
    pool->set_material_param("chunk_size", chunk_param.chunk_size);
    pool->set_material_param("skirt_size", chunk_param.skirt_size);
    pool->set_material_param("pos_scale", PageFormat::pos_scale(chunk_param.chunk_size));
    // Render positions are relative to the world origin (low words, wrapping).
    pool->set_material_param("origin_chunk_lo", WorldCoord::lo32_i(chunk_param.origin_chunk));
    page_state.resize(pool->get_page_count());

    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_initialize));
}

RID GPUDirectMesher::_rt_compile(const char *p_pass, const char *p_name) {
    const bool vertex_pass = (p_pass == GLSL_DIRECT_PASS_VERTICES);
    // Dual contouring: the vertex pass places its vertex with the shared QEF (gpu_glsl.hpp).
    const String pass = dual_contouring && vertex_pass ? dual_contouring_pass(p_pass) : String(p_pass);
    String shader = DIRECT_TEMPLATE;
    // The instance table (archetype instances) adds the DefRecord/List buffers at 8/9, the exact int64
    // helpers (i64_sub, for the table's noise lattice) and the ChunkInput global g_sdf_chunk. A terrain
    // without a table gets exactly the same shader text as before (byte-identical).
    String bindings = GLSL_DIRECT_BINDINGS;
    String helpers = String(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + String(GLSL_EXACT_NOISE);
    String set_chunk;
    if (SDFGpuTable *table = sdf->get_gpu_table()) {
        bindings += table->glsl_buffers(8, 9);
        helpers += GPUFrame::glsl_int64();
        set_chunk = table->glsl_set_chunk();
    }
    shader = shader.replace("/*GLSL_STRUCT_DEFINITION*/", String(GLSL_STRUCT_DEFINITION) + String(GLSL_FRAME_STRUCTS));
    shader = shader.replace("/*GLSL_DIRECT_BINDINGS*/", bindings);
    shader = shader.replace("/*GLSL_PAGE_FORMAT*/", GLSL_PAGE_FORMAT);
    shader = shader.replace("/*GLSL_HELPER_FUNCTION*/", helpers);
    shader = shader.replace("/*GLSL_TERRAIN_SDF*/", sdf->get_glsl_code());
    shader = shader.replace("/*GLSL_SET_CHUNK*/", set_chunk);
    shader = shader.replace("/*GLSL_QEF_VERTEX*/", String(GLSL_VERTEX_NORMAL) + (dual_contouring ? GLSL_QEF_VERTEX : ""));
    shader = CascadeGLSL::fill_mesher_pass(shader, sdf, pass, vertex_pass);
    RDUtils::dump_shader(String("direct_") + p_name, shader);

    Ref<RDShaderSource> source;
    source.instantiate();
    source->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, shader);
    Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(source);
    const String error_msg = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!error_msg.is_empty()) {
        UtilityFunctions::printerr("--- SDF SHADER COMPILATION FAILED (GPUDirectMesher) ---");
        UtilityFunctions::printerr("ERROR MESSAGE:\n", error_msg);
        PackedStringArray lines = shader.split("\n");
        for (int i = 0; i < lines.size(); i++) {
            UtilityFunctions::printerr(String::num_int64(i + 1) + " | " + lines[i]);
        }
        return RID();
    }
    return rd->shader_create_from_spirv(spirv);
}

void GPUDirectMesher::_rt_initialize() {
    rd = RenderingServer::get_singleton()->get_rendering_device();
    ERR_FAIL_NULL_MSG(rd, "GPUDirectMesher: no RenderingDevice (Compatibility renderer is not supported).");

    const uint64_t cs = chunk_param.chunk_size;
    const uint64_t cells = cs * cs * cs;
    const uint64_t batch = settings.batch_size;

    buffer_chunk_dir = rd->storage_buffer_create(batch * 16);
    buffer_input_chunk = rd->storage_buffer_create(batch * GPUFrame::CHUNK_INPUT_BYTES);
    buffer_sdf = rd->storage_buffer_create(batch * cells * 16);
    buffer_edges = rd->storage_buffer_create(batch * cells * 3 * sizeof(float));
    buffer_vertices = rd->storage_buffer_create(batch * cells * 16);
    {
        Ref<RDTextureFormat> format;
        format.instantiate();
        format->set_format(QuadPagePool::TEX_FORMAT);
        format->set_width(QuadPagePool::TEX_WIDTH);
        format->set_height(settings.pages_per_batch * pool->get_page_rows());
        format->set_usage_bits(RenderingDevice::TEXTURE_USAGE_STORAGE_BIT | RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
                RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT);
        Ref<RDTextureView> view;
        view.instantiate();
        staging_texture = rd->texture_create(format, view);
    }
    status_size_bytes = ((STATUS_HEADER_UINTS + settings.pages_per_batch) * 4 + 15) / 16 * 16;
    buffer_status = rd->storage_buffer_create(status_size_bytes);

    const PackedByteArray param_data = GPUFrame::chunk_param_bytes(chunk_param);
    buffer_chunk_param = rd->uniform_buffer_create(param_data.size(), param_data);

    for (int i = 0; i < 5; i++) {
        shaders[i] = _rt_compile(PASS_SOURCES[i], PASS_NAMES[i]);
        ERR_FAIL_COND_MSG(!shaders[i].is_valid(), String("GPUDirectMesher: pass failed to compile: ") + PASS_NAMES[i]);
        pipelines[i] = rd->compute_pipeline_create(shaders[i]);
    }

    auto make_uniform = [](int binding, RID buffer, RenderingDevice::UniformType type) {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(type);
        u->set_binding(binding);
        u->add_id(buffer);
        return u;
    };
    TypedArray<RDUniform> uniforms;
    uniforms.push_back(make_uniform(0, buffer_chunk_dir, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(1, buffer_input_chunk, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(2, buffer_sdf, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(3, buffer_edges, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(4, buffer_vertices, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    uniforms.push_back(make_uniform(5, buffer_chunk_param, RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER));
    uniforms.push_back(make_uniform(6, staging_texture, RenderingDevice::UNIFORM_TYPE_IMAGE));
    uniforms.push_back(make_uniform(7, buffer_status, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    if (SDFGpuTable *table = sdf->get_gpu_table()) {
        // Sized for the capacity up front; the records arrive with the first dispatch (_table_payload).
        buffer_table_records = rd->storage_buffer_create(uint64_t(MAX<uint32_t>(table->instance_capacity(), 1u)) * table->record_size());
        const uint64_t list_words = uint64_t(settings.batch_size) * table->max_chunk_list(); // no list is longer (build_lists)
        table_list_capacity = list_words * 4;
        buffer_table_list = rd->storage_buffer_create(table_list_capacity);
        uniforms.push_back(make_uniform(8, buffer_table_records, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
        uniforms.push_back(make_uniform(9, buffer_table_list, RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER));
    }
    // All passes declare the same bindings, so one set (created against pass 0) serves them all.
    uniform_set = rd->uniform_set_create(uniforms, shaders[0], 0);

    uint64_t cascade_mb = 0;
    if (sdf->get_cascade() != nullptr) {
        cascade = memnew(CascadePasses);
        if (!cascade->create(rd, sdf, chunk_param, settings.batch_size, buffer_input_chunk, buffer_chunk_param, buffer_sdf)) {
            rd->free_rid(uniform_set);
            uniform_set = RID(); // nothing dispatches
            ERR_FAIL_MSG("GPUDirectMesher: the cascade passes failed to compile.");
        }
        cascade_mb = cascade->scratch_bytes() / 1048576;
        mutex->lock();
        cascade_scratch_bytes = cascade->scratch_bytes();
        mutex->unlock();
    }

    VLOG(1, "GPUDirectMesher: initialized (batch ", settings.batch_size, ", ", settings.pages_per_batch,
            " pages x ", settings.page_capacity, " quads per batch, scratch ",
            (batch * cells * (16 + 12 + 16)) / 1048576, " MB, cascade scratch ", cascade_mb, " MB)");
}

void GPUDirectMesher::_rt_read_timestamps() {
    // Timestamps captured in an earlier frame are readable now; read each captured frame once, as
    // several batches may dispatch per frame. Our markers are "vx<batch>_<step>".
    const uint64_t captured_frame = rd->get_captured_timestamps_frame();
    if (captured_frame == timestamps_frame_read) {
        return;
    }
    timestamps_frame_read = captured_frame;
    struct BatchTimes {
        int64_t begin = -1;
        int64_t pass[5] = { -1, -1, -1, -1, -1 };
        int64_t cascade[3] = { -1, -1, -1 }; // after the groups j >= 3, j = 2, j = 1
        int64_t copy = -1;
    };
    HashMap<int64_t, BatchTimes> batches;
    const uint32_t n = rd->get_captured_timestamps_count();
    for (uint32_t i = 0; i < n; i++) {
        const String name = rd->get_captured_timestamp_name(i);
        if (!name.begins_with("vx")) {
            continue;
        }
        BatchTimes &b = batches[name.substr(2).get_slice("_", 0).to_int()];
        const String step = name.get_slice("_", 1);
        const int64_t t = rd->get_captured_timestamp_gpu_time(i);
        if (step == "begin") {
            b.begin = t;
        } else if (step == "copy") {
            b.copy = t;
        } else if (step.begins_with("c")) {
            const int group = step.substr(1).to_int();
            if (group >= 0 && group < 3) {
                b.cascade[group] = t;
            }
        } else if (step.begins_with("p")) {
            const int pass = step.substr(1).to_int();
            if (pass >= 0 && pass < 5) {
                b.pass[pass] = t;
            }
        }
    }
    // GPU timestamps are in nanoseconds; TimingStat stores microseconds.
    mutex->lock();
    for (const KeyValue<int64_t, BatchTimes> &kv : batches) {
        const BatchTimes &b = kv.value;
        if (b.begin < 0 || b.copy < 0 || b.pass[4] < 0) {
            continue;
        }
        int64_t prev = b.begin;
        for (int i = 0; i < 5; i++) {
            if (b.pass[i] >= prev) {
                stats.gpu_pass[i].add((b.pass[i] - prev) / 1000);
                prev = b.pass[i];
            }
        }
        if (b.cascade[0] >= b.begin && b.cascade[1] >= b.cascade[0] && b.cascade[2] >= b.cascade[1] && b.pass[0] >= b.cascade[2]) {
            const int64_t bounds[5] = { b.begin, b.cascade[0], b.cascade[1], b.cascade[2], b.pass[0] };
            for (int g = 0; g < 4; g++) {
                stats.gpu_cascade[g].add((bounds[g + 1] - bounds[g]) / 1000);
            }
        }
        stats.gpu_compute.add((b.pass[4] - b.begin) / 1000);
        stats.gpu_copy.add((b.copy - b.pass[4]) / 1000);
        if (b.copy > b.begin) {
            measured_batches.push_back({ (uint32_t)kv.key, uint64_t(b.copy - b.begin) / 1000 });
        }
    }
    mutex->unlock();
}

void GPUDirectMesher::_rt_record_samples(uint32_t p_count, int p_min_lod, const PackedByteArray &p_push, const String &p_prefix) {
    if (cascade == nullptr) {
        const uint32_t g = chunk_param.chunk_size / 8;
        const int64_t list = rd->compute_list_begin();
        rd->compute_list_bind_compute_pipeline(list, pipelines[0]);
        rd->compute_list_bind_uniform_set(list, uniform_set, 0);
        rd->compute_list_set_push_constant(list, p_push, p_push.size());
        rd->compute_list_dispatch(list, g, g, g * p_count);
        rd->compute_list_end();
        return;
    }
    // One list per level (the render graph orders them through the scratch buffers); timestamps "_c0", "_c1",
    // "_c2" after the groups j >= 3, j = 2 and j = 1, empty groups included.
    auto mark = [&](int p_group) {
        if (!p_prefix.is_empty()) {
            rd->capture_timestamp(p_prefix + String("_c") + String::num_int64(p_group));
        }
    };
    int j = cascade->level_count(p_min_lod) - 1;
    for (; j >= 3; j--) {
        cascade->record_level(j, 0, p_count);
    }
    mark(0);
    if (j == 2) {
        cascade->record_level(j--, 0, p_count);
    }
    mark(1);
    if (j == 1) {
        cascade->record_level(j--, 0, p_count);
    }
    mark(2);
    cascade->record_level(0, 0, p_count);
}

void GPUDirectMesher::_rt_upload_table(const PackedByteArray &p_list, const PackedByteArray &p_records) {
    if (!buffer_table_records.is_valid()) {
        return;
    }
    // The records the batch's lists were built against (main thread, same snapshot), when they changed.
    if (!p_records.is_empty()) {
        SDFGpuTable::upload_records(rd, buffer_table_records, p_records);
    }
    if (!p_list.is_empty()) {
        ERR_FAIL_COND_MSG(uint64_t(p_list.size()) > table_list_capacity, vformat("GPUDirectMesher: batch list %d bytes over the %d B buffer.", p_list.size(), table_list_capacity));
        rd->buffer_update(buffer_table_list, 0, p_list.size(), p_list);
    }
}

uint32_t GPUDirectMesher::_table_payload(const std::vector<ChunkId> &p_chunks, PackedByteArray &r_ids, PackedByteArray &r_list, PackedByteArray &r_records) {
    SDFGpuTable *table = sdf->get_gpu_table();
    if (!table) {
        return 0;
    }
    // Main thread, like the edits: the lists and the records come from the same snapshot.
    table->build_lists(p_chunks, chunk_param, r_ids, r_list);
    const uint32_t version = table->version();
    if (version != sent_table_version) {
        r_records = table->record_bytes(sent_table_version, sent_table_version);
    }
    return version;
}

void GPUDirectMesher::_rt_dispatch(uint32_t p_batch_id, const PackedByteArray &p_chunk_ids, uint32_t p_count,
        const PackedInt32Array &p_pages, int p_min_lod, const PackedByteArray &p_list, const PackedByteArray &p_records) {
    if (!uniform_set.is_valid()) {
        return; // initialization failed (already reported)
    }
    const uint64_t t0 = now_usec();
    _rt_read_timestamps();
    _rt_upload_table(p_list, p_records);

    rd->buffer_update(buffer_input_chunk, 0, p_chunk_ids.size(), p_chunk_ids);
    rd->buffer_clear(buffer_chunk_dir, 0, p_count * 16);
    rd->buffer_clear(buffer_status, 0, status_size_bytes);
    // Pages draw whole groups of quads: slots past the fill must be zero (degenerate), not stale.
    rd->texture_clear(staging_texture, Color(0, 0, 0, 0), 0, 1, 0, 1); // headers included

    PackedByteArray push;
    push.resize(16);
    push.encode_u32(0, p_count);
    push.encode_u32(4, p_pages.size());
    push.encode_u32(8, settings.page_capacity);
    push.encode_u32(12, 0);

    const uint32_t g = chunk_param.chunk_size / 8;
    const uint32_t groups[5][3] = {
        { g, g, g * p_count }, // sdf
        { g, 3 * g, g * p_count }, // edges (y spans the 3 axes)
        { g, g, g * p_count }, // vertices
        { 1, 1, 1 }, // reserve
        { g, g, g * p_count }, // emit
    };

    const String prefix = "vx" + String::num_int64(p_batch_id);
    rd->capture_timestamp(prefix + String("_begin"));
    for (int i = 0; i < 5; i++) {
        if (i == 0) {
            _rt_record_samples(p_count, p_min_lod, push, prefix);
            rd->capture_timestamp(prefix + String("_p0"));
            continue;
        }
        // One list per pass so a timestamp can sit between passes; the render graph orders them.
        const int64_t list = rd->compute_list_begin();
        rd->compute_list_bind_compute_pipeline(list, pipelines[i]);
        rd->compute_list_bind_uniform_set(list, uniform_set, 0);
        rd->compute_list_set_push_constant(list, push, push.size());
        rd->compute_list_dispatch(list, groups[i][0], groups[i][1], groups[i][2]);
        rd->compute_list_end();
        rd->capture_timestamp(prefix + String("_p") + String::num_int64(i));
    }

    // GPU -> GPU: quads into the page textures, used group counts into the indirect instance counts.
    for (int k = 0; k < p_pages.size(); k++) {
        pool->rt_copy_page_from(p_pages[k], staging_texture, k * pool->get_page_rows());
        const QuadPagePool::PageRD page_rd = pool->rt_get_page_rd(p_pages[k]);
        ERR_CONTINUE_MSG(!page_rd.command.is_valid(), vformat("GPUDirectMesher: batch %d: page %d has no command buffer", p_batch_id, p_pages[k]));
        rd->buffer_copy(buffer_status, page_rd.command, (STATUS_HEADER_UINTS + k) * 4,
                QuadPagePool::INDIRECT_INSTANCE_COUNT_OFFSET, 4);
    }
    rd->capture_timestamp(prefix + String("_copy"));

    // Bookkeeping only (where each chunk landed); rendering does not wait for it.
    // Counted only when queued: a refused readback never calls back (_rt_destroy waits for the count).
    rt_readbacks_in_flight += rd->buffer_get_data_async(buffer_chunk_dir, callable_mp(this, &GPUDirectMesher::_rt_on_directory).bind(p_batch_id), 0, p_count * 16) == OK;
    rt_readbacks_in_flight += rd->buffer_get_data_async(buffer_status, callable_mp(this, &GPUDirectMesher::_rt_on_status).bind(p_batch_id), 0, status_size_bytes) == OK;

    mutex->lock();
    stats.rt_record_cpu.add(now_usec() - t0);
    mutex->unlock();
}

void GPUDirectMesher::_rt_dispatch_count(const PackedByteArray &p_chunk_ids, uint32_t p_count, uint32_t p_batch_id, int p_min_lod,
        const PackedByteArray &p_list, const PackedByteArray &p_records) {
    if (!uniform_set.is_valid()) {
        return;
    }
    _rt_upload_table(p_list, p_records);
    rd->buffer_update(buffer_input_chunk, 0, p_chunk_ids.size(), p_chunk_ids);
    rd->buffer_clear(buffer_chunk_dir, 0, p_count * 16);
    rd->buffer_clear(buffer_status, 0, status_size_bytes);

    PackedByteArray push;
    push.resize(16);
    push.encode_u32(0, p_count);
    push.encode_u32(4, 0);
    push.encode_u32(8, settings.page_capacity);
    push.encode_u32(12, 0);

    // sdf, edges, vertices: the vertices pass leaves the exact quad count in chunk_dir.x.
    const uint32_t g = chunk_param.chunk_size / 8;
    const uint32_t groups[3][3] = {
        { g, g, g * p_count },
        { g, 3 * g, g * p_count },
        { g, g, g * p_count },
    };
    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            _rt_record_samples(p_count, p_min_lod, push, String());
            continue;
        }
        const int64_t list = rd->compute_list_begin();
        rd->compute_list_bind_compute_pipeline(list, pipelines[i]);
        rd->compute_list_bind_uniform_set(list, uniform_set, 0);
        rd->compute_list_set_push_constant(list, push, push.size());
        rd->compute_list_dispatch(list, groups[i][0], groups[i][1], groups[i][2]);
        rd->compute_list_end();
    }
    rt_readbacks_in_flight += rd->buffer_get_data_async(buffer_chunk_dir, callable_mp(this, &GPUDirectMesher::_rt_on_audit_directory).bind(p_batch_id), 0, p_count * 16) == OK;
}

void GPUDirectMesher::_rt_on_audit_directory(const PackedByteArray &p_data, uint32_t p_batch_id) {
    mutex->lock();
    CompletedBatch c;
    c.batch_id = p_batch_id;
    c.directory = p_data;
    audit_completed.push_back(c);
    mutex->unlock();
    _rt_readback_done();
}

void GPUDirectMesher::_rt_on_directory(const PackedByteArray &p_data, uint32_t p_batch_id) {
    mutex->lock();
    CompletedBatch &c = partial[p_batch_id];
    c.batch_id = p_batch_id;
    c.directory = p_data;
    if (!c.status.is_empty()) {
        completed.push_back(c);
        partial.erase(p_batch_id);
    }
    mutex->unlock();
    _rt_readback_done();
}

void GPUDirectMesher::_rt_on_status(const PackedByteArray &p_data, uint32_t p_batch_id) {
    mutex->lock();
    CompletedBatch &c = partial[p_batch_id];
    c.batch_id = p_batch_id;
    c.status = p_data;
    if (!c.directory.is_empty()) {
        completed.push_back(c);
        partial.erase(p_batch_id);
    }
    mutex->unlock();
    _rt_readback_done();
}

void GPUDirectMesher::_rt_kill(const PackedInt32Array &p_pages, const PackedInt64Array &p_ranges) {
    for (int i = 0; i < p_pages.size(); i++) {
        pool->rt_kill_range(p_pages[i], p_ranges[i * 2], p_ranges[i * 2 + 1]);
    }
}

void GPUDirectMesher::_rt_compact(const PackedInt32Array &p_copies, const PackedInt32Array &p_group_counts) {
    for (int i = 0; i + 4 < p_copies.size(); i += 5) {
        pool->rt_copy_rows(p_copies[i], p_copies[i + 1], p_copies[i + 2], p_copies[i + 3], p_copies[i + 4]);
    }
    for (int i = 0; i + 1 < p_group_counts.size(); i += 2) {
        pool->rt_set_group_count(p_group_counts[i], p_group_counts[i + 1]);
    }
}

AABB GPUDirectMesher::_chunk_aabb(const ChunkId &p_chunk) const {
    const Vector3 center = ChunkMath::chunk_render_center(p_chunk, chunk_param);
    const float size = ChunkMath::world_chunk_size(p_chunk, chunk_param);
    // Skirt cells extend past the chunk bounds; be generous rather than cull visible geometry.
    const float margin = size * 0.25f;
    const Vector3 half(size * 0.5f + margin, size * 0.5f + margin, size * 0.5f + margin);
    return AABB(center - half, half * 2.0f);
}

void GPUDirectMesher::_update_dirty_bounds() {
    for (const int page : dirty_bounds_pages) {
        PageState &ps = page_state[page];
        if (!ps.bounds_dirty) {
            continue; // released (and maybe reacquired) since it was marked
        }
        ps.bounds_dirty = false;
        if (ps.chunks.empty()) {
            continue;
        }
        AABB bounds = _chunk_aabb(ps.chunks[0].id);
        for (size_t i = 1; i < ps.chunks.size(); i++) {
            bounds.merge_with(_chunk_aabb(ps.chunks[i].id));
        }
        ps.bounds = bounds;
        pool->set_page_aabb(page, bounds);
    }
    dirty_bounds_pages.clear();
}

bool GPUDirectMesher::_pop_queue(ChunkId &r_chunk) {
    // Instance edits first, finest LOD first: the chunk under the edit is what the user is looking at.
    for (std::vector<ChunkId> &queue : remesh_queue) {
        if (queue.empty()) {
            continue;
        }
        r_chunk = queue.back();
        queue.pop_back();
        remesh_queued.erase(r_chunk);
        stats.remesh_pops++;
        return true;
    }
    for (std::vector<ChunkId> &queue : chunk_queue) {
        if (queue.empty()) {
            continue;
        }
        r_chunk = queue.back();
        queue.pop_back();
        return true;
    }
    return false;
}

void GPUDirectMesher::_queue_remesh(const ChunkId &p_chunk) {
    const int lod = ChunkMath::get_lod(p_chunk);
    if (lod < 0 || lod >= (int)remesh_queue.size() || remesh_queued.has(p_chunk)) {
        return; // queued once: an edit storm re-requires the version, it doesn't dispatch the chunk again
    }
    remesh_queued.insert(p_chunk);
    remesh_queue[lod].push_back(p_chunk);
}

void GPUDirectMesher::remesh_chunks(const std::vector<ChunkId> &p_chunks) {
    const uint32_t version = sdf->get_gpu_table() ? sdf->get_gpu_table()->version() : 0u;
    for (const ChunkId &chunk : p_chunks) {
        required_version[chunk] = version;
        _queue_remesh(chunk);
        stats.remesh_chunks++;
    }
    // Chunks in flight for their first mesh aren't loaded yet, so the terrain doesn't list them: their
    // result, computed from the old table, must not be kept either (it is dropped and re-queued on landing).
    for (const KeyValue<uint32_t, InFlightBatch> &kv : in_flight) {
        if (kv.value.table_version >= version) {
            continue;
        }
        for (const ChunkId &chunk : kv.value.chunks) {
            if (sdf->edit_affects_chunk(chunk)) {
                required_version[chunk] = version;
            }
        }
    }
    uint64_t queued = 0;
    for (const std::vector<ChunkId> &q : remesh_queue) {
        queued += q.size();
    }
    stats.remesh_queue_max = MAX(stats.remesh_queue_max, queued);
}

void GPUDirectMesher::_flush_kills() {
    if (pending_kill_pages.is_empty()) {
        return;
    }
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_kill)
                    .bind(pending_kill_pages, pending_kill_ranges));
    pending_kill_pages = PackedInt32Array();
    pending_kill_ranges = PackedInt64Array();
}

void GPUDirectMesher::process_queue() {
    frame++;
    // Before acquiring pages: a page released since it was marked must not be touched.
    _update_dirty_bounds();
    // Before acquiring pages too: a kill must not land on a page reused by this frame's batches.
    _flush_kills();

    // Batches of one frame share the scratch buffers, the staging texture and the status buffer: the
    // render graph orders them like batches of consecutive frames (each readback copies its own data
    // at its point in the graph). Audit batches only run once the mesh queue is empty.
    _update_cost_model();
    const bool budgeted = settings.gpu_budget_ms > 0.0f;
    const double budget_us = budgeted ? settings.gpu_budget_ms * 1000.0 : INFINITY;
    double budget_left = budget_us;
    uint32_t dispatched = 0;
    // Bound the batches waiting for their directory (each one holds pages_per_batch pages).
    while (dispatched < settings.max_batches_per_frame && in_flight.size() + audit_in_flight.size() < settings.max_in_flight &&
            budget_left > 0.0) {
        if (!_dispatch_mesh_batch(budget_left) && !_dispatch_audit_batch()) {
            break;
        }
        dispatched++;
    }
    if (dispatched > 0) {
        stats.dispatch_frames++;
        stats.in_flight_max = MAX(stats.in_flight_max, (uint64_t)(in_flight.size() + audit_in_flight.size()));
        if (budgeted) {
            stats.budget_frame_estimate.add(uint64_t(budget_us - budget_left));
        }
    }
}

void GPUDirectMesher::_update_cost_model() {
    std::vector<std::pair<uint32_t, uint64_t>> measured;
    mutex->lock();
    measured.swap(measured_batches);
    mutex->unlock();
    for (const std::pair<uint32_t, uint64_t> &m : measured) {
        HashMap<uint32_t, std::vector<uint32_t>>::Iterator it = batch_lod_counts.find(m.first);
        if (it == batch_lod_counts.end()) {
            continue;
        }
        const double estimate = cost_model.estimate(it->value);
        if (estimate > 0.0) {
            stats.model_ratio_sum += m.second / estimate;
            stats.model_ratio_count++;
        }
        cost_model.observe(it->value, (double)m.second);
        batch_lod_counts.remove(it);
    }
    // A batch whose timestamps were never captured (over the per-frame limit) is forgotten after a while.
    if (batch_lod_counts.size() > 1024) {
        std::vector<uint32_t> old;
        for (const KeyValue<uint32_t, std::vector<uint32_t>> &kv : batch_lod_counts) {
            if (next_batch_id - kv.key > 1024) {
                old.push_back(kv.key);
            }
        }
        for (uint32_t id : old) {
            batch_lod_counts.erase(id);
        }
    }
}

bool GPUDirectMesher::_dispatch_mesh_batch(double &r_budget_us) {
    std::vector<ChunkId> chunks;
    chunks.reserve(settings.batch_size);
    std::vector<uint32_t> lod_counts(lod_count, 0);
    double estimate = 0.0;
    ChunkId chunk;
    // With a budget the batch stops once its estimate reaches what's left (a frame dispatches at least
    // one chunk: budget_left starts > 0).
    while (chunks.size() < settings.batch_size && estimate < r_budget_us && _pop_queue(chunk)) {
        chunks.push_back(chunk);
        const int lod = ChunkMath::get_lod(chunk);
        if (lod >= 0 && lod < lod_count) {
            lod_counts[lod]++;
            estimate += cost_model.estimate(lod);
        }
    }
    if (chunks.empty()) {
        return false;
    }
    r_budget_us -= estimate;
    const uint64_t t0 = now_usec();

    // The queue already pops in Morton order; re-sorting covers re-queued overflow chunks.
    const ChunkId origin = chunk_param.origin_chunk;
    std::sort(chunks.begin(), chunks.end(), [origin](const ChunkId &a, const ChunkId &b) { return morton_key(a, origin) < morton_key(b, origin); });

    InFlightBatch batch;
    batch.dispatch_frame = frame;
    int min_lod = ChunkMath::get_lod(chunks[0]);
    for (const ChunkId &c : chunks) {
        min_lod = MIN(min_lod, ChunkMath::get_lod(c));
    }

    AABB bounds = _chunk_aabb(chunks[0]);
    PackedByteArray ids;
    ids.resize(chunks.size() * GPUFrame::CHUNK_INPUT_BYTES);
    for (size_t i = 0; i < chunks.size(); i++) {
        GPUFrame::encode_chunk_input(ids, (int64_t)i, chunks[i], chunk_param);
        bounds.merge_with(_chunk_aabb(chunks[i]));
        working_generation.insert(chunks[i]);
    }
    PackedByteArray list;
    PackedByteArray records;
    const uint32_t table_version = _table_payload(chunks, ids, list, records);

    PackedInt32Array pages;
    for (uint32_t k = 0; k < settings.pages_per_batch; k++) {
        const int page = pool->acquire_page();
        if ((int)page_state.size() <= page) {
            page_state.resize(page + 1);
        }
        page_state[page] = PageState();
        // Conservative until the directory says which chunks really landed in this page.
        pool->set_page_aabb(page, bounds);
        batch.pages.push_back(page);
        pages.push_back(page);
    }

    const uint32_t batch_id = next_batch_id++;
    const uint32_t count = chunks.size();
    batch_lod_counts[batch_id] = std::move(lod_counts);
    batch.table_version = table_version;
    batch.chunks = std::move(chunks);
    in_flight[batch_id] = std::move(batch);
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_dispatch)
                    .bind(batch_id, ids, count, pages, min_lod, list, records));

    stats.batches++;
    stats.dispatch_cpu.add(now_usec() - t0);
    VLOG(3, "GPUDirectMesher: dispatched batch ", batch_id, " (", (int64_t)count, " chunks)");
    return true;
}

bool GPUDirectMesher::_dispatch_audit_batch() {
    if (audit_queue.empty()) {
        return false;
    }
    const size_t n = MIN((size_t)settings.batch_size, audit_queue.size());
    std::vector<ChunkId> chunks(audit_queue.end() - n, audit_queue.end());
    audit_queue.resize(audit_queue.size() - n);

    PackedByteArray ids;
    ids.resize(chunks.size() * GPUFrame::CHUNK_INPUT_BYTES);
    int min_lod = ChunkMath::get_lod(chunks[0]);
    for (size_t i = 0; i < chunks.size(); i++) {
        GPUFrame::encode_chunk_input(ids, (int64_t)i, chunks[i], chunk_param);
        min_lod = MIN(min_lod, ChunkMath::get_lod(chunks[i]));
    }
    PackedByteArray list;
    PackedByteArray records;
    _table_payload(chunks, ids, list, records);
    const uint32_t batch_id = next_batch_id++;
    audit_in_flight[batch_id] = chunks;
    stats.audit_batches++;
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_dispatch_count)
                    .bind(ids, (uint32_t)chunks.size(), batch_id, min_lod, list, records));
    return true;
}

void GPUDirectMesher::audit_count_chunks(const std::vector<ChunkId> &p_chunks) {
    audit_queue.insert(audit_queue.end(), p_chunks.begin(), p_chunks.end());
}

void GPUDirectMesher::take_audit_counts(std::vector<std::pair<ChunkId, uint32_t>> &r_counts) {
    r_counts.insert(r_counts.end(), audit_counts.begin(), audit_counts.end());
    audit_counts.clear();
}

bool GPUDirectMesher::get_results(VoxelState &output) {
    mutex->lock();
    std::vector<CompletedBatch> done;
    done.swap(completed);
    std::vector<CompletedBatch> audit_done;
    audit_done.swap(audit_completed);
    mutex->unlock();

    for (CompletedBatch &c : audit_done) {
        HashMap<uint32_t, std::vector<ChunkId>>::Iterator it = audit_in_flight.find(c.batch_id);
        ERR_CONTINUE(it == audit_in_flight.end());
        const PackedInt32Array dir = c.directory.to_int32_array();
        for (size_t i = 0; i < it->value.size(); i++) {
            audit_counts.push_back({ it->value[i], (uint32_t)dir[i * 4 + 0] });
        }
        audit_in_flight.remove(it);
    }

    if (done.empty()) {
        return false;
    }

    for (CompletedBatch &c : done) {
        const uint64_t t0 = now_usec();
        HashMap<uint32_t, InFlightBatch>::Iterator it = in_flight.find(c.batch_id);
        ERR_CONTINUE(it == in_flight.end());
        InFlightBatch &batch = it->value;

        const PackedInt32Array status = c.status.to_int32_array();
        const PackedInt32Array dir = c.directory.to_int32_array();
        stats.quads_emitted += (uint32_t)status[0];
        stats.overflow_chunks += (uint32_t)status[1];
        stats.missing_neighbor += (uint32_t)status[2];
        stats.nan_vertices += (uint32_t)status[3];

        std::vector<AABB> page_bounds(batch.pages.size());
        struct StaleRange {
            uint32_t page_index, offset, count;
        };
        std::vector<StaleRange> stale_ranges;
        std::vector<bool> page_has_bounds(batch.pages.size(), false);

        for (size_t k = 0; k < batch.pages.size(); k++) {
            page_state[batch.pages[k]].groups = (uint32_t)status[STATUS_HEADER_UINTS + k];
            if ((uint32_t)status[STATUS_HEADER_UINTS + k] > pool->get_page_rows()) {
                ERR_PRINT(vformat("GPUDirectMesher: batch %d page %d: status fill %d groups", c.batch_id, (int64_t)k, status[STATUS_HEADER_UINTS + k]));
            }
        }

        for (size_t i = 0; i < batch.chunks.size(); i++) {
            const ChunkId &chunk_id = batch.chunks[i];
            const uint32_t count = dir[i * 4 + 0];
            const uint32_t base = dir[i * 4 + 2];
            const uint32_t flags = dir[i * 4 + 3];
            working_generation.erase(chunk_id);

            if (flags & 1u) { // overflow: try again in a later batch
                stats.overflow_requeued++;
                if (output.loaded_chunk.has(chunk_id)) {
                    // A re-mesh (instance edit): send_chunk rebuilds chunk_queue from the scan's dirty
                    // chunks only, which a loaded chunk isn't, so it would be lost there.
                    _queue_remesh(chunk_id);
                    continue;
                }
                const int lod = ChunkMath::get_lod(chunk_id);
                if (lod >= 0 && lod < (int)chunk_queue.size()) {
                    chunk_queue[lod].push_back(chunk_id);
                }
                continue;
            }

            // Staleness (instance edits): this result was computed from a table version older than the
            // chunk's latest edit. Drop it and re-mesh from the current table.
            if (required_version.has(chunk_id)) {
                const uint32_t required = required_version[chunk_id];
                if (required > batch.table_version) {
                    _queue_remesh(chunk_id);
                    stats.remesh_stale_dropped++;
                    // Its quads are already in the page: kill them, or the old surface stays drawn next
                    // to the chunk's re-mesh for as long as the page lives.
                    if (count > 0) {
                        const uint32_t k = base / settings.page_capacity;
                        if (k < batch.pages.size()) {
                            stale_ranges.push_back({ k, base % settings.page_capacity, count });
                        }
                    }
                    continue;
                }
                required_version.erase(chunk_id);
            }

            ChunkOutput out;
            out.chunk_id = chunk_id;
            out.triangle_count = count * 2;
            out.drawn_triangles = count * 2;
            if (count > 0) {
                const uint32_t k = base / settings.page_capacity;
                if (k >= batch.pages.size() || base % settings.page_capacity + count > settings.page_capacity) {
                    ERR_PRINT(vformat("GPUDirectMesher: bad directory entry: batch %d chunk %d count %d base %d flags %d", c.batch_id, (int64_t)i, count, base, flags));
                    continue;
                }
                out.owner = this;
                out.page = batch.pages[k];
                out.quad_offset = base % settings.page_capacity;
                out.quad_count = count;
                PageState &ps = page_state[out.page];
                ps.live_quads += count;
                ps.live_groups += groups_of(count);
                ps.chunks.push_back({ chunk_id, out.quad_offset, count });
                stats.live_quads += count;
                const AABB box = _chunk_aabb(chunk_id);
                page_bounds[k] = page_has_bounds[k] ? page_bounds[k].merge(box) : box;
                page_has_bounds[k] = true;
            }
            stats.chunks_meshed++;
            output.add_new_loaded(out);
        }

        // Dropped (stale) results in pages that stay: their slots are drawn until killed. A page left
        // without chunks is hidden and reset below, slots included.
        for (const StaleRange &r : stale_ranges) {
            const int page = batch.pages[r.page_index];
            if (page_state[page].chunks.empty()) {
                continue;
            }
            page_state[page].settled = false;
            pending_kill_pages.push_back(page);
            pending_kill_ranges.push_back(r.offset);
            pending_kill_ranges.push_back(r.count);
        }
        for (size_t k = 0; k < batch.pages.size(); k++) {
            const int page = batch.pages[k];
            if (page_state[page].chunks.empty()) {
                page_state[page] = PageState();
                pool->release_page(page);
                stats.pages_returned_unused++;
            } else {
                page_state[page].bounds = page_bounds[k];
                pool->set_page_aabb(page, page_bounds[k]);
            }
        }

        stats.readback_frames_total += frame - batch.dispatch_frame;
        stats.readback_frames_max = MAX(stats.readback_frames_max, frame - batch.dispatch_frame);
        in_flight.remove(it);
        stats.collect_cpu.add(now_usec() - t0);
    }
    return true;
}

void GPUDirectMesher::release_chunk(const ChunkOutput &p_chunk) {
    ERR_FAIL_INDEX(p_chunk.page, (int)page_state.size());
    PageState &ps = page_state[p_chunk.page];
    ps.live_quads -= p_chunk.quad_count;
    ps.live_groups -= groups_of(p_chunk.quad_count);
    stats.live_quads -= p_chunk.quad_count;
    stats.kills++;
    for (size_t i = 0; i < ps.chunks.size(); i++) {
        if (ps.chunks[i].id == p_chunk.chunk_id) {
            ps.chunks[i] = ps.chunks.back();
            ps.chunks.pop_back();
            break;
        }
    }
    if (ps.chunks.empty()) {
        // Last chunk in the page: hide and reset the whole page instead of clearing the range.
        ps = PageState();
        pool->release_page(p_chunk.page);
        return;
    }
    ps.settled = false;
    if (!ps.bounds_dirty) {
        ps.bounds_dirty = true;
        dirty_bounds_pages.push_back(p_chunk.page);
    }
    pending_kill_pages.push_back(p_chunk.page);
    pending_kill_ranges.push_back(p_chunk.quad_offset);
    pending_kill_ranges.push_back(p_chunk.quad_count);
}

void GPUDirectMesher::compact(VoxelState &r_state) {
    // Kills recorded this frame must reach the GPU before any page is copied, reused or freed.
    _flush_kills();
    if (settings.compaction) {
        const uint64_t t0 = now_usec();
        const uint32_t page_groups = settings.page_capacity / QuadPagePool::QUAD_GROUP;
        // Pages less than half full of live quads: dead slots left by destroyed chunks, or the
        // last, partly filled page of a batch. Pages of in-flight batches have no chunks yet.
        std::vector<std::pair<uint64_t, int>> candidates;
        for (int page = 0; page < (int)page_state.size(); page++) {
            const PageState &ps = page_state[page];
            if (ps.chunks.empty() || ps.settled || ps.live_groups * 2 > page_groups) {
                continue;
            }
            uint64_t key = UINT64_MAX;
            for (const PageChunk &c : ps.chunks) {
                key = MIN(key, morton_key(c.id, chunk_param.origin_chunk));
            }
            candidates.push_back({ key, page });
        }
        if (!candidates.empty()) {
            // Neighbouring candidates in Morton order are the ones worth merging.
            std::sort(candidates.begin(), candidates.end());
            std::vector<int> window;
            for (size_t i = 0; i < candidates.size() && window.size() < settings.compact_pages_per_frame; i++) {
                window.push_back(candidates[i].second);
            }
            if (_compact_pages(window, r_state)) {
                stats.compact_cpu.add(now_usec() - t0);
            }
        }
    }
    pool->trim(settings.spare_pages, settings.compact_pages_per_frame);
}

bool GPUDirectMesher::_compact_pages(const std::vector<int> &p_pages, VoxelState &r_state) {
    struct Move {
        uint64_t key;
        int src_page;
        PageChunk chunk;
    };
    std::vector<Move> moves;
    uint32_t src_groups = 0;
    for (const int page : p_pages) {
        for (const PageChunk &c : page_state[page].chunks) {
            moves.push_back({ morton_key(c.id, chunk_param.origin_chunk), page, c });
        }
        src_groups += page_state[page].groups;
    }
    std::sort(moves.begin(), moves.end(), [](const Move &a, const Move &b) { return a.key < b.key; });

    // Plan: pack the survivors in Morton order, like a batch, into as few pages as fit.
    struct Dest {
        uint32_t groups = 0;
        AABB bounds;
        float smallest_chunk = 0.0f;
        size_t end = 0; // moves [previous dest's end, end)
    };
    const uint32_t page_groups = settings.page_capacity / QuadPagePool::QUAD_GROUP;
    std::vector<Dest> dests;
    std::vector<uint32_t> dst_row(moves.size());
    uint32_t dst_groups = 0;
    for (size_t i = 0; i < moves.size(); i++) {
        const ChunkId &id = moves[i].chunk.id;
        const uint32_t g = groups_of(moves[i].chunk.count);
        const AABB box = _chunk_aabb(id);
        const float size = ChunkMath::world_chunk_size(id, chunk_param);
        bool fits = false;
        if (!dests.empty()) {
            const Dest &d = dests.back();
            const float smallest = MIN(d.smallest_chunk, size);
            fits = d.groups + g <= page_groups &&
                    d.bounds.merge(box).get_longest_axis_size() <= settings.compact_max_extent * smallest;
        }
        if (!fits) {
            Dest d;
            d.bounds = box;
            d.smallest_chunk = size;
            dests.push_back(d);
        }
        Dest &d = dests.back();
        dst_row[i] = d.groups;
        d.groups += g;
        d.bounds.merge_with(box);
        d.smallest_chunk = MIN(d.smallest_chunk, size);
        d.end = i + 1;
        dst_groups += g;
    }

    // Worth it if it frees a page, or a quarter of the drawn groups were dead.
    if (dests.size() >= p_pages.size() && (src_groups - dst_groups) * 4 < src_groups) {
        for (const int page : p_pages) {
            page_state[page].settled = true;
        }
        stats.compact_skipped++;
        return false;
    }

    PackedInt32Array copies; // src page, dst page, src row, dst row, rows
    PackedInt32Array group_counts; // dst page, groups
    size_t begin = 0;
    for (const Dest &d : dests) {
        const int page = pool->acquire_page();
        if ((int)page_state.size() <= page) {
            page_state.resize(page + 1);
        }
        PageState &ps = page_state[page];
        ps = PageState();
        ps.groups = d.groups;
        ps.bounds = d.bounds;
        pool->set_page_aabb(page, d.bounds);
        for (size_t i = begin; i < d.end; i++) {
            const Move &m = moves[i];
            const uint32_t rows = groups_of(m.chunk.count);
            const uint32_t src_row = m.chunk.offset / QuadPagePool::QUAD_GROUP;
            const int n = copies.size();
            // Chunks that were neighbours in the source stay neighbours: extend the last copy.
            if (n > 0 && copies[n - 5] == m.src_page && copies[n - 4] == page &&
                    copies[n - 3] + copies[n - 1] == (int)src_row && copies[n - 2] + copies[n - 1] == (int)dst_row[i]) {
                copies.set(n - 1, copies[n - 1] + rows);
            } else {
                copies.append_array({ m.src_page, page, (int)src_row, (int)dst_row[i], (int)rows });
            }
            PageChunk moved = m.chunk;
            moved.offset = dst_row[i] * QuadPagePool::QUAD_GROUP;
            ps.chunks.push_back(moved);
            ps.live_quads += moved.count;
            ps.live_groups += rows;

            ChunkOutput *out = r_state.loaded_chunk.getptr(m.chunk.id);
            if (out && out->owner == this && out->page == m.src_page) {
                out->page = page;
                out->quad_offset = moved.offset;
            } else {
                ERR_PRINT(vformat("GPUDirectMesher: compacted chunk %s is not loaded where its page says.", WorldCoord::to_string(m.chunk.id)));
            }
        }
        group_counts.append_array({ page, (int)d.groups });
        begin = d.end;
    }
    // Queued before the sources' release, so the copies read them before anything reuses them.
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_compact)
                    .bind(copies, group_counts));
    for (const int page : p_pages) {
        page_state[page] = PageState();
        pool->release_page(page);
    }

    stats.compact_rounds++;
    stats.pages_compacted += p_pages.size();
    stats.pages_compacted_into += dests.size();
    stats.chunks_moved += moves.size();
    stats.compact_copies += copies.size() / 5;
    VLOG(3, "GPUDirectMesher: compacted ", (int64_t)p_pages.size(), " pages into ", (int64_t)dests.size(),
            " (", (int64_t)moves.size(), " chunks, ", (int64_t)copies.size() / 5, " copies)");
    return true;
}

void GPUDirectMesher::send_chunk(const LODHashSet &p_chunks) {
    // Chunk ids are chunk centres in the same units at every LOD, so one Morton key orders them
    // spatially. Sorted once per scan; consecutive pops then form compact batches and pages.
    std::vector<std::vector<std::pair<uint64_t, ChunkId>>> keyed(chunk_queue.size());
    for (const ChunkId &chunk_pos : p_chunks) {
        const int lod = ChunkMath::get_lod(chunk_pos);
        if (lod < 0 || lod >= (int)chunk_queue.size()) {
            continue;
        }
        if (!working_generation.has(chunk_pos)) {
            keyed[lod].push_back({ morton_key(chunk_pos, chunk_param.origin_chunk), chunk_pos });
        }
    }
    for (size_t lod = 0; lod < chunk_queue.size(); lod++) {
        std::sort(keyed[lod].begin(), keyed[lod].end(),
                [](const std::pair<uint64_t, ChunkId> &a, const std::pair<uint64_t, ChunkId> &b) { return a.first > b.first; });
        std::vector<ChunkId> &queue = chunk_queue[lod];
        queue.clear();
        queue.reserve(keyed[lod].size());
        for (const std::pair<uint64_t, ChunkId> &k : keyed[lod]) {
            queue.push_back(k.second);
        }
    }
}

bool GPUDirectMesher::is_idle() {
    if (!in_flight.is_empty()) {
        return false;
    }
    for (const std::vector<ChunkId> &q : chunk_queue) {
        if (!q.empty()) {
            return false;
        }
    }
    // Re-meshes queued by an edit or by a stale result (get_results) are pending work too.
    return remesh_queued.is_empty();
}

void GPUDirectMesher::set_debug_view(int p_view) {
    pool->set_debug_view(p_view);
}

Dictionary GPUDirectMesher::get_stats() {
    Dictionary d;
    int64_t queued = 0;
    for (const std::vector<ChunkId> &q : chunk_queue) {
        queued += q.size();
    }
    int pages_with_live = 0;
    double extent_sum = 0.0;
    double extent_max = 0.0;
    int64_t drawn_groups = 0;
    for (const PageState &ps : page_state) {
        if (!ps.chunks.empty()) {
            pages_with_live++;
            drawn_groups += ps.groups;
            // Longest axis of the page AABB in units of its finest chunk (frustum culling
            // granularity; metres would be dominated by the kilometre-wide far LODs).
            float finest = ChunkMath::world_chunk_size(ps.chunks[0].id, chunk_param);
            for (const PageChunk &c : ps.chunks) {
                finest = MIN(finest, ChunkMath::world_chunk_size(c.id, chunk_param));
            }
            const double extent = ps.bounds.get_longest_axis_size() / finest;
            extent_sum += extent;
            extent_max = MAX(extent_max, extent);
        }
    }
    d["page_aabb_extent_chunks_mean"] = pages_with_live ? extent_sum / pages_with_live : 0.0;
    d["page_aabb_extent_chunks_max"] = extent_max;
    d["queued"] = queued;
    int64_t remesh_queued = 0;
    for (const std::vector<ChunkId> &q : remesh_queue) {
        remesh_queued += q.size();
    }
    d["remesh_queue"] = remesh_queued;
    d["remesh_queue_max"] = (int64_t)stats.remesh_queue_max;
    d["remesh_chunks"] = (int64_t)stats.remesh_chunks;
    d["remesh_pops"] = (int64_t)stats.remesh_pops;
    d["remesh_stale_dropped"] = (int64_t)stats.remesh_stale_dropped;
    d["in_flight"] = (int64_t)in_flight.size();
    d["in_flight_max"] = (int64_t)stats.in_flight_max;
    // Mesh and audit batches per frame that dispatched any.
    d["batches_per_dispatch_frame"] = stats.dispatch_frames ? double(stats.batches + stats.audit_batches) / stats.dispatch_frames : 0.0;
    d["batches"] = (int64_t)stats.batches;
    d["gpu_budget_ms"] = settings.gpu_budget_ms;
    d["budget_frame_estimate"] = stats.budget_frame_estimate.to_dict();
    // How well the cost model predicts a batch (1 = exactly), and its cost per chunk per LOD (us, -1 unmeasured).
    d["cost_model_ratio_avg"] = stats.model_ratio_count ? stats.model_ratio_sum / stats.model_ratio_count : 0.0;
    PackedFloat64Array cost_us;
    for (double c : cost_model.us_per_chunk) {
        cost_us.push_back(c);
    }
    d["cost_us_per_chunk"] = cost_us;
    d["chunks_meshed"] = (int64_t)stats.chunks_meshed;
    d["quads_emitted"] = (int64_t)stats.quads_emitted;
    d["overflow_chunks"] = (int64_t)stats.overflow_chunks;
    d["overflow_requeued"] = (int64_t)stats.overflow_requeued;
    d["audit_batches"] = (int64_t)stats.audit_batches;
    d["missing_neighbor"] = (int64_t)stats.missing_neighbor;
    d["nan_vertices"] = (int64_t)stats.nan_vertices;
    d["kills"] = (int64_t)stats.kills;
    d["pages_returned_unused"] = (int64_t)stats.pages_returned_unused;
    // Quad slots the vertex shader runs on (whole groups of live pages): dead and padding included.
    d["drawn_quad_slots"] = drawn_groups * QuadPagePool::QUAD_GROUP;
    d["compact_rounds"] = (int64_t)stats.compact_rounds;
    d["compact_skipped"] = (int64_t)stats.compact_skipped;
    d["pages_compacted"] = (int64_t)stats.pages_compacted;
    d["pages_compacted_into"] = (int64_t)stats.pages_compacted_into;
    d["chunks_moved"] = (int64_t)stats.chunks_moved;
    d["compact_copies"] = (int64_t)stats.compact_copies;
    d["compact_cpu"] = stats.compact_cpu.to_dict();
    d["live_quads"] = (int64_t)stats.live_quads;
    d["pages_with_live_chunks"] = pages_with_live;
    const int64_t slots_in_live_pages = (int64_t)pages_with_live * settings.page_capacity;
    d["page_occupancy"] = slots_in_live_pages ? (double)stats.live_quads / slots_in_live_pages : 0.0;
    d["dispatch_cpu"] = stats.dispatch_cpu.to_dict();
    d["collect_cpu"] = stats.collect_cpu.to_dict();
    mutex->lock();
    d["rt_record_cpu"] = stats.rt_record_cpu.to_dict();
    d["gpu_compute"] = stats.gpu_compute.to_dict();
    d["gpu_copy"] = stats.gpu_copy.to_dict();
    for (int i = 0; i < 5; i++) {
        d[String("gpu_pass_") + PASS_NAMES[i]] = stats.gpu_pass[i].to_dict();
    }
    if (sdf->get_cascade() != nullptr) {
        for (int g = 0; g < 4; g++) {
            d[String("gpu_cascade_") + CASCADE_GROUP_NAMES[g]] = stats.gpu_cascade[g].to_dict();
        }
        d["cascade_scratch_mb"] = double(cascade_scratch_bytes) / 1048576.0;
    }
    mutex->unlock();
    Dictionary frames;
    const uint64_t collected = stats.batches - in_flight.size();
    frames["avg"] = collected ? (double)stats.readback_frames_total / collected : 0.0;
    frames["max"] = (int64_t)stats.readback_frames_max;
    d["readback_latency_frames"] = frames;
    d["pool"] = pool->get_stats();
    return d;
}

void GPUDirectMesher::destroy() {
    chunk_queue.clear();
    // Queues its own render-thread cleanup. `pool` is not cleared: render-thread work queued before
    // (dispatch, kills) still uses it and runs first (FIFO).
    pool->destroy();
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_destroy));
}

void GPUDirectMesher::_rt_destroy() {
    if (cascade != nullptr) {
        // Its uniform set uses the input, param and sdf buffers freed below.
        cascade->free();
        memdelete(cascade);
        cascade = nullptr;
    }
    // Dependents (uniform set, pipelines) before the shaders and resources they use.
    RID rids[] = { uniform_set,
        pipelines[0], pipelines[1], pipelines[2], pipelines[3], pipelines[4],
        shaders[0], shaders[1], shaders[2], shaders[3], shaders[4],
        buffer_chunk_dir, buffer_input_chunk, buffer_sdf, buffer_edges, buffer_vertices,
        buffer_chunk_param, staging_texture, buffer_status, buffer_table_records, buffer_table_list };
    if (rd) {
        for (const RID &rid : rids) {
            if (rid.is_valid()) {
                rd->free_rid(rid);
            }
        }
    }
    // The RIDs can go now (the readbacks copied them to staging when they were recorded), not the object.
    rt_destroying = true;
    if (rt_readbacks_in_flight == 0) {
        memdelete(this);
    }
}

void GPUDirectMesher::_rt_readback_done() {
    rt_readbacks_in_flight--;
    if (rt_destroying && rt_readbacks_in_flight == 0) {
        memdelete(this);
    }
}
