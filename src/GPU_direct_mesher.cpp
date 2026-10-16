#include "GPU_direct_mesher.hpp"

#include "gpu_direct_glsl.hpp"
#include "gpu_glsl.hpp"
#include "noise.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

#include <algorithm>

static constexpr const char DIRECT_TEMPLATE[] = R"glsl(#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

/*GLSL_STRUCT_DEFINITION*/

/*GLSL_DIRECT_BINDINGS*/

/*GLSL_HELPER_FUNCTION*/

/*GLSL_QUAD_PACK*/

/*GLSL_TERRAIN_SDF*/

vec4 evaluate_sdf(vec3 pos) {
    return terrain_sdf(pos);
}

/*GLSL_SKIRT_SDF*/

shared uint wg_quad_count;

void main() {
    // Every pass must reference the push constant block, otherwise the compiler may strip it from
    // some pipelines and the shared push constant upload would mismatch. Dynamically uniform.
    if (batch.batch_count == 0u) {
        return;
    }
    ivec3 chunk_id = input_chunk[gl_GlobalInvocationID.z / chunk_param.chunk_size].xyz;
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

static constexpr uint32_t STATUS_HEADER_UINTS = 8; // must match STATUS_PAGE_FILL in the GLSL

static uint64_t morton_key(const Vector3i &p_chunk) {
    auto spread = [](uint64_t v) {
        v &= 0x1fffff;
        v = (v | v << 32) & 0x1f00000000ffffULL;
        v = (v | v << 16) & 0x1f0000ff0000ffULL;
        v = (v | v << 8) & 0x100f00f00f00f00fULL;
        v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
        v = (v | v << 2) & 0x1249249249249249ULL;
        return v;
    };
    const uint64_t bias = 1 << 20;
    return spread(p_chunk.x + bias) | (spread(p_chunk.y + bias) << 1) | (spread(p_chunk.z + bias) << 2);
}

void GPUDirectMesher::initialize(SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers, RID p_scenario, const Settings &p_settings) {
    sdf = p_sdf;
    chunk_param = p_chunk_param;
    scenario = p_scenario;
    settings = p_settings;
    mutex.instantiate();

    chunk_queue.resize(p_lod_numbers, nullptr);
    for (int i = 0; i < p_lod_numbers; ++i) {
        chunk_queue[i] = memnew(LODHashSet);
    }

    pool = memnew(QuadPagePool);
    pool->initialize(scenario, settings.page_capacity, settings.initial_pages, chunk_param.voxel_size);
    pool->set_material_param("chunk_world_dimension", chunk_param.chunk_world_dimension());
    page_state.resize(pool->get_page_count());

    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_initialize));
}

RID GPUDirectMesher::_rt_compile(const char *p_pass) {
    String shader = DIRECT_TEMPLATE;
    shader = shader.replace("/*GLSL_STRUCT_DEFINITION*/", GLSL_STRUCT_DEFINITION);
    shader = shader.replace("/*GLSL_DIRECT_BINDINGS*/", GLSL_DIRECT_BINDINGS);
    shader = shader.replace("/*GLSL_HELPER_FUNCTION*/", String(GLSL_HELPER_FUNCTION) + String(GLSL_NOISE));
    shader = shader.replace("/*GLSL_QUAD_PACK*/", GLSL_QUAD_PACK);
    shader = shader.replace("/*GLSL_TERRAIN_SDF*/", sdf->get_glsl_code());
    shader = shader.replace("/*GLSL_SKIRT_SDF*/", GLSL_SKIRT_SDF);
    shader = shader.replace("/*GLSL_ALGORITHM_PASS*/", p_pass);

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
    buffer_input_chunk = rd->storage_buffer_create(batch * 16);
    buffer_sdf = rd->storage_buffer_create(batch * cells * 16);
    buffer_edges = rd->storage_buffer_create(batch * cells * 3 * sizeof(float));
    buffer_vertices = rd->storage_buffer_create(batch * cells * 16);
    {
        Ref<RDTextureFormat> format;
        format.instantiate();
        format->set_format(RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT);
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

    PackedByteArray param_data;
    param_data.resize(16);
    param_data.encode_s32(0, chunk_param.chunk_size);
    param_data.encode_float(4, chunk_param.voxel_size);
    param_data.encode_s32(8, chunk_param.skirt_size);
    param_data.encode_s32(12, chunk_param.use_skirt ? 1 : 0);
    buffer_chunk_param = rd->uniform_buffer_create(param_data.size(), param_data);

    for (int i = 0; i < 5; i++) {
        shaders[i] = _rt_compile(PASS_SOURCES[i]);
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
    // All passes declare the same bindings, so one set (created against pass 0) serves them all.
    uniform_set = rd->uniform_set_create(uniforms, shaders[0], 0);

    VLOG(1, "GPUDirectMesher: initialized (batch ", settings.batch_size, ", ", settings.pages_per_batch,
            " pages x ", settings.page_capacity, " quads per batch, scratch ",
            (batch * cells * (16 + 12 + 16)) / 1048576, " MB)");
}

void GPUDirectMesher::_rt_read_timestamps() {
    // Timestamps captured in an earlier frame are readable now. Our markers are "vx<batch>_<step>".
    const uint32_t n = rd->get_captured_timestamps_count();
    int64_t t_begin = -1;
    int64_t t_pass[5] = { -1, -1, -1, -1, -1 };
    int64_t t_copy = -1;
    for (uint32_t i = 0; i < n; i++) {
        const String name = rd->get_captured_timestamp_name(i);
        if (!name.begins_with("vx")) {
            continue;
        }
        const int64_t t = rd->get_captured_timestamp_gpu_time(i);
        if (name.ends_with("_begin")) {
            t_begin = t;
        } else if (name.ends_with("_copy")) {
            t_copy = t;
        } else {
            const int pass = name.get_slice("_p", 1).to_int();
            if (pass >= 0 && pass < 5) {
                t_pass[pass] = t;
            }
        }
    }
    if (t_begin < 0 || t_copy < 0 || t_pass[4] < 0) {
        return;
    }
    // GPU timestamps are in nanoseconds; TimingStat stores microseconds.
    mutex->lock();
    int64_t prev = t_begin;
    for (int i = 0; i < 5; i++) {
        if (t_pass[i] >= prev) {
            stats.gpu_pass[i].add((t_pass[i] - prev) / 1000);
            prev = t_pass[i];
        }
    }
    stats.gpu_compute.add((t_pass[4] - t_begin) / 1000);
    stats.gpu_copy.add((t_copy - t_pass[4]) / 1000);
    mutex->unlock();
}

void GPUDirectMesher::_rt_dispatch(uint32_t p_batch_id, const PackedByteArray &p_chunk_ids, uint32_t p_count,
        const Array &p_page_multimeshes, const PackedInt32Array &p_pages) {
    if (!uniform_set.is_valid()) {
        return; // initialization failed (already reported)
    }
    const uint64_t t0 = now_usec();
    _rt_read_timestamps();

    rd->buffer_update(buffer_input_chunk, 0, p_chunk_ids.size(), p_chunk_ids);
    rd->buffer_clear(buffer_chunk_dir, 0, p_count * 16);
    rd->buffer_clear(buffer_status, 0, status_size_bytes);
    // Pages draw whole groups of quads: slots past the fill must be zero (degenerate), not stale.
    rd->texture_clear(staging_texture, Color(0, 0, 0, 0), 0, 1, 0, 1);

    PackedByteArray push;
    push.resize(16);
    push.encode_u32(0, p_count);
    push.encode_u32(4, p_pages.size());
    push.encode_u32(8, settings.page_capacity);
    push.encode_u32(12, 0);

    const uint32_t g = chunk_param.chunk_size / 8;
    const uint32_t groups[5][3] = {
        { g, g, g * p_count },     // sdf
        { g, 3 * g, g * p_count }, // edges (y spans the 3 axes)
        { g, g, g * p_count },     // vertices
        { 1, 1, 1 },               // reserve
        { g, g, g * p_count },     // emit
    };

    const String prefix = "vx" + String::num_int64(p_batch_id);
    rd->capture_timestamp(prefix + String("_begin"));
    for (int i = 0; i < 5; i++) {
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
        pool->rt_copy_page_from(p_pages[k], p_page_multimeshes[k], staging_texture, k * pool->get_page_rows());
        const QuadPagePool::PageRD page_rd = pool->rt_get_page_rd(p_pages[k], p_page_multimeshes[k]);
        rd->buffer_copy(buffer_status, page_rd.command, (STATUS_HEADER_UINTS + k) * 4,
                QuadPagePool::INDIRECT_INSTANCE_COUNT_OFFSET, 4);
    }
    rd->capture_timestamp(prefix + String("_copy"));

    // Bookkeeping only (where each chunk landed); rendering does not wait for it.
    rd->buffer_get_data_async(buffer_chunk_dir, callable_mp(this, &GPUDirectMesher::_rt_on_directory).bind(p_batch_id), 0, p_count * 16);
    rd->buffer_get_data_async(buffer_status, callable_mp(this, &GPUDirectMesher::_rt_on_status).bind(p_batch_id), 0, status_size_bytes);

    mutex->lock();
    stats.rt_record_cpu.add(now_usec() - t0);
    mutex->unlock();
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
}

void GPUDirectMesher::_rt_kill(const PackedInt32Array &p_pages, const Array &p_multimeshes, const PackedInt64Array &p_ranges) {
    for (int i = 0; i < p_pages.size(); i++) {
        pool->rt_kill_range(p_pages[i], p_multimeshes[i], p_ranges[i * 2], p_ranges[i * 2 + 1]);
    }
}

AABB GPUDirectMesher::_chunk_aabb(const Vector3i &p_chunk) const {
    const Vector3 center = ChunkMath::chunk_to_world(p_chunk, chunk_param);
    const float size = ChunkMath::world_chunk_size(p_chunk, chunk_param);
    // Skirt cells extend past the chunk bounds; be generous rather than cull visible geometry.
    const float margin = size * 0.25f;
    const Vector3 half(size * 0.5f + margin, size * 0.5f + margin, size * 0.5f + margin);
    return AABB(center - half, half * 2.0f);
}

bool GPUDirectMesher::_pop_queue(Vector3i &r_chunk) {
    for (LODHashSet *queue : chunk_queue) {
        if (queue == nullptr || queue->is_empty()) {
            continue;
        }
        LODHashSet::Iterator it = queue->begin();
        r_chunk = *it;
        queue->remove(it);
        return true;
    }
    return false;
}

void GPUDirectMesher::process_queue() {
    frame++;

    if (!pending_kill_pages.is_empty()) {
        RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_kill)
                        .bind(pending_kill_pages, pending_kill_multimeshes, pending_kill_ranges));
        // Packed arrays are copy-on-write, but Array is shared by reference: the bound copy must
        // not be cleared, so start a new one.
        pending_kill_pages = PackedInt32Array();
        pending_kill_multimeshes = Array();
        pending_kill_ranges = PackedInt64Array();
    }

    // Bound the number of batches waiting for their directory (each one holds pages_per_batch pages).
    if (in_flight.size() >= 4) {
        return;
    }

    std::vector<Vector3i> chunks;
    Vector3i chunk;
    while (chunks.size() < settings.batch_size && _pop_queue(chunk)) {
        chunks.push_back(chunk);
    }
    if (chunks.empty()) {
        return;
    }
    const uint64_t t0 = now_usec();

    // Spatially coherent batches -> pages with tight bounds -> page-level frustum culling works.
    std::sort(chunks.begin(), chunks.end(), [](const Vector3i &a, const Vector3i &b) { return morton_key(a) < morton_key(b); });

    InFlightBatch batch;
    batch.chunks = chunks;
    batch.dispatch_frame = frame;

    AABB bounds = _chunk_aabb(chunks[0]);
    PackedByteArray ids;
    ids.resize(chunks.size() * 16);
    for (size_t i = 0; i < chunks.size(); i++) {
        ids.encode_s32(i * 16 + 0, chunks[i].x);
        ids.encode_s32(i * 16 + 4, chunks[i].y);
        ids.encode_s32(i * 16 + 8, chunks[i].z);
        ids.encode_s32(i * 16 + 12, 0);
        bounds.merge_with(_chunk_aabb(chunks[i]));
        working_generation.insert(chunks[i]);
    }

    Array multimeshes;
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
        multimeshes.push_back(pool->get_page_multimesh(page));
    }

    const uint32_t batch_id = next_batch_id++;
    in_flight[batch_id] = batch;
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_dispatch)
                    .bind(batch_id, ids, (uint32_t)chunks.size(), multimeshes, pages));

    stats.batches++;
    stats.dispatch_cpu.add(now_usec() - t0);
    VLOG(3, "GPUDirectMesher: dispatched batch ", batch_id, " (", (int64_t)chunks.size(), " chunks)");
}

bool GPUDirectMesher::get_results(VoxelState &output) {
    mutex->lock();
    std::vector<CompletedBatch> done;
    done.swap(completed);
    mutex->unlock();

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
        std::vector<bool> page_has_bounds(batch.pages.size(), false);

        for (size_t i = 0; i < batch.chunks.size(); i++) {
            const Vector3i &chunk_id = batch.chunks[i];
            const uint32_t count = dir[i * 4 + 0];
            const uint32_t base = dir[i * 4 + 2];
            const uint32_t flags = dir[i * 4 + 3];
            working_generation.erase(chunk_id);

            if (flags & 1u) { // overflow: try again in a later batch
                const int lod = ChunkMath::get_lod(chunk_id);
                if (lod >= 0 && lod < (int)chunk_queue.size()) {
                    chunk_queue[lod]->insert(chunk_id);
                }
                continue;
            }

            ChunkOutput out;
            out.chunk_id = chunk_id;
            out.triangle_count = count * 2;
            out.drawn_triangles = count * 2;
            if (count > 0) {
                const uint32_t k = base / settings.page_capacity;
                out.owner = this;
                out.page = batch.pages[k];
                out.quad_offset = base % settings.page_capacity;
                out.quad_count = count;
                page_state[out.page].live_quads += count;
                page_state[out.page].live_chunks++;
                stats.live_quads += count;
                const AABB box = _chunk_aabb(chunk_id);
                page_bounds[k] = page_has_bounds[k] ? page_bounds[k].merge(box) : box;
                page_has_bounds[k] = true;
            }
            stats.chunks_meshed++;
            output.add_new_loaded(out);
        }

        for (size_t k = 0; k < batch.pages.size(); k++) {
            const int page = batch.pages[k];
            if (page_state[page].live_chunks == 0) {
                pool->release_page(page);
                stats.pages_returned_unused++;
            } else {
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
    ps.live_chunks--;
    stats.live_quads -= p_chunk.quad_count;
    stats.kills++;
    if (ps.live_chunks <= 0) {
        // Last chunk in the page: hide and reset the whole page instead of clearing the range.
        ps = PageState();
        pool->release_page(p_chunk.page);
        return;
    }
    pending_kill_pages.push_back(p_chunk.page);
    pending_kill_multimeshes.push_back(pool->get_page_multimesh(p_chunk.page));
    pending_kill_ranges.push_back(p_chunk.quad_offset);
    pending_kill_ranges.push_back(p_chunk.quad_count);
}

void GPUDirectMesher::send_chunk(LODHashSet p_chunks) {
    for (LODHashSet *q : chunk_queue) {
        q->clear();
    }
    for (const Vector3i &chunk_pos : p_chunks) {
        const int lod = ChunkMath::get_lod(chunk_pos);
        if (lod < 0 || lod >= (int)chunk_queue.size()) {
            continue;
        }
        if (!working_generation.has(chunk_pos)) {
            chunk_queue[lod]->insert(chunk_pos);
        }
    }
}

bool GPUDirectMesher::is_idle() {
    if (!in_flight.is_empty()) {
        return false;
    }
    for (LODHashSet *q : chunk_queue) {
        if (!q->is_empty()) {
            return false;
        }
    }
    return true;
}

void GPUDirectMesher::set_debug_view(int p_view) {
    pool->set_debug_view(p_view);
}

Dictionary GPUDirectMesher::get_stats() {
    Dictionary d;
    int64_t queued = 0;
    for (LODHashSet *q : chunk_queue) {
        queued += q->size();
    }
    int pages_with_live = 0;
    for (const PageState &ps : page_state) {
        pages_with_live += ps.live_chunks > 0 ? 1 : 0;
    }
    d["queued"] = queued;
    d["in_flight"] = (int64_t)in_flight.size();
    d["batches"] = (int64_t)stats.batches;
    d["chunks_meshed"] = (int64_t)stats.chunks_meshed;
    d["quads_emitted"] = (int64_t)stats.quads_emitted;
    d["overflow_chunks"] = (int64_t)stats.overflow_chunks;
    d["missing_neighbor"] = (int64_t)stats.missing_neighbor;
    d["nan_vertices"] = (int64_t)stats.nan_vertices;
    d["kills"] = (int64_t)stats.kills;
    d["pages_returned_unused"] = (int64_t)stats.pages_returned_unused;
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
    for (LODHashSet *q : chunk_queue) {
        memdelete(q);
    }
    chunk_queue.clear();
    // Queues its own render-thread cleanup. `pool` is not cleared: render-thread work queued before
    // (dispatch, kills) still uses it and runs first (FIFO).
    pool->destroy();
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &GPUDirectMesher::_rt_destroy));
}

void GPUDirectMesher::_rt_destroy() {
    // Dependents (uniform set, pipelines) before the shaders and resources they use.
    RID rids[] = { uniform_set,
        pipelines[0], pipelines[1], pipelines[2], pipelines[3], pipelines[4],
        shaders[0], shaders[1], shaders[2], shaders[3], shaders[4],
        buffer_chunk_dir, buffer_input_chunk, buffer_sdf, buffer_edges, buffer_vertices,
        buffer_chunk_param, staging_texture, buffer_status };
    if (rd) {
        for (const RID &rid : rids) {
            if (rid.is_valid()) {
                rd->free_rid(rid);
            }
        }
    }
    memdelete(this);
}
