/*
 * TODO (Optimization): Replace slow uniform integer division operations.
 * Currently, compute shaders use integer division and modulo (`/` and `%`)
 * with a uniform variable (`chunk_param.chunk_size`) to calculate batch offsets
 * and flatten 3D indices (e.g., `gl_GlobalInvocationID.y / chunk_param.chunk_size`).
 * GPU hardware lacks fast integer division, making uniform division quite slow.
 *
 * Once the meshing logic works perfectly, implement one of these fixes:
 * 1. Compile-Time Injection (Best): Since shaders are compiled dynamically in
 *    `compile_dynamic_shader()`, inject the chunk size into the shader string as
 *    a `#define` macro. The GLSL compiler will then automatically optimize the
 *    math into 1-cycle instructions.
 * 2. Power-of-Two Bitshifts: If `chunk_size` is guaranteed to be a power of two,
 *    pass shift and mask values as uniforms and replace divisions with bitwise
 *    shifts (`>>`) and modulos with bitwise ANDs (`&`).
 */

#include "GPU_mesher.hpp"

#include "cascade_passes.hpp"
#include "exact_noise.hpp"
#include "exact_noise_glsl.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"
#include "gpu_shader.hpp"
#include "legacy_mesher_glsl.hpp"
#include "material_library.hpp"
#include "page_format.hpp"

#include "ov/math.hpp"
#include "ov/on_error.hpp"

#include <cmath>
#include <cstring>

using ov::gpu::BufferKind;

ov::gpu::Pipeline GPUMesher::compile_dynamic_shader(const char *algorithm_pass, const ov::str &p_name) {
    // The shared legacy shader text (src/legacy_mesher_glsl.hpp).
    const SDFGpuTable *table = sdf->get_gpu_table();
    const bool dc = dual_contouring && p_name == "third";
    const ov::str code = LegacyMesherGLSL::fill_shader(algorithm_pass, table->glsl_buffers(6, 7), table->glsl_set_chunk(),
            ChunkPalette::glsl_base(table->base_palette()), CascadeGLSL::glsl_material_buffer(MATERIAL_BINDING, "readonly"), dc);
    return ov::gpu::compile(*device, "legacy_" + p_name, code);
}

void GPUMesher::initialize(
        SDFBase *p_sdf,
        ChunkParam p_chunk_param,
        uint32_t p_chunk_batch_number,
        int lod_numbers,
        MeshHost *p_host) {
    sdf = p_sdf;
    gate.bind(p_sdf, this);
    chunk_param = p_chunk_param;
    chunk_batch_number = p_chunk_batch_number;
    host = p_host;

    // Worker stuff

    worker = ov::executor().start_worker("legacy_mesher", *this);

    lod_count = lod_numbers;
    chunk_queue.resize(ChunkMath::load_queue_count(lod_numbers), nullptr);

    for (LODHashSet *&q : chunk_queue) {
        q = new LODHashSet();
    }
    remesh_queue.resize(lod_numbers, nullptr);
    for (LODHashSet *&q : remesh_queue) {
        q = new LODHashSet();
    }
};

void GPUMesher::_initialize_gpu_compute() {
    //alocate memory

    device = ov::gpu::create_local_device();
    OV_ERR_FAIL_COND_MSG(device == nullptr, "GPUMesher: no local GPU device (headless, or the Compatibility renderer): nothing is meshed.");
    const uint64_t cells = uint64_t(chunk_param.chunk_size) * chunk_param.chunk_size * chunk_param.chunk_size;

    buffer_input_chunk_id = device->create_buffer(BufferKind::STORAGE, uint64_t(chunk_batch_number) * GPUFrame::CHUNK_INPUT_BYTES);

    buffer_atomic_counter = device->create_buffer(BufferKind::STORAGE, 2 * (1 + uint64_t(chunk_batch_number)) * sizeof(uint32_t));

    buffer_sdf_sample = device->create_buffer(BufferKind::STORAGE, uint64_t(chunk_batch_number) * cells * 16);

    buffer_edge_intersect = device->create_buffer(BufferKind::STORAGE, cells * sizeof(float) * 3 * chunk_batch_number);

    // One PointMetadata per cell, 4 x uint32: position (3 x uint8), normal (snorm 3 x 8), chunk index
    // | quad mask << 16 | palette index << POINT_PALETTE_SHIFT, cell index.
    buffer_output_point = device->create_buffer(BufferKind::STORAGE, 4 * cells * sizeof(uint32_t) * chunk_batch_number);
    // One palette index per sample, like sdf_sample (the cascade's level 0 writes it).
    buffer_material = device->create_buffer(BufferKind::STORAGE, uint64_t(chunk_batch_number) * cells * sizeof(uint32_t));

    // Injection of Chunk Param in the uniform
    // std140 ChunkParam, laid out by GPUFrame.
    const ov::bytes param_data = GPUFrame::chunk_param_bytes(chunk_param);
    buffer_chunk_param = device->create_buffer(BufferKind::UNIFORM, param_data.size(), param_data.data());

    // The table, sized for the capacity up front; the records arrive with the first batch (process_queue).
    const SDFGpuTable *table = sdf->get_gpu_table();
    buffer_table_records = device->create_buffer(BufferKind::STORAGE, uint64_t(ov::max<uint32_t>(table->instance_capacity(), 1u)) * table->record_size());
    table_list_capacity = uint64_t(chunk_batch_number) * table->max_chunk_list() * 4; // no list is longer (build_lists)
    buffer_table_list = device->create_buffer(BufferKind::STORAGE, table_list_capacity);

    pipeline_second_pass = compile_dynamic_shader(LegacyMesherGLSL::SECOND_PASS, "second");
    pipeline_third_pass = compile_dynamic_shader(LegacyMesherGLSL::THIRD_PASS, "third");

    // Both passes declare the same bindings (GLSL binding points 0 to 8): one set serves them.
    uniform_set = device->create_bind_set(pipeline_second_pass,
            { { 0, BufferKind::STORAGE, buffer_atomic_counter }, { 1, BufferKind::STORAGE, buffer_input_chunk_id },
                    { 2, BufferKind::STORAGE, buffer_sdf_sample }, { 3, BufferKind::STORAGE, buffer_edge_intersect },
                    { 4, BufferKind::STORAGE, buffer_output_point }, { 5, BufferKind::UNIFORM, buffer_chunk_param },
                    { 6, BufferKind::STORAGE, buffer_table_records }, { 7, BufferKind::STORAGE, buffer_table_list },
                    { uint32_t(MATERIAL_BINDING), BufferKind::STORAGE, buffer_material } });

    // Pass 1 (the samples) is the cascade passes.
    cascade = new CascadePasses();
    if (!cascade->create(device.get(), sdf, chunk_param, ov::min(chunk_batch_number, CASCADE_MAX_CHUNKS), buffer_input_chunk_id, buffer_chunk_param,
                buffer_sdf_sample, ov::gpu::Buffer(), nullptr, buffer_table_records, buffer_table_list, buffer_material)) {
        OV_ERR_PRINT("GPUMesher: the cascade passes failed to compile: no pass runs (empty meshes).");
    }
    VLOG(1, "GPUMesher: cascade scratch ", int64_t(cascade->scratch_bytes() / 1048576), " MB (", int64_t(cascade->get_max_chunks()), " chunks per sub-batch)");
}

void GPUMesher::enter() {
    _initialize_gpu_compute();
    if (device == nullptr) {
        return; // never available: no batch is handed over
    }
    // Only now may process_queue build a batch: it sizes the lists by table_list_capacity, set above (before, a batch's
    // lists were cut to 0 words).
    mutex.lock();
    worker_available = true;
    mutex.unlock();
    VLOG(1, "GPUMesher : Worker started");
}

// The batch process_queue handed over (worker_available false), if any.
bool GPUMesher::step() {
    mutex.lock();
    if (worker_available) {
        mutex.unlock();
        return false;
    }
    const ov::bytes local_chunk_id = input_chunk_id;
    const ov::bytes local_table_list = input_table_list;
    const ov::bytes local_table_records = input_table_records;
    const uint32_t local_count = batch_count;
    const ov::ivec2 local_lods = batch_lods;
    mutex.unlock();

    const uint64_t t_batch_start = now_usec();
    _execute_gpu_pipeline(local_chunk_id, local_count, local_lods, local_table_list, local_table_records);

    mutex.lock();
    stats.batch_total.add(now_usec() - t_batch_start);
    worker_available = true;
    result_available = true;
    mutex.unlock();
    return false;
}

void GPUMesher::exit() {
    // The local device must be freed on the thread that created it.
    if (cascade != nullptr) {
        // Its bind sets use the input, param and sdf buffers freed below.
        cascade->free();
        delete cascade;
        cascade = nullptr;
    }
    if (device == nullptr) {
        return;
    }
    const ov::gpu::Handle resources[] = { uniform_set, pipeline_second_pass, pipeline_third_pass, buffer_input_chunk_id, buffer_atomic_counter,
        buffer_sdf_sample, buffer_edge_intersect, buffer_output_point, buffer_chunk_param, buffer_table_records, buffer_table_list, buffer_material };
    for (const ov::gpu::Handle h : resources) {
        if (h.is_valid()) {
            device->free(h);
        }
    }
    device.reset();
    VLOG(1, "GPUMesher : Worker stopped");
}

void GPUMesher::stop() {
    if (!worker) {
        return;
    }
    worker->stop();
    worker.reset();

    // Free the meshes of results that were produced but never collected.
    for (ChunkOutput &out : out_queue) {
        out.free();
    }
    out_queue.clear();

    for (LODHashSet *q : chunk_queue) {
        delete q;
    }
    chunk_queue.clear();
    lod_count = 0;
    for (LODHashSet *q : remesh_queue) {
        delete q;
    }
    remesh_queue.clear();
}

void GPUMesher::release_chunk(const ChunkOutput &p_chunk) {
    mutex.lock();
    const uint64_t *found = chunk_meshes.getptr(p_chunk.handle);
    const uint64_t freed = found ? *found : 0;
    chunk_meshes.erase(p_chunk.handle);
    mutex.unlock();
    OV_ERR_FAIL_NULL_MSG(found, "GPUMesher: a chunk released twice (its handle is unknown).");
    if (freed != 0) {
        host->free_mesh(freed);
        DebugLog::chunk_rids_freed++;
    }
}

uint64_t GPUMesher::mesh_of(const ChunkOutput &p_chunk) {
    mutex.lock();
    const uint64_t *found = chunk_meshes.getptr(p_chunk.handle);
    const uint64_t mesh = found ? *found : 0;
    mutex.unlock();
    return mesh;
}

bool GPUMesher::is_idle() {
    mutex.lock();
    const bool idle = worker_available && !result_available;
    mutex.unlock();
    if (!idle || !gate.idle(ChunkGate::CHUNK) || !gate.idle(ChunkGate::REMESH)) {
        return false;
    }
    for (const LODHashSet *q : remesh_queue) {
        if (!q->is_empty()) {
            return false;
        }
    }
    return true;
}

ov::Value GPUMesher::get_stats() {
    ov::Value d = ov::Value::object();
    mutex.lock();
    d["batch_total"] = stats.batch_total.to_value();
    d["gpu_submit_sync"] = stats.gpu_submit_sync.to_value();
    d["points_fetch"] = stats.points_fetch.to_value();
    d["points_process"] = stats.points_process.to_value();
    d["triangulate"] = stats.triangulate.to_value();
    d["render_upload"] = stats.render_upload.to_value();
    d["chunks_meshed"] = (int64_t)stats.chunks_meshed;
    d["points"] = (int64_t)stats.points;
    d["triangles"] = (int64_t)stats.triangles;
    d["queued"] = (int64_t)[this]() {
        int64_t n = 0;
        for (LODHashSet *q : chunk_queue) {
            n += q->size();
        }
        return n;
    }
    ();
    d["remesh_queue"] = (int64_t)[this]() {
        int64_t n = 0;
        for (LODHashSet *q : remesh_queue) {
            n += q->size();
        }
        return n;
    }
    ();
    d["in_flight"] = (int64_t)(worker_available ? 0 : batch_count);
    mutex.unlock();
    return d;
}

ov::ivec3 GPUMesher::_unflatten_index_z(uint32_t flat_idx) {
    uint32_t x = flat_idx % chunk_param.chunk_size;
    uint32_t y = (flat_idx / chunk_param.chunk_size) % chunk_param.chunk_size;
    uint32_t z = flat_idx / (chunk_param.chunk_size * chunk_param.chunk_size);
    return ov::ivec3(int32_t(x), int32_t(y), int32_t(z));
}

uint32_t GPUMesher::_get_flat_index_z(int p_x, int p_y, int p_z) {
    return uint32_t(p_x + p_y * chunk_param.chunk_size + p_z * chunk_param.chunk_size * chunk_param.chunk_size);
}

void GPUMesher::_execute_gpu_pipeline(const ov::bytes &p_chunk_id, uint32_t p_count, const ov::ivec2 &p_lods, const ov::bytes &p_table_list,
        const ov::bytes &p_table_records) {
    // The instance table (archetype instances): the records the main thread captured with this batch's
    // lists (same snapshot) when they changed, then the lists.
    if (!p_table_records.empty()) {
        SDFGpuTable::upload_records(*device, buffer_table_records, p_table_records);
    }
    if (!p_table_list.empty()) {
        OV_ERR_FAIL_COND_MSG(uint64_t(p_table_list.size()) > table_list_capacity, "GPUMesher: batch instance list over its buffer.");
        device->update_buffer(buffer_table_list, 0, p_table_list.size(), p_table_list.data());
    }
    const uint64_t cells = uint64_t(chunk_param.chunk_size) * chunk_param.chunk_size * chunk_param.chunk_size;
    // 1. RESET ATOMIC COUNTERS TO 0
    device->clear_buffer(buffer_atomic_counter, 0, 2 * (1 + uint64_t(chunk_batch_number)) * sizeof(uint32_t));

    // Erase the active edges buffer to 0.0
    // (So passes read eX_t > 0.0 correctly)
    device->clear_buffer(buffer_edge_intersect, 0, cells * sizeof(float) * 3 * chunk_batch_number);

    // 2. put the input inside
    device->update_buffer(buffer_input_chunk_id, 0, p_chunk_id.size(), p_chunk_id.data());

    // Pass 1 is the cascade's levels for the batch's real chunks (their own dispatches, before the passes below). Slots past the batch keep stale samples: they are computed but never used (see
    // render_chunk). A cascade that failed to build runs nothing.
    const bool run_passes = cascade->is_valid();
    if (run_passes && p_count > 0) {
        cascade->record_all(p_lods, p_count, !p_table_list.empty()); // no instance listed: the bare cascade
    }

    // --- GPU SUBMISSION PIPELINE ---
    uint32_t dispatch_size = chunk_param.chunk_size / 8;

    if (run_passes) {
        // ---- PASS 2: Calculate Edge Intersections ----
        device->dispatch(pipeline_second_pass, uniform_set, nullptr, 0, dispatch_size, dispatch_size * 3, dispatch_size * chunk_batch_number);

        // ---- PASS 3 (after pass 2's edge_intersect): Generate Vertex Centroids, Normals & Metadata ----
        device->dispatch(pipeline_third_pass, uniform_set, nullptr, 0, dispatch_size, dispatch_size, dispatch_size * chunk_batch_number);
    }

    const uint64_t submit_start = now_usec();
    device->sync();
    const uint64_t t_submit_sync = now_usec() - submit_start;

    uint64_t t_points_fetch = 0;
    uint64_t t_points_process = 0;
    uint64_t t_triangulate = 0;

    // 1. Unpack Atomic Counters to get total point count
    const uint64_t t_counters_fetch_start = now_usec();
    const ov::bytes counter_data = device->read_buffer(buffer_atomic_counter, 0, 2 * (1 + uint64_t(chunk_batch_number)) * sizeof(uint32_t));
    t_points_fetch += now_usec() - t_counters_fetch_start;
    auto counter = [&counter_data](size_t p_i) { return (p_i + 1) * 4 <= counter_data.size() ? ov::get_u32(counter_data.data() + p_i * 4) : 0u; };

    uint32_t total_points = counter(0);
    uint32_t total_triangles = counter(1);

    if (total_points > 0) {
        // 2. Unpack points
        const uint64_t t_points_fetch_start = now_usec();
        const ov::bytes point_data = device->read_buffer(buffer_output_point, 0, uint64_t(total_points) * 4 * sizeof(uint32_t));
        t_points_fetch += now_usec() - t_points_fetch_start;
        OV_ERR_FAIL_COND_MSG(point_data.size() < size_t(total_points) * 16, "GPUMesher: the points readback came back short.");
        const uint8_t *points = point_data.data();

        for (uint32_t i = 0; i < chunk_batch_number; i++) {
            uint32_t local_point_count = counter(2 + 2 * i);
            uint32_t local_triangle_count = counter(3 + 2 * i);

            // 1 triangle = 3 indices
            unpacked_chunks[i].resize(local_point_count, local_triangle_count * 3);
        }

        // 3. Process every point
        const uint64_t t_points_process_start = now_usec();
        for (uint32_t i = 0; i < total_points; i++) {
            // Struct layout: int a, int b, int c, int d;
            const size_t offset = size_t(i) * 16;
            uint32_t a = ov::get_u32(points + offset + 0);
            uint32_t b = ov::get_u32(points + offset + 4);
            uint32_t c = ov::get_u32(points + offset + 8);
            uint32_t d = ov::get_u32(points + offset + 12);

            // --- Unpack Flat Index (d) ---
            uint32_t flat_idx = d;
            const ov::ivec3 cell_coord = _unflatten_index_z(flat_idx);

            // --- Unpack Position (a) ---
            Float3 local_vertex;
            local_vertex.x = float(a & 0xFF) / 255.0f + cell_coord.x;
            local_vertex.y = float((a >> 8) & 0xFF) / 255.0f + cell_coord.y;
            local_vertex.z = float((a >> 16) & 0xFF) / 255.0f + cell_coord.z;

            // --- Unpack Normal (b) ---
            // packSnorm4x8 mapping: signed normalized -1.0 to 1.0 mapped to 0 to 255
            Float3 surface_normal;
            surface_normal.x = ov::clamp(float((int8_t)(b & 0xFF)) / 127.0f, -1.0f, 1.0f);
            surface_normal.y = ov::clamp(float((int8_t)((b >> 8) & 0xFF)) / 127.0f, -1.0f, 1.0f);
            surface_normal.z = ov::clamp(float((int8_t)((b >> 16) & 0xFF)) / 127.0f, -1.0f, 1.0f);

            float length = std::sqrt(surface_normal.x * surface_normal.x +
                    surface_normal.y * surface_normal.y +
                    surface_normal.z * surface_normal.z);
            if (length > 0.0001f) {
                surface_normal.x /= length;
                surface_normal.y /= length;
                surface_normal.z /= length;
            }

            // --- Unpack Metadata (c) ---
            uint32_t chunk_idx = c & 0xFFFF; // Changed to uint32_t for safe array indexing
            uint8_t quad_mask = (c >> 16) & 0xF;
            const uint32_t palette_index = (c >> POINT_PALETTE_SHIFT) & (PageFormat::PALETTE_SIZE - 1);
            const std::vector<uint32_t> &palette = unpacked_chunks[chunk_idx].palette;
            const uint32_t material = palette_index < palette.size() ? palette[palette_index] : Material::GROUND;

            // Fill the unpack chunk points and normals
            uint32_t pt_idx = unpacked_chunks[chunk_idx].point_offset++;
            unpacked_chunks[chunk_idx].mesh.points[pt_idx] = local_vertex;
            unpacked_chunks[chunk_idx].mesh.normals[pt_idx] = surface_normal;
            unpacked_chunks[chunk_idx].mesh.materials[pt_idx] = float(material);
            unpacked_chunks[chunk_idx].quad_masks[pt_idx] = quad_mask;
            unpacked_chunks[chunk_idx].point_cells[pt_idx] = int32_t(flat_idx);
            unpacked_chunks[chunk_idx].index_storage[flat_idx] = pt_idx;
        }
        t_points_process = now_usec() - t_points_process_start;

        // 4. Generate Triangles based on dual contouring quad masks
        const uint64_t t_tri_start = now_usec();
        for (uint32_t chunk_idx = 0; chunk_idx < chunk_batch_number; ++chunk_idx) {
            uint32_t tri_offset = 0;

            std::vector<int32_t> &tris = unpacked_chunks[chunk_idx].mesh.indices;
            auto put = [&](uint32_t p_v) {
                if (tri_offset < tris.size()) {
                    tris[tri_offset] = int32_t(p_v);
                } else {
                    OV_ERR_PRINT_ONCE("GPUMesher: more indices than the GPU counted: dropped.");
                }
                tri_offset++;
            };
            auto emit_quad = [&](uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3, bool positive_winding) {
                if (positive_winding) {
                    put(p0);
                    put(p1);
                    put(p3);

                    put(p0);
                    put(p3);
                    put(p2);
                } else {
                    put(p0);
                    put(p3);
                    put(p1);

                    put(p0);
                    put(p2);
                    put(p3);
                }
            };

            for (uint32_t pt_idx = 0; pt_idx < unpacked_chunks[chunk_idx].point_offset; ++pt_idx) {
                uint8_t quad_mask = unpacked_chunks[chunk_idx].quad_masks[pt_idx];

                // Mask out the sign bit to check if any geometry exists (mask & 7 checks the first 3 bits)
                if ((quad_mask & 7) == 0)
                    continue;

                // True if the origin cell is solid, false if it is empty air
                bool origin_solid = (quad_mask & 8) != 0;

                const ov::ivec3 cell_coord = _unflatten_index_z(uint32_t(unpacked_chunks[chunk_idx].point_cells[pt_idx]));
                ov::HashMap<uint32_t, uint32_t, ov::DefaultHash> &index_storage = unpacked_chunks[chunk_idx].index_storage;
                // The point of the cell at cell_coord - (p_dx, p_dy, p_dz) (0 if none: godot's operator[] added it).
                auto at = [&](int p_dx, int p_dy, int p_dz) { return index_storage[_get_flat_index_z(cell_coord.x - p_dx, cell_coord.y - p_dy, cell_coord.z - p_dz)]; };

                // x-axis edge quad
                if ((quad_mask & 1) && cell_coord.y > 0 && cell_coord.z > 0) {
                    uint32_t p0 = pt_idx; // Current cell
                    uint32_t p1 = at(0, 1, 0);
                    uint32_t p2 = at(0, 0, 1);
                    uint32_t p3 = at(0, 1, 1);

                    emit_quad(p0, p2, p1, p3, origin_solid); // axis 0 = X
                }

                // y-axis edge quad
                if ((quad_mask & 2) && cell_coord.x > 0 && cell_coord.z > 0) {
                    uint32_t p0 = pt_idx;
                    uint32_t p1 = at(1, 0, 0);
                    uint32_t p2 = at(0, 0, 1);
                    uint32_t p3 = at(1, 0, 1);

                    emit_quad(p0, p1, p2, p3, origin_solid); // axis 1 = Y
                }

                // z-axis edge quad
                if ((quad_mask & 4) && cell_coord.x > 0 && cell_coord.y > 0) {
                    uint32_t p0 = pt_idx;
                    uint32_t p1 = at(1, 0, 0);
                    uint32_t p2 = at(0, 1, 0);
                    uint32_t p3 = at(1, 1, 0);

                    emit_quad(p0, p2, p1, p3, origin_solid); // axis 2 = Z
                }
            }
            unpacked_chunks[chunk_idx].index_count = tri_offset;
        }
        t_triangulate = now_usec() - t_tri_start;
    }

    const uint64_t render_start = now_usec();
    render_chunk();
    const uint64_t t_render = now_usec() - render_start;

    VLOG(2, "GPUMesher batch: chunks=", batch_count, " points=", total_points, " gpu_tri_counter=", total_triangles,
            " submit_sync=", t_submit_sync / 1000.0, "ms fetch=", t_points_fetch / 1000.0,
            "ms process=", t_points_process / 1000.0, "ms triangulate=", t_triangulate / 1000.0,
            "ms upload=", t_render / 1000.0, "ms");

    uint64_t emitted_triangles = 0;
    mutex.lock();
    // Only the first batch_count slots hold real chunks; the rest of the batch is unused.
    for (uint32_t chunk_idx = 0; chunk_idx < batch_count; ++chunk_idx) {
        UnpackChunk &chunk = unpacked_chunks[chunk_idx];

        ChunkOutput out;
        out.chunk_id = chunk.chunk_id;
        out.triangle_count = chunk.index_count / 3;
        out.drawn_triangles = chunk.host_mesh != 0 ? uint32_t(chunk.mesh.indices.size() / 3) : 0;
        if (chunk.host_mesh != 0) {
            out.owner = this;
            out.handle = ++next_handle;
            chunk_meshes[out.handle] = chunk.host_mesh;
        }
        emitted_triangles += out.triangle_count;

        out_queue.push_back(out);
    }
    stats.gpu_submit_sync.add(t_submit_sync);
    stats.points_fetch.add(t_points_fetch);
    stats.points_process.add(t_points_process);
    stats.triangulate.add(t_triangulate);
    stats.render_upload.add(t_render);
    stats.chunks_meshed += batch_count;
    stats.points += total_points;
    stats.triangles += emitted_triangles;
    mutex.unlock();
}

void GPUMesher::render_chunk() {
    const std::lock_guard<std::mutex> lock(origin_mutex);
    // Slots past batch_count are unused but still computed (zeroed chunk id): never build meshes for them.
    for (uint32_t chunk_idx = 0; chunk_idx < batch_count; ++chunk_idx) {
        UnpackChunk &chunk = unpacked_chunks[chunk_idx];

        if (chunk.mesh.points.empty() || chunk.mesh.indices.empty()) {
            continue;
        }

        if (trim_indices) {
            chunk.mesh.indices.resize(chunk.index_count);
            if (chunk.mesh.indices.empty()) {
                continue;
            }
        }

        // Mesh vertices are in the chunk's vertex grid: scale by the cell size, origin at grid (0,0,0).
        const float cell_size = chunk_param.voxel_size * float(ChunkMath::get_lod_size(chunk.chunk_id));
        chunk.host_mesh = host->create_mesh(chunk.mesh, cell_size, ChunkMath::grid_to_render(chunk.chunk_id, Float3{}, chunk_param));
        DebugLog::chunk_rids_created += chunk.host_mesh != 0 ? 1 : 0;
    }
}

void GPUMesher::set_render_origin(const ChunkId &p_origin, const Float3 &p_shift) {
    const std::lock_guard<std::mutex> lock(origin_mutex);
    chunk_param.origin_chunk = p_origin;
    host->shift_meshes(p_shift);
}

bool GPUMesher::_is_available() {
    bool available;

    mutex.lock();
    available = worker_available;
    mutex.unlock();

    return available;
}

bool GPUMesher::get_results(VoxelState &output) {
    mutex.lock();
    const bool is_result_available = result_available;

    if (is_result_available) {
        for (ChunkOutput &out : out_queue) {
            output.add_new_loaded(out);
            working_generation.erase(out.chunk_id);
        }

        out_queue.clear();
        result_available = false;
    }
    mutex.unlock();

    return is_result_available;
}

void GPUMesher::process_queue() {
    // The parked chunks whose records came, back in their queues (every frame, the worker busy or not).
    gate.take_ready([this](const ChunkId &p_chunk, ChunkGate::Kind p_kind) { _requeue(p_chunk, p_kind == ChunkGate::REMESH); });
    if (!_is_available())
        return;

    ov::bytes new_batch(size_t(chunk_batch_number) * GPUFrame::CHUNK_INPUT_BYTES, 0);

    unpacked_chunks.clear();
    unpacked_chunks.resize(chunk_batch_number);

    uint32_t count = 0;
    ChunkId new_chunk;

    ChunkGate::Kind kind = ChunkGate::CHUNK;
    const auto pop = [this](ChunkId &r_chunk, ChunkGate::Kind &r_kind, int &r_queue) { return _pop_queue(r_chunk, r_kind, r_queue); };
    const auto requeue = [this](const ChunkId &p_chunk, ChunkGate::Kind p_kind) { _requeue(p_chunk, p_kind == ChunkGate::REMESH); };
    gate.begin_frame(0); // one batch at a time: the batch size bounds the admits
    while (count < chunk_batch_number && gate.next(new_chunk, kind, pop, requeue)) {
        working_generation.insert(new_chunk);

        GPUFrame::encode_chunk_input(new_batch.data() + size_t(count) * GPUFrame::CHUNK_INPUT_BYTES, new_chunk, chunk_param);

        unpacked_chunks[count].chunk_id = new_chunk;

        count++;
    }

    if (count > 0) {
        ov::bytes table_list;
        ov::bytes table_records;
        in_flight_ids.clear();
        for (uint32_t i = 0; i < count; i++) {
            in_flight_ids.push_back(unpacked_chunks[i].chunk_id);
        }
        SDFGpuTable *table = sdf->get_gpu_table();
        std::vector<std::vector<uint32_t>> palettes;
        const uint64_t t_lists = now_usec();
        const uint32_t kept = table->build_lists(in_flight_ids, chunk_param, new_batch, table_list, uint32_t(table_list_capacity / 4), &palettes);
        dispatch_times.build_lists += now_usec() - t_lists;
        // The chunks whose lists didn't fit the buffer go first in a later batch (remesh_queue holds first meshes too,
        // remesh_chunks puts the batch in flight there).
        for (uint32_t i = kept; i < count; i++) {
            const ChunkId &chunk = in_flight_ids[i];
            working_generation.erase(chunk);
            remesh_queue[ChunkMath::get_lod(chunk)]->insert(chunk);
            unpacked_chunks[i] = UnpackChunk();
            std::memset(new_batch.data() + size_t(i) * GPUFrame::CHUNK_INPUT_BYTES, 0, GPUFrame::CHUNK_INPUT_BYTES); // an unused slot
        }
        in_flight_ids.resize(kept);
        count = kept;
        for (uint32_t i = 0; i < count; i++) {
            unpacked_chunks[i].palette = std::move(palettes[i]);
        }
        if (table->upload_cursor() != sent_records) {
            const uint64_t t_records = now_usec();
            table_records = table->record_bytes(sent_records, sent_records);
            dispatch_times.record_bytes += now_usec() - t_records;
        }
        mutex.lock();
        input_chunk_id = new_batch;
        input_table_list = table_list;
        input_table_records = table_records;
        batch_count = count;
        batch_lods = ChunkMath::lod_range(in_flight_ids);
        worker_available = false;
        mutex.unlock();

        worker->wake();
    }
}

bool GPUMesher::_pop_queue(ChunkId &new_chunk, ChunkGate::Kind &r_kind, int &r_queue) {
    // Instance edits first, finest LOD first (as GPUDirectMesher::_pop_queue).
    for (const std::vector<LODHashSet *> *queues : { &remesh_queue, &chunk_queue }) {
        for (size_t q = 0; q < queues->size(); q++) {
            LODHashSet *queue = (*queues)[q];
            if (queue == nullptr || queue->is_empty()) {
                continue;
            }
            LODHashSet::Iterator it = queue->begin();
            new_chunk = *it;
            queue->remove(it);
            r_kind = queues == &remesh_queue ? ChunkGate::REMESH : ChunkGate::CHUNK;
            r_queue = int(q);
            return true;
        }
    }
    return false;
}

void GPUMesher::_requeue(const ChunkId &p_chunk, bool p_remesh) {
    const int lod = ChunkMath::get_lod(p_chunk);
    if (lod < 0 || lod >= lod_count) {
        return;
    }
    LODHashSet *queue = p_remesh ? remesh_queue[lod] : chunk_queue[ChunkMath::load_queue(p_chunk, lod_count, load_view)];
    if (queue != nullptr) {
        queue->insert(p_chunk);
    }
}

void GPUMesher::remesh_chunks(const std::vector<ChunkId> &p_chunks) {
    std::vector<ChunkId> chunks = p_chunks;
    // The batch in flight was built from the old table: its chunks the edit reaches are meshed again after
    // it lands (a first mesh isn't in loaded_chunk, so the terrain doesn't list it).
    // (remesh_queue is a set: a chunk already loaded and listed is queued once.)
    for (const ChunkId &chunk : in_flight_ids) {
        if (sdf->edit_affects_chunk(chunk)) {
            chunks.push_back(chunk);
        }
    }
    for (const ChunkId &chunk : chunks) {
        const int lod = ChunkMath::get_lod(chunk);
        if (lod >= 0 && lod < (int)remesh_queue.size() && remesh_queue[lod] != nullptr) {
            remesh_queue[lod]->insert(chunk);
        }
    }
}

void GPUMesher::send_chunk(const LODHashSet &p_chunks, const ScanView &p_view) {
    load_view = p_view;
    gate.next_scan(p_chunks, [this, &p_view](const ChunkId &p_chunk) { return ChunkMath::load_queue(p_chunk, lod_count, p_view); });
    for (LODHashSet *q : chunk_queue) {
        q->clear();
    }

    for (const ChunkId &chunk_pos : p_chunks) {
        const int lod = ChunkMath::get_lod(chunk_pos);
        if (lod < 0 || lod >= lod_count) {
            continue;
        }
        if (!working_generation.has(chunk_pos) && !gate.keeps(chunk_pos)) {
            chunk_queue[ChunkMath::load_queue(chunk_pos, lod_count, p_view)]->insert(chunk_pos);
        }
    }
}
