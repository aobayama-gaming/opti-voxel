#include "GPU_direct_mesher.hpp"

#include "cascade_passes.hpp"
#include "exact_noise.hpp"
#include "exact_noise_glsl.hpp"
#include "gpu_direct_glsl.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"
#include "gpu_shader.hpp"
#include "page_format.hpp"
#include "trace.hpp"

#include "ov/math.hpp"
#include "ov/on_error.hpp"
#include "ov/str.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <utility>

static constexpr const char DIRECT_TEMPLATE[] = R"glsl(#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

/*GLSL_STRUCT_DEFINITION*/

/*GLSL_DIRECT_BINDINGS*/

/*GLSL_HELPER_FUNCTION*/

/*GLSL_VERTEX_GRADIENT*/

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

// The mesher's own compute passes (shaders[i] / pipelines[i]). Pass 0 of the stats and timestamps (`sdf`, `_p0`) is
// the samples, the cascade passes (CascadePasses, _rt_record_samples); compute pass i is stat / timestamp i + 1.
static const char *PASS_SOURCES[GPUDirectMesher::COMPUTE_PASSES] = {
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
    const ChunkId rel(ov::clamp(d.x, -bias, bias - 1), ov::clamp(d.y, -bias, bias - 1), ov::clamp(d.z, -bias, bias - 1));
    return spread(uint64_t(rel.x + bias)) | (spread(uint64_t(rel.y + bias)) << 1) | (spread(uint64_t(rel.z + bias)) << 2);
}

void GPUDirectMesher::initialize(SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers, ov::gpu::Device *p_device, PageHost *p_host,
        const Settings &p_settings) {
    sdf = p_sdf;
    gate.bind(p_sdf, this);
    chunk_param = p_chunk_param;
    device = p_device;
    host = p_host;
    settings = p_settings;
    settings.max_batches_per_frame = ov::clamp(settings.max_batches_per_frame, 1u, MAX_BATCHES_PER_FRAME);
    if (settings.max_in_flight == 0) {
        settings.max_in_flight = 3 * settings.max_batches_per_frame;
    }
    if (settings.spare_pages == 0) {
        settings.spare_pages = settings.max_in_flight * settings.pages_per_batch;
    }
    if (settings.max_admits_per_frame == 0) {
        settings.max_admits_per_frame = 4 * settings.batch_size;
    }

    lod_count = p_lod_numbers;
    chunk_queue.resize(ChunkMath::load_queue_count(p_lod_numbers));
    remesh_queue.resize(p_lod_numbers);
    cost_model.resize(p_lod_numbers);

    pool = new QuadPagePool();
    pool->initialize(device, p_host, settings.page_capacity, settings.initial_pages);
    p_host->set_decode(_page_decode());
    p_host->set_materials(sdf->get_materials().make_image());
    page_state.resize(pool->get_page_count());
    palettes.initialize(settings.palette_capacity, sdf->get_gpu_table()->base_palette());

    device->post([this]() { _rt_initialize(); });
}

ov::gpu::Pipeline GPUDirectMesher::_rt_compile(const char *p_pass, const char *p_name) {
    const bool vertex_pass = (p_pass == GLSL_DIRECT_PASS_VERTICES);
    // Dual contouring: the vertex pass places its vertex with the shared QEF (gpu_glsl.hpp).
    const ov::str pass = dual_contouring && vertex_pass ? dual_contouring_pass(p_pass) : ov::str(p_pass);
    // The instance table adds the DefRecord/List buffers at 8/9, the exact int64 helpers (i64_sub, for the
    // table's noise lattice) and the ChunkInput global g_sdf_chunk.
    const SDFGpuTable *table = sdf->get_gpu_table();
    const ov::str bindings = ov::str(GLSL_DIRECT_BINDINGS) + table->glsl_buffers(8, 9);
    const ov::str helpers = ov::str(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + GLSL_EXACT_NOISE + GPUFrame::glsl_int64();
    const ov::str material_bindings = CascadeGLSL::glsl_material_buffer(MATERIAL_BINDING, "readonly") +
            ov::format("layout(set = 0, binding = %d, rgba16ui) uniform restrict writeonly uimage2D palette_image;\n"
                       "layout(set = 0, binding = %d, std430) restrict readonly buffer PaletteBatch { uint palette_batch[]; };\n"
                       "const uint PALETTE_GROUND_BLOCK = %du;\n",
                    PALETTE_IMAGE_BINDING, PALETTE_BATCH_BINDING, PaletteAlloc::GROUND_BLOCK) +
            ChunkPalette::glsl_base(table->base_palette());
    ov::str code = ov::replace_all(DIRECT_TEMPLATE,
            { { "/*GLSL_STRUCT_DEFINITION*/", ov::str(GLSL_STRUCT_DEFINITION) + GLSL_FRAME_STRUCTS },
                    { "/*GLSL_DIRECT_BINDINGS*/", bindings },
                    { "/*GLSL_PAGE_FORMAT*/", PageFormat::glsl_encode() },
                    { "/*GLSL_MATERIAL_BINDINGS*/", material_bindings },
                    { "/*GLSL_PALETTE_BATCH*/", PalettePool::glsl_batch() },
                    { "/*GLSL_HELPER_FUNCTION*/", helpers },
                    { "/*GLSL_SET_CHUNK*/", table->glsl_set_chunk() },
                    { "/*GLSL_QEF_VERTEX*/", ov::str(GLSL_VERTEX_NORMAL) + (dual_contouring ? GLSL_QEF_VERTEX : "") } });
    return ov::gpu::compile(*device, ov::str("direct_") + p_name, CascadeGLSL::fill_mesher_pass(code, pass));
}

void GPUDirectMesher::_rt_initialize() {
    using ov::gpu::Binding;
    using ov::gpu::BufferKind;
    const uint64_t cs = chunk_param.chunk_size;
    const uint64_t cells = cs * cs * cs;
    const uint64_t batch = settings.batch_size;

    buffer_chunk_dir = device->create_buffer(BufferKind::STORAGE, batch * 16);
    buffer_input_chunk = device->create_buffer(BufferKind::STORAGE, batch * GPUFrame::CHUNK_INPUT_BYTES);
    buffer_sdf = device->create_buffer(BufferKind::STORAGE, batch * cells * 16);
    buffer_edges = device->create_buffer(BufferKind::STORAGE, batch * cells * 3 * sizeof(float));
    buffer_vertices = device->create_buffer(BufferKind::STORAGE, batch * cells * 16);
    buffer_material = device->create_buffer(BufferKind::STORAGE, batch * cells * 4);
    buffer_palette_batch = device->create_buffer(BufferKind::STORAGE, PalettePool::batch_bytes(settings.batch_size));
    palettes.rt_create(device);
    host->rt_set_palette_texture(palettes.rt_texture());
    staging_texture = device->create_texture(QuadPagePool::TEX_FORMAT, QuadPagePool::TEX_WIDTH, settings.pages_per_batch * pool->get_page_rows(),
            ov::gpu::TEXTURE_STORAGE | ov::gpu::TEXTURE_COPY_SRC | ov::gpu::TEXTURE_COPY_DST);
    status_size_bytes = ((STATUS_HEADER_UINTS + settings.pages_per_batch) * 4 + 15) / 16 * 16;
    buffer_status = device->create_buffer(BufferKind::STORAGE, status_size_bytes);

    const ov::bytes param_data = GPUFrame::chunk_param_bytes(chunk_param);
    buffer_chunk_param = device->create_buffer(BufferKind::UNIFORM, param_data.size(), param_data.data());

    for (int i = 0; i < COMPUTE_PASSES; i++) {
        pipelines[i] = _rt_compile(PASS_SOURCES[i], PASS_NAMES[i + 1]);
        OV_ERR_FAIL_COND_MSG(!pipelines[i].is_valid(), ov::str("GPUDirectMesher: pass failed to compile: ") + PASS_NAMES[i + 1]);
    }

    // The table, sized for the capacity up front; the records arrive with the first dispatch (_table_payload).
    const SDFGpuTable *table = sdf->get_gpu_table();
    buffer_table_records = device->create_buffer(BufferKind::STORAGE, uint64_t(ov::max<uint32_t>(table->instance_capacity(), 1u)) * table->record_size());
    table_list_capacity = uint64_t(settings.batch_size) * table->max_chunk_list() * 4; // no list is longer (build_lists)
    buffer_table_list = device->create_buffer(BufferKind::STORAGE, table_list_capacity);
    // All passes declare the same bindings, so one set serves them all.
    uniform_set = device->create_bind_set(pipelines[0],
            { { 0, BufferKind::STORAGE, buffer_chunk_dir }, { 1, BufferKind::STORAGE, buffer_input_chunk }, { 2, BufferKind::STORAGE, buffer_sdf },
                    { 3, BufferKind::STORAGE, buffer_edges }, { 4, BufferKind::STORAGE, buffer_vertices }, { 5, BufferKind::UNIFORM, buffer_chunk_param },
                    Binding::storage_image(6, staging_texture), { 7, BufferKind::STORAGE, buffer_status }, { 8, BufferKind::STORAGE, buffer_table_records },
                    { 9, BufferKind::STORAGE, buffer_table_list }, { uint32_t(MATERIAL_BINDING), BufferKind::STORAGE, buffer_material },
                    Binding::storage_image(PALETTE_IMAGE_BINDING, palettes.rt_texture()),
                    { uint32_t(PALETTE_BATCH_BINDING), BufferKind::STORAGE, buffer_palette_batch } });

    cascade = new CascadePasses();
    if (!cascade->create(device, sdf, chunk_param, settings.batch_size, buffer_input_chunk, buffer_chunk_param, buffer_sdf, ov::gpu::Buffer(), nullptr,
                buffer_table_records, buffer_table_list, buffer_material)) {
        device->free(uniform_set);
        uniform_set = ov::gpu::BindSet(); // nothing dispatches
        OV_ERR_FAIL_MSG("GPUDirectMesher: the cascade passes failed to compile.");
    }
    const uint64_t cascade_mb = cascade->scratch_bytes() / 1048576;
    mutex.lock();
    cascade_scratch_bytes = cascade->scratch_bytes();
    mutex.unlock();

    VLOG(1, "GPUDirectMesher: initialized (batch ", settings.batch_size, ", ", settings.pages_per_batch,
            " pages x ", settings.page_capacity, " quads per batch, scratch ",
            (batch * cells * (16 + 12 + 16 + 4)) / 1048576, " MB, cascade scratch ", cascade_mb, " MB)");
}

void GPUDirectMesher::_rt_read_timestamps() {
    // Timestamps captured in an earlier frame are readable now; read each captured frame once, as
    // several batches may dispatch per frame. Our markers are "vx<batch>_<step>".
    const uint64_t captured_frame = device->timestamps_frame();
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
    // By batch id, in ascending order (the measured batches feed the cost model in it).
    std::map<int64_t, BatchTimes> batches;
    for (const ov::gpu::Timestamp &ts : device->timestamps()) {
        const ov::str &name = ts.name;
        if (name.compare(0, 2, "vx") != 0) {
            continue;
        }
        const size_t sep = name.find('_');
        BatchTimes &b = batches[std::strtoll(name.substr(2, sep == ov::str::npos ? ov::str::npos : sep - 2).c_str(), nullptr, 10)];
        const ov::str step = sep == ov::str::npos ? ov::str() : name.substr(sep + 1, name.find('_', sep + 1) - sep - 1);
        const int64_t t = int64_t(ts.gpu_ns);
        if (step == "begin") {
            b.begin = t;
        } else if (step == "copy") {
            b.copy = t;
        } else if (!step.empty() && step[0] == 'c') {
            const int group = int(std::strtol(step.c_str() + 1, nullptr, 10));
            if (group >= 0 && group < 3) {
                b.cascade[group] = t;
            }
        } else if (!step.empty() && step[0] == 'p') {
            const int pass = int(std::strtol(step.c_str() + 1, nullptr, 10));
            if (pass >= 0 && pass < 5) {
                b.pass[pass] = t;
            }
        }
    }
    // GPU timestamps are in nanoseconds; TimingStat stores microseconds.
    mutex.lock();
    for (const std::pair<const int64_t, BatchTimes> &kv : batches) {
        const BatchTimes &b = kv.second;
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
            const uint64_t sdf_ns = b.pass[0] >= b.begin ? uint64_t(b.pass[0] - b.begin) : 0;
            measured_batches.push_back({ uint32_t(kv.first), uint64_t(b.copy - b.begin) / 1000, sdf_ns / 1000 });
        }
    }
    mutex.unlock();
}

void GPUDirectMesher::_rt_record_samples(uint32_t p_count, const ov::ivec2 &p_lods, const ov::str &p_prefix, const ov::bytes &p_list) {
    OV_TRACE("mesher", "rt_record_samples");
    // One list per level (the render graph orders them through the scratch buffers); timestamps "_c0", "_c1",
    // "_c2" after the groups j >= 3, j = 2 and j = 1, empty groups included.
    auto mark = [&](int p_group) {
        if (!p_prefix.empty()) {
            device->timestamp(p_prefix + "_c" + ov::itos(p_group));
        }
    };
    int j = cascade->level_count(p_lods.x) - 1;
    for (; j >= 3; j--) {
        cascade->record_level(j, 0, p_count, p_lods);
    }
    mark(0);
    if (j == 2) {
        cascade->record_level(j--, 0, p_count, p_lods);
    }
    mark(1);
    if (j == 1) {
        cascade->record_level(j--, 0, p_count, p_lods);
    }
    mark(2);
    cascade->record_level(0, 0, p_count, p_lods, !p_list.empty()); // no instance listed: the bare cascade
}

void GPUDirectMesher::_rt_upload_table(const ov::bytes &p_list, const ov::bytes &p_records) {
    OV_TRACE("mesher", "rt_upload_table");
    // The records the batch's lists were built against (main thread, same snapshot), when they changed.
    if (!p_records.empty()) {
        SDFGpuTable::upload_records(*device, buffer_table_records, p_records);
    }
    if (!p_list.empty()) {
        OV_ERR_FAIL_COND_MSG(uint64_t(p_list.size()) > table_list_capacity,
                ov::format("GPUDirectMesher: batch list %d bytes over the %d B buffer.", p_list.size(), table_list_capacity));
        device->update_buffer(buffer_table_list, 0, p_list.size(), p_list.data());
    }
}

uint32_t GPUDirectMesher::_table_payload(std::vector<ChunkId> &r_chunks, std::vector<ChunkId> &r_rest, ov::bytes &r_ids, ov::bytes &r_list,
        ov::bytes &r_records, std::vector<std::vector<uint32_t>> *r_palettes, std::vector<uint32_t> *r_streamed) {
    OV_TRACE("mesher", "table_payload");
    SDFGpuTable *table = sdf->get_gpu_table();
    // Main thread, like the edits: the lists and the records come from the same snapshot.
    const uint64_t t_lists = now_usec();
    const uint32_t kept = table->build_lists(r_chunks, chunk_param, r_ids, r_list, uint32_t(table_list_capacity / 4), r_palettes, r_streamed);
    dispatch_times.build_lists += now_usec() - t_lists;
    r_rest.assign(r_chunks.begin() + kept, r_chunks.end());
    r_chunks.resize(kept);
    r_ids.resize(size_t(kept) * GPUFrame::CHUNK_INPUT_BYTES);
    if (table->upload_cursor() != sent_records) {
        const uint64_t t_records = now_usec();
        r_records = table->record_bytes(sent_records, sent_records);
        dispatch_times.record_bytes += now_usec() - t_records;
    }
    return table->version();
}

void GPUDirectMesher::_rt_read(ov::gpu::Buffer p_buffer, uint64_t p_size, void (GPUDirectMesher::*p_done)(const ov::bytes &, uint32_t), uint32_t p_batch_id) {
    rt_readbacks_in_flight++;
    device->read_buffer_async(p_buffer, 0, p_size, [this, p_done, p_batch_id](const ov::bytes &p_data) {
        if (!p_data.empty()) {
            (this->*p_done)(p_data, p_batch_id);
        }
        _rt_readback_done();
    });
}

// Push constants of the mesher's passes: batch count, page count, page capacity, 0.
static void mesher_push(uint8_t r_push[16], uint32_t p_count, uint32_t p_pages, uint32_t p_page_capacity) {
    ov::put_u32(r_push, p_count);
    ov::put_u32(r_push + 4, p_pages);
    ov::put_u32(r_push + 8, p_page_capacity);
    ov::put_u32(r_push + 12, 0);
}

void GPUDirectMesher::_rt_dispatch(uint32_t p_batch_id, const ov::bytes &p_chunk_ids, uint32_t p_count, const std::vector<int> &p_pages,
        const ov::ivec2 &p_lods, const ov::bytes &p_list, const ov::bytes &p_records, const ov::bytes &p_palette_batch) {
    if (!uniform_set.is_valid()) {
        return; // initialization failed (already reported)
    }
    Trace::name_thread("render");
    Trace::Scope span("mesher", "rt_dispatch");
    span.set_arg(p_count);
    Trace::flow(Trace::Kind::FLOW_STEP, "batch", p_batch_id);
    const uint64_t t0 = now_usec();
    _rt_read_timestamps();
    _rt_upload_table(p_list, p_records);
    device->update_buffer(buffer_palette_batch, 0, p_palette_batch.size(), p_palette_batch.data());

    device->update_buffer(buffer_input_chunk, 0, p_chunk_ids.size(), p_chunk_ids.data());
    device->clear_buffer(buffer_chunk_dir, 0, uint64_t(p_count) * 16);
    device->clear_buffer(buffer_status, 0, status_size_bytes);
    // Pages draw whole groups of quads: slots past the fill must be zero (degenerate), not stale.
    device->clear_texture(staging_texture); // headers included

    uint8_t push[16];
    mesher_push(push, p_count, uint32_t(p_pages.size()), settings.page_capacity);

    const uint32_t g = chunk_param.chunk_size / 8;
    const uint32_t groups[COMPUTE_PASSES][3] = {
        { g, 3 * g, g * p_count }, // edges (y spans the 3 axes)
        { g, g, g * p_count }, // vertices
        { 1, 1, 1 }, // reserve
        { g, g, g * p_count }, // emit
    };

    const ov::str prefix = "vx" + ov::itos(p_batch_id);
    device->timestamp(prefix + "_begin");
    _rt_record_samples(p_count, p_lods, prefix, p_list);
    device->timestamp(prefix + "_p0");
    for (int i = 0; i < COMPUTE_PASSES; i++) {
        OV_TRACE("mesher", "rt_record_pass");
        // One dispatch per pass so a timestamp can sit between passes.
        device->dispatch(pipelines[i], uniform_set, push, sizeof(push), groups[i][0], groups[i][1], groups[i][2]);
        device->timestamp(prefix + "_p" + ov::itos(i + 1));
    }

    // GPU -> GPU: quads into the page textures, used group counts into the indirect instance counts.
    for (size_t k = 0; k < p_pages.size(); k++) {
        OV_TRACE("pages", "rt_copy_page");
        pool->rt_copy_page_from(p_pages[k], staging_texture, uint32_t(k) * pool->get_page_rows());
        const QuadPagePool::PageRD page_rd = pool->rt_get_page_rd(p_pages[k]);
        OV_ERR_CONTINUE_MSG(!page_rd.command.is_valid(), ov::format("GPUDirectMesher: batch %d: page %d has no command buffer", p_batch_id, p_pages[k]));
        device->copy_buffer(buffer_status, page_rd.command, (STATUS_HEADER_UINTS + k) * 4, QuadPagePool::INDIRECT_INSTANCE_COUNT_OFFSET, 4);
    }
    device->timestamp(prefix + "_copy");

    // Bookkeeping only (where each chunk landed); rendering does not wait for it.
    _rt_read(buffer_chunk_dir, uint64_t(p_count) * 16, &GPUDirectMesher::_rt_on_directory, p_batch_id);
    _rt_read(buffer_status, status_size_bytes, &GPUDirectMesher::_rt_on_status, p_batch_id);

    mutex.lock();
    stats.rt_record_cpu.add(now_usec() - t0);
    mutex.unlock();
}

void GPUDirectMesher::_rt_dispatch_count(const ov::bytes &p_chunk_ids, uint32_t p_count, uint32_t p_batch_id, const ov::ivec2 &p_lods,
        const ov::bytes &p_list, const ov::bytes &p_records) {
    if (!uniform_set.is_valid()) {
        return;
    }
    _rt_upload_table(p_list, p_records);
    device->update_buffer(buffer_input_chunk, 0, p_chunk_ids.size(), p_chunk_ids.data());
    device->clear_buffer(buffer_chunk_dir, 0, uint64_t(p_count) * 16);
    device->clear_buffer(buffer_status, 0, status_size_bytes);
    // Counting only: every chunk on the ground block, no texel (only the base palette's materials are read).
    device->clear_buffer(buffer_palette_batch, 0, (uint64_t(p_count) + 1) * 4);

    uint8_t push[16];
    mesher_push(push, p_count, 0, settings.page_capacity);

    // samples (the cascade passes), edges, vertices: the vertices pass leaves the exact quad count in chunk_dir.x.
    const uint32_t g = chunk_param.chunk_size / 8;
    const uint32_t groups[2][3] = {
        { g, 3 * g, g * p_count },
        { g, g, g * p_count },
    };
    _rt_record_samples(p_count, p_lods, ov::str(), p_list);
    for (int i = 0; i < 2; i++) {
        device->dispatch(pipelines[i], uniform_set, push, sizeof(push), groups[i][0], groups[i][1], groups[i][2]);
    }
    _rt_read(buffer_chunk_dir, uint64_t(p_count) * 16, &GPUDirectMesher::_rt_on_audit_directory, p_batch_id);
}

void GPUDirectMesher::_rt_on_audit_directory(const ov::bytes &p_data, uint32_t p_batch_id) {
    mutex.lock();
    CompletedBatch c;
    c.batch_id = p_batch_id;
    c.directory = p_data;
    audit_completed.push_back(c);
    mutex.unlock();
}

void GPUDirectMesher::_rt_on_directory(const ov::bytes &p_data, uint32_t p_batch_id) {
    OV_TRACE("mesher", "readback");
    Trace::flow(Trace::Kind::FLOW_STEP, "batch", p_batch_id);
    mutex.lock();
    CompletedBatch &c = partial[p_batch_id];
    c.batch_id = p_batch_id;
    c.directory = p_data;
    if (!c.status.empty()) {
        completed.push_back(c);
        partial.erase(p_batch_id);
    }
    mutex.unlock();
}

void GPUDirectMesher::_rt_on_status(const ov::bytes &p_data, uint32_t p_batch_id) {
    OV_TRACE("mesher", "readback");
    mutex.lock();
    CompletedBatch &c = partial[p_batch_id];
    c.batch_id = p_batch_id;
    c.status = p_data;
    if (!c.directory.empty()) {
        completed.push_back(c);
        partial.erase(p_batch_id);
    }
    mutex.unlock();
}

void GPUDirectMesher::_rt_kill(const std::vector<int> &p_pages, const std::vector<int64_t> &p_ranges, uint64_t p_flow) {
    OV_TRACE("pages", "rt_kill");
    Trace::flow(Trace::Kind::FLOW_END, "kill", p_flow);
    for (size_t i = 0; i < p_pages.size(); i++) {
        pool->rt_kill_range(p_pages[i], uint32_t(p_ranges[i * 2]), uint32_t(p_ranges[i * 2 + 1]));
    }
}

void GPUDirectMesher::_rt_compact(const std::vector<int> &p_copies, const std::vector<int> &p_group_counts, uint64_t p_flow) {
    OV_TRACE("pages", "rt_compact");
    Trace::flow(Trace::Kind::FLOW_END, "compact", p_flow);
    for (size_t i = 0; i + 4 < p_copies.size(); i += 5) {
        pool->rt_copy_rows(p_copies[i], p_copies[i + 1], uint32_t(p_copies[i + 2]), uint32_t(p_copies[i + 3]), uint32_t(p_copies[i + 4]));
    }
    for (size_t i = 0; i + 1 < p_group_counts.size(); i += 2) {
        pool->rt_set_group_count(p_group_counts[i], uint32_t(p_group_counts[i + 1]));
    }
}

template <typename IdOf>
RenderBox GPUDirectMesher::_merged_aabb(size_t p_count, IdOf p_id) const {
    RenderBox bounds = _chunk_aabb(p_id(0));
    for (size_t i = 1; i < p_count; i++) {
        bounds.merge_with(_chunk_aabb(p_id(i)));
    }
    return bounds;
}

PageDecode GPUDirectMesher::_page_decode() const {
    // The page shader rebuilds render positions from the group header's chunk id, relative to the render origin (low
    // words, wrapping).
    PageDecode decode;
    decode.chunk_size = chunk_param.chunk_size;
    decode.skirt_size = chunk_param.skirt_size;
    decode.pos_scale = PageFormat::pos_scale(chunk_param.chunk_size);
    decode.voxel_size = chunk_param.voxel_size;
    decode.origin_chunk_lo = WorldCoord::lo32_i(chunk_param.origin_chunk);
    decode.origin_voxel_lo = WorldCoord::lo32_i(chunk_param.origin_voxel());
    return decode;
}

void GPUDirectMesher::set_render_origin(const ChunkId &p_origin) {
    chunk_param.origin_chunk = p_origin;
    host->set_decode(_page_decode());
    for (size_t page = 0; page < page_state.size(); page++) {
        if (!page_state[page].bounds_dirty) {
            page_state[page].bounds_dirty = true;
            dirty_bounds_pages.push_back(int(page));
        }
    }
    _update_dirty_bounds();
    // A batch in flight: its pages hold the batch's conservative box until its directory lands.
    for (const std::pair<uint32_t, InFlightBatch> &kv : in_flight) {
        const std::vector<ChunkId> &chunks = kv.second.chunks;
        const RenderBox box = _merged_aabb(chunks.size(), [&](size_t i) { return chunks[i]; });
        for (const int page : kv.second.pages) {
            if (page_state[page].chunks.empty()) {
                pool->set_page_aabb(page, box);
            }
        }
    }
}

RenderBox GPUDirectMesher::_chunk_aabb(const ChunkId &p_chunk) const {
    const Float3 center = ChunkMath::chunk_render_center(p_chunk, chunk_param);
    const float size = ChunkMath::world_chunk_size(p_chunk, chunk_param);
    // Skirt cells extend past the chunk bounds; be generous rather than cull visible geometry.
    const float margin = size * 0.25f;
    const float half = size * 0.5f + margin;
    return RenderBox{ Float3{ center.x - half, center.y - half, center.z - half }, Float3{ half * 2.0f, half * 2.0f, half * 2.0f } };
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
        ps.bounds = _merged_aabb(ps.chunks.size(), [&](size_t i) { return ps.chunks[i].id; });
        pool->set_page_aabb(page, ps.bounds);
    }
    dirty_bounds_pages.clear();
}

bool GPUDirectMesher::_pop_queue(ChunkId &r_chunk, ChunkGate::Kind &r_kind, int &r_queue) {
    // Instance edits first, finest LOD first: the chunk under the edit is what the user is looking at.
    for (std::vector<ChunkId> &queue : remesh_queue) {
        if (queue.empty()) {
            continue;
        }
        r_chunk = queue.back();
        queue.pop_back();
        remesh_queued.erase(r_chunk);
        stats.remesh_pops++;
        r_kind = ChunkGate::REMESH;
        return true;
    }
    for (size_t q = 0; q < chunk_queue.size(); q++) {
        std::vector<ChunkId> &queue = chunk_queue[q];
        if (queue.empty()) {
            continue;
        }
        r_chunk = queue.back();
        queue.pop_back();
        r_kind = ChunkGate::CHUNK;
        r_queue = int(q);
        return true;
    }
    return false;
}

void GPUDirectMesher::_requeue(const ChunkId &p_chunk, bool p_remesh) {
    if (p_remesh) {
        _queue_remesh(p_chunk);
        return;
    }
    const int lod = ChunkMath::get_lod(p_chunk);
    if (lod >= 0 && lod < lod_count) {
        chunk_queue[ChunkMath::load_queue(p_chunk, lod_count, load_view)].push_back(p_chunk);
    }
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
    const uint32_t version = sdf->get_gpu_table()->version();
    for (const ChunkId &chunk : p_chunks) {
        required_version[chunk] = version;
        _queue_remesh(chunk);
        stats.remesh_chunks++;
    }
    // Chunks in flight for their first mesh aren't loaded yet, so the terrain doesn't list them: their
    // result, computed from the old table, must not be kept either (it is dropped and re-queued on landing).
    for (const std::pair<uint32_t, InFlightBatch> &kv : in_flight) {
        if (kv.second.table_version >= version) {
            continue;
        }
        for (const ChunkId &chunk : kv.second.chunks) {
            if (sdf->edit_affects_chunk(chunk)) {
                required_version[chunk] = version;
            }
        }
    }
    uint64_t queued = 0;
    for (const std::vector<ChunkId> &q : remesh_queue) {
        queued += q.size();
    }
    stats.remesh_queue_max = ov::max(stats.remesh_queue_max, queued);
}

void GPUDirectMesher::_flush_kills() {
    if (pending_kill_pages.empty()) {
        return;
    }
    OV_TRACE("pages", "flush_kills");
    Trace::flow(Trace::Kind::FLOW_START, "kill", ++rt_posts);
    device->post([this, pages = std::move(pending_kill_pages), ranges = std::move(pending_kill_ranges), flow = rt_posts]() { _rt_kill(pages, ranges, flow); });
    pending_kill_pages.clear();
    pending_kill_ranges.clear();
}

void GPUDirectMesher::process_queue() {
    frame++;
    // The parked chunks whose records came, back in their queues: they pop first.
    gate.take_ready([this](const ChunkId &p_chunk, ChunkGate::Kind p_kind) {
        if (p_kind == ChunkGate::AUDIT) {
            audit_queue.push_back(p_chunk);
        } else {
            _requeue(p_chunk, p_kind == ChunkGate::REMESH);
        }
    });
    gate.begin_frame(settings.max_admits_per_frame);
    // Before acquiring pages: a page released since it was marked must not be touched.
    _update_dirty_bounds();
    // Before acquiring pages too: a kill must not land on a page reused by this frame's batches.
    _flush_kills();
    _flush_hidden();

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
        stats.in_flight_max = ov::max(stats.in_flight_max, uint64_t(in_flight.size() + audit_in_flight.size()));
        if (budgeted) {
            stats.budget_frame_estimate.add(uint64_t(budget_us - budget_left));
        }
    }
}

void GPUDirectMesher::_update_cost_model() {
    std::vector<MeasuredBatch> measured;
    mutex.lock();
    measured.swap(measured_batches);
    mutex.unlock();
    for (const MeasuredBatch &m : measured) {
        const BatchCost *cost = batch_costs.getptr(m.batch_id);
        if (cost == nullptr) {
            continue;
        }
        const uint32_t words = cost->list_words;
        const double estimate = cost_model.estimate(cost->lod_counts, words);
        if (estimate > 0.0) {
            stats.model_ratio_sum += m.total_us / estimate;
            stats.model_ratio_count++;
        }
        cost_model.observe(cost->lod_counts, double(m.total_us), words);
        ListWordsBin &bin = gpu_by_list_words[words == 0 ? 0 : 32 - __builtin_clz(words)];
        bin.batches++;
        bin.words += words;
        bin.sdf_us += m.sdf_us;
        bin.total_us += m.total_us;
        batch_costs.erase(m.batch_id);
    }
    // A batch whose timestamps were never captured (over the per-frame limit) is forgotten after a while.
    if (batch_costs.size() > 1024) {
        std::vector<uint32_t> old;
        for (const std::pair<uint32_t, BatchCost> &kv : batch_costs) {
            if (next_batch_id - kv.first > 1024) {
                old.push_back(kv.first);
            }
        }
        for (uint32_t id : old) {
            batch_costs.erase(id);
        }
    }
}

void GPUDirectMesher::_pop_batch(double p_budget_us, std::vector<ChunkId> &r_chunks, ov::HashSet<ChunkId, Vector3i64Hasher> &r_remeshes,
        ov::HashMap<ChunkId, uint64_t, Vector3i64Hasher> &r_bounds) {
    OV_TRACE("mesher", "pop_batch");
    double estimate = 0.0;
    ChunkId chunk;
    ChunkGate::Kind kind = ChunkGate::CHUNK;
    uint64_t words = 0;
    const auto pop = [this](ChunkId &r_chunk, ChunkGate::Kind &r_kind, int &r_queue) { return _pop_queue(r_chunk, r_kind, r_queue); };
    const auto requeue = [this](const ChunkId &p_chunk, ChunkGate::Kind p_kind) { _requeue(p_chunk, p_kind == ChunkGate::REMESH); };
    // With a budget the batch stops once its estimate reaches what's left (a frame dispatches at least
    // one chunk: budget_left starts > 0).
    while (r_chunks.size() < settings.batch_size && estimate < p_budget_us && gate.next(chunk, kind, pop, requeue, &words)) {
        r_chunks.push_back(chunk);
        if (kind == ChunkGate::REMESH) {
            r_remeshes.insert(chunk);
        }
        r_bounds.insert(chunk, words);
        const int lod = ChunkMath::get_lod(chunk);
        if (lod >= 0 && lod < lod_count) {
            estimate += cost_model.estimate_chunk(lod, words);
        }
    }
}

uint64_t GPUDirectMesher::_observe_lists(const std::vector<ChunkId> &p_chunks, const ov::bytes &p_ids, const std::vector<uint32_t> &p_streamed,
        const ov::HashMap<ChunkId, uint64_t, Vector3i64Hasher> &p_bounds) {
    std::vector<uint64_t> bounds(lod_count, 0), streamed(lod_count, 0), other(lod_count, 0);
    std::vector<uint32_t> counts(lod_count, 0);
    uint64_t total = 0;
    for (size_t i = 0; i < p_chunks.size(); i++) {
        const uint64_t *known = p_bounds.getptr(p_chunks[i]);
        const uint64_t bound = known ? *known : 0;
        total += bound;
        const int lod = ChunkMath::get_lod(p_chunks[i]);
        if (lod >= 0 && lod < lod_count) {
            counts[lod]++;
            bounds[lod] += bound;
            const uint64_t words = uint64_t(ov::max(0, ov::get_s32(p_ids.data() + i * GPUFrame::CHUNK_INPUT_BYTES + GPUFrame::CHUNK_LIST_COUNT_AT)));
            const uint64_t from_stream = i < p_streamed.size() ? ov::min(uint64_t(p_streamed[i]), words) : 0;
            streamed[lod] += from_stream;
            other[lod] += words - from_stream;
        }
    }
    for (int lod = 0; lod < lod_count; lod++) {
        cost_model.observe_list(lod, bounds[lod], streamed[lod], other[lod], counts[lod]);
    }
    return total;
}

bool GPUDirectMesher::_dispatch_mesh_batch(double &r_budget_us) {
    std::vector<ChunkId> chunks;
    chunks.reserve(settings.batch_size);
    ov::HashSet<ChunkId, Vector3i64Hasher> remeshes; // popped from remesh_queue: they go back there if they wait
    ov::HashMap<ChunkId, uint64_t, Vector3i64Hasher> bounds; // each chunk's list bound (SDFBase::admit)
    _pop_batch(r_budget_us, chunks, remeshes, bounds);
    if (chunks.empty()) {
        return false;
    }
    Trace::Scope span("mesher", "dispatch_batch");
    span.set_arg(chunks.size());
    const uint64_t t0 = now_usec();

    // The queue already pops in Morton order; re-sorting covers re-queued overflow chunks.
    const ChunkId origin = chunk_param.origin_chunk;
    std::sort(chunks.begin(), chunks.end(), [origin](const ChunkId &a, const ChunkId &b) { return morton_key(a, origin) < morton_key(b, origin); });

    ov::bytes ids(chunks.size() * GPUFrame::CHUNK_INPUT_BYTES);
    for (size_t i = 0; i < chunks.size(); i++) {
        GPUFrame::encode_chunk_input(ids.data() + i * GPUFrame::CHUNK_INPUT_BYTES, chunks[i], chunk_param);
    }
    ov::bytes list;
    ov::bytes records;
    std::vector<std::vector<uint32_t>> chunk_palettes;
    std::vector<ChunkId> rest;
    std::vector<uint32_t> streamed_words;
    const uint32_t table_version = _table_payload(chunks, rest, ids, list, records, &chunk_palettes, &streamed_words);
    for (const ChunkId &c : rest) {
        _requeue(c, remeshes.has(c));
        stats.list_requeued++;
    }

    InFlightBatch batch;
    batch.dispatch_frame = frame;
    std::vector<uint32_t> lod_counts(lod_count, 0);
    const uint32_t list_words = uint32_t(list.size() / 4);
    const uint64_t list_bound = _observe_lists(chunks, ids, streamed_words, bounds);
    const ov::ivec2 lods = ChunkMath::lod_range(chunks);
    const RenderBox box = _merged_aabb(chunks.size(), [&](size_t i) { return chunks[i]; });
    for (const ChunkId &c : chunks) {
        const int lod = ChunkMath::get_lod(c);
        if (lod >= 0 && lod < lod_count) {
            lod_counts[lod]++;
        }
        working_generation.insert(c);
    }
    r_budget_us -= cost_model.estimate(lod_counts, list_words);
    ov::bytes palette_batch = palettes.place_batch(chunk_palettes, batch.palettes);

    std::vector<int> pages;
    for (uint32_t k = 0; k < settings.pages_per_batch; k++) {
        const int page = pool->acquire_page();
        if ((int)page_state.size() <= page) {
            page_state.resize(page + 1);
        }
        page_state[page] = PageState();
        // Conservative until the directory says which chunks really landed in this page.
        pool->set_page_aabb(page, box);
        batch.pages.push_back(page);
        pages.push_back(page);
    }

    const uint32_t batch_id = next_batch_id++;
    const uint32_t count = uint32_t(chunks.size());
    stats.list_words_total += list_words;
    stats.list_bound_total += list_bound;
    stats.list_words_max = ov::max(stats.list_words_max, uint64_t(list_words));
    batch_costs[batch_id] = BatchCost{ std::move(lod_counts), list_words };
    batch.table_version = table_version;
    batch.chunks = std::move(chunks);
    in_flight[batch_id] = std::move(batch);
    Trace::flow(Trace::Kind::FLOW_START, "batch", batch_id);
    device->post([this, batch_id, ids = std::move(ids), count, pages = std::move(pages), lods, list = std::move(list), records = std::move(records),
                         palette_batch = std::move(palette_batch)]() { _rt_dispatch(batch_id, ids, count, pages, lods, list, records, palette_batch); });

    stats.batches++;
    stats.dispatch_cpu.add(now_usec() - t0);
    VLOG(3, "GPUDirectMesher: dispatched batch ", batch_id, " (", int64_t(count), " chunks)");
    return true;
}

bool GPUDirectMesher::_dispatch_audit_batch() {
    if (audit_queue.empty()) {
        return false;
    }
    std::vector<ChunkId> chunks;
    ChunkId chunk;
    ChunkGate::Kind kind = ChunkGate::AUDIT;
    const auto pop = [this](ChunkId &r_chunk, ChunkGate::Kind &r_kind, int &) {
        if (audit_queue.empty()) {
            return false;
        }
        r_chunk = audit_queue.back();
        r_kind = ChunkGate::AUDIT;
        audit_queue.pop_back();
        return true;
    };
    const auto requeue = [this](const ChunkId &p_chunk, ChunkGate::Kind) { audit_queue.push_back(p_chunk); };
    while (chunks.size() < settings.batch_size && gate.next(chunk, kind, pop, requeue)) {
        chunks.push_back(chunk);
    }
    if (chunks.empty()) {
        return false;
    }

    ov::bytes ids(chunks.size() * GPUFrame::CHUNK_INPUT_BYTES);
    const ov::ivec2 lods = ChunkMath::lod_range(chunks);
    for (size_t i = 0; i < chunks.size(); i++) {
        GPUFrame::encode_chunk_input(ids.data() + i * GPUFrame::CHUNK_INPUT_BYTES, chunks[i], chunk_param);
    }
    ov::bytes list;
    ov::bytes records;
    std::vector<ChunkId> rest;
    _table_payload(chunks, rest, ids, list, records);
    audit_queue.insert(audit_queue.end(), rest.begin(), rest.end());
    const uint32_t batch_id = next_batch_id++;
    audit_in_flight[batch_id] = chunks;
    stats.audit_batches++;
    device->post([this, ids = std::move(ids), count = uint32_t(chunks.size()), batch_id, lods, list = std::move(list), records = std::move(records)]() {
        _rt_dispatch_count(ids, count, batch_id, lods, list, records);
    });
    return true;
}

void GPUDirectMesher::audit_count_chunks(const std::vector<ChunkId> &p_chunks) {
    audit_queue.insert(audit_queue.end(), p_chunks.begin(), p_chunks.end());
}

void GPUDirectMesher::take_audit_counts(std::vector<std::pair<ChunkId, uint32_t>> &r_counts) {
    r_counts.insert(r_counts.end(), audit_counts.begin(), audit_counts.end());
    audit_counts.clear();
}

// Word p_i of a readback (int32, host order like the GPU's).
static int32_t word(const ov::bytes &p_data, size_t p_i) {
    return (p_i + 1) * 4 <= p_data.size() ? ov::get_s32(p_data.data() + p_i * 4) : 0;
}

bool GPUDirectMesher::get_results(VoxelState &output) {
    mutex.lock();
    std::vector<CompletedBatch> done;
    done.swap(completed);
    std::vector<CompletedBatch> audit_done;
    audit_done.swap(audit_completed);
    mutex.unlock();

    for (CompletedBatch &c : audit_done) {
        const std::vector<ChunkId> *chunks = audit_in_flight.getptr(c.batch_id);
        OV_ERR_CONTINUE(chunks == nullptr);
        for (size_t i = 0; i < chunks->size(); i++) {
            audit_counts.push_back({ (*chunks)[i], uint32_t(word(c.directory, i * 4 + 0)) });
        }
        audit_in_flight.erase(c.batch_id);
    }

    if (done.empty()) {
        return false;
    }

    for (CompletedBatch &c : done) {
        OV_TRACE("mesher", "collect_batch");
        Trace::flow(Trace::Kind::FLOW_END, "batch", c.batch_id);
        const uint64_t t0 = now_usec();
        InFlightBatch *found = in_flight.getptr(c.batch_id);
        OV_ERR_CONTINUE(found == nullptr);
        InFlightBatch &batch = *found;

        const ov::bytes &status = c.status;
        const ov::bytes &dir = c.directory;
        stats.quads_emitted += uint32_t(word(status, 0));
        stats.overflow_chunks += uint32_t(word(status, 1));
        stats.missing_neighbor += uint32_t(word(status, 2));
        stats.nan_vertices += uint32_t(word(status, 3));

        std::vector<RenderBox> page_bounds(batch.pages.size());
        struct StaleRange {
            uint32_t page_index, offset, count;
        };
        std::vector<StaleRange> stale_ranges;
        std::vector<bool> page_has_bounds(batch.pages.size(), false);

        for (size_t k = 0; k < batch.pages.size(); k++) {
            page_state[batch.pages[k]].groups = uint32_t(word(status, STATUS_HEADER_UINTS + k));
            if (uint32_t(word(status, STATUS_HEADER_UINTS + k)) > pool->get_page_rows()) {
                OV_ERR_PRINT(ov::format("GPUDirectMesher: batch %d page %d: status fill %d groups", c.batch_id, int64_t(k), word(status, STATUS_HEADER_UINTS + k)));
            }
        }

        for (size_t i = 0; i < batch.chunks.size(); i++) {
            const ChunkId &chunk_id = batch.chunks[i];
            const uint32_t count = uint32_t(word(dir, i * 4 + 0));
            const uint32_t base = uint32_t(word(dir, i * 4 + 2));
            const uint32_t flags = uint32_t(word(dir, i * 4 + 3));
            working_generation.erase(chunk_id);

            if (flags & 1u) { // overflow: try again in a later batch
                stats.overflow_requeued++;
                // A loaded chunk is a re-mesh (instance edit): send_chunk rebuilds chunk_queue from the scan's dirty
                // chunks only, which a loaded chunk isn't, so it would be lost there.
                _requeue(chunk_id, output.loaded_chunk.has(chunk_id));
                continue;
            }

            // Staleness (instance edits): this result was computed from a table version older than the
            // chunk's latest edit. Drop it and re-mesh from the current table.
            if (const uint32_t *required_ptr = required_version.getptr(chunk_id)) {
                const uint32_t required = *required_ptr;
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
                    OV_ERR_PRINT(ov::format("GPUDirectMesher: bad directory entry: batch %d chunk %d count %d base %d flags %d", c.batch_id, int64_t(i), count, base, flags));
                    continue;
                }
                out.owner = this;
                out.palette = std::exchange(batch.palettes[i], PalettePool::Block()); // the chunk owns it now
                out.page = batch.pages[k];
                out.quad_offset = base % settings.page_capacity;
                out.quad_count = count;
                PageState &ps = page_state[out.page];
                ps.live_quads += count;
                ps.live_groups += groups_of(count);
                ps.chunks.push_back({ chunk_id, out.quad_offset, count });
                stats.live_quads += count;
                const RenderBox box = _chunk_aabb(chunk_id);
                page_bounds[k] = page_has_bounds[k] ? page_bounds[k].merge(box) : box;
                page_has_bounds[k] = true;
            }
            stats.chunks_meshed++;
            output.add_new_loaded(out);
        }

        // The palettes no loaded chunk took (overflow, stale, empty): nothing names their blocks any more (a stale
        // result's rows are killed below, before any batch that could reuse them).
        for (const PalettePool::Block &b : batch.palettes) {
            palettes.release(b);
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
        stats.readback_frames_max = ov::max(stats.readback_frames_max, frame - batch.dispatch_frame);
        in_flight.erase(c.batch_id);
        stats.collect_cpu.add(now_usec() - t0);
    }
    return true;
}

void GPUDirectMesher::release_chunk(const ChunkOutput &p_chunk) {
    OV_TRACE("pages", "release_chunk");
    OV_ERR_FAIL_INDEX(p_chunk.page, int(page_state.size()));
    // Its rows are killed (or its page reset) before a batch that could reuse the block is dispatched:
    // process_queue flushes the kills first, on the render thread's FIFO.
    palettes.release(p_chunk.palette);
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

void GPUDirectMesher::set_hidden(const ChunkOutput &p_chunk, uint64_t p_mask) {
    OV_ERR_FAIL_INDEX(p_chunk.page, int(page_state.size()));
    pending_hidden.push_back(p_chunk.page);
    pending_hidden.push_back(p_chunk.quad_offset / QuadPagePool::QUAD_GROUP);
    pending_hidden.push_back(groups_of(p_chunk.quad_count));
    pending_hidden.push_back(int64_t(p_mask));
    pending_hidden.push_back(int64_t(WorldCoord::lo32(p_chunk.chunk_id.z)) | int64_t(p_chunk.palette.offset) << 32);
    stats.hidden_rows += groups_of(p_chunk.quad_count);
}

void GPUDirectMesher::_rt_set_hidden(const std::vector<int64_t> &p_writes, uint64_t p_flow) {
    OV_TRACE("pages", "rt_set_hidden");
    Trace::flow(Trace::Kind::FLOW_END, "hidden", p_flow);
    pool->rt_set_hidden(p_writes);
}

void GPUDirectMesher::_flush_hidden() {
    if (pending_hidden.empty()) {
        return;
    }
    // Recorded after the last compaction (VoxelState::push_coverage), flushed before the next one moves rows.
    OV_TRACE("pages", "flush_hidden");
    Trace::flow(Trace::Kind::FLOW_START, "hidden", ++rt_posts);
    device->post([this, writes = std::move(pending_hidden), flow = rt_posts]() { _rt_set_hidden(writes, flow); });
    pending_hidden.clear();
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
                key = ov::min(key, morton_key(c.id, chunk_param.origin_chunk));
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
    OV_TRACE("pages", "compact_pages");
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
        RenderBox bounds;
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
        const RenderBox box = _chunk_aabb(id);
        const float size = ChunkMath::world_chunk_size(id, chunk_param);
        bool fits = false;
        if (!dests.empty()) {
            const Dest &d = dests.back();
            const float smallest = ov::min(d.smallest_chunk, size);
            fits = d.groups + g <= page_groups && d.bounds.merge(box).longest_axis_size() <= settings.compact_max_extent * smallest;
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
        d.smallest_chunk = ov::min(d.smallest_chunk, size);
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

    std::vector<int> copies; // src page, dst page, src row, dst row, rows
    std::vector<int> group_counts; // dst page, groups
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
            const size_t n = copies.size();
            // Chunks that were neighbours in the source stay neighbours: extend the last copy.
            if (n > 0 && copies[n - 5] == m.src_page && copies[n - 4] == page && copies[n - 3] + copies[n - 1] == int(src_row) &&
                    copies[n - 2] + copies[n - 1] == int(dst_row[i])) {
                copies[n - 1] += int(rows);
            } else {
                copies.insert(copies.end(), { m.src_page, page, int(src_row), int(dst_row[i]), int(rows) });
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
                OV_ERR_PRINT(ov::format("GPUDirectMesher: compacted chunk %s is not loaded where its page says.", WorldCoord::to_string(m.chunk.id)));
            }
        }
        group_counts.insert(group_counts.end(), { page, int(d.groups) });
        begin = d.end;
    }
    // Queued before the sources' release, so the copies read them before anything reuses them.
    Trace::flow(Trace::Kind::FLOW_START, "compact", ++rt_posts);
    device->post([this, copies, group_counts, flow = rt_posts]() { _rt_compact(copies, group_counts, flow); });
    for (const int page : p_pages) {
        page_state[page] = PageState();
        pool->release_page(page);
    }

    stats.compact_rounds++;
    stats.pages_compacted += p_pages.size();
    stats.pages_compacted_into += dests.size();
    stats.chunks_moved += moves.size();
    stats.compact_copies += copies.size() / 5;
    VLOG(3, "GPUDirectMesher: compacted ", int64_t(p_pages.size()), " pages into ", int64_t(dests.size()), " (", int64_t(moves.size()), " chunks, ",
            int64_t(copies.size() / 5), " copies)");
    return true;
}

void GPUDirectMesher::send_chunk(const LODHashSet &p_chunks, const ScanView &p_view) {
    OV_TRACE("mesher", "queue_chunks");
    // Chunk ids are chunk centres in the same units at every LOD, so one Morton key orders them
    // spatially. Sorted once per scan; consecutive pops then form compact batches and pages.
    load_view = p_view;
    gate.next_scan(p_chunks, [this, &p_view](const ChunkId &p_chunk) { return ChunkMath::load_queue(p_chunk, lod_count, p_view); });
    std::vector<std::vector<std::pair<uint64_t, ChunkId>>> keyed(chunk_queue.size());
    for (const ChunkId &chunk_pos : p_chunks) {
        const int lod = ChunkMath::get_lod(chunk_pos);
        if (lod < 0 || lod >= lod_count) {
            continue;
        }
        if (!working_generation.has(chunk_pos) && !gate.keeps(chunk_pos)) {
            keyed[ChunkMath::load_queue(chunk_pos, lod_count, p_view)].push_back({ morton_key(chunk_pos, chunk_param.origin_chunk), chunk_pos });
        }
    }
    for (size_t q = 0; q < chunk_queue.size(); q++) {
        std::sort(keyed[q].begin(), keyed[q].end(),
                [](const std::pair<uint64_t, ChunkId> &a, const std::pair<uint64_t, ChunkId> &b) { return a.first > b.first; });
        std::vector<ChunkId> &queue = chunk_queue[q];
        queue.clear();
        queue.reserve(keyed[q].size());
        for (const std::pair<uint64_t, ChunkId> &k : keyed[q]) {
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
    // Re-meshes queued by an edit or by a stale result (get_results) are pending work too, and the parked chunks.
    return remesh_queued.is_empty() && gate.idle(ChunkGate::CHUNK) && gate.idle(ChunkGate::REMESH);
}

bool GPUDirectMesher::is_edit_settled() const {
    if (!remesh_queued.is_empty() || !gate.idle(ChunkGate::REMESH)) {
        return false;
    }
    const uint32_t version = sdf->get_gpu_table()->version();
    for (const std::pair<uint32_t, InFlightBatch> &kv : in_flight) {
        if (kv.second.table_version < version) {
            return false;
        }
        for (const ChunkId &chunk : kv.second.chunks) {
            if (required_version.has(chunk)) {
                return false;
            }
        }
    }
    return true;
}

void GPUDirectMesher::set_debug_view(int p_view) {
    host->set_debug_view(p_view);
}

ov::Value GPUDirectMesher::get_stats() {
    ov::Value d = ov::Value::object();
    d["palettes"] = palettes.get_stats();
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
                finest = ov::min(finest, ChunkMath::world_chunk_size(c.id, chunk_param));
            }
            const double extent = ps.bounds.longest_axis_size() / finest;
            extent_sum += extent;
            extent_max = ov::max(extent_max, extent);
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
    d["budget_frame_estimate"] = stats.budget_frame_estimate.to_value();
    // How well the cost model predicts a batch (1 = exactly), and its cost per chunk per LOD (us, -1 unmeasured).
    d["cost_model_ratio_avg"] = stats.model_ratio_count ? stats.model_ratio_sum / stats.model_ratio_count : 0.0;
    ov::Value cost_us = ov::Value::array();
    for (double c : cost_model.us_per_chunk) {
        cost_us.push_back(c);
    }
    d["cost_us_per_chunk"] = cost_us;
    // A list word's cost (us), the share of the chunks' list bounds (their keys' records) their lists hold, and their words past it.
    d["cost_us_per_word"] = cost_model.us_per_word;
    ov::Value list_share = ov::Value::array();
    ov::Value words_per_chunk = ov::Value::array();
    for (int lod = 0; lod < cost_model.lods(); lod++) {
        list_share.push_back(cost_model.list_share[lod]);
        words_per_chunk.push_back(cost_model.words_per_chunk[lod]);
    }
    d["cost_list_share"] = list_share; // per LOD
    d["cost_words_per_chunk"] = words_per_chunk;
    d["list_bound_total"] = (int64_t)stats.list_bound_total;
    d["chunks_meshed"] = (int64_t)stats.chunks_meshed;
    d["quads_emitted"] = (int64_t)stats.quads_emitted;
    d["overflow_chunks"] = (int64_t)stats.overflow_chunks;
    d["overflow_requeued"] = (int64_t)stats.overflow_requeued;
    d["list_requeued"] = (int64_t)stats.list_requeued;
    d["admit_parked"] = (int64_t)gate.parked;
    d["admit_later"] = (int64_t)gate.later;
    d["admit_capped_frames"] = (int64_t)gate.capped_frames;
    d["max_admits_per_frame"] = (int64_t)settings.max_admits_per_frame;
    d["list_words_total"] = (int64_t)stats.list_words_total;
    d["list_words_max"] = (int64_t)stats.list_words_max;
    d["gpu_by_list_words"] = _gpu_by_list_words_stats();
    d["audit_batches"] = (int64_t)stats.audit_batches;
    d["missing_neighbor"] = (int64_t)stats.missing_neighbor;
    d["nan_vertices"] = (int64_t)stats.nan_vertices;
    d["kills"] = (int64_t)stats.kills;
    d["hidden_rows"] = (int64_t)stats.hidden_rows;
    d["pages_returned_unused"] = (int64_t)stats.pages_returned_unused;
    // Quad slots the vertex shader runs on (whole groups of live pages): dead and padding included.
    d["drawn_quad_slots"] = drawn_groups * QuadPagePool::QUAD_GROUP;
    d["compact_rounds"] = (int64_t)stats.compact_rounds;
    d["compact_skipped"] = (int64_t)stats.compact_skipped;
    d["pages_compacted"] = (int64_t)stats.pages_compacted;
    d["pages_compacted_into"] = (int64_t)stats.pages_compacted_into;
    d["chunks_moved"] = (int64_t)stats.chunks_moved;
    d["compact_copies"] = (int64_t)stats.compact_copies;
    d["compact_cpu"] = stats.compact_cpu.to_value();
    d["live_quads"] = (int64_t)stats.live_quads;
    d["pages_with_live_chunks"] = pages_with_live;
    const int64_t slots_in_live_pages = (int64_t)pages_with_live * settings.page_capacity;
    d["page_occupancy"] = slots_in_live_pages ? (double)stats.live_quads / slots_in_live_pages : 0.0;
    d["dispatch_cpu"] = stats.dispatch_cpu.to_value();
    d["collect_cpu"] = stats.collect_cpu.to_value();
    mutex.lock();
    d["rt_record_cpu"] = stats.rt_record_cpu.to_value();
    d["gpu_compute"] = stats.gpu_compute.to_value();
    d["gpu_copy"] = stats.gpu_copy.to_value();
    for (int i = 0; i < 5; i++) {
        d[ov::str("gpu_pass_") + PASS_NAMES[i]] = stats.gpu_pass[i].to_value();
    }
    for (int g = 0; g < 4; g++) {
        d[ov::str("gpu_cascade_") + CASCADE_GROUP_NAMES[g]] = stats.gpu_cascade[g].to_value();
    }
    d["cascade_scratch_mb"] = double(cascade_scratch_bytes) / 1048576.0;
    mutex.unlock();
    ov::Value frames = ov::Value::object();
    const uint64_t collected = stats.batches - in_flight.size();
    frames["avg"] = collected ? (double)stats.readback_frames_total / collected : 0.0;
    frames["max"] = int64_t(stats.readback_frames_max);
    d["readback_latency_frames"] = frames;
    d["pool"] = pool->get_stats();
    return d;
}

ov::Value GPUDirectMesher::_gpu_by_list_words_stats() const {
    ov::Value out = ov::Value::array();
    for (int b = 0; b < LIST_WORD_BINS; b++) {
        const ListWordsBin &bin = gpu_by_list_words[b];
        if (bin.batches == 0) {
            continue;
        }
        ov::Value d = ov::Value::object();
        d["words_lo"] = b == 0 ? int64_t(0) : int64_t(1) << (b - 1);
        d["words_hi"] = b == 0 ? int64_t(0) : (int64_t(1) << b) - 1;
        d["batches"] = (int64_t)bin.batches;
        d["words_avg"] = double(bin.words) / double(bin.batches);
        d["sdf_us_avg"] = double(bin.sdf_us) / double(bin.batches);
        d["total_us_avg"] = double(bin.total_us) / double(bin.batches);
        out.push_back(d);
    }
    return out;
}

void GPUDirectMesher::destroy() {
    chunk_queue.clear();
    lod_count = 0;
    // Posts its own cleanup. `pool` is not cleared: the device-thread work posted before (dispatch, kills) still uses it
    // and runs first (FIFO).
    pool->destroy();
    device->post([this]() { _rt_destroy(); });
}

void GPUDirectMesher::_rt_destroy() {
    if (cascade != nullptr) {
        // Its bind sets use the input, param and sdf buffers freed below.
        cascade->free();
        delete cascade;
        cascade = nullptr;
    }
    // Dependents (the bind set) before the pipelines and resources they use.
    const ov::gpu::Handle handles[] = { uniform_set, pipelines[0], pipelines[1], pipelines[2], pipelines[3], buffer_chunk_dir, buffer_input_chunk,
        buffer_sdf, buffer_edges, buffer_vertices, buffer_material, buffer_palette_batch, buffer_chunk_param, staging_texture, buffer_status,
        buffer_table_records, buffer_table_list };
    for (const ov::gpu::Handle h : handles) {
        if (h.is_valid()) {
            device->free(h);
        }
    }
    host->rt_set_palette_texture(ov::gpu::Texture());
    palettes.rt_free(device); // after the bind set that uses it
    // The objects can go now (the readbacks copied them to staging when they were recorded), not this one.
    rt_destroying = true;
    if (rt_readbacks_in_flight == 0) {
        delete this;
    }
}

void GPUDirectMesher::_rt_readback_done() {
    rt_readbacks_in_flight--;
    if (rt_destroying && rt_readbacks_in_flight == 0) {
        delete this;
    }
}
