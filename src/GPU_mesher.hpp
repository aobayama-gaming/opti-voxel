#pragma once

#include "chunk_gate.hpp"
#include "chunk_math.hpp"
#include "chunk_memory.hpp"
#include "debug_log.hpp"
#include "gpu_device.hpp"
#include "legacy_mesher_glsl.hpp"
#include "lod_structures.hpp"
#include "mesh_host.hpp"
#include "sdf_base.hpp"

#include "ov/bytes.hpp"
#include "ov/containers.hpp"
#include "ov/executor.hpp"
#include "ov/types.hpp"
#include "ov/value.hpp"

#include <memory>
#include <mutex>

struct UnpackChunk {
    ChunkId chunk_id;

    // points, normals, indices; materials per point: its global material (material_library.hpp), from the chunk's
    // palette (the validator compares it).
    ChunkMesh mesh;
    std::vector<uint32_t> palette; // the chunk's palette (build_lists), main thread -> worker with the batch

    uint32_t point_offset = 0; // used to fill the array
    uint32_t index_count = 0; // indices actually written by the CPU triangulation

    ov::HashMap<uint32_t, uint32_t, ov::DefaultHash> index_storage;

    std::vector<int32_t> quad_masks;
    // Cell (flat index) of each point. The triangulation must not derive it from the quantised position:
    // a vertex on its cell's upper face (local coordinate 1.0, a crossing at t = 1) floors into the next
    // cell, and its quads then take neighbour indices from the wrong cells.
    std::vector<int32_t> point_cells;

    uint64_t host_mesh = 0; // MeshHost::create_mesh's

    void resize(int point_number, int triangles_number) {
        mesh.points.resize(size_t(point_number));
        mesh.normals.resize(size_t(point_number));
        mesh.materials.resize(size_t(point_number));
        quad_masks.resize(size_t(point_number));
        point_cells.resize(size_t(point_number));

        mesh.indices.resize(size_t(triangles_number) * 3);
    }
};

// Legacy pipeline (pipeline_mode 1), kept as the correctness oracle of GPU Direct (MeshValidator)
// and as its perf baseline. A worker owns a local device and runs three compute
// passes per batch (SDF samples, edge crossings, one point per surface cell with its quad mask),
// reads everything back and triangulates on the CPU: one mesh + instance per chunk.
class CascadePasses;

class GPUMesher : public ChunkReleaser, public ov::WorkerLoop {
private:
    // Each chunk's host mesh, under its output's handle: written by the worker as it emits the chunk, freed on the main
    // thread (release_chunk), so under the mutex. Keyed by a serial handle, not the chunk id: a re-meshed chunk's new
    // output exists before the old one is freed.
    ov::HashMap<uint64_t, uint64_t, ov::DefaultHash> chunk_meshes;
    uint64_t next_handle = 0;

    // Constant between run
    SDFBase *sdf;
    ChunkParam chunk_param;

    // The worker (a worker of the core's executor): enter() makes the device, a step() per batch handed over, exit()
    // frees the device, on its thread.
    std::unique_ptr<ov::Worker> worker;

    //worker availability
    std::mutex mutex;
    // Held by the worker while it creates a host mesh at chunk_param.origin_chunk, and by set_render_origin while it
    // moves the origin and the meshes made: a mesh is made either before the move (and shifted) or after it.
    std::mutex origin_mutex;
    bool worker_available = false; // set by the worker once its device and buffers exist (process_queue sizes lists by them)
    bool result_available = false;
    uint32_t batch_count = 0; // number of real chunks in the current batch (<= chunk_batch_number)
    ov::ivec2 batch_lods; // smallest and largest LOD of the current batch (cascade levels and variants to run)
    bool trim_indices = false; // drop the zero padding at the end of the index buffer (legacy bug)
    // Dual contouring of Hermite data (DC-lite) in the vertex pass (gpu_glsl.hpp). Set before
    // initialize(): the worker compiles the vertex shader there.
    bool dual_contouring = false;

    // Stats (written by the worker under mutex, read by get_stats()).
    struct Stats {
        TimingStat batch_total; // whole worker iteration
        TimingStat gpu_submit_sync;
        TimingStat points_fetch;
        TimingStat points_process;
        TimingStat triangulate;
        TimingStat render_upload;
        uint64_t chunks_meshed = 0;
        uint64_t points = 0;
        uint64_t triangles = 0;
    } stats;

    MeshHost *host = nullptr;

    uint32_t chunk_batch_number;

    //input
    ov::bytes input_chunk_id;
    ov::bytes input_table_list; // the batch's instance list (archetype instances)
    ov::bytes input_table_records; // the table's records changed since the last batch (record_bytes spans)

    // GPU stuff: the worker's local device

    std::unique_ptr<ov::gpu::Device> device;

    // Buffers

    ov::gpu::Buffer buffer_input_chunk_id;

    ov::gpu::Buffer buffer_atomic_counter;
    ov::gpu::Buffer buffer_sdf_sample;
    ov::gpu::Buffer buffer_edge_intersect;

    ov::gpu::Buffer buffer_output_point;
    ov::gpu::Buffer buffer_chunk_param;
    // The instance table's buffers (archetype instances), on the local device.
    ov::gpu::Buffer buffer_table_records;
    ov::gpu::Buffer buffer_table_list;
    ov::gpu::Buffer buffer_material; // per sample: its palette index (cascade level 0 -> third pass)
    uint64_t table_list_capacity = 0; // bytes
    SDFGpuTable::UploadCursor sent_records; // main thread: the records last handed to the worker
    std::vector<ChunkId> in_flight_ids; // main thread: the chunks of the batch handed to the worker
    // Uniform set

    ov::gpu::BindSet uniform_set;

    ov::gpu::Pipeline pipeline_second_pass;
    ov::gpu::Pipeline pipeline_third_pass;

    // The cascade passes: pass 1 (the samples), in sub-batches of at most
    // CASCADE_MAX_CHUNKS chunks (its scratch, ~92 KB per chunk with 4 channels, is not sized for 1024).
    static constexpr uint32_t CASCADE_MAX_CHUNKS = 128;
    // sdf_material's binding (after the table's 6, 7), and the palette index's place in a point's word c: above the
    // chunk index (16 bits) and the quad mask (4), within an int.
    static constexpr int MATERIAL_BINDING = 8;
    static constexpr uint32_t POINT_PALETTE_SHIFT = LegacyMesherGLSL::POINT_PALETTE_SHIFT;
    static_assert(POINT_PALETTE_SHIFT + PageFormat::PALETTE_BITS <= 31, "a point's word c is an int");
    CascadePasses *cascade = nullptr;

    // One set per ChunkMath::load_queue: in the view cone first, then by LOD.
    std::vector<LODHashSet *> chunk_queue;
    ScanView load_view; // the view of the last send_chunk, for the chunks queued again
    int lod_count = 0;
    // Instance edits' re-meshes (loaded chunks), popped before chunk_queue. Kept apart because send_chunk
    // rebuilds chunk_queue from each scan's dirty chunks, which a loaded chunk isn't.
    std::vector<LODHashSet *> remesh_queue;

    LODHashSet working_generation;
    ChunkGate gate; // the SDF's gating of the popped chunks (streamed records)
    DispatchTimes dispatch_times; // gate_admit read from the gate

    ov::gpu::Pipeline compile_dynamic_shader(const char *algorithm_pass, const ov::str &p_name);

    void _execute_gpu_pipeline(const ov::bytes &p_chunk_id, uint32_t p_count, const ov::ivec2 &p_lods, const ov::bytes &p_table_list,
            const ov::bytes &p_table_records);
    // ov::WorkerLoop.
    void enter() override;
    bool step() override;
    void exit() override;
    void _initialize_gpu_compute();

    // r_kind: REMESH from remesh_queue, else CHUNK; r_queue: its index in its queue list (ChunkGate::next's pop).
    bool _pop_queue(ChunkId &new_chunk, ChunkGate::Kind &r_kind, int &r_queue);
    // A popped chunk back in its queue.
    void _requeue(const ChunkId &p_chunk, bool p_remesh);

    bool _is_available();

    ov::ivec3 _unflatten_index_z(uint32_t flat_idx);
    uint32_t _get_flat_index_z(int p_x, int p_y, int p_z);

    void render_chunk();

    std::vector<UnpackChunk> unpacked_chunks;

    std::vector<ChunkOutput> out_queue;

public:
    // p_host draws the chunks' meshes (and must outlive stop()).
    void initialize(SDFBase *p_sdf, ChunkParam p_chunk_param, uint32_t p_chunk_batch_number, int lod_numbers, MeshHost *p_host);

    void process_queue();

    // The chunks to mesh (replacing the queue), loaded in the order of ChunkMath::load_queue for p_view.
    void send_chunk(const LODHashSet &p_chunks, const ScanView &p_view);
    // Re-mesh chunks an instance edit changed (legacy path; the direct mesher has the versioned queue).
    void remesh_chunks(const std::vector<ChunkId> &p_chunks);

    bool get_results(VoxelState &output);

    // ChunkReleaser: frees the chunk's host mesh (main thread). The legacy meshes draw everything.
    void release_chunk(const ChunkOutput &p_chunk) override;
    void set_hidden(const ChunkOutput &, uint64_t) override {}
    // The host mesh of a chunk this mesher made (0 for an empty one): the validator's legacy triangles.
    uint64_t mesh_of(const ChunkOutput &p_chunk);

    // Stops the worker thread and frees the local device (on the worker, which owns it).
    void stop();

    bool is_idle();
    // The render origin moved (World::rebase): the meshes made follow it (MeshHost::shift_meshes, p_shift metres), the
    // ones the worker makes from now on start from it.
    void set_render_origin(const ChunkId &p_origin, const Float3 &p_shift);

    void set_trim_indices(bool p_trim) { trim_indices = p_trim; }

    // Dual contouring of Hermite data (DC-lite) in the vertex pass: sharp UNION / SUBTRACT creases are
    // placed on the crease instead of the Surface Nets mass point (gpu_glsl.hpp). Set it before
    // initialize(). Both meshers expose it so VoxelTerrain does not branch on the pipeline.
    void set_dual_contouring(bool p_enable) { dual_contouring = p_enable; }

    ov::Value get_stats();
    // Main thread: the dispatch's sub-timers since the start (VoxelTerrain's sections).
    DispatchTimes get_dispatch_times() const {
        DispatchTimes t = dispatch_times;
        t.gate_admit = gate.admit_usec;
        return t;
    }
};
