#pragma once

#include "chunk_math.hpp"
#include <godot_cpp/core/object.hpp>

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include <godot_cpp/classes/thread.hpp>

#include "chunk_memory.hpp"
#include "debug_log.hpp"
#include "lod_structures.hpp"
#include "sdf_base.hpp"
#include <godot_cpp/classes/rendering_server.hpp>

using namespace godot;

struct UnpackChunk {
    ChunkId chunk_id;

    PackedVector3Array output_points;
    PackedVector3Array output_normals;
    PackedInt32Array output_triangles;
    // Per point: its global material (material_library.hpp), from the chunk's palette. A float vertex channel
    // (ARRAY_CUSTOM0, R float: ids < Material::MAX_MATERIALS are exact) the validator compares.
    PackedFloat32Array output_materials;
    std::vector<uint32_t> palette; // the chunk's palette (build_lists), main thread -> worker with the batch

    uint32_t point_offset = 0; // used to fill the array
    uint32_t index_count = 0; // indices actually written by the CPU triangulation

    HashMap<uint32_t, uint32_t> index_storage;

    PackedInt32Array quad_masks;
    // Cell (flat index) of each point. The triangulation must not derive it from the quantised position:
    // a vertex on its cell's upper face (local coordinate 1.0, a crossing at t = 1) floors into the next
    // cell, and its quads then take neighbour indices from the wrong cells.
    PackedInt32Array point_cells;

    RID mesh_rid;
    RID instance_rid;

    void resize(int point_number, int triangles_number) {
        output_points.resize(point_number);
        output_normals.resize(point_number);
        output_materials.resize(point_number);
        quad_masks.resize(point_number);
        point_cells.resize(point_number);

        output_triangles.resize(triangles_number * 3);
    }
};

// Legacy pipeline (pipeline_mode 1), kept as the correctness oracle of GPU Direct (MeshValidator)
// and as its perf baseline. A worker thread owns a local RenderingDevice and runs three compute
// passes per batch (SDF samples, edge crossings, one point per surface cell with its quad mask),
// reads everything back and triangulates on the CPU: one mesh + instance per chunk.
class CascadePasses;

class GPUMesher : public Object {
private:
    // Constant between run
    SDFBase *sdf;
    ChunkParam chunk_param;

    //worker
    Ref<Thread> worker;

    //worker availability
    Ref<Mutex> mutex;
    Ref<Semaphore> semaphore;
    bool worker_available = true;
    bool result_available = false;
    bool exit_thread = false;
    uint32_t batch_count = 0; // number of real chunks in the current batch (<= chunk_batch_number)
    int batch_min_lod = 0; // smallest LOD of the current batch (cascade levels to run)
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

    RID scenario;

    uint32_t chunk_batch_number;

    //input
    PackedByteArray input_chunk_id;
    PackedByteArray input_table_list; // the batch's instance list (archetype instances)
    PackedByteArray input_table_records; // the table's records changed since the last batch (record_bytes spans)

    // GPU stuff

    RenderingDevice *rd;

    // Buffers

    RID buffer_input_chunk_id;

    RID buffer_atomic_counter;
    RID buffer_sdf_sample;
    RID buffer_edge_intersect;

    RID buffer_output_point;
    RID buffer_chunk_param;
    // The instance table's buffers (archetype instances), on the local device.
    RID buffer_table_records;
    RID buffer_table_list;
    RID buffer_material; // per sample: its palette index (cascade level 0 -> third pass)
    uint64_t table_list_capacity = 0; // bytes
    uint32_t sent_table_version = UINT32_MAX; // main thread: the records version last handed to the worker
    std::vector<ChunkId> in_flight_ids; // main thread: the chunks of the batch handed to the worker
    // Uniform set

    RID uniform_set;

    RID shader_second_pass;
    RID shader_third_pass;

    RID pipeline_second_pass;
    RID pipeline_third_pass;

    // The cascade passes: pass 1 (the samples), in sub-batches of at most
    // CASCADE_MAX_CHUNKS chunks (its scratch, ~92 KB per chunk with 4 channels, is not sized for 1024).
    static constexpr uint32_t CASCADE_MAX_CHUNKS = 128;
    CascadePasses *cascade = nullptr;

    // One set per ChunkMath::load_queue: in the view cone first, then by LOD.
    std::vector<LODHashSet *> chunk_queue;
    int lod_count = 0;
    // Instance edits' re-meshes (loaded chunks), popped before chunk_queue. Kept apart because send_chunk
    // rebuilds chunk_queue from each scan's dirty chunks, which a loaded chunk isn't.
    std::vector<LODHashSet *> remesh_queue;

    LODHashSet working_generation;

    RID compile_dynamic_shader(const String &algorithm_pass, const String &p_name);

    void _execute_gpu_pipeline(PackedByteArray p_chunk_id, uint32_t p_count, int p_min_lod, const PackedByteArray &p_table_list,
            const PackedByteArray &p_table_records);
    void _execute_whole_pipeline();
    void _initialize_gpu_compute();

    bool _pop_queue(ChunkId &new_chunk);

    bool _is_available();

    Vector3i _unflatten_index_z(uint32_t flat_idx);
    uint32_t _get_flat_index_z(Vector3i cell);

    void render_chunk();

    std::vector<UnpackChunk> unpacked_chunks;

    std::vector<ChunkOutput> out_queue;

public:
    void initialize(SDFBase *p_sdf, ChunkParam p_chunk_param, uint32_t p_chunk_batch_number, int lod_numbers, RID p_scenario);

    void process_queue();

    // The chunks to mesh (replacing the queue), loaded in the order of ChunkMath::load_queue for p_view.
    void send_chunk(const LODHashSet &p_chunks, const ScanView &p_view);
    // Re-mesh chunks an instance edit changed (legacy path; the direct mesher has the versioned queue).
    void remesh_chunks(const std::vector<ChunkId> &p_chunks);

    bool get_results(VoxelState &output);

    // Stops the worker thread and frees the local RenderingDevice (on the worker, which owns it).
    void stop();

    bool is_idle();

    void set_trim_indices(bool p_trim) { trim_indices = p_trim; }

    // Dual contouring of Hermite data (DC-lite) in the vertex pass: sharp UNION / SUBTRACT creases are
    // placed on the crease instead of the Surface Nets mass point (gpu_glsl.hpp). Set it before
    // initialize(). Both meshers expose it so VoxelTerrain does not branch on the pipeline.
    void set_dual_contouring(bool p_enable) { dual_contouring = p_enable; }

    Dictionary get_stats();
};
