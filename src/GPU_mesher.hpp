#pragma once

#include <godot_cpp/core/object.hpp>
#include "chunk_math.hpp"

#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/semaphore.hpp>

#include <godot_cpp/classes/rendering_server.hpp>
#include"sdf_base.hpp"
#include "lod_structures.hpp"
#include "chunk_memory.hpp"
#include "debug_log.hpp"

using namespace godot;

struct UnpackChunk{

    ChunkId chunk_id;

    PackedVector3Array output_points;
    PackedVector3Array output_normals;
    PackedInt32Array output_triangles;

    uint32_t point_offset = 0; // used to fill the array
    uint32_t index_count = 0; // indices actually written by the CPU triangulation

    HashMap<uint32_t,uint32_t> index_storage;

    PackedInt32Array quad_masks;

    RID mesh_rid;
    RID instance_rid;

    void resize(int point_number, int triangles_number){
        output_points.resize(point_number);
        output_normals.resize(point_number);
        quad_masks.resize(point_number);

        output_triangles.resize(triangles_number*3);
    }

};

class GPUMesher : public Object{

    private :
    
    // Constant between run
    SDFBase* sdf;
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
    bool trim_indices = false; // drop the zero padding at the end of the index buffer (legacy bug)

    // Stats (written by the worker under mutex, read by get_stats()).
    struct Stats {
        TimingStat batch_total;   // whole worker iteration
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
    PackedInt32Array input_chunk_id;

    // GPU stuff

    RenderingDevice *rd;

    // Buffers

    RID buffer_input_chunk_id;

    RID buffer_atomic_counter;
    RID buffer_sdf_sample;
    RID buffer_edge_intersect;

    RID buffer_output_point;    
    RID buffer_chunk_param;
    // Uniform set

    
    RID uniform_set;

    //no need this now we use all at once.
    // RID uniform_set_input_chunk_id;

    // RID uniform_set_atomic_counter;
    // RID uniform_set_sdf_sample;
    // RID uniform_set_edge_intersect;

    // RID uniform_set_output_point;

    RID shader_first_pass;
    RID shader_second_pass;
    RID shader_third_pass;

    RID pipeline_first_pass;
    RID pipeline_second_pass;
    RID pipeline_third_pass;


    std::vector<LODHashSet *> chunk_queue;

    LODHashSet working_generation;

    RID compile_dynamic_shader(const String &custom_sdf_code);

    void _execute_gpu_pipeline(PackedByteArray p_chunk_id);
    void _execute_whole_pipeline();
    void _initialize_gpu_compute();

    bool _pop_queue(ChunkId &new_chunk);

    bool _is_available();

    Vector3i _unflatten_index_z(uint32_t flat_idx);
    uint32_t _get_flat_index_z(Vector3i cell);

    void render_chunk();

    std::vector<UnpackChunk> unpacked_chunks;

    std::vector<ChunkOutput> out_queue;
    std::vector<ChunkOutput> del_queue;

    public : 
    
    void initialize(SDFBase* p_sdf,ChunkParam p_chunk_param, uint32_t p_chunk_batch_number,int lod_numbers, RID p_scenario);

    void process_queue();

    void send_remove_chunk(LODHashSet p_chunks);

    void send_chunk(LODHashSet p_chunks);

    bool get_results(VoxelState &output);

    // Stops the worker thread and frees the local RenderingDevice (on the worker, which owns it).
    void stop();

    bool is_idle();

    void set_trim_indices(bool p_trim) { trim_indices = p_trim; }

    Dictionary get_stats();

    

};

/*
Algorithm will be quite different because the different passes will all be brute force:

Pass 1 (Vertex): Determine the sdf sample

**compute list barrier**

Pass 2, 3, 4 (Edge): Determine X/Y/Z edge crossing

**compute list barrier**

Pass 5 (Cell): QEF/ point - normals + Quad_metadata


Variable and compression:

GLOBAL

global_point_atomic_counter uint32
global_triangle_atomic_counter uint32

PER_CHUNK (assume that batch fit in 65,535 and doesnt overflow the global atomic counter)

SDF_sample DENSE (chunk_size+1)^3 : float32

edge_intersect X/Y/Z DENSE (chunk_size-1) * chunk_size^2 : 3 float32

per_chunk_point_atomic_counter uint32
per_chunk_triangle_atomic_counter uint32

point output :

    output_point SPARSE (chunk_size)^3 : uint8 x 3 (compression, local position inside the cell)
    output_normal SPARSE (chunk_size)^3 : uint8 x 3 (packSnorm4x8)
    output_point_metadata SPARSE (chunk_size)^3 : uint32 (address of the point) + uint16** (address of the chunk) + 3bitmask (triangle metadata)

    total SPARSE (chunk_size)^3 : 4 x uint32

** Probably less than 16 can use the end to store other data

*/