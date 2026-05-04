#pragma once

#include <godot_cpp/core/object.hpp>
#include "chunk_math.hpp"
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include"sdf_base.hpp"

using namespace godot;

class GPUMesher : public Object{

    private :
    
    // Constant between run
    SDFBase* sdf;
    ChunkParam chunk_param;
    Ref<Thread> worker;

    uint32_t chunk_batch_number;

    RenderingDevice *rd;

    // Buffers

    RID buffer_input_chunk_id;

    RID buffer_atomic_counter;
    RID buffer_sdf_sample;
    RID buffer_edge_intersect;

    RID buffer_output_point;

    // Uniform set

    RID uniform_set_input_chunk_id;

    RID uniform_set_atomic_counter;
    RID uniform_set_sdf_sample;
    RID uniform_set_edge_intersect;

    RID uniform_set_output_point;



    void _execute_pipeline();

    public : 
    
    void initialize(SDFBase* p_sdf,ChunkParam p_chunk_param, uint32_t p_chunk_batch_number);

    RID compile_dynamic_shader(const String &custom_sdf_code);

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