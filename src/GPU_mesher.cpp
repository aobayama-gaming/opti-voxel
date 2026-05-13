#include "GPU_mesher.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>

//debug only
#include <godot_cpp/classes/scene_tree.hpp>


static constexpr const char GLSL_STRUCT_DEFINITION[] = R"glsl(

#ifndef WORLD_NUDGE
#define WORLD_NUDGE 0.0 // Define this to your desired nudge value or pass as uniform
#endif

struct PointMetadata { int a; int b; int c; int d; };

// Equivalent to your C++ ChunkParam class
struct ChunkParam {
    int chunk_size;
    float voxel_size;
    int skirt_size;
    int use_skirt; // nearly useless but need the padding
};

float chunk_voxel_dimension(ChunkParam param) {
    return float(param.chunk_size - param.skirt_size * 2);
}

float chunk_world_dimension(ChunkParam param) {
    return chunk_voxel_dimension(param) * param.voxel_size;
}

)glsl";

static constexpr const char GLSL_HELPER_FUNCTION[] =  R"glsl(

// ---------------------------------------------------------
// Bitwise & LOD functions
// ---------------------------------------------------------

int get_parent_from_child(int coord, out int lod) {
    int lsb_index = findLSB(coord);
    if (lsb_index == -1) { // findLSB returns -1 when coord is 0
        lod = 0;
        return 0; 
    }

    lod = lsb_index + 1;
    
    int cleared_coord = coord & ~(1 << lsb_index);
    int parent_coord = cleared_coord | (1 << (lsb_index + 1));

    return parent_coord;
}

int get_parent_from_child(int coord) {
    int dummy_lod;
    return get_parent_from_child(coord, dummy_lod);
}

ivec3 get_parent_from_child(ivec3 coord, out int lod) {
    // Evaluated sequentially to guarantee safe `out` parameter assignment
    int px = get_parent_from_child(coord.x, lod);
    int py = get_parent_from_child(coord.y);
    int pz = get_parent_from_child(coord.z);
    return ivec3(px, py, pz);
}

ivec3 get_parent_from_child(ivec3 coord) {
    return ivec3(
        get_parent_from_child(coord.x),
        get_parent_from_child(coord.y),
        get_parent_from_child(coord.z)
    );
}

int get_lod(ivec3 coord) {
    int lod_x = findLSB(coord.x);
    if (lod_x == -1) {
        return 0; 
    }
    return lod_x;
}

int get_lod_size(ivec3 coord) {
    return (1 << get_lod(coord));
}

ivec3 get_parent_from_child_until(ivec3 coord, int target_lod) {
    ivec3 parent = coord;
    int current_lod = -1;

    while (current_lod < target_lod) {
        parent = get_parent_from_child(parent, current_lod);
    }

    return parent;
}

// ---------------------------------------------------------
// World <-> Chunk Transforms
// ---------------------------------------------------------

float world_chunk_size(ivec3 chunk_coord, ChunkParam chunk_param) {
    int lod = get_lod(chunk_coord);
    return float(1 << lod) * chunk_world_dimension(chunk_param); 
}

float world_chunk_spherical_radius(ivec3 chunk_coord, ChunkParam chunk_param) {
    float half_size = world_chunk_size(chunk_coord, chunk_param) * 0.5;
    return sqrt(half_size * half_size * 3.0); 
}

vec3 chunk_to_world(ivec3 chunk_coord, ChunkParam chunk_param) {
    return vec3(chunk_coord) * (chunk_world_dimension(chunk_param) * 0.5) + WORLD_NUDGE;
}

ivec3 world_to_chunk(vec3 world_coord, ChunkParam chunk_param) {
    ivec3 cell = ivec3(floor(world_coord / chunk_world_dimension(chunk_param)));
    return cell * 2 + 1;
}

// ---------------------------------------------------------
// Vertex <-> World Transforms
// ---------------------------------------------------------

float direct_transform(float inputValue, float chunk_dim, ChunkParam chunk_param) {
    return (inputValue - float(chunk_param.skirt_size)) * chunk_dim / chunk_voxel_dimension(chunk_param);
}

float inverse_transform(float inputValue, float chunk_dim, ChunkParam chunk_param) {
    return inputValue * chunk_voxel_dimension(chunk_param) / chunk_dim + float(chunk_param.skirt_size);
}

vec3 vertices_to_world(ivec3 chunk_coord, vec3 local_coord, ChunkParam chunk_param) {
    float chunk_size = world_chunk_size(chunk_coord, chunk_param);
    vec3 chunk_center = chunk_to_world(chunk_coord, chunk_param);
    vec3 chunk_min_corner = chunk_center - vec3(chunk_size * 0.5);

    return chunk_min_corner + vec3(
        direct_transform(local_coord.x, chunk_size, chunk_param),
        direct_transform(local_coord.y, chunk_size, chunk_param),
        direct_transform(local_coord.z, chunk_size, chunk_param)
    );
}

vec3 world_to_vertices(ivec3 chunk_coord, vec3 world_coord, ChunkParam chunk_param) {
    float chunk_size = world_chunk_size(chunk_coord, chunk_param);
    vec3 chunk_center = chunk_to_world(chunk_coord, chunk_param);
    vec3 chunk_min_corner = chunk_center - vec3(chunk_size * 0.5);
    
    vec3 local_offset = world_coord - chunk_min_corner;
    return vec3(
        inverse_transform(local_offset.x, chunk_size, chunk_param),
        inverse_transform(local_offset.y, chunk_size, chunk_param),
        inverse_transform(local_offset.z, chunk_size, chunk_param)
    );
}

bool vertices_out_of_bound(vec3 vertices_coord, ChunkParam chunk_param) {
    // Utilizes GLSL vector relational functions for a cleaner check
    return any(lessThan(vertices_coord, vec3(0.0))) || 
           any(greaterThan(vertices_coord, vec3(chunk_world_dimension(chunk_param))));
}

)glsl";

static constexpr const char GLSL_BUFFER_BINDING[] = R"glsl(
layout(set = 0, binding = 0, std430) restrict buffer AtomicCounter {
    uint atomic_counter[];
};

layout(set = 0, binding = 1, std430) restrict buffer InputChunk {
    ivec3 input_chunk[];
};

layout(set = 0, binding = 2, std430) restrict buffer SDFSample {
    vec4 sdf_sample[];
};

layout(set = 0, binding = 3, std430) restrict buffer EdgeIntersect {
    float edge_intersect[];
};

layout(set = 0, binding = 4, std430) restrict buffer OutputPoint {
    PointMetadata output_point[];
};

layout(set = 0, binding = 5, std140) uniform ChunkParamUniform {
    ChunkParam chunk_param;
};

)glsl";

RID GPUMesher::compile_dynamic_shader(const String &algorithm_pass) {
    
    // 1. Define your base Compute Shader using a C++ Raw String
    String shader = R"glsl(#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

/*GLSL_STRUCT_DEFINITION*/

/*GLSL_BUFFER_BINDING*/

/*GLSL_HELPER_FUNCTION*/

//Will inject vec4 terrain_sdf(vec3 pos)
/*GLSL_TERRAIN_SDF*/

// A placeholder function that we will inject code into!
vec4 evaluate_sdf(vec3 pos) {

    //injected function
    return terrain_sdf(pos);

}

void main() {

    ivec3 chunk_id = input_chunk[gl_GlobalInvocationID.z/chunk_param.chunk_size];

    ivec3 cell_coordinate = ivec3(
        gl_GlobalInvocationID.x,
        gl_GlobalInvocationID.y,
        gl_GlobalInvocationID.z%chunk_param.chunk_size
        );

    uint flat_index = gl_GlobalInvocationID.x + 
        (gl_GlobalInvocationID.y+ gl_GlobalInvocationID.z*chunk_param.chunk_size)*chunk_param.chunk_size;

    /*GLSL_ALGORITHM_PASS*/
}
    )glsl";

    // 2. Inject the dynamic code
    shader = shader.replace("/*GLSL_STRUCT_DEFINITION*/",GLSL_STRUCT_DEFINITION);
    shader = shader.replace("/*GLSL_BUFFER_BINDING*/",GLSL_BUFFER_BINDING);
    shader = shader.replace("/*GLSL_HELPER_FUNCTION*/","");
    shader = shader.replace("/*GLSL_TERRAIN_SDF*/",sdf->get_glsl_code());
    shader = shader.replace("/*GLSL_ALGORITHM_PASS*/", algorithm_pass);

    // 3. Create a Shader Source object
    Ref<RDShaderSource> shader_source;
    shader_source.instantiate();
    
    // Tell Godot this is a Compute Shader
    shader_source->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, shader);

    // 4. Compile the GLSL to SPIR-V bytecode at runtime!
    Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(shader_source);

    // 5. CRITICAL: Check for compilation errors
    String error_msg = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!error_msg.is_empty()) {
        UtilityFunctions::printerr("--- SDF SHADER COMPILATION FAILED ---");
        UtilityFunctions::printerr("ERROR MESSAGE:\n", error_msg);
        UtilityFunctions::printerr("-------------------------------------");
        UtilityFunctions::printerr("SHADER SOURCE CODE:\n");
        
        // Print shader code with line numbers
        PackedStringArray lines = shader.split("\n");
        for (int i = 0; i < lines.size(); i++) {
            UtilityFunctions::printerr(String::num_int64(i + 1) + " | " + lines[i]);
        }
        
        UtilityFunctions::printerr("-------------------------------------");

        return RID(); // Return invalid RID
    }

    // 6. Create the final shader RID
    RID shader_rid = rd->shader_create_from_spirv(spirv);
    
    return shader_rid;
}

static constexpr const char GLSL_FIRST_PASS[] = R"glsl(

vec3 

sdf_sample[flat_index] = vec4(float(chunk_id.x), float(chunk_id.y), float(chunk_id.z),0.0);

)glsl";


void GPUMesher::initialize(
    SDFBase* p_sdf,
    ChunkParam p_chunk_param, 
    uint32_t p_chunk_batch_number,
    int lod_numbers
){

    sdf = p_sdf;
    chunk_param = p_chunk_param;
    chunk_batch_number = p_chunk_batch_number;

    // Worker stuff

    worker.instantiate();
    mutex.instantiate();
    semaphore.instantiate();

    worker->start(callable_mp(this,&GPUMesher::_execute_whole_pipeline),Thread::PRIORITY_NORMAL);

    chunk_queue.resize(lod_numbers, nullptr);

    for (int i = 0; i < lod_numbers; ++i) {
        chunk_queue[i] = memnew(LODHashSet);
    }

};

void GPUMesher::_initialize_gpu_compute(){

    //alocate memory

    rd = RenderingServer::get_singleton()->create_local_rendering_device();

    PackedByteArray empty_data;

    buffer_input_chunk_id = rd->storage_buffer_create((chunk_batch_number)*sizeof(Vector4i),empty_data); // need to be Vector4i due to padding...

    buffer_atomic_counter = rd->storage_buffer_create(2*(1+chunk_batch_number)*sizeof(uint32_t),empty_data);
    
    buffer_sdf_sample = rd->storage_buffer_create(chunk_batch_number*chunk_param.chunk_size*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(Vector4),empty_data);

    buffer_edge_intersect =rd->storage_buffer_create((chunk_param.chunk_size-1)*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(float_t)*3,empty_data);
    /*
    Compressed position uint8 x 3
    Compressed normal uint8 x 3 
    Address of point uint32
    Address of chunk + quad 3bit mask 16bit
    
    Total of 3 uint32_t
    */
    buffer_output_point = rd->storage_buffer_create(3*chunk_param.chunk_size*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(uint32_t),empty_data);

    // Injection of Chunk Param in the uniform
    // Match the std140 layout for ChunkParam: 4 x 4-byte fields = 16 bytes
    PackedByteArray param_data;
    param_data.resize(16);
    
    param_data.encode_s32(0, chunk_param.chunk_size);
    param_data.encode_float(4, chunk_param.voxel_size);
    param_data.encode_s32(8, chunk_param.skirt_size);
    param_data.encode_s32(12, chunk_param.use_skirt ? 1 : 0);
    
    buffer_chunk_param = rd->uniform_buffer_create(param_data.size(), param_data);

    // Helper lambda to keep code clean
    auto create_storage_uniform =[](int binding, RID buffer,RenderingDevice::UniformType type = RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER) -> Ref<RDUniform> {
        Ref<RDUniform> uniform;
        uniform.instantiate();
        uniform->set_uniform_type(type);
        uniform->set_binding(binding);
        uniform->add_id(buffer);
        return uniform;
    };

    TypedArray<RDUniform> uniforms;
    // 1. Create the Uniforms mapping to GLSL binding points (0 to 6)
    
    uniforms.push_back(create_storage_uniform(0, buffer_atomic_counter));
    uniforms.push_back(create_storage_uniform(1, buffer_input_chunk_id));
    uniforms.push_back(create_storage_uniform(2, buffer_sdf_sample));
    uniforms.push_back(create_storage_uniform(3, buffer_edge_intersect));
    uniforms.push_back(create_storage_uniform(4, buffer_output_point));
    uniforms.push_back(create_storage_uniform(5, buffer_chunk_param, RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER));
    
    
    // 2. Create the Uniform Set (CACHE THIS RID!)
    // 'shader_rid' is the RID returned when you compiled your GLSL.
    // '0' is the layout(set = 0) index.
    // RID master_uniform_set = rd->uniform_set_create(uniforms, shader_rid, 0);

    shader_first_pass = compile_dynamic_shader(GLSL_FIRST_PASS);

    //since we assume all uniform for all the pass we do this
    uniform_set = rd->uniform_set_create(uniforms,shader_first_pass,0);

}

void GPUMesher::_execute_whole_pipeline(){

    _initialize_gpu_compute();
    UtilityFunctions::print("GPUMesher : Worker started");

    while(true){
    
    semaphore->wait();
    //UtilityFunctions::print("GPUMesher : pipeline started");
    mutex->lock();
    PackedByteArray local_chunk_id = input_chunk_id.to_byte_array();
    mutex->unlock();

    _execute_gpu_pipeline(local_chunk_id);

    //render_chunk();

    //debug pipeline
    //render_chunk_box();

    mutex->lock();
    worker_available=true;
    result_available=true;
    mutex->unlock();

    //UtilityFunctions::print("GPUMesher : pipeline ended");
    }

}

void GPUMesher::_execute_gpu_pipeline(PackedByteArray p_chunk_id){

    // 1. RESET ATOMIC COUNTERS TO 0
    PackedByteArray counter_zeros;
    counter_zeros.resize(2*(1+chunk_batch_number) * sizeof(uint32_t));
    counter_zeros.fill(0);
    rd->buffer_update(buffer_atomic_counter, 0, counter_zeros.size(), counter_zeros);

    // 2. put the input inside

    rd->buffer_update(buffer_input_chunk_id, 0,p_chunk_id.size(),p_chunk_id);

    //first step pipeline
    RID first_pass = rd->compute_pipeline_create(shader_first_pass);

    // --- GPU SUBMISSION PIPELINE ---
    int64_t compute_list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(compute_list,first_pass);
    rd->compute_list_bind_uniform_set(compute_list,uniform_set,0); // bind for all pass actually

    uint32_t dispatch_size = chunk_param.chunk_size/8;

    rd->compute_list_dispatch(compute_list,dispatch_size,dispatch_size,dispatch_size*chunk_batch_number); // Arise the requirement of chunk 8 secable
    rd->compute_list_end();

    rd->submit();
    rd->sync();

    //debug retrieve sdf value
    PackedByteArray debug_sdf;
    debug_sdf = rd->buffer_get_data(buffer_sdf_sample);

    PackedFloat32Array untangle_sdf;
    untangle_sdf = debug_sdf.to_float32_array();

    UtilityFunctions::print("Output: ",untangle_sdf);

}

bool GPUMesher::_is_available(){

    bool available;

    mutex->lock();
    available = worker_available;
    //available &= !result_available; // since result is an array, change behavior from CPU
    mutex->unlock();

    //UtilityFunctions::print("GPUMesher : status = ",available);

    return available;

}

void GPUMesher::process_queue(){

    if(!_is_available()) return;

    PackedInt32Array new_batch;
    new_batch.resize(chunk_batch_number * 4); // 4 int32 per ivec3 for std430 alignment

    int count = 0;
    Vector3i new_chunk;

    //UtilityFunctions::print("GPUMesher : prepare for ",chunk_batch_number);
    
    while(count < chunk_batch_number && _pop_queue(new_chunk)){

        //UtilityFunctions::print("Dispatching chunk: ", new_chunk);

        working_generation.insert(new_chunk);
        
        new_batch[count * 4 + 0] = new_chunk.x;
        new_batch[count * 4 + 1] = new_chunk.y;
        new_batch[count * 4 + 2] = new_chunk.z;
        new_batch[count * 4 + 3] = 0; // padding
        
        count++;
    }

    if(count > 0) {
        mutex->lock();
        input_chunk_id = new_batch;
        worker_available = false;
        mutex->unlock();
        
        semaphore->post(); // wake up the worker thread
        //UtilityFunctions::print("GPUMesher : Semaphore posted.");
    }

}

bool GPUMesher::_pop_queue(Vector3i &new_chunk) {
    for (LODHashSet *queue : chunk_queue) {
        if (queue == nullptr || queue->is_empty()) {
            continue;
        }

        LODHashSet::Iterator it = queue->begin();
        if (!it) {
            continue;
        }

        Vector3i chunk_pos = *it;
        queue->remove(it);
        new_chunk = chunk_pos;
        return true;
    }

    return false;
}

void GPUMesher::send_chunk(LODHashSet p_chunks){

    for (LODHashSet *q : chunk_queue) {
        if (q) {
            q->clear();
        }
    }

    for (const Vector3i &chunk_pos : p_chunks) {

    UtilityFunctions::print("GPUMesher : Adding ",chunk_pos);

    const int lod = ChunkMath::get_lod(chunk_pos);
    if (lod < 0 || lod >= static_cast<int>(chunk_queue.size())) {
        continue;
    }

    LODHashSet *q = chunk_queue[lod];
    if (q && !working_generation.has(chunk_pos)) {
        q->insert(chunk_pos);
    }
    }

}
