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

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>

#include "godot_cpp/templates/hash_map.hpp"

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/surface_tool.hpp>

#include "noise.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"

//debug only
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/time.hpp>




static constexpr const char GLSL_BUFFER_BINDING[] = R"glsl(
layout(set = 0, binding = 0, std430) restrict buffer AtomicCounter {
    uint atomic_counter[];
};

layout(set = 0, binding = 1, std430) restrict buffer InputChunk {
    ChunkInput input_chunk[];
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

uint get_flat_index_z(uvec3 cell_coordinate) {
    return cell_coordinate.x + (cell_coordinate.y * chunk_param.chunk_size) + (cell_coordinate.z * chunk_param.chunk_size * chunk_param.chunk_size);
}

uint get_edge_index(uint chunk_idx, uint axis, uvec3 local_c) {
    uint cs = chunk_param.chunk_size;
    uint local_idx = local_c.x + (local_c.y * cs) + (local_c.z * cs * cs);
    
    // Memory Layout: [Chunk Batch] -> [Axis 0/1/2] -> [Local Index]
    return (chunk_idx * 3 * cs * cs * cs) + (axis * cs * cs * cs) + local_idx;
}

)glsl";

RID GPUMesher::compile_dynamic_shader(const String &algorithm_pass) {
    
    // 1. Define your base Compute Shader using a C++ Raw String
    String shader = R"glsl(#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

/*GLSL_STRUCT_DEFINITION*/

/*GLSL_BUFFER_BINDING*/

/*GLSL_HELPER_FUNCTION*/

//Will inject vec4 terrain_sdf(WorldPos p)
/*GLSL_TERRAIN_SDF*/

vec4 evaluate_sdf(WorldPos p) {
    return terrain_sdf(p);
}

/*GLSL_SKIRT_SDF*/

void main() {



    ChunkFrame chunk = chunk_frame(input_chunk[gl_GlobalInvocationID.z/chunk_param.chunk_size], chunk_param);
    ivec3 chunk_id = chunk.id_lo;

    ivec3 cell_chunk_coordinate = ivec3(
        gl_GlobalInvocationID.x,
        gl_GlobalInvocationID.y%chunk_param.chunk_size, // x / y / z batching
        gl_GlobalInvocationID.z%chunk_param.chunk_size // chunk batching
        );

    /*GLSL_ALGORITHM_PASS*/
}
    )glsl";

    // 2. Inject the dynamic code
    shader = shader.replace("/*GLSL_STRUCT_DEFINITION*/", String(GLSL_STRUCT_DEFINITION) + String(GLSL_FRAME_STRUCTS));
    shader = shader.replace("/*GLSL_BUFFER_BINDING*/",GLSL_BUFFER_BINDING);
    shader = shader.replace("/*GLSL_HELPER_FUNCTION*/", String(GLSL_HELPER_FUNCTION) + String(GLSL_NOISE) + GPUFrame::glsl_functions());
    shader = shader.replace("/*GLSL_SKIRT_SDF*/", GLSL_SKIRT_SDF);
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

sdf_sample[get_flat_index_z(gl_GlobalInvocationID)] = skirt_sdf(chunk,vec3(cell_chunk_coordinate),chunk_param);

)glsl";

static constexpr const char GLSL_SECOND_PASS[] = R"glsl(

uint axis = gl_GlobalInvocationID.y / chunk_param.chunk_size;

if (cell_chunk_coordinate[axis] < chunk_param.chunk_size - 1) {

    // global_cell is used ONLY for safely reading from the SDF buffer
    uvec3 global_cell = uvec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y % chunk_param.chunk_size, gl_GlobalInvocationID.z);

    uvec3 offset = uvec3(0);
    offset[axis] = 1;

    vec4 center_sdf = sdf_sample[get_flat_index_z(global_cell)];
    vec4 offset_sdf = sdf_sample[get_flat_index_z(global_cell + offset)];

    if ((center_sdf.w > 0.0) != (offset_sdf.w > 0.0)) {
        float t = center_sdf.w / (center_sdf.w - offset_sdf.w + 0.00001);
        
        // Extract chunk batch to safely write edge data
        uint chunk_idx = gl_GlobalInvocationID.z / chunk_param.chunk_size;
        
        // Write using the strictly local cell_chunk_coordinate
        edge_intersect[get_edge_index(chunk_idx, axis, uvec3(cell_chunk_coordinate))] = t;
    }
}

)glsl";

static constexpr const char GLSL_THIRD_PASS[] = R"glsl(

if (all(lessThan(cell_chunk_coordinate, ivec3(chunk_param.chunk_size - 1)))) {

uint chunk_idx = gl_GlobalInvocationID.z / chunk_param.chunk_size;

// Helper to construct a local grid coordinate for neighbor cells
#define offset(x, y, z) (uvec3(cell_chunk_coordinate) + uvec3(x, y, z))

// --- X-Axis Edges ---
float e0_t = edge_intersect[get_edge_index(chunk_idx, 0, offset(0,0,0))];
float e2_t = edge_intersect[get_edge_index(chunk_idx, 0, offset(0,1,0))];
float e4_t = edge_intersect[get_edge_index(chunk_idx, 0, offset(0,0,1))];
float e6_t = edge_intersect[get_edge_index(chunk_idx, 0, offset(0,1,1))];

// --- Y-Axis Edges ---
float e3_t = edge_intersect[get_edge_index(chunk_idx, 1, offset(0,0,0))];
float e1_t = edge_intersect[get_edge_index(chunk_idx, 1, offset(1,0,0))];
float e7_t = edge_intersect[get_edge_index(chunk_idx, 1, offset(0,0,1))];
float e5_t = edge_intersect[get_edge_index(chunk_idx, 1, offset(1,0,1))];

// --- Z-Axis Edges ---
float e8_t = edge_intersect[get_edge_index(chunk_idx, 2, offset(0,0,0))];
float e9_t = edge_intersect[get_edge_index(chunk_idx, 2, offset(1,0,0))];
float e11_t = edge_intersect[get_edge_index(chunk_idx, 2, offset(0,1,0))];
float e10_t = edge_intersect[get_edge_index(chunk_idx, 2, offset(1,1,0))];

//storing results

vec3 local_vertex = vec3(0.0);
int active_edge_count = 0;

// --- X-Axis Edges (varying X, static Y/Z) ---
if (e0_t > 0.0) { local_vertex += vec3(e0_t, 0.0, 0.0); active_edge_count++; }
if (e2_t > 0.0) { local_vertex += vec3(e2_t, 1.0, 0.0); active_edge_count++; }
if (e4_t > 0.0) { local_vertex += vec3(e4_t, 0.0, 1.0); active_edge_count++; }
if (e6_t > 0.0) { local_vertex += vec3(e6_t, 1.0, 1.0); active_edge_count++; }

// --- Y-Axis Edges (varying Y, static X/Z) ---
if (e3_t > 0.0) { local_vertex += vec3(0.0, e3_t, 0.0); active_edge_count++; }
if (e1_t > 0.0) { local_vertex += vec3(1.0, e1_t, 0.0); active_edge_count++; }
if (e7_t > 0.0) { local_vertex += vec3(0.0, e7_t, 1.0); active_edge_count++; }
if (e5_t > 0.0) { local_vertex += vec3(1.0, e5_t, 1.0); active_edge_count++; }

// --- Z-Axis Edges (varying Z, static X/Y) ---
if (e8_t > 0.0) { local_vertex += vec3(0.0, 0.0, e8_t); active_edge_count++; }
if (e9_t > 0.0) { local_vertex += vec3(1.0, 0.0, e9_t); active_edge_count++; }
if (e11_t > 0.0) { local_vertex += vec3(0.0, 1.0, e11_t); active_edge_count++; }
if (e10_t > 0.0) { local_vertex += vec3(1.0, 1.0, e10_t); active_edge_count++; }

// --- Calculate Cell Centroid ---
if (active_edge_count > 0) {
    // Average the positions to find the center of mass
    local_vertex /= float(active_edge_count);
    
    // 1. Pack Position (3 x uint8 = 24 bits)
    // Convert 0.0->1.0 to 0->255
    uvec3 byte_pos = uvec3(local_vertex * 255.0);
    uint packed_pos = byte_pos.x | (byte_pos.y << 8) | (byte_pos.z << 16);

    // 2. Pack Normal (using built in packSnorm4x8)
    // Find the exact world position of our local_vertex
    
    // Resample the SDF to get the exact normal at this vertex

    vec3 chunk_vertex = local_vertex+vec3(cell_chunk_coordinate);

    vec3 surface_normal = normalize(skirt_sdf(chunk,chunk_vertex,chunk_param).xyz);
    uint packed_normal = packSnorm4x8(vec4(surface_normal, 0.0));

    // 3. Triangle Metadata (3 bit mask for X/Y/Z quad generation in Dual Contouring/Surface nets)
    // We check if edges 0, 3, and 8 have intersections to know if this cell spawns a quad
    uint quad_mask = 0;
    if (e0_t > 0.0) quad_mask |= 1; // X-axis crossing quad
    if (e3_t > 0.0) quad_mask |= 2; // Y-axis crossing quad
    if (e8_t > 0.0) quad_mask |= 4; // Z-axis crossing quad

    if (sdf_sample[get_flat_index_z(gl_GlobalInvocationID)].w < 0.0) {
        quad_mask |= 8; // Store "Origin is Solid" in the 4th bit!
    }

    // 4. Update Atomic Counters    
    // Get sparse index from global counter
    uint global_point_idx = atomicAdd(atomic_counter[0], 1);
    

    // Increment the individual chunk counter (Index 2 + 2 * chunk_id)
    uint local_chunk_point_idx = atomicAdd(atomic_counter[2 + 2 * chunk_idx], 1);

    // Increment triangle counters (2 triangles per quad)
    uint quad_count = bitCount(quad_mask);
    if (quad_count > 0) {
        atomicAdd(atomic_counter[1], quad_count * 2);
        atomicAdd(atomic_counter[3 + 2 * chunk_idx], quad_count * 2);
    }

    // 5. Pack Metadata (Global Address + Chunk Address + Quad Mask)
    // We have effectively 4 integers in PointMetadata based on your struct: {a, b, c, d}
    // Note: If you reduce PointMetadata to 3 ints in C++, adjust this assignment.
    
    output_point[global_point_idx].a = int(packed_pos);
    output_point[global_point_idx].b = int(packed_normal);
    
    // Store chunk index and mask in 'c' 
    output_point[global_point_idx].c = int((chunk_idx & 0xFFFF) | (quad_mask << 16));
    
    // Store localized point address in 'd'
    output_point[global_point_idx].d = int(get_flat_index_z(cell_chunk_coordinate));
}


}

)glsl";

void GPUMesher::initialize(
    SDFBase* p_sdf,
    ChunkParam p_chunk_param, 
    uint32_t p_chunk_batch_number,
    int lod_numbers,
    RID p_scenario
){

    sdf = p_sdf;
    chunk_param = p_chunk_param;
    chunk_batch_number = p_chunk_batch_number;
    scenario = p_scenario;

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

    buffer_input_chunk_id = rd->storage_buffer_create(chunk_batch_number * GPUFrame::CHUNK_INPUT_BYTES, empty_data);

    buffer_atomic_counter = rd->storage_buffer_create(2*(1+chunk_batch_number)*sizeof(uint32_t),empty_data);
    
    buffer_sdf_sample = rd->storage_buffer_create(chunk_batch_number*chunk_param.chunk_size*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(Vector4),empty_data);

    buffer_edge_intersect =rd->storage_buffer_create(chunk_param.chunk_size*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(float_t)*3*chunk_batch_number,empty_data);
    //to simplify debug erase the optimise allocation
    
    /*
    Compressed position uint8 x 3
    Compressed normal uint8 x 3 
    Address of point uint32
    Address of chunk + quad 3bit mask 16bit
    
    Total of 3 uint32_t
    */
    buffer_output_point = rd->storage_buffer_create(4*chunk_param.chunk_size*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(uint32_t)*chunk_batch_number,empty_data);

    // Injection of Chunk Param in the uniform
    // Match the std140 layout for ChunkParam: 4 x 4-byte fields = 16 bytes
    const PackedByteArray param_data = GPUFrame::chunk_param_bytes(chunk_param);
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
    shader_second_pass = compile_dynamic_shader(GLSL_SECOND_PASS);
    shader_third_pass = compile_dynamic_shader(GLSL_THIRD_PASS);

    //first step pipeline
    pipeline_first_pass = rd->compute_pipeline_create(shader_first_pass);
    pipeline_second_pass = rd->compute_pipeline_create(shader_second_pass);
    pipeline_third_pass = rd->compute_pipeline_create(shader_third_pass);

    //since we assume all uniform for all the pass we do this
    uniform_set = rd->uniform_set_create(uniforms,shader_first_pass,0);

}

void GPUMesher::_execute_whole_pipeline(){

    _initialize_gpu_compute();
    VLOG(1, "GPUMesher : Worker started");

    while(true){
    
    semaphore->wait();
    //UtilityFunctions::print("GPUMesher : pipeline started");
    mutex->lock();
    const bool should_exit = exit_thread;
    PackedByteArray local_chunk_id = input_chunk_id;
    mutex->unlock();

    if (should_exit) {
        break;
    }

    const uint64_t t_batch_start = now_usec();
    _execute_gpu_pipeline(local_chunk_id);

    mutex->lock();
    stats.batch_total.add(now_usec() - t_batch_start);
    mutex->unlock();

    //render_chunk();

    //debug pipeline
    //render_chunk_box();

    mutex->lock();
    worker_available=true;
    result_available=true;
    mutex->unlock();

    //UtilityFunctions::print("GPUMesher : pipeline ended");
    }

    // The local device must be freed on the thread that created it.
    RID resources[] = {
        uniform_set, pipeline_first_pass, pipeline_second_pass, pipeline_third_pass,
        shader_first_pass, shader_second_pass, shader_third_pass,
        buffer_input_chunk_id, buffer_atomic_counter, buffer_sdf_sample,
        buffer_edge_intersect, buffer_output_point, buffer_chunk_param
    };
    for (const RID &rid : resources) {
        if (rid.is_valid()) {
            rd->free_rid(rid);
        }
    }
    memdelete(rd);
    rd = nullptr;
    VLOG(1, "GPUMesher : Worker stopped");
}

void GPUMesher::stop() {
    if (worker.is_null()) {
        return;
    }
    mutex->lock();
    exit_thread = true;
    mutex->unlock();
    semaphore->post();
    worker->wait_to_finish();
    worker.unref();

    // Free the meshes of results that were produced but never collected.
    for (ChunkOutput &out : out_queue) {
        out.free();
    }
    out_queue.clear();

    for (LODHashSet *q : chunk_queue) {
        memdelete(q);
    }
    chunk_queue.clear();
}

bool GPUMesher::is_idle() {
    mutex->lock();
    const bool idle = worker_available && !result_available;
    mutex->unlock();
    return idle;
}

Dictionary GPUMesher::get_stats() {
    Dictionary d;
    mutex->lock();
    d["batch_total"] = stats.batch_total.to_dict();
    d["gpu_submit_sync"] = stats.gpu_submit_sync.to_dict();
    d["points_fetch"] = stats.points_fetch.to_dict();
    d["points_process"] = stats.points_process.to_dict();
    d["triangulate"] = stats.triangulate.to_dict();
    d["render_upload"] = stats.render_upload.to_dict();
    d["chunks_meshed"] = (int64_t)stats.chunks_meshed;
    d["points"] = (int64_t)stats.points;
    d["triangles"] = (int64_t)stats.triangles;
    d["queued"] = (int64_t)[this]() { int64_t n = 0; for (LODHashSet *q : chunk_queue) { n += q->size(); } return n; }();
    d["in_flight"] = (int64_t)(worker_available ? 0 : batch_count);
    mutex->unlock();
    return d;
}



Vector3i GPUMesher::_unflatten_index_z(uint32_t flat_idx) {
    uint32_t x = flat_idx % chunk_param.chunk_size;
    uint32_t y = (flat_idx / chunk_param.chunk_size) % chunk_param.chunk_size;
    uint32_t z = flat_idx / (chunk_param.chunk_size * chunk_param.chunk_size);
    return Vector3i(x, y, z);
}

uint32_t GPUMesher::_get_flat_index_z(Vector3i cell) {
    return cell.x + cell.y * chunk_param.chunk_size + cell.z * chunk_param.chunk_size * chunk_param.chunk_size;
}

void GPUMesher::_execute_gpu_pipeline(PackedByteArray p_chunk_id){

    // 1. RESET ATOMIC COUNTERS TO 0
    rd->buffer_clear(buffer_atomic_counter, 0, 2*(1+chunk_batch_number) * sizeof(uint32_t));

    // Erase the active edges buffer to 0.0 
    // (So passes read eX_t > 0.0 correctly)
    rd->buffer_clear(buffer_edge_intersect, 0, chunk_param.chunk_size * chunk_param.chunk_size * chunk_param.chunk_size * sizeof(float_t) * 3 * chunk_batch_number);

    // 2. put the input inside
    rd->buffer_update(buffer_input_chunk_id, 0,p_chunk_id.size(),p_chunk_id);

    // --- GPU SUBMISSION PIPELINE ---
    int64_t compute_list = rd->compute_list_begin();

    rd->compute_list_bind_uniform_set(compute_list,uniform_set,0); // bind for all pass actually

    uint32_t dispatch_size = chunk_param.chunk_size/8;

    // ---- PASS 1: Sample SDF Distances ----
    rd->compute_list_bind_compute_pipeline(compute_list, pipeline_first_pass);
    rd->compute_list_dispatch(compute_list, dispatch_size, dispatch_size, dispatch_size * chunk_batch_number);

    // Barrier: Wait for Pass 1 to finish writing to sdf_sample
    rd->compute_list_add_barrier(compute_list);

    // ---- PASS 2: Calculate Edge Intersections ----
    rd->compute_list_bind_compute_pipeline(compute_list, pipeline_second_pass);
    rd->compute_list_dispatch(compute_list, dispatch_size, dispatch_size*3, dispatch_size * chunk_batch_number);

    // Barrier: Wait for Pass 2 to finish writing to edge_intersect
    rd->compute_list_add_barrier(compute_list);

    // ---- PASS 3: Generate Vertex Centroids, Normals & Metadata ----
    rd->compute_list_bind_compute_pipeline(compute_list, pipeline_third_pass);
    rd->compute_list_dispatch(compute_list, dispatch_size, dispatch_size, dispatch_size * chunk_batch_number);

    rd->compute_list_end();

    const uint64_t submit_start = now_usec();
    rd->submit();
    rd->sync();
    const uint64_t t_submit_sync = now_usec() - submit_start;

    uint64_t t_points_fetch = 0;
    uint64_t t_points_process = 0;
    uint64_t t_triangulate = 0;

    // 1. Unpack Atomic Counters to get total point count
    const uint64_t t_counters_fetch_start = now_usec();
    PackedByteArray counter_data = rd->buffer_get_data(buffer_atomic_counter);
    PackedInt32Array counters = counter_data.to_int32_array();
    t_points_fetch += now_usec() - t_counters_fetch_start;

    uint32_t total_points = counters[0];
    uint32_t total_triangles = counters[1];

    if (total_points > 0) {
        // 2. Unpack points
        const uint64_t t_points_fetch_start = now_usec();
        PackedByteArray point_data = rd->buffer_get_data(buffer_output_point,0,total_points*4*sizeof(uint32_t));
        PackedInt32Array points = point_data.to_int32_array();
        t_points_fetch += now_usec() - t_points_fetch_start;

        for (uint32_t i = 0; i < chunk_batch_number; i++) {
            uint32_t local_point_count = counters[2 + 2 * i];
            uint32_t local_triangle_count = counters[3 + 2 * i];
            
            // 1 triangle = 3 indices
            unpacked_chunks[i].resize(local_point_count, local_triangle_count * 3);
            
            // Reconstruct the 3D chunk ID using byte array correctly
            // We need to fetch the chunk ID from p_chunk_id based on `i`
            // int32_t cx = p_chunk_id.decode_s32(i * 16 + 0);
            // int32_t cy = p_chunk_id.decode_s32(i * 16 + 4);
            // int32_t cz = p_chunk_id.decode_s32(i * 16 + 8);
            // unpacked_chunks[i].chunk_id = Vector3i(cx, cy, cz);
        }
    
        // 3. Process every point
        const uint64_t t_points_process_start = now_usec();
        for (uint32_t i = 0; i < total_points; i++) {
            // Struct layout: int a, int b, int c, int d;
            int offset = i * 4;
            uint32_t a = points[offset + 0];
            uint32_t b = points[offset + 1];
            uint32_t c = points[offset + 2];
            uint32_t d = points[offset + 3];

            // --- Unpack Flat Index (d) ---
            uint32_t flat_idx = d;
            Vector3i cell_coord = _unflatten_index_z(flat_idx);

            // --- Unpack Position (a) ---
            Vector3 local_vertex;
            local_vertex.x = float(a & 0xFF) / 255.0f + cell_coord.x;
            local_vertex.y = float((a >> 8) & 0xFF) / 255.0f + cell_coord.y;
            local_vertex.z = float((a >> 16) & 0xFF) / 255.0f + cell_coord.z;

            // --- Unpack Normal (b) ---
            // packSnorm4x8 mapping: signed normalized -1.0 to 1.0 mapped to 0 to 255
            Vector3 surface_normal;
            surface_normal.x = Math::clamp(float((int8_t)(b & 0xFF)) / 127.0f, -1.0f, 1.0f);
            surface_normal.y = Math::clamp(float((int8_t)((b >> 8) & 0xFF)) / 127.0f, -1.0f, 1.0f);
            surface_normal.z = Math::clamp(float((int8_t)((b >> 16) & 0xFF)) / 127.0f, -1.0f, 1.0f);

            // (After your unpacking block)
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

            // Fill the unpack chunk points and normals
            uint32_t pt_idx = unpacked_chunks[chunk_idx].point_offset++;
            unpacked_chunks[chunk_idx].output_points.set(pt_idx, local_vertex);
            unpacked_chunks[chunk_idx].output_normals.set(pt_idx, surface_normal);
            unpacked_chunks[chunk_idx].quad_masks[pt_idx] = quad_mask;
            unpacked_chunks[chunk_idx].index_storage[flat_idx] = pt_idx;

        }
        t_points_process = now_usec() - t_points_process_start;

        // 4. Generate Triangles based on dual contouring quad masks
        const uint64_t t_tri_start = now_usec();
        for (uint32_t chunk_idx = 0; chunk_idx < chunk_batch_number; ++chunk_idx) {
            uint32_t tri_offset = 0;

            auto emit_quad = [&](uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3, bool positive_winding) {
                if (positive_winding) {
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p0);
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p1);
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p3);

                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p0);
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p3);
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p2);
                } else {
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p0);
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p3);
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p1);

                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p0);
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p2);
                    unpacked_chunks[chunk_idx].output_triangles.set(tri_offset++, p3);
                }
            };
            
            for (uint32_t pt_idx = 0; pt_idx < unpacked_chunks[chunk_idx].point_offset; ++pt_idx) {
                uint8_t quad_mask = unpacked_chunks[chunk_idx].quad_masks[pt_idx];

                // Mask out the sign bit to check if any geometry exists (mask & 7 checks the first 3 bits)
                if ((quad_mask & 7) == 0) continue; 

                // True if the origin cell is solid, false if it is empty air
                bool origin_solid = (quad_mask & 8) != 0;

                Vector3 local_pos = unpacked_chunks[chunk_idx].output_points[pt_idx];
                Vector3i cell_coord(Math::floor(local_pos.x), Math::floor(local_pos.y), Math::floor(local_pos.z));
                
                // x-axis edge quad
                if ((quad_mask & 1) && cell_coord.y > 0 && cell_coord.z > 0) {
                    uint32_t p0 = pt_idx; // Current cell
                    uint32_t p1 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(0, 1, 0))];
                    uint32_t p2 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(0, 0, 1))];
                    uint32_t p3 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(0, 1, 1))];

                    emit_quad(p0, p2, p1, p3, origin_solid); // axis 0 = X
                }
                
                // y-axis edge quad
                if ((quad_mask & 2) && cell_coord.x > 0 && cell_coord.z > 0) {
                    uint32_t p0 = pt_idx;
                    uint32_t p1 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(1, 0, 0))];
                    uint32_t p2 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(0, 0, 1))];
                    uint32_t p3 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(1, 0, 1))];

                    emit_quad(p0, p1, p2, p3, origin_solid); // axis 1 = Y
                }
                
                // z-axis edge quad
                if ((quad_mask & 4) && cell_coord.x > 0 && cell_coord.y > 0) {
                    uint32_t p0 = pt_idx;
                    uint32_t p1 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(1, 0, 0))];
                    uint32_t p2 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(0, 1, 0))];
                    uint32_t p3 = unpacked_chunks[chunk_idx].index_storage[_get_flat_index_z(cell_coord - Vector3i(1, 1, 0))];

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

    //debug leak
    // RenderingServer *rs = RenderingServer::get_singleton();
    // for (uint32_t chunk_idx = 0; chunk_idx < chunk_batch_number; ++chunk_idx) {
    //     UnpackChunk &chunk = unpacked_chunks[chunk_idx];

    //     rs->free_rid(chunk.instance_rid);
    //     rs->free_rid(chunk.mesh_rid);

    // }
    uint64_t emitted_triangles = 0;
    mutex->lock();  
    // Only the first batch_count slots hold real chunks; the rest of the batch is unused.
    for (uint32_t chunk_idx = 0; chunk_idx < batch_count; ++chunk_idx) {
        UnpackChunk &chunk = unpacked_chunks[chunk_idx];

        ChunkOutput out;
        out.chunk_id = chunk.chunk_id;
        out.mesh_rid = chunk.mesh_rid;
        out.instance_rid = chunk.instance_rid;
        out.triangle_count = chunk.index_count / 3;
        out.drawn_triangles = out.mesh_rid.is_valid() ? chunk.output_triangles.size() / 3 : 0;
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
    mutex->unlock();

}

void GPUMesher::render_chunk() {
    RenderingServer *rs = RenderingServer::get_singleton();
    constexpr bool debug_generate_normals = false;

    // Slots past batch_count are unused but still computed (zeroed chunk id): never build meshes for them.
    for (uint32_t chunk_idx = 0; chunk_idx < batch_count; ++chunk_idx) {
        UnpackChunk &chunk = unpacked_chunks[chunk_idx];

        if (chunk.output_points.is_empty() || chunk.output_triangles.is_empty()) {
            continue;
        }

        if (trim_indices) {
            chunk.output_triangles.resize(chunk.index_count);
            if (chunk.output_triangles.is_empty()) {
                continue;
            }
        }

        Array arrays;
        arrays.resize(Mesh::ARRAY_MAX);
        arrays[Mesh::ARRAY_VERTEX] = chunk.output_points;
        arrays[Mesh::ARRAY_INDEX] = chunk.output_triangles;
        arrays[Mesh::ARRAY_NORMAL] = chunk.output_normals;

        if (debug_generate_normals) {
            Ref<SurfaceTool> surface_tool;
            surface_tool.instantiate();
            surface_tool->create_from_arrays(arrays, Mesh::PRIMITIVE_TRIANGLES);
            surface_tool->generate_normals();

            Array lit_arrays = surface_tool->commit_to_arrays();

            //RID mesh_rid = rs->mesh_create();
            chunk.mesh_rid = rs->mesh_create();
            rs->mesh_add_surface_from_arrays(chunk.mesh_rid, RenderingServer::PRIMITIVE_TRIANGLES, lit_arrays);
        } else {
            //RID mesh_rid = rs->mesh_create();
            chunk.mesh_rid = rs->mesh_create();
            rs->mesh_add_surface_from_arrays(chunk.mesh_rid, RenderingServer::PRIMITIVE_TRIANGLES, arrays);
        }

        //RID instance_rid = rs->instance_create();
        chunk.instance_rid = rs->instance_create();
        DebugLog::chunk_rids_created++;

        rs->instance_set_scenario(chunk.instance_rid, scenario);
        rs->instance_set_base(chunk.instance_rid, chunk.mesh_rid);
        
        // Mesh vertices are in the chunk's vertex grid: scale by the cell size, origin at grid (0,0,0).
        const float cell_size = chunk_param.voxel_size * float(ChunkMath::get_lod_size(chunk.chunk_id));
        Basis scaled_basis;
        scaled_basis.scale(Vector3(cell_size, cell_size, cell_size));

        rs->instance_set_transform(
            chunk.instance_rid,
            Transform3D(scaled_basis, ChunkMath::grid_to_render(chunk.chunk_id, Vector3(), chunk_param))
        );


    }
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

bool GPUMesher::get_results(VoxelState &output){

    mutex->lock();
    const bool is_result_available = result_available; 

    if(is_result_available){
        for (ChunkOutput &out : out_queue) {
            output.add_new_loaded(out);
            working_generation.erase(out.chunk_id);
        }

        out_queue.clear();
        result_available = false;
    }
    mutex->unlock();

    return is_result_available;
}

void GPUMesher::process_queue(){

    if(!_is_available()) return;

    PackedByteArray new_batch;
    new_batch.resize(chunk_batch_number * GPUFrame::CHUNK_INPUT_BYTES);
    new_batch.fill(0);

    unpacked_chunks.clear();
    unpacked_chunks.resize(chunk_batch_number);

    int count = 0;
    ChunkId new_chunk;

    //UtilityFunctions::print("GPUMesher : prepare for ",chunk_batch_number);
    
    while(count < chunk_batch_number && _pop_queue(new_chunk)){

        //UtilityFunctions::print("Dispatching chunk: ", new_chunk);

        working_generation.insert(new_chunk);
        
        GPUFrame::encode_chunk_input(new_batch, count, new_chunk, chunk_param);

        unpacked_chunks[count].chunk_id = new_chunk;
        
        count++;
    }

    if(count > 0) {
        mutex->lock();
        input_chunk_id = new_batch;
        batch_count = count;
        worker_available = false;
        mutex->unlock();
        
        semaphore->post(); // wake up the worker thread
        //UtilityFunctions::print("GPUMesher : Semaphore posted.");
    }

}

bool GPUMesher::_pop_queue(ChunkId &new_chunk) {
    for (LODHashSet *queue : chunk_queue) {
        if (queue == nullptr || queue->is_empty()) {
            continue;
        }

        LODHashSet::Iterator it = queue->begin();
        if (!it) {
            continue;
        }

        ChunkId chunk_pos = *it;
        queue->remove(it);
        new_chunk = chunk_pos;
        return true;
    }

    return false;
}

void GPUMesher::send_remove_chunk(LODHashSet p_chunks) {
    mutex->lock();

    for (const ChunkId &chunk_pos : p_chunks) {
        if (working_generation.has(chunk_pos)) {
            continue;
        }

        ChunkOutput tombstone;
        tombstone.chunk_id = chunk_pos;
        del_queue.push_back(tombstone);
    }

    mutex->unlock();
}

void GPUMesher::send_chunk(const LODHashSet &p_chunks){

    for (LODHashSet *q : chunk_queue) {
        if (q) {
            q->clear();
        }
    }

    for (const ChunkId &chunk_pos : p_chunks) {

    //UtilityFunctions::print("GPUMesher : Adding ",chunk_pos);

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
