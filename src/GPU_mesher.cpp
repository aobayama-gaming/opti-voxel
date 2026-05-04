#include "GPU_mesher.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>

static constexpr const char GLSL_STRUCT_DEFINITION[] = R"(

struct PointMetadata { int a; int b; int c; int d; };

)";

static constexpr const char GLSL_HELPER_FUNCTION[] = R"(
)";

static constexpr const char GLSL_BUFFER_BINDING[] = R"(
layout(set = 0, binding = 0, std430) restrict buffer AtomicCounter {
    uint counter[];
}atomic_counter;

layout(set = 0, binding = 1, std430) restrict buffer InputChunk {
    ivec3 id[];
}input_chunk;

layout(set = 0, binding = 2, std430) restrict buffer SDFSample {
    vec4 sample[];
}sdf;

layout(set = 0, binding = 3, std430) restrict buffer EdgeIntersect {
    float value[];
}edge_intersect;

layout(set = 0, binding = 4, std430) restrict buffer OutputPoint {
    PointMetadata metadata[];
}output_point;

)";

RID GPUMesher::compile_dynamic_shader(const String &algorithm_pass) {
    
    // 1. Define your base Compute Shader using a C++ Raw String
    String shader = R"(
        #[compute]
        #version 450

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

        /*GLSL_STRUCT_DEFINITION*/

        /*GLSL_BUFFER_BINDING*/

        /*GLSL_HELPER_FUNCTION*/

        //Will inject vec4 terrain_sdf(vec3 pos)
        /*GLSL_TERRAIN_SDF*/

        // A placeholder function that we will inject code into!
        vec4 evaluate_sdf(vec3 pos) {

            //injected function
            return terrain_sdf(vec3 pos);

        }

        void main() {
            /*GLSL_ALGORITHM_PASS*/
        }
    )";

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
        UtilityFunctions::printerr("Failed to compile injected SDF Shader!\n", error_msg);
        return RID(); // Return invalid RID
    }

    // 6. Create the final shader RID
    RID shader_rid = rd->shader_create_from_spirv(spirv);
    
    return shader_rid;
}

static constexpr const char GLSL_FIRST_PASS[] = R"(



)";


void GPUMesher::initialize(SDFBase* p_sdf,ChunkParam p_chunk_param, uint32_t p_chunk_batch_number){

    sdf = p_sdf;
    chunk_param = p_chunk_param;
    chunk_batch_number = p_chunk_batch_number;

    //alocate memory

    rd = RenderingServer::get_singleton()->get_rendering_device();

    PackedByteArray empty_data;

    buffer_input_chunk_id = rd->storage_buffer_create((chunk_batch_number)*sizeof(Vector4i),empty_data); // need to be Vector4i due to padding...

    buffer_atomic_counter = rd->storage_buffer_create(2*(1+chunk_batch_number)*sizeof(uint32_t),empty_data);
    
    buffer_sdf_sample = rd->storage_buffer_create(chunk_param.chunk_size*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(Vector4),empty_data);

    buffer_edge_intersect =rd->storage_buffer_create((chunk_param.chunk_size-1)*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(float_t)*3,empty_data);
    /*
    Compressed position uint8 x 3
    Compressed normal uint8 x 3 
    Address of point uint32
    Address of chunk + quad 3bit mask 16bit
    
    Total of 3 uint32_t
    */
    buffer_output_point = rd->storage_buffer_create(3*chunk_param.chunk_size*chunk_param.chunk_size*chunk_param.chunk_size*sizeof(uint32_t),empty_data);

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

    // 2. Create the Uniform Set (CACHE THIS RID!)
    // 'shader_rid' is the RID returned when you compiled your GLSL.
    // '0' is the layout(set = 0) index.
    // RID master_uniform_set = rd->uniform_set_create(uniforms, shader_rid, 0);

};

void GPUMesher::_execute_pipeline(){

    // 1. RESET ATOMIC COUNTERS TO 0
    PackedByteArray counter_zeros;
    counter_zeros.resize(2*(1+chunk_batch_number) * sizeof(uint32_t));
    counter_zeros.fill(0);
    rd->buffer_update(buffer_atomic_counter, 0, counter_zeros.size(), counter_zeros);

    // --- GPU SUBMISSION PIPELINE ---
    int64_t compute_list = rd->compute_list_begin();

}