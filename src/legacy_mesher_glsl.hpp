#pragma once

// The legacy mesher's GLSL: one shader template and two algorithm passes (edges, vertices), engine-free so a
// non-Godot host (the Dawn adapter) meshes with the same text as the GDExtension. The bindings are the legacy
// mesher's: counters, chunk inputs, samples, edges, points, the ChunkParam uniform, then the table and palette.
//
// The dynamic pieces are the terrain's cascade code, the instance table's declarations, the chunk palette and the
// material buffer; fill_legacy_shader() splices them, exactly as GPUMesher::compile_dynamic_shader does.

#include "cascade_glsl.hpp"
#include "exact_noise_glsl.hpp"
#include "gpu_frame.hpp"
#include "gpu_glsl.hpp"
#include "ov/str.hpp"

namespace LegacyMesherGLSL {

// A point's word c: chunk index (16 bits) | quad mask << 16 | palette index << 20.
inline constexpr uint32_t POINT_PALETTE_SHIFT = 20;

inline constexpr const char BUFFER_BINDING[] = R"glsl(
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

// Per sample: its chunk palette index (written by the cascade's level 0 for chunks with a list).
/*GLSL_MATERIAL_BUFFER*/

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

inline constexpr const char SHADER_TEMPLATE[] = R"glsl(#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

/*GLSL_STRUCT_DEFINITION*/

/*GLSL_BUFFER_BINDING*/

/*GLSL_HELPER_FUNCTION*/

/*GLSL_VERTEX_GRADIENT*/

/*GLSL_QEF_VERTEX*/

void main() {



    ChunkFrame chunk = chunk_frame(input_chunk[gl_GlobalInvocationID.z/chunk_param.chunk_size], chunk_param);
    ivec3 chunk_id = chunk.id_lo;/*GLSL_SET_CHUNK*/

    ivec3 cell_chunk_coordinate = ivec3(
        gl_GlobalInvocationID.x,
        gl_GlobalInvocationID.y%chunk_param.chunk_size, // x / y / z batching
        gl_GlobalInvocationID.z%chunk_param.chunk_size // chunk batching
        );

    /*GLSL_ALGORITHM_PASS*/
}
    )glsl";

inline constexpr const char SECOND_PASS[] = R"glsl(

uint axis = gl_GlobalInvocationID.y / chunk_param.chunk_size;

if (cell_chunk_coordinate[axis] < chunk_param.chunk_size - 1) {

    // global_cell is used ONLY for safely reading from the SDF buffer
    uvec3 global_cell = uvec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y % chunk_param.chunk_size, gl_GlobalInvocationID.z);

    uvec3 offset = uvec3(0);
    offset[axis] = 1;

    vec4 center_sdf = sdf_sample[get_flat_index_z(global_cell)];
    vec4 offset_sdf = sdf_sample[get_flat_index_z(global_cell + offset)];

    // Solid is !(w > 0) in every pass. A crossing is t > 0 downstream, so a sample of exactly 0.0
    // must not give t = 0 (lost quad = hole); the signs differ, so the denominator is never 0.
    if ((center_sdf.w > 0.0) != (offset_sdf.w > 0.0)) {
        float t = max(center_sdf.w / (center_sdf.w - offset_sdf.w), 1e-6);

        // Extract chunk batch to safely write edge data
        uint chunk_idx = gl_GlobalInvocationID.z / chunk_param.chunk_size;

        // Write using the strictly local cell_chunk_coordinate
        edge_intersect[get_edge_index(chunk_idx, axis, uvec3(cell_chunk_coordinate))] = t;
    }
}

)glsl";

inline constexpr const char THIRD_PASS[] = R"glsl(

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

    vec3 surface_normal = vertex_normal(vec3(0.0), cascade_vertex_gradient(gl_GlobalInvocationID, local_vertex));
    uint packed_normal = packSnorm4x8(vec4(surface_normal, 0.0));

    // 3. Triangle Metadata (3 bit mask for X/Y/Z quad generation in Dual Contouring/Surface nets)
    // We check if edges 0, 3, and 8 have intersections to know if this cell spawns a quad
    uint quad_mask = 0;
    if (e0_t > 0.0) quad_mask |= 1; // X-axis crossing quad
    if (e3_t > 0.0) quad_mask |= 2; // Y-axis crossing quad
    if (e8_t > 0.0) quad_mask |= 4; // Z-axis crossing quad

    if (!(sdf_sample[get_flat_index_z(gl_GlobalInvocationID)].w > 0.0)) { // same side rule as the edge pass
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
    output_point[global_point_idx].a = int(packed_pos);
    output_point[global_point_idx].b = int(packed_normal);

    // Store chunk index and mask in 'c'
    // (and the vertex's palette index at POINT_PALETTE_SHIFT; without a climate's base palette a chunk without a list
    // is all ground, its samples' materials unwritten)
    uint palette = input_chunk[chunk_idx].anchor_hi.w == 0 && !GROUND_MATERIALS ? 0u : cascade_vertex_material(gl_GlobalInvocationID);
    output_point[global_point_idx].c = int((chunk_idx & 0xFFFF) | (quad_mask << 16) | (palette << POINT_PALETTE_SHIFT));

    // Store localized point address in 'd'
    output_point[global_point_idx].d = int(get_flat_index_z(cell_chunk_coordinate));
}


}

)glsl";

// The full compute shader for one algorithm pass. p_table_glsl: the instance table's declarations at bindings 6 and
// 7 (empty for a terrain without a table); p_set_chunk: the statement after `ivec3 chunk_id = chunk.id_lo;`;
// p_base_palette: ChunkPalette::glsl_base's text; p_material: the per-sample material buffer declaration.
inline ov::str fill_shader(const char *p_pass, const ov::str &p_table_glsl, const ov::str &p_set_chunk, const ov::str &p_base_palette,
        const ov::str &p_material, bool p_dual_contouring) {
    const ov::str bindings = ov::replace_all(BUFFER_BINDING, "/*GLSL_MATERIAL_BUFFER*/", p_material) + p_table_glsl +
            ov::format("const uint POINT_PALETTE_SHIFT = %du;\n", uint64_t(POINT_PALETTE_SHIFT)) + p_base_palette;
    const ov::str helpers = ov::str(GLSL_HELPER_FUNCTION) + GPUFrame::glsl_functions() + ov::str(GLSL_EXACT_NOISE) + GPUFrame::glsl_int64();
    const ov::str code = ov::replace_all(SHADER_TEMPLATE,
            { { "/*GLSL_STRUCT_DEFINITION*/", ov::str(GLSL_STRUCT_DEFINITION) + GLSL_FRAME_STRUCTS },
                    { "/*GLSL_BUFFER_BINDING*/", bindings },
                    { "/*GLSL_HELPER_FUNCTION*/", helpers },
                    { "/*GLSL_SET_CHUNK*/", p_set_chunk },
                    { "/*GLSL_QEF_VERTEX*/", ov::str(GLSL_VERTEX_NORMAL) + (p_dual_contouring ? GLSL_QEF_VERTEX : "") } });
    const ov::str pass = p_dual_contouring ? dual_contouring_pass(p_pass) : ov::str(p_pass);
    return CascadeGLSL::fill_mesher_pass(code, pass);
}

} // namespace LegacyMesherGLSL
