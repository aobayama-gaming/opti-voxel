#pragma once

// Compute passes of the direct mesher. Same front end as the legacy GPUMesher (SDF sample and
// edge passes), then the triangulation the legacy path did on the CPU is done on the GPU:
//
//   1 sdf       per vertex sample            (legacy pass 1, unchanged)
//   2 edges     per axis edge crossing t     (legacy pass 2, but always writes -> no clear needed)
//   3 vertices  dense per-cell vertex (chunk-local) + normal, per-chunk quad count (workgroup-reduced
//               atomics)
//   4 reserve   one thread packs chunks into the batch's pages, each chunk starting on a group
//               (texture row) boundary, no chunk straddling a page
//   5 emit      one quad per sign-changing edge, written at base + atomicAdd(cursor) as 4 corners
//               of 8 bytes, plus the group header for the first quad of each group, into the staging
//               image (layout: page_format.hpp); staging rows are then copied into the page textures
//
// The quad topology, corner order and winding mirror GPUMesher's CPU triangulation exactly
// (see emit pass), so the two paths can be compared triangle for triangle.

inline constexpr const char GLSL_DIRECT_BINDINGS[] = R"glsl(

// Per chunk of the batch: x = quad count, y = emit cursor, z = base slot, w = flags
#define CHUNK_FLAG_OVERFLOW 1u
layout(set = 0, binding = 0, std430) restrict buffer ChunkDirectory {
    uvec4 chunk_dir[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer InputChunk {
    ChunkInput input_chunk[];
};

layout(set = 0, binding = 2, std430) restrict buffer SDFSample {
    vec4 sdf_sample[];
};

layout(set = 0, binding = 3, std430) restrict buffer EdgeIntersect {
    float edge_intersect[];
};

// Dense per cell: xyz position in the chunk's vertex grid, w = oct22 normal + 1 (0 = no vertex)
layout(set = 0, binding = 4, std430) restrict buffer CellVertex {
    vec4 cell_vertex[];
};

layout(set = 0, binding = 5, std140) uniform ChunkParamUniform {
    ChunkParam chunk_param;
};

// K pages stacked vertically, one group of quads per row (page_format.hpp).
#define QUAD_GROUP 64u
layout(set = 0, binding = 6, rgba16ui) uniform restrict writeonly uimage2D staging;

/*GLSL_PAGE_FORMAT*/

// Debug/status counters + per page fill (instance count), read back asynchronously.
#define STATUS_TOTAL_QUADS 0
#define STATUS_OVERFLOW_CHUNKS 1
#define STATUS_MISSING_NEIGHBOR 2
#define STATUS_NAN_VERTICES 3
#define STATUS_OUT_OF_BOUNDS 4
#define STATUS_PAGE_FILL 8
layout(set = 0, binding = 7, std430) restrict buffer Status {
    uint status[];
};

layout(push_constant, std430) uniform BatchParams {
    uint batch_count;
    uint page_count;
    uint page_capacity;
    uint pad;
} batch;

uint get_flat_index_z(uvec3 cell_coordinate) {
    return cell_coordinate.x + (cell_coordinate.y * chunk_param.chunk_size) + (cell_coordinate.z * chunk_param.chunk_size * chunk_param.chunk_size);
}

uint get_edge_index(uint chunk_idx, uint axis, uvec3 local_c) {
    uint cs = chunk_param.chunk_size;
    uint local_idx = local_c.x + (local_c.y * cs) + (local_c.z * cs * cs);
    return (chunk_idx * 3 * cs * cs * cs) + (axis * cs * cs * cs) + local_idx;
}

)glsl";

// Same as legacy pass 2, except it writes 0.0 for edges without a crossing: every entry the
// later passes read is rewritten each batch, so the 48 MB clear of the legacy path is not needed.
inline constexpr const char GLSL_DIRECT_PASS_EDGES[] = R"glsl(

uint axis = gl_GlobalInvocationID.y / chunk_param.chunk_size;

if (cell_chunk_coordinate[axis] < chunk_param.chunk_size - 1) {
    uvec3 global_cell = uvec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y % chunk_param.chunk_size, gl_GlobalInvocationID.z);

    uvec3 offset = uvec3(0);
    offset[axis] = 1;

    vec4 center_sdf = sdf_sample[get_flat_index_z(global_cell)];
    vec4 offset_sdf = sdf_sample[get_flat_index_z(global_cell + offset)];

    // Solid is !(w > 0) in every pass. A crossing is t > 0 downstream, so a sample of exactly 0.0
    // must not give t = 0 (lost quad = hole); the signs differ, so the denominator is never 0.
    float t = 0.0;
    if ((center_sdf.w > 0.0) != (offset_sdf.w > 0.0)) {
        t = max(center_sdf.w / (center_sdf.w - offset_sdf.w), 1e-6);
    }
    uint chunk_idx = gl_GlobalInvocationID.z / chunk_param.chunk_size;
    edge_intersect[get_edge_index(chunk_idx, axis, uvec3(cell_chunk_coordinate))] = t;
}

)glsl";

// Vertex placement identical to legacy pass 3 (mass point of edge crossings, normal from the SDF
// gradient at the vertex), stored densely per cell in chunk-local vertex coordinates; quantised
// only when written to the page (1/512 cell, legacy uses a coarser grid).
inline constexpr const char GLSL_DIRECT_PASS_VERTICES[] = R"glsl(

uint chunk_idx = gl_GlobalInvocationID.z / chunk_param.chunk_size;
uint flat_idx = get_flat_index_z(gl_GlobalInvocationID);
uint quads_here = 0u;
vec4 out_vertex = vec4(0.0);

if (gl_LocalInvocationIndex == 0u) {
    wg_quad_count = 0u;
}
barrier();

if (all(lessThan(cell_chunk_coordinate, ivec3(chunk_param.chunk_size - 1)))) {

#define offset(x, y, z) (uvec3(cell_chunk_coordinate) + uvec3(x, y, z))
    float e0_t = edge_intersect[get_edge_index(chunk_idx, 0, offset(0,0,0))];
    float e2_t = edge_intersect[get_edge_index(chunk_idx, 0, offset(0,1,0))];
    float e4_t = edge_intersect[get_edge_index(chunk_idx, 0, offset(0,0,1))];
    float e6_t = edge_intersect[get_edge_index(chunk_idx, 0, offset(0,1,1))];
    float e3_t = edge_intersect[get_edge_index(chunk_idx, 1, offset(0,0,0))];
    float e1_t = edge_intersect[get_edge_index(chunk_idx, 1, offset(1,0,0))];
    float e7_t = edge_intersect[get_edge_index(chunk_idx, 1, offset(0,0,1))];
    float e5_t = edge_intersect[get_edge_index(chunk_idx, 1, offset(1,0,1))];
    float e8_t = edge_intersect[get_edge_index(chunk_idx, 2, offset(0,0,0))];
    float e9_t = edge_intersect[get_edge_index(chunk_idx, 2, offset(1,0,0))];
    float e11_t = edge_intersect[get_edge_index(chunk_idx, 2, offset(0,1,0))];
    float e10_t = edge_intersect[get_edge_index(chunk_idx, 2, offset(1,1,0))];
#undef offset

    vec3 local_vertex = vec3(0.0);
    int active_edge_count = 0;
    if (e0_t > 0.0) { local_vertex += vec3(e0_t, 0.0, 0.0); active_edge_count++; }
    if (e2_t > 0.0) { local_vertex += vec3(e2_t, 1.0, 0.0); active_edge_count++; }
    if (e4_t > 0.0) { local_vertex += vec3(e4_t, 0.0, 1.0); active_edge_count++; }
    if (e6_t > 0.0) { local_vertex += vec3(e6_t, 1.0, 1.0); active_edge_count++; }
    if (e3_t > 0.0) { local_vertex += vec3(0.0, e3_t, 0.0); active_edge_count++; }
    if (e1_t > 0.0) { local_vertex += vec3(1.0, e1_t, 0.0); active_edge_count++; }
    if (e7_t > 0.0) { local_vertex += vec3(0.0, e7_t, 1.0); active_edge_count++; }
    if (e5_t > 0.0) { local_vertex += vec3(1.0, e5_t, 1.0); active_edge_count++; }
    if (e8_t > 0.0) { local_vertex += vec3(0.0, 0.0, e8_t); active_edge_count++; }
    if (e9_t > 0.0) { local_vertex += vec3(1.0, 0.0, e9_t); active_edge_count++; }
    if (e11_t > 0.0) { local_vertex += vec3(0.0, 1.0, e11_t); active_edge_count++; }
    if (e10_t > 0.0) { local_vertex += vec3(1.0, 1.0, e10_t); active_edge_count++; }

    if (active_edge_count > 0) {
        local_vertex /= float(active_edge_count);
        vec3 chunk_vertex = local_vertex + vec3(cell_chunk_coordinate);
        vec3 surface_normal = vertex_normal(vec3(0.0), cascade_vertex_gradient(gl_GlobalInvocationID, local_vertex));
        if (any(isnan(chunk_vertex)) || any(isinf(chunk_vertex)) || any(isnan(surface_normal))) {
            atomicAdd(status[STATUS_NAN_VERTICES], 1u);
        } else {
            out_vertex = vec4(chunk_vertex, float(pack_normal_oct22(surface_normal)) + 1.0);
        }

        // Count exactly the quads the emit pass will produce (same rules as the CPU triangulation).
        ivec3 c = cell_chunk_coordinate;
        if (e0_t > 0.0 && c.y > 0 && c.z > 0) quads_here++;
        if (e3_t > 0.0 && c.x > 0 && c.z > 0) quads_here++;
        if (e8_t > 0.0 && c.x > 0 && c.y > 0) quads_here++;
    }
}

cell_vertex[flat_idx] = out_vertex;

if (quads_here > 0u) {
    atomicAdd(wg_quad_count, quads_here);
}
barrier();
// One global atomic per workgroup (all 8x8x8 invocations of a workgroup belong to the same chunk).
if (gl_LocalInvocationIndex == 0u && wg_quad_count > 0u) {
    atomicAdd(chunk_dir[chunk_idx].x, wg_quad_count);
}

)glsl";

inline constexpr const char GLSL_DIRECT_PASS_RESERVE[] = R"glsl(

if (gl_GlobalInvocationID == uvec3(0)) {
    uint page = 0u;
    uint fill = 0u;
    uint total = 0u;
    for (uint i = 0u; i < batch.batch_count; i++) {
        uint count = chunk_dir[i].x;
        chunk_dir[i].y = 0u;
        if (count == 0u) {
            chunk_dir[i].z = 0u;
            chunk_dir[i].w = 0u;
            continue;
        }
        // Chunks start on a group: each one is whole texture rows, so it can be moved or zeroed
        // with one copy. The tail of its last group stays zero (degenerate).
        uint padded = (count + QUAD_GROUP - 1u) / QUAD_GROUP * QUAD_GROUP;
        if (fill + padded > batch.page_capacity) { // never straddle pages: start the next one
            page++;
            fill = 0u;
        }
        if (count > batch.page_capacity || page >= batch.page_count) {
            chunk_dir[i].z = 0u;
            chunk_dir[i].w = CHUNK_FLAG_OVERFLOW;
            status[STATUS_OVERFLOW_CHUNKS] += 1u;
            continue;
        }
        chunk_dir[i].z = page * batch.page_capacity + fill;
        chunk_dir[i].w = 0u;
        fill += padded;
        total += count;
        status[STATUS_PAGE_FILL + page] = fill / QUAD_GROUP;
    }
    status[STATUS_TOTAL_QUADS] = total;
}

)glsl";

// Corner order mirrors GPUMesher::_execute_gpu_pipeline's emit_quad(a, b, c, d, positive):
// positive -> triangles (a,b,d),(a,d,c) == base quad A,B,D / A,D,C with A=a B=b C=c D=d;
// negative -> triangles (a,d,b),(a,c,d) == the same with b and c swapped.
inline constexpr const char GLSL_DIRECT_PASS_EMIT[] = R"glsl(

uint chunk_idx = gl_GlobalInvocationID.z / chunk_param.chunk_size;
uint flat_idx = get_flat_index_z(gl_GlobalInvocationID);
uvec4 dir = chunk_dir[chunk_idx];
vec4 v0 = cell_vertex[flat_idx];

if (v0.w > 0.0 && (dir.w & CHUNK_FLAG_OVERFLOW) == 0u && all(lessThan(cell_chunk_coordinate, ivec3(chunk_param.chunk_size - 1)))) {
    ivec3 c = cell_chunk_coordinate;
    bool origin_solid = !(sdf_sample[flat_idx].w > 0.0); // same side rule as the edge pass
    uint sx = 1u;
    uint sy = uint(chunk_param.chunk_size);
    uint sz = uint(chunk_param.chunk_size * chunk_param.chunk_size);

    for (uint axis = 0u; axis < 3u; axis++) {
        float t = edge_intersect[get_edge_index(chunk_idx, axis, uvec3(c))];
        if (!(t > 0.0)) continue;
        // The two other axes, and the neighbour cells sharing this edge.
        uint s1; uint s2; bool ok; bool a_first;
        if (axis == 0u) { ok = c.y > 0 && c.z > 0; s1 = sy; s2 = sz; a_first = false; } // legacy: emit_quad(p0, p2, p1, p3)
        else if (axis == 1u) { ok = c.x > 0 && c.z > 0; s1 = sx; s2 = sz; a_first = true; } // legacy: emit_quad(p0, p1, p2, p3)
        else { ok = c.x > 0 && c.y > 0; s1 = sx; s2 = sy; a_first = false; } // legacy: emit_quad(p0, p2, p1, p3)
        if (!ok) continue;

        vec4 v1 = cell_vertex[flat_idx - s1];
        vec4 v2 = cell_vertex[flat_idx - s2];
        vec4 v3 = cell_vertex[flat_idx - s1 - s2];
        if (!(v1.w > 0.0 && v2.w > 0.0 && v3.w > 0.0)) {
            atomicAdd(status[STATUS_MISSING_NEIGHBOR], 1u);
            continue;
        }
        // emit_quad arguments (a, b, c, d)
        vec4 qa = v0;
        vec4 qb = a_first ? v1 : v2;
        vec4 qc = a_first ? v2 : v1;
        vec4 qd = v3;
        if (!origin_solid) { // negative winding: swap b and c
            vec4 tmp = qb; qb = qc; qc = tmp;
        }
        uint slot = dir.z + atomicAdd(chunk_dir[chunk_idx].y, 1u);
        // The chunk starts on a group (reserve pass), so its slots cover each of its groups' first
        // slot exactly once.
        if (slot % QUAD_GROUP == 0u) {
            store_group_header(slot / QUAD_GROUP, chunk_id);
        }
        float pos_scale = page_pos_scale(chunk_param.chunk_size);
        store_corner(slot, 0u, encode_corner(qa.xyz, uint(qa.w - 1.0), pos_scale));
        store_corner(slot, 1u, encode_corner(qb.xyz, uint(qb.w - 1.0), pos_scale));
        store_corner(slot, 2u, encode_corner(qc.xyz, uint(qc.w - 1.0), pos_scale));
        store_corner(slot, 3u, encode_corner(qd.xyz, uint(qd.w - 1.0), pos_scale));
    }
}

)glsl";
