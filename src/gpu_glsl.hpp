#pragma once

// GLSL snippets of the compute mesher (GPU_direct_mesher), stitched together at runtime by its
// compile step.

inline constexpr const char GLSL_STRUCT_DEFINITION[] = R"glsl(

// Mirror of the C++ ChunkParam, std140 (laid out by GPUFrame::chunk_param_bytes).
struct ChunkParam {
    int chunk_size;
    float voxel_size;
    int skirt_size;
    float skirt_depth; // in voxels of the chunk's LOD
    uvec4 origin_lo; // world origin in LOD0 voxels, low words (see gpu_frame.hpp)
    ivec4 origin_hi; // high words
};

float chunk_voxel_dimension(ChunkParam param) {
    return float(param.chunk_size - param.skirt_size * 2);
}

)glsl";

inline constexpr const char GLSL_HELPER_FUNCTION[] = R"glsl(

// ---------------------------------------------------------
// Bitwise & LOD functions
// ---------------------------------------------------------

int get_lod(ivec3 coord) {
    int lod_x = findLSB(coord.x);
    if (lod_x == -1) {
        return 0;
    }
    return lod_x;
}

)glsl";

// The vertex pass's normal.
inline constexpr const char GLSL_VERTEX_NORMAL[] = R"glsl(

// The vertex normal: the SDF gradient at the vertex, else straight up. A NaN normal drops the vertex: a
// broken quad, a hole. The gradient vanishes exactly on a medial axis, and a cell symmetric about one puts
// its vertex there: a capsule thinner than the cell on a lattice line (the scatter's pillars, thin trunks
// at LOD 2). The shape is finer than the cell, so any finite normal will do.
vec3 vertex_normal(vec3 p_gradient) {
    precise float len2 = p_gradient.x * p_gradient.x + p_gradient.y * p_gradient.y + p_gradient.z * p_gradient.z;
    return len2 > 0.0 ? normalize(p_gradient) : vec3(0.0, 1.0, 0.0);
}

)glsl";
