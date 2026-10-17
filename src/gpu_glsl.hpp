#pragma once

// GLSL snippets shared by the compute meshers (GPU_mesher legacy path and GPU_direct_mesher).
// They are stitched together at runtime by each mesher's compile step.

inline constexpr const char GLSL_STRUCT_DEFINITION[] = R"glsl(

struct PointMetadata { int a; int b; int c; int d; };

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

float chunk_world_dimension(ChunkParam param) {
    return chunk_voxel_dimension(param) * param.voxel_size;
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

int get_lod_size(ivec3 coord) {
    return (1 << get_lod(coord));
}


// ---------------------------------------------------------
// SDF transform
// ---------------------------------------------------------

vec4 sdf_negate(vec4 sdf) {
    // If f(x) = -g(x), then grad(f) = -grad(g)
    return vec4(-sdf.xyz, -sdf.w);
}

vec4 sdf_scale(vec4 sdf, float scale) {
    // If f(x) = a * g(x), then grad(f) = a * grad(g)
    return vec4(sdf.xyz * scale, sdf.w * scale);
}

vec4 sdf_add(vec4 sdf1, vec4 sdf2) {
    // If f(x) = g(x) + h(x), then grad(f) = grad(g) + grad(h)
    return vec4(sdf1.xyz + sdf2.xyz, sdf1.w + sdf2.w);
}

vec4 sdf_subtract(vec4 sdf1, vec4 sdf2) {
    // If f(x) = g(x) - h(x), then grad(f) = grad(g) - grad(h)
    return vec4(sdf1.xyz - sdf2.xyz, sdf1.w - sdf2.w);
}

vec4 sdf_abs(vec4 sdf) {
    // Absolute value flips gradient if value is negative
    if(sdf.w < 0.0) {
        return vec4(-sdf.xyz, -sdf.w);
    } else {
        return sdf;
    }
}

vec4 sdf_constant(float constant) {
    // Constant has zero gradient
    return vec4(0.0, 0.0, 0.0, constant);
}

vec4 sdf_max(vec4 sdf1, vec4 sdf2) {
    // Max operation takes the gradient of the active surface
    if(sdf1.w > sdf2.w) {
        return sdf1;
    } else {
        return sdf2;
    }
}

vec4 sdf_min(vec4 sdf1, vec4 sdf2) {
    // Max operation takes the gradient of the active surface
    if(sdf1.w < sdf2.w) {
        return sdf1;
    } else {
        return sdf2;
    }
}

vec4 sdf_softmax(vec4 a, vec4 b, float intensity) {
    // Calculate the blend factor based on the distance between the two SDFs
    // "intensity" is the smoothing radius in meters
    float h = clamp(0.5 + 0.5 * (a.w - b.w) / intensity, 0.0, 1.0);
    
    // Smoothly blend the distance (and add the polynomial smoothing curve)
    float smooth_dist = mix(b.w, a.w, h) + intensity * h * (1.0 - h);
    
    // Because of the Chain Rule, the gradient of the smooth max is simply 
    // the linearly interpolated gradients of the two surfaces!
    vec3 smooth_grad = normalize(mix(b.xyz, a.xyz, h));
    
    return vec4(smooth_grad, smooth_dist);
}

)glsl";

inline constexpr const char GLSL_SKIRT_SDF[] = R"glsl(

// Signed distance of a vertex-grid coordinate to the chunk bounds along one axis, in voxels of the
// chunk's LOD: positive inside, negative in the overhang. The grid overhangs by skirt_size voxels on
// the min side (index skirt_size is the chunk's minimum) and skirt_size - 1 on the max side.
float _skirt_inside_distance(float coord, ChunkParam chunk_param){
    float lo = float(chunk_param.skirt_size);
    float hi = lo + chunk_voxel_dimension(chunk_param);
    return min(coord - lo, hi - coord);
}

// Skirt: past the chunk bounds the SDF is pushed towards air by up to skirt_depth voxels, so the
// overhanging rim of the mesh turns down into the solid and hides under the neighbour's surface
// (which covers that overhang), whatever its LOD. Inside the bounds the SDF is untouched, so the
// visible surface is exact. The gradient (normal) stays the base one.
vec4 skirt_sdf(ChunkFrame chunk, vec3 point_coord, ChunkParam chunk_param){

    vec4 base_sdf = evaluate_sdf(world_pos_from_grid(chunk, point_coord, chunk_param), chunk.lod);

    if(chunk_param.skirt_size > 0 && chunk_param.skirt_depth > 0.0){
        float inside = min(_skirt_inside_distance(point_coord.x, chunk_param),
                       min(_skirt_inside_distance(point_coord.y, chunk_param),
                           _skirt_inside_distance(point_coord.z, chunk_param)));
        float depth_m = chunk_param.skirt_depth * chunk_param.voxel_size * float(chunk.lod_size);
        base_sdf.w += clamp(-inside, 0.0, 1.0) * depth_m;
    }
    return base_sdf;
}

)glsl";
