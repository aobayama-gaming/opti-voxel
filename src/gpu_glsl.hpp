#pragma once

// GLSL snippets shared by the compute meshers (GPU_mesher legacy path and GPU_direct_mesher).
// They are stitched together at runtime by each mesher's compile step.

inline constexpr const char GLSL_STRUCT_DEFINITION[] = R"glsl(

#ifndef WORLD_NUDGE
#define WORLD_NUDGE 0.123456 // Define this to your desired nudge value or pass as uniform
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

inline constexpr const char GLSL_HELPER_FUNCTION[] =  R"glsl(

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

vec2 _calculate_skirt_offset(float coord,ChunkParam chunk_param){

    float dist_from_min = coord;
    float dist_from_max = float(chunk_param.chunk_size-1) - coord;

    float offset_direction = 1.0;
    float nearest_edge =dist_from_min;


    if (dist_from_max<dist_from_min){
        nearest_edge=dist_from_max;
        offset_direction = -1.0;
    }

    float skirt_trimmed = float(chunk_param.skirt_size)*0.75;

    float offset = 0.0;
    float offset_grad = 0;

    if(nearest_edge < skirt_trimmed){
        offset = 1.0 - (nearest_edge/skirt_trimmed);
        offset_grad = - offset_direction / skirt_trimmed;
    }

    return vec2(offset_grad,offset);

}

vec4 skirt_sdf(ivec3 chunk_coord,vec3 point_coord, ChunkParam chunk_param){

    vec4 base_sdf = evaluate_sdf(vertices_to_world(chunk_coord,vec3(point_coord),chunk_param));

    if(chunk_param.skirt_size > 0){

        float gradient_magnitude = length(base_sdf.xyz);
        //float offset_size = chunk_param.voxel_size*0.3*float(get_lod_size(chunk_coord)) / max(gradient_magnitude, 0.1);
        float offset_size = chunk_param.voxel_size*0.3*float(get_lod_size(chunk_coord)) / gradient_magnitude/1000.0;

        vec2 x_offset = _calculate_skirt_offset(point_coord.x,chunk_param);
        vec2 y_offset = _calculate_skirt_offset(point_coord.y,chunk_param);
        vec2 z_offset = _calculate_skirt_offset(point_coord.z,chunk_param);

        // vec4 offset = vec4(x_offset.x,0.0,0.0,x_offset.y);
        // offset +=vec4(0.0,y_offset.x,0.0,y_offset.y);
        // offset +=vec4(0.0,0.0,z_offset.x,z_offset.y);

        vec4 offset = vec4(0.0,0.0,0.0,x_offset.y);
        offset +=vec4(0.0,0.0,0.0,y_offset.y);
        offset +=vec4(0.0,0.0,0.0,z_offset.y);

        return base_sdf+offset*offset_size;
    }
    else{
        return base_sdf;
    }
}

)glsl";
