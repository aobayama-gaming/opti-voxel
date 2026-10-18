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

// Dual contouring of Hermite data. Both meshers' vertex pass replaces its mass point with a call to
// qef_vertex (GPU_mesher / GPU_direct_mesher), so the placement is one function: the two meshers
// compute the same vertex, which is what MeshValidator's direct == legacy comparison relies on.
//
// The QEF
//
//     minimise  sum_i (n_i . (x - p_i))^2 + tau * |x - x0|^2      (x0 = the mass point)
//  =>  (sum_i n_i n_i^T + tau I) x = sum_i n_i (n_i . p_i) + tau x0
//
// places the vertex at the intersection of the crossing tangent planes, which is the crease/corner
// for a hard UNION / SUBTRACT, instead of the mass point (Surface Nets), which rounds it into a
// staircase. p_i is the crossing, n_i the Hermite normal there (qef_crossing_normal). tau is scaled
// to the QEF's spectrum (see the regularisation below), which keeps the solve well posed on a flat
// face (rank-1) and on a curved patch (where the plane intersection runs away), without an SVD.
inline constexpr const char GLSL_QEF_VERTEX[] = R"glsl(

// The Hermite normal at a crossing: the SDF gradient at the crossing point (a point query). It is
// the exact Hermite normal, and unlike the lerp of the cell's corner gradients it does not carry the
// crossing's t: the two meshers run on different RenderingDevices and compute t ~1e-3 apart, which a
// near-cancelling lerp amplifies into degrees of normal error (and the QEF then amplifies that).
// CascadeGLSL::fill_mesher_pass swaps the body for the corner-gradient lerp: a cascade only exists on
// lattice points, so it has no off-lattice evaluation.
vec3 qef_crossing_normal(ChunkFrame chunk, vec3 crossing, float t, vec3 ga, vec3 gb) {
    return skirt_sdf(chunk, crossing, chunk_param).xyz;
}

vec3 qef_vertex(ChunkFrame chunk, vec3 cell, uvec3 gid, vec3 mass_point, float edge_t[12]) {
    // Every float here is `precise`: the two meshers compile this same text into two different
    // shaders (different pipelines, and the legacy one on its own RenderingDevice), and without
    // `precise` the compilers may reassociate or fuse multiply-adds differently, which the
    // near-singular solve below would then amplify past the validator's tolerance.
    // The cell's 8 corner gradients, corner index c = x + 2 y + 4 z (the sample grid), for the
    // cascade fallback of qef_crossing_normal.
    precise vec3 corner_gradient[8] = vec3[8](
            sdf_sample[get_flat_index_z(gid + uvec3(0u, 0u, 0u))].xyz,
            sdf_sample[get_flat_index_z(gid + uvec3(1u, 0u, 0u))].xyz,
            sdf_sample[get_flat_index_z(gid + uvec3(0u, 1u, 0u))].xyz,
            sdf_sample[get_flat_index_z(gid + uvec3(1u, 1u, 0u))].xyz,
            sdf_sample[get_flat_index_z(gid + uvec3(0u, 0u, 1u))].xyz,
            sdf_sample[get_flat_index_z(gid + uvec3(1u, 0u, 1u))].xyz,
            sdf_sample[get_flat_index_z(gid + uvec3(0u, 1u, 1u))].xyz,
            sdf_sample[get_flat_index_z(gid + uvec3(1u, 1u, 1u))].xyz);

    // The 12 edges in the pass's order (e0, e2, e4, e6, e3, e1, e7, e5, e8, e9, e11, e10): the two
    // corner indices each. Edge 0-3 varies x, 4-7 y, 8-11 z.
    const int edge_corner_a[12] = int[12](0, 2, 4, 6, 0, 1, 4, 5, 0, 1, 2, 3);
    const int edge_corner_b[12] = int[12](1, 3, 5, 7, 2, 3, 6, 7, 4, 5, 6, 7);

    precise float a00 = 0.0, a01 = 0.0, a02 = 0.0;
    precise float a11 = 0.0, a12 = 0.0, a22 = 0.0;
    precise float b0 = 0.0, b1 = 0.0, b2 = 0.0;

    for (int e = 0; e < 12; e++) {
        const float t = edge_t[e];
        if (!(t > 0.0)) {
            continue;
        }
        const int ca = edge_corner_a[e];
        precise vec3 p = vec3(float(ca & 1), float((ca >> 1) & 1), float((ca >> 2) & 1));
        if (e < 4) {
            p.x += t;
        } else if (e < 8) {
            p.y += t;
        } else {
            p.z += t;
        }
        precise vec3 n = qef_crossing_normal(chunk, cell + p, t, corner_gradient[ca], corner_gradient[edge_corner_b[e]]);
        precise float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (!(len2 > 0.0)) {
            continue; // a flat gradient: no Hermite normal to place with
        }
        n *= inversesqrt(len2);

        a00 += n.x * n.x;
        a01 += n.x * n.y;
        a02 += n.x * n.z;
        a11 += n.y * n.y;
        a12 += n.y * n.z;
        a22 += n.z * n.z;
        precise float ndp = n.x * p.x + n.y * p.y + n.z * p.z;
        b0 += n.x * ndp;
        b1 += n.y * ndp;
        b2 += n.z * ndp;
    }

    // Regularise the QEF with a mass-point prior (Tikhonov): the minimiser of
    //
    //     sum_i (n_i . (x - p_i))^2 + tau * |x - x0|^2      (x0 = the mass point)
    //
    // is (A + tau I) x = b + tau x0 with A = sum n_i n_i^T, b = sum n_i (n_i . p_i). tau is scaled to
    // the QEF's own spectrum: the directions the crossing normals barely span (a curved patch's
    // tangential directions, where the tangent planes meet far outside the cell) fall back to the mass
    // point, while the directions they do span (a crease) still reach the plane intersection. A fixed
    // tiny tau leaves those weak directions unconstrained, so the vertex runs off to the tangent-plane
    // intersection: that is the spikes. This is a fixed-formula stand-in for clamping the QEF's small
    // singular values (no SVD, no eigenvectors).
    precise float qef_trace = (a00 + a11 + a22) / 3.0;
    precise float TAU = 0.15 * max(qef_trace, 1e-4);
    a00 += TAU;
    a11 += TAU;
    a22 += TAU;
    b0 += TAU * mass_point.x;
    b1 += TAU * mass_point.y;
    b2 += TAU * mass_point.z;

    precise float c00 = a11 * a22 - a12 * a12;
    precise float c01 = a12 * a02 - a01 * a22;
    precise float c02 = a01 * a12 - a11 * a02;
    precise float c11 = a00 * a22 - a02 * a02;
    precise float c12 = a01 * a02 - a00 * a12;
    precise float c22 = a00 * a11 - a01 * a01;
    precise float det = a00 * c00 + a01 * c01 + a02 * c02;
    if (!(abs(det) > 1e-12)) {
        return mass_point; // rank-deficient even with the prior: keep the mass point
    }
    precise vec3 x;
    x.x = (c00 * b0 + c01 * b1 + c02 * b2) / det;
    x.y = (c01 * b0 + c11 * b1 + c12 * b2) / det;
    x.z = (c02 * b0 + c12 * b1 + c22 * b2) / det;
    // A regularised QEF solution can still sit outside the cell on an under-sampled (coarse-LOD) cell
    // or a strongly curved patch; bound the move from the mass point to a fraction of the cell before
    // the AABB clamp, so a cell whose fit is bad cannot hurl its vertex across the cell (a jagged
    // silhouette) while a crease, which sits within a fraction of a cell of the mass point, is still
    // reached.
    precise vec3 move = x - mass_point;
    precise float move_len2 = move.x * move.x + move.y * move.y + move.z * move.z;
    const float move_cap = 0.25;
    if (move_len2 > move_cap * move_cap) {
        move *= move_cap * inversesqrt(move_len2);
    }
    precise vec3 candidate = clamp(mass_point + move, vec3(0.0), vec3(1.0));
    // A placed vertex can land where the SDF gradient vanishes (the pass then normalizes a zero
    // vector and the vertex normal is NaN: a hole). Keep the mass point there.
    precise vec3 vertex_gradient = qef_crossing_normal(chunk, cell + candidate, 0.0, corner_gradient[0], corner_gradient[0]);
    precise float vertex_gradient2 = vertex_gradient.x * vertex_gradient.x + vertex_gradient.y * vertex_gradient.y + vertex_gradient.z * vertex_gradient.z;
    if (!(vertex_gradient2 > 0.0)) {
        return mass_point;
    }
    return candidate;
}

)glsl";

// The mass-point line both vertex passes rewrite to qef_vertex when the terrain has dual contouring
// enabled. Kept next to the function so the two call sites cannot drift. The rewritten line is the
// division that turns the sum of the crossings into their mean, so the call divides: handed the sum,
// qef_vertex clamped the vertex onto the cell's far faces (terraces).
inline constexpr const char GLSL_MASS_POINT_VERTEX[] = "local_vertex /= float(active_edge_count);";
inline constexpr const char GLSL_QEF_VERTEX_CALL[] =
        "local_vertex = qef_vertex(chunk, vec3(cell_chunk_coordinate), gl_GlobalInvocationID, "
        "local_vertex / float(active_edge_count), "
        "float[12](e0_t, e2_t, e4_t, e6_t, e3_t, e1_t, e7_t, e5_t, e8_t, e9_t, e11_t, e10_t));";

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
