
#include "chunk_math.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#else
// GCC/Clang equivalent of the MSVC intrinsic: writes the index of the lowest set bit, returns 0 if mask is 0.
static inline unsigned char _BitScanForward(unsigned long *index, uint32_t mask) {
    if (mask == 0) {
        return 0;
    }
    *index = __builtin_ctz(mask);
    return 1;
}
#endif

constexpr inline float WORLD_NUDGE=0.123567f;

int32_t ChunkMath::get_parent_from_child(int32_t coord, int &lod) {
    // Get the center coordinate of the parent chunk.

    // 1. Find the index of the least significant bit (LSB).
    //    For 5 (101), the LSB is at index 0.
    //    For 6 (110), the LSB is at index 1.
    unsigned long lsb_index;
    if (_BitScanForward(&lsb_index, coord) == 0) {
        return 0; // If coord is 0, its parent is 0.
    }

    lod = lsb_index+1; // The LOD is the index of the LSB. +1 because we are going to output the lod of parent

    // 2. "set to 0 the least importance"
    //    Create a mask to clear that one bit.
    int32_t cleared_coord = coord & ~(1 << lsb_index);

    // 3. "flip to 1 the one after"
    //    Set the bit at the next higher index.
    int32_t parent_coord = cleared_coord | (1 << (lsb_index + 1));

    return parent_coord;
}

int32_t ChunkMath::get_parent_from_child(int32_t coord ) {

    int dummy_lod;
    return get_parent_from_child(coord, dummy_lod);
}

Vector3i ChunkMath::get_parent_from_child(Vector3i coord,int &lod) {
    return Vector3i(
        get_parent_from_child(coord.x, lod),
        get_parent_from_child(coord.y),
        get_parent_from_child(coord.z)
    );
}

Vector3i ChunkMath::get_parent_from_child(Vector3i coord) {
    return Vector3i(
        get_parent_from_child(coord.x),
        get_parent_from_child(coord.y),
        get_parent_from_child(coord.z)
    );
}

int32_t ChunkMath::get_lod(Vector3i coord) {
    // Get the LOD of the chunk at the given coordinate. The LOD is determined by the largest power of 2 from any coordinate.

    unsigned long lod_x = 0;
    if (_BitScanForward(&lod_x, coord.x) == 0) {
        return 0; // If coord.x is 0, it contributes no LOD.
    }

    return lod_x;
}

Vector3i ChunkMath::get_parent_from_child_until(Vector3i coord, int target_lod) {
    // Get the parent chunk at the target LOD. This is done by repeatedly getting the parent until the desired LOD is reached.

    Vector3i parent = coord;
    int current_lod = -1;

    while (current_lod < target_lod) {
        parent = get_parent_from_child(parent,current_lod);
    }

    return parent;
}

Vector3 ChunkMath::chunk_to_world(Vector3i chunk_coord,ChunkParam chunk_param = ChunkParam()) {
    // Convert chunk coordinates to world coordinates. This is done by multiplying the chunk coordinates by the chunk size and voxel size.

    return Vector3(
        (chunk_coord.x) * chunk_param.chunk_world_dimension()/2+WORLD_NUDGE,
        (chunk_coord.y) * chunk_param.chunk_world_dimension()/2+WORLD_NUDGE,
        (chunk_coord.z) * chunk_param.chunk_world_dimension()/2+WORLD_NUDGE
    );
}

float _direct_transform(float inputValue,float chunk_dim,ChunkParam chunk_param){

    return  (inputValue-chunk_param.skirt_size)*chunk_dim/chunk_param.chunk_voxel_dimension();

}

float _inverse_transform(float inputValue,float chunk_dim,ChunkParam chunk_param){

    return inputValue*chunk_param.chunk_voxel_dimension()/chunk_dim+chunk_param.skirt_size;

}

Vector3 ChunkMath::vertices_to_world(Vector3i chunk_coord, Vector3 local_coord,ChunkParam chunk_param = ChunkParam()) {
    // Convert a vertex coordinate inside a chunk to a world-space position.
    // Uses fixed-width edge scheme with _direct_transform.
    const float chunk_size = world_chunk_size(chunk_coord,chunk_param);
    const Vector3 chunk_center = chunk_to_world(chunk_coord,chunk_param);
    const Vector3 chunk_min_corner = chunk_center - Vector3(
        chunk_size * 0.5f,
        chunk_size * 0.5f,
        chunk_size * 0.5f
    );

    return chunk_min_corner + Vector3(
        _direct_transform(static_cast<float>(local_coord.x), chunk_size,chunk_param),
        _direct_transform(static_cast<float>(local_coord.y), chunk_size,chunk_param),
        _direct_transform(static_cast<float>(local_coord.z), chunk_size,chunk_param)
    );
}

Vector3 ChunkMath::world_to_vertices(Vector3i chunk_coord, Vector3 world_coord,ChunkParam chunk_param = ChunkParam()) {
    // Convert a world-space position back to the chunk's vertex-local coordinate system.
    // Uses fixed-width edge scheme with _inverse_transform.
    const float chunk_size = world_chunk_size(chunk_coord,chunk_param);
    const Vector3 chunk_center = chunk_to_world(chunk_coord,chunk_param);
    const Vector3 chunk_min_corner = chunk_center - Vector3(
        chunk_size * 0.5f,
        chunk_size * 0.5f,
        chunk_size * 0.5f
    );
    
    Vector3 local_offset = world_coord - chunk_min_corner;
    return Vector3(
        _inverse_transform(local_offset.x, chunk_size,chunk_param),
        _inverse_transform(local_offset.y, chunk_size,chunk_param),
        _inverse_transform(local_offset.z, chunk_size,chunk_param)
    );
}

float ChunkMath::world_chunk_size(Vector3i chunk_coord,ChunkParam chunk_param = ChunkParam()) {
    // Get the world size of the chunk at the given coordinate. The world size is determined by the chunk size multiplied by the LOD.
    int lod = ChunkMath::get_lod(chunk_coord);
    return (1 << lod) * chunk_param.chunk_world_dimension(); 
}

float ChunkMath::world_chunk_spherical_radius(Vector3i chunk_coord,ChunkParam chunk_param = ChunkParam()) {
    // Get the world spherical radius of the chunk at the given coordinate. This is used for distance calculations when determining which chunks to load.
    // The spherical radius is the distance from the center of the chunk to any of its corners, which can be calculated using the Pythagorean theorem in 3D.

    float half_size = world_chunk_size(chunk_coord,chunk_param) * 0.5f;
    return sqrt(half_size * half_size * 3); // sqrt(x^2 + y^2 + z^2) where x=y=z=half_size
}

Vector3i ChunkMath::world_to_chunk(Vector3 world_coord,ChunkParam chunk_param = ChunkParam()) {
    // Convert world coordinates to the nearest LOD0 chunk coordinates.

    const int32_t cell_x = static_cast<int32_t>(Math::floor(world_coord.x / chunk_param.chunk_world_dimension()));
    const int32_t cell_y = static_cast<int32_t>(Math::floor(world_coord.y / chunk_param.chunk_world_dimension()));
    const int32_t cell_z = static_cast<int32_t>(Math::floor(world_coord.z / chunk_param.chunk_world_dimension()));

    // LOD0 chunk centers are odd chunk coordinates: ..., -3, -1, 1, 3, ...
    return Vector3i(
        cell_x * 2 + 1,
        cell_y * 2 + 1,
        cell_z * 2 + 1
    );
}

bool ChunkMath::vertices_out_of_bound(Vector3 vertices_coord,ChunkParam chunk_param = ChunkParam()) {
    // Check if vertices coordinate is out of bounds in any dimension.
    // Valid range is [0, CHUNK_SIZE] in each dimension.
    
    return vertices_coord.x < 0.0f || vertices_coord.x > chunk_param.chunk_world_dimension() ||
           vertices_coord.y < 0.0f || vertices_coord.y > chunk_param.chunk_world_dimension() ||
           vertices_coord.z < 0.0f || vertices_coord.z > chunk_param.chunk_world_dimension();
}

int32_t ChunkMath::get_lod_size(Vector3i coord){
    return (1<<get_lod(coord));
}