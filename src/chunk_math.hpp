#pragma once

#include "world_coord.hpp"

#include <algorithm>

using namespace godot;

struct ChunkParam {
    int chunk_size = 32;
    float voxel_size = 0.5f;
    int skirt_size = 2;
    // Depth of the skirt in voxels of the chunk's own LOD: the overhang past the chunk bounds is
    // pushed this far into the solid, under the neighbour's surface, to hide LOD seams.
    float skirt_depth = 1.0f;
    // Integer world origin in chunk-id units (snapped to the root chunk span, see VoxelTerrain::_ready).
    // Godot space is the integer frame shifted by it: render positions are metres relative to the
    // origin, so they stay small wherever the world is.
    ChunkId origin_chunk;

    int chunk_voxel_dimension() const noexcept {
        return (chunk_size - skirt_size * 2);
    }

    float chunk_world_dimension() const noexcept {
        return chunk_voxel_dimension() * voxel_size;
    }

    // One chunk-id unit is half a LOD0 chunk (chunk_voxel_dimension must be even).
    int64_t half_chunk_voxels() const noexcept {
        return chunk_voxel_dimension() / 2;
    }

    VoxelPos origin_voxel() const noexcept {
        return origin_chunk * half_chunk_voxels();
    }
};

// Where a scan looks from, in chunk-id units (render axes are the chunk-id axes). The view cone has its
// apex at the camera chunk's centre and a half-angle of the frustum's diagonal half-FOV plus a margin. The
// LOD rule coarsens the nodes outside it (ChunkMath::lod_distance), the meshers load the chunks inside it
// first (ChunkMath::load_queue).
struct ScanView {
    ChunkId camera_chunk;
    Vector3 forward; // unit view axis; zero: no direction, every chunk is in the cone
    double cos_half_angle = -1.0;
    double sin_half_angle = 0.0;
    // How much coarser the LOD rule gets away from the cone: lod_pixel_error_back / lod_pixel_error - 1.
    // 0: the scanned set ignores the direction.
    double back_scale = 0.0;
    // The views the camera had (ChunkDiscoverer::remember_view): a camera chunk, a unit view axis, and a
    // fade = max(0, distance from the camera chunk to that one - the memory distance), so a view fades out
    // as the camera walks away from where it was. The LOD rule coarsens away from the union of their cones,
    // each pushed back by its fade. One of them is the camera chunk's own, within RESCAN_TURN_DEG of
    // `forward` (the only one without memory). Sorted by increasing fade. None (a view not made by
    // remember_view): the cone around `forward`.
    struct ViewMemory {
        ChunkId apex;
        Vector3 dir;
        double fade = 0.0;
        uint32_t used = 0; // memory_clock when the camera last had this view (eviction only, not compared)
    };
    static constexpr int MAX_VIEW_MEMORIES = 64;
    ViewMemory memories[MAX_VIEW_MEMORIES];
    int memory_count = 0;
    uint32_t memory_clock = 0; // remember_view calls

    void set_half_angle(double p_rad) {
        const double a = CLAMP(p_rad, 0.0, Math_PI);
        cos_half_angle = std::cos(a);
        sin_half_angle = std::sin(a);
    }
    void copy_memories_from(const ScanView &p_other) {
        memory_count = p_other.memory_count;
        memory_clock = p_other.memory_clock;
        std::copy(p_other.memories, p_other.memories + p_other.memory_count, memories);
    }
};

// Chunk ids are int64 chunk centres in chunk-id units (half a LOD0 chunk): LOD0 chunks sit on odd
// coordinates, a LOD-L chunk's coordinates have their lowest set bit at L and it spans 2^(L+1) units.
namespace ChunkMath {

// The most LODs a terrain has (0 .. MAX_LOD_NUMBERS - 1): the GPU reads a chunk's LOD from its id's low 32 bits.
inline constexpr int MAX_LOD_NUMBERS = 31;
// A LOD clamped into per-LOD stat arrays of MAX_LOD_NUMBERS.
constexpr int lod_slot(int p_lod) { return p_lod < 0 ? 0 : (p_lod >= MAX_LOD_NUMBERS ? MAX_LOD_NUMBERS - 1 : p_lod); }

int64_t get_parent_from_child(int64_t coord, int &lod);
int64_t get_parent_from_child(int64_t coord);

ChunkId get_parent_from_child(const ChunkId &coord, int &lod);
ChunkId get_parent_from_child(const ChunkId &coord);

int get_lod(const ChunkId &coord);
int64_t get_lod_size(const ChunkId &coord);

ChunkId get_parent_from_child_until(const ChunkId &coord, int target_lod);

// The sub-cubes of p_ancestor (Subcube::AXIS^3 of them, bits Subcube::bit) that its descendant
// p_descendant overlaps: one when at least two LODs finer, the 8 of an octant for a child (and below a LOD-1 ancestor,
// whose sub-cubes are smaller than a LOD0 chunk).
uint64_t subcube_bits(const ChunkId &p_ancestor, const ChunkId &p_descendant);

// Edge length of the chunk in metres (a size, not a position).
float world_chunk_size(const ChunkId &chunk_coord, const ChunkParam &chunk_param);

// LOD0 voxel of the chunk's minimum corner (vertex grid index skirt_size) and of its centre.
VoxelPos chunk_min_voxel(const ChunkId &chunk_coord, const ChunkParam &chunk_param);
VoxelPos chunk_center_voxel(const ChunkId &chunk_coord, const ChunkParam &chunk_param);
// The LOD-p_lod chunk whose span [chunk_min_voxel, + span) holds LOD0 voxel p_voxel.
ChunkId chunk_at_voxel(const VoxelPos &p_voxel, int p_lod, const ChunkParam &chunk_param);

// Render space (Godot space): metres relative to the world origin. Only these helpers produce metres.
Vector3 chunk_render_center(const ChunkId &chunk_coord, const ChunkParam &chunk_param);
// An LOD0 voxel (its lower corner).
Vector3 voxel_to_render(const VoxelPos &p_voxel, const ChunkParam &chunk_param);
// A point of the chunk's vertex grid (the grid the SDF is sampled on, skirt included).
Vector3 grid_to_render(const ChunkId &chunk_coord, const Vector3 &grid, const ChunkParam &chunk_param);
// Same from a page header, which only carries the low 32 bits of the id (mirrors the page shader).
Vector3 grid_to_render_lo32(const Vector3i &chunk_lo, const Vector3 &grid, const ChunkParam &chunk_param);
// LOD0 chunk holding a render-space position.
ChunkId chunk_at_render(const Vector3 &render_pos, const ChunkParam &chunk_param);

// Bounding sphere radius of a LOD-p_lod chunk in chunk-id units (its half extent is 2^lod).
float chunk_bounding_radius(int p_lod);
// Distance from the bounding sphere of `p_chunk` (LOD p_lod) to the view cone, in chunk-id units; 0 when
// they meet. 1-Lipschitz in the chunk's centre, like any distance to a set.
double view_cone_gap(const ChunkId &p_chunk, int p_lod, const ScanView &p_view);
// The LOD rule's gap: the least, over p_view.memories, of the gap to the memory's cone (apex at its camera
// chunk) plus its fade; view_cone_gap without memories. Each term is a distance plus a constant, so it is
// still 1-Lipschitz in the chunk's centre.
double lod_cone_gap(const ChunkId &p_chunk, int p_lod, const ScanView &p_view);
// The distance the LOD rule compares to its LOD's threshold: from the camera chunk to the chunk's bounding
// sphere, plus back_scale times its gap to the view cones (lod_cone_gap). In the cone it is the plain distance; far behind,
// (1 + back_scale) times it: the thresholds of lod_pixel_error_back. It changes by at most (1 + back_scale)
// per unit of position, so face neighbours stay within one LOD (unit-tested at error 1 | 4).
double lod_distance(const ChunkId &p_chunk, int p_lod, const ScanView &p_view);
// lod_distance(p_chunk, p_lod, p_view) < p_threshold, bit for bit, without the full minimum over the memories:
// the plain distance alone may already fail, and any one memory under the threshold is enough.
bool lod_split(const ChunkId &p_chunk, int p_lod, const ScanView &p_view, double p_threshold);
// The half-angle of the view cone: the frustum's diagonal half-FOV (p_fov_rad vertical, p_aspect width /
// height) plus p_margin_rad.
float view_cone_half_angle(float p_fov_rad, float p_aspect, float p_margin_rad);
// The meshers' load queue of a chunk, popped in increasing order: the chunks in the view cone first, then
// finest LOD first. In [0, load_queue_count(p_lod_count)).
int load_queue(const ChunkId &p_chunk, int p_lod_count, const ScanView &p_view);
// The number of load queues.
constexpr int load_queue_count(int p_lod_count) {
    return 2 * p_lod_count;
}

// Absolute position of a vertex-grid point: CPU mirror of world_pos_from_grid (gpu_frame.hpp).
WorldPos sample_pos(const ChunkId &chunk_coord, const Float3 &grid, const ChunkParam &chunk_param);
} //namespace ChunkMath
