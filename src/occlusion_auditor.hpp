#pragma once

#include "lod_structures.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/variant/dictionary.hpp>

#include <vector>

using namespace godot;

// Occlusion at discovery, the debug side: what the hidden chunks hold, and the oracle that checks them.
class OcclusionAuditor {
public:
    // The hidden share of the loaded chunks and of their quads (live triangles / 2), per LOD band (near 0-3,
    // mid 4-6, far 7+) and in total. With occlusion_mode 1 (dry run) hidden chunks are meshed, so it is exact.
    static Dictionary hidden_stats(const LODHashSet &p_hidden, const ChunkHashMap &p_loaded);

    // The oracle (--occlusion_audit=N): up to p_count hidden chunks, spread over the distances, with rays from
    // each eye of p_eyes (LOD0 voxels, p_eye_base + offset; the scan's eye first) to 9 points of each chunk's
    // vertex-grid box. A ray is blocked before reaching the box by a cell whose 8 lattice corners are solid at
    // the LOD of the scanned chunk holding it (a surface-nets mesh is closed around it), by solid where the scan
    // culled, or by another hidden chunk (checked in its turn: the nearest hidden region is behind drawn solid).
    // Air is crossed by sphere tracing at twice the steepest occluder bound. A ray that reaches the box is a
    // miss; ok = no miss and no ray out of steps. CPU, rays spread over the WorkerThreadPool.
    static Dictionary audit(const SDFBase &p_sdf, const ChunkParam &p_param, const LODHashSet &p_scan, const LODHashSet &p_hidden,
            const ChunkHashMap &p_loaded, const VoxelPos &p_eye_base, const std::vector<Vector3> &p_eyes, int p_count);

    // What the real scene (trees included) hides from one eye (--ray_census=MIN_LOD): for up to p_per_lod scanned
    // chunks of each LOD from p_min_lod, the rays of audit() to 9 points of each, marched the same way (no hidden
    // chunk blocks). Per LOD: the share of rays blocked (what a depth test would cull) and of chunks with all 9
    // blocked (what chunk culling could skip; 9 rays are a sample, an upper bound).
    static Dictionary ray_census(const SDFBase &p_sdf, const ChunkParam &p_param, const LODHashSet &p_scan, const VoxelPos &p_eye_base,
            const Vector3 &p_eye, int p_min_lod, int p_per_lod);

    // Step 0 of occlusion by the meshed scene (--occlusion_estimate=MAX_LOD): what the scanned chunks' solid
    // blocks would hide from one eye. The occupancy the mesher would give (blocks of B cells whose (B + 1)^3
    // sample corners are solid) is computed on the CPU for the scanned chunks of LOD <= p_max_lod, merged into
    // boxes (OccluderSet), rasterised (OcclusionMap::add_box) and every scanned chunk tested, for blocks of 2, 4
    // and 7 cells and margins of 0, 0.25 and 1 m. Per configuration: the hidden chunks per LOD, the hidden share
    // of the loaded quads, and the build and test times (the runtime job's cost).
    static Dictionary occlusion_estimate(const SDFBase &p_sdf, const ChunkParam &p_param, const LODHashSet &p_scan, const ChunkHashMap &p_loaded,
            const VoxelPos &p_eye_base, const Vector3 &p_eye, int p_max_lod);
    // The scanned chunks occlusion_estimate's occluders (blocks of p_block cells) hide from one eye, one set per
    // margin.
    static std::vector<LODHashSet> hidden_by_blocks(const SDFBase &p_sdf, const ChunkParam &p_param, const LODHashSet &p_scan,
            const VoxelPos &p_eye_base, const Vector3 &p_eye, int p_max_lod, int p_block, const std::vector<double> &p_margins_m);
};
