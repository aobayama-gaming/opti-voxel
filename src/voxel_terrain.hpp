#pragma once

#include "godot_cpp/classes/node3d.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/variant.hpp"

#include "chunk_discoverer.hpp"

#include "godot_cpp/classes/camera3d.hpp"

#include "godot_cpp/classes/editor_interface.hpp"
#include "godot_cpp/classes/sub_viewport.hpp"
#include "godot_cpp/classes/viewport.hpp"

#include <vector>

#include "chunk_memory.hpp"

#include "GPU_direct_mesher.hpp"
#include "GPU_mesher.hpp"
#include "debug_log.hpp"
#include "gpu_profiler.hpp"
#include "mesh_validator.hpp"
#include "page_auditor.hpp"

//debug
#include "sdf_archetype.hpp"
#include "sdf_cascade.hpp"

using namespace godot;

class VoxelTerrain : public Node3D {
    GDCLASS(VoxelTerrain, Node3D)

public:
    enum PipelineMode {
        PIPELINE_GPU_LEGACY = 1,
        PIPELINE_GPU_DIRECT = 2,
    };

protected:
    static void _bind_methods();

private:
    Camera3D *camera = nullptr;

    ChunkParam chunk_param; // ChunkParam's defaults; voxel_size and skirt_depth are properties

    ChunkDiscoverer *chunk_discoverer = nullptr;

    RID scenario;

    GPUMesher *gpu_mesher = nullptr;

    GPUDirectMesher *direct_mesher = nullptr;

    MeshValidator *validator = nullptr;
    GPUProfiler *gpu_profiler = nullptr;
    PageAuditor *auditor = nullptr;
    Dictionary audit_report;
    bool debug_gpu_profile = false;
    Dictionary validation_report;

    SDFBase *sdf = memnew(SDFCascade);

    VoxelState visual_mesh_data;

    // Settings (must be set before the node enters the tree).
    PipelineMode pipeline_mode = PIPELINE_GPU_LEGACY;
    int terrain_mode = 1;
    int legacy_batch_size = 1024;
    // Number of LOD levels (0 .. lod_numbers - 1). The GPU reads a chunk's LOD from the low 32 bits
    // of its id (findLSB), so at most MAX_LOD_NUMBERS.
    int lod_numbers = 15;
    bool legacy_trim_indices = false;
    // Dual contouring of Hermite data (DC-lite) in the meshers' vertex pass: sharp UNION / SUBTRACT
    // creases land on the crease instead of the Surface Nets mass point (gpu_glsl.hpp).
    bool dual_contouring = false;
    float lod_pixel_error = 2.0f;
    VoxelPos world_origin_request; // LOD0 voxels, snapped in _ready
    GPUDirectMesher::Settings direct_settings;
    int debug_view = 0;
    int discovery_cull_mode = ChunkDiscoverer::CULL_LEGACY;
    bool discovery_cache = true;
    bool discovery_verify = false;
    bool destroy_verify = false;
    int discoverer_mode = -1; // -1 auto: GPU for a cascade terrain, CPU otherwise
    bool discovery_compare = false;

    bool started = false;

    // update_destroy is needed (a scan or mesher results arrived since it last ran).
    bool destroy_dirty = false;
    // Camera chunk of the scan currently in visual_mesh_data.scanned_chunk.
    ChunkId scan_camera_chunk;

    // Discovery audit (debug_audit_discovery): quad counts of every chunk the cull dropped.
    struct DiscoveryAudit {
        bool running = false;
        Dictionary base; // sizes, lod jump: filled when it starts
        int64_t expected = 0;
        int64_t checked = 0;
        int64_t misses = 0;
        int64_t miss_quads = 0;
        Array miss_examples;
        Dictionary misses_by_lod;
        uint64_t start_usec = 0;
    } discovery_audit;
    Dictionary discovery_audit_report;
    void _poll_discovery_audit();

    void _teardown();

    bool _can_set_world_origin(int64_t p_voxel) const;

    //debug

    uint32_t exec_time = 0;
    bool scan_received = false;
    int64_t frames_until_loaded = -1;
    uint64_t start_usec = 0;
    int64_t usec_until_loaded = -1;

    struct ProcessStats {
        TimingStat process_total;
        TimingStat scan_ingest; // get_scanned_chunks + add_scan_chunk + send_chunk
        TimingStat scan_apply; // part of scan_ingest: VoxelState::add_scan_chunk (the scan delta)
        TimingStat destroy_from_scan; // VoxelState::update_destroy
        TimingStat destroy_verify; // VoxelState::verify_against_full (--destroy_verify)
        TimingStat mesher_dispatch; // process_queue
        TimingStat mesher_collect; // get_results
        TimingStat destroy_chunks; // VoxelState::destroy_marked
        TimingStat compact; // GPUDirectMesher::compact
    } process_stats;

public:
    void _ready() override;
    void _process(double delta) override;
    void _exit_tree() override;

    void set_pipeline_mode(int p_mode);
    int get_pipeline_mode() const;

    // Terrain: 1 the Hermite cascade terrain (SDFCascade, default), 2 its continents preset, 3 the archetype
    // IR's demo scene (SDFArchetype, --terrain=archetypes). 0 was SDFDummy (removed): saved scenes keep
    // their numbers.
    void set_terrain_mode(int p_mode);
    int get_terrain_mode() const;

    void set_legacy_batch_size(int p_size);
    int get_legacy_batch_size() const;

    // Edge length of a LOD 0 voxel in metres (each LOD level doubles it).
    void set_voxel_size(float p_size);
    float get_voxel_size() const;
    // Depth of the LOD-seam skirt, in voxels of each chunk's LOD (0 disables it).
    void set_skirt_depth(float p_depth);
    float get_skirt_depth() const;

    // Integer world origin in LOD0 voxels: Godot space (0, 0, 0) is this voxel of the integer frame
    // (plus the legacy lattice offset). Snapped down to a multiple of the root chunk span when the
    // terrain starts, so the LOD layout is the same around any origin. |value| <= 2^60.
    void set_world_origin_x(int64_t p_voxel);
    int64_t get_world_origin_x() const;
    void set_world_origin_y(int64_t p_voxel);
    int64_t get_world_origin_y() const;
    void set_world_origin_z(int64_t p_voxel);
    int64_t get_world_origin_z() const;

    // Screen-space error target of the LOD selection (higher = coarser = fewer chunks/triangles).
    void set_lod_pixel_error(float p_error);
    float get_lod_pixel_error() const;

    static constexpr int MAX_LOD_NUMBERS = 31;
    void set_lod_numbers(int p_count);
    int get_lod_numbers() const;
    // Distance from the camera (metres) that bounds every chunk the discoverer can load: the corner of
    // its scan grid. 0 before the terrain is ready.
    double get_load_radius() const;
    // SDF (metres, LOD 0) at a render-space position: the camera's distance to the terrain, for the
    // clip planes above 15 LODs.
    double get_sdf_at(const Vector3 &p_render_pos) const;

    void set_legacy_trim_indices(bool p_trim);
    bool get_legacy_trim_indices() const;

    // Dual contouring of Hermite data (DC-lite, both meshers): sharp UNION / SUBTRACT creases are
    // placed on the crease instead of the Surface Nets mass point. Off by default (Surface Nets).
    void set_dual_contouring(bool p_enable);
    bool get_dual_contouring() const;

    int get_pending_chunk_count() const;

    // Per-pass GPU timings in get_debug_stats()["gpu_breakdown"] (needs the viewport to measure its
    // render time, which is what makes the renderer capture timestamps).
    void set_debug_gpu_profile(bool p_enable);
    bool get_debug_gpu_profile() const;

    void set_direct_batch_size(int p_size);
    int get_direct_batch_size() const;
    void set_direct_pages_per_batch(int p_pages);
    int get_direct_pages_per_batch() const;
    void set_direct_batches_per_frame(int p_batches);
    int get_direct_batches_per_frame() const;
    void set_direct_max_in_flight(int p_batches);
    int get_direct_max_in_flight() const;
    void set_direct_page_capacity(int p_capacity);
    int get_direct_page_capacity() const;
    void set_direct_initial_pages(int p_pages);
    int get_direct_initial_pages() const;
    void set_direct_compaction(bool p_enable);
    bool get_direct_compaction() const;
    void set_direct_compact_max_extent(float p_chunks);
    float get_direct_compact_max_extent() const;

    // How the discoverer drops LOD tree nodes holding no surface (ChunkDiscoverer::CullMode).
    void set_discovery_cull_mode(int p_mode);
    int get_discovery_cull_mode() const;
    // Remember each LOD tree node's cull verdict across scans (exact: the SDF does not change).
    void set_discovery_cache(bool p_enable);
    bool get_discovery_cache() const;
    // Debug: recompute every cached verdict and report mismatches as errors.
    void set_discovery_verify(bool p_enable);
    bool get_discovery_verify() const;
    // Debug: check every scan delta against the full scans and every incremental destroy check against
    // the full recompute (VoxelState::verify_against_full), ERROR on a difference.
    void set_destroy_verify(bool p_enable);
    bool get_destroy_verify() const;
    // Where the discoverer evaluates the corners of its cull test: -1 auto (GPU when the terrain is a cascade,
    // CPU otherwise), 0 CPU, 1 GPU (DiscovererGPU, a local device on the discoverer thread; CPU with a warning
    // if it can't be created or the terrain has no cascade). Both give the same scans.
    void set_discoverer_mode(int p_mode);
    int get_discoverer_mode() const;
    // Debug: diff every scan against a fresh CPU scan, ERROR on any difference (--discovery_compare).
    void set_discovery_compare(bool p_enable);
    bool get_discovery_compare() const;

    // GPU Direct only: 0 shaded, 1 normals, 2 lod, 3 page, 4 wireframe, 5 chunk, 6 group, 7 quad index.
    void set_debug_view(int p_view);
    int get_debug_view() const;

    void set_log_level(int p_level);
    int get_log_level() const;

    // True once every scanned chunk is meshed and the mesher has nothing in flight.
    bool is_load_complete();

    Dictionary get_debug_stats();

    // GPU Direct only: re-mesh `count` loaded chunks with the legacy mesher and compare triangle by
    // triangle. Asynchronous: poll get_validation_report() (empty until done). OBJ files of the first
    // mismatching chunk are written to `dump_dir` when it is not empty.
    bool debug_validate(int p_count, const String &p_dump_dir);
    Dictionary get_validation_report() const;

    // GPU Direct only: read back every page holding loaded chunks and classify all drawn quad slots
    // (live / dead / stale / partial). Asynchronous: poll get_audit_report().
    bool debug_audit_pages();
    Dictionary get_audit_report() const;

    // GPU Direct only: rebuild the current scan without culling and count the quads of every chunk
    // the cull dropped (GPU, nothing loaded). A culled chunk with quads is a miss: a hole. Asynchronous:
    // poll get_discovery_audit_report(). Run it with the camera still and the load complete.
    bool debug_audit_discovery();
    Dictionary get_discovery_audit_report() const;

    VoxelTerrain() = default;
    ~VoxelTerrain() override;
};

VARIANT_ENUM_CAST(VoxelTerrain::PipelineMode);
