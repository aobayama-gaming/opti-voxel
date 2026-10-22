#pragma once

#include "godot_cpp/classes/node3d.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/packed_float32_array.hpp"
#include "godot_cpp/variant/variant.hpp"

#include "chunk_discoverer.hpp"

#include "godot_cpp/classes/camera3d.hpp"

#include "godot_cpp/classes/editor_interface.hpp"
#include "godot_cpp/classes/sub_viewport.hpp"
#include "godot_cpp/classes/viewport.hpp"

#include <utility>
#include <vector>

#include "chunk_memory.hpp"

#include "GPU_direct_mesher.hpp"
#include "GPU_mesher.hpp"
#include "debug_log.hpp"
#include "gpu_profiler.hpp"
#include "mesh_validator.hpp"
#include "page_auditor.hpp"

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
    bool debug_gpu_profile_frames = false; // record every captured frame (GPUProfiler::take_frames)
    Dictionary validation_report;

    SDFCascade *sdf = memnew(SDFCascade); // the one terrain: the cascade ground under the instance table

    VoxelState visual_mesh_data;

    // Settings (must be set before the node enters the tree).
    PipelineMode pipeline_mode = PIPELINE_GPU_LEGACY;
    int terrain_mode = 1;
    int legacy_batch_size = 1024;
    // Number of LOD levels (0 .. lod_numbers - 1). The GPU reads a chunk's LOD from the low 32 bits
    // of its id (findLSB), so at most MAX_LOD_NUMBERS.
    int lod_numbers = 15;
    int archetype_instances = 0;
    double archetype_instance_area = 200.0;
    int archetype_trees = 0;
    double archetype_tree_area = 60.0;
    double biome_radius = 0.0;
    String climate_path;
    bool archetype_band = true;
    bool archetype_moments = true;
    float archetype_erode = Archetype::ERODE_CELLS;
    float archetype_resolve = Archetype::ErodeRule().resolve;
    bool legacy_trim_indices = false;
    // Dual contouring of Hermite data (DC-lite) in the meshers' vertex pass: sharp UNION / SUBTRACT
    // creases land on the crease instead of the Surface Nets mass point (gpu_glsl.hpp).
    bool dual_contouring = false;
    float lod_pixel_error = 2.0f;
    float lod_pixel_error_back = 0.0f; // 0: lod_pixel_error everywhere
    float lod_view_margin_deg = 20.0f;
    float lod_view_memory_m = -1.0f; // < 0: auto (ChunkDiscoverer::set_view_memory)
    VoxelPos world_origin_request; // LOD0 voxels, snapped in _ready
    GPUDirectMesher::Settings direct_settings;
    int debug_view = 0;
    int discovery_cull_mode = ChunkDiscoverer::CULL_LEGACY;
    bool discovery_cache = true;
    bool discovery_verify = false;
    bool destroy_verify = false;
    int discoverer_mode = -1; // -1 auto: GPU (CPU when no local device can be made)
    bool discovery_compare = false;

    bool started = false;

    // update_destroy is needed (a scan or mesher results arrived since it last ran).
    bool destroy_dirty = false;
    // View of the scan currently in visual_mesh_data.scanned_chunk.
    ScanView scan_view;

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

    // Re-mesh the loaded chunks an instance edit can change and invalidate the discoverer's cull cache.
    void _apply_instance_edit();
    // The camera's view axis in render space.
    Vector3 _camera_forward() const;
    struct EditStats {
        int64_t edits = 0;
        int64_t remesh_total = 0;
        int64_t remesh_last = 0;
        TimingStat apply; // the edit + the re-mesh enqueue
    } edit_stats;

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
        TimingStat coverage; // VoxelState::push_coverage

        // The sections by name, for the reports.
        std::vector<std::pair<const char *, const TimingStat *>> sections() const {
            return { { "total", &process_total }, { "scan_ingest", &scan_ingest }, { "scan_apply", &scan_apply },
                { "destroy_from_scan", &destroy_from_scan }, { "destroy_verify", &destroy_verify },
                { "mesher_dispatch", &mesher_dispatch }, { "mesher_collect", &mesher_collect },
                { "destroy_chunks", &destroy_chunks }, { "compact", &compact }, { "coverage", &coverage } };
        }
    } process_stats;

public:
    void _ready() override;
    void _process(double delta) override;
    void _exit_tree() override;

    void set_pipeline_mode(int p_mode);
    int get_pipeline_mode() const;

    // Terrain: 1 the Hermite cascade terrain (SDFCascade, default), 2 its continents preset, 4 its flat preset under
    // the shapes gallery (SDFTable::Scene::SHAPES, --terrain=shapes, res://shapes_scene.tscn). 0 was SDFDummy and 3
    // the archetype demo on an fBm ground (SDFArchetype), both removed: saved scenes keep their numbers.
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

    // A deterministic scatter of --instances rocks and pillars on the ground, in a square of side
    // --instance_area metres around SDFTable::SCENE_X / Z. Must be set before the node enters the tree.
    void set_archetype_instances(int p_count);
    int get_archetype_instances() const;
    void set_archetype_instance_area(double p_area);
    double get_archetype_instance_area() const;
    // A forest of about --trees grown trees (tree_archetype.hpp) in a square of side
    // --tree_area metres around SDFTable::SCENE_X / Z. Set before the node enters the tree.
    void set_archetype_trees(int p_count);
    int get_archetype_trees() const;
    void set_archetype_tree_area(double p_area);
    double get_archetype_tree_area() const;
    // The biome features (biome_features.hpp: lava lakes, floating islands) within this many metres of
    // SDFTable::SCENE_X / Z, 0 for none (--biomes=KM). Set before the node enters the tree.
    void set_biome_radius(double p_metres);
    double get_biome_radius() const;
    // A climate table (JSON, src/climate.cpp's format: res:// or absolute), carried by the cascade and modulating the
    // ground's octaves per biome; empty for none (--climate=PATH). Set before the node enters the tree.
    void set_climate_path(const String &p_path);
    String get_climate_path() const;
    // The band rule for archetype instances (Archetype::set_band); false keeps every instance at every LOD
    // (the reference of the band report, --instance_band=0). Must be set before the node enters the tree.
    void set_archetype_band(bool p_enabled);
    bool get_archetype_band() const;
    // The far field of the dropped instances (default on): their moments, per LOD, as blobs the coarse chunks fold into a
    // surface (moment_layers.hpp; --moments=0 to leave it out). Must be set before the node enters the tree.
    void set_archetype_moments(bool p_enabled);
    bool get_archetype_moments() const;
    // The band rule's erosion rate, cells per LOD (default 1, in (0, 4]): the skirt's depth and the fastest a part
    // shrinks before its drop (--instance_erode). Must be set before the node enters the tree.
    void set_archetype_erode(float p_cells);
    float get_archetype_erode() const;
    // The band rule's resolution, cells (default 1): a part is kept while its half thickness (its moments) holds this
    // many cells, and erodes only just before its drop (Archetype::set_band; --instance_resolve). Must be set before
    // the node enters the tree.
    void set_archetype_resolve(float p_cells);
    float get_archetype_resolve() const;

    // Screen-space error target of the LOD selection (higher = coarser = fewer chunks/triangles).
    void set_lod_pixel_error(float p_error);
    float get_lod_pixel_error() const;
    // The error away from the view: coarser LODs outside the view cone, from lod_pixel_error at its edge to
    // this one behind the camera (0 or <= lod_pixel_error: off). See lod_view_memory_m for turning.
    void set_lod_pixel_error_back(float p_error);
    float get_lod_pixel_error_back() const;
    // The view cone: the frustum's diagonal half-FOV plus this margin, in degrees (at least
    // ChunkDiscoverer::RESCAN_TURN_DEG, the most the remembered view axis lags the camera's). It keeps lod_pixel_error
    // around the view while turning (a turn of ChunkDiscoverer::RESCAN_TURN_DEG rescans), and the meshers
    // load the chunks inside it first.
    void set_lod_view_margin_deg(float p_degrees);
    float get_lod_view_margin_deg() const;
    // The views the camera had keep lod_pixel_error in their cone until the camera walks this many metres
    // from where it had them; past that they fade (the LOD coarsens with the extra distance). Turning in
    // place ends up with the front set all around and stops re-meshing; walking on coarsens what is behind
    // as the views fade. < 0 (default): the distance where LOD ChunkDiscoverer::VIEW_MEMORY_AUTO_LOD starts.
    // 0: only the current view (every turn re-meshes the cone's edges).
    void set_lod_view_memory_m(float p_metres);
    float get_lod_view_memory_m() const;

    static constexpr int MAX_LOD_NUMBERS = ChunkMath::MAX_LOD_NUMBERS;
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
    void set_debug_gpu_profile_frames(bool p_enable);
    bool get_debug_gpu_profile_frames() const;
    // The frames recorded since the last call (debug_gpu_profile_frames), see GPUProfiler::take_frames.
    Array take_gpu_profile_frames();
    // { section: total ms since start } of _process, cheap enough to read every frame.
    Dictionary get_process_times_ms() const;

    void set_direct_batch_size(int p_size);
    int get_direct_batch_size() const;
    void set_direct_pages_per_batch(int p_pages);
    int get_direct_pages_per_batch() const;
    void set_direct_batches_per_frame(int p_batches);
    int get_direct_batches_per_frame() const;
    void set_direct_gpu_budget_ms(float p_ms);
    float get_direct_gpu_budget_ms() const;
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
    // Where the discoverer evaluates the corners of its cull test: 0 CPU, 1 (and -1, auto) GPU (DiscovererGPU, a
    // local device on the discoverer thread; CPU with a warning if it can't be created). Both give the same scans.
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

    // Instance edits (archetype table). The anchor is in LOD0 voxels (int64, never through Vector3i);
    // rotation is a 9-float row-major basis and params up to 8 floats (empty: identity / zero). op is a
    // Combine (0 union, 1 smooth union, 2 subtract, 3 smooth subtract). Returns the new id or -1.
    // add_instance / move_instance re-mesh the chunks the edit can change and rescan the discoverer.
    int add_instance(int p_type, int64_t p_x, int64_t p_y, int64_t p_z, const PackedFloat32Array &p_rotation, const PackedFloat32Array &p_params,
            int p_op, float p_k);
    bool remove_instance(int p_id);
    bool move_instance(int p_id, int64_t p_x, int64_t p_y, int64_t p_z, const PackedFloat32Array &p_rotation);

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
