#pragma once

#include "GPU_direct_mesher.hpp"
#include "GPU_mesher.hpp"
#include "atmosphere.hpp"
#include "chunk_discoverer.hpp"
#include "chunk_memory.hpp"
#include "clouds.hpp"
#include "debug_log.hpp"
#include "gpu_device.hpp"
#include "mesh_host.hpp"
#include "mesh_validator.hpp"
#include "page_auditor.hpp"
#include "page_host.hpp"
#include "sdf_cascade.hpp"

#include "ov/value.hpp"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace ov {

// The engine: one terrain world, driven frame by frame by its host (Godot's VoxelTerrain, the Dawn adapter). It owns
// the terrain (SDFCascade), the discoverer, a mesher and the loaded chunks (VoxelState); the host lends it a GPU device
// and draws the meshers' output (PageHost / MeshHost). Main thread only, like the host's frame loop.
class World {
public:
    enum PipelineMode {
        PIPELINE_GPU_LEGACY = 1,
        PIPELINE_GPU_DIRECT = 2,
    };

    // What the terrain is, set before start().
    struct Settings {
        PipelineMode pipeline_mode = PIPELINE_GPU_LEGACY;
        // 1 the Hermite cascade terrain, 2 its continents preset, 4 its flat preset under the shapes gallery.
        int terrain_mode = 1;
        // > 0: a planet of this radius (metres) centred on the voxel (0, 0, 0), its +y pole at y = radius (SDFCascade); 0 flat.
        double planet_radius_m = 0.0;
        // The volumetric atmosphere over the base (Atmosphere), its scale (AtmosphereMedium::for_base: 0 from the base).
        bool atmosphere = true;
        double atmosphere_scale = 0.0;
        // The volumetric clouds in it (Clouds; only with the atmosphere), their layer's scale as the atmosphere's.
        bool clouds = true;
        int legacy_batch_size = 1024;
        int lod_numbers = 15; // LODs 0 .. lod_numbers - 1 (at most ChunkMath::MAX_LOD_NUMBERS)
        ChunkParam chunk_param; // voxel_size and skirt_depth set by the host; the render origin by start() and recenter()
        VoxelPos world_origin_request; // LOD0 voxels, snapped down to the root chunk span by start()
        // recenter() moves the render origin to the camera once it is farther than this from it (metres, on any axis):
        // render positions stay small, so float precision holds anywhere in the world. 0: never (a fixed origin).
        float rebase_distance_m = 1024.0f;
        int archetype_instances = 0;
        double archetype_instance_area = 200.0;
        int archetype_trees = 0;
        double archetype_tree_area = 60.0;
        double biome_radius = 0.0;
        str climate_path; // a host path (ov::read_file), empty for none
        bool archetype_band = true;
        bool archetype_stream = true;
        int archetype_stream_part_sides = int(Archetype::ScatterSource::STREAM_PART_CELL_SIDES);
        bool archetype_moments = true;
        float archetype_erode = Archetype::ERODE_CELLS;
        float archetype_resolve = Archetype::ErodeRule().resolve;
        bool legacy_trim_indices = false;
        bool dual_contouring = false;
        float lod_pixel_error = 2.0f;
        float lod_pixel_error_back = 0.0f; // 0: lod_pixel_error everywhere
        std::vector<float> lod_pixel_errors; // per LOD, replaces lod_pixel_error (0 or past the end: lod_pixel_error)
        float lod_view_margin_deg = 20.0f;
        float lod_view_memory_m = -1.0f; // < 0: auto (ChunkDiscoverer::set_view_memory)
        GPUDirectMesher::Settings direct;
        int debug_view = 0;
        int discovery_cull_mode = ChunkDiscoverer::CULL_LEGACY;
        bool discovery_cache = true;
        bool discovery_verify = false;
        bool destroy_verify = false;
        int discoverer_mode = -1; // -1 auto: GPU (CPU when no local device can be made), 0 CPU, 1 GPU
        bool discovery_compare = false;
        bool walk_gpu = true;
        bool walk_verify = false;
        int walk_batch = int(SDFTable::WALK_BATCH_GPU);
    };

    // The camera's projection, which the LOD distances are measured against.
    struct Projection {
        float viewport_height = 1080.0f; // the 3D render's height in pixels
        float fov_rad = 0.0f; // vertical; 0: 75 degrees
        float aspect = 16.0f / 9.0f;
    };

    // What the host lends: the main device (its thread runs the meshers' GPU work, Device::post) and the drawers.
    struct Hosts {
        gpu::Device *device = nullptr;
        PageHost *pages = nullptr; // GPU Direct's pages
        MeshHost *meshes = nullptr; // the legacy meshes
        // The validator's legacy meshes: a MeshHost that draws nothing (debug_validate).
        std::function<std::unique_ptr<MeshHost>()> hidden_meshes;
        // The atmosphere's textures for the host's shaders, on the device's thread (Atmosphere::TexturesFn).
        Atmosphere::TexturesFn atmosphere_textures;
        // The clouds' textures, the same way (Clouds::TexturesFn).
        Clouds::TexturesFn cloud_textures;
    };

    World();
    ~World();
    World(const World &) = delete;
    World &operator=(const World &) = delete;

    Settings &settings() { return config; }
    const Settings &settings() const { return config; }

    // The terrain built and the discoverer and mesher started; false (an error) without a device or with bad settings.
    bool start(const Projection &p_projection, const Hosts &p_hosts, const Float3 *p_camera_pos, const Float3 *p_camera_forward,
            double p_camera_far_m = 0.0);
    // One frame: the camera (render space, null: none), the scans and meshes that landed, the next batches.
    // p_camera_far_m: the depth of the camera's far plane in metres (0: none, an infinite projection): nothing farther
    // than its frustum's corners reach (far / cos of the diagonal half-FOV), in any direction, is scanned (ChunkMath::beyond_far).
    void frame(const Float3 *p_camera_pos, const Float3 *p_camera_forward, double p_camera_far_m = 0.0);
    // Frees the loaded chunks, the mesher (its GPU objects on the device's thread) and the discoverer.
    void stop();
    bool is_started() const { return started; }
    // The direction towards the sun (render space, unit): the atmosphere's light from the next frame on.
    void set_sun(const Float3 &p_towards_sun) { sun = p_towards_sun; }
    // The atmosphere (null: off, or before start).
    const Atmosphere *get_atmosphere() const { return atmosphere; }
    // The clouds (null: off, or before start), and their weather: the cover in [0, 1] and the wind (render metres per
    // second), from the next frame on.
    const Clouds *get_clouds() const { return clouds; }
    void set_cloud_weather(float p_cover, const Float3 &p_wind_mps);

    SDFCascade &terrain() { return *sdf; }
    const SDFCascade &terrain() const { return *sdf; }
    const ChunkParam &chunk_param() const { return config.chunk_param; }
    uint32_t frames() const { return exec_time; }

    // True once every scanned chunk is meshed and the mesher has nothing in flight; p_camera_pos (the camera now, null:
    // none) must still be in the scanned camera chunk.
    bool is_load_complete(const Float3 *p_camera_pos);
    // The instance edits made so far are meshed (the legacy mesher: idle).
    bool is_edit_settled();
    int64_t pending_chunk_count() const { return int64_t(state.dirty_chunk.size()); }
    // Distance from the camera (metres) that bounds every chunk the discoverer can load (0 before start).
    double load_radius() const;
    // SDF (metres, LOD 0) at a render-space position: the camera's distance to the terrain.
    double sdf_at(const double p_render_pos[3]) const;
    // The LOD0 voxel holding a render-space position (an edit's anchor).
    VoxelPos voxel_at(const double p_render_pos[3]) const;
    // The render-space point (u, v) metres along the base's reference tangent plane (BaseSurface::tangent_point).
    void tangent_position(double p_u_m, double p_v_m, double r_render_pos[3]) const;
    // What the hosts add to their scenes' render positions (poses made for the flat world from its world origin, its
    // datum BaseSurface::DATUM_M up) so they stand the same over a planet's +y pole: 0 on the flat world. Before start
    // too (from the settings), so a host places its scene before the first scan.
    void scene_shift(double r_shift_m[3]) const;
    // The render origin's LOD0 voxel; before start, the world origin start() will take (start_origin).
    VoxelPos render_origin_voxel() const;
    // The base's up at a render-space position (BaseSurface::up_dir): +y on the flat world, the planet's radial direction.
    UpAxis up_at(const double p_render_pos[3]) const;
    // The base in render space (BaseSurface::render), for the shaders' altitude, up and level spans (glsl_render): it
    // changes with the render origin.
    BaseSurface::Render base_render() const;

    // Render-origin rebasing. Render space (every Float3 the host passes or gets) is metres from the render origin,
    // which starts at the world origin. Called by the host before frame(): when the camera is farther than
    // rebase_distance_m from the render origin, moves it under the camera (ChunkMath::recentered_origin) and returns
    // true with r_shift, the metres the host subtracts from every render position it holds (its camera, nodes, paths)
    // in the same frame. Deferred (false) while a validation or an audit holds render positions.
    bool recenter(const Float3 &p_camera_pos, Float3 &r_shift);
    // Moves the render origin to p_origin (even: ChunkMath::is_render_origin), r_shift as recenter's. The meshes
    // follow (the pages' bounds and decode, the legacy meshes' transforms); nothing is re-meshed.
    bool rebase(const ChunkId &p_origin, Float3 &r_shift);
    const ChunkId &render_origin() const { return config.chunk_param.origin_chunk; }
    // The world origin start() snapped (the render origin it started at).
    const ChunkId &world_origin() const { return world_origin_chunk; }

    void set_debug_view(int p_view);

    // Instance edits (archetype table): the anchor in LOD0 voxels, a 9-float row-major rotation and 8 params (null:
    // identity / zero), op a Combine. The chunks the edit can change are re-meshed, the discoverer rescans.
    int add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k);
    bool remove_instance(int p_id);
    bool move_instance(int p_id, const VoxelPos &p_anchor, const float *p_rotation);

    ov::Value debug_stats();
    // { section: total ms since start } of frame(), cheap enough to read every frame.
    ov::Value process_times_ms() const;

    // GPU Direct only: re-mesh p_count loaded chunks with the legacy mesher and compare triangle by triangle (the report
    // once done, empty until then). OBJ files of the first mismatching chunk go to p_dump_dir when it is not empty.
    bool debug_validate(int p_count, const str &p_dump_dir);
    const ov::Value &validation_report() const { return validation; }
    // GPU Direct only: read back every page holding loaded chunks and classify all drawn quad slots.
    bool debug_audit_pages();
    const ov::Value &audit_report() const { return audit; }
    // GPU Direct only: rebuild the current scan without culling and count the quads of every chunk the cull dropped (a
    // culled chunk with quads is a hole). Run it with the camera still and the load complete.
    bool debug_audit_discovery();
    const ov::Value &discovery_audit_report() const { return discovery_audit_result; }

private:
    Settings config;
    Hosts hosts;
    std::unique_ptr<SDFCascade> sdf;
    ChunkDiscoverer *chunk_discoverer = nullptr;
    GPUMesher *gpu_mesher = nullptr;
    GPUDirectMesher *direct_mesher = nullptr;
    Atmosphere *atmosphere = nullptr; // deletes itself on the device's thread (destroy)
    Clouds *clouds = nullptr; // the same; reads the atmosphere's textures, so goes first
    float cloud_cover = 0.5f;
    Float3 cloud_wind;
    Float3 sun{ 0.0f, 1.0f, 0.0f };
    MeshValidator *validator = nullptr;
    std::unique_ptr<MeshHost> validator_host;
    PageAuditor *auditor = nullptr;
    ov::Value validation = ov::Value::object();
    ov::Value audit = ov::Value::object();
    VoxelState state;

    bool started = false;
    // update_destroy is needed (a scan or mesher results arrived since it last ran).
    bool destroy_dirty = false;
    // View of the scan currently in state.scanned_chunk.
    ScanView scan_view;
    Float3 last_camera; // the camera of the last frame (and start), render space
    ChunkId world_origin_chunk;
    // The world origin start() takes: the request snapped down to the root chunk span (2^lod_numbers chunk-id units), so
    // the start is the same for any requested origin; on a planet a request of (0, 0, 0) stands for its pole
    // (pole_lift: the camera starts near the render origin there), any other is absolute. It is also the first render
    // origin; a rebase then moves that one by even ids only (ChunkMath::recentered_origin), so render-relative Morton keys
    // are no longer root-aligned after one.
    ChunkId start_origin() const;
    // The settings' planet base (SDFCascade::planet_base), the default flat one without a planet: what start() configures.
    BaseSurface settings_planet() const;
    int64_t rebases = 0;
    bool has_camera = false;
    double far_reach = 1.0; // the farthest point drawn / the far plane's depth: 1 / cos(the diagonal half-FOV)

    // Discovery audit (debug_audit_discovery): quad counts of every chunk the cull dropped.
    struct DiscoveryAudit {
        bool running = false;
        ov::Value base = ov::Value::object(); // sizes, lod jump: filled when it starts
        int64_t expected = 0;
        int64_t checked = 0;
        int64_t misses = 0;
        int64_t miss_quads = 0;
        ov::Value miss_examples = ov::Value::array();
        ov::Value misses_by_lod = ov::Value::object();
        uint64_t start_usec = 0;
    } discovery_audit;
    ov::Value discovery_audit_result = ov::Value::object();
    void _poll_discovery_audit();

    void _apply_instance_edit();
    void _free_validator();
    struct EditStats {
        int64_t edits = 0;
        int64_t remesh_total = 0;
        int64_t remesh_last = 0;
        TimingStat apply; // the edit + the re-mesh enqueue
    } edit_stats;

    uint32_t exec_time = 0;
    bool scan_received = false;
    int64_t frames_until_loaded = -1;
    uint64_t start_usec = 0;
    int64_t usec_until_loaded = -1;

    struct ProcessStats {
        TimingStat process_total;
        TimingStat stream_poll; // SDFBase::poll (the streamed keys landed, the stream's evictions)
        TimingStat scan_ingest; // get_scanned_chunks + add_scan_chunk + send_chunk
        TimingStat scan_apply; // part of scan_ingest: VoxelState::add_scan_chunk (the scan delta)
        TimingStat destroy_from_scan; // VoxelState::update_destroy
        TimingStat destroy_verify; // VoxelState::verify_against_full (destroy_verify)
        TimingStat mesher_dispatch; // process_queue
        // Parts of mesher_dispatch (the mesher's DispatchTimes, per frame).
        TimingStat gate_admit; // ChunkGate::admit
        TimingStat build_lists; // SDFGpuTable::build_lists
        TimingStat record_bytes; // SDFGpuTable::record_bytes
        TimingStat mesher_collect; // get_results
        TimingStat destroy_chunks; // VoxelState::destroy_marked
        TimingStat compact; // GPUDirectMesher::compact
        TimingStat coverage; // VoxelState::push_coverage

        // The sections by name, for the reports.
        std::vector<std::pair<const char *, const TimingStat *>> sections() const {
            return { { "total", &process_total }, { "stream_poll", &stream_poll }, { "scan_ingest", &scan_ingest }, { "scan_apply", &scan_apply },
                { "destroy_from_scan", &destroy_from_scan }, { "destroy_verify", &destroy_verify }, { "mesher_dispatch", &mesher_dispatch },
                { "gate_admit", &gate_admit }, { "build_lists", &build_lists }, { "record_bytes", &record_bytes },
                { "mesher_collect", &mesher_collect }, { "destroy_chunks", &destroy_chunks }, { "compact", &compact }, { "coverage", &coverage } };
        }
    } process_stats;
    DispatchTimes last_dispatch_times; // the mesher's, at the previous frame
    // The mesher's dispatch sub-timers since the previous frame, into their sections.
    void _record_dispatch_times(const DispatchTimes &p_now);
};

} // namespace ov
