#pragma once

#include "godot_cpp/classes/node3d.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/packed_float32_array.hpp"
#include "godot_cpp/variant/variant.hpp"

#include "godot_cpp/classes/camera3d.hpp"

#include "godot_cpp/classes/editor_interface.hpp"
#include "godot_cpp/classes/sub_viewport.hpp"
#include "godot_cpp/classes/viewport.hpp"

#include "dig_tool.hpp"
#include "godot_mesh_host.hpp"
#include "godot_page_host.hpp"
#include "gpu_profiler.hpp"
#include "rd_device.hpp"
#include "world.hpp"

using namespace godot;

// The terrain node: Godot's host of an ov::World. Its properties are the world's settings (set before the node enters
// the tree); it lends the world the main RenderingDevice (RDDevice) and draws its output (GodotPageHost: GPU Direct's
// pages; GodotMeshHost: the legacy meshes), and each frame hands it the camera.
class VoxelTerrain : public Node3D {
    GDCLASS(VoxelTerrain, Node3D)

public:
    enum PipelineMode {
        PIPELINE_GPU_LEGACY = ov::World::PIPELINE_GPU_LEGACY,
        PIPELINE_GPU_DIRECT = ov::World::PIPELINE_GPU_DIRECT,
    };

protected:
    static void _bind_methods();

private:
    Camera3D *camera = nullptr;
    ov::World world;
    ov::DigTool dig_tool{ world };
    ov::World::Settings &cfg() { return world.settings(); }
    const ov::World::Settings &cfg() const { return world.settings(); }

    // The main RenderingDevice as the world's device (used on the render thread), and the hosts drawing its output.
    RDDevice *main_device = nullptr;
    GodotPageHost *page_host = nullptr; // GPU Direct's pages
    GodotMeshHost *mesh_host = nullptr; // the legacy meshes

    GPUProfiler *gpu_profiler = nullptr;
    bool debug_gpu_profile = false;
    bool debug_gpu_profile_frames = false; // record every captured frame (GPUProfiler::take_frames)
    Dictionary page_params; // page shader uniforms set from outside (VoxelSky's clouds), applied when the pages exist
    bool lod_far_limit = true; // hand the world the camera's far plane (set_lod_far_limit)

    void _teardown();
    // The camera's view axis in render space.
    Vector3 _camera_forward() const;
    // What the world gets as the camera's far plane: Camera3D.far, 0 (none) without a camera or the far limit.
    double _camera_far() const;
    bool _can_set_world_origin(int64_t p_voxel) const;
    // The render origin moved by p_shift metres (World::recenter): the Node3D siblings follow, then origin_shifted.
    void _move_render_origin(const Float3 &p_shift);
    // What follows the render origin outside the meshers: the sea level colours, the VoxelSky siblings' frame.
    void _push_render_frame();
    // The atmosphere's uniforms to the pages and the VoxelSky siblings (World::start; its textures once made).
    void _push_atmosphere(const Dictionary &p_params);
    // The atmosphere's textures (Atmosphere::TexturesFn, render thread): their RenderingServer RIDs, then
    // _push_atmosphere on the main thread; invalid ones free them.
    void _rt_atmosphere_textures(ov::gpu::Texture p_volume, ov::gpu::Texture p_multiscatter);
    RID atmosphere_rids[2]; // render thread
    float atmosphere_range_sent = -1.0f; // the last global atmo_range_m

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
    // A planet of this radius centred on the voxel (0, 0, 0) (ov::World::Settings::planet_radius_m); 0 the flat world.
    void set_planet_radius(double p_metres);
    double get_planet_radius() const;
    // The volumetric atmosphere over the base (ov::Atmosphere, World::Settings::atmosphere) and its scale (0: from the
    // base's curvature). Set before the node enters the tree.
    void set_atmosphere(bool p_enabled);
    bool get_atmosphere() const;
    void set_atmosphere_scale(double p_scale);
    double get_atmosphere_scale() const;
    // The direction towards the sun (render space; VoxelSky, from its light): the atmosphere's camera volume.
    void set_sun_direction(const Vector3 &p_towards_sun);
    // A climate table (JSON, src/climate.cpp's format: res:// or absolute), carried by the cascade and modulating the
    // ground's octaves per biome; empty for none (--climate=PATH). Set before the node enters the tree.
    void set_climate_path(const String &p_path);
    String get_climate_path() const;
    // The band rule for archetype instances (Archetype::set_band); false keeps every instance at every LOD
    // (the reference of the band report, --instance_band=0). Must be set before the node enters the tree.
    void set_archetype_band(bool p_enabled);
    bool get_archetype_band() const;
    // The scatter and the forest stream (default on: their keys computed as the chunks ask for them,
    // instance_stream.hpp); false places them whole at configure (--stream=0). Must be set before the node enters the tree.
    void set_archetype_stream(bool p_enabled);
    bool get_archetype_stream() const;
    // The streamed sources' moment layers read the records while their cell is under this many source cells, above it
    // proxies (default 1; 3 = the whole build's blobs, ScatterSource::PART_CELL_SIDES; --stream_part_sides). Must be set
    // before the node enters the tree.
    void set_archetype_stream_part_sides(int p_sides);
    int get_archetype_stream_part_sides() const;
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
    // The error of each LOD (index = LOD; 0 or past the end: lod_pixel_error), which sets how many of its chunks
    // load: e.g. {4, 4, 2, 1, 1} has fewer LOD 0-1 chunks and more LOD 3-4 ones. lod_pixel_error_back stays a ratio
    // to lod_pixel_error, applied on top. An error 2x coarser than the LOD below leaves that LOD no band (warned).
    void set_lod_pixel_errors(const PackedFloat32Array &p_errors);
    PackedFloat32Array get_lod_pixel_errors() const;
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
    // Scan nothing farther from the camera than its far plane (Camera3D.far, read every frame): it is never drawn.
    // With --lods above 15 the far plane follows the altitude (Presets.clip_planes), so a low camera drops the
    // coarsest LODs. Default on.
    void set_lod_far_limit(bool p_enabled);
    bool get_lod_far_limit() const;

    static constexpr int MAX_LOD_NUMBERS = ChunkMath::MAX_LOD_NUMBERS;
    void set_lod_numbers(int p_count);
    int get_lod_numbers() const;
    // Distance from the camera (metres) that bounds every chunk the discoverer can load: the corner of
    // its scan grid. 0 before the terrain is ready.
    double get_load_radius() const;
    // LOD0 voxels per chunk-id unit (half a LOD0 chunk, ChunkParam::half_chunk_voxels): for tools that rebuild
    // chunk boxes from their ids.
    int64_t get_half_chunk_voxels() const;
    // SDF (metres, LOD 0) at a render-space position: the camera's distance to the terrain, for the
    // clip planes above 15 LODs.
    double get_sdf_at(const Vector3 &p_render_pos) const;
    // The render-space point (u, v) metres along the base's reference tangent plane (BaseSurface::tangent_point; flat: x,
    // z from the world's (0, 0) column at the datum): the player's --spawn and the climate report's coordinates.
    Vector3 get_tangent_position(const Vector2 &p_tangent_m) const;
    // The base's up at a render-space position (World::up_at): +y on the flat world, a planet's local vertical. The player and the
    // bench stand the camera on it.
    Vector3 get_up_at(const Vector3 &p_render_pos) const;
    // World::scene_shift: the offset a scene's flat-world poses take to stand over a planet's pole (zero on the flat world).
    Vector3 get_scene_shift() const;

    // Render-origin rebasing (World::recenter): once the camera is farther than this many metres from the render
    // origin on any axis, the origin moves under it, every Node3D sibling of the terrain moves by the shift in the same
    // frame and origin_shifted(shift) is emitted for the render positions scripts hold (subtract it). 0: a fixed
    // origin. Default 1024. Set before the node enters the tree.
    void set_rebase_distance_m(float p_metres);
    float get_rebase_distance_m() const;
    // The render origin in LOD0 voxels (absolute, int64): render (0, 0, 0) is this voxel. The world origin until the
    // first rebase.
    int64_t get_render_origin_x() const;
    int64_t get_render_origin_y() const;
    int64_t get_render_origin_z() const;
    // Render metres from the world origin to the render origin: a render position plus this is the position the scene
    // had before any rebase (a float: for reports and replayable poses, not exact far away).
    Vector3 get_render_offset_m() const;

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
    // The engine trace (trace.hpp): every thread's spans and hand-offs, the GPU's passes. Start clears it; write
    // saves it as a Chrome trace and returns { events, dropped, bytes } (bytes -1 when the file can't be written).
    void debug_trace_start(int64_t p_max_events);
    void debug_trace_stop();
    Dictionary debug_trace_write(const String &p_path);
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
    // Where the discoverer evaluates the corners of its cull test: 0 CPU, 1 (and -1, auto) GPU (LocalPointsGPU, a
    // local device on the discoverer thread; CPU with a warning if it can't be created). Both give the same scans.
    void set_discoverer_mode(int p_mode);
    int get_discoverer_mode() const;
    // Debug: diff every scan against a fresh CPU scan, ERROR on any difference (--discovery_compare).
    void set_discovery_compare(bool p_enable);
    bool get_discovery_compare() const;
    // Whether the streamed sources' site walks read the ground on the GPU (FieldGPU, a local device on its own thread,
    // attached once ready; the CPU with a warning if it can't be created). Both give the same sites.
    void set_walk_gpu(bool p_enable);
    bool get_walk_gpu() const;
    // Debug: every batch of the GPU's readings read again on the CPU, ERROR on a difference (--walk_verify).
    void set_walk_verify(bool p_enable);
    bool get_walk_verify() const;
    // The walk keys a drain task walks together with the GPU field (SDFTable::WALK_BATCH_GPU by default).
    void set_walk_batch(int p_keys);
    int get_walk_batch() const;

    // GPU Direct only: 0 shaded, 1 normals, 2 lod, 3 page, 4 wireframe, 5 chunk, 6 group, 7 quad index.
    void set_debug_view(int p_view);
    int get_debug_view() const;
    // GPU Direct: page shader uniforms by name (VoxelSky hands the cloud layer's over), kept across the pool's creation.
    void set_page_params(const Dictionary &p_params);

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

    // The player's dig tools (ov::DigTool): origin and unit direction in render space. dig_spray / dig_tube every frame
    // their button is held, dig_release when it is released. set_dig_settings takes any of dig_radius, dig_length,
    // dig_k, spray_radius, drag_radius, drag_length (metres) and drag_union (bool: a solid tube).
    bool dig_hole(const Vector3 &p_origin, const Vector3 &p_dir);
    void dig_spray(const Vector3 &p_origin, const Vector3 &p_dir, double p_delta);
    void dig_tube(const Vector3 &p_origin, const Vector3 &p_dir);
    void dig_release();
    int get_dig_edits() const;
    void set_dig_settings(const Dictionary &p_settings);

    // True once every scanned chunk is meshed and the mesher has nothing in flight.
    bool is_load_complete();
    // The instance edits made so far are meshed (the legacy mesher: idle).
    bool is_edit_settled();

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
