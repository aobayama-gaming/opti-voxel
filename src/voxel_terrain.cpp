#include "voxel_terrain.hpp"
#include "rd_utils.hpp"
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

void VoxelTerrain::_destroy_chunk() {
    for (ChunkHashMap::Iterator it = visual_mesh_data.to_destroy_chunk.begin();
            it != visual_mesh_data.to_destroy_chunk.end();) {
        const ChunkId chunk_pos = it->key;
        ChunkOutput output = it->value;

        output.free();

        visual_mesh_data.loaded_chunk.erase(chunk_pos);
        visual_mesh_data.to_destroy_chunk.erase(chunk_pos);

        it = visual_mesh_data.to_destroy_chunk.begin();
    }
}

void VoxelTerrain::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_pipeline_mode", "mode"), &VoxelTerrain::set_pipeline_mode);
    ClassDB::bind_method(D_METHOD("get_pipeline_mode"), &VoxelTerrain::get_pipeline_mode);
    ClassDB::bind_method(D_METHOD("set_legacy_batch_size", "size"), &VoxelTerrain::set_legacy_batch_size);
    ClassDB::bind_method(D_METHOD("get_legacy_batch_size"), &VoxelTerrain::get_legacy_batch_size);
    ClassDB::bind_method(D_METHOD("set_voxel_size", "size"), &VoxelTerrain::set_voxel_size);
    ClassDB::bind_method(D_METHOD("get_voxel_size"), &VoxelTerrain::get_voxel_size);
    ClassDB::bind_method(D_METHOD("set_world_origin_x", "voxel"), &VoxelTerrain::set_world_origin_x);
    ClassDB::bind_method(D_METHOD("get_world_origin_x"), &VoxelTerrain::get_world_origin_x);
    ClassDB::bind_method(D_METHOD("set_world_origin_y", "voxel"), &VoxelTerrain::set_world_origin_y);
    ClassDB::bind_method(D_METHOD("get_world_origin_y"), &VoxelTerrain::get_world_origin_y);
    ClassDB::bind_method(D_METHOD("set_world_origin_z", "voxel"), &VoxelTerrain::set_world_origin_z);
    ClassDB::bind_method(D_METHOD("get_world_origin_z"), &VoxelTerrain::get_world_origin_z);
    ClassDB::bind_method(D_METHOD("set_debug_test_rig", "enable"), &VoxelTerrain::set_debug_test_rig);
    ClassDB::bind_method(D_METHOD("get_debug_test_rig"), &VoxelTerrain::get_debug_test_rig);
    ClassDB::bind_method(D_METHOD("set_debug_zero_pillar", "mode"), &VoxelTerrain::set_debug_zero_pillar);
    ClassDB::bind_method(D_METHOD("get_debug_zero_pillar"), &VoxelTerrain::get_debug_zero_pillar);
    ClassDB::bind_method(D_METHOD("set_lod_pixel_error", "error"), &VoxelTerrain::set_lod_pixel_error);
    ClassDB::bind_method(D_METHOD("get_lod_pixel_error"), &VoxelTerrain::get_lod_pixel_error);
    ClassDB::bind_method(D_METHOD("set_lod_numbers", "count"), &VoxelTerrain::set_lod_numbers);
    ClassDB::bind_method(D_METHOD("get_lod_numbers"), &VoxelTerrain::get_lod_numbers);
    ClassDB::bind_method(D_METHOD("get_load_radius"), &VoxelTerrain::get_load_radius);
    ClassDB::bind_method(D_METHOD("set_skirt_depth", "depth"), &VoxelTerrain::set_skirt_depth);
    ClassDB::bind_method(D_METHOD("get_skirt_depth"), &VoxelTerrain::get_skirt_depth);
    ClassDB::bind_method(D_METHOD("set_legacy_trim_indices", "trim"), &VoxelTerrain::set_legacy_trim_indices);
    ClassDB::bind_method(D_METHOD("get_legacy_trim_indices"), &VoxelTerrain::get_legacy_trim_indices);
    ClassDB::bind_method(D_METHOD("get_pending_chunk_count"), &VoxelTerrain::get_pending_chunk_count);
    ClassDB::bind_method(D_METHOD("set_debug_gpu_profile", "enable"), &VoxelTerrain::set_debug_gpu_profile);
    ClassDB::bind_method(D_METHOD("get_debug_gpu_profile"), &VoxelTerrain::get_debug_gpu_profile);
    ClassDB::bind_method(D_METHOD("set_direct_batch_size", "size"), &VoxelTerrain::set_direct_batch_size);
    ClassDB::bind_method(D_METHOD("get_direct_batch_size"), &VoxelTerrain::get_direct_batch_size);
    ClassDB::bind_method(D_METHOD("set_direct_pages_per_batch", "pages"), &VoxelTerrain::set_direct_pages_per_batch);
    ClassDB::bind_method(D_METHOD("get_direct_pages_per_batch"), &VoxelTerrain::get_direct_pages_per_batch);
    ClassDB::bind_method(D_METHOD("set_direct_page_capacity", "capacity"), &VoxelTerrain::set_direct_page_capacity);
    ClassDB::bind_method(D_METHOD("get_direct_page_capacity"), &VoxelTerrain::get_direct_page_capacity);
    ClassDB::bind_method(D_METHOD("set_direct_initial_pages", "pages"), &VoxelTerrain::set_direct_initial_pages);
    ClassDB::bind_method(D_METHOD("get_direct_initial_pages"), &VoxelTerrain::get_direct_initial_pages);
    ClassDB::bind_method(D_METHOD("set_direct_compaction", "enable"), &VoxelTerrain::set_direct_compaction);
    ClassDB::bind_method(D_METHOD("get_direct_compaction"), &VoxelTerrain::get_direct_compaction);
    ClassDB::bind_method(D_METHOD("set_direct_compact_max_extent", "chunks"), &VoxelTerrain::set_direct_compact_max_extent);
    ClassDB::bind_method(D_METHOD("get_direct_compact_max_extent"), &VoxelTerrain::get_direct_compact_max_extent);
    ClassDB::bind_method(D_METHOD("set_debug_view", "view"), &VoxelTerrain::set_debug_view);
    ClassDB::bind_method(D_METHOD("get_debug_view"), &VoxelTerrain::get_debug_view);
    ClassDB::bind_method(D_METHOD("set_log_level", "level"), &VoxelTerrain::set_log_level);
    ClassDB::bind_method(D_METHOD("get_log_level"), &VoxelTerrain::get_log_level);
    ClassDB::bind_method(D_METHOD("is_load_complete"), &VoxelTerrain::is_load_complete);
    ClassDB::bind_method(D_METHOD("get_debug_stats"), &VoxelTerrain::get_debug_stats);
    ClassDB::bind_method(D_METHOD("debug_validate", "count", "dump_dir"), &VoxelTerrain::debug_validate, DEFVAL(String()));
    ClassDB::bind_method(D_METHOD("get_validation_report"), &VoxelTerrain::get_validation_report);
    ClassDB::bind_method(D_METHOD("debug_audit_pages"), &VoxelTerrain::debug_audit_pages);
    ClassDB::bind_method(D_METHOD("get_audit_report"), &VoxelTerrain::get_audit_report);
    ClassDB::bind_method(D_METHOD("debug_audit_discovery"), &VoxelTerrain::debug_audit_discovery);
    ClassDB::bind_method(D_METHOD("get_discovery_audit_report"), &VoxelTerrain::get_discovery_audit_report);
    ClassDB::bind_method(D_METHOD("set_discovery_cull_mode", "mode"), &VoxelTerrain::set_discovery_cull_mode);
    ClassDB::bind_method(D_METHOD("get_discovery_cull_mode"), &VoxelTerrain::get_discovery_cull_mode);
    ClassDB::bind_method(D_METHOD("set_discovery_cache", "enable"), &VoxelTerrain::set_discovery_cache);
    ClassDB::bind_method(D_METHOD("get_discovery_cache"), &VoxelTerrain::get_discovery_cache);
    ClassDB::bind_method(D_METHOD("set_discovery_verify", "enable"), &VoxelTerrain::set_discovery_verify);
    ClassDB::bind_method(D_METHOD("get_discovery_verify"), &VoxelTerrain::get_discovery_verify);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "pipeline_mode", PROPERTY_HINT_ENUM, "GPU Legacy:1,GPU Direct:2"), "set_pipeline_mode", "get_pipeline_mode");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "legacy_batch_size", PROPERTY_HINT_RANGE, "1,4096"), "set_legacy_batch_size", "get_legacy_batch_size");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size", PROPERTY_HINT_RANGE, "0.01,8,0.01,suffix:m"), "set_voxel_size", "get_voxel_size");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_pixel_error", PROPERTY_HINT_RANGE, "0.25,64,0.25"), "set_lod_pixel_error", "get_lod_pixel_error");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_numbers", PROPERTY_HINT_RANGE, "1,31"), "set_lod_numbers", "get_lod_numbers");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "skirt_depth", PROPERTY_HINT_RANGE, "0,4,0.05,suffix:voxels"), "set_skirt_depth", "get_skirt_depth");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "legacy_trim_indices"), "set_legacy_trim_indices", "get_legacy_trim_indices");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "discovery_cull_mode", PROPERTY_HINT_ENUM, "None:0,Legacy:1"), "set_discovery_cull_mode", "get_discovery_cull_mode");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "discovery_cache"), "set_discovery_cache", "get_discovery_cache");
    ADD_GROUP("World Origin", "world_origin_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_x", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_x", "get_world_origin_x");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_y", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_y", "get_world_origin_y");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_z", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_z", "get_world_origin_z");
    ADD_GROUP("GPU Direct", "direct_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_batch_size", PROPERTY_HINT_RANGE, "1,1024"), "set_direct_batch_size", "get_direct_batch_size");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_pages_per_batch", PROPERTY_HINT_RANGE, "1,256"), "set_direct_pages_per_batch", "get_direct_pages_per_batch");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_page_capacity", PROPERTY_HINT_RANGE, "1024,1048576"), "set_direct_page_capacity", "get_direct_page_capacity");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_initial_pages", PROPERTY_HINT_RANGE, "0,4096"), "set_direct_initial_pages", "get_direct_initial_pages");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "direct_compaction"), "set_direct_compaction", "get_direct_compaction");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "direct_compact_max_extent", PROPERTY_HINT_RANGE, "1,1000000"), "set_direct_compact_max_extent", "get_direct_compact_max_extent");
    ADD_GROUP("Debug", "");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_view", PROPERTY_HINT_ENUM, "Shaded,Normals,LOD,Page,Wireframe,Chunk,Group,Quad index"), "set_debug_view", "get_debug_view");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "log_level", PROPERTY_HINT_RANGE, "0,3"), "set_log_level", "get_log_level");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_test_rig"), "set_debug_test_rig", "get_debug_test_rig");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_zero_pillar", PROPERTY_HINT_ENUM, "Off,Exact top,Top +1 cm"), "set_debug_zero_pillar", "get_debug_zero_pillar");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_gpu_profile"), "set_debug_gpu_profile", "get_debug_gpu_profile");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "discovery_verify"), "set_discovery_verify", "get_discovery_verify");

    BIND_ENUM_CONSTANT(PIPELINE_GPU_LEGACY);
    BIND_ENUM_CONSTANT(PIPELINE_GPU_DIRECT);
}

void VoxelTerrain::set_pipeline_mode(int p_mode) {
    ERR_FAIL_COND_MSG(started, "pipeline_mode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_mode != PIPELINE_GPU_LEGACY && p_mode != PIPELINE_GPU_DIRECT,
            "pipeline_mode must be 1 (GPU Legacy) or 2 (GPU Direct).");
    pipeline_mode = (PipelineMode)p_mode;
}

int VoxelTerrain::get_pipeline_mode() const {
    return pipeline_mode;
}

void VoxelTerrain::set_legacy_batch_size(int p_size) {
    ERR_FAIL_COND_MSG(started, "legacy_batch_size must be set before the terrain enters the tree.");
    legacy_batch_size = p_size;
}

int VoxelTerrain::get_legacy_batch_size() const {
    return legacy_batch_size;
}

void VoxelTerrain::set_voxel_size(float p_size) {
    ERR_FAIL_COND_MSG(started, "voxel_size must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_size <= 0.0f, "voxel_size must be positive.");
    chunk_param.voxel_size = p_size;
}

float VoxelTerrain::get_voxel_size() const {
    return chunk_param.voxel_size;
}

void VoxelTerrain::set_discovery_cull_mode(int p_mode) {
    ERR_FAIL_COND_MSG(started, "discovery_cull_mode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_mode < ChunkDiscoverer::CULL_NONE || p_mode > ChunkDiscoverer::CULL_LEGACY, "Unknown discovery_cull_mode.");
    discovery_cull_mode = p_mode;
}

int VoxelTerrain::get_discovery_cull_mode() const {
    return discovery_cull_mode;
}

void VoxelTerrain::set_discovery_cache(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "discovery_cache must be set before the terrain enters the tree.");
    discovery_cache = p_enable;
}

bool VoxelTerrain::get_discovery_cache() const {
    return discovery_cache;
}

void VoxelTerrain::set_discovery_verify(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "discovery_verify must be set before the terrain enters the tree.");
    discovery_verify = p_enable;
}

bool VoxelTerrain::get_discovery_verify() const {
    return discovery_verify;
}

void VoxelTerrain::set_skirt_depth(float p_depth) {
    ERR_FAIL_COND_MSG(started, "skirt_depth must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_depth < 0.0f, "skirt_depth must not be negative.");
    chunk_param.skirt_depth = p_depth;
}

float VoxelTerrain::get_skirt_depth() const {
    return chunk_param.skirt_depth;
}

bool VoxelTerrain::_can_set_world_origin(int64_t p_voxel) const {
    ERR_FAIL_COND_V_MSG(started, false, "world_origin must be set before the terrain enters the tree.");
    ERR_FAIL_COND_V_MSG(p_voxel > WorldCoord::MAX_ABS_ORIGIN_VOXEL || p_voxel < -WorldCoord::MAX_ABS_ORIGIN_VOXEL, false,
            "world_origin must be within +-2^60 voxels.");
    return true;
}

void VoxelTerrain::set_world_origin_x(int64_t p_voxel) {
    if (_can_set_world_origin(p_voxel)) {
        world_origin_request.x = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_x() const {
    return world_origin_request.x;
}

void VoxelTerrain::set_world_origin_y(int64_t p_voxel) {
    if (_can_set_world_origin(p_voxel)) {
        world_origin_request.y = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_y() const {
    return world_origin_request.y;
}

void VoxelTerrain::set_world_origin_z(int64_t p_voxel) {
    if (_can_set_world_origin(p_voxel)) {
        world_origin_request.z = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_z() const {
    return world_origin_request.z;
}

void VoxelTerrain::set_debug_test_rig(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "debug_test_rig must be set before the terrain enters the tree.");
    debug_test_rig = p_enable;
}

bool VoxelTerrain::get_debug_test_rig() const {
    return debug_test_rig;
}

void VoxelTerrain::set_debug_zero_pillar(int p_mode) {
    ERR_FAIL_COND_MSG(started, "debug_zero_pillar must be set before the terrain enters the tree.");
    debug_zero_pillar = p_mode;
}

int VoxelTerrain::get_debug_zero_pillar() const {
    return debug_zero_pillar;
}

void VoxelTerrain::set_lod_pixel_error(float p_error) {
    ERR_FAIL_COND_MSG(started, "lod_pixel_error must be set before the terrain enters the tree.");
    lod_pixel_error = p_error;
}

float VoxelTerrain::get_lod_pixel_error() const {
    return lod_pixel_error;
}

void VoxelTerrain::set_lod_numbers(int p_count) {
    ERR_FAIL_COND_MSG(started, "lod_numbers must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_count < 1 || p_count > MAX_LOD_NUMBERS, vformat("lod_numbers must be in [1, %d].", MAX_LOD_NUMBERS));
    lod_numbers = p_count;
}

int VoxelTerrain::get_lod_numbers() const {
    return lod_numbers;
}

double VoxelTerrain::get_load_radius() const {
    if (chunk_discoverer == nullptr) {
        return 0.0;
    }
    // The camera is within one step of its top-LOD ancestor, the grid spans `count` steps on each
    // side of it and each top chunk reaches one step (its half extent) past its centre.
    int64_t step;
    int64_t count;
    chunk_discoverer->get_scan_grid(step, count);
    const double ids = double((count + 2) * step) * std::sqrt(3.0);
    return ids * double(chunk_param.half_chunk_voxels()) * double(chunk_param.voxel_size);
}

void VoxelTerrain::set_legacy_trim_indices(bool p_trim) {
    ERR_FAIL_COND_MSG(started, "legacy_trim_indices must be set before the terrain enters the tree.");
    legacy_trim_indices = p_trim;
}

bool VoxelTerrain::get_legacy_trim_indices() const {
    return legacy_trim_indices;
}

void VoxelTerrain::set_debug_gpu_profile(bool p_enable) {
    debug_gpu_profile = p_enable;
}

bool VoxelTerrain::get_debug_gpu_profile() const {
    return debug_gpu_profile;
}

int VoxelTerrain::get_pending_chunk_count() const {
    return visual_mesh_data.dirty_chunk.size();
}

void VoxelTerrain::set_direct_batch_size(int p_size) {
    ERR_FAIL_COND_MSG(started, "direct_batch_size must be set before the terrain enters the tree.");
    direct_settings.batch_size = p_size;
}

int VoxelTerrain::get_direct_batch_size() const {
    return direct_settings.batch_size;
}

void VoxelTerrain::set_direct_pages_per_batch(int p_pages) {
    ERR_FAIL_COND_MSG(started, "direct_pages_per_batch must be set before the terrain enters the tree.");
    direct_settings.pages_per_batch = p_pages;
}

int VoxelTerrain::get_direct_pages_per_batch() const {
    return direct_settings.pages_per_batch;
}

void VoxelTerrain::set_direct_page_capacity(int p_capacity) {
    ERR_FAIL_COND_MSG(started, "direct_page_capacity must be set before the terrain enters the tree.");
    direct_settings.page_capacity = p_capacity;
}

int VoxelTerrain::get_direct_page_capacity() const {
    return direct_settings.page_capacity;
}

void VoxelTerrain::set_direct_initial_pages(int p_pages) {
    ERR_FAIL_COND_MSG(started, "direct_initial_pages must be set before the terrain enters the tree.");
    direct_settings.initial_pages = p_pages;
}

int VoxelTerrain::get_direct_initial_pages() const {
    return direct_settings.initial_pages;
}

void VoxelTerrain::set_direct_compaction(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "direct_compaction must be set before the terrain enters the tree.");
    direct_settings.compaction = p_enable;
}

bool VoxelTerrain::get_direct_compaction() const {
    return direct_settings.compaction;
}

void VoxelTerrain::set_direct_compact_max_extent(float p_chunks) {
    ERR_FAIL_COND_MSG(started, "direct_compact_max_extent must be set before the terrain enters the tree.");
    direct_settings.compact_max_extent = p_chunks;
}

float VoxelTerrain::get_direct_compact_max_extent() const {
    return direct_settings.compact_max_extent;
}

void VoxelTerrain::set_debug_view(int p_view) {
    debug_view = p_view;
    if (direct_mesher) {
        direct_mesher->set_debug_view(debug_view);
    }
}

int VoxelTerrain::get_debug_view() const {
    return debug_view;
}

void VoxelTerrain::set_log_level(int p_level) {
    DebugLog::level.store(p_level);
}

int VoxelTerrain::get_log_level() const {
    return DebugLog::level.load();
}

void VoxelTerrain::_ready() {
    // One chunk-id unit is half a LOD0 chunk, a whole number of voxels.
    ERR_FAIL_COND_MSG(chunk_param.chunk_voxel_dimension() % 2 != 0, "VoxelTerrain: chunk_size - 2 * skirt_size must be even.");
    ERR_FAIL_COND_MSG(lod_numbers < 1 || lod_numbers > MAX_LOD_NUMBERS, "VoxelTerrain: lod_numbers out of range.");

    started = true;
    start_usec = now_usec();

    // The world origin, snapped down to the root chunk span (2^lod_numbers chunk-id units): the
    // octree around the camera, the LOD layout and the GPU low words are then the same at any origin.
    {
        const int64_t root_units = int64_t(1) << lod_numbers;
        const int64_t root_voxels = root_units * chunk_param.half_chunk_voxels();
        chunk_param.origin_chunk = ChunkId(WorldCoord::floor_div(world_origin_request.x, root_voxels) * root_units,
                WorldCoord::floor_div(world_origin_request.y, root_voxels) * root_units,
                WorldCoord::floor_div(world_origin_request.z, root_voxels) * root_units);
        if (chunk_param.origin_voxel() != world_origin_request) {
            VLOG(1, "VoxelTerrain: world origin snapped to ", WorldCoord::to_string(chunk_param.origin_voxel()), " voxels");
        }
    }

    // sdf is always an SDFDummy for now (see the member initialiser).
    static_cast<SDFDummy *>(sdf)->test_rig = debug_test_rig;
    static_cast<SDFDummy *>(sdf)->zero_pillar = debug_zero_pillar;
    sdf->configure(chunk_param, lod_numbers);

    visual_mesh_data.set_max_lod(lod_numbers);

    if (Engine::get_singleton()->is_editor_hint()) {
        camera = EditorInterface::get_singleton()->get_editor_viewport_3d()->get_camera_3d();
    } else {
        camera = get_viewport()->get_camera_3d();
    }

    chunk_discoverer = memnew(ChunkDiscoverer);

    chunk_discoverer->set_config(chunk_param, sdf);
    chunk_discoverer->set_cull_mode((ChunkDiscoverer::CullMode)discovery_cull_mode);
    chunk_discoverer->set_cache(discovery_cache, discovery_verify);

    // The pixel error is measured against the camera's real vertical FOV and viewport height.
    float viewport_height = 1080.0f;
    float fov_rad = Math::deg_to_rad(75.0f);
    if (camera != nullptr) {
        const Vector2 viewport_size = camera->get_viewport()->get_visible_rect().size;
        viewport_height = MAX(viewport_size.y, 1.0f);
        fov_rad = Math::deg_to_rad(float(camera->get_fov()));
        if (camera->get_keep_aspect_mode() == Camera3D::KEEP_WIDTH && viewport_size.x > 0.0f) {
            // fov is horizontal: convert it to vertical.
            fov_rad = 2.0f * std::atan(std::tan(fov_rad * 0.5f) * viewport_size.y / viewport_size.x);
        }
    }

    chunk_discoverer->setup_lod(
            lod_numbers,
            lod_pixel_error,
            viewport_height,
            fov_rad);

    chunk_discoverer->start();

    scenario = get_world_3d()->get_scenario();

    if (pipeline_mode == PIPELINE_GPU_DIRECT) {
        direct_mesher = memnew(GPUDirectMesher);
        direct_mesher->initialize(sdf, chunk_param, lod_numbers, scenario, direct_settings);
        direct_mesher->set_debug_view(debug_view);
    } else {
        gpu_mesher = memnew(GPUMesher);

        gpu_mesher->initialize(
                sdf,
                chunk_param,
                legacy_batch_size, // chunk_batch
                lod_numbers,
                scenario);
        gpu_mesher->set_trim_indices(legacy_trim_indices);
    }

    if (camera != nullptr) {
        chunk_discoverer->set_camera_chunk(camera->get_global_position());
    }

    VLOG(1, "VoxelTerrain: started, pipeline_mode=", (int)pipeline_mode);
}

void VoxelTerrain::_process(double) {
    if (!started) {
        return;
    }

    const uint64_t t_process_start = now_usec();

    if (camera != nullptr) {
        chunk_discoverer->set_camera_chunk(camera->get_global_position());
    }

    uint64_t t0 = now_usec();
    std::unique_ptr<LODHashSet> new_scanned_chunk;

    bool dirty_scan = chunk_discoverer->get_scanned_chunks(new_scanned_chunk, scan_camera_chunk);

    if (dirty_scan) {
        scan_received = true;
        destroy_dirty = true;

        visual_mesh_data.add_scan_chunk(std::move(new_scanned_chunk));
        if (direct_mesher) {
            direct_mesher->send_chunk(visual_mesh_data.dirty_chunk);
        } else {
            gpu_mesher->send_chunk(visual_mesh_data.dirty_chunk);
        }
    }
    process_stats.scan_ingest.add(now_usec() - t0);

    // Only a new scan or new results change what can be destroyed.
    if (destroy_dirty) {
        destroy_dirty = false;
        t0 = now_usec();
        visual_mesh_data.update_destroy_from_scan();
        process_stats.destroy_from_scan.add(now_usec() - t0);
    }

    t0 = now_usec();
    if (direct_mesher) {
        direct_mesher->process_queue();
    } else {
        gpu_mesher->process_queue();
    }
    process_stats.mesher_dispatch.add(now_usec() - t0);

    t0 = now_usec();
    const bool results = direct_mesher ? direct_mesher->get_results(visual_mesh_data) : gpu_mesher->get_results(visual_mesh_data);
    destroy_dirty = destroy_dirty || results;
    process_stats.mesher_collect.add(now_usec() - t0);

    t0 = now_usec();
    _destroy_chunk();
    process_stats.destroy_chunks.add(now_usec() - t0);

    // After destruction (to_destroy_chunk holds copies of the outputs it moves), and never while the
    // validator or auditor hold page ranges.
    t0 = now_usec();
    if (direct_mesher && !validator && !auditor) {
        direct_mesher->compact(visual_mesh_data);
    }
    process_stats.compact.add(now_usec() - t0);

    if (debug_gpu_profile && exec_time % 15 == 0) {
        if (!gpu_profiler) {
            gpu_profiler = memnew(GPUProfiler);
        }
        gpu_profiler->poll();
    }

    if (auditor && auditor->step()) {
        audit_report = auditor->get_report();
        auditor->destroy();
        auditor = nullptr;
        VLOG(1, "VoxelTerrain: page audit: ", audit_report);
    }

    if (discovery_audit.running) {
        _poll_discovery_audit();
    }

    if (validator && validator->step()) {
        validation_report = validator->get_report();
        validator->destroy();
        validator = nullptr;
        VLOG(1, "VoxelTerrain: validation report: ", validation_report);
    }

    exec_time++;
    process_stats.process_total.add(now_usec() - t_process_start);

    if (frames_until_loaded < 0 && is_load_complete()) {
        frames_until_loaded = exec_time;
        usec_until_loaded = now_usec() - start_usec;
        VLOG(1, "VoxelTerrain: load complete after ", exec_time, " frames, ", usec_until_loaded / 1000.0, " ms");
    }
}

bool VoxelTerrain::debug_validate(int p_count, const String &p_dump_dir) {
    ERR_FAIL_COND_V_MSG(!direct_mesher, false, "debug_validate needs pipeline_mode = GPU Direct.");
    ERR_FAIL_COND_V_MSG(validator, false, "A validation is already running.");
    std::vector<MeshValidator::Target> candidates;
    for (const KeyValue<ChunkId, ChunkOutput> &kv : visual_mesh_data.loaded_chunk) {
        if (kv.value.owner) {
            MeshValidator::Target t;
            t.chunk_id = kv.key;
            t.page = kv.value.page;
            t.offset = kv.value.quad_offset;
            t.count = kv.value.quad_count;
            candidates.push_back(t);
        }
    }
    ERR_FAIL_COND_V_MSG(candidates.empty(), false, "No non-empty loaded chunk to validate.");
    // Evenly spaced sample (hash map order is arbitrary but stable enough for a spread of LODs).
    std::vector<MeshValidator::Target> targets;
    const int n = MIN(p_count, (int)candidates.size());
    for (int i = 0; i < n; i++) {
        targets.push_back(candidates[(size_t)i * candidates.size() / n]);
    }
    validation_report = Dictionary();
    validator = memnew(MeshValidator);
    validator->start(targets, sdf, chunk_param, lod_numbers, direct_mesher->get_pool(), p_dump_dir);
    return true;
}

Dictionary VoxelTerrain::get_validation_report() const {
    return validation_report;
}

bool VoxelTerrain::debug_audit_pages() {
    ERR_FAIL_COND_V_MSG(!direct_mesher, false, "debug_audit_pages needs pipeline_mode = GPU Direct.");
    ERR_FAIL_COND_V_MSG(auditor, false, "An audit is already running.");
    HashMap<int, int> index_of_page;
    std::vector<PageAuditor::PageInfo> pages;
    for (const KeyValue<ChunkId, ChunkOutput> &kv : visual_mesh_data.loaded_chunk) {
        if (!kv.value.owner) {
            continue;
        }
        if (!index_of_page.has(kv.value.page)) {
            index_of_page[kv.value.page] = pages.size();
            PageAuditor::PageInfo info;
            info.page = kv.value.page;
            pages.push_back(info);
        }
        pages[index_of_page[kv.value.page]].live.push_back({ kv.value.quad_offset, kv.value.quad_count, kv.key });
    }
    audit_report = Dictionary();
    auditor = memnew(PageAuditor);
    auditor->start(direct_mesher->get_pool(), pages, chunk_param);
    return true;
}

Dictionary VoxelTerrain::get_audit_report() const {
    return audit_report;
}

// LOD of the scanned chunk holding the LOD0 chunk `p_point` (odd coordinates), or -1.
static int scanned_lod_at(const LODHashSet &p_scan, const ChunkId &p_point, int p_max_lod) {
    for (int lod = 0; lod <= p_max_lod; lod++) {
        const int64_t span = int64_t(2) << lod; // a LOD-lod chunk spans 2^(lod+1) units
        auto centre = [span, lod](int64_t v) { return WorldCoord::floor_div(v, span) * span + (int64_t(1) << lod); };
        if (p_scan.has(ChunkId(centre(p_point.x), centre(p_point.y), centre(p_point.z)))) {
            return lod;
        }
    }
    return -1;
}

bool VoxelTerrain::debug_audit_discovery() {
    ERR_FAIL_COND_V_MSG(!direct_mesher, false, "debug_audit_discovery needs pipeline_mode = GPU Direct.");
    ERR_FAIL_COND_V_MSG(discovery_audit.running, false, "A discovery audit is already running.");
    ERR_FAIL_COND_V_MSG(!scan_received, false, "No scan received yet.");

    const uint64_t t0 = now_usec();
    const LODHashSet &scanned = *visual_mesh_data.scanned_chunk;
    LODHashSet reference;
    ChunkDiscoverer::ScanCounters counters;
    chunk_discoverer->scan(scan_camera_chunk, ChunkDiscoverer::CULL_NONE, reference, counters);

    int64_t scanned_not_in_ref = 0;
    for (const ChunkId &id : scanned) {
        if (!reference.has(id)) {
            scanned_not_in_ref++;
        }
    }
    std::vector<ChunkId> culled;
    for (const ChunkId &id : reference) {
        if (!scanned.has(id)) {
            culled.push_back(id);
        }
    }

    // LOD difference across chunk faces (the skirt hides a jump of 1). One probe per face, just past
    // its minimum corner: the LOD0 chunk there, then whichever scanned chunk holds it.
    const int max_lod = chunk_discoverer->get_max_lod();
    int max_jump = 0;
    Dictionary jumps; // jump -> faces
    for (const ChunkId &id : scanned) {
        const int lod = ChunkMath::get_lod(id);
        const int64_t half = int64_t(1) << lod;
        const ChunkId corner = id - ChunkId(half - 1, half - 1, half - 1); // LOD0 chunk in the min corner
        for (int axis = 0; axis < 3; axis++) {
            for (int side = -1; side <= 1; side += 2) {
                ChunkId probe = corner;
                int64_t *c = axis == 0 ? &probe.x : (axis == 1 ? &probe.y : &probe.z);
                const int64_t centre = axis == 0 ? id.x : (axis == 1 ? id.y : id.z);
                *c = side < 0 ? centre - half - 1 : centre + half + 1;
                const int other = scanned_lod_at(scanned, probe, max_lod);
                if (other < 0) {
                    continue;
                }
                const int jump = ABS(other - lod);
                max_jump = MAX(max_jump, jump);
                jumps[jump] = int64_t(jumps.get(jump, 0)) + 1;
            }
        }
    }

    discovery_audit = DiscoveryAudit();
    discovery_audit.running = true;
    discovery_audit.start_usec = t0;
    discovery_audit.expected = (int64_t)culled.size();
    Dictionary base;
    base["camera_chunk"] = WorldCoord::to_string(scan_camera_chunk);
    base["cull_mode"] = discovery_cull_mode;
    base["reference"] = (int64_t)reference.size();
    base["reference_nodes_visited"] = counters.nodes_visited;
    base["scanned"] = (int64_t)scanned.size();
    base["culled"] = (int64_t)culled.size();
    base["scanned_not_in_reference"] = scanned_not_in_ref;
    base["max_lod_jump"] = max_jump;
    base["lod_jump_faces"] = jumps;
    base["reference_ms"] = (now_usec() - t0) / 1000.0;
    discovery_audit.base = base;
    discovery_audit_report = Dictionary();

    direct_mesher->audit_count_chunks(culled);
    VLOG(1, "VoxelTerrain: discovery audit started: reference ", (int64_t)reference.size(), " chunks, scanned ",
            (int64_t)scanned.size(), ", checking ", (int64_t)culled.size(), " culled chunks");
    return true;
}

void VoxelTerrain::_poll_discovery_audit() {
    std::vector<std::pair<ChunkId, uint32_t>> counts;
    direct_mesher->take_audit_counts(counts);
    for (const std::pair<ChunkId, uint32_t> &c : counts) {
        discovery_audit.checked++;
        if (c.second == 0) {
            continue;
        }
        discovery_audit.misses++;
        discovery_audit.miss_quads += c.second;
        const int lod = ChunkMath::get_lod(c.first);
        discovery_audit.misses_by_lod[lod] = int64_t(discovery_audit.misses_by_lod.get(lod, 0)) + 1;
        if (discovery_audit.miss_examples.size() < 20) {
            Dictionary e;
            e["id"] = WorldCoord::to_string(c.first);
            e["lod"] = lod;
            e["quads"] = (int64_t)c.second;
            e["distance_chunks"] = scan_camera_chunk.distance_to(c.first);
            discovery_audit.miss_examples.push_back(e);
        }
    }
    if (!direct_mesher->is_audit_idle()) {
        return;
    }
    Dictionary r = discovery_audit.base.duplicate();
    r["checked"] = discovery_audit.checked;
    r["misses"] = discovery_audit.misses;
    r["miss_quads"] = discovery_audit.miss_quads;
    r["misses_by_lod"] = discovery_audit.misses_by_lod;
    r["miss_examples"] = discovery_audit.miss_examples;
    r["ms"] = (now_usec() - discovery_audit.start_usec) / 1000.0;
    r["ok"] = discovery_audit.misses == 0 && discovery_audit.checked == discovery_audit.expected &&
            int64_t(r["scanned_not_in_reference"]) == 0;
    discovery_audit_report = r;
    discovery_audit.running = false;
    VLOG(1, "VoxelTerrain: discovery audit: ", r);
}

Dictionary VoxelTerrain::get_discovery_audit_report() const {
    return discovery_audit_report;
}

bool VoxelTerrain::is_load_complete() {
    if (!started || !scan_received || !visual_mesh_data.dirty_chunk.is_empty()) {
        return false;
    }
    if (direct_mesher) {
        return direct_mesher->is_idle();
    }
    return gpu_mesher->is_idle();
}

Dictionary VoxelTerrain::get_debug_stats() {
    Dictionary d;
    if (!started) {
        return d;
    }

    d["pipeline_mode"] = (int)pipeline_mode;
    Dictionary origin;
    origin["requested_voxel"] = WorldCoord::to_string(world_origin_request);
    origin["voxel"] = WorldCoord::to_string(chunk_param.origin_voxel());
    origin["chunk"] = WorldCoord::to_string(chunk_param.origin_chunk);
    d["world_origin"] = origin;
    d["frames"] = (int64_t)exec_time;
    d["load_complete"] = is_load_complete();
    d["frames_until_loaded"] = frames_until_loaded;
    d["ms_until_loaded"] = usec_until_loaded < 0 ? -1.0 : usec_until_loaded / 1000.0;

    int64_t live_triangles = 0;
    int64_t drawn_triangles = 0;
    int64_t non_empty_chunks = 0;
    for (const KeyValue<ChunkId, ChunkOutput> &kv : visual_mesh_data.loaded_chunk) {
        live_triangles += kv.value.triangle_count;
        drawn_triangles += kv.value.drawn_triangles;
        non_empty_chunks += kv.value.triangle_count > 0 ? 1 : 0;
    }

    Dictionary chunks;
    chunks["scanned"] = (int64_t)visual_mesh_data.scanned_chunk->size();
    chunks["loaded"] = (int64_t)visual_mesh_data.loaded_chunk.size();
    chunks["loaded_non_empty"] = non_empty_chunks;
    chunks["pending"] = (int64_t)visual_mesh_data.dirty_chunk.size();
    // Settle check: once the load is complete both must be 0 (a scanned chunk not loaded is a hole).
    int64_t loaded_not_scanned = 0;
    int64_t scanned_not_loaded = 0;
    for (const KeyValue<ChunkId, ChunkOutput> &kv : visual_mesh_data.loaded_chunk) {
        loaded_not_scanned += visual_mesh_data.scanned_chunk->has(kv.key) ? 0 : 1;
    }
    for (const ChunkId &id : *visual_mesh_data.scanned_chunk) {
        scanned_not_loaded += visual_mesh_data.loaded_chunk.has(id) ? 0 : 1;
    }
    chunks["loaded_not_scanned"] = loaded_not_scanned;
    chunks["scanned_not_loaded"] = scanned_not_loaded;
    chunks["live_triangles"] = live_triangles;
    chunks["drawn_triangles"] = drawn_triangles;
    d["chunks"] = chunks;

    Dictionary process;
    process["total"] = process_stats.process_total.to_dict();
    process["scan_ingest"] = process_stats.scan_ingest.to_dict();
    process["destroy_from_scan"] = process_stats.destroy_from_scan.to_dict();
    process["mesher_dispatch"] = process_stats.mesher_dispatch.to_dict();
    process["mesher_collect"] = process_stats.mesher_collect.to_dict();
    process["destroy_chunks"] = process_stats.destroy_chunks.to_dict();
    process["compact"] = process_stats.compact.to_dict();
    d["process"] = process;

    if (gpu_profiler) {
        d["gpu_breakdown"] = gpu_profiler->get_breakdown();
    }
    d["chunk_rids_alive"] = DebugLog::chunk_rids_created.load() - DebugLog::chunk_rids_freed.load();
    d["discoverer"] = chunk_discoverer->get_stats();
    d["mesher"] = direct_mesher ? direct_mesher->get_stats() : gpu_mesher->get_stats();

    return d;
}

void VoxelTerrain::_teardown() {
    if (!started) {
        return;
    }
    started = false;

    // Debug tools first: the validator owns a legacy mesher, and both hold pages of the direct one.
    // A run may be in flight: they delete themselves once their readbacks have arrived.
    if (validator) {
        validator->destroy();
        validator = nullptr;
    }
    if (auditor) {
        auditor->destroy();
        auditor = nullptr;
    }
    if (gpu_profiler) {
        RDUtils::delete_on_render_thread(gpu_profiler);
        gpu_profiler = nullptr;
    }

    chunk_discoverer->stop();
    memdelete(chunk_discoverer);
    chunk_discoverer = nullptr;

    // Release loaded chunks first: GPU Direct chunks hand their range back to the mesher.
    for (KeyValue<ChunkId, ChunkOutput> &kv : visual_mesh_data.loaded_chunk) {
        kv.value.free();
    }
    visual_mesh_data.loaded_chunk.clear();

    if (direct_mesher) {
        direct_mesher->destroy(); // deletes itself on the render thread
        direct_mesher = nullptr;
    }
    if (gpu_mesher) {
        gpu_mesher->stop();
        memdelete(gpu_mesher);
        gpu_mesher = nullptr;
    }

    VLOG(1, "VoxelTerrain: teardown, chunk instances created=", DebugLog::chunk_rids_created.load(),
            " freed=", DebugLog::chunk_rids_freed.load());
    visual_mesh_data.to_destroy_chunk.clear();
    visual_mesh_data.dirty_chunk.clear();
}

void VoxelTerrain::_exit_tree() {
    _teardown();
}

VoxelTerrain::~VoxelTerrain() {
    _teardown();
    if (sdf) {
        memdelete(sdf);
        sdf = nullptr;
    }
}
