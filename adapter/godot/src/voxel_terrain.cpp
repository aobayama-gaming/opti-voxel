#include "voxel_terrain.hpp"

#include "godot_callable.hpp"
#include "godot_conv.hpp"
#include "ov/executor.hpp"
#include "rd_utils.hpp"
#include "trace.hpp"
#include "voxel_sky.hpp"
#include <cmath>
#include <fstream>
#include <godot_cpp/classes/directional_light3d.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <map>

void VoxelTerrain::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_world_path", "path"), &VoxelTerrain::set_world_path);
    ClassDB::bind_method(D_METHOD("get_world_path"), &VoxelTerrain::get_world_path);
    ClassDB::bind_method(D_METHOD("set_world_json", "json", "dir"), &VoxelTerrain::set_world_json);
    ClassDB::bind_method(D_METHOD("restart_world", "json", "dir"), &VoxelTerrain::restart_world);
    ClassDB::bind_method(D_METHOD("get_world_error"), &VoxelTerrain::get_world_error);
    ClassDB::bind_method(D_METHOD("get_voxel_size"), &VoxelTerrain::get_voxel_size);
    ClassDB::bind_method(D_METHOD("set_world_origin_x", "voxel"), &VoxelTerrain::set_world_origin_x);
    ClassDB::bind_method(D_METHOD("get_world_origin_x"), &VoxelTerrain::get_world_origin_x);
    ClassDB::bind_method(D_METHOD("set_world_origin_y", "voxel"), &VoxelTerrain::set_world_origin_y);
    ClassDB::bind_method(D_METHOD("get_world_origin_y"), &VoxelTerrain::get_world_origin_y);
    ClassDB::bind_method(D_METHOD("set_world_origin_z", "voxel"), &VoxelTerrain::set_world_origin_z);
    ClassDB::bind_method(D_METHOD("get_world_origin_z"), &VoxelTerrain::get_world_origin_z);
    ClassDB::bind_method(D_METHOD("set_sun_direction", "towards_sun"), &VoxelTerrain::set_sun_direction);
    ClassDB::bind_method(D_METHOD("get_day_angle"), &VoxelTerrain::get_day_angle);
    ClassDB::bind_method(D_METHOD("get_planet_radius"), &VoxelTerrain::get_planet_radius);
    ClassDB::bind_method(D_METHOD("set_archetype_stream", "enabled"), &VoxelTerrain::set_archetype_stream);
    ClassDB::bind_method(D_METHOD("get_archetype_stream"), &VoxelTerrain::get_archetype_stream);
    ClassDB::bind_method(D_METHOD("set_lod_pixel_error", "error"), &VoxelTerrain::set_lod_pixel_error);
    ClassDB::bind_method(D_METHOD("get_lod_pixel_error"), &VoxelTerrain::get_lod_pixel_error);
    ClassDB::bind_method(D_METHOD("set_lod_pixel_error_back", "error"), &VoxelTerrain::set_lod_pixel_error_back);
    ClassDB::bind_method(D_METHOD("get_lod_pixel_error_back"), &VoxelTerrain::get_lod_pixel_error_back);
    ClassDB::bind_method(D_METHOD("set_lod_pixel_errors", "errors"), &VoxelTerrain::set_lod_pixel_errors);
    ClassDB::bind_method(D_METHOD("get_lod_pixel_errors"), &VoxelTerrain::get_lod_pixel_errors);
    ClassDB::bind_method(D_METHOD("set_lod_view_margin_deg", "degrees"), &VoxelTerrain::set_lod_view_margin_deg);
    ClassDB::bind_method(D_METHOD("get_lod_view_margin_deg"), &VoxelTerrain::get_lod_view_margin_deg);
    ClassDB::bind_method(D_METHOD("set_lod_view_memory_m", "metres"), &VoxelTerrain::set_lod_view_memory_m);
    ClassDB::bind_method(D_METHOD("get_lod_view_memory_m"), &VoxelTerrain::get_lod_view_memory_m);
    ClassDB::bind_method(D_METHOD("set_lod_view_max", "lod"), &VoxelTerrain::set_lod_view_max);
    ClassDB::bind_method(D_METHOD("get_lod_view_max"), &VoxelTerrain::get_lod_view_max);
    ClassDB::bind_method(D_METHOD("set_lod_centre", "position"), &VoxelTerrain::set_lod_centre);
    ClassDB::bind_method(D_METHOD("clear_lod_centre"), &VoxelTerrain::clear_lod_centre);
    ClassDB::bind_method(D_METHOD("set_lod_view_min", "lod"), &VoxelTerrain::set_lod_view_min);
    ClassDB::bind_method(D_METHOD("get_lod_view_min"), &VoxelTerrain::get_lod_view_min);
    ClassDB::bind_method(D_METHOD("get_lod_at_distance", "metres"), &VoxelTerrain::get_lod_at_distance);
    ClassDB::bind_method(D_METHOD("set_lod_far_limit", "enabled"), &VoxelTerrain::set_lod_far_limit);
    ClassDB::bind_method(D_METHOD("get_lod_far_limit"), &VoxelTerrain::get_lod_far_limit);
    ClassDB::bind_method(D_METHOD("get_lod_numbers"), &VoxelTerrain::get_lod_numbers);
    ClassDB::bind_method(D_METHOD("get_load_radius"), &VoxelTerrain::get_load_radius);
    ClassDB::bind_method(D_METHOD("get_half_chunk_voxels"), &VoxelTerrain::get_half_chunk_voxels);
    ClassDB::bind_method(D_METHOD("get_sdf_at", "render_pos"), &VoxelTerrain::get_sdf_at);
    ClassDB::bind_method(D_METHOD("get_sea_height_at", "render_pos"), &VoxelTerrain::get_sea_height_at);
    ClassDB::bind_method(D_METHOD("get_weather_at", "render_pos"), &VoxelTerrain::get_weather_at);
    ClassDB::bind_method(D_METHOD("get_tangent_position", "tangent_m"), &VoxelTerrain::get_tangent_position);
    ClassDB::bind_method(D_METHOD("get_up_at", "render_pos"), &VoxelTerrain::get_up_at);
    ClassDB::bind_method(D_METHOD("get_scene_shift"), &VoxelTerrain::get_scene_shift);
    ClassDB::bind_method(D_METHOD("set_rebase_distance_m", "metres"), &VoxelTerrain::set_rebase_distance_m);
    ClassDB::bind_method(D_METHOD("get_rebase_distance_m"), &VoxelTerrain::get_rebase_distance_m);
    ClassDB::bind_method(D_METHOD("get_render_origin_x"), &VoxelTerrain::get_render_origin_x);
    ClassDB::bind_method(D_METHOD("get_render_origin_y"), &VoxelTerrain::get_render_origin_y);
    ClassDB::bind_method(D_METHOD("get_render_origin_z"), &VoxelTerrain::get_render_origin_z);
    ClassDB::bind_method(D_METHOD("get_render_offset_m"), &VoxelTerrain::get_render_offset_m);
    ClassDB::bind_method(D_METHOD("set_skirt_depth", "depth"), &VoxelTerrain::set_skirt_depth);
    ClassDB::bind_method(D_METHOD("get_skirt_depth"), &VoxelTerrain::get_skirt_depth);
    ClassDB::bind_method(D_METHOD("get_pending_chunk_count"), &VoxelTerrain::get_pending_chunk_count);
    ClassDB::bind_method(D_METHOD("set_debug_gpu_profile", "enable"), &VoxelTerrain::set_debug_gpu_profile);
    ClassDB::bind_method(D_METHOD("get_debug_gpu_profile"), &VoxelTerrain::get_debug_gpu_profile);
    ClassDB::bind_method(D_METHOD("set_debug_gpu_profile_frames", "enable"), &VoxelTerrain::set_debug_gpu_profile_frames);
    ClassDB::bind_method(D_METHOD("get_debug_gpu_profile_frames"), &VoxelTerrain::get_debug_gpu_profile_frames);
    ClassDB::bind_method(D_METHOD("take_gpu_profile_frames"), &VoxelTerrain::take_gpu_profile_frames);
    ClassDB::bind_method(D_METHOD("debug_trace_start", "max_events"), &VoxelTerrain::debug_trace_start, DEFVAL(2000000));
    ClassDB::bind_method(D_METHOD("debug_trace_stop"), &VoxelTerrain::debug_trace_stop);
    ClassDB::bind_method(D_METHOD("debug_trace_write", "path"), &VoxelTerrain::debug_trace_write);
    ClassDB::bind_method(D_METHOD("get_process_times_ms"), &VoxelTerrain::get_process_times_ms);
    ClassDB::bind_method(D_METHOD("set_direct_batch_size", "size"), &VoxelTerrain::set_direct_batch_size);
    ClassDB::bind_method(D_METHOD("get_direct_batch_size"), &VoxelTerrain::get_direct_batch_size);
    ClassDB::bind_method(D_METHOD("set_direct_pages_per_batch", "pages"), &VoxelTerrain::set_direct_pages_per_batch);
    ClassDB::bind_method(D_METHOD("get_direct_pages_per_batch"), &VoxelTerrain::get_direct_pages_per_batch);
    ClassDB::bind_method(D_METHOD("set_direct_batches_per_frame", "batches"), &VoxelTerrain::set_direct_batches_per_frame);
    ClassDB::bind_method(D_METHOD("get_direct_batches_per_frame"), &VoxelTerrain::get_direct_batches_per_frame);
    ClassDB::bind_method(D_METHOD("set_direct_gpu_budget_ms", "ms"), &VoxelTerrain::set_direct_gpu_budget_ms);
    ClassDB::bind_method(D_METHOD("get_direct_gpu_budget_ms"), &VoxelTerrain::get_direct_gpu_budget_ms);
    ClassDB::bind_method(D_METHOD("set_direct_max_in_flight", "batches"), &VoxelTerrain::set_direct_max_in_flight);
    ClassDB::bind_method(D_METHOD("get_direct_max_in_flight"), &VoxelTerrain::get_direct_max_in_flight);
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
    ClassDB::bind_method(D_METHOD("set_page_params", "params"), &VoxelTerrain::set_page_params);
    ClassDB::bind_method(D_METHOD("set_log_level", "level"), &VoxelTerrain::set_log_level);
    ClassDB::bind_method(D_METHOD("get_log_level"), &VoxelTerrain::get_log_level);
    ClassDB::bind_method(D_METHOD("is_load_complete"), &VoxelTerrain::is_load_complete);
    ClassDB::bind_method(D_METHOD("is_edit_settled"), &VoxelTerrain::is_edit_settled);
    ClassDB::bind_method(D_METHOD("dig_hole", "origin", "dir"), &VoxelTerrain::dig_hole);
    ClassDB::bind_method(D_METHOD("dig_spray", "origin", "dir", "delta"), &VoxelTerrain::dig_spray);
    ClassDB::bind_method(D_METHOD("dig_tube", "origin", "dir"), &VoxelTerrain::dig_tube);
    ClassDB::bind_method(D_METHOD("dig_release"), &VoxelTerrain::dig_release);
    ClassDB::bind_method(D_METHOD("get_dig_edits"), &VoxelTerrain::get_dig_edits);
    ClassDB::bind_method(D_METHOD("set_dig_settings", "settings"), &VoxelTerrain::set_dig_settings);
    ClassDB::bind_method(D_METHOD("get_debug_stats"), &VoxelTerrain::get_debug_stats);
    ClassDB::bind_method(D_METHOD("debug_audit_pages"), &VoxelTerrain::debug_audit_pages);
    ClassDB::bind_method(D_METHOD("get_audit_report"), &VoxelTerrain::get_audit_report);
    ClassDB::bind_method(D_METHOD("debug_audit_discovery"), &VoxelTerrain::debug_audit_discovery);
    ClassDB::bind_method(D_METHOD("get_discovery_audit_report"), &VoxelTerrain::get_discovery_audit_report);
    ClassDB::bind_method(D_METHOD("set_specialize_shaders", "enable"), &VoxelTerrain::set_specialize_shaders);
    ClassDB::bind_method(D_METHOD("get_specialize_shaders"), &VoxelTerrain::get_specialize_shaders);
    ClassDB::bind_method(D_METHOD("set_destroy_verify", "enable"), &VoxelTerrain::set_destroy_verify);
    ClassDB::bind_method(D_METHOD("get_destroy_verify"), &VoxelTerrain::get_destroy_verify);
    ClassDB::bind_method(D_METHOD("set_walk_gpu", "enable"), &VoxelTerrain::set_walk_gpu);
    ClassDB::bind_method(D_METHOD("get_walk_gpu"), &VoxelTerrain::get_walk_gpu);
    ClassDB::bind_method(D_METHOD("set_walk_verify", "enable"), &VoxelTerrain::set_walk_verify);
    ClassDB::bind_method(D_METHOD("get_walk_verify"), &VoxelTerrain::get_walk_verify);
    ClassDB::bind_method(D_METHOD("set_discoverer_mode", "mode"), &VoxelTerrain::set_discoverer_mode);
    ClassDB::bind_method(D_METHOD("get_discoverer_mode"), &VoxelTerrain::get_discoverer_mode);
    ClassDB::bind_method(D_METHOD("set_discovery_compare", "enable"), &VoxelTerrain::set_discovery_compare);
    ClassDB::bind_method(D_METHOD("get_discovery_compare"), &VoxelTerrain::get_discovery_compare);
    ClassDB::bind_method(D_METHOD("add_instance", "type", "x", "y", "z", "rotation", "params", "op", "k"), &VoxelTerrain::add_instance,
            DEFVAL(PackedFloat32Array()), DEFVAL(PackedFloat32Array()), DEFVAL(0), DEFVAL(0.0f));
    ClassDB::bind_method(D_METHOD("remove_instance", "id"), &VoxelTerrain::remove_instance);
    ClassDB::bind_method(D_METHOD("move_instance", "id", "x", "y", "z", "rotation"), &VoxelTerrain::move_instance, DEFVAL(PackedFloat32Array()));
    ClassDB::bind_method(D_METHOD("get_type_index", "name"), &VoxelTerrain::get_type_index);
    ClassDB::bind_method(D_METHOD("save_edits", "path"), &VoxelTerrain::save_edits);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "world_path", PROPERTY_HINT_FILE, "*.json"), "set_world_path", "get_world_path");
    // Read-only, from the world file.
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_voxel_size");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_numbers", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_lod_numbers");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "planet_radius", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_planet_radius");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "archetype_stream"), "set_archetype_stream", "get_archetype_stream");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_pixel_error", PROPERTY_HINT_RANGE, "0.25,64,0.25"), "set_lod_pixel_error", "get_lod_pixel_error");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_pixel_error_back", PROPERTY_HINT_RANGE, "0,64,0.25"), "set_lod_pixel_error_back", "get_lod_pixel_error_back");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "lod_pixel_errors"), "set_lod_pixel_errors", "get_lod_pixel_errors");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_view_margin_deg", PROPERTY_HINT_RANGE, vformat("%d,180,1,suffix:deg", int(ChunkDiscoverer::RESCAN_TURN_DEG))), "set_lod_view_margin_deg", "get_lod_view_margin_deg");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_view_memory_m", PROPERTY_HINT_RANGE, "-1,100000,1,or_greater,suffix:m"), "set_lod_view_memory_m", "get_lod_view_memory_m");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_view_max", PROPERTY_HINT_RANGE, "-1,63,1"), "set_lod_view_max", "get_lod_view_max");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_view_min", PROPERTY_HINT_RANGE, "0,63,1"), "set_lod_view_min", "get_lod_view_min");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lod_far_limit"), "set_lod_far_limit", "get_lod_far_limit");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "skirt_depth", PROPERTY_HINT_RANGE, "0,4,0.05,suffix:voxels"), "set_skirt_depth", "get_skirt_depth");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "specialize_shaders"), "set_specialize_shaders", "get_specialize_shaders");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "discoverer_mode", PROPERTY_HINT_ENUM, "CPU:0,GPU:1"), "set_discoverer_mode", "get_discoverer_mode");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "walk_gpu"), "set_walk_gpu", "get_walk_gpu");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "walk_verify"), "set_walk_verify", "get_walk_verify");
    ADD_GROUP("World Origin", "world_origin_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_x", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_x", "get_world_origin_x");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_y", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_y", "get_world_origin_y");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_z", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_z", "get_world_origin_z");
    ADD_GROUP("", "");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rebase_distance_m", PROPERTY_HINT_RANGE, "0,1000000,1,suffix:m"), "set_rebase_distance_m", "get_rebase_distance_m");
    ADD_SIGNAL(MethodInfo("origin_shifted", PropertyInfo(Variant::VECTOR3, "shift")));
    ADD_GROUP("GPU Direct", "direct_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_batch_size", PROPERTY_HINT_RANGE, "1,1024"), "set_direct_batch_size", "get_direct_batch_size");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_pages_per_batch", PROPERTY_HINT_RANGE, "1,256"), "set_direct_pages_per_batch", "get_direct_pages_per_batch");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_batches_per_frame", PROPERTY_HINT_RANGE, vformat("1,%d", GPUDirectMesher::MAX_BATCHES_PER_FRAME)), "set_direct_batches_per_frame", "get_direct_batches_per_frame");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "direct_gpu_budget_ms", PROPERTY_HINT_RANGE, "0,100,0.1,suffix:ms"), "set_direct_gpu_budget_ms", "get_direct_gpu_budget_ms");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_max_in_flight", PROPERTY_HINT_RANGE, "0,256"), "set_direct_max_in_flight", "get_direct_max_in_flight");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_page_capacity", PROPERTY_HINT_RANGE, "1024,1048576"), "set_direct_page_capacity", "get_direct_page_capacity");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "direct_initial_pages", PROPERTY_HINT_RANGE, "0,4096"), "set_direct_initial_pages", "get_direct_initial_pages");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "direct_compaction"), "set_direct_compaction", "get_direct_compaction");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "direct_compact_max_extent", PROPERTY_HINT_RANGE, "1,1000000"), "set_direct_compact_max_extent", "get_direct_compact_max_extent");
    ADD_GROUP("Debug", "");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_view", PROPERTY_HINT_ENUM, "Shaded:0,Normals:1,LOD:2,Page:3,Wireframe:4,Chunk:5,Group:6,Quad index:7,Faceted:8,Material:9,Biome:12"), "set_debug_view", "get_debug_view");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "log_level", PROPERTY_HINT_RANGE, "0,3"), "set_log_level", "get_log_level");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_gpu_profile"), "set_debug_gpu_profile", "get_debug_gpu_profile");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_gpu_profile_frames"), "set_debug_gpu_profile_frames", "get_debug_gpu_profile_frames");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "destroy_verify"), "set_destroy_verify", "get_destroy_verify");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "discovery_compare"), "set_discovery_compare", "get_discovery_compare");
}

void VoxelTerrain::set_world_path(const String &p_path) {
    if (!_settable("world_path")) {
        return;
    }
    cfg().world_path = ov_str(p_path);
    cfg().world_json.clear();
    world->load_world(); // the voxel size and the scene shift are read before the terrain starts
}

void VoxelTerrain::set_world_json(const String &p_json, const String &p_dir) {
    if (!_settable("the world")) {
        return;
    }
    cfg().world_path.clear();
    cfg().world_json = ov_str(p_json);
    cfg().world_dir = ov_str(p_dir);
    world->load_world(); // as set_world_path
}

String VoxelTerrain::get_world_path() const {
    return godot_str(cfg().world_path);
}

float VoxelTerrain::get_voxel_size() const {
    return cfg().chunk_param.voxel_size;
}

// Read on each start (restart_world too): the climate's shaders specialised to its values, or generic (SDFCascade::specialize).
void VoxelTerrain::set_specialize_shaders(bool p_enable) {
    cfg().specialize_shaders = p_enable;
}

bool VoxelTerrain::get_specialize_shaders() const {
    return cfg().specialize_shaders;
}

void VoxelTerrain::set_destroy_verify(bool p_enable) {
    if (!_settable("destroy_verify")) {
        return;
    }
    cfg().destroy_verify = p_enable;
}

bool VoxelTerrain::get_destroy_verify() const {
    return cfg().destroy_verify;
}

void VoxelTerrain::set_discoverer_mode(int p_mode) {
    if (!_settable("discoverer_mode")) {
        return;
    }
    cfg().discoverer_mode = p_mode;
}

int VoxelTerrain::get_discoverer_mode() const {
    return cfg().discoverer_mode;
}

void VoxelTerrain::set_walk_gpu(bool p_enable) {
    if (!_settable("walk_gpu")) {
        return;
    }
    cfg().walk_gpu = p_enable;
}

bool VoxelTerrain::get_walk_gpu() const {
    return cfg().walk_gpu;
}

void VoxelTerrain::set_walk_verify(bool p_enable) {
    if (!_settable("walk_verify")) {
        return;
    }
    cfg().walk_verify = p_enable;
}

bool VoxelTerrain::get_walk_verify() const {
    return cfg().walk_verify;
}

void VoxelTerrain::set_discovery_compare(bool p_enable) {
    if (!_settable("discovery_compare")) {
        return;
    }
    cfg().discovery_compare = p_enable;
}

bool VoxelTerrain::get_discovery_compare() const {
    return cfg().discovery_compare;
}

void VoxelTerrain::set_skirt_depth(float p_depth) {
    if (!_settable("skirt_depth")) {
        return;
    }
    cfg().chunk_param.skirt_depth = p_depth;
}

float VoxelTerrain::get_skirt_depth() const {
    return cfg().chunk_param.skirt_depth;
}

// The settings are only read by start(), which checks their ranges (World::Settings::validate).
bool VoxelTerrain::_settable(const char *p_name) const {
    ERR_FAIL_COND_V_MSG(world->is_started(), false, vformat("%s must be set before the terrain enters the tree.", p_name));
    return true;
}

void VoxelTerrain::set_world_origin_x(int64_t p_voxel) {
    if (_settable("world_origin")) {
        cfg().world_origin_request.x = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_x() const {
    return cfg().world_origin_request.x;
}

void VoxelTerrain::set_world_origin_y(int64_t p_voxel) {
    if (_settable("world_origin")) {
        cfg().world_origin_request.y = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_y() const {
    return cfg().world_origin_request.y;
}

void VoxelTerrain::set_world_origin_z(int64_t p_voxel) {
    if (_settable("world_origin")) {
        cfg().world_origin_request.z = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_z() const {
    return cfg().world_origin_request.z;
}

double VoxelTerrain::get_planet_radius() const {
    return world->world_file().planet_radius_m;
}

void VoxelTerrain::set_sun_direction(const Vector3 &p_towards_sun) {
    world->set_sun(ov_f3(p_towards_sun.normalized()));
}

double VoxelTerrain::get_day_angle() const {
    return world->day_angle();
}

void VoxelTerrain::set_archetype_stream(bool p_enabled) {
    if (!_settable("archetype_stream")) {
        return;
    }
    cfg().archetype_stream = p_enabled;
}

bool VoxelTerrain::get_archetype_stream() const {
    return cfg().archetype_stream;
}

void VoxelTerrain::set_lod_pixel_error(float p_error) {
    if (!_settable("lod_pixel_error")) {
        return;
    }
    cfg().lod_pixel_error = p_error;
}

float VoxelTerrain::get_lod_pixel_error() const {
    return cfg().lod_pixel_error;
}

void VoxelTerrain::set_lod_pixel_error_back(float p_error) {
    if (!_settable("lod_pixel_error_back")) {
        return;
    }
    cfg().lod_pixel_error_back = p_error;
}

float VoxelTerrain::get_lod_pixel_error_back() const {
    return cfg().lod_pixel_error_back;
}

void VoxelTerrain::set_lod_pixel_errors(const PackedFloat32Array &p_errors) {
    if (!_settable("lod_pixel_errors")) {
        return;
    }
    cfg().lod_pixel_errors.assign(p_errors.ptr(), p_errors.ptr() + p_errors.size());
}

PackedFloat32Array VoxelTerrain::get_lod_pixel_errors() const {
    PackedFloat32Array errors;
    for (float e : cfg().lod_pixel_errors) {
        errors.push_back(e);
    }
    return errors;
}

void VoxelTerrain::set_lod_view_margin_deg(float p_degrees) {
    if (!_settable("lod_view_margin_deg")) {
        return;
    }
    cfg().lod_view_margin_deg = p_degrees;
}

float VoxelTerrain::get_lod_view_margin_deg() const {
    return cfg().lod_view_margin_deg;
}

void VoxelTerrain::set_lod_view_memory_m(float p_metres) {
    if (!_settable("lod_view_memory_m")) {
        return;
    }
    cfg().lod_view_memory_m = p_metres;
}

float VoxelTerrain::get_lod_view_memory_m() const {
    return cfg().lod_view_memory_m;
}

void VoxelTerrain::set_lod_view_max(int p_lod) {
    if (!_settable("lod_view_max")) {
        return;
    }
    cfg().lod_view_max = p_lod;
}

int VoxelTerrain::get_lod_view_max() const {
    return cfg().lod_view_max;
}

void VoxelTerrain::set_lod_centre(const Vector3 &p_position) {
    lod_centre = p_position;
    has_lod_centre = true;
}

void VoxelTerrain::clear_lod_centre() {
    has_lod_centre = false;
}

void VoxelTerrain::set_lod_view_min(int p_lod) {
    cfg().lod_view_min = p_lod;
}

int VoxelTerrain::get_lod_view_min() const {
    return cfg().lod_view_min;
}

int VoxelTerrain::get_lod_at_distance(double p_metres) const {
    return world->lod_at_distance(p_metres);
}

void VoxelTerrain::set_lod_far_limit(bool p_enabled) {
    if (!_settable("lod_far_limit")) {
        return;
    }
    lod_far_limit = p_enabled;
}

bool VoxelTerrain::get_lod_far_limit() const {
    return lod_far_limit;
}

int VoxelTerrain::get_lod_numbers() const {
    return world->world_file().lods;
}

void VoxelTerrain::set_direct_batch_size(int p_size) {
    if (!_settable("direct_batch_size")) {
        return;
    }
    cfg().direct.batch_size = p_size;
}

int VoxelTerrain::get_direct_batch_size() const {
    return cfg().direct.batch_size;
}

void VoxelTerrain::set_direct_pages_per_batch(int p_pages) {
    if (!_settable("direct_pages_per_batch")) {
        return;
    }
    cfg().direct.pages_per_batch = p_pages;
}

int VoxelTerrain::get_direct_pages_per_batch() const {
    return cfg().direct.pages_per_batch;
}

void VoxelTerrain::set_direct_batches_per_frame(int p_batches) {
    if (!_settable("direct_batches_per_frame")) {
        return;
    }
    cfg().direct.max_batches_per_frame = p_batches;
}

int VoxelTerrain::get_direct_batches_per_frame() const {
    return cfg().direct.max_batches_per_frame;
}

void VoxelTerrain::set_direct_gpu_budget_ms(float p_ms) {
    if (!_settable("direct_gpu_budget_ms")) {
        return;
    }
    cfg().direct.gpu_budget_ms = p_ms;
}

float VoxelTerrain::get_direct_gpu_budget_ms() const {
    return cfg().direct.gpu_budget_ms;
}

void VoxelTerrain::set_direct_max_in_flight(int p_batches) {
    if (!_settable("direct_max_in_flight")) {
        return;
    }
    ERR_FAIL_COND_MSG(p_batches < 0, "direct_max_in_flight must be >= 0 (0: 3 * direct_batches_per_frame)."); // the setting is unsigned
    cfg().direct.max_in_flight = p_batches;
}

int VoxelTerrain::get_direct_max_in_flight() const {
    return cfg().direct.max_in_flight;
}

void VoxelTerrain::set_direct_page_capacity(int p_capacity) {
    if (!_settable("direct_page_capacity")) {
        return;
    }
    cfg().direct.page_capacity = p_capacity;
}

int VoxelTerrain::get_direct_page_capacity() const {
    return cfg().direct.page_capacity;
}

void VoxelTerrain::set_direct_initial_pages(int p_pages) {
    if (!_settable("direct_initial_pages")) {
        return;
    }
    cfg().direct.initial_pages = p_pages;
}

int VoxelTerrain::get_direct_initial_pages() const {
    return cfg().direct.initial_pages;
}

void VoxelTerrain::set_direct_compaction(bool p_enable) {
    if (!_settable("direct_compaction")) {
        return;
    }
    cfg().direct.compaction = p_enable;
}

bool VoxelTerrain::get_direct_compaction() const {
    return cfg().direct.compaction;
}

void VoxelTerrain::set_direct_compact_max_extent(float p_chunks) {
    if (!_settable("direct_compact_max_extent")) {
        return;
    }
    cfg().direct.compact_max_extent = p_chunks;
}

float VoxelTerrain::get_direct_compact_max_extent() const {
    return cfg().direct.compact_max_extent;
}

Vector3 VoxelTerrain::_camera_forward() const {
    return -camera->get_global_transform().basis.get_column(2);
}

double VoxelTerrain::_camera_far() const {
    return camera != nullptr && lod_far_limit ? double(camera->get_far()) : 0.0;
}

double VoxelTerrain::get_sdf_at(const Vector3 &p_render_pos) const {
    const double pos[3] = { double(p_render_pos.x), double(p_render_pos.y), double(p_render_pos.z) };
    return world->sdf_at(pos);
}

double VoxelTerrain::get_sea_height_at(const Vector3 &p_render_pos) const {
    const double pos[3] = { double(p_render_pos.x), double(p_render_pos.y), double(p_render_pos.z) };
    return world->sea_height_at(pos);
}

Dictionary VoxelTerrain::get_weather_at(const Vector3 &p_render_pos) const {
    Dictionary d;
    const double pos[3] = { double(p_render_pos.x), double(p_render_pos.y), double(p_render_pos.z) };
    Weather::Sample s;
    if (!world->weather_at(pos, s)) {
        return d;
    }
    d["wind"] = Vector3(s.wind_mps.x, s.wind_mps.y, s.wind_mps.z);
    d["cover"] = s.cover;
    d["type"] = s.type;
    d["rain"] = s.rain;
    d["temp"] = s.temp;
    d["vapour"] = s.vapour;
    return d;
}

Vector3 VoxelTerrain::get_tangent_position(const Vector2 &p_tangent_m) const {
    double pos[3];
    world->tangent_position(double(p_tangent_m.x), double(p_tangent_m.y), pos);
    return Vector3(real_t(pos[0]), real_t(pos[1]), real_t(pos[2]));
}

Vector3 VoxelTerrain::get_up_at(const Vector3 &p_render_pos) const {
    const double pos[3] = { double(p_render_pos.x), double(p_render_pos.y), double(p_render_pos.z) };
    const UpAxis up = world->up_at(pos);
    return Vector3(real_t(up[0]), real_t(up[1]), real_t(up[2]));
}

Vector3 VoxelTerrain::get_scene_shift() const {
    double shift[3];
    world->scene_shift(shift);
    return Vector3(real_t(shift[0]), real_t(shift[1]), real_t(shift[2]));
}

void VoxelTerrain::set_rebase_distance_m(float p_metres) {
    if (!_settable("rebase_distance_m")) {
        return;
    }
    cfg().rebase_distance_m = p_metres;
}

float VoxelTerrain::get_rebase_distance_m() const {
    return cfg().rebase_distance_m;
}

int64_t VoxelTerrain::get_render_origin_x() const {
    return world->render_origin_voxel().x;
}

int64_t VoxelTerrain::get_render_origin_y() const {
    return world->render_origin_voxel().y;
}

int64_t VoxelTerrain::get_render_origin_z() const {
    return world->render_origin_voxel().z;
}

Vector3 VoxelTerrain::get_render_offset_m() const {
    if (!world->is_started()) {
        return Vector3();
    }
    return godot_v3(ChunkMath::origin_shift_m(world->world_origin(), world->render_origin(), world->chunk_param()));
}

void VoxelTerrain::_move_render_origin(const Float3 &p_shift) {
    const Vector3 shift = godot_v3(p_shift);
    const Vector3 camera_before = camera->get_global_position();
    lod_centre -= shift;
    Node *parent = get_parent();
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        Node3D *node = Object::cast_to<Node3D>(parent->get_child(i));
        if (node != nullptr && node != this) {
            node->set_global_position(node->get_global_position() - shift);
        }
    }
    // A camera outside the siblings (top_level, elsewhere in the tree) moves itself: it must, or every frame rebases.
    if (camera->get_global_position() == camera_before) {
        camera->set_global_position(camera_before - shift);
    }
    _push_render_frame();
    emit_signal("origin_shifted", shift);
}

namespace {

// A struct's uniforms (its visit: BaseSurface::Render, AtmosphereMedium) into shader parameters.
struct ShaderParams {
    Dictionary &out;
    void operator()(const char *p_name, int p_v) { out[p_name] = p_v; }
    void operator()(const char *p_name, float p_v) { out[p_name] = p_v; }
    void operator()(const char *p_name, const Float3 &p_v) { out[p_name] = godot_v3(p_v); }
};

} // namespace

// Sets a global shader uniform whose value changed since the last frame (sent: the last values by name).
struct SetChangedGlobal {
    Dictionary &sent;
    void set(const char *p_name, const Variant &p_value) {
        if (sent.get(p_name, Variant()) != p_value) {
            sent[p_name] = p_value;
            RenderingServer::get_singleton()->global_shader_parameter_set(p_name, p_value);
        }
    }
    void operator()(const char *p_name, const Float3 &p_v) { set(p_name, godot_v3(p_v)); }
    void operator()(const char *p_name, float p_v) { set(p_name, p_v); }
};

void VoxelTerrain::_push_sun_without_sky() {
    Node *parent = get_parent();
    DirectionalLight3D *light = nullptr;
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        if (Object::cast_to<VoxelSky>(parent->get_child(i))) {
            return; // VoxelSky::_process sets it
        }
        light = light ? light : Object::cast_to<DirectionalLight3D>(parent->get_child(i));
    }
    const Vector3 towards = light ? light->get_global_transform().basis.get_column(2).normalized() : VoxelSky::DEFAULT_SUN_DIR;
    if (towards != sun_dir_sent) {
        sun_dir_sent = towards;
        RenderingServer::get_singleton()->global_shader_parameter_set("atmo_sun_dir", towards);
    }
}

void VoxelTerrain::_push_atmosphere(const Dictionary &p_params) {
    set_page_params(p_params);
    Node *parent = get_parent();
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        if (VoxelSky *sky = Object::cast_to<VoxelSky>(parent->get_child(i))) {
            sky->set_atmosphere(p_params);
        }
    }
}

void VoxelTerrain::_rt_share_textures(RDDevice *p_device, std::shared_ptr<std::vector<RID>> p_rids, ObjectID p_terrain, const char *p_enabled,
        const std::vector<const char *> &p_names, const std::vector<ov::gpu::Texture> &p_textures) {
    RenderingServer *rs = RenderingServer::get_singleton();
    for (const RID &rid : *p_rids) {
        if (rid.is_valid()) {
            rs->free_rid(rid);
        }
    }
    p_rids->clear();
    if (!p_textures[0].is_valid()) {
        return; // the world stops (_teardown turned them off on the pages and the sky)
    }
    // The materials are set on the main thread.
    Dictionary params;
    params[p_enabled] = true; // with the textures: never on with unbound samplers
    for (size_t i = 0; i < p_textures.size(); i++) {
        p_rids->push_back(rs->texture_rd_create(p_device->get_rid(p_textures[i])));
        params[p_names[i]] = p_rids->back();
    }
    godot_callable([p_terrain, params]() {
        if (VoxelTerrain *terrain = Object::cast_to<VoxelTerrain>(ObjectDB::get_instance(p_terrain))) {
            terrain->_push_atmosphere(params);
        }
    }).call_deferred();
}

void VoxelTerrain::_push_render_frame() {
    const ChunkParam &param = world->chunk_param();
    // The base in render space (World::base_render), for the pages' up, altitude and cloud shadows and the sky's frame and
    // cloud layer.
    Dictionary base;
    world->base_render().visit(ShaderParams{ base });
    set_page_params(base);
    Dictionary sea;
    world->sea_render().visit(ShaderParams{ sea });
    set_page_params(sea);
    const Vector3i origin_lo = godot_v3i(WorldCoord::lo32_i(param.origin_voxel()));
    Node *parent = get_parent();
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        if (VoxelSky *sky = Object::cast_to<VoxelSky>(parent->get_child(i))) {
            sky->set_render_frame(origin_lo, param.voxel_size, base);
        }
    }
}

double VoxelTerrain::get_load_radius() const {
    return world->load_radius();
}

int64_t VoxelTerrain::get_half_chunk_voxels() const {
    return cfg().chunk_param.half_chunk_voxels();
}

void VoxelTerrain::set_debug_gpu_profile(bool p_enable) {
    debug_gpu_profile = p_enable;
}

bool VoxelTerrain::get_debug_gpu_profile() const {
    return debug_gpu_profile;
}

void VoxelTerrain::set_debug_gpu_profile_frames(bool p_enable) {
    debug_gpu_profile_frames = p_enable;
    if (gpu_profiler) {
        gpu_profiler->set_recording(p_enable);
    }
}

bool VoxelTerrain::get_debug_gpu_profile_frames() const {
    return debug_gpu_profile_frames;
}

Array VoxelTerrain::take_gpu_profile_frames() {
    return gpu_profiler ? gpu_profiler->take_frames() : Array();
}

void VoxelTerrain::debug_trace_start(int64_t p_max_events) {
    Trace::name_thread("main");
    Trace::start(uint64_t(MAX(p_max_events, int64_t(1))));
}

void VoxelTerrain::debug_trace_stop() {
    Trace::stop();
}

Dictionary VoxelTerrain::debug_trace_write(const String &p_path) {
    const bool was = Trace::enabled();
    Trace::stop(); // a consistent copy: no thread appends while it is written
    const std::string json = Trace::to_json();
    std::ofstream file(ProjectSettings::get_singleton()->globalize_path(p_path).utf8().get_data(), std::ios::binary);
    file.write(json.data(), std::streamsize(json.size()));
    Dictionary d;
    d["events"] = int64_t(Trace::event_count());
    d["dropped"] = int64_t(Trace::dropped_count());
    d["bytes"] = file.good() ? int64_t(json.size()) : int64_t(-1);
    Trace::recording.store(was);
    return d;
}

Dictionary VoxelTerrain::get_process_times_ms() const {
    return godot_dict(world->process_times_ms());
}

int VoxelTerrain::get_pending_chunk_count() const {
    return int(world->pending_chunk_count());
}

void VoxelTerrain::set_debug_view(int p_view) {
    world->set_debug_view(p_view);
}

int VoxelTerrain::get_debug_view() const {
    return cfg().debug_view;
}

void VoxelTerrain::set_page_params(const Dictionary &p_params) {
    Array keys = p_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        page_params[keys[i]] = p_params[keys[i]];
        if (page_host) {
            page_host->set_material_param(keys[i], p_params[keys[i]]);
        }
    }
}

void VoxelTerrain::set_log_level(int p_level) {
    DebugLog::level.store(p_level);
}

int VoxelTerrain::get_log_level() const {
    return DebugLog::level.load();
}

void VoxelTerrain::_ready() {
    // Without a RenderingDevice (--headless, the Compatibility renderer, the editor's headless --import) the terrain
    // stays empty.
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    ERR_FAIL_NULL_MSG(rd, "VoxelTerrain: no RenderingDevice (--headless or the Compatibility renderer): the terrain is disabled.");

    if (Engine::get_singleton()->is_editor_hint()) {
        camera = EditorInterface::get_singleton()->get_editor_viewport_3d()->get_camera_3d();
    } else {
        camera = get_viewport()->get_camera_3d();
    }
    main_device = new RDDevice(rd);
    VoxelSky::add_atmosphere_globals(); // before the page shader compiles
    hosts.device = main_device;
    page_host = new GodotPageHost(get_world_3d()->get_scenario(), main_device);
    hosts.pages = page_host;
    hosts.atmosphere_textures = [device = main_device, rids = std::make_shared<std::vector<RID>>(), id = ObjectID(get_instance_id())](
                                        ov::gpu::Texture p_volume, ov::gpu::Texture p_multiscatter) {
        _rt_share_textures(device, rids, id, "atmo_enabled", { "atmo_volume", "atmo_multiscatter" }, { p_volume, p_multiscatter });
    };
    hosts.cloud_textures = [device = main_device, rids = std::make_shared<std::vector<RID>>(), id = ObjectID(get_instance_id())](ov::gpu::Texture p_light,
                                   ov::gpu::Texture p_depth, ov::gpu::Texture p_shadow) {
        const auto &names = Clouds::HOST_TEXTURES;
        _rt_share_textures(device, rids, id, "cloud_enabled", { names[0], names[1], names[2] }, { p_light, p_depth, p_shadow });
    };
    start_chunk_param = cfg().chunk_param;
    _start_world();
}

bool VoxelTerrain::_start_world() {
    // The pixel error is measured against the camera's real vertical FOV and the 3D render's height: an upscaler's
    // render scale (scaling_3d_scale) makes its pixels larger, and the LOD coarser with them.
    ov::World::Projection projection;
    if (camera != nullptr) {
        Viewport *viewport = camera->get_viewport();
        const Vector2 viewport_size = viewport->get_visible_rect().size * float(viewport->get_scaling_3d_scale());
        projection.viewport_height = MAX(viewport_size.y, 1.0f);
        projection.aspect = viewport_size.x > 0.0f ? viewport_size.x / projection.viewport_height : projection.aspect;
        projection.fov_rad = Math::deg_to_rad(float(camera->get_fov()));
        if (camera->get_keep_aspect_mode() == Camera3D::KEEP_WIDTH && viewport_size.x > 0.0f) {
            // fov is horizontal: convert it to vertical.
            projection.fov_rad = 2.0f * std::atan(std::tan(projection.fov_rad * 0.5f) * viewport_size.y / viewport_size.x);
        }
    }
    const Float3 camera_pos = camera != nullptr ? ov_f3(camera->get_global_position()) : Float3();
    const Float3 camera_forward = camera != nullptr ? ov_f3(_camera_forward()) : Float3();
    const double camera_far = _camera_far();
    if (!world->start(projection, hosts, camera != nullptr ? &camera_pos : nullptr, camera != nullptr ? &camera_forward : nullptr, camera_far)) {
        if (page_host) {
            page_host->set_sea(false); // a refused world draws no sea (a restart's last one had one)
        }
        return false;
    }
    if (page_host) {
        // Skirt depth per LOD: the page shader's height colours undo the skirt push.
        const SDFCascade &sdf = world->terrain();
        PackedFloat32Array skirt_depths;
        skirt_depths.resize(32);
        for (int l = 0; l < 32; l++) {
            skirt_depths.set(l, l < get_lod_numbers() ? sdf.skirt_depth_m(l) : 0.0f);
        }
        page_host->set_material_param("skirt_depth_m", skirt_depths);
        page_host->set_sea(world->world_file().sea.on);
        set_page_params(page_params); // set before the pool existed (VoxelSky's clouds)
    }
    _push_render_frame();
    if (const Atmosphere *atmosphere = world->get_atmosphere()) {
        // The medium, the longest path (constant for a world) and the switch; the textures follow once made.
        Dictionary params;
        atmosphere->host_constants().visit(ShaderParams{ params });
        _push_atmosphere(params);
    }
    return true;
}

bool VoxelTerrain::restart_world(const String &p_json, const String &p_dir) {
    ERR_FAIL_NULL_V_MSG(main_device, false, "restart_world: the terrain is not in the tree (set_world_json before it enters).");
    OV_TRACE("terrain", "restart_world");
    main_device->post([device = main_device]() { device->park(); });
    // The live edits go on in the next world (a world refused before has none: the next one replays its file's).
    const bool keep_edits = world->is_started();
    const std::vector<SDFTable::Edit> edits = world->live_edits();
    ov::World::Kept kept; // the next world's mesher draws on the same pages, its discoverer on the same local device
    {
        OV_TRACE("terrain", "restart_stop");
        _stop_world(&kept);
    }
    ov::World::Settings settings = cfg();
    settings.chunk_param = start_chunk_param;
    const ov::DigTool::Settings dig = dig_tool->settings;
    dig_tool.reset();
    world = std::make_unique<ov::World>();
    world->reuse(std::move(kept));
    dig_tool = std::make_unique<ov::DigTool>(*world);
    dig_tool->settings = dig;
    cfg() = settings;
    cfg().world_path.clear();
    cfg().world_json = ov_str(p_json);
    cfg().world_dir = ov_str(p_dir);
    if (keep_edits) {
        world->set_start_edits(edits);
    }
    bool started = false;
    {
        OV_TRACE("terrain", "restart_load");
        started = world->load_world();
    }
    if (started) {
        OV_TRACE("terrain", "restart_start");
        started = _start_world();
    }
    // After the new world's pipelines were made (posted in order): the old ones it didn't take.
    main_device->post([device = main_device]() { device->drop_parked(); });
    return started;
}

String VoxelTerrain::get_world_error() const {
    return godot_str(world->get_error());
}

// The instance edit's rotation (9 floats) and params (8): null when not given (identity / zero).
static const float *edit_floats(const PackedFloat32Array &p_values, int p_count, float *r_out, bool p_pad) {
    if (p_values.size() < (p_pad ? 1 : p_count)) {
        return nullptr;
    }
    for (int i = 0; i < p_count; i++) {
        r_out[i] = i < p_values.size() ? p_values[i] : 0.0f;
    }
    return r_out;
}

int VoxelTerrain::add_instance(int p_type, int64_t p_x, int64_t p_y, int64_t p_z, const PackedFloat32Array &p_rotation,
        const PackedFloat32Array &p_params, int p_op, float p_k) {
    float rot[9];
    float params[8] = {};
    return world->add_instance(p_type, VoxelPos(p_x, p_y, p_z), edit_floats(p_rotation, 9, rot, false), edit_floats(p_params, 8, params, true), p_op, p_k);
}

bool VoxelTerrain::remove_instance(int p_id) {
    return world->remove_instance(p_id);
}

bool VoxelTerrain::move_instance(int p_id, int64_t p_x, int64_t p_y, int64_t p_z, const PackedFloat32Array &p_rotation) {
    float rot[9];
    return world->move_instance(p_id, VoxelPos(p_x, p_y, p_z), edit_floats(p_rotation, 9, rot, false));
}

int VoxelTerrain::get_type_index(const String &p_name) const {
    return world->type_index(ov_str(p_name));
}

String VoxelTerrain::save_edits(const String &p_path) const {
    return godot_str(world->save_edits(ov_str(p_path)));
}

bool VoxelTerrain::dig_hole(const Vector3 &p_origin, const Vector3 &p_dir) {
    return dig_tool->dig(ov_f3(p_origin), ov_f3(p_dir));
}

void VoxelTerrain::dig_spray(const Vector3 &p_origin, const Vector3 &p_dir, double p_delta) {
    dig_tool->spray(ov_f3(p_origin), ov_f3(p_dir), p_delta);
}

void VoxelTerrain::dig_tube(const Vector3 &p_origin, const Vector3 &p_dir) {
    dig_tool->tube(ov_f3(p_origin), ov_f3(p_dir));
}

void VoxelTerrain::dig_release() {
    dig_tool->release();
}

int VoxelTerrain::get_dig_edits() const {
    return dig_tool->edit_count();
}

void VoxelTerrain::set_dig_settings(const Dictionary &p_settings) {
    ov::DigTool::Settings &s = dig_tool->settings;
    const std::pair<const char *, double *> metres[] = { { "dig_radius", &s.dig_radius_m }, { "dig_length", &s.dig_length_m }, { "dig_k", &s.dig_k_m },
        { "spray_radius", &s.spray_radius_m }, { "drag_radius", &s.drag_radius_m }, { "drag_length", &s.drag_length_m } };
    for (const auto &m : metres) {
        if (p_settings.has(m.first)) {
            *m.second = double(p_settings[m.first]);
        }
    }
    if (p_settings.has("drag_union")) {
        s.drag_op = bool(p_settings["drag_union"]) ? Archetype::Combine::UNION : Archetype::Combine::SUBTRACT;
    }
}

void VoxelTerrain::_process(double) {
    if (!world->is_started()) {
        return;
    }
    if (camera != nullptr) {
        Float3 shift;
        if (!Engine::get_singleton()->is_editor_hint() && world->recenter(ov_f3(camera->get_global_position()), shift)) {
            _move_render_origin(shift);
        }
        const Float3 pos = ov_f3(camera->get_global_position());
        const Float3 forward = ov_f3(_camera_forward());
        const Float3 centre = ov_f3(lod_centre);
        world->frame(&pos, &forward, _camera_far(), has_lod_centre ? &centre : nullptr);
    } else {
        world->frame(nullptr, nullptr);
    }
    if (const Atmosphere *atmosphere = world->get_atmosphere(); atmosphere && atmosphere->range_m() != atmosphere_range_sent) {
        atmosphere_range_sent = atmosphere->range_m();
        RenderingServer::get_singleton()->global_shader_parameter_set("atmo_range_m", atmosphere_range_sent);
    }
    if (const Clouds *clouds = world->get_clouds()) {
        // The shadow map's frame, as the pass of this frame takes it (the globals that changed).
        Clouds::visit_host_frame(clouds->last_frame(), SetChangedGlobal{ cloud_globals_sent });
    }
    if (const Vector3 wind = godot_v3(world->camera_wind()); wind != weather_wind_sent) {
        // The leaves' sway (VoxelSky::glsl_clouds' wind_sway): the weather's wind here.
        weather_wind_sent = wind;
        RenderingServer::get_singleton()->global_shader_parameter_set("weather_wind", wind);
    }
    if (world->world_file().sea.on) {
        _push_sun_without_sky();
    }
    // Tracing reads every captured frame (the GPU track).
    if ((debug_gpu_profile && world->frames() % 15 == 0) || debug_gpu_profile_frames || Trace::enabled()) {
        if (!gpu_profiler) {
            gpu_profiler = memnew(GPUProfiler);
            gpu_profiler->set_recording(debug_gpu_profile_frames);
        }
        gpu_profiler->poll();
    }
}

bool VoxelTerrain::debug_audit_pages() {
    return world->debug_audit_pages();
}

Dictionary VoxelTerrain::get_audit_report() const {
    return godot_dict(world->audit_report());
}

bool VoxelTerrain::debug_audit_discovery() {
    return world->debug_audit_discovery();
}

Dictionary VoxelTerrain::get_discovery_audit_report() const {
    return godot_dict(world->discovery_audit_report());
}

bool VoxelTerrain::is_load_complete() {
    if (camera == nullptr) {
        return world->is_load_complete(nullptr);
    }
    const Float3 pos = ov_f3(has_lod_centre ? lod_centre : camera->get_global_position());
    return world->is_load_complete(&pos);
}

bool VoxelTerrain::is_edit_settled() {
    return world->is_edit_settled();
}

Dictionary VoxelTerrain::get_debug_stats() {
    Dictionary d = godot_dict(world->debug_stats());
    if (gpu_profiler && world->is_started()) {
        d["gpu_breakdown"] = gpu_profiler->get_breakdown();
    }
    return d;
}

void VoxelTerrain::_stop_world(ov::World::Kept *r_kept) {
    if (world->get_atmosphere()) {
        // Off on the pages and the sky before its textures go (the stop's last call frees them).
        Dictionary off;
        off["atmo_enabled"] = false;
        off["atmo_volume"] = RID();
        off["atmo_multiscatter"] = RID();
        off["cloud_enabled"] = false;
        for (const char *name : Clouds::HOST_TEXTURES) {
            off[name] = RID();
        }
        _push_atmosphere(off);
    }
    world->stop(r_kept);
}

void VoxelTerrain::_teardown() {
    if (gpu_profiler) {
        RDUtils::delete_on_render_thread(gpu_profiler);
        gpu_profiler = nullptr;
    }
    _stop_world();
    if (page_host) {
        page_host->destroy(); // after the mesher's teardown, on the render thread too
        page_host = nullptr;
    }
    if (main_device) {
        // Last on the render thread: everything posted before still uses it.
        main_device->post([device = main_device]() { delete device; });
        main_device = nullptr;
    }
}

void VoxelTerrain::_exit_tree() {
    _teardown();
}

VoxelTerrain::~VoxelTerrain() {
    _teardown();
}
