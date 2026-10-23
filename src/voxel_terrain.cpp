#include "voxel_terrain.hpp"
#include "godot_conv.hpp"
#include "rd_utils.hpp"
#include <cmath>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

void VoxelTerrain::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_pipeline_mode", "mode"), &VoxelTerrain::set_pipeline_mode);
    ClassDB::bind_method(D_METHOD("get_pipeline_mode"), &VoxelTerrain::get_pipeline_mode);
    ClassDB::bind_method(D_METHOD("set_terrain_mode", "mode"), &VoxelTerrain::set_terrain_mode);
    ClassDB::bind_method(D_METHOD("get_terrain_mode"), &VoxelTerrain::get_terrain_mode);
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
    ClassDB::bind_method(D_METHOD("set_archetype_instances", "count"), &VoxelTerrain::set_archetype_instances);
    ClassDB::bind_method(D_METHOD("get_archetype_instances"), &VoxelTerrain::get_archetype_instances);
    ClassDB::bind_method(D_METHOD("set_archetype_instance_area", "area"), &VoxelTerrain::set_archetype_instance_area);
    ClassDB::bind_method(D_METHOD("get_archetype_instance_area"), &VoxelTerrain::get_archetype_instance_area);
    ClassDB::bind_method(D_METHOD("set_archetype_trees", "count"), &VoxelTerrain::set_archetype_trees);
    ClassDB::bind_method(D_METHOD("get_archetype_trees"), &VoxelTerrain::get_archetype_trees);
    ClassDB::bind_method(D_METHOD("set_archetype_tree_area", "area"), &VoxelTerrain::set_archetype_tree_area);
    ClassDB::bind_method(D_METHOD("get_archetype_tree_area"), &VoxelTerrain::get_archetype_tree_area);
    ClassDB::bind_method(D_METHOD("set_biome_radius", "metres"), &VoxelTerrain::set_biome_radius);
    ClassDB::bind_method(D_METHOD("get_biome_radius"), &VoxelTerrain::get_biome_radius);
    ClassDB::bind_method(D_METHOD("set_climate_path", "path"), &VoxelTerrain::set_climate_path);
    ClassDB::bind_method(D_METHOD("get_climate_path"), &VoxelTerrain::get_climate_path);
    ClassDB::bind_method(D_METHOD("set_archetype_band", "enabled"), &VoxelTerrain::set_archetype_band);
    ClassDB::bind_method(D_METHOD("get_archetype_band"), &VoxelTerrain::get_archetype_band);
    ClassDB::bind_method(D_METHOD("set_archetype_stream", "enabled"), &VoxelTerrain::set_archetype_stream);
    ClassDB::bind_method(D_METHOD("get_archetype_stream"), &VoxelTerrain::get_archetype_stream);
    ClassDB::bind_method(D_METHOD("set_archetype_stream_part_sides", "sides"), &VoxelTerrain::set_archetype_stream_part_sides);
    ClassDB::bind_method(D_METHOD("get_archetype_stream_part_sides"), &VoxelTerrain::get_archetype_stream_part_sides);
    ClassDB::bind_method(D_METHOD("set_archetype_moments", "enabled"), &VoxelTerrain::set_archetype_moments);
    ClassDB::bind_method(D_METHOD("get_archetype_moments"), &VoxelTerrain::get_archetype_moments);
    ClassDB::bind_method(D_METHOD("set_archetype_erode", "cells"), &VoxelTerrain::set_archetype_erode);
    ClassDB::bind_method(D_METHOD("get_archetype_erode"), &VoxelTerrain::get_archetype_erode);
    ClassDB::bind_method(D_METHOD("set_archetype_resolve", "cells"), &VoxelTerrain::set_archetype_resolve);
    ClassDB::bind_method(D_METHOD("get_archetype_resolve"), &VoxelTerrain::get_archetype_resolve);
    ClassDB::bind_method(D_METHOD("set_lod_pixel_error", "error"), &VoxelTerrain::set_lod_pixel_error);
    ClassDB::bind_method(D_METHOD("get_lod_pixel_error"), &VoxelTerrain::get_lod_pixel_error);
    ClassDB::bind_method(D_METHOD("set_lod_pixel_error_back", "error"), &VoxelTerrain::set_lod_pixel_error_back);
    ClassDB::bind_method(D_METHOD("get_lod_pixel_error_back"), &VoxelTerrain::get_lod_pixel_error_back);
    ClassDB::bind_method(D_METHOD("set_lod_view_margin_deg", "degrees"), &VoxelTerrain::set_lod_view_margin_deg);
    ClassDB::bind_method(D_METHOD("get_lod_view_margin_deg"), &VoxelTerrain::get_lod_view_margin_deg);
    ClassDB::bind_method(D_METHOD("set_lod_view_memory_m", "metres"), &VoxelTerrain::set_lod_view_memory_m);
    ClassDB::bind_method(D_METHOD("get_lod_view_memory_m"), &VoxelTerrain::get_lod_view_memory_m);
    ClassDB::bind_method(D_METHOD("set_lod_numbers", "count"), &VoxelTerrain::set_lod_numbers);
    ClassDB::bind_method(D_METHOD("get_lod_numbers"), &VoxelTerrain::get_lod_numbers);
    ClassDB::bind_method(D_METHOD("set_view_lods", "count"), &VoxelTerrain::set_view_lods);
    ClassDB::bind_method(D_METHOD("get_view_lods"), &VoxelTerrain::get_view_lods);
    ClassDB::bind_method(D_METHOD("get_load_radius"), &VoxelTerrain::get_load_radius);
    ClassDB::bind_method(D_METHOD("get_sdf_at", "render_pos"), &VoxelTerrain::get_sdf_at);
    ClassDB::bind_method(D_METHOD("get_tangent_position", "tangent_m"), &VoxelTerrain::get_tangent_position);
    ClassDB::bind_method(D_METHOD("set_skirt_depth", "depth"), &VoxelTerrain::set_skirt_depth);
    ClassDB::bind_method(D_METHOD("get_skirt_depth"), &VoxelTerrain::get_skirt_depth);
    ClassDB::bind_method(D_METHOD("set_legacy_trim_indices", "trim"), &VoxelTerrain::set_legacy_trim_indices);
    ClassDB::bind_method(D_METHOD("get_legacy_trim_indices"), &VoxelTerrain::get_legacy_trim_indices);
    ClassDB::bind_method(D_METHOD("set_dual_contouring", "enable"), &VoxelTerrain::set_dual_contouring);
    ClassDB::bind_method(D_METHOD("get_dual_contouring"), &VoxelTerrain::get_dual_contouring);
    ClassDB::bind_method(D_METHOD("get_pending_chunk_count"), &VoxelTerrain::get_pending_chunk_count);
    ClassDB::bind_method(D_METHOD("set_debug_gpu_profile", "enable"), &VoxelTerrain::set_debug_gpu_profile);
    ClassDB::bind_method(D_METHOD("get_debug_gpu_profile"), &VoxelTerrain::get_debug_gpu_profile);
    ClassDB::bind_method(D_METHOD("set_debug_gpu_profile_frames", "enable"), &VoxelTerrain::set_debug_gpu_profile_frames);
    ClassDB::bind_method(D_METHOD("get_debug_gpu_profile_frames"), &VoxelTerrain::get_debug_gpu_profile_frames);
    ClassDB::bind_method(D_METHOD("take_gpu_profile_frames"), &VoxelTerrain::take_gpu_profile_frames);
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
    ClassDB::bind_method(D_METHOD("set_destroy_verify", "enable"), &VoxelTerrain::set_destroy_verify);
    ClassDB::bind_method(D_METHOD("get_destroy_verify"), &VoxelTerrain::get_destroy_verify);
    ClassDB::bind_method(D_METHOD("set_discovery_verify", "enable"), &VoxelTerrain::set_discovery_verify);
    ClassDB::bind_method(D_METHOD("get_discovery_verify"), &VoxelTerrain::get_discovery_verify);
    ClassDB::bind_method(D_METHOD("set_walk_gpu", "enable"), &VoxelTerrain::set_walk_gpu);
    ClassDB::bind_method(D_METHOD("get_walk_gpu"), &VoxelTerrain::get_walk_gpu);
    ClassDB::bind_method(D_METHOD("set_walk_verify", "enable"), &VoxelTerrain::set_walk_verify);
    ClassDB::bind_method(D_METHOD("set_walk_batch", "keys"), &VoxelTerrain::set_walk_batch);
    ClassDB::bind_method(D_METHOD("get_walk_batch"), &VoxelTerrain::get_walk_batch);
    ClassDB::bind_method(D_METHOD("get_walk_verify"), &VoxelTerrain::get_walk_verify);
    ClassDB::bind_method(D_METHOD("set_discoverer_mode", "mode"), &VoxelTerrain::set_discoverer_mode);
    ClassDB::bind_method(D_METHOD("get_discoverer_mode"), &VoxelTerrain::get_discoverer_mode);
    ClassDB::bind_method(D_METHOD("set_discovery_compare", "enable"), &VoxelTerrain::set_discovery_compare);
    ClassDB::bind_method(D_METHOD("get_discovery_compare"), &VoxelTerrain::get_discovery_compare);
    ClassDB::bind_method(D_METHOD("add_instance", "type", "x", "y", "z", "rotation", "params", "op", "k"), &VoxelTerrain::add_instance,
            DEFVAL(PackedFloat32Array()), DEFVAL(PackedFloat32Array()), DEFVAL(0), DEFVAL(0.0f));
    ClassDB::bind_method(D_METHOD("remove_instance", "id"), &VoxelTerrain::remove_instance);
    ClassDB::bind_method(D_METHOD("move_instance", "id", "x", "y", "z", "rotation"), &VoxelTerrain::move_instance, DEFVAL(PackedFloat32Array()));

    ADD_PROPERTY(PropertyInfo(Variant::INT, "pipeline_mode", PROPERTY_HINT_ENUM, "GPU Legacy:1,GPU Direct:2"), "set_pipeline_mode", "get_pipeline_mode");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "terrain_mode", PROPERTY_HINT_ENUM, "Cascade:1,Continents:2,Shapes:4"), "set_terrain_mode", "get_terrain_mode");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "legacy_batch_size", PROPERTY_HINT_RANGE, "1,4096"), "set_legacy_batch_size", "get_legacy_batch_size");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "archetype_instances", PROPERTY_HINT_RANGE, "0,1000000,1"), "set_archetype_instances", "get_archetype_instances");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "archetype_instance_area", PROPERTY_HINT_RANGE, "1,1000000,1,suffix:m"), "set_archetype_instance_area", "get_archetype_instance_area");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "archetype_trees", PROPERTY_HINT_RANGE, "0,100000000,1"), "set_archetype_trees", "get_archetype_trees");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "archetype_tree_area", PROPERTY_HINT_RANGE, "1,1000000,1,suffix:m"), "set_archetype_tree_area", "get_archetype_tree_area");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "biome_radius", PROPERTY_HINT_RANGE, "0,100000,1,suffix:m"), "set_biome_radius", "get_biome_radius");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "climate_path", PROPERTY_HINT_FILE, "*.json"), "set_climate_path", "get_climate_path");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "archetype_band"), "set_archetype_band", "get_archetype_band");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "archetype_stream"), "set_archetype_stream", "get_archetype_stream");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "archetype_stream_part_sides", PROPERTY_HINT_RANGE,
                         vformat("1,%d,1", Archetype::ScatterSource::STREAM_PART_CELL_SIDES_MAX)),
            "set_archetype_stream_part_sides",
            "get_archetype_stream_part_sides");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "archetype_moments"), "set_archetype_moments", "get_archetype_moments");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "archetype_erode", PROPERTY_HINT_RANGE, "0.05,4,0.05"), "set_archetype_erode", "get_archetype_erode");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "archetype_resolve", PROPERTY_HINT_RANGE, "0,8,0.25"), "set_archetype_resolve", "get_archetype_resolve");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size", PROPERTY_HINT_RANGE, "0.01,8,0.01,suffix:m"), "set_voxel_size", "get_voxel_size");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_pixel_error", PROPERTY_HINT_RANGE, "0.25,64,0.25"), "set_lod_pixel_error", "get_lod_pixel_error");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_pixel_error_back", PROPERTY_HINT_RANGE, "0,64,0.25"), "set_lod_pixel_error_back", "get_lod_pixel_error_back");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_view_margin_deg", PROPERTY_HINT_RANGE, vformat("%d,180,1,suffix:deg", int(ChunkDiscoverer::RESCAN_TURN_DEG))), "set_lod_view_margin_deg", "get_lod_view_margin_deg");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_view_memory_m", PROPERTY_HINT_RANGE, "-1,100000,1,or_greater,suffix:m"), "set_lod_view_memory_m", "get_lod_view_memory_m");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_numbers", PROPERTY_HINT_RANGE, "1,31"), "set_lod_numbers", "get_lod_numbers");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "view_lods", PROPERTY_HINT_RANGE, "0,31"), "set_view_lods", "get_view_lods");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "skirt_depth", PROPERTY_HINT_RANGE, "0,4,0.05,suffix:voxels"), "set_skirt_depth", "get_skirt_depth");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "legacy_trim_indices"), "set_legacy_trim_indices", "get_legacy_trim_indices");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dual_contouring"), "set_dual_contouring", "get_dual_contouring");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "discovery_cull_mode", PROPERTY_HINT_ENUM, "None:0,Legacy:1"), "set_discovery_cull_mode", "get_discovery_cull_mode");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "discovery_cache"), "set_discovery_cache", "get_discovery_cache");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "discoverer_mode", PROPERTY_HINT_ENUM, "Auto:-1,CPU:0,GPU:1"), "set_discoverer_mode", "get_discoverer_mode");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "walk_gpu"), "set_walk_gpu", "get_walk_gpu");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "walk_verify"), "set_walk_verify", "get_walk_verify");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "walk_batch", PROPERTY_HINT_RANGE, "1,4096"), "set_walk_batch", "get_walk_batch");
    ADD_GROUP("World Origin", "world_origin_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_x", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_x", "get_world_origin_x");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_y", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_y", "get_world_origin_y");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "world_origin_z", PROPERTY_HINT_NONE, "suffix:voxels"), "set_world_origin_z", "get_world_origin_z");
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
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "discovery_verify"), "set_discovery_verify", "get_discovery_verify");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "destroy_verify"), "set_destroy_verify", "get_destroy_verify");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "discovery_compare"), "set_discovery_compare", "get_discovery_compare");

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

void VoxelTerrain::set_terrain_mode(int p_mode) {
    ERR_FAIL_COND_MSG(started, "terrain_mode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_mode != 1 && p_mode != 2 && p_mode != 4, "terrain_mode must be 1 (cascade), 2 (continents preset) or 4 (shapes gallery).");
    terrain_mode = p_mode;
}

int VoxelTerrain::get_terrain_mode() const {
    return terrain_mode;
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

void VoxelTerrain::set_destroy_verify(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "destroy_verify must be set before the terrain enters the tree.");
    destroy_verify = p_enable;
}

bool VoxelTerrain::get_destroy_verify() const {
    return destroy_verify;
}

void VoxelTerrain::set_discoverer_mode(int p_mode) {
    ERR_FAIL_COND_MSG(started, "discoverer_mode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_mode < -1 || p_mode > 1, "Unknown discoverer_mode (-1 auto, 0 CPU, 1 GPU).");
    discoverer_mode = p_mode;
}

int VoxelTerrain::get_discoverer_mode() const {
    return discoverer_mode;
}

void VoxelTerrain::set_walk_gpu(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "walk_gpu must be set before the terrain enters the tree.");
    walk_gpu = p_enable;
}

bool VoxelTerrain::get_walk_gpu() const {
    return walk_gpu;
}

void VoxelTerrain::set_walk_verify(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "walk_verify must be set before the terrain enters the tree.");
    walk_verify = p_enable;
}

bool VoxelTerrain::get_walk_verify() const {
    return walk_verify;
}

void VoxelTerrain::set_walk_batch(int p_keys) {
    ERR_FAIL_COND_MSG(started, "walk_batch must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_keys < 1, "walk_batch must be at least 1.");
    walk_batch = p_keys;
}

int VoxelTerrain::get_walk_batch() const {
    return walk_batch;
}

void VoxelTerrain::set_discovery_compare(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "discovery_compare must be set before the terrain enters the tree.");
    discovery_compare = p_enable;
}

bool VoxelTerrain::get_discovery_compare() const {
    return discovery_compare;
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

void VoxelTerrain::set_archetype_instances(int p_count) {
    ERR_FAIL_COND_MSG(started, "archetype_instances must be set before the terrain enters the tree.");
    archetype_instances = p_count;
}

int VoxelTerrain::get_archetype_instances() const {
    return archetype_instances;
}

void VoxelTerrain::set_archetype_instance_area(double p_area) {
    ERR_FAIL_COND_MSG(started, "archetype_instance_area must be set before the terrain enters the tree.");
    archetype_instance_area = p_area;
}

double VoxelTerrain::get_archetype_instance_area() const {
    return archetype_instance_area;
}

void VoxelTerrain::set_archetype_trees(int p_count) {
    ERR_FAIL_COND_MSG(started, "archetype_trees must be set before the terrain enters the tree.");
    archetype_trees = p_count;
}

int VoxelTerrain::get_archetype_trees() const {
    return archetype_trees;
}

void VoxelTerrain::set_archetype_tree_area(double p_area) {
    ERR_FAIL_COND_MSG(started, "archetype_tree_area must be set before the terrain enters the tree.");
    archetype_tree_area = p_area;
}

double VoxelTerrain::get_archetype_tree_area() const {
    return archetype_tree_area;
}

void VoxelTerrain::set_biome_radius(double p_metres) {
    ERR_FAIL_COND_MSG(started, "biome_radius must be set before the terrain enters the tree.");
    biome_radius = p_metres;
}

double VoxelTerrain::get_biome_radius() const {
    return biome_radius;
}

void VoxelTerrain::set_climate_path(const String &p_path) {
    ERR_FAIL_COND_MSG(started, "climate_path must be set before the terrain enters the tree.");
    climate_path = p_path;
}

String VoxelTerrain::get_climate_path() const {
    return climate_path;
}

void VoxelTerrain::set_archetype_band(bool p_enabled) {
    ERR_FAIL_COND_MSG(started, "archetype_band must be set before the terrain enters the tree.");
    archetype_band = p_enabled;
}

bool VoxelTerrain::get_archetype_band() const {
    return archetype_band;
}

void VoxelTerrain::set_archetype_stream(bool p_enabled) {
    ERR_FAIL_COND_MSG(started, "archetype_stream must be set before the terrain enters the tree.");
    archetype_stream = p_enabled;
}

bool VoxelTerrain::get_archetype_stream() const {
    return archetype_stream;
}

void VoxelTerrain::set_archetype_stream_part_sides(int p_sides) {
    ERR_FAIL_COND_MSG(started, "archetype_stream_part_sides must be set before the terrain enters the tree.");
    // Past it, far layers become part layers whose blob tiles read cubically many LOD-0 part tiles.
    ERR_FAIL_COND_MSG(p_sides < 1 || p_sides > int(Archetype::ScatterSource::STREAM_PART_CELL_SIDES_MAX),
            vformat("archetype_stream_part_sides must be within [1, %d] source cells.", Archetype::ScatterSource::STREAM_PART_CELL_SIDES_MAX));
    archetype_stream_part_sides = p_sides;
}

int VoxelTerrain::get_archetype_stream_part_sides() const {
    return archetype_stream_part_sides;
}

void VoxelTerrain::set_archetype_moments(bool p_enabled) {
    ERR_FAIL_COND_MSG(started, "archetype_moments must be set before the terrain enters the tree.");
    archetype_moments = p_enabled;
}

bool VoxelTerrain::get_archetype_moments() const {
    return archetype_moments;
}

void VoxelTerrain::set_archetype_erode(float p_cells) {
    ERR_FAIL_COND_MSG(started, "archetype_erode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(!(p_cells > 0.0f && p_cells <= 4.0f), "archetype_erode must be in (0, 4] cells per LOD.");
    archetype_erode = p_cells;
}

float VoxelTerrain::get_archetype_erode() const {
    return archetype_erode;
}

void VoxelTerrain::set_archetype_resolve(float p_cells) {
    ERR_FAIL_COND_MSG(started, "archetype_resolve must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(!(p_cells >= 0.0f && p_cells <= 8.0f), "archetype_resolve must be in [0, 8] cells.");
    archetype_resolve = p_cells;
}

float VoxelTerrain::get_archetype_resolve() const {
    return archetype_resolve;
}

void VoxelTerrain::set_lod_pixel_error(float p_error) {
    ERR_FAIL_COND_MSG(started, "lod_pixel_error must be set before the terrain enters the tree.");
    lod_pixel_error = p_error;
}

float VoxelTerrain::get_lod_pixel_error() const {
    return lod_pixel_error;
}

void VoxelTerrain::set_lod_pixel_error_back(float p_error) {
    ERR_FAIL_COND_MSG(started, "lod_pixel_error_back must be set before the terrain enters the tree.");
    lod_pixel_error_back = p_error;
}

float VoxelTerrain::get_lod_pixel_error_back() const {
    return lod_pixel_error_back;
}

void VoxelTerrain::set_lod_view_margin_deg(float p_degrees) {
    ERR_FAIL_COND_MSG(started, "lod_view_margin_deg must be set before the terrain enters the tree.");
    // The remembered view axis lags the camera's by up to RESCAN_TURN_DEG: a smaller margin would leave the
    // frustum's edge outside the cone.
    ERR_FAIL_COND_MSG(!(p_degrees >= ChunkDiscoverer::RESCAN_TURN_DEG && p_degrees <= 180.0f),
            vformat("lod_view_margin_deg must be in [%d, 180].", int(ChunkDiscoverer::RESCAN_TURN_DEG)));
    lod_view_margin_deg = p_degrees;
}

float VoxelTerrain::get_lod_view_margin_deg() const {
    return lod_view_margin_deg;
}

void VoxelTerrain::set_lod_view_memory_m(float p_metres) {
    ERR_FAIL_COND_MSG(started, "lod_view_memory_m must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(std::isnan(p_metres), "lod_view_memory_m must be a number.");
    lod_view_memory_m = p_metres;
}

float VoxelTerrain::get_lod_view_memory_m() const {
    return lod_view_memory_m;
}

Vector3 VoxelTerrain::_camera_forward() const {
    return -camera->get_global_transform().basis.get_column(2);
}

void VoxelTerrain::set_lod_numbers(int p_count) {
    ERR_FAIL_COND_MSG(started, "lod_numbers must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_count < 1 || p_count > MAX_LOD_NUMBERS, vformat("lod_numbers must be in [1, %d].", MAX_LOD_NUMBERS));
    lod_numbers = p_count;
}

double VoxelTerrain::get_sdf_at(const Vector3 &p_render_pos) const {
    const VoxelPos origin = chunk_param.origin_voxel();
    const double vs = double(chunk_param.voxel_size);
    const double offset[3] = { double(p_render_pos.x) / vs, double(p_render_pos.y) / vs, double(p_render_pos.z) / vs };
    WorldPos p = WorldCoord::offset_pos(origin, offset);
    p.frac = Float3(); // the cascade only exists on lattice points: the LOD0 voxel corner (within one voxel)
    return sdf->evaluate_view(p, 0); // called every frame: streamed instances come once resident
}

Vector3 VoxelTerrain::get_tangent_position(const Vector2 &p_tangent_m) const {
    const double vs = double(chunk_param.voxel_size);
    const VoxelPos p = sdf->get_base().tangent_point(std::llround(double(p_tangent_m.x) / vs), std::llround(double(p_tangent_m.y) / vs));
    const VoxelPos origin = chunk_param.origin_voxel();
    return Vector3(real_t(double(p.x - origin.x) * vs), real_t(double(p.y - origin.y) * vs), real_t(double(p.z - origin.z) * vs));
}

int VoxelTerrain::get_lod_numbers() const {
    return lod_numbers;
}

void VoxelTerrain::set_view_lods(int p_count) {
    ERR_FAIL_COND_MSG(started, "view_lods must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_count < 0 || p_count > MAX_LOD_NUMBERS, vformat("view_lods must be in [0, %d].", MAX_LOD_NUMBERS));
    view_lods = p_count;
}

int VoxelTerrain::get_view_lods() const {
    return view_lods;
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

void VoxelTerrain::set_dual_contouring(bool p_enable) {
    ERR_FAIL_COND_MSG(started, "dual_contouring must be set before the terrain enters the tree.");
    dual_contouring = p_enable;
}

bool VoxelTerrain::get_dual_contouring() const {
    return dual_contouring;
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

Dictionary VoxelTerrain::get_process_times_ms() const {
    Dictionary d;
    for (const auto &[name, stat] : process_stats.sections()) {
        d[name] = stat->total / 1000.0;
    }
    return d;
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

void VoxelTerrain::set_direct_batches_per_frame(int p_batches) {
    ERR_FAIL_COND_MSG(started, "direct_batches_per_frame must be set before the terrain enters the tree.");
    ERR_FAIL_COND(p_batches < 1 || p_batches > (int)GPUDirectMesher::MAX_BATCHES_PER_FRAME);
    direct_settings.max_batches_per_frame = p_batches;
}

int VoxelTerrain::get_direct_batches_per_frame() const {
    return direct_settings.max_batches_per_frame;
}

void VoxelTerrain::set_direct_gpu_budget_ms(float p_ms) {
    ERR_FAIL_COND_MSG(started, "direct_gpu_budget_ms must be set before the terrain enters the tree.");
    ERR_FAIL_COND(!(p_ms >= 0.0f)); // 0: no budget
    direct_settings.gpu_budget_ms = p_ms;
}

float VoxelTerrain::get_direct_gpu_budget_ms() const {
    return direct_settings.gpu_budget_ms;
}

void VoxelTerrain::set_direct_max_in_flight(int p_batches) {
    ERR_FAIL_COND_MSG(started, "direct_max_in_flight must be set before the terrain enters the tree.");
    ERR_FAIL_COND(p_batches < 0); // 0: 3 * direct_batches_per_frame
    direct_settings.max_in_flight = p_batches;
}

int VoxelTerrain::get_direct_max_in_flight() const {
    return direct_settings.max_in_flight;
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

void VoxelTerrain::set_page_params(const Dictionary &p_params) {
    Array keys = p_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        page_params[keys[i]] = p_params[keys[i]];
        if (direct_mesher) {
            direct_mesher->get_pool()->set_material_param(keys[i], p_params[keys[i]]);
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
    // One chunk-id unit is half a LOD0 chunk, a whole number of voxels.
    ERR_FAIL_COND_MSG(chunk_param.chunk_voxel_dimension() % 2 != 0, "VoxelTerrain: chunk_size - 2 * skirt_size must be even.");
    ERR_FAIL_COND_MSG(lod_numbers < 1 || lod_numbers > MAX_LOD_NUMBERS, "VoxelTerrain: lod_numbers out of range.");
    // Both meshers need a RenderingDevice (GPU Direct the main one, Legacy a local one, which needs it too).
    // Without one (--headless, the Compatibility renderer) the page pool would dereference null on the
    // render thread: the terrain stays empty instead. This also covers the editor's headless --import.
    ERR_FAIL_NULL_MSG(RenderingServer::get_singleton()->get_rendering_device(),
            "VoxelTerrain: no RenderingDevice (--headless or the Compatibility renderer): the terrain is disabled.");

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
            VLOG(1, "VoxelTerrain: world origin snapped to ", godot_str(chunk_param.origin_voxel()), " voxels");
        }
    }

    // One terrain, the cascade: mode 2 is its continents preset, 4 its flat preset under the shapes gallery.
    if (terrain_mode == 2) {
        sdf->preset = SDFCascade::Preset::CONTINENTS;
    } else if (terrain_mode == 4) {
        sdf->preset = SDFCascade::Preset::FLAT;
        sdf->set_scene(SDFTable::Scene::SHAPES);
    }
    sdf->set_scatter(archetype_instances, archetype_instance_area);
    sdf->set_forest(archetype_trees, archetype_tree_area);
    sdf->set_stream_sources(archetype_stream);
    sdf->set_stream_part_sides(archetype_stream_part_sides);
    sdf->set_biome_radius(biome_radius);
    if (!climate_path.is_empty()) {
        // Parsed and checked by configure() (an ERR there: no climate).
        const String err = Climate::load(climate_path, sdf->climate_data);
        if (!err.is_empty()) {
            ERR_PRINT("VoxelTerrain: " + err + String(" No climate."));
        }
    }
    sdf->set_instance_band(archetype_band && terrain_mode != 4); // the gallery: a crease test at every LOD, never eroded
    sdf->set_moment_blobs(archetype_moments);
    sdf->set_instance_erode(archetype_erode);
    sdf->set_instance_resolve(archetype_resolve);
    sdf->set_walk_batch_gpu(size_t(walk_batch));
    const float min_skirt = sdf->min_skirt_depth(chunk_param, lod_numbers);
    if (chunk_param.skirt_depth < min_skirt) {
        VLOG(1, "VoxelTerrain: skirt_depth raised from ", chunk_param.skirt_depth, " to ", min_skirt, " voxels (the terrain's band steps)");
        chunk_param.skirt_depth = min_skirt;
    }
    sdf->configure(chunk_param, lod_numbers);
    sdf->set_walk_verify(walk_verify);
    if (walk_gpu) {
        sdf->start_field_gpu(chunk_param); // attached by sdf->poll() once its device is ready
    }

    visual_mesh_data.set_max_lod(lod_numbers);
    visual_mesh_data.verify = destroy_verify;

    if (Engine::get_singleton()->is_editor_hint()) {
        camera = EditorInterface::get_singleton()->get_editor_viewport_3d()->get_camera_3d();
    } else {
        camera = get_viewport()->get_camera_3d();
    }

    chunk_discoverer = memnew(ChunkDiscoverer);

    chunk_discoverer->set_config(chunk_param, sdf);
    chunk_discoverer->set_cull_mode((ChunkDiscoverer::CullMode)discovery_cull_mode);
    chunk_discoverer->set_cache(discovery_cache, discovery_verify);
    chunk_discoverer->set_gpu(discoverer_mode != 0);
    chunk_discoverer->set_compare(discovery_compare);

    // The pixel error is measured against the camera's real vertical FOV and the 3D render's height: an upscaler's
    // render scale (scaling_3d_scale) makes its pixels larger, and the LOD coarser with them.
    float viewport_height = 1080.0f;
    float fov_rad = Math::deg_to_rad(75.0f);
    float aspect = 16.0f / 9.0f;
    if (camera != nullptr) {
        Viewport *viewport = camera->get_viewport();
        const Vector2 viewport_size = viewport->get_visible_rect().size * float(viewport->get_scaling_3d_scale());
        viewport_height = MAX(viewport_size.y, 1.0f);
        aspect = viewport_size.x > 0.0f ? viewport_size.x / viewport_height : aspect;
        fov_rad = Math::deg_to_rad(float(camera->get_fov()));
        if (camera->get_keep_aspect_mode() == Camera3D::KEEP_WIDTH && viewport_size.x > 0.0f) {
            // fov is horizontal: convert it to vertical.
            fov_rad = 2.0f * std::atan(std::tan(fov_rad * 0.5f) * viewport_size.y / viewport_size.x);
        }
    }

    // The view cone holds the frustum: its half-angle is the half-FOV along the screen diagonal.
    chunk_discoverer->setup_lod(
            view_lods > 0 ? MIN(view_lods, lod_numbers) : lod_numbers,
            lod_pixel_error,
            viewport_height,
            fov_rad,
            lod_pixel_error_back,
            ChunkMath::view_cone_half_angle(fov_rad, aspect, Math::deg_to_rad(lod_view_margin_deg)));
    chunk_discoverer->set_view_memory(lod_view_memory_m);

    chunk_discoverer->start();

    scenario = get_world_3d()->get_scenario();

    if (pipeline_mode == PIPELINE_GPU_DIRECT) {
        direct_mesher = memnew(GPUDirectMesher);
        direct_mesher->set_dual_contouring(dual_contouring);
        direct_mesher->initialize(sdf, chunk_param, lod_numbers, scenario, direct_settings);
        direct_mesher->set_debug_view(debug_view);
        // Skirt depth per LOD: the page shader's height colours undo the skirt push.
        PackedFloat32Array skirt_depths;
        skirt_depths.resize(32);
        for (int l = 0; l < 32; l++) {
            skirt_depths.set(l, l < lod_numbers ? sdf->skirt_depth_m(l) : 0.0f);
        }
        direct_mesher->get_pool()->set_material_param("skirt_depth_m", skirt_depths);
        const UpAxis up = sdf->up_at(VoxelPos(0, sdf->get_datum_y(), 0));
        direct_mesher->get_pool()->set_material_param("base_up", Vector3(real_t(up[0]), real_t(up[1]), real_t(up[2])));
        if (sdf->preset == SDFCascade::Preset::CONTINENTS) {
            // Continents preset: the page shader colours heights around sea level (the cascade's datum).
            const double sea_m = ChunkMath::voxel_to_render(VoxelPos(0, sdf->get_datum_y(), 0), chunk_param).y;
            direct_mesher->get_pool()->set_material_param("sea_level_m", sea_m);
        }
        set_page_params(page_params); // set before the pool existed (VoxelSky's clouds)
    } else {
        gpu_mesher = memnew(GPUMesher);

        gpu_mesher->set_dual_contouring(dual_contouring);
        gpu_mesher->initialize(
                sdf,
                chunk_param,
                legacy_batch_size, // chunk_batch
                lod_numbers,
                scenario);
        gpu_mesher->set_trim_indices(legacy_trim_indices);
    }
    last_dispatch_times = DispatchTimes(); // the new mesher's start from 0

    if (camera != nullptr) {
        chunk_discoverer->set_camera(camera->get_global_position(), _camera_forward());
    }

    VLOG(1, "VoxelTerrain: started, pipeline_mode=", (int)pipeline_mode);
}

int VoxelTerrain::add_instance(int p_type, int64_t p_x, int64_t p_y, int64_t p_z, const PackedFloat32Array &p_rotation,
        const PackedFloat32Array &p_params, int p_op, float p_k) {
    if (!sdf) {
        return -1;
    }
    float rot[9];
    const float *rp = nullptr;
    if (p_rotation.size() >= 9) {
        for (int i = 0; i < 9; i++) {
            rot[i] = p_rotation[i];
        }
        rp = rot;
    }
    float params[8] = {};
    const float *pp = nullptr;
    if (!p_params.is_empty()) {
        for (int i = 0; i < 8; i++) {
            params[i] = i < p_params.size() ? p_params[i] : 0.0f;
        }
        pp = params;
    }
    const int id = sdf->add_instance(p_type, VoxelPos(p_x, p_y, p_z), rp, pp, p_op, p_k);
    if (id >= 0) {
        _apply_instance_edit();
    }
    return id;
}

bool VoxelTerrain::remove_instance(int p_id) {
    if (!sdf) {
        return false;
    }
    const bool ok = sdf->remove_instance(p_id);
    if (ok) {
        _apply_instance_edit();
    }
    return ok;
}

bool VoxelTerrain::move_instance(int p_id, int64_t p_x, int64_t p_y, int64_t p_z, const PackedFloat32Array &p_rotation) {
    if (!sdf) {
        return false;
    }
    float rot[9];
    const float *rp = nullptr;
    if (p_rotation.size() >= 9) {
        for (int i = 0; i < 9; i++) {
            rot[i] = p_rotation[i];
        }
        rp = rot;
    }
    const bool ok = sdf->move_instance(p_id, VoxelPos(p_x, p_y, p_z), rp);
    if (ok) {
        _apply_instance_edit();
    }
    return ok;
}

void VoxelTerrain::_apply_instance_edit() {
    const uint64_t t0 = now_usec();
    const VoxelBox &box = sdf->last_edit_box();
    if (!box.valid) {
        return;
    }
    // The loaded chunks whose samples the edit can change: their box grown by the locality margin at
    // their LOD meets the edit's region.
    std::vector<ChunkId> remesh;
    for (const auto &kv : visual_mesh_data.loaded_chunk) {
        if (sdf->edit_affects_chunk(kv.first)) {
            remesh.push_back(kv.first);
        }
    }
    if (direct_mesher) {
        direct_mesher->remesh_chunks(remesh);
    } else if (gpu_mesher) {
        gpu_mesher->remesh_chunks(remesh);
    }
    if (chunk_discoverer) {
        chunk_discoverer->invalidate(box);
    }
    destroy_dirty = true;
    edit_stats.edits++;
    edit_stats.remesh_last = (int64_t)remesh.size();
    edit_stats.remesh_total += (int64_t)remesh.size();
    edit_stats.apply.add(now_usec() - t0);
}

void VoxelTerrain::_record_dispatch_times(const DispatchTimes &p_now) {
    process_stats.gate_admit.add(p_now.gate_admit - last_dispatch_times.gate_admit);
    process_stats.build_lists.add(p_now.build_lists - last_dispatch_times.build_lists);
    process_stats.record_bytes.add(p_now.record_bytes - last_dispatch_times.record_bytes);
    last_dispatch_times = p_now;
}

void VoxelTerrain::_process(double) {
    if (!started) {
        return;
    }

    const uint64_t t_process_start = now_usec();
    visual_mesh_data.frame = exec_time;
    sdf->poll(); // an edit's rebuilt index base, the streamed keys computed since the last frame (before the meshers ask)
    process_stats.stream_poll.add(now_usec() - t_process_start);

    if (camera != nullptr) {
        chunk_discoverer->set_camera(camera->get_global_position(), _camera_forward());
    }

    uint64_t t0 = now_usec();
    std::unique_ptr<LODHashSet> new_scanned_chunk;
    std::unique_ptr<ScanDelta> scan_delta;

    bool dirty_scan = chunk_discoverer->get_scanned_chunks(new_scanned_chunk, scan_delta, scan_view);

    if (dirty_scan) {
        scan_received = true;
        destroy_dirty = true;

        const uint64_t t_apply = now_usec();
        visual_mesh_data.add_scan_chunk(std::move(new_scanned_chunk), *scan_delta);
        process_stats.scan_apply.add(now_usec() - t_apply);
        if (direct_mesher) {
            direct_mesher->send_chunk(visual_mesh_data.dirty_chunk, scan_view);
        } else {
            gpu_mesher->send_chunk(visual_mesh_data.dirty_chunk, scan_view);
        }
    }
    process_stats.scan_ingest.add(now_usec() - t0);

    // Only a new scan or new results change what can be destroyed.
    if (destroy_dirty) {
        destroy_dirty = false;
        t0 = now_usec();
        visual_mesh_data.update_destroy();
        process_stats.destroy_from_scan.add(now_usec() - t0);
        if (destroy_verify) {
            t0 = now_usec();
            const int64_t mismatches = visual_mesh_data.verify_against_full();
            process_stats.destroy_verify.add(now_usec() - t0);
            if (mismatches > 0) {
                ERR_PRINT(String("VoxelTerrain: incremental destroy check differs from the full recompute: ") + String::num_int64(mismatches) +
                        " differences (frame " + String::num_int64(exec_time) + ").");
            }
        }
    }

    t0 = now_usec();
    if (direct_mesher) {
        direct_mesher->process_queue();
    } else {
        gpu_mesher->process_queue();
    }
    process_stats.mesher_dispatch.add(now_usec() - t0);
    _record_dispatch_times(direct_mesher ? direct_mesher->get_dispatch_times() : gpu_mesher->get_dispatch_times());

    t0 = now_usec();
    const bool results = direct_mesher ? direct_mesher->get_results(visual_mesh_data) : gpu_mesher->get_results(visual_mesh_data);
    destroy_dirty = destroy_dirty || results;
    process_stats.mesher_collect.add(now_usec() - t0);

    t0 = now_usec();
    visual_mesh_data.destroy_marked();
    process_stats.destroy_chunks.add(now_usec() - t0);

    // After destruction (to_destroy_chunk holds copies of the outputs it moves), and never while the
    // validator or auditor hold page ranges.
    t0 = now_usec();
    if (direct_mesher && !validator && !auditor) {
        direct_mesher->compact(visual_mesh_data);
    }
    process_stats.compact.add(now_usec() - t0);

    // The stalled chunks hide the sub-cubes their replacements cover. After compaction: the rows are current.
    t0 = now_usec();
    visual_mesh_data.push_coverage();
    process_stats.coverage.add(now_usec() - t0);

    if ((debug_gpu_profile && exec_time % 15 == 0) || debug_gpu_profile_frames) {
        if (!gpu_profiler) {
            gpu_profiler = memnew(GPUProfiler);
            gpu_profiler->set_recording(debug_gpu_profile_frames);
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
    for (const auto &kv : visual_mesh_data.loaded_chunk) {
        if (kv.second.owner) {
            MeshValidator::Target t;
            t.chunk_id = kv.first;
            t.page = kv.second.page;
            t.offset = kv.second.quad_offset;
            t.count = kv.second.quad_count;
            t.palette = kv.second.palette;
            candidates.push_back(t);
        }
    }
    ERR_FAIL_COND_V_MSG(candidates.empty(), false, "No non-empty loaded chunk to validate.");
    // (hash map order is arbitrary but stable enough for a spread of LODs)
    const std::vector<MeshValidator::Target> targets = MeshValidator::pick_targets(candidates, p_count);
    validation_report = Dictionary();
    validator = memnew(MeshValidator);
    validator->start(targets, sdf, chunk_param, lod_numbers, direct_mesher->get_pool(), p_dump_dir, dual_contouring);
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
    for (const auto &kv : visual_mesh_data.loaded_chunk) {
        if (!kv.second.owner) {
            continue;
        }
        if (!index_of_page.has(kv.second.page)) {
            index_of_page[kv.second.page] = pages.size();
            PageAuditor::PageInfo info;
            info.page = kv.second.page;
            pages.push_back(info);
        }
        pages[index_of_page[kv.second.page]].live.push_back({ kv.second.quad_offset, kv.second.quad_count, kv.first, kv.second.palette });
    }
    audit_report = Dictionary();
    auditor = memnew(PageAuditor);
    auditor->start(direct_mesher->get_pool(), pages, chunk_param);
    return true;
}

Dictionary VoxelTerrain::get_audit_report() const {
    return audit_report;
}

bool VoxelTerrain::debug_audit_discovery() {
    ERR_FAIL_COND_V_MSG(!direct_mesher, false, "debug_audit_discovery needs pipeline_mode = GPU Direct.");
    ERR_FAIL_COND_V_MSG(discovery_audit.running, false, "A discovery audit is already running.");
    ERR_FAIL_COND_V_MSG(!scan_received, false, "No scan received yet.");

    const uint64_t t0 = now_usec();
    const LODHashSet &scanned = *visual_mesh_data.scanned_chunk;
    LODHashSet reference;
    ChunkDiscoverer::ScanCounters counters;
    chunk_discoverer->scan(scan_view, ChunkDiscoverer::CULL_NONE, reference, counters);

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

    // LOD difference across chunk faces (the skirt hides a jump of 1).
    std::vector<int64_t> faces;
    face_lod_jumps(scanned, chunk_discoverer->get_max_lod(), faces);
    int max_jump = 0;
    Dictionary jumps; // jump -> faces
    for (int j = 0; j < (int)faces.size(); j++) {
        if (faces[j] > 0) {
            jumps[j] = faces[j];
            max_jump = j;
        }
    }

    discovery_audit = DiscoveryAudit();
    discovery_audit.running = true;
    discovery_audit.start_usec = t0;
    discovery_audit.expected = (int64_t)culled.size();
    Dictionary base;
    base["camera_chunk"] = godot_str(scan_view.camera_chunk);
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
            e["id"] = godot_str(c.first);
            e["lod"] = lod;
            e["quads"] = (int64_t)c.second;
            e["distance_chunks"] = scan_view.camera_chunk.distance_to(c.first);
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
    // The received scan must be the camera's current one: a camera moved (bench --above_ground) or turned
    // after _ready is otherwise "loaded" with the first scan while the next one is still running.
    // The view of the last set_camera (_process) carries the remembered views; a camera moved since is
    // caught by its chunk, a turn since by the next frame.
    if (camera != nullptr && (ChunkDiscoverer::view_changed(scan_view, chunk_discoverer->get_view()) || ChunkMath::chunk_at_render(ov_f3(camera->get_global_position()), chunk_param) != scan_view.camera_chunk)) {
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
    d["terrain_mode"] = terrain_mode;
    Dictionary origin;
    origin["requested_voxel"] = godot_str(world_origin_request);
    origin["voxel"] = godot_str(chunk_param.origin_voxel());
    origin["chunk"] = godot_str(chunk_param.origin_chunk);
    d["world_origin"] = origin;
    if (sdf) {
        Dictionary table = sdf->get_gpu_table()->get_stats();
        const Dictionary walk = sdf->walk_gpu_stats();
        if (!walk.is_empty()) {
            table["walk_gpu"] = walk;
        }
        d["table"] = table;
        if (const Climate::Table *climate = sdf->get_climate()) {
            Dictionary c;
            c["path"] = climate_path;
            c["fields"] = int64_t(climate->fields.size());
            c["biomes"] = int64_t(climate->biomes.size());
            c["modulated"] = sdf->get_cascade()->modulation != nullptr;
            d["climate"] = c;
        }
    }
    d["frames"] = (int64_t)exec_time;
    d["load_complete"] = is_load_complete();
    d["frames_until_loaded"] = frames_until_loaded;
    d["ms_until_loaded"] = usec_until_loaded < 0 ? -1.0 : usec_until_loaded / 1000.0;

    int64_t live_triangles = 0;
    int64_t drawn_triangles = 0;
    int64_t non_empty_chunks = 0;
    for (const auto &kv : visual_mesh_data.loaded_chunk) {
        live_triangles += kv.second.triangle_count;
        drawn_triangles += kv.second.drawn_triangles;
        non_empty_chunks += kv.second.triangle_count > 0 ? 1 : 0;
    }

    Dictionary chunks;
    chunks["scanned"] = (int64_t)visual_mesh_data.scanned_chunk->size();
    chunks["loaded"] = (int64_t)visual_mesh_data.loaded_chunk.size();
    chunks["loaded_non_empty"] = non_empty_chunks;
    chunks["pending"] = (int64_t)visual_mesh_data.dirty_chunk.size();
    // Settle check: once the load is complete both must be 0 (a scanned chunk not loaded is a hole).
    int64_t loaded_not_scanned = 0;
    int64_t scanned_not_loaded = 0;
    for (const auto &kv : visual_mesh_data.loaded_chunk) {
        loaded_not_scanned += visual_mesh_data.scanned_chunk->has(kv.first) ? 0 : 1;
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
    for (const auto &[name, stat] : process_stats.sections()) {
        process[name] = stat->to_dict();
    }
    d["process"] = process;

    const VoxelState::Stats &vs = visual_mesh_data.stats;
    Dictionary destroy;
    destroy["leaving"] = (int64_t)visual_mesh_data.leaving.size();
    destroy["leaving_max"] = vs.leaving_max;
    destroy["stalled"] = (int64_t)visual_mesh_data.stalled.size();
    destroy["rescans"] = vs.rescans;
    destroy["rescan_reanchored"] = vs.rescan_reanchored;
    destroy["rescan_ms_total"] = vs.rescan_usec / 1000.0;
    destroy["verify"] = destroy_verify;
    destroy["verify_checks"] = vs.verify_checks;
    destroy["verify_mismatches"] = vs.verify_mismatches;
    destroy["delta_mismatches"] = vs.delta_mismatches;
    destroy["octree_violations"] = vs.octree_violations;
    destroy["coverage_hidden_subcubes"] = visual_mesh_data.hidden_subcubes(); // now
    destroy["coverage_written"] = vs.coverage_written;
    destroy["coverage_written_max"] = vs.coverage_written_max;
    // Frames a stalled parent stays drawn over its children (first stalled sub-cube to the last cleared), cumulative.
    destroy["handoff_frames"] = godot_dict(vs.handoff_frames);
    d["destroy"] = destroy;

    if (gpu_profiler) {
        d["gpu_breakdown"] = gpu_profiler->get_breakdown();
    }
    d["chunk_rids_alive"] = DebugLog::chunk_rids_created.load() - DebugLog::chunk_rids_freed.load();
    d["discoverer"] = chunk_discoverer->get_stats();
    d["mesher"] = direct_mesher ? direct_mesher->get_stats() : gpu_mesher->get_stats();

    Dictionary edits;
    edits["edits"] = edit_stats.edits;
    edits["remesh_total"] = edit_stats.remesh_total;
    edits["remesh_last"] = edit_stats.remesh_last;
    edits["apply"] = edit_stats.apply.to_dict();
    d["edits"] = edits;

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
    sdf->stop_field_gpu(); // its thread frees its device; a walk still computing falls back to the CPU's field

    // Release loaded chunks first: GPU Direct chunks hand their range back to the mesher.
    for (auto &kv : visual_mesh_data.loaded_chunk) {
        kv.second.free();
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
    visual_mesh_data.clear();
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
