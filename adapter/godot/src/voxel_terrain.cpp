#include "voxel_terrain.hpp"

#include "godot_conv.hpp"
#include "ov/executor.hpp"
#include "rd_utils.hpp"
#include "trace.hpp"
#include "voxel_sky.hpp"
#include <cmath>
#include <fstream>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <map>

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
    ClassDB::bind_method(D_METHOD("set_lod_pixel_errors", "errors"), &VoxelTerrain::set_lod_pixel_errors);
    ClassDB::bind_method(D_METHOD("get_lod_pixel_errors"), &VoxelTerrain::get_lod_pixel_errors);
    ClassDB::bind_method(D_METHOD("set_lod_view_margin_deg", "degrees"), &VoxelTerrain::set_lod_view_margin_deg);
    ClassDB::bind_method(D_METHOD("get_lod_view_margin_deg"), &VoxelTerrain::get_lod_view_margin_deg);
    ClassDB::bind_method(D_METHOD("set_lod_view_memory_m", "metres"), &VoxelTerrain::set_lod_view_memory_m);
    ClassDB::bind_method(D_METHOD("get_lod_view_memory_m"), &VoxelTerrain::get_lod_view_memory_m);
    ClassDB::bind_method(D_METHOD("set_lod_far_limit", "enabled"), &VoxelTerrain::set_lod_far_limit);
    ClassDB::bind_method(D_METHOD("get_lod_far_limit"), &VoxelTerrain::get_lod_far_limit);
    ClassDB::bind_method(D_METHOD("set_lod_numbers", "count"), &VoxelTerrain::set_lod_numbers);
    ClassDB::bind_method(D_METHOD("get_lod_numbers"), &VoxelTerrain::get_lod_numbers);
    ClassDB::bind_method(D_METHOD("get_load_radius"), &VoxelTerrain::get_load_radius);
    ClassDB::bind_method(D_METHOD("get_half_chunk_voxels"), &VoxelTerrain::get_half_chunk_voxels);
    ClassDB::bind_method(D_METHOD("get_sdf_at", "render_pos"), &VoxelTerrain::get_sdf_at);
    ClassDB::bind_method(D_METHOD("get_tangent_position", "tangent_m"), &VoxelTerrain::get_tangent_position);
    ClassDB::bind_method(D_METHOD("set_rebase_distance_m", "metres"), &VoxelTerrain::set_rebase_distance_m);
    ClassDB::bind_method(D_METHOD("get_rebase_distance_m"), &VoxelTerrain::get_rebase_distance_m);
    ClassDB::bind_method(D_METHOD("get_render_origin_x"), &VoxelTerrain::get_render_origin_x);
    ClassDB::bind_method(D_METHOD("get_render_origin_y"), &VoxelTerrain::get_render_origin_y);
    ClassDB::bind_method(D_METHOD("get_render_origin_z"), &VoxelTerrain::get_render_origin_z);
    ClassDB::bind_method(D_METHOD("get_render_offset_m"), &VoxelTerrain::get_render_offset_m);
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
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "lod_pixel_errors"), "set_lod_pixel_errors", "get_lod_pixel_errors");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_view_margin_deg", PROPERTY_HINT_RANGE, vformat("%d,180,1,suffix:deg", int(ChunkDiscoverer::RESCAN_TURN_DEG))), "set_lod_view_margin_deg", "get_lod_view_margin_deg");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_view_memory_m", PROPERTY_HINT_RANGE, "-1,100000,1,or_greater,suffix:m"), "set_lod_view_memory_m", "get_lod_view_memory_m");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lod_far_limit"), "set_lod_far_limit", "get_lod_far_limit");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_numbers", PROPERTY_HINT_RANGE, "1,31"), "set_lod_numbers", "get_lod_numbers");
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
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "discovery_verify"), "set_discovery_verify", "get_discovery_verify");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "destroy_verify"), "set_destroy_verify", "get_destroy_verify");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "discovery_compare"), "set_discovery_compare", "get_discovery_compare");

    BIND_ENUM_CONSTANT(PIPELINE_GPU_LEGACY);
    BIND_ENUM_CONSTANT(PIPELINE_GPU_DIRECT);
}

void VoxelTerrain::set_pipeline_mode(int p_mode) {
    ERR_FAIL_COND_MSG(world.is_started(), "pipeline_mode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_mode != PIPELINE_GPU_LEGACY && p_mode != PIPELINE_GPU_DIRECT,
            "pipeline_mode must be 1 (GPU Legacy) or 2 (GPU Direct).");
    cfg().pipeline_mode = ov::World::PipelineMode(p_mode);
}

int VoxelTerrain::get_pipeline_mode() const {
    return cfg().pipeline_mode;
}

void VoxelTerrain::set_terrain_mode(int p_mode) {
    ERR_FAIL_COND_MSG(world.is_started(), "terrain_mode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_mode != 1 && p_mode != 2 && p_mode != 4, "terrain_mode must be 1 (cascade), 2 (continents preset) or 4 (shapes gallery).");
    cfg().terrain_mode = p_mode;
}

int VoxelTerrain::get_terrain_mode() const {
    return cfg().terrain_mode;
}

void VoxelTerrain::set_legacy_batch_size(int p_size) {
    ERR_FAIL_COND_MSG(world.is_started(), "legacy_batch_size must be set before the terrain enters the tree.");
    cfg().legacy_batch_size = p_size;
}

int VoxelTerrain::get_legacy_batch_size() const {
    return cfg().legacy_batch_size;
}

void VoxelTerrain::set_voxel_size(float p_size) {
    ERR_FAIL_COND_MSG(world.is_started(), "voxel_size must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_size <= 0.0f, "voxel_size must be positive.");
    cfg().chunk_param.voxel_size = p_size;
}

float VoxelTerrain::get_voxel_size() const {
    return cfg().chunk_param.voxel_size;
}

void VoxelTerrain::set_discovery_cull_mode(int p_mode) {
    ERR_FAIL_COND_MSG(world.is_started(), "discovery_cull_mode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_mode < ChunkDiscoverer::CULL_NONE || p_mode > ChunkDiscoverer::CULL_LEGACY, "Unknown discovery_cull_mode.");
    cfg().discovery_cull_mode = p_mode;
}

int VoxelTerrain::get_discovery_cull_mode() const {
    return cfg().discovery_cull_mode;
}

void VoxelTerrain::set_discovery_cache(bool p_enable) {
    ERR_FAIL_COND_MSG(world.is_started(), "discovery_cache must be set before the terrain enters the tree.");
    cfg().discovery_cache = p_enable;
}

bool VoxelTerrain::get_discovery_cache() const {
    return cfg().discovery_cache;
}

void VoxelTerrain::set_discovery_verify(bool p_enable) {
    ERR_FAIL_COND_MSG(world.is_started(), "discovery_verify must be set before the terrain enters the tree.");
    cfg().discovery_verify = p_enable;
}

bool VoxelTerrain::get_discovery_verify() const {
    return cfg().discovery_verify;
}

void VoxelTerrain::set_destroy_verify(bool p_enable) {
    ERR_FAIL_COND_MSG(world.is_started(), "destroy_verify must be set before the terrain enters the tree.");
    cfg().destroy_verify = p_enable;
}

bool VoxelTerrain::get_destroy_verify() const {
    return cfg().destroy_verify;
}

void VoxelTerrain::set_discoverer_mode(int p_mode) {
    ERR_FAIL_COND_MSG(world.is_started(), "discoverer_mode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_mode < -1 || p_mode > 1, "Unknown discoverer_mode (-1 auto, 0 CPU, 1 GPU).");
    cfg().discoverer_mode = p_mode;
}

int VoxelTerrain::get_discoverer_mode() const {
    return cfg().discoverer_mode;
}

void VoxelTerrain::set_walk_gpu(bool p_enable) {
    ERR_FAIL_COND_MSG(world.is_started(), "walk_gpu must be set before the terrain enters the tree.");
    cfg().walk_gpu = p_enable;
}

bool VoxelTerrain::get_walk_gpu() const {
    return cfg().walk_gpu;
}

void VoxelTerrain::set_walk_verify(bool p_enable) {
    ERR_FAIL_COND_MSG(world.is_started(), "walk_verify must be set before the terrain enters the tree.");
    cfg().walk_verify = p_enable;
}

bool VoxelTerrain::get_walk_verify() const {
    return cfg().walk_verify;
}

void VoxelTerrain::set_walk_batch(int p_keys) {
    ERR_FAIL_COND_MSG(world.is_started(), "walk_batch must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_keys < 1, "walk_batch must be at least 1.");
    cfg().walk_batch = p_keys;
}

int VoxelTerrain::get_walk_batch() const {
    return cfg().walk_batch;
}

void VoxelTerrain::set_discovery_compare(bool p_enable) {
    ERR_FAIL_COND_MSG(world.is_started(), "discovery_compare must be set before the terrain enters the tree.");
    cfg().discovery_compare = p_enable;
}

bool VoxelTerrain::get_discovery_compare() const {
    return cfg().discovery_compare;
}

void VoxelTerrain::set_skirt_depth(float p_depth) {
    ERR_FAIL_COND_MSG(world.is_started(), "skirt_depth must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_depth < 0.0f, "skirt_depth must not be negative.");
    cfg().chunk_param.skirt_depth = p_depth;
}

float VoxelTerrain::get_skirt_depth() const {
    return cfg().chunk_param.skirt_depth;
}

bool VoxelTerrain::_can_set_world_origin(int64_t p_voxel) const {
    ERR_FAIL_COND_V_MSG(world.is_started(), false, "world_origin must be set before the terrain enters the tree.");
    ERR_FAIL_COND_V_MSG(p_voxel > WorldCoord::MAX_ABS_ORIGIN_VOXEL || p_voxel < -WorldCoord::MAX_ABS_ORIGIN_VOXEL, false,
            "world_origin must be within +-2^60 voxels.");
    return true;
}

void VoxelTerrain::set_world_origin_x(int64_t p_voxel) {
    if (_can_set_world_origin(p_voxel)) {
        cfg().world_origin_request.x = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_x() const {
    return cfg().world_origin_request.x;
}

void VoxelTerrain::set_world_origin_y(int64_t p_voxel) {
    if (_can_set_world_origin(p_voxel)) {
        cfg().world_origin_request.y = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_y() const {
    return cfg().world_origin_request.y;
}

void VoxelTerrain::set_world_origin_z(int64_t p_voxel) {
    if (_can_set_world_origin(p_voxel)) {
        cfg().world_origin_request.z = p_voxel;
    }
}

int64_t VoxelTerrain::get_world_origin_z() const {
    return cfg().world_origin_request.z;
}

void VoxelTerrain::set_archetype_instances(int p_count) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_instances must be set before the terrain enters the tree.");
    cfg().archetype_instances = p_count;
}

int VoxelTerrain::get_archetype_instances() const {
    return cfg().archetype_instances;
}

void VoxelTerrain::set_archetype_instance_area(double p_area) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_instance_area must be set before the terrain enters the tree.");
    cfg().archetype_instance_area = p_area;
}

double VoxelTerrain::get_archetype_instance_area() const {
    return cfg().archetype_instance_area;
}

void VoxelTerrain::set_archetype_trees(int p_count) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_trees must be set before the terrain enters the tree.");
    cfg().archetype_trees = p_count;
}

int VoxelTerrain::get_archetype_trees() const {
    return cfg().archetype_trees;
}

void VoxelTerrain::set_archetype_tree_area(double p_area) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_tree_area must be set before the terrain enters the tree.");
    cfg().archetype_tree_area = p_area;
}

double VoxelTerrain::get_archetype_tree_area() const {
    return cfg().archetype_tree_area;
}

void VoxelTerrain::set_biome_radius(double p_metres) {
    ERR_FAIL_COND_MSG(world.is_started(), "biome_radius must be set before the terrain enters the tree.");
    cfg().biome_radius = p_metres;
}

double VoxelTerrain::get_biome_radius() const {
    return cfg().biome_radius;
}

void VoxelTerrain::set_climate_path(const String &p_path) {
    ERR_FAIL_COND_MSG(world.is_started(), "climate_path must be set before the terrain enters the tree.");
    cfg().climate_path = ov_str(p_path);
}

String VoxelTerrain::get_climate_path() const {
    return godot_str(cfg().climate_path);
}

void VoxelTerrain::set_archetype_band(bool p_enabled) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_band must be set before the terrain enters the tree.");
    cfg().archetype_band = p_enabled;
}

bool VoxelTerrain::get_archetype_band() const {
    return cfg().archetype_band;
}

void VoxelTerrain::set_archetype_stream(bool p_enabled) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_stream must be set before the terrain enters the tree.");
    cfg().archetype_stream = p_enabled;
}

bool VoxelTerrain::get_archetype_stream() const {
    return cfg().archetype_stream;
}

void VoxelTerrain::set_archetype_stream_part_sides(int p_sides) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_stream_part_sides must be set before the terrain enters the tree.");
    // Past it, far layers become part layers whose blob tiles read cubically many LOD-0 part tiles.
    ERR_FAIL_COND_MSG(p_sides < 1 || p_sides > int(Archetype::ScatterSource::STREAM_PART_CELL_SIDES_MAX),
            vformat("archetype_stream_part_sides must be within [1, %d] source cells.", Archetype::ScatterSource::STREAM_PART_CELL_SIDES_MAX));
    cfg().archetype_stream_part_sides = p_sides;
}

int VoxelTerrain::get_archetype_stream_part_sides() const {
    return cfg().archetype_stream_part_sides;
}

void VoxelTerrain::set_archetype_moments(bool p_enabled) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_moments must be set before the terrain enters the tree.");
    cfg().archetype_moments = p_enabled;
}

bool VoxelTerrain::get_archetype_moments() const {
    return cfg().archetype_moments;
}

void VoxelTerrain::set_archetype_erode(float p_cells) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_erode must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(!(p_cells > 0.0f && p_cells <= 4.0f), "archetype_erode must be in (0, 4] cells per LOD.");
    cfg().archetype_erode = p_cells;
}

float VoxelTerrain::get_archetype_erode() const {
    return cfg().archetype_erode;
}

void VoxelTerrain::set_archetype_resolve(float p_cells) {
    ERR_FAIL_COND_MSG(world.is_started(), "archetype_resolve must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(!(p_cells >= 0.0f && p_cells <= 8.0f), "archetype_resolve must be in [0, 8] cells.");
    cfg().archetype_resolve = p_cells;
}

float VoxelTerrain::get_archetype_resolve() const {
    return cfg().archetype_resolve;
}

void VoxelTerrain::set_lod_pixel_error(float p_error) {
    ERR_FAIL_COND_MSG(world.is_started(), "lod_pixel_error must be set before the terrain enters the tree.");
    cfg().lod_pixel_error = p_error;
}

float VoxelTerrain::get_lod_pixel_error() const {
    return cfg().lod_pixel_error;
}

void VoxelTerrain::set_lod_pixel_error_back(float p_error) {
    ERR_FAIL_COND_MSG(world.is_started(), "lod_pixel_error_back must be set before the terrain enters the tree.");
    cfg().lod_pixel_error_back = p_error;
}

float VoxelTerrain::get_lod_pixel_error_back() const {
    return cfg().lod_pixel_error_back;
}

void VoxelTerrain::set_lod_pixel_errors(const PackedFloat32Array &p_errors) {
    ERR_FAIL_COND_MSG(world.is_started(), "lod_pixel_errors must be set before the terrain enters the tree.");
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
    ERR_FAIL_COND_MSG(world.is_started(), "lod_view_margin_deg must be set before the terrain enters the tree.");
    // The remembered view axis lags the camera's by up to RESCAN_TURN_DEG: a smaller margin would leave the
    // frustum's edge outside the cone.
    ERR_FAIL_COND_MSG(!(p_degrees >= ChunkDiscoverer::RESCAN_TURN_DEG && p_degrees <= 180.0f),
            vformat("lod_view_margin_deg must be in [%d, 180].", int(ChunkDiscoverer::RESCAN_TURN_DEG)));
    cfg().lod_view_margin_deg = p_degrees;
}

float VoxelTerrain::get_lod_view_margin_deg() const {
    return cfg().lod_view_margin_deg;
}

void VoxelTerrain::set_lod_view_memory_m(float p_metres) {
    ERR_FAIL_COND_MSG(world.is_started(), "lod_view_memory_m must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(std::isnan(p_metres), "lod_view_memory_m must be a number.");
    cfg().lod_view_memory_m = p_metres;
}

float VoxelTerrain::get_lod_view_memory_m() const {
    return cfg().lod_view_memory_m;
}

void VoxelTerrain::set_lod_far_limit(bool p_enabled) {
    ERR_FAIL_COND_MSG(world.is_started(), "lod_far_limit must be set before the terrain enters the tree.");
    lod_far_limit = p_enabled;
}

bool VoxelTerrain::get_lod_far_limit() const {
    return lod_far_limit;
}

void VoxelTerrain::set_lod_numbers(int p_count) {
    ERR_FAIL_COND_MSG(world.is_started(), "lod_numbers must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(p_count < 1 || p_count > MAX_LOD_NUMBERS, vformat("lod_numbers must be in [1, %d].", MAX_LOD_NUMBERS));
    cfg().lod_numbers = p_count;
}

int VoxelTerrain::get_lod_numbers() const {
    return cfg().lod_numbers;
}

void VoxelTerrain::set_legacy_trim_indices(bool p_trim) {
    ERR_FAIL_COND_MSG(world.is_started(), "legacy_trim_indices must be set before the terrain enters the tree.");
    cfg().legacy_trim_indices = p_trim;
}

bool VoxelTerrain::get_legacy_trim_indices() const {
    return cfg().legacy_trim_indices;
}

void VoxelTerrain::set_dual_contouring(bool p_enable) {
    ERR_FAIL_COND_MSG(world.is_started(), "dual_contouring must be set before the terrain enters the tree.");
    cfg().dual_contouring = p_enable;
}

bool VoxelTerrain::get_dual_contouring() const {
    return cfg().dual_contouring;
}

void VoxelTerrain::set_direct_batch_size(int p_size) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_batch_size must be set before the terrain enters the tree.");
    cfg().direct.batch_size = p_size;
}

int VoxelTerrain::get_direct_batch_size() const {
    return cfg().direct.batch_size;
}

void VoxelTerrain::set_direct_pages_per_batch(int p_pages) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_pages_per_batch must be set before the terrain enters the tree.");
    cfg().direct.pages_per_batch = p_pages;
}

int VoxelTerrain::get_direct_pages_per_batch() const {
    return cfg().direct.pages_per_batch;
}

void VoxelTerrain::set_direct_batches_per_frame(int p_batches) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_batches_per_frame must be set before the terrain enters the tree.");
    ERR_FAIL_COND(p_batches < 1 || p_batches > (int)GPUDirectMesher::MAX_BATCHES_PER_FRAME);
    cfg().direct.max_batches_per_frame = p_batches;
}

int VoxelTerrain::get_direct_batches_per_frame() const {
    return cfg().direct.max_batches_per_frame;
}

void VoxelTerrain::set_direct_gpu_budget_ms(float p_ms) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_gpu_budget_ms must be set before the terrain enters the tree.");
    ERR_FAIL_COND(!(p_ms >= 0.0f)); // 0: no budget
    cfg().direct.gpu_budget_ms = p_ms;
}

float VoxelTerrain::get_direct_gpu_budget_ms() const {
    return cfg().direct.gpu_budget_ms;
}

void VoxelTerrain::set_direct_max_in_flight(int p_batches) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_max_in_flight must be set before the terrain enters the tree.");
    ERR_FAIL_COND(p_batches < 0); // 0: 3 * direct_batches_per_frame
    cfg().direct.max_in_flight = p_batches;
}

int VoxelTerrain::get_direct_max_in_flight() const {
    return cfg().direct.max_in_flight;
}

void VoxelTerrain::set_direct_page_capacity(int p_capacity) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_page_capacity must be set before the terrain enters the tree.");
    cfg().direct.page_capacity = p_capacity;
}

int VoxelTerrain::get_direct_page_capacity() const {
    return cfg().direct.page_capacity;
}

void VoxelTerrain::set_direct_initial_pages(int p_pages) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_initial_pages must be set before the terrain enters the tree.");
    cfg().direct.initial_pages = p_pages;
}

int VoxelTerrain::get_direct_initial_pages() const {
    return cfg().direct.initial_pages;
}

void VoxelTerrain::set_direct_compaction(bool p_enable) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_compaction must be set before the terrain enters the tree.");
    cfg().direct.compaction = p_enable;
}

bool VoxelTerrain::get_direct_compaction() const {
    return cfg().direct.compaction;
}

void VoxelTerrain::set_direct_compact_max_extent(float p_chunks) {
    ERR_FAIL_COND_MSG(world.is_started(), "direct_compact_max_extent must be set before the terrain enters the tree.");
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
    return world.sdf_at(pos);
}

Vector3 VoxelTerrain::get_tangent_position(const Vector2 &p_tangent_m) const {
    double pos[3];
    world.tangent_position(double(p_tangent_m.x), double(p_tangent_m.y), pos);
    return Vector3(real_t(pos[0]), real_t(pos[1]), real_t(pos[2]));
}

void VoxelTerrain::set_rebase_distance_m(float p_metres) {
    ERR_FAIL_COND_MSG(world.is_started(), "rebase_distance_m must be set before the terrain enters the tree.");
    ERR_FAIL_COND_MSG(!(p_metres >= 0.0f), "rebase_distance_m must be >= 0.");
    cfg().rebase_distance_m = p_metres;
}

float VoxelTerrain::get_rebase_distance_m() const {
    return cfg().rebase_distance_m;
}

int64_t VoxelTerrain::get_render_origin_x() const {
    return cfg().chunk_param.origin_voxel().x;
}

int64_t VoxelTerrain::get_render_origin_y() const {
    return cfg().chunk_param.origin_voxel().y;
}

int64_t VoxelTerrain::get_render_origin_z() const {
    return cfg().chunk_param.origin_voxel().z;
}

Vector3 VoxelTerrain::get_render_offset_m() const {
    if (!world.is_started()) {
        return Vector3();
    }
    return godot_v3(ChunkMath::origin_shift_m(world.world_origin(), world.render_origin(), world.chunk_param()));
}

void VoxelTerrain::_move_render_origin(const Float3 &p_shift) {
    const Vector3 shift = godot_v3(p_shift);
    Node *parent = get_parent();
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        Node3D *node = Object::cast_to<Node3D>(parent->get_child(i));
        if (node != nullptr && node != this) {
            node->set_global_position(node->get_global_position() - shift);
        }
    }
    _push_render_frame();
    emit_signal("origin_shifted", shift);
}

void VoxelTerrain::_push_render_frame() {
    const ChunkParam &param = world.chunk_param();
    const SDFCascade &sdf = world.terrain();
    if (page_host && sdf.preset == SDFCascade::Preset::CONTINENTS) {
        // Continents preset: the page shader colours heights around sea level (the cascade's datum), in render metres.
        page_host->set_material_param("sea_level_m", ChunkMath::voxel_to_render(VoxelPos(0, sdf.get_datum_y(), 0), param).y);
    }
    const float rise_m = ChunkMath::origin_shift_m(world.world_origin(), world.render_origin(), param).y;
    const Vector3i origin_lo = godot_v3i(WorldCoord::lo32_i(param.origin_voxel()));
    Node *parent = get_parent();
    for (int i = 0; parent && i < parent->get_child_count(); i++) {
        if (VoxelSky *sky = Object::cast_to<VoxelSky>(parent->get_child(i))) {
            sky->set_render_frame(origin_lo, param.voxel_size, rise_m);
        }
    }
}

double VoxelTerrain::get_load_radius() const {
    return world.load_radius();
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
    return godot_dict(world.process_times_ms());
}

int VoxelTerrain::get_pending_chunk_count() const {
    return int(world.pending_chunk_count());
}

void VoxelTerrain::set_debug_view(int p_view) {
    world.set_debug_view(p_view);
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

    main_device = new RDDevice(rd);
    ov::World::Hosts hosts;
    hosts.device = main_device;
    if (cfg().pipeline_mode == ov::World::PIPELINE_GPU_DIRECT) {
        page_host = new GodotPageHost(get_world_3d()->get_scenario(), main_device);
        hosts.pages = page_host;
    } else {
        mesh_host = new GodotMeshHost(get_world_3d()->get_scenario());
        hosts.meshes = mesh_host;
    }
    hosts.hidden_meshes = []() -> std::unique_ptr<MeshHost> { return std::make_unique<GodotMeshHost>(); };
    const Float3 camera_pos = camera != nullptr ? ov_f3(camera->get_global_position()) : Float3();
    const Float3 camera_forward = camera != nullptr ? ov_f3(_camera_forward()) : Float3();
    const double camera_far = _camera_far();
    if (!world.start(projection, hosts, camera != nullptr ? &camera_pos : nullptr, camera != nullptr ? &camera_forward : nullptr, camera_far)) {
        return;
    }
    if (page_host) {
        // Skirt depth per LOD: the page shader's height colours undo the skirt push.
        const SDFCascade &sdf = world.terrain();
        PackedFloat32Array skirt_depths;
        skirt_depths.resize(32);
        for (int l = 0; l < 32; l++) {
            skirt_depths.set(l, l < cfg().lod_numbers ? sdf.skirt_depth_m(l) : 0.0f);
        }
        page_host->set_material_param("skirt_depth_m", skirt_depths);
        const UpAxis up = sdf.up_at(VoxelPos(0, sdf.get_datum_y(), 0));
        page_host->set_material_param("base_up", Vector3(real_t(up[0]), real_t(up[1]), real_t(up[2])));
        set_page_params(page_params); // set before the pool existed (VoxelSky's clouds)
    }
    _push_render_frame();
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
    return world.add_instance(p_type, VoxelPos(p_x, p_y, p_z), edit_floats(p_rotation, 9, rot, false), edit_floats(p_params, 8, params, true), p_op, p_k);
}

bool VoxelTerrain::remove_instance(int p_id) {
    return world.remove_instance(p_id);
}

bool VoxelTerrain::move_instance(int p_id, int64_t p_x, int64_t p_y, int64_t p_z, const PackedFloat32Array &p_rotation) {
    float rot[9];
    return world.move_instance(p_id, VoxelPos(p_x, p_y, p_z), edit_floats(p_rotation, 9, rot, false));
}

bool VoxelTerrain::dig_hole(const Vector3 &p_origin, const Vector3 &p_dir) {
    return dig_tool.dig(ov_f3(p_origin), ov_f3(p_dir));
}

void VoxelTerrain::dig_spray(const Vector3 &p_origin, const Vector3 &p_dir, double p_delta) {
    dig_tool.spray(ov_f3(p_origin), ov_f3(p_dir), p_delta);
}

void VoxelTerrain::dig_tube(const Vector3 &p_origin, const Vector3 &p_dir) {
    dig_tool.tube(ov_f3(p_origin), ov_f3(p_dir));
}

void VoxelTerrain::dig_release() {
    dig_tool.release();
}

int VoxelTerrain::get_dig_edits() const {
    return dig_tool.edit_count();
}

void VoxelTerrain::set_dig_settings(const Dictionary &p_settings) {
    ov::DigTool::Settings &s = dig_tool.settings;
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
    if (!world.is_started()) {
        return;
    }
    if (camera != nullptr) {
        Float3 shift;
        if (world.recenter(ov_f3(camera->get_global_position()), shift)) {
            _move_render_origin(shift);
        }
        const Float3 pos = ov_f3(camera->get_global_position());
        const Float3 forward = ov_f3(_camera_forward());
        world.frame(&pos, &forward, _camera_far());
    } else {
        world.frame(nullptr, nullptr);
    }
    // Tracing reads every captured frame (the GPU track).
    if ((debug_gpu_profile && world.frames() % 15 == 0) || debug_gpu_profile_frames || Trace::enabled()) {
        if (!gpu_profiler) {
            gpu_profiler = memnew(GPUProfiler);
            gpu_profiler->set_recording(debug_gpu_profile_frames);
        }
        gpu_profiler->poll();
    }
}

bool VoxelTerrain::debug_validate(int p_count, const String &p_dump_dir) {
    return world.debug_validate(p_count, ov_str(p_dump_dir));
}

Dictionary VoxelTerrain::get_validation_report() const {
    return godot_dict(world.validation_report());
}

bool VoxelTerrain::debug_audit_pages() {
    return world.debug_audit_pages();
}

Dictionary VoxelTerrain::get_audit_report() const {
    return godot_dict(world.audit_report());
}

bool VoxelTerrain::debug_audit_discovery() {
    return world.debug_audit_discovery();
}

Dictionary VoxelTerrain::get_discovery_audit_report() const {
    return godot_dict(world.discovery_audit_report());
}

bool VoxelTerrain::is_load_complete() {
    if (camera == nullptr) {
        return world.is_load_complete(nullptr);
    }
    const Float3 pos = ov_f3(camera->get_global_position());
    return world.is_load_complete(&pos);
}

bool VoxelTerrain::is_edit_settled() {
    return world.is_edit_settled();
}

Dictionary VoxelTerrain::get_debug_stats() {
    Dictionary d = godot_dict(world.debug_stats());
    if (gpu_profiler && world.is_started()) {
        d["gpu_breakdown"] = gpu_profiler->get_breakdown();
    }
    return d;
}

void VoxelTerrain::_teardown() {
    if (gpu_profiler) {
        RDUtils::delete_on_render_thread(gpu_profiler);
        gpu_profiler = nullptr;
    }
    world.stop();
    if (page_host) {
        page_host->destroy(); // after the mesher's teardown, on the render thread too
        page_host = nullptr;
    }
    delete mesh_host;
    mesh_host = nullptr;
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
