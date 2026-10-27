extends CanvasLayer
## Corner readout for the integer-frame test scenes: where the camera is, in Godot space (metres
## from the render origin, which follows the camera: VoxelTerrain.rebase_distance_m) and in the integer
## frame (absolute LOD0 voxel), and how far it is from the world origin. Past the Morton locality range
## (2^20 chunk-id units from the render origin) batches and pages lose spatial locality.

@export var terrain_path: NodePath

var terrain: Node

@onready var label: Label = $Label


func _ready() -> void:
	terrain = get_node_or_null(terrain_path)


func _process(_delta: float) -> void:
	var camera := get_viewport().get_camera_3d()
	if terrain == null or camera == null:
		return
	var p: Vector3 = camera.global_position
	var voxel_size: float = terrain.voxel_size
	var voxel := [
		terrain.get_render_origin_x() + int(floor(p.x / voxel_size)),
		terrain.get_render_origin_y() + int(floor(p.y / voxel_size)),
		terrain.get_render_origin_z() + int(floor(p.z / voxel_size)),
	]
	# One chunk-id unit is half a LOD0 chunk: 14 voxels with the default chunk_size 32, skirt 2.
	var morton_range_km := pow(2.0, 20) * 14.0 * voxel_size / 1000.0
	var distance_km := (p + terrain.get_render_offset_m()).length() / 1000.0
	var text := "position (m, from render origin)\n  X %.1f   Y %.1f   Z %.1f\n" % [p.x, p.y, p.z]
	text += "absolute voxel (%.2f m)\n  X %d\n  Y %d\n  Z %d\n" % [voxel_size, voxel[0], voxel[1], voxel[2]]
	text += "distance from world origin %.2f km (Morton range %.0f km around the render origin)" % [distance_km, morton_range_km]
	label.text = text
