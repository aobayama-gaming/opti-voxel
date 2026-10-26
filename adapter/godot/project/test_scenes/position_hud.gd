extends CanvasLayer
## Corner readout for the integer-frame test scenes: where the camera is, in Godot space (metres
## from the world origin) and in the integer frame (absolute LOD0 voxel), and how far it is from the
## origin. Past the Morton locality range (2^20 chunk-id units from the origin) batches and pages lose
## spatial locality; render floats also degrade with distance (no origin rebasing yet).

@export var terrain_path: NodePath

var terrain: Node
var origin_voxel := []  # [x, y, z] int, the snapped origin, read once the terrain has started

@onready var label: Label = $Label


func _ready() -> void:
	terrain = get_node_or_null(terrain_path)


func _process(_delta: float) -> void:
	var camera := get_viewport().get_camera_3d()
	if terrain == null or camera == null:
		return
	if origin_voxel.is_empty():
		var stats: Dictionary = terrain.get_debug_stats()
		if not stats.has("world_origin"):
			return
		# "(x, y, z)" string: int64 values do not survive a float round trip.
		var parts: PackedStringArray = String(stats["world_origin"]["voxel"]).trim_prefix("(").trim_suffix(")").split(",")
		origin_voxel = [parts[0].strip_edges().to_int(), parts[1].strip_edges().to_int(), parts[2].strip_edges().to_int()]

	var p: Vector3 = camera.global_position
	var voxel_size: float = terrain.voxel_size
	var voxel := [
		origin_voxel[0] + int(floor(p.x / voxel_size)),
		origin_voxel[1] + int(floor(p.y / voxel_size)),
		origin_voxel[2] + int(floor(p.z / voxel_size)),
	]
	# One chunk-id unit is half a LOD0 chunk: 14 voxels with the default chunk_size 32, skirt 2.
	var morton_range_km := pow(2.0, 20) * 14.0 * voxel_size / 1000.0
	var distance_km := p.length() / 1000.0
	var text := "position (m, from world origin)\n  X %.1f   Y %.1f   Z %.1f\n" % [p.x, p.y, p.z]
	text += "absolute voxel (%.2f m)\n  X %d\n  Y %d\n  Z %d\n" % [voxel_size, voxel[0], voxel[1], voxel[2]]
	text += "distance from origin %.2f km (Morton range %.0f km)" % [distance_km, morton_range_km]
	if distance_km > morton_range_km:
		text += "\n  OUTSIDE the Morton locality range"
	label.text = text
