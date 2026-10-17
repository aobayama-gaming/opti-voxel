extends Node3D

var speed: float = 50.0
var mouse_sensitivity: float = 0.002

var pitch: float = 0.0
var yaw: float = 0.0

@onready var camera = $Camera3D

# Hole marks: H saves the camera pose, a screenshot, the terrain's chunk counts and a discovery audit
# (are culled chunks missing?) to marks.jsonl in the marks dir; closing the window saves a last pose.
# Dir: `-- --marks=DIR` on the command line, default user://marks.
var marks_dir := "user://marks"
var mark_index := 0
var pending_audit := {}  # the mark waiting for its discovery audit report

# `-- --lods=N` sets the sibling VoxelTerrain's lod_numbers (default 15): it must be set before the
# terrain's _ready, and every node enters the tree before any is ready.
func _enter_tree() -> void:
	var terrain := get_parent().get_node_or_null("VoxelTerrain")
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--lods=") and terrain:
			terrain.lod_numbers = a.trim_prefix("--lods=").to_int()

func _ready():
	# Far plane: the scene's at the default LOD count, the discoverer's load range with more LODs.
	var terrain := get_parent().get_node_or_null("VoxelTerrain")
	if terrain and terrain.lod_numbers > 15:
		camera.far = maxf(camera.far, terrain.get_load_radius())
		# Godot's float projection loses the far plane past far/near ~2^24 (the light culler then
		# ERRORs every frame): near rises with far (~1.7 km at 25 LODs).
		camera.near = maxf(camera.near, camera.far / 2097152.0)
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	yaw = rotation.y
	pitch = camera.rotation.x
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--marks="):
			marks_dir = a.trim_prefix("--marks=")
	DirAccess.make_dir_recursive_absolute(marks_dir)
	get_tree().auto_accept_quit = false

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		_mark("close", false)
		get_tree().quit()

func _terrain() -> Node:
	return get_tree().current_scene.find_child("VoxelTerrain", true, false)

func _mark(kind: String, audit: bool) -> void:
	mark_index += 1
	var t: Transform3D = camera.global_transform
	var m := {
		"index": mark_index, "kind": kind, "time": Time.get_datetime_string_from_system(),
		"frame": Engine.get_process_frames(),
		"cam_pos": [t.origin.x, t.origin.y, t.origin.z],
		"forward": [-t.basis.z.x, -t.basis.z.y, -t.basis.z.z],
		"yaw_deg": rad_to_deg(yaw), "pitch_deg": rad_to_deg(pitch), "fov": camera.fov,
		"window": [get_viewport().get_visible_rect().size.x, get_viewport().get_visible_rect().size.y],
	}
	var shot := "%s/mark_%03d.png" % [marks_dir, mark_index]
	get_viewport().get_texture().get_image().save_png(shot)
	m["screenshot"] = ProjectSettings.globalize_path(shot)
	var terrain := _terrain()
	if terrain:
		var stats: Dictionary = terrain.get_debug_stats()
		m["chunks"] = stats.get("chunks", {})
		m["load_complete"] = terrain.is_load_complete()
		m["lod_pixel_error"] = terrain.lod_pixel_error
		m["voxel_size"] = terrain.voxel_size
		if audit and pending_audit.is_empty() and terrain.debug_audit_discovery():
			pending_audit = m
			print("MARK %d: discovery audit running" % mark_index)
			return
	_write_mark(m)

func _write_mark(m: Dictionary) -> void:
	var path := marks_dir + "/marks.jsonl"
	var f := FileAccess.open(path, FileAccess.READ_WRITE if FileAccess.file_exists(path) else FileAccess.WRITE)
	f.seek_end()
	f.store_line(JSON.stringify(m))
	f.close()
	print("MARK ", JSON.stringify(m))

func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		yaw -= event.relative.x * mouse_sensitivity
		pitch -= event.relative.y * mouse_sensitivity
		pitch = clamp(pitch, -PI/2, PI/2)
		
		rotation.y = yaw
		camera.rotation.x = pitch
		camera.rotation.y = 0
		camera.rotation.z = 0
		
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			speed *= 1.2
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			speed /= 1.2

	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_H:
		_mark("hole", true)

	if event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		else:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _process(delta: float) -> void:
	if not pending_audit.is_empty():
		var report: Dictionary = _terrain().get_discovery_audit_report()
		if not report.is_empty():
			pending_audit["discovery_audit"] = report
			_write_mark(pending_audit)
			pending_audit = {}

	var input_dir = Vector3.ZERO
	if Input.is_physical_key_pressed(KEY_W) or Input.is_action_pressed("ui_up"):
		input_dir.z -= 1.0
	if Input.is_physical_key_pressed(KEY_S) or Input.is_action_pressed("ui_down"):
		input_dir.z += 1.0
	if Input.is_physical_key_pressed(KEY_A) or Input.is_action_pressed("ui_left"):
		input_dir.x -= 1.0
	if Input.is_physical_key_pressed(KEY_D) or Input.is_action_pressed("ui_right"):
		input_dir.x += 1.0
	if Input.is_physical_key_pressed(KEY_SPACE):
		input_dir.y += 1.0
	if Input.is_physical_key_pressed(KEY_SHIFT):
		input_dir.y -= 1.0

	var direction = (transform.basis * input_dir).normalized()
	position += direction * speed * delta
