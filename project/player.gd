extends Node3D

var speed: float = 50.0
var mouse_sensitivity: float = 0.002

var pitch: float = 0.0
var yaw: float = 0.0

@onready var camera: Camera3D = $Camera3D

# Hole marks: H saves the camera pose, a screenshot, the terrain's chunk counts and a discovery audit
# (are culled chunks missing?) to marks.jsonl in the marks dir; closing the window saves a last pose.
# Dir: `-- --marks=DIR` on the command line, default user://marks.
var marks_dir := "user://marks"
var mark_index := 0
var pending_audit := {}  # the mark waiting for its discovery audit report
var far_lods_terrain: Node = null  # set above 15 LODs: clip planes follow the altitude
var base_near := 0.05

# `-- --lods=N` sets the sibling VoxelTerrain's lod_numbers (default 15), `-- --terrain=cascade|continents|archetypes`
# its terrain_mode (the Hermite cascade test terrain, its continents demo preset, or the archetype IR demo) and
# `-- --discoverer=cpu|gpu` its discoverer_mode: they must be set before the terrain's _ready, and every
# node enters the tree before any is ready.
var spawn_above_ground := false
var spawn_altitude := 200.0  # --altitude=M: start M metres above the ground (cascade terrains)
var cascade_terrain: Node = null  # set with --terrain=cascade
var cached_altitude := 0.0
var altitude_age := 1.0

# ---- Archetype dig tools (feature/archetype-dig) -----------------------------------------------
# Right click digs a hole: a SMOOTH_SUBTRACT capsule along the view, centred on the ray-march hit.
# Left click (held): one tube re-placed every frame in front of the camera (move_instance), removed on
# release. Both use VoxelTerrain.add_instance / move_instance / remove_instance (archetype table only).
var dig_radius := 1.5       # --dig_radius=M
var dig_length := 6.0       # --dig_length=M
var drag_length := 20.0     # --drag_length=M: how far the tube carves INWARD from the surface
var drag_radius := 3.0      # --drag_radius=M: the tube's radius
var dig_k := 0.3            # smooth radius (metres)
var drag_op := 2            # Combine: 2 subtract (a hole that follows you), 0 union
var drag_active := false
var drag_id := -1
var edits_count := 0
var hud: Label = null
var hud_layer: CanvasLayer = null
var hud_visible := true

func _enter_tree() -> void:
	var terrain := get_parent().get_node_or_null("VoxelTerrain")
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--lods=") and terrain:
			terrain.lod_numbers = a.trim_prefix("--lods=").to_int()
		if a == "--terrain=cascade" and terrain:
			terrain.terrain_mode = 1
			spawn_above_ground = true
		if a.begins_with("--discoverer=") and terrain:
			var modes := {"cpu": 0, "gpu": 1}
			var mode := a.trim_prefix("--discoverer=")
			if modes.has(mode):
				terrain.discoverer_mode = modes[mode]
			else:
				push_error("player: unknown --discoverer=%s (expected cpu or gpu)" % mode)
		if a == "--terrain=continents" and terrain:
			terrain.terrain_mode = 2
			spawn_above_ground = true
		if a == "--terrain=archetypes" and terrain:
			terrain.terrain_mode = 3
		if a.begins_with("--altitude="):
			spawn_altitude = a.trim_prefix("--altitude=").to_float()
		if a.begins_with("--voxel=") and terrain:
			terrain.voxel_size = a.trim_prefix("--voxel=").to_float()
		if a.begins_with("--lod_error=") and terrain:
			terrain.lod_pixel_error = a.trim_prefix("--lod_error=").to_float()
		if a.begins_with("--view=") and terrain:
			terrain.debug_view = a.trim_prefix("--view=").to_int()
		if a.begins_with("--dig_radius="):
			dig_radius = a.trim_prefix("--dig_radius=").to_float()
		if a.begins_with("--dig_length="):
			dig_length = a.trim_prefix("--dig_length=").to_float()
		if a.begins_with("--drag_length="):
			drag_length = a.trim_prefix("--drag_length=").to_float()
		if a == "--drag_op=subtract":
			drag_op = 2
		if a == "--drag_op=union":
			drag_op = 0
		if a.begins_with("--drag_radius="):
			drag_radius = a.trim_prefix("--drag_radius=").to_float()

func _ready():
	# Clip planes: the scene's at the default LOD count; above 15 LODs they follow the altitude
	# (_process): near = altitude / 1000, far = near * 2^21 capped at the discoverer's load range.
	# Godot's float projection loses its far plane past far/near ~2^24 (the light culler ERRORs).
	var terrain := get_parent().get_node_or_null("VoxelTerrain")
	if terrain and terrain.lod_numbers > 15:
		far_lods_terrain = terrain
		base_near = camera.near
	if spawn_above_ground and terrain:
		cascade_terrain = terrain
		# The cascade terrain's relief is kilometres high: start spawn_altitude above the ground under the spawn.
		for i in 32:
			var d: float = terrain.get_sdf_at(global_position)
			if absf(d - spawn_altitude) < spawn_altitude * 0.25:
				break
			global_position.y += spawn_altitude - d
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	hud_layer = CanvasLayer.new()
	add_child(hud_layer)
	hud = Label.new()
	hud.position = Vector2(10, 10)
	hud.add_theme_color_override("font_color", Color(1, 1, 1))
	hud.add_theme_color_override("font_shadow_color", Color(0, 0, 0))
	hud.add_theme_constant_override("shadow_offset_x", 1)
	hud.add_theme_constant_override("shadow_offset_y", 1)
	hud_layer.add_child(hud)
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

	# Dig tools (archetype table): right click digs a hole, left click (held) drags a tube.
	if event is InputEventMouseButton:
		if event.pressed and event.button_index == MOUSE_BUTTON_RIGHT:
			_dig()
		elif event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				drag_active = true
			else:
				_end_drag()

	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_TAB:
		hud_visible = not hud_visible

	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_H:
		_mark("hole", true)

	# V cycles the debug views (VoxelTerrain.debug_view): shaded, normals, LOD, page, wireframe, chunk, group, quad.
	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_V:
		var terrain := _terrain()
		if terrain:
			terrain.debug_view = (terrain.debug_view + 1) % 8
			print("debug view ", terrain.debug_view)

	if event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		else:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

# The dig tools: a ray-march along the view (get_sdf_at) and the archetype edit API.
func _raymarch(terrain: Node, origin: Vector3, dir: Vector3) -> Vector3:
	var t := 0.0
	while t < 200.0:
		var p := origin + dir * t
		var d: float = terrain.get_sdf_at(p)
		if d < 0.5:
			return p
		t += maxf(absf(d) * 0.5, 0.1)
	return origin + dir * 10.0

# LOD0 voxel of a render-space position (int64; returned as an Array, never through Vector3i).
func _voxel_of(terrain: Node, pos: Vector3) -> Array:
	var vt: float = terrain.voxel_size
	return [int(round(float(terrain.world_origin_x) + pos.x / vt)),
			int(round(float(terrain.world_origin_y) + pos.y / vt)),
			int(round(float(terrain.world_origin_z) + pos.z / vt))]

# World -> local basis (row major) with local y along fwd: the capsule's axis.
func _basis9(fwd: Vector3) -> PackedFloat32Array:
	var up := Vector3.UP
	if absf(fwd.dot(up)) > 0.99:
		up = Vector3.RIGHT
	var right := fwd.cross(up).normalized()
	var u2 := right.cross(fwd).normalized()
	return PackedFloat32Array([right.x, right.y, right.z, fwd.x, fwd.y, fwd.z, u2.x, u2.y, u2.z])

func _dig() -> void:
	var terrain := _terrain()
	if terrain == null:
		return
	var vt: float = terrain.voxel_size
	var fwd := -camera.global_transform.basis.z
	var hit := _raymarch(terrain, camera.global_position, fwd)
	var p := _voxel_of(terrain, hit)
	var params := PackedFloat32Array([dig_length / vt, dig_radius / vt])
	var id: int = terrain.add_instance(1, p[0], p[1], p[2], _basis9(fwd), params, 3, dig_k / vt)
	if id >= 0:
		edits_count += 1

func _update_drag() -> void:
	if not drag_active:
		return
	var terrain := _terrain()
	if terrain == null:
		return
	var vt: float = terrain.voxel_size
	var fwd := -camera.global_transform.basis.z
	# The tube starts at the surface the view hits and carves drag_length metres INWARD along the view, so
	# it goes deep into the mountain instead of eating only the sliver it overlaps. The capsule's local y
	# runs along the view; its origin is 1 m before the surface (so the mouth opens) and it is
	# drag_length + 1 m long. If the view hits nothing (sky) or the camera is already inside solid, the
	# hit is the camera and it carves from there. SUBTRACT by default; --drag_op=union makes it solid.
	var hit := _raymarch(terrain, camera.global_position, fwd)
	var p := _voxel_of(terrain, hit - fwd * 1.0)
	var rot := _basis9(fwd)
	var params := PackedFloat32Array([(drag_length + 1.0) / vt, drag_radius / vt])
	if drag_id < 0:
		drag_id = terrain.add_instance(1, p[0], p[1], p[2], rot, params, drag_op, dig_k / vt)
		if drag_id < 0:
			push_error("dig: terrain.add_instance returned -1 (is the terrain an archetype table?)")
			drag_active = false
			return
		edits_count += 1
	else:
		terrain.move_instance(drag_id, p[0], p[1], p[2], rot)
		edits_count += 1

func _end_drag() -> void:
	drag_active = false
	if drag_id >= 0:
		var terrain := _terrain()
		if terrain:
			terrain.remove_instance(drag_id)
		drag_id = -1

func _update_hud() -> void:
	if hud == null:
		return
	hud.visible = hud_visible
	if not hud_visible:
		return
	var terrain := _terrain()
	if terrain == null:
		return
	var s: Dictionary = terrain.get_debug_stats()
	var edits: Dictionary = s.get("edits", {})
	var ms: Dictionary = s.get("mesher", {})
	var apply_ms := 0.0
	if edits.get("apply") is Dictionary:
		apply_ms = float((edits["apply"] as Dictionary).get("avg", 0.0))
	hud.text = "fps %d   edits %d (+%d tool)\nre-mesh queue %d  dispatched %d  stale %d\nlast edit %d chunks, apply %.2f ms" % [
		Engine.get_frames_per_second(), int(edits.get("edits", 0)), edits_count,
		int(ms.get("remesh_queue", 0)), int(ms.get("remesh_pops", 0)), int(ms.get("remesh_stale_dropped", 0)),
		int(edits.get("remesh_last", 0)), apply_ms]

func _process(delta: float) -> void:
	_update_drag()
	_update_hud()
	if far_lods_terrain:
		var altitude: float = absf(far_lods_terrain.get_sdf_at(camera.global_position))
		camera.near = maxf(base_near, altitude / 1000.0)
		camera.far = minf(far_lods_terrain.get_load_radius(), camera.near * 2097152.0)
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
	var move_speed := speed
	if cascade_terrain:
		# Cascade terrain: kilometre-scale relief, fly faster when high (half the altitude per second).
		# The altitude is a CPU cascade point query: refreshed 4 times per second.
		altitude_age += delta
		if altitude_age >= 0.25:
			altitude_age = 0.0
			cached_altitude = absf(cascade_terrain.get_sdf_at(camera.global_position))
		move_speed = maxf(speed, cached_altitude * 0.5)
	position += direction * move_speed * delta
