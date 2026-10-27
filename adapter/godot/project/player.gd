extends Node3D

const Presets = preload("res://debug/presets.gd")
const VideoOverlay = preload("res://video_overlay.gd")

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
var clip_terrain: Node = null  # the clip planes follow the altitude (Presets.clip_planes)
var base_near := 0.05

# `-- --lods=N` sets the sibling VoxelTerrain's lod_numbers (default 15), `-- --terrain=cascade|continents|shapes`
# its terrain_mode (the Hermite cascade terrain, the default, its continents preset, or its flat preset under the
# hard-CSG shapes gallery) and
# `-- --discoverer=cpu|gpu` its discoverer_mode, `-- --walk_gpu=0|1` its walk_gpu: they must be set before the terrain's _ready, and every
# node enters the tree before any is ready.
var spawn_altitude := 200.0  # --altitude=M: start M metres above the ground (not in the shapes gallery)
var spawn_tangent = null  # --spawn=U,V: metres along the base's tangent plane (Vector2), null: the scene's spawn
var altitude_terrain: Node = null  # flight speed follows the altitude over it; not in the shapes gallery
var smoothed_altitude := 0.0  # 0: not read yet
const ALTITUDE_TAU := 0.2  # seconds: the altitude reading's time constant, in log space
const FLY_RATE := 1.0  # altitudes per second

# ---- Dig tools (VoxelTerrain.dig_*: the core's DigTool, on the instance table) -----------------------------------
# Right click digs a hole (a capsule from where the view meets the ground). Left click held sprays small spheres where the
# view meets the surface, or with --left_click=tube carries a tube with the camera until the release. They only act
# while the mouse is captured. --dig_radius= --dig_length= --spray_radius= --drag_radius= --drag_length= (metres) and
# --drag_op=subtract|union go to VoxelTerrain.set_dig_settings.
var dig_settings := {}
var left_held := false
var left_tube := false      # --left_click=tube: left click drags the tube instead of spraying
var hud: Label = null
var hud_visible := true     # TAB
var hud_age := 1.0          # seconds since the HUD's stats were read (get_debug_stats walks every chunk)
var overlay: CanvasLayer = null  # --overlay=video: the frame-time graph and scene facts, in place of the HUD; none: hidden
var flashlight: SpotLight3D = null  # --flashlight=1: a shadowed spot carried by the camera, with a wide flood; F toggles it
var pixelate := 1           # --pixelate=N: the 3D drawn in N x N blocks, under a full-resolution HUD

func _enter_tree() -> void:
	_setup_upscaling()  # before the terrain's _ready: its LOD reads the render scale
	var terrain := get_parent().get_node_or_null("VoxelTerrain")
	for a in Presets.user_args():
		if a.begins_with("--lods=") and terrain:
			terrain.lod_numbers = a.trim_prefix("--lods=").to_int()
		if a == "--terrain=cascade" and terrain:
			terrain.terrain_mode = 1
		if a.begins_with("--discoverer=") and terrain:
			var modes := {"cpu": 0, "gpu": 1}
			var mode := a.trim_prefix("--discoverer=")
			if modes.has(mode):
				terrain.discoverer_mode = modes[mode]
			else:
				push_error("player: unknown --discoverer=%s (expected cpu or gpu)" % mode)
		if a.begins_with("--walk_gpu=") and terrain:
			terrain.walk_gpu = a.trim_prefix("--walk_gpu=") == "1"
		if a == "--terrain=continents" and terrain:
			terrain.terrain_mode = 2
		if a == "--terrain=shapes" and terrain:
			terrain.terrain_mode = 4
		if a.begins_with("--altitude="):
			spawn_altitude = a.trim_prefix("--altitude=").to_float()
		# --spawn=U,V: metres along the base's tangent plane (VoxelTerrain.get_tangent_position; the flat world's x, z
		# from its (0, 0) column, origin 0); tools/climate_report.sh lists biome borders in these coordinates.
		if a.begins_with("--spawn="):
			var uv := a.trim_prefix("--spawn=").split(",")
			if uv.size() == 2:
				spawn_tangent = Vector2(uv[0].to_float(), uv[1].to_float())
			else:
				push_error("player: --spawn=%s (expected U,V in metres)" % a.trim_prefix("--spawn="))
		if a.begins_with("--voxel=") and terrain:
			terrain.voxel_size = a.trim_prefix("--voxel=").to_float()
		if a.begins_with("--lod_error=") and terrain:
			terrain.lod_pixel_error = a.trim_prefix("--lod_error=").to_float()
		if a.begins_with("--lod_errors=") and terrain:
			terrain.lod_pixel_errors = PackedFloat32Array(Array(a.trim_prefix("--lod_errors=").split(",")).map(func(e): return e.to_float()))
		if a.begins_with("--lod_error_back=") and terrain:
			terrain.lod_pixel_error_back = a.trim_prefix("--lod_error_back=").to_float()
		if a.begins_with("--lod_view_margin=") and terrain:
			terrain.lod_view_margin_deg = a.trim_prefix("--lod_view_margin=").to_float()
		if a.begins_with("--lod_view_memory=") and terrain:
			terrain.lod_view_memory_m = a.trim_prefix("--lod_view_memory=").to_float()
		if a.begins_with("--lod_far_limit=") and terrain:
			terrain.lod_far_limit = a != "--lod_far_limit=0"
		if a.begins_with("--mesher_budget=") and terrain:
			terrain.direct_gpu_budget_ms = a.trim_prefix("--mesher_budget=").to_float()
		if a.begins_with("--view=") and terrain:
			terrain.debug_view = a.trim_prefix("--view=").to_int()
		if a == "--grade=0":
			Presets.ungrade(get_parent())
		if a == "--ssao=0" and Presets.environment(get_parent()):
			Presets.environment(get_parent()).ssao_enabled = false
		if a.begins_with("--ambient=") and Presets.environment(get_parent()):
			Presets.environment(get_parent()).ambient_light_energy = a.trim_prefix("--ambient=").to_float()
		for key in ["dig_radius", "dig_length", "spray_radius", "drag_radius", "drag_length"]:
			if a.begins_with("--%s=" % key):
				dig_settings[key] = a.trim_prefix("--%s=" % key).to_float()
		if a.begins_with("--drag_op="):
			dig_settings["drag_union"] = a == "--drag_op=union"
		if a == "--left_click=tube":
			left_tube = true
		if a.begins_with("--instances=") and terrain:
			terrain.archetype_instances = a.trim_prefix("--instances=").to_int()
		if a.begins_with("--instance_area=") and terrain:
			terrain.archetype_instance_area = a.trim_prefix("--instance_area=").to_float()
		if a == "--moments=0" and terrain:
			terrain.archetype_moments = false
		if a == "--instance_band=0" and terrain:
			terrain.archetype_band = false
		if a == "--stream=0" and terrain:
			terrain.archetype_stream = false
		if a.begins_with("--stream_part_sides=") and terrain:
			terrain.archetype_stream_part_sides = a.trim_prefix("--stream_part_sides=").to_int()
		if a.begins_with("--instance_erode=") and terrain:
			terrain.archetype_erode = a.trim_prefix("--instance_erode=").to_float()
		if a.begins_with("--instance_resolve=") and terrain:
			terrain.archetype_resolve = a.trim_prefix("--instance_resolve=").to_float()
		if a.begins_with("--trees=") and terrain:
			terrain.archetype_trees = a.trim_prefix("--trees=").to_int()
		if a.begins_with("--tree_area=") and terrain:
			terrain.archetype_tree_area = a.trim_prefix("--tree_area=").to_float()
		if a.begins_with("--biomes=") and terrain:
			terrain.biome_radius = 1000.0 * a.trim_prefix("--biomes=").to_float()
		if a.begins_with("--climate=") and terrain:
			terrain.climate_path = a.trim_prefix("--climate=")
		if a == "--dual_contouring=1" and terrain:
			terrain.dual_contouring = true
	if terrain:
		terrain.set_dig_settings(dig_settings)

func _ready():
	var terrain := get_parent().get_node_or_null("VoxelTerrain")
	if terrain:
		clip_terrain = terrain
		base_near = camera.near
	if terrain and terrain.terrain_mode != 4:  # the shapes gallery keeps the scene's spawn
		altitude_terrain = terrain
		if spawn_tangent != null:
			global_position = terrain.get_tangent_position(spawn_tangent)
		# The cascade terrain's relief is kilometres high: start spawn_altitude above the ground under the spawn.
		for i in 32:
			var d: float = terrain.get_sdf_at(global_position)
			if absf(d - spawn_altitude) < spawn_altitude * 0.25:
				break
			global_position.y += spawn_altitude - d
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	if terrain:
		if "--overlay=video" in Presets.user_args():
			overlay = VideoOverlay.new()
			overlay.setup(terrain, camera)
			overlay.pixelate = pixelate
			add_child(overlay)
		else:
			_make_hud()
			hud_visible = not "--overlay=none" in Presets.user_args()  # a clean screen; TAB shows the HUD
	yaw = rotation.y
	pitch = camera.rotation.x
	_make_flashlight("--flashlight=1" in Presets.user_args())
	for a in Presets.user_args():
		if a.begins_with("--marks="):
			marks_dir = a.trim_prefix("--marks=")
	DirAccess.make_dir_recursive_absolute(marks_dir)
	get_tree().auto_accept_quit = false

# Godot's built-in 3D upscalers (no DLSS in 4.6): `--upscale=fsr2|fsr1|bilinear` renders the 3D at
# `--render_scale=F` (default 0.5 per axis: a quarter of the pixels) and upscales to the window. The LOD's pixel error
# is in render pixels, so 0.5 draws the meshes of twice the error. FSR 2 needs `--render-thread safe`: on the separate
# render thread Godot 4.6.3 exits silently while loading its shaders.
# `--pixelate=N` (a fake-resolution look): the 3D renders at 1/N of the 3D scale (`--pixelate=2 --upscale=fsr2`: a
# quarter of the window per axis), is upscaled to the window, then a screen pass copies one pixel over each N x N block.
# The HUD draws on a layer above it, at full resolution. The LOD reads the 3D render's size, so it follows.
func _setup_upscaling() -> void:
	for a in Presets.user_args():
		if a.begins_with("--pixelate="):
			pixelate = maxi(a.trim_prefix("--pixelate=").to_int(), 1)
	if pixelate > 1:
		_make_pixel_blocks()
		get_viewport().scaling_3d_scale = clampf(1.0 / pixelate, 0.25, 1.0)
	var modes := {
		"bilinear": Viewport.SCALING_3D_MODE_BILINEAR,
		"fsr1": Viewport.SCALING_3D_MODE_FSR,
		"fsr2": Viewport.SCALING_3D_MODE_FSR2,
	}
	var mode := ""
	var render_scale := 0.5
	for a in Presets.user_args():
		if a.begins_with("--upscale="):
			mode = a.trim_prefix("--upscale=")
		if a.begins_with("--render_scale="):
			render_scale = a.trim_prefix("--render_scale=").to_float()
	if mode.is_empty():
		return
	if not modes.has(mode):
		push_error("--upscale=%s: expected fsr2, fsr1 or bilinear" % mode)
		return
	var viewport := get_viewport()
	viewport.scaling_3d_mode = modes[mode]
	viewport.scaling_3d_scale = clampf(render_scale / pixelate, 0.25, 2.0)

# Below the HUD (layer 1): every pixel of an N x N block takes the screen's colour at the block's first pixel.
func _make_pixel_blocks() -> void:
	var shader := Shader.new()
	shader.code = """shader_type canvas_item;
uniform sampler2D screen_tex : hint_screen_texture, filter_nearest;
uniform float block = 2.0;
void fragment() {
	vec2 px = floor(FRAGCOORD.xy / block) * block + 0.5;
	COLOR = vec4(texture(screen_tex, px * SCREEN_PIXEL_SIZE).rgb, 1.0);
}"""
	var material := ShaderMaterial.new()
	material.shader = shader
	material.set_shader_parameter("block", float(pixelate))
	var rect := ColorRect.new()
	rect.material = material
	rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var layer := CanvasLayer.new()
	layer.layer = 0
	layer.add_child(rect)
	add_child(layer)

func _make_hud() -> void:
	var hud_layer := CanvasLayer.new()
	add_child(hud_layer)
	hud = Label.new()
	hud.position = Vector2(8, 8)
	hud.add_theme_font_size_override("font_size", 11)
	hud.add_theme_color_override("font_color", Color(1, 1, 1))
	hud.add_theme_color_override("font_shadow_color", Color(0, 0, 0))
	hud.add_theme_constant_override("shadow_offset_x", 1)
	hud.add_theme_constant_override("shadow_offset_y", 1)
	hud_layer.add_child(hud)

# A torch: a spot along the view, slightly below and right of the eye so its shadows read as cast from the hand.
func _make_flashlight(p_on: bool) -> void:
	flashlight = SpotLight3D.new()
	flashlight.position = Vector3(0.25, -0.2, 0.0)
	flashlight.light_color = Color(1.0, 0.92, 0.8)
	flashlight.light_energy = 3.0
	flashlight.spot_range = 60.0
	flashlight.spot_angle = 32.0
	flashlight.spot_attenuation = 0.7
	flashlight.shadow_enabled = true
	flashlight.visible = p_on
	camera.add_child(flashlight)
	# The flood around the beam: wide, dimmer, shorter, no shadows (a second shadowed spot doubles the shadow cost).
	var flood := SpotLight3D.new()
	flood.light_color = flashlight.light_color
	flood.light_energy = 1.5
	flood.spot_range = 40.0
	flood.spot_angle = 75.0
	flood.spot_attenuation = 0.5
	flashlight.add_child(flood)

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		_mark("close", false)
		get_tree().quit()

func _terrain() -> Node:
	return get_tree().current_scene.find_child("VoxelTerrain", true, false)

func _mark(kind: String, audit: bool) -> void:
	mark_index += 1
	var t: Transform3D = camera.global_transform
	var terrain_node := _terrain()
	var offset: Vector3 = terrain_node.get_render_offset_m() if terrain_node else Vector3.ZERO
	var m := {
		"index": mark_index, "kind": kind, "time": Time.get_datetime_string_from_system(),
		"frame": Engine.get_process_frames(),
		# Relative to the world origin (bench.sh --cam_pos), whatever the render origin moved to.
		"cam_pos": [t.origin.x + offset.x, t.origin.y + offset.y, t.origin.z + offset.z],
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
		m["lod_pixel_error_back"] = terrain.lod_pixel_error_back
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

	# Dig tools: right click digs a hole, left click (held) sprays or drags the tube. Only while the mouse is captured:
	# a click that gives the window focus back (after Escape) must not dig. A release always ends the drag.
	if event is InputEventMouseButton:
		var captured := Input.mouse_mode == Input.MOUSE_MODE_CAPTURED
		var terrain := _terrain()
		if terrain and event.pressed and captured and event.button_index == MOUSE_BUTTON_RIGHT:
			terrain.dig_hole(camera.global_position, -camera.global_transform.basis.z)
		elif event.button_index == MOUSE_BUTTON_LEFT:
			left_held = event.pressed and captured
			if terrain and not event.pressed:
				terrain.dig_release()

	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_TAB:
		hud_visible = not hud_visible
		if overlay:
			overlay.visible = hud_visible

	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_F:
		flashlight.visible = not flashlight.visible

	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_H:
		_mark("hole", true)

	# V cycles the debug views (VoxelTerrain.debug_view): shaded, normals, LOD, page, wireframe, chunk, group, quad, faceted,
	# material, biome (12; 10 / 11 are the bench's bark probe). B toggles the biome view.
	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_V:
		var terrain := _terrain()
		if terrain:
			var views := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
			terrain.debug_view = views[(views.find(terrain.debug_view) + 1) % views.size()]
			print("debug view ", terrain.debug_view)
	if event is InputEventKey and event.pressed and not event.echo and event.physical_keycode == KEY_B:
		var terrain := _terrain()
		if terrain:
			terrain.debug_view = 0 if terrain.debug_view == 12 else 12
			print("debug view ", terrain.debug_view)

	if event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		else:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _update_dig(delta: float) -> void:
	var terrain := _terrain()
	if not left_held or terrain == null:
		return
	var fwd := -camera.global_transform.basis.z
	if left_tube:
		terrain.dig_tube(camera.global_position, fwd)
	else:
		terrain.dig_spray(camera.global_position, fwd, delta)

func _update_hud(delta: float) -> void:
	if hud == null:
		return
	hud.visible = hud_visible
	hud_age += delta
	if not hud_visible or hud_age < 0.25:
		return
	hud_age = 0.0
	var terrain := _terrain()
	if terrain == null:
		return
	var s: Dictionary = terrain.get_debug_stats()
	var edits: Dictionary = s.get("edits", {})
	var ms: Dictionary = s.get("mesher", {})
	var apply_ms := 0.0
	if edits.get("apply") is Dictionary:
		apply_ms = float((edits["apply"] as Dictionary).get("avg", 0.0))
	var chunks: Dictionary = s.get("chunks", {})
	hud.text = "fps %d  frame %.1f ms  lod_error %.1f  %s
chunks %d loaded, %d pending  triangles %.1f M  mesher queue %d
edits %d (%d by the tools)
re-mesh queue %d  dispatched %d  stale %d
last edit %d chunks, apply %.2f ms" % [
		Engine.get_frames_per_second(), 1000.0 / maxf(Engine.get_frames_per_second(), 1.0), terrain.lod_pixel_error,
		"loaded" if s.get("load_complete", false) else "loading",
		int(chunks.get("loaded", 0)), int(chunks.get("pending", 0)), float(chunks.get("live_triangles", 0)) / 1e6,
		int(ms.get("queued", 0)), int(edits.get("edits", 0)), terrain.get_dig_edits(),
		int(ms.get("remesh_queue", 0)), int(ms.get("remesh_pops", 0)), int(ms.get("remesh_stale_dropped", 0)),
		int(edits.get("remesh_last", 0)), apply_ms]

func _process(delta: float) -> void:
	_update_dig(delta)
	_update_hud(delta)
	if clip_terrain:
		Presets.clip_planes(camera, clip_terrain, base_near)
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
	if altitude_terrain:
		# Cascade terrain: kilometre-scale relief, fly faster when high (FLY_RATE altitudes per second, at least
		# `speed`). The altitude is a CPU cascade point query, every frame; only the reading is smoothed (a canopy
		# or a ridge passing under steps it), in log space and briefly: a longer lag on the speed itself left a climb
		# at half speed for seconds.
		var altitude := maxf(absf(altitude_terrain.get_sdf_at(camera.global_position)), 0.01)
		if smoothed_altitude <= 0.0:
			smoothed_altitude = altitude
		smoothed_altitude *= pow(altitude / smoothed_altitude, 1.0 - exp(-delta / ALTITUDE_TAU))
		move_speed = maxf(speed, smoothed_altitude * FLY_RATE)
	position += direction * move_speed * delta
