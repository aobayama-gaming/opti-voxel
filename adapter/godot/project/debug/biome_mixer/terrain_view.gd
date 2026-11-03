extends SubViewportContainer
## The mixer's 3D view: the game's own terrain (a VoxelTerrain, the GPU Direct mesher) on the world being edited, in a
## viewport of its own with the game's sky, fog and sun. An edit restarts the terrain's world on the new one in place
## (VoxelTerrain.restart_world: its device, pages and pipelines kept): the climate's numbers are shader params
## (Cascade::ShaderParams), so a world whose structure didn't change builds no pipeline, and the time is the reload of
## the chunks in view. A drag restarts it as often as the loads keep up. The view holds the last terrain's frame (a
## copy over it: a SubViewport stopped with UPDATE_DISABLED showed black, noise and stripes, not its last frame) until the
## new one's chunks are loaded (no blank ground between them). Without the world's clouds (their shadows),
## trees, rocks and features (half the workers' time and the 10-33 ms instance lists of a drag's frames).
## The chunks load round the orbit's pivot (VoxelTerrain.set_lod_centre), from the LOD the camera's distance needs there
## (lod_view_min) to LOD_VIEW_MAX, which at the pixel error reaches ~54 km from it.
## The orbit's pivot is a player: a capsule of a person's height (a scale), its feet on the ground (get_sdf_at, the CPU's
## SDF) unless noclip. Move: left drag (walking, it follows the ground; noclip: it flies where the camera looks); orbit:
## right or middle drag; zoom: the wheel, down to the player. Nothing without a RenderingDevice (the headless check).

signal status_changed

## The pixel error of the view's LODs, the game's: at 12 the LOD seams showed as ledges, for a reload only ~10% shorter.
const LOD_PIXEL_ERROR := 4.0
## The coarsest LOD the view loads (12.8 km chunks): its reach, ~54 km round the pivot.
const LOD_VIEW_MAX := 13
## The pause after an edit before the terrain restarts (seconds): a slider's edits of a few frames restart it once.
const RESTART_AFTER := 0.05
## A restart waits for the last one's chunks in view at most this long (seconds): a drag shows each load finished.
const LOAD_WAIT := 1.0
## The game's look: its sky, fog and sun are this scene's.
const GAME_SCENE := "res://main_scene_5cm.tscn"
## The game's fog over views of kilometres instead of metres.
const FOG_DENSITY := 0.00004
## The player: a capsule of a person's height and its radius (metres).
const PLAYER_HEIGHT := 1.7
const PLAYER_RADIUS := 0.25
## The camera's closest orbit (metres from the player's feet) and its least height above the ground.
const NEAREST := 2.5
const CAMERA_CLEARANCE := 0.3
## First person: the eyes' height above the feet (metres), a drag's move in metres per pixel.
const EYE_HEIGHT := 1.6
const FPV_STEP := 0.05
## The nearest far plane (metres): the LODs' reach round the player (LOD_VIEW_MAX, ~54 km), zoomed in or not.
const VIEW_FAR := 60000.0
## The aim's distance: cast this often (seconds), at most this many steps, each this share of the SDF (under 1: the
## SDF is a height difference, not a true distance, and overshoots on slopes).
const AIM_EVERY := 0.1
const AIM_STEPS := 256
const AIM_STEP := 0.7
## A game character's run (metres a second; Minecraft sprints at 5.6, most action games run at 5-7): WASD in first
## person, and the aim's distance read as the time to run it.
const KEY_SPEED := 6.0

var _viewport := SubViewport.new()
var _held := TextureRect.new()  # the last terrain's frame while the new one loads
var _aim_label := Label.new()  # the crosshair and the distance to the terrain under it
var _aim_timer := 0.0
var _camera := Camera3D.new()
var _player := MeshInstance3D.new()
var _terrain: Node3D  # VoxelTerrain
var _restart := Timer.new()
var _json := ""
var _dir := ""
var _started := ""  # the JSON the running terrain was started on
var _restart_usec := 0
var _target := Vector3.ZERO
var _aim := Vector3.ZERO  # the target asked: tangent metres (x, z) and the altitude above the base (y)
var _aim_pending := false
var _up := Vector3.UP  # the ground's up at the target: the orbit's axis
var _shifted := Vector3.ZERO  # the running terrain's render origin shifts (it moved the camera with them)
var _yaw := 0.7
var _pitch := 0.45
var _dist := 8000.0
var status := ""  # the last restart's line
var loaded_ms := -1  # the last restart until its chunks in view were loaded, -1 while loading
var profile := {}  # --drag: "restart_ms" and "loaded_ms" appended to, when set
var noclip := false  # the player flies through the ground, else walks on it
var fpv := false  # the camera at the player's eyes, else orbiting them


func _init() -> void:
	stretch = true
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	mouse_filter = Control.MOUSE_FILTER_STOP
	focus_mode = Control.FOCUS_CLICK  # WASD while focused
	_viewport.own_world_3d = true
	_viewport.render_target_update_mode = SubViewport.UPDATE_WHEN_VISIBLE
	add_child(_viewport)
	_held.set_anchors_preset(Control.PRESET_FULL_RECT)
	_held.stretch_mode = TextureRect.STRETCH_SCALE
	_held.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_held.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_held.visible = false
	add_child(_held)
	_aim_label.set_anchors_preset(Control.PRESET_FULL_RECT)
	_aim_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_aim_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_aim_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_aim_label.add_theme_color_override("font_outline_color", Color.BLACK)
	_aim_label.add_theme_constant_override("outline_size", 4)
	add_child(_aim_label)
	_add_game_look()
	_viewport.add_child(_camera)
	var capsule := CapsuleMesh.new()
	capsule.height = PLAYER_HEIGHT
	capsule.radius = PLAYER_RADIUS
	var look := StandardMaterial3D.new()
	look.albedo_color = Color(1.0, 0.45, 0.1)
	look.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	capsule.material = look
	_player.mesh = capsule
	_viewport.add_child(_player)
	_restart.one_shot = true
	_restart.wait_time = RESTART_AFTER
	_restart.timeout.connect(_restart_terrain)
	add_child(_restart)


# GAME_SCENE's WorldEnvironment (its fog at FOG_DENSITY) and sun, read from the scene's state (nothing instanced).
func _add_game_look() -> void:
	var state := (load(GAME_SCENE) as PackedScene).get_state()
	for n in state.get_node_count():
		var node: Node
		match state.get_node_type(n):
			&"WorldEnvironment":
				node = WorldEnvironment.new()
			&"DirectionalLight3D":
				node = DirectionalLight3D.new()
			_:
				continue
		for p in state.get_node_property_count(n):
			node.set(state.get_node_property_name(n, p), state.get_node_property_value(n, p))
		if node is WorldEnvironment:
			node.environment = node.environment.duplicate(true)  # its sky's material too: the terrain writes into it
			node.environment.fog_density = FOG_DENSITY
		_viewport.add_child(node)


# p_json: the world the 2D view draws (a world file's text, its paths from p_dir). The terrain restarts on it once the
# edits pause; p_frame: the camera fitted to p_extent_km round p_centre_m, its target at p_ground_m.
func show_world(p_json: String, p_dir: String, p_frame: bool, p_extent_km: float, p_centre_m: Vector2, p_ground_m: float) -> void:
	_json = p_json
	_dir = p_dir
	if p_frame:
		_dist = p_extent_km * 1000.0 * 0.6
		_aim = Vector3(p_centre_m.x, p_ground_m, p_centre_m.y)
		_aim_pending = true
		if _json == _started:
			_aim_camera()  # else on the terrain that starts on p_json
	_schedule_restart()


func _schedule_restart() -> void:
	if _json != _started and is_visible_in_tree() and RenderingServer.get_rendering_device() != null:
		loaded_ms = -1
		if _restart.is_stopped():  # not pushed back: maps faster than RESTART_AFTER (a drag's) would never let it fire
			_restart.start()


# The terrain restarted on the world it shows (an edit's path, the same structure: every pipeline kept), now.
func restart() -> void:
	_restart.stop()
	_restart_terrain()


func _restart_terrain() -> void:
	if not is_inside_tree() or not is_visible_in_tree():
		return  # once shown (visibility_changed)
	if _restart_usec != 0 and Time.get_ticks_usec() - _restart_usec < int(LOAD_WAIT * 1e6):
		_restart.start()  # the last edit's chunks still loading
		return
	_restart_usec = Time.get_ticks_usec()
	if _terrain != null:
		_hold(true)  # the last frame on screen while the new world loads
		if profile.has("hold_ms"):
			profile.hold_ms.append((Time.get_ticks_usec() - _restart_usec) / 1000.0)
		_unshift_camera()  # the world starts again at its first render origin
		_terrain.restart_world(_view_world(_json), _dir)
	else:
		_terrain = ClassDB.instantiate("VoxelTerrain")
		_terrain.pipeline_mode = 2
		_terrain.lod_pixel_error = LOD_PIXEL_ERROR
		_terrain.lod_far_limit = true
		_terrain.lod_view_max = LOD_VIEW_MAX
		# The discoverer on the GPU: its local device is kept across restarts (World::Kept), a scan ~1.5 ms (46 on the
		# CPU); only the first world pays the device and its point pipelines (~2 s). The site walk on the CPU: its
		# device is made per world (~2 s a restart), and the view draws no instances.
		_terrain.discoverer_mode = 1
		_terrain.walk_gpu = false
		_terrain.direct_batches_per_frame = 8  # 4 (the game's): a reload 116 ms p50, 8: 100, 16: 108 (heavier frames)
		# The generic shader (SDFCascade::specialize): an edit of a value, a shape's type or a 0 included, keeps the
		# cascade's GPU text, so a restart compiles nothing (the specialised text froze the view 3.5 s on such edits).
		_terrain.specialize_shaders = false
		_terrain.set_world_json(_view_world(_json), _dir)
		_terrain.origin_shifted.connect(func(p_shift: Vector3) -> void:
			_target -= p_shift
			_shifted += p_shift)
		_viewport.add_child(_terrain)  # starts the world: its pipelines, then the chunks
	_aim_camera()
	_to_ground()  # the edited ground under the player
	_place_camera()  # the new terrain's LOD centre
	_started = _json
	loaded_ms = -1
	if profile.has("restart_ms"):
		profile.restart_ms.append((Time.get_ticks_usec() - _restart_usec) / 1000.0)
	status = "terrain restarted in %d ms, loading" % ((Time.get_ticks_usec() - _restart_usec) / 1000)
	status_changed.emit()


# p_json as the view draws it: its look's clouds off (over a view of kilometres their shadows hide the ground's relief)
# and no instances or features, the world's nor a biome's ([] is none; the view shows the ground being edited).
static func _view_world(p_json: String) -> String:
	var world: Dictionary = JSON.parse_string(p_json)
	var look: Dictionary = world.get("look", {})
	look["clouds"] = false
	world["look"] = look
	world["instances"] = []
	world["features"] = []
	for biome: Dictionary in world.get("biomes", []):
		biome["instances"] = []
		biome["features"] = []
	return JSON.stringify(world)


# The terrain stopped (its world, its GPU pages): the camera back in the first render origin's frame, where the next
# terrain starts, the view kept.
func _stop_terrain() -> void:
	if _terrain == null:
		return
	_viewport.remove_child(_terrain)  # its teardown: the world stops
	_terrain.free()
	_terrain = null
	_started = ""
	_restart_usec = 0
	_hold(false)
	_unshift_camera()


func _unshift_camera() -> void:
	_target += _shifted
	_shifted = Vector3.ZERO
	_place_camera()


func _notification(p_what: int) -> void:
	if p_what != NOTIFICATION_VISIBILITY_CHANGED:
		return
	if is_visible_in_tree():
		_schedule_restart()
	else:
		_restart.stop()
		_stop_terrain()  # nothing runs behind the 2D view


func _process(p_delta: float) -> void:
	_keys_move(p_delta)
	_aim_timer -= p_delta
	if _aim_timer <= 0.0:
		_aim_timer = AIM_EVERY
		var d := aim_distance()
		# The crosshair, the distance under it beneath (a second line keeps the "+" on the view's centre).
		_aim_label.text = "+\n%s" % ("—" if d < 0.0 else "%s · %s" % [_metres(d), _run_time(d / KEY_SPEED)])
	if _terrain == null or _restart_usec == 0:
		return
	if _terrain.is_load_complete() or Time.get_ticks_usec() - _restart_usec > int(LOAD_WAIT * 1e6):
		_hold(false)
		loaded_ms = (Time.get_ticks_usec() - _restart_usec) / 1000
		if profile.has("loaded_ms"):
			profile.loaded_ms.append(loaded_ms)
		status = "terrain loaded in %d ms (restart and chunks in view)" % loaded_ms
		_restart_usec = 0
		status_changed.emit()


# p_on: a copy of the viewport's frame over it (the viewport renders on below), else the viewport itself. A hold that
# is on keeps its frame (the last loaded one, not a half-loaded one).
func _hold(p_on: bool) -> void:
	if p_on and not _held.visible:
		_held.texture = ImageTexture.create_from_image(_viewport.get_texture().get_image())
	_held.visible = p_on
	if not p_on:
		_held.texture = null


# The camera's target on the asked point: the terrain's base there (its render position, VoxelTerrain's tangent frame)
# plus the altitude along its up. Waits for a terrain.
func _aim_camera() -> void:
	if not _aim_pending or _terrain == null or not _terrain.is_inside_tree():
		return
	var base: Vector3 = _terrain.get_tangent_position(Vector2(_aim.x, _aim.z))
	_up = _terrain.get_up_at(base)
	_target = base + _up * _aim.y
	_aim_pending = false
	_to_ground()
	_place_camera()


# The orbit's horizontal axes round _up (x, z of a flat world).
func _ground_axes() -> Array[Vector3]:
	var z := Vector3.RIGHT.cross(_up)
	z = z.normalized() if z.length() > 1e-3 else Vector3.BACK
	return [_up.cross(z), z]


static func _metres(p_m: float) -> String:
	return "%.1f m" % p_m if p_m < 1000.0 else "%.2f km" % (p_m / 1000.0)


# p_s seconds as a run's time: "12 s", "3 min 20 s", "1 h 05 min".
static func _run_time(p_s: float) -> String:
	var s := int(round(p_s))
	if s < 60:
		return "%d s run" % s
	if s < 3600:
		return "%d min %02d s run" % [s / 60, s % 60]
	return "%d h %02d min run" % [s / 3600, (s % 3600) / 60]


# The distance (metres) from the camera to the terrain at the view's centre, -1 past VIEW_FAR or without a terrain:
# sphere tracing on the CPU's SDF (get_sdf_at, steps of AIM_STEP of it), a crossing refined by bisection.
func aim_distance() -> float:
	if _terrain == null or not _terrain.is_inside_tree():
		return -1.0
	var from := _camera.global_position
	var dir := -_camera.global_transform.basis.z
	var t := 0.0
	var last := 0.0
	for i in AIM_STEPS:
		var d: float = _terrain.get_sdf_at(from + dir * t)
		if d <= 0.0:
			var lo := last
			var hi := t
			for j in 16:  # inside: the crossing between the last point outside and this one
				var mid := (lo + hi) * 0.5
				if _terrain.get_sdf_at(from + dir * mid) > 0.0:
					lo = mid
				else:
					hi = mid
			return hi
		last = t
		t += maxf(d * AIM_STEP, 0.02)
		if t > VIEW_FAR:
			return -1.0
	return -1.0


# The player's feet on the ground under them (along the terrain's up, by its SDF), unless noclip.
func _to_ground() -> void:
	if noclip or _terrain == null or not _terrain.is_inside_tree():
		return
	for i in 32:
		var d: float = _terrain.get_sdf_at(_target)
		if absf(d) < 0.01:
			break
		_target -= _terrain.get_up_at(_target) * d


func set_noclip(p_on: bool) -> void:
	noclip = p_on
	_to_ground()
	_place_camera()


func set_fpv(p_on: bool) -> void:
	fpv = p_on
	_place_camera()


# From the player towards the orbiting camera (yaw round _up, pitch above the ground); first person looks against it.
func _view_dir() -> Vector3:
	var axes := _ground_axes()
	return (axes[0] * sin(_yaw) + axes[1] * cos(_yaw)) * cos(_pitch) + _up * sin(_pitch)


# The player's right and forward: along the ground walking, forward where the camera looks in noclip.
func _move_axes() -> Array[Vector3]:
	var axes := _ground_axes()
	var right := axes[0] * cos(_yaw) - axes[1] * sin(_yaw)
	var fwd := -_view_dir() if noclip else -(axes[0] * sin(_yaw) + axes[1] * cos(_yaw))
	return [right, fwd]


# WASD while the view has focus (a click gives it): along the ground walking; in noclip forward where the camera looks,
# Space up and Shift down. KEY_SPEED metres a second in first person, the orbit's distance a second otherwise.
func _keys_move(p_delta: float) -> void:
	if not has_focus():
		return
	var mv := _move_axes()
	var v := Vector3.ZERO
	v += mv[1] * (float(Input.is_key_pressed(KEY_W)) - float(Input.is_key_pressed(KEY_S)))
	v += mv[0] * (float(Input.is_key_pressed(KEY_D)) - float(Input.is_key_pressed(KEY_A)))
	if noclip:
		v += _up * (float(Input.is_key_pressed(KEY_SPACE)) - float(Input.is_key_pressed(KEY_SHIFT)))
	if v == Vector3.ZERO:
		return
	_target += v.normalized() * (KEY_SPEED if fpv else maxf(_dist, 10.0)) * p_delta
	_to_ground()
	_place_camera()


func _place_camera() -> void:
	var dir := _view_dir()
	var eye := _target + _up * (PLAYER_HEIGHT * 0.5)  # the orbit round the capsule's middle
	if fpv:
		eye = _target + _up * EYE_HEIGHT
		_camera.position = eye
		_camera.look_at(eye - dir, _up)
	else:
		_camera.position = eye + dir * _dist
		if not noclip and _terrain != null and _terrain.is_inside_tree():
			var d: float = _terrain.get_sdf_at(_camera.position)
			if d < CAMERA_CLEARANCE:  # never under the ground the player walks
				_camera.position += _up * (CAMERA_CLEARANCE - d)
		_camera.look_at(eye, _up)
	_player.visible = not fpv
	_player.transform = Transform3D(Basis(Quaternion(Vector3.UP, _up)), _target + _up * (PLAYER_HEIGHT * 0.5))
	var dist := EYE_HEIGHT if fpv else _dist
	if _terrain != null:
		_terrain.set_lod_centre(_target)  # the LOD rings round the pivot, not the camera
		_terrain.lod_view_min = _terrain.get_lod_at_distance(dist)  # nothing finer than the pivot needs from here
	# The planes follow the orbit: a world's view is hundreds of km (the terrain loads to the far plane, lod_far_limit).
	_camera.far = maxf(_dist * 4.0, VIEW_FAR)
	_camera.near = clampf(dist * 0.0002, 0.05, 50.0)


func _gui_input(p_event: InputEvent) -> void:
	if p_event is InputEventMouseMotion:
		var m := p_event as InputEventMouseMotion
		if m.button_mask & (MOUSE_BUTTON_MASK_RIGHT | MOUSE_BUTTON_MASK_MIDDLE):
			_yaw -= m.relative.x * 0.008
			_pitch = clampf(_pitch + m.relative.y * 0.008, -1.55 if noclip or fpv else 0.05, 1.55)
		elif m.button_mask & MOUSE_BUTTON_MASK_LEFT:
			var mv := _move_axes()
			var step := FPV_STEP if fpv else maxf(_dist, 10.0) * 0.0015
			_target += (-mv[0] * m.relative.x + mv[1] * m.relative.y) * step
			_to_ground()
		else:
			return
		_place_camera()
		accept_event()
	elif p_event is InputEventMouseButton and p_event.pressed:
		var b := p_event as InputEventMouseButton
		if b.button_index == MOUSE_BUTTON_WHEEL_UP:
			_dist = maxf(_dist * 0.9, NEAREST)
		elif b.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_dist /= 0.9
		else:
			return
		_place_camera()
		accept_event()


func terrain() -> Node3D:
	return _terrain


func image() -> Image:
	return _viewport.get_texture().get_image()
