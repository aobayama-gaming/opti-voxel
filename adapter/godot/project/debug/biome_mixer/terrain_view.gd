extends SubViewportContainer
## The mixer's 3D view: the game's own terrain (a VoxelTerrain, the GPU Direct mesher) on the world being edited, in a
## viewport of its own with the game's sky, fog and sun. An edit restarts the terrain's world on the new one in place
## (VoxelTerrain.restart_world: its device, pages and pipelines kept): the climate's numbers are shader params
## (Cascade::ShaderParams), so a world whose structure didn't change builds no pipeline, and the time is the reload of
## the chunks in view. A drag restarts it as often as the loads keep up. The view holds the last terrain's frame until
## the new one's chunks are loaded (no blank ground between them). Without the world's clouds (their shadows).
## Orbit: left drag; pan: right or middle drag; zoom: the wheel. Nothing without a RenderingDevice (the headless check).

signal status_changed

## The pixel error of the view's LODs, the game's: at 12 the LOD seams showed as ledges, for a reload only ~10% shorter.
const LOD_PIXEL_ERROR := 4.0
## The coarsest LOD the view loads (1.6 km chunks): its reach, the orbit's tens of km.
const LOD_VIEW_MAX := 10
## The pause after an edit before the terrain restarts (seconds): a slider's edits of a few frames restart it once.
const RESTART_AFTER := 0.05
## A restart waits for the last one's chunks in view at most this long (seconds): a drag shows each load finished.
const LOAD_WAIT := 1.0
## The game's look: its sky, fog and sun are this scene's.
const GAME_SCENE := "res://main_scene_5cm.tscn"
## The game's fog over views of kilometres instead of metres.
const FOG_DENSITY := 0.00004

var _viewport := SubViewport.new()
var _camera := Camera3D.new()
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


func _init() -> void:
	stretch = true
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	mouse_filter = Control.MOUSE_FILTER_STOP
	_viewport.own_world_3d = true
	_viewport.render_target_update_mode = SubViewport.UPDATE_WHEN_VISIBLE
	add_child(_viewport)
	_add_game_look()
	_viewport.add_child(_camera)
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
		_unshift_camera()  # the world starts again at its first render origin
		_terrain.restart_world(_without_clouds(_json), _dir)
	else:
		_terrain = ClassDB.instantiate("VoxelTerrain")
		_terrain.pipeline_mode = 2
		_terrain.lod_pixel_error = LOD_PIXEL_ERROR
		_terrain.lod_far_limit = true
		_terrain.lod_view_max = LOD_VIEW_MAX
		# The discoverer and the site walk on the CPU: their GPU variants build their point pipelines on a local device
		# of their own, ~9 s per restart (the driver, not glslang), for a view of a few hundred chunks.
		_terrain.discoverer_mode = 0
		_terrain.walk_gpu = false
		_terrain.set_world_json(_without_clouds(_json), _dir)
		_terrain.origin_shifted.connect(func(p_shift: Vector3) -> void:
			_target -= p_shift
			_shifted += p_shift)
		_viewport.add_child(_terrain)  # starts the world: its pipelines, then the chunks
	_aim_camera()
	_started = _json
	loaded_ms = -1
	status = "terrain restarted in %d ms, loading" % ((Time.get_ticks_usec() - _restart_usec) / 1000)
	status_changed.emit()


# p_json with its look's clouds off: over a view of kilometres their shadows hide the ground's relief.
static func _without_clouds(p_json: String) -> String:
	var world: Dictionary = JSON.parse_string(p_json)
	var look: Dictionary = world.get("look", {})
	look["clouds"] = false
	world["look"] = look
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


func _process(_delta: float) -> void:
	if _terrain == null or _restart_usec == 0:
		return
	if _terrain.is_load_complete() or Time.get_ticks_usec() - _restart_usec > int(LOAD_WAIT * 1e6):
		_hold(false)
		loaded_ms = (Time.get_ticks_usec() - _restart_usec) / 1000
		status = "terrain loaded in %d ms (restart and chunks in view)" % loaded_ms
		_restart_usec = 0
		status_changed.emit()


# p_on: the viewport keeps its last frame (renders nothing), else renders while shown.
func _hold(p_on: bool) -> void:
	_viewport.render_target_update_mode = SubViewport.UPDATE_DISABLED if p_on else SubViewport.UPDATE_WHEN_VISIBLE


# The camera's target on the asked point: the terrain's base there (its render position, VoxelTerrain's tangent frame)
# plus the altitude along its up. Waits for a terrain.
func _aim_camera() -> void:
	if not _aim_pending or _terrain == null or not _terrain.is_inside_tree():
		return
	var base: Vector3 = _terrain.get_tangent_position(Vector2(_aim.x, _aim.z))
	_up = _terrain.get_up_at(base)
	_target = base + _up * _aim.y
	_aim_pending = false
	_place_camera()


# The orbit's horizontal axes round _up (x, z of a flat world).
func _ground_axes() -> Array[Vector3]:
	var z := Vector3.RIGHT.cross(_up)
	z = z.normalized() if z.length() > 1e-3 else Vector3.BACK
	return [_up.cross(z), z]


func _place_camera() -> void:
	var axes := _ground_axes()
	var dir := (axes[0] * sin(_yaw) + axes[1] * cos(_yaw)) * cos(_pitch) + _up * sin(_pitch)
	_camera.position = _target + dir * _dist
	_camera.look_at(_target, _up)
	# The planes follow the orbit: a world's view is hundreds of km (the terrain loads to the far plane, lod_far_limit).
	_camera.far = maxf(_dist * 4.0, 2000.0)
	_camera.near = clampf(_dist * 0.0002, 0.05, 50.0)


func _gui_input(p_event: InputEvent) -> void:
	if p_event is InputEventMouseMotion:
		var m := p_event as InputEventMouseMotion
		if m.button_mask & MOUSE_BUTTON_MASK_LEFT:
			_yaw -= m.relative.x * 0.008
			_pitch = clampf(_pitch + m.relative.y * 0.008, 0.05, 1.55)
		elif m.button_mask & (MOUSE_BUTTON_MASK_RIGHT | MOUSE_BUTTON_MASK_MIDDLE):
			var axes := _ground_axes()
			var right := axes[0] * cos(_yaw) - axes[1] * sin(_yaw)
			var fwd := -(axes[0] * sin(_yaw) + axes[1] * cos(_yaw))
			_target += (-right * m.relative.x + fwd * m.relative.y) * _dist * 0.0015
		else:
			return
		_place_camera()
		accept_event()
	elif p_event is InputEventMouseButton and p_event.pressed:
		var b := p_event as InputEventMouseButton
		if b.button_index == MOUSE_BUTTON_WHEEL_UP:
			_dist = maxf(_dist * 0.9, 20.0)
		elif b.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_dist /= 0.9
		else:
			return
		_place_camera()
		accept_event()


func image() -> Image:
	return _viewport.get_texture().get_image()
