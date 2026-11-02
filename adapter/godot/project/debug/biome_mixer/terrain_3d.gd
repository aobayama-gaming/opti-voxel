extends SubViewportContainer
## The mixer's 3D view: the engine's map (ClimateReportNode.preview) as a surface, each point at the altitude the engine
## traced, coloured by the map the 2D view shows (the engine's shading or blended biome colours, unlit). Nothing here
## computes terrain. Orbit: left drag; pan: right or middle drag; zoom: the wheel.

var exaggeration := 1.0
var _viewport := SubViewport.new()
var _camera := Camera3D.new()
var _mesh := MeshInstance3D.new()
var _material := StandardMaterial3D.new()
var _target := Vector3.ZERO
var _yaw := 0.7
var _pitch := 0.5
var _dist := 20000.0
var _map := {}
var _image: Image


func _init() -> void:
	stretch = true
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	mouse_filter = Control.MOUSE_FILTER_STOP
	_viewport.render_target_update_mode = SubViewport.UPDATE_WHEN_VISIBLE
	_viewport.msaa_3d = Viewport.MSAA_4X
	add_child(_viewport)
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color(0.11, 0.12, 0.14)
	_viewport.add_child(env)
	_material.vertex_color_use_as_albedo = true
	_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	_mesh.material_override = _material
	_viewport.add_child(_mesh)
	_camera.far = 1.0e7
	_camera.near = 1.0
	_viewport.add_child(_camera)


# p_map: ClimateReportNode.preview's dictionary; p_image: the map the 2D view shows (its colours). p_frame: the
# camera's distance fitted to the map's extent (a new extent, not an edit).
func show_map(p_map: Dictionary, p_image: Image, p_frame: bool) -> void:
	_map = p_map
	_image = p_image
	_build()
	if p_frame:
		_dist = float(p_map.size) * float(p_map.spacing_m) * 0.9
		_target = Vector3.ZERO
	_place_camera()


func set_exaggeration(p_x: float) -> void:
	exaggeration = p_x
	_build()


func _build() -> void:
	if _map.is_empty() or _image == null:
		return
	var n: int = _map.size
	var spacing := float(_map.spacing_m)
	var alt: PackedFloat32Array = _map.altitude_m
	var verts := PackedVector3Array()
	var colors := PackedColorArray()
	verts.resize(n * n)
	colors.resize(n * n)
	var half := float(n / 2)
	for i in n:
		for j in n:
			var k := i * n + j
			verts[k] = Vector3((j - half) * spacing, alt[k] * exaggeration, (i - half) * spacing)
			colors[k] = _image.get_pixel(j, i)
	var idx := PackedInt32Array()
	idx.resize((n - 1) * (n - 1) * 6)
	var w := 0
	for i in n - 1:
		for j in n - 1:
			var a := i * n + j
			idx[w] = a
			idx[w + 1] = a + 1
			idx[w + 2] = a + n
			idx[w + 3] = a + 1
			idx[w + 4] = a + n + 1
			idx[w + 5] = a + n
			w += 6
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_INDEX] = idx
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	_mesh.mesh = mesh


func _place_camera() -> void:
	var dir := Vector3(cos(_pitch) * sin(_yaw), sin(_pitch), cos(_pitch) * cos(_yaw))
	_camera.position = _target + dir * _dist
	_camera.look_at(_target, Vector3.UP)


func _gui_input(p_event: InputEvent) -> void:
	if p_event is InputEventMouseMotion:
		var m := p_event as InputEventMouseMotion
		if m.button_mask & MOUSE_BUTTON_MASK_LEFT:
			_yaw -= m.relative.x * 0.008
			_pitch = clampf(_pitch + m.relative.y * 0.008, 0.05, 1.55)
		elif m.button_mask & (MOUSE_BUTTON_MASK_RIGHT | MOUSE_BUTTON_MASK_MIDDLE):
			var right := _camera.global_transform.basis.x
			var fwd := Vector3(-sin(_yaw), 0.0, -cos(_yaw))
			_target += (-right * m.relative.x + fwd * m.relative.y) * _dist * 0.0015
		else:
			return
		_place_camera()
		accept_event()
	elif p_event is InputEventMouseButton and p_event.pressed:
		var b := p_event as InputEventMouseButton
		if b.button_index == MOUSE_BUTTON_WHEEL_UP:
			_dist *= 0.9
		elif b.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_dist /= 0.9
		else:
			return
		_place_camera()
		accept_event()


func image() -> Image:
	return _viewport.get_texture().get_image()
