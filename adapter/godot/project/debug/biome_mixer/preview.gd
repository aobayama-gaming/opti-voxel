extends SubViewportContainer
## The mixer's raymarched view: one full-screen ColorRect in a SubViewport, drawn only when something changed (an
## edit, the camera, a resize): the laptop's GPU idles otherwise. While a drag goes on it draws at a coarser
## resolution, then once more at the chosen one. Orbit: left drag; pan: right or middle drag; zoom: the wheel.

const SETTLE_S := 0.25

var rect := ColorRect.new()
var viewport := SubViewport.new()
var mat := ShaderMaterial.new()
var code := ""
var quality := 1  # the stretch shrink when settled: 1 full resolution, 2 half, ...
var settle := Timer.new()

var target := Vector3.ZERO
var yaw := 0.7
var pitch := 0.42
var dist := 3000.0
var fov_deg := 50.0


func _init() -> void:
	stretch = true
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	mouse_filter = Control.MOUSE_FILTER_STOP
	viewport.render_target_update_mode = SubViewport.UPDATE_ONCE
	viewport.handle_input_locally = false
	add_child(viewport)
	rect.material = mat
	rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	viewport.add_child(rect)
	settle.one_shot = true
	settle.wait_time = SETTLE_S
	settle.timeout.connect(redraw)
	add_child(settle)
	resized.connect(redraw)


func set_code(p_code: String) -> void:
	if p_code == code:
		return
	code = p_code
	var s := Shader.new()
	s.code = p_code
	mat.shader = s


func set_uniforms(p_uniforms: Dictionary) -> void:
	for name in p_uniforms:
		mat.set_shader_parameter(name, p_uniforms[name])


# Draw once more: coarse now if p_moving (a drag), then at the chosen quality once the edits pause.
func redraw(p_moving: bool = false) -> void:
	var shrink := maxi(quality, 3) if p_moving else quality
	if stretch_shrink != shrink:
		stretch_shrink = shrink
	_camera()
	viewport.render_target_update_mode = SubViewport.UPDATE_ONCE
	if p_moving:
		settle.start()


func _camera() -> void:
	var dir := Vector3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
	var pos := target + dir * dist
	var fwd := -dir
	var right := fwd.cross(Vector3.UP).normalized()
	var up := right.cross(fwd)
	var px := Vector2(maxf(viewport.size.x, 1.0), maxf(viewport.size.y, 1.0))
	mat.set_shader_parameter("u_cam_pos", pos)
	mat.set_shader_parameter("u_cam_fwd", fwd)
	mat.set_shader_parameter("u_cam_right", right)
	mat.set_shader_parameter("u_cam_up", up)
	mat.set_shader_parameter("u_tan_half_fov", tan(deg_to_rad(fov_deg) * 0.5))
	mat.set_shader_parameter("u_view_px", px)
	mat.set_shader_parameter("u_far", maxf(dist * 8.0, 20000.0))


# Over the whole biome: the target at its middle, the ground's bound in view.
func frame(p_hmax: float) -> void:
	target = Vector3.ZERO
	# The bound is the worst case (every note at its peak at once); the ground's real relief is a fraction of it.
	dist = clampf(p_hmax * 1.5, 200.0, 40000.0)
	pitch = 0.42
	yaw = 0.7
	redraw()


func _gui_input(p_event: InputEvent) -> void:
	if p_event is InputEventMouseMotion:
		var mm := p_event as InputEventMouseMotion
		if mm.button_mask & MOUSE_BUTTON_MASK_LEFT:
			yaw -= mm.relative.x * 0.006
			pitch = clampf(pitch + mm.relative.y * 0.006, -0.2, 1.5)
			_moved()
		elif mm.button_mask & (MOUSE_BUTTON_MASK_RIGHT | MOUSE_BUTTON_MASK_MIDDLE):
			var right := Vector3(cos(yaw), 0.0, -sin(yaw))
			var back := Vector3(sin(yaw), 0.0, cos(yaw))
			var k := dist * 0.0015
			target += (-right * mm.relative.x - back * mm.relative.y) * k
			_moved()
	elif p_event is InputEventMouseButton and p_event.pressed:
		var mb := p_event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			dist = maxf(dist / 1.15, 5.0)
			_moved()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			dist = minf(dist * 1.15, 200000.0)
			_moved()


func _moved() -> void:
	redraw(true)
	accept_event()


# The rendered view as an image (the --shot flag).
func image() -> Image:
	return viewport.get_texture().get_image()
