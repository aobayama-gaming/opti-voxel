extends Control
## The mixer's raymarched view, drawn only when something changed (an edit, the camera, a resize): the laptop's GPU
## idles otherwise. Two layers of the same shader: a coarse one (1/3 of the resolution), drawn on every change, and,
## once the changes pause, the chosen resolution over it in strips of about STRIP_MS each, so no frame waits on the
## whole view and an edit or a drag cancels the strips at once. Orbit: left drag; pan: right or middle drag; zoom:
## the wheel. The map (map = true) is the same shader from above (u_mode 1): any drag pans, the wheel zooms. sample()
## draws the chart's samples (u_mode 2) into a small float viewport and reads them back.

const SETTLE_S := 0.2
const COARSE := 3  # the coarse layer's shrink
const STRIP_MS := 25.0
const SAMPLES := 96  # the chart's samples per side
const ShaderLib := preload("res://debug/biome_mixer/shader_lib.gd")

signal sharpened  # a sharp pass ended (sharp_ms: how long it took)

var mat := ShaderMaterial.new()
var sample_mat := ShaderMaterial.new()
var sampler := SubViewport.new()
var map := false
var code := ""
var quality := 1  # the sharp layer's shrink: 1 full resolution, 2 half, ...
var settle := Timer.new()

var coarse := SubViewport.new()
var sharp := SubViewport.new()
var coarse_rect := ColorRect.new()
var strip := ColorRect.new()  # the sharp layer's next strip
var coarse_tex := TextureRect.new()
var sharp_tex := TextureRect.new()
var strip_y := -1  # the next strip's first row, -1 when no sharp pass runs
var strip_rows := 64.0  # rows per strip, fitted to STRIP_MS from the last strip's time
var strip_usec := 0
var pass_usec := 0
var sharp_ms := 0.0

var target := Vector3.ZERO
var yaw := 0.7
var pitch := 0.42
var dist := 3000.0
var fov_deg := 50.0


func _init() -> void:
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	mouse_filter = Control.MOUSE_FILTER_STOP
	clip_contents = true
	sharp.transparent_bg = true
	for vp in [coarse, sharp]:
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.handle_input_locally = false
		add_child(vp)
	for r in [coarse_rect, strip]:
		r.material = mat
		r.mouse_filter = Control.MOUSE_FILTER_IGNORE
	coarse_rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	coarse.add_child(coarse_rect)
	sharp.add_child(strip)
	for t in [coarse_tex, sharp_tex]:
		t.set_anchors_preset(Control.PRESET_FULL_RECT)
		t.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		t.stretch_mode = TextureRect.STRETCH_SCALE
		t.mouse_filter = Control.MOUSE_FILTER_IGNORE
		add_child(t)
	coarse_tex.texture = coarse.get_texture()
	sharp_tex.texture = sharp.get_texture()
	sampler.use_hdr_2d = true
	sampler.size = Vector2i(SAMPLES, SAMPLES)
	sampler.render_target_update_mode = SubViewport.UPDATE_DISABLED
	var sample_rect := ColorRect.new()
	sample_rect.material = sample_mat
	sample_rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	sample_rect.size = Vector2(SAMPLES, SAMPLES)
	sampler.add_child(sample_rect)
	add_child(sampler)
	settle.one_shot = true
	settle.wait_time = SETTLE_S
	settle.timeout.connect(_start_sharp)
	add_child(settle)
	resized.connect(redraw)
	RenderingServer.frame_post_draw.connect(_next_strip)


func set_code(p_code: String) -> void:
	if p_code == code:
		return
	code = p_code
	# A new shader's cost is unknown (a world's is many biomes'): its first strips start thin and grow.
	strip_rows = 8.0
	var s := Shader.new()
	s.code = p_code
	mat.shader = s
	sample_mat.shader = s


func set_uniforms(p_uniforms: Dictionary) -> void:
	for name in p_uniforms:
		mat.set_shader_parameter(name, p_uniforms[name])
		sample_mat.set_shader_parameter(name, p_uniforms[name])


# The fields p_cx, p_cy at SAMPLES^2 points over p_extent metres around p_centre (x, z), from the shader's own
# fields: the chart's cloud of where the ground goes.
func sample(p_cx: int, p_cy: int, p_centre: Vector2, p_extent: float) -> PackedVector2Array:
	sample_mat.set_shader_parameter("u_mode", 2)
	sample_mat.set_shader_parameter("u_cx", p_cx)
	sample_mat.set_shader_parameter("u_cy", p_cy)
	sample_mat.set_shader_parameter("u_map_centre", p_centre)
	sample_mat.set_shader_parameter("u_map_extent", p_extent)
	sampler.render_target_update_mode = SubViewport.UPDATE_ONCE
	await RenderingServer.frame_post_draw
	var img := sampler.get_texture().get_image()
	var out := PackedVector2Array()
	if img == null:
		return out
	for y in img.get_height():
		for x in img.get_width():
			var c := img.get_pixel(x, y)
			out.append(Vector2(c.r - ShaderLib.SAMPLE_BIAS, c.g - ShaderLib.SAMPLE_BIAS))
	return out


# Draw again: the coarse layer now, the sharp one once the edits pause (p_moving: a drag goes on).
func redraw(p_moving: bool = false) -> void:
	_camera()
	coarse.size = _px(COARSE)
	coarse.render_target_update_mode = SubViewport.UPDATE_ONCE
	strip_y = -1
	strip_usec = 0
	sharp_tex.visible = false
	if quality >= COARSE:
		return
	if p_moving:
		settle.start()
	else:
		_start_sharp()


# The whole view at the chosen resolution in one frame (the --shot flag: its time is the view's cost).
func draw_full() -> void:
	_camera()
	sharp.size = _px(quality)
	sharp.render_target_clear_mode = SubViewport.CLEAR_MODE_ONCE
	strip.position = Vector2.ZERO
	strip.size = Vector2(sharp.size)
	sharp.render_target_update_mode = SubViewport.UPDATE_ONCE
	sharp_tex.visible = true
	strip_y = -1


func _px(p_shrink: int) -> Vector2i:
	return Vector2i((size / float(p_shrink)).ceil().max(Vector2.ONE))


func _start_sharp() -> void:
	settle.stop()
	sharp.size = _px(quality)
	sharp.render_target_clear_mode = SubViewport.CLEAR_MODE_ONCE
	strip_y = 0
	strip_usec = 0
	pass_usec = Time.get_ticks_usec()
	_draw_strip()
	sharp_tex.visible = true


func _draw_strip() -> void:
	var rows := mini(int(strip_rows), sharp.size.y - strip_y)
	strip.position = Vector2(0, strip_y)
	strip.size = Vector2(sharp.size.x, rows)
	strip_y += rows
	sharp.render_target_update_mode = SubViewport.UPDATE_ONCE


# After each frame of a sharp pass: fit the strip to the frame's time, draw the next one. A frame only queues its
# strip; the GPU's wait lands on the next one, so the period between two frames is what a strip costs.
func _next_strip() -> void:
	if strip_y < 0:
		return
	var now := Time.get_ticks_usec()
	if strip_usec > 0:
		var ms := (now - strip_usec) / 1000.0
		strip_rows = clampf(strip.size.y * STRIP_MS / maxf(ms, 1.0), 8.0, strip_rows * 2.0)
	strip_usec = now
	if strip_y >= sharp.size.y:
		strip_y = -1
		strip_usec = 0
		sharp_ms = (now - pass_usec) / 1000.0
		sharpened.emit()
		return
	_draw_strip()


func _camera() -> void:
	mat.set_shader_parameter("u_mode", 1 if map else 0)
	mat.set_shader_parameter("u_map_centre", Vector2(target.x, target.z))
	mat.set_shader_parameter("u_map_extent", dist * 2.0)
	var dir := Vector3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
	var pos := target + dir * dist
	var fwd := -dir
	var right := fwd.cross(Vector3.UP).normalized()
	var up := right.cross(fwd)
	mat.set_shader_parameter("u_cam_pos", pos)
	mat.set_shader_parameter("u_cam_fwd", fwd)
	mat.set_shader_parameter("u_cam_right", right)
	mat.set_shader_parameter("u_cam_up", up)
	mat.set_shader_parameter("u_tan_half_fov", tan(deg_to_rad(fov_deg) * 0.5))
	mat.set_shader_parameter("u_far", maxf(dist * 8.0, 20000.0))


# Over a world: p_extent metres across in view, from higher up.
func frame_world(p_extent: float) -> void:
	target = Vector3.ZERO
	dist = p_extent * 0.5
	pitch = 0.75
	yaw = 0.7
	redraw()


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
		if map and mm.button_mask != 0:
			# The map is dist * 2 metres across the view's width.
			var k := dist * 2.0 / maxf(size.x, 1.0)
			target += Vector3(-mm.relative.x, 0.0, -mm.relative.y) * k
			_moved()
		elif mm.button_mask & MOUSE_BUTTON_MASK_LEFT:
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
			dist = minf(dist * 1.15, 4000000.0)
			_moved()


func _moved() -> void:
	redraw(true)
	accept_event()


# The rendered view as an image (the --shot flag).
func image() -> Image:
	return sharp.get_texture().get_image()
