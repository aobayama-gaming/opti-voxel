extends RefCounted
## The mixer's view options under the preview (quality, steps, sun, water, contours, the view's centre; in the World
## mode the map and the chart's extent): saved in the biome's "view" (the world's in the World mode). p_ctx is the
## mixer.

const World := preload("res://debug/biome_mixer/world.gd")


# The view's options under the preview, in p_ctx.view_box.
static func build(p_ctx: Object) -> void:
	var box: Container = p_ctx.view_box
	for c in box.get_children():
		box.remove_child(c)
		c.queue_free()
	var view: Dictionary = p_ctx.view_settings()
	var set_view := func(p_key: String, p_value: Variant, p_recompile: bool = false) -> void:
		view[p_key] = p_value
		if p_recompile:
			p_ctx.recompile(true)
		else:
			apply(p_ctx)
			p_ctx.preview.redraw(true)
		p_ctx.autosave.start()
	var q := OptionButton.new()
	for t in ["Full", "1/2", "1/3", "1/4"]:
		q.add_item(t)
	q.select(int(view.quality) - 1)
	q.tooltip_text = "Resolution once still (dragging draws at 1/3 at most)"
	q.item_selected.connect(func(i: int) -> void:
		p_ctx.preview.quality = i + 1
		set_view.call("quality", i + 1))
	p_ctx.preview.quality = int(view.quality)
	_item(box, "Quality", q)
	_item(box, "Steps", _spin(view.steps, 32, 2048, 1, func(v: float) -> void: set_view.call("steps", int(v))), "The most march steps per pixel")
	_item(box, "Step", _spin(view.step, 0.1, 2.0, 0.05, func(v: float) -> void: set_view.call("step", v)), "The march step over its safe bound: lower if thin ridges show holes")
	var sh := CheckBox.new()
	sh.button_pressed = view.shadows
	sh.toggled.connect(func(on: bool) -> void: set_view.call("shadows", on))
	_item(box, "Shadows", sh)
	_item(box, "Sun", _spin(view.sun_az, 0.0, 360.0, 5.0, func(v: float) -> void: set_view.call("sun_az", v)), "Azimuth, degrees")
	_item(box, "high", _spin(view.sun_el, 2.0, 90.0, 1.0, func(v: float) -> void: set_view.call("sun_el", v)), "Elevation, degrees")
	_item(box, "Exaggerate", _spin(view.exag, 0.1, 10.0, 0.1, func(v: float) -> void: set_view.call("exag", v, true)), "Heights times this (the view only)")
	var w := CheckBox.new()
	w.button_pressed = view.water_on
	w.toggled.connect(func(on: bool) -> void: set_view.call("water_on", on))
	_item(box, "Water", w)
	_item(box, "at m", _spin(view.water, -5000.0, 5000.0, 1.0, func(v: float) -> void: set_view.call("water", v)))
	_item(box, "Contours m", _spin(view.contour, 0.0, 1000.0, 1.0, func(v: float) -> void: set_view.call("contour", v)), "A line every this many metres (0: none)")
	var holder: Dictionary = p_ctx.world if p_ctx.world_mode() else p_ctx.model
	var origin: Array = holder.get("origin_m", [0.0, 0.0])
	holder.origin_m = origin
	_item(box, "Centre x m", _spin(origin[0], -1e9, 1e9, 100.0, func(v: float) -> void:
		holder.origin_m[0] = v
		p_ctx.recompile(false)), "Where in the world the preview sits (the notes and fields move under it)")
	_item(box, "z m", _spin(origin[1], -1e9, 1e9, 100.0, func(v: float) -> void:
		holder.origin_m[1] = v
		p_ctx.recompile(false)))
	if p_ctx.world_mode():
		var wv: Dictionary = p_ctx.world.view
		var map := CheckBox.new()
		map.button_pressed = bool(wv.map)
		map.toggled.connect(func(on: bool) -> void:
			wv.map = on
			p_ctx.preview.map = on
			p_ctx.preview.redraw()
			p_ctx.autosave.start())
		_item(box, "Map", map, "From above: the biomes' colours over the ground's shading (drag pans, the wheel zooms)")
		_item(box, "Chart km", _spin(float(wv.extent_m) / 1000.0, 10.0, 20000.0, 10.0, func(v: float) -> void:
			wv.extent_m = v * 1000.0
			p_ctx.resample(true)), "How far around the view's centre the chart samples the world")


static func _item(p_box: Container, p_label: String, p_control: Control, p_tip: String = "") -> void:
	var row := HBoxContainer.new()
	var l := Label.new()
	l.text = p_label
	row.add_child(l)
	row.add_child(p_control)
	row.tooltip_text = p_tip
	p_box.add_child(row)


static func _spin(p_value: float, p_min: float, p_max: float, p_step: float, p_set: Callable) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = p_min
	s.max_value = p_max
	s.step = p_step
	s.value = p_value
	s.custom_minimum_size.x = 80
	s.value_changed.connect(p_set)
	return s


# The view's settings as the preview's uniforms.
static func apply(p_ctx: Object) -> void:
	var v: Dictionary = p_ctx.view_settings()
	var az := deg_to_rad(float(v.sun_az))
	var el := deg_to_rad(float(v.sun_el))
	p_ctx.preview.set_uniforms({
		"u_max_steps": int(v.steps),
		# The slope bound is a worst case (every note at its steepest at once): twice its step still holds on
		# the imported biomes, at half the frame.
		"u_step": float(v.step) * 2.0,
		"u_shadows": bool(v.shadows),
		"u_sun": Vector3(cos(el) * sin(az), sin(el), cos(el) * cos(az)),
		"u_water_on": bool(v.water_on),
		"u_water": float(v.water),
		"u_contour": float(v.contour),
	})
