extends Node3D
## The weather debugger (WeatherDebugNode over WeatherDebug, docs/weather.md "Debugger"): a world's weather alone, no
## terrain. The planet is the weather's cells coloured by a field at a level (or a column field: the ground and its
## fluxes); click a cell for its column (every level's values) and the section through it (its face's row, the levels up).
## Run: godot --path project res://debug/weather_debug/weather_debug.tscn -- [--world=res://worlds/ocean_test.planet.json]
##   [--spin_up=<s>, the world's by default] [--run: running from the start] [--speed=<x>: game seconds per real
##   second, 20 at most: past what the worker keeps up with, it ticks flat out] [--field=<name>] [--level=<k>] [--pick=<face,i,j>] [--shot=<png>
##   --ticks=<n>: n ticks, a screenshot, quit] [--profiles=<m,m,...>: headless, the profile tables (all, sea, land) at
##   those game minutes after the start, printed, then quit] [--trace=<m>: headless, one line a game minute for m minutes
##   (the strongest wind and its cell, the mean, the strongest w, theta's departure, qv, qc, the cover; the strongest
##   wind's column the first time it passes 25 m/s), then quit; --trace_s=<s> another interval, --trace_from=<m> a later
##   first line]
## Keys: space run / pause, N one tick, M ten, up / down the level, left / right the field, W the wind, R the range.
## Mouse: left click picks a cell, right drag turns the planet, the wheel zooms.

const TABLE_FIELDS := ["temperature", "theta - sounding", "qv", "qs (saturation)", "relative humidity", "qc (cloud)",
		"qr (rain)", "cloudy share", "Q1", "sigma_s (spread)", "updraft qc", "w (up)", "wind speed", "buoyancy"]
const TABLE_SHORT := ["T C", "th'", "qv", "qs", "RH", "qc", "qr", "share", "Q1", "sig", "upd qc", "w", "|u|", "b"]

var sim: WeatherDebugNode
var grid := {}
var fields: Array = []
var field := 0
var level := 0
var running := false
var speed := 1.0
var shown_s := -1.0
var picked := Vector3i(-1, -1, -1)
var auto_range := true
var vmin := 0.0
var vmax := 1.0
var show_wind := true
var yaw := 0.6
var pitch := 0.4
var distance := 5000.0
var args := {}
var shot_ticks := -1

var planet: MeshInstance3D
var wind_mesh: MeshInstance3D
var camera: Camera3D
var field_mat: ShaderMaterial
var section_rect: TextureRect
var legend_rect: TextureRect
var status: Label
var legend_label: Label
var table: RichTextLabel
var field_menu: OptionButton
var level_slider: HSlider
var level_label: Label
var world_edit: LineEdit
var spin_edit: SpinBox


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv: PackedStringArray = a.substr(2).split("=", true, 1)
			args[kv[0]] = kv[1] if kv.size() > 1 else "true"
	speed = float(args.get("speed", str(speed)))
	running = args.has("run")
	fields = WeatherDebugNode.fields()
	_build_scene()
	_build_ui()
	if args.has("field"):
		for f in fields.size():
			if fields[f]["name"] == args["field"]:
				field = f
	level = int(args.get("level", "0"))
	field_menu.select(field)
	_open(args.get("world", "res://worlds/ocean_test.planet.json"), float(args.get("spin_up", "-1")))
	if args.has("pick"):
		var p: PackedStringArray = String(args["pick"]).split(",")
		if p.size() == 3:
			picked = Vector3i(int(p[0]), int(p[1]), int(p[2]))
	if args.has("shot"):
		shot_ticks = int(args.get("ticks", "0"))


func _build_scene() -> void:
	sim = WeatherDebugNode.new()
	add_child(sim)
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color(0.07, 0.07, 0.09)
	add_child(env)
	camera = Camera3D.new()
	camera.far = 1e6
	camera.near = 1.0
	add_child(camera)
	planet = MeshInstance3D.new()
	field_mat = ShaderMaterial.new()
	field_mat.shader = load("res://debug/weather_debug/weather_field.gdshader")
	add_child(planet)
	wind_mesh = MeshInstance3D.new()
	var wind_mat := StandardMaterial3D.new()
	wind_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	wind_mat.albedo_color = Color(1, 1, 1, 0.9)
	wind_mesh.material_override = wind_mat
	add_child(wind_mesh)


func _labelled(box: Container, text: String, control: Control) -> void:
	var row := HBoxContainer.new()
	var l := Label.new()
	l.text = text
	l.custom_minimum_size.x = 70
	row.add_child(l)
	control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(control)
	box.add_child(row)


func _button(box: Container, text: String, fn: Callable) -> void:
	var b := Button.new()
	b.text = text
	b.pressed.connect(fn)
	box.add_child(b)


func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var left := PanelContainer.new()
	left.position = Vector2(8, 8)
	left.custom_minimum_size = Vector2(330, 0)
	layer.add_child(left)
	var box := VBoxContainer.new()
	left.add_child(box)
	world_edit = LineEdit.new()
	_labelled(box, "world", world_edit)
	spin_edit = SpinBox.new()
	spin_edit.min_value = -1
	spin_edit.max_value = 36000
	spin_edit.value = -1
	spin_edit.tooltip_text = "spin-up (game s); -1: the world file's"
	_labelled(box, "spin-up s", spin_edit)
	var row := HBoxContainer.new()
	_button(row, "Open", func(): _open(world_edit.text, spin_edit.value))
	_button(row, "Run/Pause", func(): running = not running)
	_button(row, "+1", func(): _step(1))
	_button(row, "+10", func(): _step(10))
	box.add_child(row)
	var speed_box := SpinBox.new()
	speed_box.min_value = 0.1
	speed_box.max_value = 20
	speed_box.step = 0.1
	speed_box.value = speed
	speed_box.value_changed.connect(func(v): speed = v)
	_labelled(box, "speed", speed_box)
	field_menu = OptionButton.new()
	for f in fields:
		field_menu.add_item("%s%s" % [f["name"], "" if f["levels"] else "  (ground)"])
	field_menu.item_selected.connect(func(i): _set_field(i))
	_labelled(box, "field", field_menu)
	level_slider = HSlider.new()
	level_slider.step = 1
	level_slider.value_changed.connect(func(v): level = int(v); _refresh(true))
	_labelled(box, "level", level_slider)
	level_label = Label.new()
	box.add_child(level_label)
	var range_box := CheckBox.new()
	range_box.text = "auto range (R)"
	range_box.button_pressed = true
	range_box.toggled.connect(func(on): auto_range = on; _refresh(true))
	box.add_child(range_box)
	var wind_box := CheckBox.new()
	wind_box.text = "wind at the level (W)"
	wind_box.button_pressed = show_wind
	wind_box.toggled.connect(func(on): show_wind = on; _refresh(true))
	box.add_child(wind_box)
	legend_rect = _flat_rect(Vector2(310, 14), true)
	box.add_child(legend_rect)
	legend_label = Label.new()
	box.add_child(legend_label)
	status = Label.new()
	status.autowrap_mode = TextServer.AUTOWRAP_WORD
	status.custom_minimum_size.x = 310
	box.add_child(status)
	var right := PanelContainer.new()
	right.anchor_left = 1.0
	right.anchor_right = 1.0
	right.offset_left = -640
	right.offset_right = -8
	right.offset_top = 8
	layer.add_child(right)
	var rbox := VBoxContainer.new()
	right.add_child(rbox)
	table = RichTextLabel.new()
	table.bbcode_enabled = true
	table.fit_content = true
	table.custom_minimum_size = Vector2(620, 0)
	table.add_theme_font_size_override("normal_font_size", 12)
	table.add_theme_font_size_override("mono_font_size", 12)
	rbox.add_child(table)
	section_rect = _flat_rect(Vector2(620, 160), false)
	rbox.add_child(section_rect)


func _flat_rect(size: Vector2, legend: bool) -> TextureRect:
	var r := TextureRect.new()
	r.custom_minimum_size = size
	r.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	r.stretch_mode = TextureRect.STRETCH_SCALE
	var m := ShaderMaterial.new()
	m.shader = load("res://debug/weather_debug/weather_flat.gdshader")
	m.set_shader_parameter("legend", legend)
	r.material = m
	var img := Image.create(1, 1, false, Image.FORMAT_RF)
	r.texture = ImageTexture.create_from_image(img)
	return r


func _open(path: String, spin_up: float) -> void:
	world_edit.text = path
	spin_edit.value = spin_up
	var error: String = sim.open(path, spin_up)
	if error != "":
		status.text = "refused: " + error
		push_error(error)
		return
	grid = sim.grid()
	var r: float = grid["radius_m"]
	planet.mesh = sim.planet_mesh(0.0)
	planet.material_override = field_mat
	distance = r * 2.6
	level_slider.max_value = int(grid["levels"]) - 1
	level = clampi(level, 0, int(grid["levels"]) - 1)
	level_slider.set_value_no_signal(level)
	shown_s = -1.0
	status.text = "forcing and spin-up..."


func _set_field(i: int) -> void:
	field = clampi(i, 0, fields.size() - 1)
	field_menu.select(field)
	_refresh(true)


func _step(n: int) -> void:
	running = false
	sim.step(n)
	_refresh(false)


func _process(delta: float) -> void:
	_place_camera()
	# Collects what the worker published (paused: the clock stays).
	sim.run(delta * speed if running else 0.0)
	if not sim.is_ready():
		return
	if args.has("profiles"):
		_profiles()
		return
	if args.has("trace"):
		_trace()
		return
	if shot_ticks >= 0:
		sim.step(shot_ticks)
		_refresh(true)
		shot_ticks = -1
		await get_tree().process_frame
		await get_tree().process_frame
		var img := get_viewport().get_texture().get_image()
		if img:
			img.save_png(args["shot"])
		print("WEATHER_DEBUG shot at game ", shown_s, " s: ", table.get_parsed_text().substr(0, 2000))
		get_tree().quit(0)
		return
	_refresh(false)


# The profile tables at each game minute of --profiles, printed between WEATHER_PROFILE markers; then quit.
func _profiles() -> void:
	var start: float = sim.stats().get("snapshot_game_s", 0.0)
	var mins := String(args["profiles"]).split(",")
	args.erase("profiles")
	for m in mins:
		while float(sim.stats().get("snapshot_game_s", 0.0)) < start + float(m) * 60.0 - 1e-3:
			sim.step(1)
		var st: Dictionary = sim.stats()
		print("WEATHER_PROFILE minute %s (game %.0f s, tick %.1f ms)" % [m, st.get("snapshot_game_s", 0.0), st.get("tick", {}).get("avg_ms", 0.0)])
		for c in 3:
			print("### ", ["all", "sea", "land"][c], "\n", sim.profile_table(c))
	print("WEATHER_PROFILE end")
	get_tree().quit(0)


# One line a game minute for --trace minutes, between WEATHER_TRACE markers (the drift's time series); then quit.
func _trace() -> void:
	var minutes := float(args["trace"])
	var every := float(args.get("trace_s", 60.0))
	var from := float(args.get("trace_from", 0.0))
	args.erase("trace")
	var shown := false
	var names := {}
	for n in fields.size():
		names[fields[n]["name"]] = n
	var levels: int = grid["levels"]
	var start: float = sim.stats().get("snapshot_game_s", 0.0)
	print("WEATHER_TRACE   min  game_s  tick_ms  max|u| (level, face, i, j)  mean|u|  max|w|  mean th'  mean qv  mean qc  cover")
	var lines := int(round((minutes - from) * 60.0 / every))
	for line in lines + 1:
		var m := from + float(line) * every / 60.0
		while float(sim.stats().get("snapshot_game_s", 0.0)) < start + m * 60.0 - 1e-3:
			sim.step(1)
		var st: Dictionary = sim.stats()
		var umax := 0.0
		var ulev := 0
		var uat := 0
		var usum := 0.0
		var wmax := 0.0
		var th := 0.0
		var qv := 0.0
		var qc := 0.0
		var count := 0
		for k in levels:
			var u: PackedFloat32Array = sim.slice(names["wind speed"], k)
			var w: PackedFloat32Array = sim.slice(names["w (up)"], k)
			var t: PackedFloat32Array = sim.slice(names["theta - sounding"], k)
			var v: PackedFloat32Array = sim.slice(names["qv"], k)
			var c: PackedFloat32Array = sim.slice(names["qc (cloud)"], k)
			for i in u.size():
				if u[i] > umax:
					umax = u[i]
					ulev = k
					uat = i
				usum += u[i]
				wmax = maxf(wmax, absf(w[i]))
				th += t[i]
				qv += v[i]
				qc += c[i]
			count += u.size()
		var cov: PackedFloat32Array = sim.slice(names["cover (max share)"], 0)
		var cover := 0.0
		for x in cov:
			cover += x
		var wd: int = grid["width"]
		var face := uat / (wd * int(grid["height"]))
		var cj := (uat / wd) % int(grid["height"])
		var ci := uat % wd
		print("WEATHER_TRACE %5.2f %7.0f %7.1f %7.2f (%2d, %d, %2d, %2d) %7.2f %7.2f %+8.3f %7.3f %7.4f %6.3f" % [m, st.get("snapshot_game_s", 0.0),
				st.get("tick", {}).get("avg_ms", 0.0), umax, ulev, face, ci, cj, usum / count, wmax, th / count, qv / count, qc / count, cover / cov.size()])
		if umax > 25.0 and not shown:
			shown = true
			var col: Array = sim.column(face, ci, cj)
			print("WEATHER_TRACE column (face %d, %d, %d), level: |u|, w, w below, theta', qv, qs, qc, buoyancy, pressure" % [face, ci, cj])
			for k in levels:
				print("WEATHER_TRACE   %2d %7.2f %7.2f %7.2f %+7.3f %7.3f %7.3f %7.4f %+8.4f %+9.3f" % [k, col[names["wind speed"]][k], col[names["w (up)"]][k],
						col[names["w (interface below)"]][k], col[names["theta - sounding"]][k], col[names["qv"]][k], col[names["qs (saturation)"]][k],
						col[names["qc (cloud)"]][k], col[names["buoyancy"]][k], col[names["pressure"]][k]])
	print("WEATHER_TRACE end")
	get_tree().quit(0)


func _refresh(force: bool) -> void:
	if grid.is_empty() or not sim.is_ready():
		return
	var st: Dictionary = sim.stats()
	var t: float = st.get("snapshot_game_s", 0.0)
	if t == shown_s and not force:
		return
	shown_s = t
	var f: Dictionary = fields[field]
	var w: int = grid["width"]
	var rows: int = int(grid["faces"]) * int(grid["height"])
	var values: PackedFloat32Array = sim.slice(field, level)
	if auto_range:
		_auto_range(values, f["signed"])
	var img := Image.create_from_data(w, rows, false, Image.FORMAT_RF, values.to_byte_array())
	field_mat.set_shader_parameter("field", ImageTexture.create_from_image(img))
	for m in [field_mat, legend_rect.material, section_rect.material]:
		m.set_shader_parameter("vmin", vmin)
		m.set_shader_parameter("vmax", vmax)
		m.set_shader_parameter("diverging", f["signed"])
	legend_label.text = "%s  %s .. %s %s" % [f["name"], _num(vmin), _num(vmax), f["unit"]]
	var faces: PackedFloat64Array = grid["faces_m"]
	level_label.text = ("level %d: %.0f to %.0f m (centre %.0f m)" % [level, faces[level], faces[level + 1], 0.5 * (faces[level] + faces[level + 1])]) if f["levels"] else "a ground field (no level)"
	_draw_wind()
	_status(st)
	_show_column()


func _auto_range(values: PackedFloat32Array, signed_field: bool) -> void:
	var lo := INF
	var hi := -INF
	for v in values:
		if not is_nan(v):
			lo = minf(lo, v)
			hi = maxf(hi, v)
	if lo > hi:
		lo = 0.0
		hi = 1.0
	if signed_field:
		var m := maxf(absf(lo), absf(hi))
		lo = -m
		hi = m
	if hi - lo < 1e-12:
		hi = lo + 1e-6
	vmin = lo
	vmax = hi


func _status(st: Dictionary) -> void:
	var tick: Dictionary = st.get("tick", {})
	status.text = "game %.1f s (%.2f days of %.0f s)   ticks %d\ntick %.1f ms avg, %.1f max   behind %d, dropped %d\nupdrafts %.0f%% of columns, top %.0f m   storms %d\nground day %.2f K / night %.2f K\ngrid %s, cells %.1f m, %d levels of %.1f m" % [
		st.get("snapshot_game_s", 0.0), st.get("snapshot_game_s", 0.0) / _day_s(st), _day_s(st), st.get("ticks", 0),
		tick.get("avg_ms", 0.0), tick.get("max_ms", 0.0), st.get("behind_ticks", 0), st.get("dropped_ticks", 0),
		100.0 * st.get("convecting", 0.0), st.get("convection_top_m", 0.0), st.get("storms", 0),
		st.get("ground_day_k", 0.0), st.get("ground_night_k", 0.0), st.get("grid", ""), st.get("cell_m", 0.0),
		st.get("levels", 0), (grid["faces_m"] as PackedFloat64Array)[1] if grid.has("faces_m") else 0.0] + ("\nRUNNING x%.1f" % speed if running else "\npaused (space)")


func _day_s(st: Dictionary) -> float:
	return 2.0 * PI / maxf(float(st.get("omega_rad_s", 0.0)), 1e-9)


func _draw_wind() -> void:
	if not show_wind or not fields[field]["levels"]:
		wind_mesh.mesh = null
		return
	# Just over the planet's cells: the arrows of the level shown, 6 s of travel each.
	var lines: PackedVector3Array = sim.wind_lines(level, 2.0, 6.0, 2)
	if lines.is_empty():
		return
	var im := ImmediateMesh.new()
	im.surface_begin(Mesh.PRIMITIVE_LINES)
	for i in range(0, lines.size(), 2):
		im.surface_set_color(Color.WHITE)
		im.surface_add_vertex(lines[i])
		im.surface_add_vertex(lines[i + 1])
	im.surface_end()
	wind_mesh.mesh = im


func _show_column() -> void:
	if picked.x < 0:
		table.text = "[i]click a cell for its column[/i]"
		section_rect.visible = false
		field_mat.set_shader_parameter("picked_uv", Vector2(-1, -1))
		return
	var w: int = grid["width"]
	var h: int = grid["height"]
	var rows: int = int(grid["faces"]) * h
	field_mat.set_shader_parameter("picked_uv", Vector2((picked.y + 0.5) / w, (picked.x * h + picked.z + 0.5) / rows))
	var col: Array = sim.column(picked.x, picked.y, picked.z)
	if col.is_empty():
		return
	var idx := []
	for name in TABLE_FIELDS:
		idx.append(_field_index(name))
	var faces: PackedFloat64Array = grid["faces_m"]
	var s := "[b]cell face %d (%d, %d)[/b]\n[code]" % [picked.x, picked.y, picked.z]
	s += "%6s" % "z m"
	for n in TABLE_SHORT:
		s += "%8s" % n
	s += "\n"
	for k in range(int(grid["levels"]) - 1, -1, -1):
		var line := "%6.0f" % (0.5 * (faces[k] + faces[k + 1]))
		for i in idx:
			line += "%8.3f" % (col[i] as PackedFloat32Array)[k]
		s += ("[color=yellow]%s[/color]\n" % line) if k == level else line + "\n"
	s += "[/code]\n"
	for f in fields.size():
		if not fields[f]["levels"]:
			s += "%s %s %s   " % [fields[f]["name"], _num((col[f] as PackedFloat32Array)[0]), fields[f]["unit"]]
	table.text = s
	var sec: PackedFloat32Array = sim.section(field, picked.x, picked.z)
	if sec.is_empty():
		return
	var sec_rows: int = sec.size() / w
	var img := Image.create_from_data(w, sec_rows, false, Image.FORMAT_RF, sec.to_byte_array())
	section_rect.texture = ImageTexture.create_from_image(img)
	section_rect.material.set_shader_parameter("picked_x", float(picked.y))
	section_rect.tooltip_text = "the face's row through the cell: x across, the levels up (the ground at the bottom)"
	section_rect.visible = true


# Four significant digits (GDScript's % has no %g).
func _num(v: float) -> String:
	if v == 0.0 or (absf(v) >= 0.01 and absf(v) < 1e5):
		return "%.4f" % v if absf(v) < 10.0 else "%.2f" % v
	return String.num_scientific(snappedf(v, pow(10.0, floorf(log(absf(v)) / log(10.0)) - 3)))


func _field_index(name: String) -> int:
	for f in fields.size():
		if fields[f]["name"] == name:
			return f
	return 0


func _place_camera() -> void:
	var dir := Vector3(cos(pitch) * cos(yaw), sin(pitch), cos(pitch) * sin(yaw))
	camera.position = dir * distance
	camera.look_at(Vector3.ZERO, Vector3.UP if absf(pitch) < 1.5 else Vector3.FORWARD)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and (event.button_mask & MOUSE_BUTTON_MASK_RIGHT):
		yaw += event.relative.x * 0.005
		pitch = clampf(pitch + event.relative.y * 0.005, -1.55, 1.55)
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			distance = maxf(distance * 0.9, float(grid.get("radius_m", 1000.0)) * 1.05)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			distance = minf(distance * 1.1, float(grid.get("radius_m", 1000.0)) * 20.0)
		elif event.button_index == MOUSE_BUTTON_LEFT:
			_pick(event.position)
	elif event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_SPACE: running = not running
			KEY_N: _step(1)
			KEY_M: _step(10)
			KEY_UP: level_slider.value = mini(level + 1, int(level_slider.max_value))
			KEY_DOWN: level_slider.value = maxi(level - 1, 0)
			KEY_RIGHT: _set_field(field + 1)
			KEY_LEFT: _set_field(field - 1)
			KEY_W: show_wind = not show_wind; _refresh(true)
			KEY_R: auto_range = not auto_range; _refresh(true)


func _pick(at: Vector2) -> void:
	var o := camera.project_ray_origin(at)
	var d := camera.project_ray_normal(at)
	var r: float = grid.get("radius_m", 0.0)
	var b := o.dot(d)
	var c := o.dot(o) - r * r
	var disc := b * b - c
	if disc < 0.0:
		return
	var t := -b - sqrt(disc)
	if t < 0.0:
		return
	picked = sim.pick(o + d * t)
	_refresh(true)
