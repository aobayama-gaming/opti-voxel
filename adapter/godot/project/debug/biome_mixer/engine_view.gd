extends VBoxContainer
## The mixer's view: the engine's maps of the world being edited (ClimateReportNode.preview: the world sampled by the
## engine's own cascade on the CPU, each column traced down to its ground), never a copy of the terrain. Biome mode: the
## biome alone, its shaded relief. World mode: the biomes' debug colours blended by their weights, shaded by the relief;
## the chart reads the same samples' fields. An edit draws a small map at once (FAST points a side), then a sharp one
## (SHARP, on a thread: the window never waits on it) once the edits pause. "3D" shows the game's own terrain on the
## same world (terrain_view.gd: a VoxelTerrain, restarted on each edit). "Open in game" runs the game on the world.

signal view_changed  # the extent or the centre moved: draw again
signal open_game

const FAST := 48
const SHARP := 192
# The pause after an edit before the sharp map (seconds).
const SETTLE := 0.35
# The world files' directory: a preview's relative paths (its engine) resolve from it.
const DIR := "res://climate/"
const TerrainView := preload("res://debug/biome_mixer/terrain_view.gd")

var extent_km := 20.0
var centre_m := Vector2.ZERO
var last := {}  # the last map drawn: ClimateReportNode.preview's dictionary
var _json := ""
var _world_mode := false
var _rect := TextureRect.new()
var _settle := Timer.new()
var _extent := SpinBox.new()
var _x := SpinBox.new()
var _z := SpinBox.new()
var _info := Label.new()
var _three := CheckButton.new()
var _view3d := TerrainView.new()
var _framed := Vector3(-1, 0, 0)  # the extent and centre the 3D camera was fitted to
var _serial := 0  # one per map asked: a sharp map from an older one is dropped
var _thread: Thread
var _sharp_again := false  # a sharp map was asked while one ran


func _ready() -> void:
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	_rect.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_rect.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
	add_child(_rect)
	_view3d.visible = false
	_view3d.status_changed.connect(_show_info)
	add_child(_view3d)
	var bar := HFlowContainer.new()
	add_child(bar)
	_spin(bar, _extent, "Across km", 0.5, 20000.0, extent_km)
	_spin(bar, _x, "Centre x km", -100000.0, 100000.0, 0.0)
	_spin(bar, _z, "z km", -100000.0, 100000.0, 0.0)
	_extent.value_changed.connect(func(v: float) -> void:
		extent_km = v
		view_changed.emit())
	_x.value_changed.connect(func(v: float) -> void:
		centre_m.x = v * 1000.0
		view_changed.emit())
	_z.value_changed.connect(func(v: float) -> void:
		centre_m.y = v * 1000.0
		view_changed.emit())
	_three.text = "3D"
	_three.button_pressed = true
	_three.toggled.connect(func(p_on: bool) -> void:
		_show(last)
		_show_terrain()
		if not p_on:
			_settle.start())
	bar.add_child(_three)
	var game := Button.new()
	game.text = "Open in game"
	game.tooltip_text = "Runs the game on this world (the engine's terrain in 3D, with its trees, rocks and features)."
	game.pressed.connect(func() -> void: open_game.emit())
	bar.add_child(game)
	_info.add_theme_color_override("font_color", Color(0.7, 0.72, 0.76))
	bar.add_child(_info)
	_settle.one_shot = true
	_settle.wait_time = SETTLE
	_settle.timeout.connect(_sharp_async)
	add_child(_settle)


func _spin(p_bar: Container, p_box: SpinBox, p_label: String, p_lo: float, p_hi: float, p_value: float) -> void:
	var l := Label.new()
	l.text = p_label
	p_bar.add_child(l)
	p_box.min_value = p_lo
	p_box.max_value = p_hi
	p_box.step = 0.1
	p_box.value = p_value
	p_box.custom_minimum_size.x = 90
	p_bar.add_child(p_box)


# The view of p_json (a world file's text): small now, sharp after SETTLE (p_sharp: sharp now, a shot or a check).
# Returns the map drawn now (its error if the engine refused the world).
func show_world(p_json: String, p_world_mode: bool, p_sharp: bool = false) -> Dictionary:
	_json = p_json
	_world_mode = p_world_mode
	_serial += 1
	_settle.stop()
	_draw_map(SHARP if p_sharp else FAST)
	_show_terrain()
	if not p_sharp and not _three.button_pressed and last.get("error", "") == "" and is_inside_tree():
		_settle.start()  # the sharp map, for the 2D view only (in 3D the small one gives the info and the centre)
	return last


func is_3d() -> bool:
	return _three.button_pressed


func set_extent(p_km: float) -> void:
	extent_km = p_km
	_extent.set_value_no_signal(p_km)


func _draw_map(p_size: int) -> void:
	if _json != "":
		_show(draw(_json, extent_km, p_size, centre_m))


# The sharp map on a thread; its result is shown if nothing was asked since.
func _sharp_async() -> void:
	if _thread != null:
		_sharp_again = true
		return
	var serial := _serial
	var args := [_json, extent_km, centre_m]
	_thread = Thread.new()
	_thread.start(func() -> void:
		var map := draw(args[0], args[1], SHARP, args[2])
		_sharp_done.call_deferred(map, serial))


func _sharp_done(p_map: Dictionary, p_serial: int) -> void:
	_thread.wait_to_finish()
	_thread = null
	if p_serial == _serial:
		_show(p_map)
	if _sharp_again:
		_sharp_again = false
		_sharp_async()


func _show(p_map: Dictionary) -> void:
	if p_map.is_empty():
		return
	last = p_map
	if String(last.error) != "":
		_info.text = "The engine refuses the world: " + String(last.error)
		return
	var img: Image = last.biomes if _world_mode else last.shaded
	_rect.texture = ImageTexture.create_from_image(img)
	_rect.visible = not _three.button_pressed
	_view3d.visible = _three.button_pressed
	_show_info()


func _show_info() -> void:
	if last.get("error", "x") != "":
		return
	_info.text = "%d x %d points, %s m apart, %d us a point (the engine, CPU)" % [last.size, last.size, snappedf(last.spacing_m, 0.1), int(last.us_per_point)]
	if _three.button_pressed and _view3d.status != "":
		_info.text += "; " + _view3d.status


# The game's terrain on the last world, its camera fitted to a new extent or centre (its target on the map's centre).
func _show_terrain() -> void:
	if last.get("error", "x") != "":
		return
	var view := Vector3(extent_km, centre_m.x, centre_m.y)
	var alt: PackedFloat32Array = last.altitude_m
	var ground := alt[alt.size() / 2 + int(last.size) / 2] if alt.size() > 0 else 0.0
	_view3d.show_world(_json, DIR, view != _framed, extent_km, centre_m, ground)
	_framed = view


func _exit_tree() -> void:
	if _thread != null:
		_thread.wait_to_finish()


# The last world drawn again (the extent or the centre moved): small now, sharp after SETTLE.
func redraw() -> void:
	show_world(_json, _world_mode)


# The engine's maps of p_json (a world file's text, its paths from DIR), p_size points a side across p_extent_km round
# p_centre_m: ClimateReportNode.preview's dictionary (error "" when drawn).
static func draw(p_json: String, p_extent_km: float, p_size: int, p_centre_m: Vector2 = Vector2.ZERO) -> Dictionary:
	var map: Dictionary = ClassDB.class_call_static("ClimateReportNode", "preview", p_json, DIR, p_extent_km, p_size, true, p_centre_m)
	map["error"] = String(map.get("error", ""))
	return map


# The last map's fields p_fx and p_fy (indices in its field_names) at every point: the chart's cloud.
func samples(p_fx: int, p_fy: int) -> PackedVector2Array:
	var out := PackedVector2Array()
	if last.get("error", "x") != "":
		return out
	var nf: int = last.field_names.size()
	var fields: PackedFloat32Array = last.fields
	if p_fx < 0 or p_fy < 0 or p_fx >= nf or p_fy >= nf:
		return out
	out.resize(fields.size() / nf)
	for i in out.size():
		out[i] = Vector2(fields[i * nf + p_fx], fields[i * nf + p_fy])
	return out


func image() -> Image:
	if _three.button_pressed:
		return _view3d.image()
	return _rect.texture.get_image() if _rect.texture != null else Image.create(1, 1, false, Image.FORMAT_RGB8)


# Until the 3D view's terrain has loaded the chunks in view (at most p_timeout_s): its load in ms, -1 if it didn't. Then
# p_restarts more restarts on the same world, timed the same (an edit's path): their loads appended.
func wait_terrain(p_timeout_s: float, p_restarts: int = 0) -> Array:
	var out := []
	for r in p_restarts + 1:
		if r > 0:
			_view3d.restart()
		var end := Time.get_ticks_msec() + int(p_timeout_s * 1000.0)
		while _view3d.loaded_ms < 0 and Time.get_ticks_msec() < end:
			await get_tree().process_frame
		out.append(_view3d.loaded_ms)
	return out
