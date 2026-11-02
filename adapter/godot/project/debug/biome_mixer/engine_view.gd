extends VBoxContainer
## The mixer's view: the engine's maps of the world being edited (ClimateReportNode.preview: the world sampled by the
## engine's own cascade on the CPU, each column traced down to its ground), never a copy of the terrain. Biome mode: the
## biome alone, its shaded relief. World mode: the biomes' debug colours blended by their weights, shaded by the relief;
## the chart reads the same samples' fields. An edit draws a small map at once (FAST points a side), then a sharp one
## (SHARP) once the edits pause. "Open in game" runs the game on the world (the real terrain, trees and all).

signal view_changed  # the extent or the centre moved: draw again
signal open_game

const FAST := 48
const SHARP := 192
# The pause after an edit before the sharp map (seconds).
const SETTLE := 0.35
# The world files' directory: a preview's relative paths (its engine) resolve from it.
const DIR := "res://climate/"

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


func _ready() -> void:
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	_rect.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_rect.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
	add_child(_rect)
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
	var game := Button.new()
	game.text = "Open in game"
	game.tooltip_text = "Runs the game on this world (the engine's terrain in 3D, with its trees, rocks and features)."
	game.pressed.connect(func() -> void: open_game.emit())
	bar.add_child(game)
	_info.add_theme_color_override("font_color", Color(0.7, 0.72, 0.76))
	bar.add_child(_info)
	_settle.one_shot = true
	_settle.wait_time = SETTLE
	_settle.timeout.connect(func() -> void: _draw_map(SHARP))
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
	_settle.stop()
	_draw_map(SHARP if p_sharp else FAST)
	if not p_sharp and last.get("error", "") == "" and is_inside_tree():
		_settle.start()
	return last


func set_extent(p_km: float) -> void:
	extent_km = p_km
	_extent.set_value_no_signal(p_km)


func _draw_map(p_size: int) -> void:
	if _json == "":
		return
	last = ClassDB.class_call_static("ClimateReportNode", "preview", _json, DIR, extent_km, p_size, true, centre_m)
	if String(last.get("error", "")) != "":
		_info.text = "The engine refuses the world: " + String(last.error)
		return
	var img: Image = _composite(last) if _world_mode else last.shaded
	_rect.texture = ImageTexture.create_from_image(img)
	_info.text = "%d x %d points, %s m apart, %d us a point (the engine, CPU)" % [last.size, last.size, snappedf(last.spacing_m, 0.1), int(last.us_per_point)]


# The biomes' blend shaded by the relief (each pixel's colour times the shaded map's brightness).
func _composite(p_map: Dictionary) -> Image:
	var blend: Image = p_map.blend
	var shaded: Image = p_map.shaded
	var out := Image.create(blend.get_width(), blend.get_height(), false, Image.FORMAT_RGB8)
	for y in blend.get_height():
		for x in blend.get_width():
			var s := shaded.get_pixel(x, y).get_luminance()
			out.set_pixel(x, y, blend.get_pixel(x, y) * (0.3 + 0.9 * s))
	return out


# The last map's fields p_fx and p_fy (indices in its field_names) at every point: the chart's cloud.
func samples(p_fx: int, p_fy: int) -> PackedVector2Array:
	var out := PackedVector2Array()
	if last.get("error", "x") != "":
		return out
	var nf: int = last.field_names.size()
	var fields: PackedFloat32Array = last.fields
	if p_fx < 0 or p_fy < 0 or p_fx >= nf or p_fy >= nf:
		return out
	for i in fields.size() / nf:
		out.append(Vector2(fields[i * nf + p_fx], fields[i * nf + p_fy]))
	return out


# The relief's half range in metres (the curve editor's input span).
func relief_m() -> float:
	var alt: PackedFloat32Array = last.get("altitude_m", PackedFloat32Array())
	var r := 0.0
	for a in alt:
		r = maxf(r, absf(a))
	return r


func image() -> Image:
	return _rect.texture.get_image() if _rect.texture != null else Image.create(1, 1, false, Image.FORMAT_RGB8)
