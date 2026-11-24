extends Window
## The world's weather as a time-lapse (docs/weather.md "Report"): the engine runs it on the CPU over the mixer's world
## (ClimateReportNode.weather_frames, on a thread) and maps the whole planet each game minute; a slider steps through
## the maps, the cover (grey, the rain blue) or the wind (hue: the way it blows, east red, north yellow-green, west cyan,
## south violet; brightness: its speed).

const MINUTES := 20 # maps at 0, 1, ... MINUTES game minutes after the spin-up
const WIDTH := 512

var _frames: Array = []
var _thread: Thread
var _image := TextureRect.new()
var _slider := HSlider.new()
var _layer := OptionButton.new()
var _info := Label.new()


func _init() -> void:
	title = "Weather time-lapse"
	size = Vector2i(WIDTH + 40, WIDTH / 2 + 140)
	close_requested.connect(hide)
	var box := VBoxContainer.new()
	box.set_anchors_preset(Control.PRESET_FULL_RECT)
	box.offset_left = 8
	box.offset_right = -8
	box.offset_top = 8
	box.offset_bottom = -8
	add_child(box)
	var bar := HBoxContainer.new()
	box.add_child(bar)
	_layer.add_item("Cover and rain")
	_layer.add_item("Wind")
	_layer.item_selected.connect(func(_i: int) -> void: _show())
	bar.add_child(_layer)
	_slider.min_value = 0
	_slider.max_value = MINUTES
	_slider.step = 1
	_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_slider.value_changed.connect(func(_v: float) -> void: _show())
	bar.add_child(_slider)
	_image.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_image.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_image.size_flags_vertical = Control.SIZE_EXPAND_FILL
	box.add_child(_image)
	_info.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(_info)


## Runs the weather of p_json (a world file's text, its relative paths from p_dir) and shows its maps once done.
func run(p_json: String, p_dir: String) -> void:
	if _thread != null:
		return
	_frames = []
	_image.texture = null
	_info.text = "Running %d game minutes of weather (the engine, CPU)..." % MINUTES
	var minutes := PackedFloat32Array()
	for m in MINUTES + 1:
		minutes.append(float(m))
	_thread = Thread.new()
	_thread.start(func() -> void:
		var r: Dictionary = ClassDB.class_call_static("ClimateReportNode", "weather_frames", p_json, p_dir, minutes, WIDTH)
		_done.call_deferred(r))


func _done(p_result: Dictionary) -> void:
	_thread.wait_to_finish()
	_thread = null
	if String(p_result.get("error", "")) != "":
		_info.text = "No weather: " + String(p_result.error)
		return
	_frames = p_result.frames
	var stats: Dictionary = p_result.get("stats", {})
	_info.set_meta("cost", "forcing %d ms, a tick %.2f ms" % [int(stats.get("forcing", {}).get("total_ms", 0)), float(stats.get("tick", {}).get("avg_ms", 0))])
	_show()


func _show() -> void:
	var i := int(_slider.value)
	if i >= _frames.size():
		return
	var f: Dictionary = _frames[i]
	_image.texture = ImageTexture.create_from_image(f.cover if _layer.selected == 0 else f.wind)
	_info.text = "Minute %d: cover %d%%, raining %.1f%% of the planet, wind up to %.1f m/s, %d storms so far (%s)" % [
		int(f.minutes), int(round(100.0 * float(f.mean_cover))), 100.0 * float(f.raining), float(f.max_wind_mps), int(f.storms),
		String(_info.get_meta("cost", ""))]


func _exit_tree() -> void:
	if _thread != null:
		_thread.wait_to_finish()
