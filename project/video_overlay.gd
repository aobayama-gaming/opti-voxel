extends CanvasLayer

# Overlay for recording videos (`-- --overlay=video` in the player): a rolling frame-time graph in the top-left
# corner and a few lines a visitor can read (frame times, scene settings, live counts) instead of the dev HUD.
# The graph is fed every frame; the text is refreshed every REFRESH_S from get_debug_stats(), which walks every chunk.
# Record with a screen recorder: Godot's --write-movie runs on a fixed time step, so the graph would show nothing real.

const SAMPLES := 300        # frames in the graph: 5 s at 60 fps, one pixel each
const GRAPH_H := 56.0
const GRAPH_MAX_MS := 50.0  # the top of the graph; longer frames are clipped
const REFRESH_S := 0.25
const ID_VOXELS := 14       # voxels per chunk-id unit (half a LOD0 chunk): the root span is 14 * 2^lod_numbers

var terrain: Node
var camera: Camera3D
var frame_ms := PackedFloat32Array()
var head := 0               # next slot written in frame_ms
var filled := 0
var age := REFRESH_S
var last_cam_pos = null     # Vector3 at the last refresh, null before the first
var static_lines := ""
var stats_line: Label
var info: Label
var graph: Control

class Graph:
	extends Control
	var overlay

	func _draw() -> void:
		var o = overlay
		var h := size.y
		var lines := PackedVector2Array()
		var colors := PackedColorArray()
		for i in o.filled:
			# Oldest on the left: slot (head - filled + i), wrapped.
			var ms: float = o.frame_ms[(o.head - o.filled + i + SAMPLES) % SAMPLES]
			var x := float(SAMPLES - o.filled + i) + 0.5
			lines.append(Vector2(x, h))
			lines.append(Vector2(x, h - h * minf(ms, GRAPH_MAX_MS) / GRAPH_MAX_MS))
			colors.append(_color(ms))
		if not lines.is_empty():
			draw_multiline_colors(lines, colors)
		var font := get_theme_default_font()
		for g in [[1000.0 / 60.0, "60"], [1000.0 / 30.0, "30"]]:
			var y: float = h - h * g[0] / GRAPH_MAX_MS
			draw_dashed_line(Vector2(0, y), Vector2(size.x, y), Color(1, 1, 1, 0.45), 1.0, 4.0)
			draw_string(font, Vector2(size.x + 4, y + 4), g[1], HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(1, 1, 1, 0.7))

	func _color(ms: float) -> Color:
		if ms <= 1000.0 / 58.0:
			return Color(0.35, 0.85, 0.45)
		if ms <= 1000.0 / 30.0:
			return Color(0.95, 0.8, 0.3)
		return Color(0.95, 0.35, 0.3)

func setup(p_terrain: Node, p_camera: Camera3D) -> void:
	terrain = p_terrain
	camera = p_camera

func _ready() -> void:
	frame_ms.resize(SAMPLES)
	var panel := PanelContainer.new()
	panel.position = Vector2(12, 12)
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0, 0, 0, 0.55)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(10)
	panel.add_theme_stylebox_override("panel", style)
	add_child(panel)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 6)
	panel.add_child(box)
	stats_line = _label(15)
	box.add_child(stats_line)
	graph = Graph.new()
	graph.overlay = self
	graph.custom_minimum_size = Vector2(SAMPLES + 22, GRAPH_H)  # room for the 60 / 30 labels
	box.add_child(graph)
	info = _label(13)
	box.add_child(info)
	static_lines = _static_lines()

func _label(p_size: int) -> Label:
	var l := Label.new()
	var font := SystemFont.new()
	font.font_names = PackedStringArray(["JetBrains Mono", "DejaVu Sans Mono", "monospace"])
	l.add_theme_font_override("font", font)
	l.add_theme_font_size_override("font_size", p_size)
	l.add_theme_color_override("font_color", Color(1, 1, 1))
	l.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.8))
	l.add_theme_constant_override("shadow_offset_x", 1)
	l.add_theme_constant_override("shadow_offset_y", 1)
	return l

# The settings: they don't change during a run.
func _static_lines() -> String:
	var vp := get_viewport()
	var res := vp.get_visible_rect().size
	var res_text := "%dx%d" % [res.x, res.y]
	if vp.scaling_3d_mode != Viewport.SCALING_3D_MODE_BILINEAR or not is_equal_approx(vp.scaling_3d_scale, 1.0):
		res_text += " (3D at %d%%)" % roundi(vp.scaling_3d_scale * 100.0)
	var vs: float = terrain.voxel_size
	var lods: int = terrain.lod_numbers
	var span_m := float(ID_VOXELS) * pow(2.0, lods) * vs
	var front: float = terrain.lod_pixel_error
	var back: float = terrain.lod_pixel_error_back
	var error_text := "LOD error %s px" % _num(front)
	if back > 0.0 and not is_equal_approx(back, front):
		error_text = "LOD error %s px ahead, %s px behind" % [_num(front), _num(back)]
	var scene := []
	var trees: int = terrain.archetype_trees
	if trees > 0:
		scene.append("%s trees" % _thousands(trees))
	scene.append("view %s" % _distance(terrain.get_load_radius()))
	return "\n".join([
		"%s · %s" % [RenderingServer.get_video_adapter_name(), res_text],
		"voxel %s · %d LODs over %s" % [_distance(vs), lods, _distance(span_m)],
		error_text,
		" · ".join(scene),
	])

func _process(delta: float) -> void:
	frame_ms[head] = delta * 1000.0
	head = (head + 1) % SAMPLES
	filled = mini(filled + 1, SAMPLES)
	graph.queue_redraw()
	age += delta
	if age < REFRESH_S or not visible:
		return
	_refresh(age)
	age = 0.0

func _refresh(p_elapsed: float) -> void:
	var sorted := frame_ms.slice(0, filled) if filled < SAMPLES else frame_ms.duplicate()
	sorted.sort()
	var p50: float = sorted[sorted.size() >> 1]
	stats_line.text = "%d fps   median %.1f ms   worst %.1f ms" % [
		Engine.get_frames_per_second(), p50, sorted[sorted.size() - 1]]

	var s: Dictionary = terrain.get_debug_stats()
	var chunks: Dictionary = s.get("chunks", {})
	var pos := camera.global_position
	var speed := 0.0
	if last_cam_pos != null:
		speed = pos.distance_to(last_cam_pos) / p_elapsed
	last_cam_pos = pos
	var altitude: float = terrain.get_sdf_at(pos)
	var load_text := "all loaded" if s.get("load_complete", false) else "streaming"
	var first_ms: float = s.get("ms_until_loaded", -1.0)
	if first_ms >= 0.0:
		load_text += " (first load %.1f s)" % (first_ms / 1000.0)
	info.text = static_lines + "\n" + "\n".join([
		"altitude %s · %d km/h" % [_distance(altitude), roundi(speed * 3.6)],
		"%.1f M triangles · %s chunks · %s" % [
			float(chunks.get("live_triangles", 0)) / 1e6, _thousands(int(chunks.get("loaded", 0))), load_text],
	])

func _num(v: float) -> String:
	return str(int(v)) if is_equal_approx(v, roundf(v)) else "%.1f" % v

func _distance(m: float) -> String:
	if absf(m) < 1.0:
		return "%d cm" % roundi(m * 100.0)
	if absf(m) < 1000.0:
		return "%d m" % roundi(m)
	return "%.1f km" % (m / 1000.0)

func _thousands(n: int) -> String:
	var s := str(absi(n))
	var out := ""
	while s.length() > 3:
		out = "," + s.right(3) + out
		s = s.left(s.length() - 3)
	return ("-" if n < 0 else "") + s + out
