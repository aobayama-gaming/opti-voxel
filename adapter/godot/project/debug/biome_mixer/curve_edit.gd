extends Control
## The Height curve node's editor: R(h), metres in (x) and out (y), through its points. Drag a point, double-click to
## add one, right-click a point to remove it. Drawn through compile.gd's curve_eval, the shader's curve.

const Compile := preload("res://debug/biome_mixer/compile.gd")
const Kinds := preload("res://debug/biome_mixer/kinds.gd")

signal points_changed(p_points: Array)

const PICK_PX := 7.0

var points: Array = []
var span := 1000.0  # the axes cover [-span, span] metres at least (the height's bound)
var drag := -1


func _init() -> void:
	custom_minimum_size = Vector2(260, 180)
	mouse_filter = Control.MOUSE_FILTER_STOP
	clip_contents = true


func set_points(p_points: Array, p_span: float) -> void:
	points = p_points.duplicate(true)
	span = maxf(p_span, 1.0)
	queue_redraw()


# The metres shown on each axis: the span and every point.
func _range() -> Vector2:
	var r := span
	for p in points:
		r = maxf(r, maxf(absf(p[0]), absf(p[1])))
	return Vector2(-r * 1.05, r * 1.05)


func _to_px(p_m: Vector2) -> Vector2:
	var r := _range()
	return Vector2((p_m.x - r.x) / (r.y - r.x) * size.x, size.y - (p_m.y - r.x) / (r.y - r.x) * size.y)


func _to_m(p_px: Vector2) -> Vector2:
	var r := _range()
	return Vector2(r.x + p_px.x / size.x * (r.y - r.x), r.x + (size.y - p_px.y) / size.y * (r.y - r.x))


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Kinds.PLOT_BG)
	var r := _range()
	var grid := Color(1, 1, 1, 0.06)
	var step := pow(10.0, floor(log(r.y) / log(10.0)))
	var g: float = ceil(r.x / step) * step
	while g < r.y:
		draw_line(_to_px(Vector2(g, r.x)), _to_px(Vector2(g, r.y)), grid)
		draw_line(_to_px(Vector2(r.x, g)), _to_px(Vector2(r.y, g)), grid)
		g += step
	draw_line(_to_px(Vector2(r.x, r.x)), _to_px(Vector2(r.y, r.y)), Color(1, 1, 1, 0.25))
	var c := Compile.curve_segments(points)
	var line := PackedVector2Array()
	for i in 129:
		var h := lerpf(r.x, r.y, i / 128.0)
		line.append(_to_px(Vector2(h, Compile.curve_eval(c, h))))
	draw_polyline(line, Kinds.PORT_COLORS[Kinds.FIELD], 2.0, true)
	for i in points.size():
		var p := _to_px(Vector2(points[i][0], points[i][1]))
		draw_circle(p, 4.5, Color(1.0, 0.8, 0.35) if i == drag else Color(0.95, 0.95, 0.95))
	var font := get_theme_default_font()
	draw_string(font, Vector2(4, 14), "out %s m" % _fmt(r.y), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(1, 1, 1, 0.5))
	draw_string(font, Vector2(size.x - 70, size.y - 4), "in %s m" % _fmt(r.y), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(1, 1, 1, 0.5))


static func _fmt(p_m: float) -> String:
	return "%.0f" % p_m if absf(p_m) >= 10.0 else "%.1f" % p_m


func _pick(p_px: Vector2) -> int:
	for i in points.size():
		if _to_px(Vector2(points[i][0], points[i][1])).distance_to(p_px) < PICK_PX:
			return i
	return -1


func _gui_input(p_event: InputEvent) -> void:
	if p_event is InputEventMouseButton:
		var mb := p_event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT and mb.double_click:
			var m := _to_m(mb.position)
			points.append([m.x, m.y])
			points.sort_custom(func(a, b): return a[0] < b[0])
			_changed()
		elif mb.button_index == MOUSE_BUTTON_LEFT:
			drag = _pick(mb.position) if mb.pressed else -1
			queue_redraw()
		elif mb.button_index == MOUSE_BUTTON_RIGHT and mb.pressed:
			var i := _pick(mb.position)
			if i >= 0:
				points.remove_at(i)
				_changed()
		accept_event()
	elif p_event is InputEventMouseMotion and drag >= 0:
		var m := _to_m((p_event as InputEventMouseMotion).position)
		# A point stays between its neighbours (the curve's x increase).
		var lo: float = points[drag - 1][0] + 0.01 if drag > 0 else -INF
		var hi: float = points[drag + 1][0] - 0.01 if drag + 1 < points.size() else INF
		points[drag] = [clampf(m.x, lo, hi), m.y]
		_changed()
		accept_event()


func _changed() -> void:
	queue_redraw()
	points_changed.emit(points.duplicate(true))
