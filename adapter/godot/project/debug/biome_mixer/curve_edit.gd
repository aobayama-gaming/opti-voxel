extends Control
## The Height curve node's editor: R(h), metres in (x) and out (y), through its points. Drag a point, double-click to
## add one, right-click a point to remove it. Drawn through compile.gd's curve_eval, the shader's curve. In spans the
## heights the curve's input reaches (compile.gd's curve_in), out what R makes of them: points beyond don't count.

const Compile := preload("res://debug/biome_mixer/compile.gd")
const Kinds := preload("res://debug/biome_mixer/kinds.gd")

signal points_changed(p_points: Array)

const PICK_PX := 7.0

var points: Array = []
var span := 1000.0  # in covers [-span, span] metres: the input's bound
var drag := -1
var view := Rect2(-1, -1, 2, 2)  # the metres shown (_range), kept until the points or the span change


func _init() -> void:
	custom_minimum_size = Vector2(260, 180)
	mouse_filter = Control.MOUSE_FILTER_STOP
	clip_contents = true


func set_points(p_points: Array, p_span: float) -> void:
	points = p_points.duplicate(true)
	set_span(p_span)


# The axes hold still under a drag (each step recompiles and comes back here) and fit again on its release.
func set_span(p_span: float) -> void:
	if drag < 0 or not is_equal_approx(span, maxf(p_span, 1.0)):
		span = maxf(p_span, 1.0)
		view = _range()
	queue_redraw()


# The metres shown: in (x) the input's span, out (y) R over it (and the points in it), both a little wider.
func _range() -> Rect2:
	var c := Compile.curve_segments(points)
	var lo := INF
	var hi := -INF
	for i in 65:
		var y := Compile.curve_eval(c, lerpf(-span, span, i / 64.0))
		lo = minf(lo, y)
		hi = maxf(hi, y)
	for p in points:
		if absf(p[0]) <= span:
			lo = minf(lo, p[1])
			hi = maxf(hi, p[1])
	var pad := maxf((hi - lo) * 0.05, 1.0)
	return Rect2(-span * 1.05, lo - pad, span * 2.1, hi - lo + 2.0 * pad)


func _to_px(p_m: Vector2) -> Vector2:
	var r := view
	return Vector2((p_m.x - r.position.x) / r.size.x * size.x, size.y - (p_m.y - r.position.y) / r.size.y * size.y)


func _to_m(p_px: Vector2) -> Vector2:
	var r := view
	return Vector2(r.position.x + p_px.x / size.x * r.size.x, r.position.y + (size.y - p_px.y) / size.y * r.size.y)


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Kinds.PLOT_BG)
	var r := view
	var x0 := r.position.x
	var x1 := r.end.x
	var grid := Color(1, 1, 1, 0.06)
	for axis in 2:
		var lo: float = r.position[axis]
		var hi: float = r.end[axis]
		var step := pow(10.0, floor(log(hi - lo) / log(10.0)))
		var g: float = ceil(lo / step) * step
		while g < hi:
			if axis == 0:
				draw_line(_to_px(Vector2(g, r.position.y)), _to_px(Vector2(g, r.end.y)), grid)
			else:
				draw_line(_to_px(Vector2(x0, g)), _to_px(Vector2(x1, g)), grid)
			g += step
	# No remap (out = in), and the input's ends.
	draw_line(_to_px(Vector2(x0, x0)), _to_px(Vector2(x1, x1)), Color(1, 1, 1, 0.25))
	for e in [-span, span]:
		draw_line(_to_px(Vector2(e, r.position.y)), _to_px(Vector2(e, r.end.y)), Color(1, 1, 1, 0.15))
	var c := Compile.curve_segments(points)
	var line := PackedVector2Array()
	for i in 129:
		var h := lerpf(x0, x1, i / 128.0)
		line.append(_to_px(Vector2(h, Compile.curve_eval(c, h))))
	draw_polyline(line, Kinds.PORT_COLORS[Kinds.FIELD], 2.0, true)
	var outside := 0
	for i in points.size():
		if absf(points[i][0]) > span and i != drag:
			outside += 1
			continue
		var p := _to_px(Vector2(points[i][0], points[i][1]))
		draw_circle(p, 4.5, Color(1.0, 0.8, 0.35) if i == drag else Color(0.95, 0.95, 0.95))
	var font := get_theme_default_font()
	var dim := Color(1, 1, 1, 0.5)
	draw_string(font, Vector2(4, 14), "out %s .. %s m" % [_fmt(r.position.y), _fmt(r.end.y)], HORIZONTAL_ALIGNMENT_LEFT, -1, 11, dim)
	draw_string(font, Vector2(4, size.y - 4), "in +-%s m (the input's bound)" % _fmt(span), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, dim)
	if outside > 0:
		draw_string(font, Vector2(4, 28), "%d points beyond the input: never reached" % outside, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, dim)


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
			set_span(span)
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
	set_span(span)
	points_changed.emit(points.duplicate(true))
