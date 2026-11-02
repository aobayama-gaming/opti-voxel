extends Control
## The climate chart: two of a world's fields as axes, each biome a box, its centre +- span on the fields it names
## (the full axis on one it doesn't: it doesn't care), its whole core (Octaves.core of the span) filled. Under them the
## world's points: where the ground actually goes in the fields' space (the preview's samples, a density image).
## Drag a box to move its centre, its edge to change its span (both sides); a click selects a biome. The default
## biome has no box: it fills what the others leave.

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const World := preload("res://debug/biome_mixer/world.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")

signal placed(p_biome: int)  # a biome's centre or span moved (the world's "at" holds it)
signal selected(p_biome: int)

const EDGE_PX := 7.0

var world := {}
var names := []  # the world's field names (World.field_names), read on every draw and hover
var fx := 0  # the fields on x and y (world.fields indices)
var fy := 1
var lo := Vector2(-1.2, -1.2)  # the axes' ranges (field values)
var hi := Vector2(1.2, 1.2)
var density: Texture2D = null  # the world's points, binned over lo .. hi
var current := -1
var drag := {}  # {biome, what: "move" | "x" | "y", start (field values), at (the biome's at then)}


func _init() -> void:
	custom_minimum_size = Vector2(320, 320)
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	mouse_filter = Control.MOUSE_FILTER_STOP
	clip_contents = true


func set_world(p_world: Dictionary) -> void:
	world = p_world
	names = World.field_names(world)
	queue_redraw()


# Field values to pixels and back.
func to_px(p_v: Vector2) -> Vector2:
	return Vector2((p_v.x - lo.x) / (hi.x - lo.x) * size.x, (1.0 - (p_v.y - lo.y) / (hi.y - lo.y)) * size.y)


func to_value(p_px: Vector2) -> Vector2:
	return Vector2(lo.x + p_px.x / size.x * (hi.x - lo.x), lo.y + (1.0 - p_px.y / size.y) * (hi.y - lo.y))


# Biome p_b's box in field values: [min, max] per axis (the whole range where it names no such field), scaled by
# p_scale (1: its span, CORE: its whole core).
func box(p_b: int, p_scale: float) -> Rect2:
	var at: Dictionary = world.biomes[p_b].at
	var r := Rect2(lo, hi - lo)
	for axis in 2:
		var f := _axis_field(axis)
		if at.has(f):
			var s: float = at[f][1] * p_scale
			r.position[axis] = at[f][0] - s
			r.size[axis] = 2.0 * s
	return r


func _px_rect(p_r: Rect2) -> Rect2:
	var a := to_px(p_r.position)
	var b := to_px(p_r.end)
	return Rect2(Vector2(minf(a.x, b.x), minf(a.y, b.y)), (b - a).abs())


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Kinds.PLOT_BG)
	if density != null:
		draw_texture_rect(density, Rect2(Vector2.ZERO, size), false, Color(1, 1, 1, 0.8))
	var axis_col := Color(1, 1, 1, 0.15)
	draw_line(to_px(Vector2(0.0, lo.y)), to_px(Vector2(0.0, hi.y)), axis_col)
	draw_line(to_px(Vector2(lo.x, 0.0)), to_px(Vector2(hi.x, 0.0)), axis_col)
	if world.is_empty():
		return
	var font := get_theme_default_font()
	var labels := []  # where the names went: a biome with the same box (apart on another field) goes under
	for b in world.biomes.size() - 1:
		var col := World.debug_color(world.biomes[b])
		var outer := _px_rect(box(b, 1.0))
		var core := _px_rect(box(b, Octaves.core))
		var sel: bool = b == current
		draw_rect(core, Color(col, 0.28 if sel else 0.16))
		draw_rect(core, Color(col, 0.9), false, 2.0 if sel else 1.0)
		draw_rect(outer, Color(col, 0.75 if sel else 0.4), false, 2.0 if sel else 1.0)
		var at := core.get_center() - Vector2(30, -4)
		while labels.any(func(l): return l.distance_to(at) < 14.0):
			at.y += 14.0
		labels.append(at)
		draw_string(font, at, world.biomes[b].name, HORIZONTAL_ALIGNMENT_CENTER, 60, 12, Color(1, 1, 1, 0.95 if sel else 0.7))
	var d: Dictionary = world.biomes.back()
	draw_string(font, Vector2(6, size.y - 6), "%s (default): fills the rest" % d.name, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(1, 1, 1, 0.6))
	draw_string(font, Vector2(size.x - 160, size.y - 6), "x %s  %.1f .. %.1f" % [_axis_field(0), lo.x, hi.x], HORIZONTAL_ALIGNMENT_RIGHT, 154, 11, Color(1, 1, 1, 0.6))
	draw_string(font, Vector2(6, 14), "y %s  %.1f .. %.1f" % [_axis_field(1), lo.y, hi.y], HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(1, 1, 1, 0.6))


# The biome under p_px and what a drag there does: an outer edge resizes that axis, inside moves the box. Boxes drawn
# later (higher in the list) win, the selected one first.
func _hit(p_px: Vector2) -> Dictionary:
	var order := range(world.biomes.size() - 1)
	order.reverse()
	if current >= 0 and current < world.biomes.size() - 1:
		order.erase(current)
		order.push_front(current)
	for b in order:
		var r := _px_rect(box(b, 1.0))
		if not r.grow(EDGE_PX).has_point(p_px):
			continue
		var near_x := absf(p_px.x - r.position.x) < EDGE_PX or absf(p_px.x - r.end.x) < EDGE_PX
		var near_y := absf(p_px.y - r.position.y) < EDGE_PX or absf(p_px.y - r.end.y) < EDGE_PX
		if near_x and _names_axis(b, 0):
			return {"biome": b, "what": "x"}
		if near_y and _names_axis(b, 1):
			return {"biome": b, "what": "y"}
		if r.has_point(p_px):
			return {"biome": b, "what": "move"}
	return {}


func _axis_field(p_axis: int) -> String:
	var i := fx if p_axis == 0 else fy
	return names[i] if i < names.size() else ""


func _names_axis(p_b: int, p_axis: int) -> bool:
	return world.biomes[p_b].at.has(_axis_field(p_axis))


func _gui_input(p_event: InputEvent) -> void:
	if world.is_empty():
		return
	if p_event is InputEventMouseButton and p_event.button_index == MOUSE_BUTTON_LEFT:
		if p_event.pressed:
			var h := _hit(p_event.position)
			if h.is_empty():
				return
			current = h.biome
			selected.emit(current)
			drag = h
			drag.start = to_value(p_event.position)
			drag.at = world.biomes[h.biome].at.duplicate(true)
			queue_redraw()
		else:
			drag = {}
		accept_event()
	elif p_event is InputEventMouseMotion:
		if drag.is_empty():
			var h := _hit(p_event.position)
			mouse_default_cursor_shape = CURSOR_ARROW if h.is_empty() else (CURSOR_MOVE if h.what == "move" else (CURSOR_HSIZE if h.what == "x" else CURSOR_VSIZE))
			return
		var v := to_value(p_event.position)
		var at: Dictionary = world.biomes[drag.biome].at
		for axis in 2:
			var f := _axis_field(axis)
			if not drag.at.has(f):
				continue
			var c: float = drag.at[f][0]
			var s: float = drag.at[f][1]
			var dv: float = (v - drag.start)[axis]
			if drag.what == "move":
				at[f] = [snappedf(c + dv, 0.001), s]
			elif drag.what == ("x" if axis == 0 else "y"):
				# The edge follows the pointer: the span is the pointer's distance from the centre.
				at[f] = [c, snappedf(maxf(absf(v[axis] - c), 0.01), 0.001)]
		placed.emit(drag.biome)
		queue_redraw()
		accept_event()


# The world's points (the preview's samples: x, y field values per texel) binned over the axes' range into a density
# image, the range fitted to them first (their 1st .. 99th percentile, padded) when p_fit.
func set_samples(p_values: PackedVector2Array, p_fit: bool) -> void:
	if p_values.is_empty():
		density = null
		queue_redraw()
		return
	if p_fit:
		var xs := PackedFloat32Array()
		var ys := PackedFloat32Array()
		for v in p_values:
			xs.append(v.x)
			ys.append(v.y)
		xs.sort()
		ys.sort()
		var q := func(a: PackedFloat32Array, t: float) -> float: return a[clampi(int(t * (a.size() - 1)), 0, a.size() - 1)]
		var pad := Vector2(0.15, 0.15)
		lo = Vector2(q.call(xs, 0.01), q.call(ys, 0.01)) - pad
		hi = Vector2(q.call(xs, 0.99), q.call(ys, 0.99)) + pad
	var n := 96
	var bins := PackedFloat32Array()
	bins.resize(n * n)
	var peak := 0.0
	for v in p_values:
		var px := Vector2((v.x - lo.x) / (hi.x - lo.x), 1.0 - (v.y - lo.y) / (hi.y - lo.y))
		if px.x < 0.0 or px.x >= 1.0 or px.y < 0.0 or px.y >= 1.0:
			continue
		var i := int(px.y * n) * n + int(px.x * n)
		bins[i] += 1.0
		peak = maxf(peak, bins[i])
	var bytes := PackedByteArray()
	bytes.resize(n * n * 4)
	for i in n * n:
		bytes[4 * i] = 191
		bytes[4 * i + 1] = 204
		bytes[4 * i + 2] = 230
		bytes[4 * i + 3] = int(sqrt(bins[i] / maxf(peak, 1.0)) * 0.7 * 255.0)
	var img := Image.create_from_data(n, n, false, Image.FORMAT_RGBA8, bytes)
	if density is ImageTexture and density.get_size() == Vector2(n, n):
		density.update(img)
	else:
		density = ImageTexture.create_from_image(img)
	queue_redraw()
