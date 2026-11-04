extends RefCounted
## A graph node's GraphNode, built from the engine's kind (EngineInfo.kind): a port per input (a mix's rows carry their
## volume, mute and solo), one output, then a row per parameter the node has (an optional one can be taken out) and a
## menu adding the optional ones it hasn't. Edits go through the mixer (p_ctx.set_params): the node, as world.json holds
## it, is the truth; the UI only shows it.

const EngineInfo := preload("res://debug/biome_mixer/engine_info.gd")
const Graph := preload("res://debug/biome_mixer/graph.gd")
const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const CurveEdit := preload("res://debug/biome_mixer/curve_edit.gd")

const LABEL_W := 84.0
# A surface colour the biome doesn't set yet starts the picker at these (the engine's procedural ground otherwise).
const SURFACE := {"flat": "#5c7a3a", "steep": "#6b625a"}


# A shape node's phi(n) over n in [-1, 1], the engine's (EngineInfo.shape_phi), or why the engine refuses the shape.
class PhiGraph:
	extends Control
	var node := {}

	func _init() -> void:
		custom_minimum_size = Vector2(200, 70)

	func _draw() -> void:
		draw_rect(Rect2(Vector2.ZERO, size), Kinds.PLOT_BG)
		var ns := PackedFloat32Array()
		for i in 65:
			ns.append(lerpf(-1.0, 1.0, i / 64.0))
		var r := EngineInfo.shape_phi(node, ns)
		var font := get_theme_default_font()
		if String(r.error) != "":
			draw_string(font, Vector2(4, 14), String(r.error), HORIZONTAL_ALIGNMENT_LEFT, size.x - 8, 10, Color(1.0, 0.5, 0.45))
			return
		var ys: PackedFloat32Array = r.phi
		var lo := -1.0
		var hi := 1.0
		for y in ys:
			lo = minf(lo, y)
			hi = maxf(hi, y)
		var line := PackedVector2Array()
		for i in ys.size():
			line.append(Vector2(i / 64.0 * size.x, size.y - (ys[i] - lo) / (hi - lo) * size.y))
		var zero := size.y - (0.0 - lo) / (hi - lo) * size.y
		draw_line(Vector2(0, zero), Vector2(size.x, zero), Color(1, 1, 1, 0.15))
		draw_polyline(line, Kinds.LINK, 2.0, true)
		draw_string(font, Vector2(4, 12), "phi(n), n in [-1, 1]", HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(1, 1, 1, 0.45))


static func fmt_m(p_m: float) -> String:
	if p_m >= 1000.0:
		return "%s km" % String.num(p_m / 1000.0, 1 if p_m < 100000.0 else 0)
	return "%s m" % String.num(p_m, 2 if p_m < 1.0 else 1)


static func title(p_node: Dictionary) -> String:
	var t := Kinds.title(p_node.kind)
	match String(p_node.kind):
		"note":
			t += " %s" % fmt_m(float(p_node.get("wavelength_m", 0.0)))
		"shape":
			t += " %s" % ("coeffs" if p_node.has("coeffs") else String(p_node.get("shape", "plain")))
		"warp":
			t += " %s" % fmt_m(float(p_node.get("amount_m", 0.0)))
	return "%s  (%s)" % [t, p_node.id]


static func build(p_node: Dictionary, p_ctx: Object) -> GraphNode:
	var gn := GraphNode.new()
	var id := String(p_node.id)
	gn.name = id
	var pos: Array = p_node.get("pos", [0.0, 0.0])
	gn.position_offset = Vector2(float(pos[0]), float(pos[1]))
	gn.tooltip_text = Kinds.hint(p_node.kind)
	var ins := Graph.input_count(p_node)
	var out := Graph.has_output(p_node)
	for i in maxi(ins, 1 if out else 0):
		var out_label := "out" if i == 0 and out else ""
		if p_node.kind == "mix" and i < Graph.rows(p_node).size():
			gn.add_child(_mix_row(id, p_node, i, p_ctx, out_label))
		else:
			gn.add_child(_port_row(Graph.input_name(p_node, i) if i < ins else "", out_label))
		gn.set_slot(i, i < ins, 0, Kinds.LINK, i == 0 and out, 0, Kinds.LINK)
	var absent := []
	for p in EngineInfo.kind(p_node.kind).get("params", []):
		if p.type == "rows":
			continue
		if p_node.has(p.name):
			gn.add_child(_param_row(id, p_node, p, p_ctx))
		else:
			absent.append(p.name)
	if p_node.kind == "shape":
		var phi := PhiGraph.new()
		gn.add_child(phi)
		gn.set_meta("phi", phi)
	elif p_node.kind == "output":
		for key in ["flat", "steep"]:
			gn.add_child(_surface_row(key, p_ctx))
	if not absent.is_empty():
		var add := MenuButton.new()
		add.text = "+ parameter"
		add.flat = false
		add.tooltip_text = "Add one of the kind's optional parameters (the engine's default until it is set)"
		for a in absent:
			add.get_popup().add_item(a)
		add.get_popup().id_pressed.connect(func(i: int) -> void:
			var name: String = absent[i]
			p_ctx.set_params(id, {name: Graph.default_value(EngineInfo.param(p_node.kind, name), p_ctx.world.fields)}, true))
		gn.add_child(add)
	var x := Button.new()
	x.text = "x"
	x.flat = true
	x.tooltip_text = "Delete this node (or select it and press Delete): what read it reads its input"
	x.pressed.connect(func() -> void: p_ctx.delete_nodes([id]))
	gn.get_titlebar_hbox().add_child(x)
	refresh(gn, p_node, p_ctx)
	return gn


# What a node shows from its parameters (title, phi graph), after an edit.
static func refresh(p_gn: GraphNode, p_node: Dictionary, p_ctx: Object) -> void:
	p_gn.title = title(p_node)
	if p_gn.has_meta("phi"):
		var phi: PhiGraph = p_gn.get_meta("phi")
		phi.node = p_node
		phi.queue_redraw()


static func _port_row(p_in: String, p_out: String) -> Control:
	var row := HBoxContainer.new()
	var a := Label.new()
	a.text = p_in
	a.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var b := Label.new()
	b.text = p_out
	b.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	row.add_child(a)
	row.add_child(b)
	return row


static func _label(p_text: String, p_w: float = LABEL_W) -> Label:
	var l := Label.new()
	l.text = p_text
	l.custom_minimum_size.x = p_w
	return l


# A slider and a number box on one value (the box takes any value, the slider shows its range).
static func _slider(p_value: float, p_range: Array, p_set: Callable) -> Control:
	var row := HBoxContainer.new()
	var s := HSlider.new()
	s.min_value = p_range[0]
	s.max_value = p_range[1]
	s.step = p_range[2]
	s.value = p_value
	s.custom_minimum_size.x = 110
	s.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	s.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	var sp := SpinBox.new()
	sp.min_value = p_range[0]
	sp.max_value = p_range[1]
	sp.step = p_range[2] / 10.0
	sp.allow_greater = true
	sp.allow_lesser = true
	sp.value = p_value
	sp.custom_minimum_size.x = 84
	s.value_changed.connect(func(v: float) -> void:
		sp.set_value_no_signal(v)
		p_set.call(v))
	sp.value_changed.connect(func(v: float) -> void:
		s.set_value_no_signal(v)
		p_set.call(v))
	row.add_child(s)
	row.add_child(sp)
	return row


static func _toggle(p_text: String, p_on: bool, p_tip: String, p_set: Callable) -> Button:
	var b := Button.new()
	b.text = p_text
	b.toggle_mode = true
	b.button_pressed = p_on
	b.tooltip_text = p_tip
	b.toggled.connect(p_set)
	return b


static func _options(p_items: Array, p_value: String, p_set: Callable) -> OptionButton:
	var ob := OptionButton.new()
	for it in p_items:
		ob.add_item(String(it))
	ob.select(p_items.find(p_value))
	ob.item_selected.connect(func(i: int) -> void: p_set.call(String(p_items[i])))
	return ob


# A mix row: what it plays, its volume x, mute and solo (the view's only), and a button that takes it out.
static func _mix_row(p_id: String, p_node: Dictionary, p_i: int, p_ctx: Object, p_out: String) -> Control:
	var row := HBoxContainer.new()
	var r: Dictionary = Graph.rows(p_node)[p_i]
	row.add_child(_label(String(r.get("in", "")), 64))
	row.add_child(_slider(float(r.get("x", 1.0)), Kinds.range_of("mix", "x"), func(v: float) -> void:
		var rows: Array = Graph.rows(p_ctx.model_node(p_id)).duplicate(true)
		rows[p_i].x = v
		p_ctx.set_params(p_id, {"rows": rows})))
	row.add_child(_toggle("M", p_ctx.row_flag(p_id, p_i, "mute"), "Mute this row in the view (the file keeps its volume)", func(on: bool) -> void: p_ctx.set_row_flag(p_id, p_i, "mute", on)))
	row.add_child(_toggle("S", p_ctx.row_flag(p_id, p_i, "solo"), "Solo: only the soloed rows sound in the view", func(on: bool) -> void: p_ctx.set_row_flag(p_id, p_i, "solo", on)))
	var x := Button.new()
	x.text = "x"
	x.flat = true
	x.tooltip_text = "Remove this row (what it played stays)"
	x.pressed.connect(func() -> void: p_ctx.remove_mix_row(p_id, p_i))
	row.add_child(x)
	if p_out != "":
		var o := Label.new()
		o.text = p_out
		row.add_child(o)
	return row


# The output's surface colour p_key (the biome's "surface", not the graph's).
static func _surface_row(p_key: String, p_ctx: Object) -> Control:
	var row := HBoxContainer.new()
	row.add_child(_label(p_key))
	var cp := ColorPickerButton.new()
	cp.color = Color(String(p_ctx.model.get("surface", {}).get(p_key, SURFACE[p_key])))
	cp.custom_minimum_size = Vector2(70, 24)
	cp.color_changed.connect(func(c: Color) -> void: p_ctx.set_surface(p_key, "#" + c.to_html(false)))
	row.add_child(cp)
	return row


# A note's wavelength: one of the octaves the cascade inserts (2^k voxels, its lattice Λ_k), from 8 voxels (the finest
# the engine takes) to the top note of the LOD count (k = lods + 4: 26 km at 5 cm and 15 LODs). A value between octaves
# (an older file) shows as its nearest, which is what the engine reads.
static func _octaves(p_m: float, p_set: Callable) -> OptionButton:
	var ob := OptionButton.new()
	var lo := 3
	var hi := EngineInfo.lods + 4
	for k in range(lo, hi + 1):
		var m := EngineInfo.voxel_m * pow(2.0, k)
		ob.add_item("%.1f m" % m if m < 1000.0 else "%.1f km" % (m / 1000.0))
	var near := clampi(int(round(log(maxf(p_m, 1e-6) / EngineInfo.voxel_m) / log(2.0))), lo, hi)
	ob.select(near - lo)
	ob.item_selected.connect(func(i: int) -> void: p_set.call(snappedf(EngineInfo.voxel_m * pow(2.0, lo + i), 0.0001)))
	return ob


static func _param_row(p_id: String, p_node: Dictionary, p_d: Dictionary, p_ctx: Object) -> Control:
	var row := HBoxContainer.new()
	row.add_child(_label(p_d.name))
	var v: Variant = p_node[p_d.name]
	var name: String = p_d.name
	var set1 := func(p_value: Variant) -> void: p_ctx.set_params(p_id, {name: p_value})
	var fields: Array = p_ctx.world.fields.map(func(f): return String(f.get("name", "")))
	match String(p_d.type):
		"number" when p_node.kind == "note" and name == "wavelength_m":
			row.add_child(_octaves(float(v), set1))
		"number":
			var c := _slider(float(v), Kinds.range_of(p_node.kind, name), set1)
			c.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			row.add_child(c)
		"seed":
			var sp := SpinBox.new()
			sp.max_value = 4294967295.0
			sp.rounded = true
			sp.value = int(v)
			sp.value_changed.connect(func(x: float) -> void: p_ctx.set_params(p_id, {name: int(x)}))
			row.add_child(sp)
		"pow2":
			# A power of two: a step up doubles, a step down halves.
			var sp := SpinBox.new()
			sp.min_value = 1
			sp.max_value = 256
			sp.rounded = true
			sp.value = int(v)
			var cur := [int(v)]
			sp.value_changed.connect(func(x: float) -> void:
				var i := int(x)
				if i & (i - 1) != 0:
					i = cur[0] * 2 if i > cur[0] else maxi(cur[0] / 2, 1)
					sp.set_value_no_signal(i)
				cur[0] = i
				p_ctx.set_params(p_id, {name: i}))
			row.add_child(sp)
		"field":
			row.add_child(_options(fields, String(v), set1))
		"shape":
			row.add_child(_options(Array(p_d.options), String(v), func(s: String) -> void: p_ctx.set_params(p_id, {name: s}, true)))
		"coeffs":
			for i in 4:
				var k := i
				var sp := SpinBox.new()
				sp.min_value = -4.0
				sp.max_value = 4.0
				sp.step = 0.001
				sp.allow_greater = true
				sp.allow_lesser = true
				sp.value = float(v[k]) if v is Array and k < v.size() else 0.0
				sp.tooltip_text = ["c on 1", "c on n", "c on |n|", "c on n^2"][k]
				sp.value_changed.connect(func(x: float) -> void:
					var c: Array = (p_node[name] as Array).duplicate()
					c[k] = x
					p_ctx.set_params(p_id, {name: c}))
				row.add_child(sp)
		"points":
			var box := VBoxContainer.new()
			var ce := CurveEdit.new()
			ce.set_points(v, p_ctx.curve_span())
			ce.points_changed.connect(func(p: Array) -> void: p_ctx.set_params(p_id, {name: p}))
			p_ctx.curve_ui[p_id] = ce
			var reset := Button.new()
			reset.text = "Reset to no remap"
			reset.pressed.connect(func() -> void: p_ctx.set_params(p_id, {name: Kinds.identity_curve(p_ctx.curve_span())}, true))
			box.add_child(ce)
			box.add_child(reset)
			row.add_child(box)
	if not p_d.required:
		var x := Button.new()
		x.text = "-"
		x.flat = true
		x.tooltip_text = "Take %s out (the engine's default)" % name
		x.pressed.connect(func() -> void: p_ctx.set_params(p_id, {name: null}, true))
		row.add_child(x)
	return row
