extends RefCounted
## A biome graph node's GraphNode: its port rows (a Mix's rows carry their volume, mute, solo and centre), then one
## row per parameter. Edits go through the mixer (p_ctx.set_params): the model is the truth, the UI only shows it.

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const CurveEdit := preload("res://debug/biome_mixer/curve_edit.gd")

const LABEL_W := 84.0


# A Shape's phi(n) over n in [-1, 1].
class PhiGraph:
	extends Control
	var coeffs := [0.0, 1.0, 0.0, 0.0]

	func _init() -> void:
		custom_minimum_size = Vector2(200, 70)

	func _draw() -> void:
		draw_rect(Rect2(Vector2.ZERO, size), Kinds.PLOT_BG)
		var ys := []
		var lo := -1.0
		var hi := 1.0
		for i in 65:
			var y := Kinds.phi(coeffs, lerpf(-1.0, 1.0, i / 64.0))
			ys.append(y)
			lo = minf(lo, y)
			hi = maxf(hi, y)
		var line := PackedVector2Array()
		for i in 65:
			line.append(Vector2(i / 64.0 * size.x, size.y - (ys[i] - lo) / (hi - lo) * size.y))
		var zero := size.y - (0.0 - lo) / (hi - lo) * size.y
		draw_line(Vector2(0, zero), Vector2(size.x, zero), Color(1, 1, 1, 0.15))
		draw_polyline(line, Kinds.PORT_COLORS[Kinds.FIELD], 2.0, true)
		draw_string(get_theme_default_font(), Vector2(4, 12), "phi(n), n in [-1, 1]", HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(1, 1, 1, 0.45))


static func fmt_m(p_m: float) -> String:
	if p_m >= 1000.0:
		return "%s km" % String.num(p_m / 1000.0, 1 if p_m < 100000.0 else 0)
	return "%s m" % String.num(p_m, 2 if p_m < 1.0 else 1)


static func title(p_node: Dictionary, p_voxel_m: float) -> String:
	var t: String = Kinds.KINDS[p_node.kind]["title"]
	match p_node.kind:
		"note":
			t += " %s" % fmt_m(Octaves.wavelength_m(int(Kinds.param(p_node, "k")), p_voxel_m))
			if Kinds.param(p_node, "warp"):
				t += ", warped"
		"shape":
			t += " %s" % Kinds.param(p_node, "preset")
	return t


static func build(p_id: String, p_node: Dictionary, p_ctx: Object) -> GraphNode:
	var gn := GraphNode.new()
	gn.name = p_id
	gn.position_offset = Vector2(p_node.pos[0], p_node.pos[1])
	gn.tooltip_text = Kinds.KINDS[p_node.kind].get("hint", "")
	var ins := Kinds.inputs(p_node)
	var outs := Kinds.outputs(p_node)
	for i in maxi(ins.size(), outs.size()):
		var out_label: String = outs[i][0] if i < outs.size() else ""
		if p_node.kind == "mix" and i < Kinds.rows(p_node).size():
			gn.add_child(_mix_row(p_id, p_node, i, p_ctx, out_label))
		elif p_node.kind == "mix":
			gn.add_child(_port_row("+ link a note here for a new row", out_label))
		else:
			gn.add_child(_port_row(ins[i][0] if i < ins.size() else "", out_label))
		var tl: int = ins[i][1] if i < ins.size() else 0
		var tr: int = outs[i][1] if i < outs.size() else 0
		gn.set_slot(i, i < ins.size(), tl, Kinds.PORT_COLORS[tl], i < outs.size(), tr, Kinds.PORT_COLORS[tr])
	var coeffs: bool = Kinds.param(p_node, "preset") == "coeffs"
	for d in Kinds.KINDS[p_node.kind]["params"]:
		if d.type == "rows":
			continue
		if p_node.kind == "shape" and (d.name in ["c0", "c1", "c2", "c3"]) != coeffs and d.name != "preset":
			continue
		gn.add_child(_param_row(p_id, p_node, d, p_ctx))
	if p_node.kind == "shape":
		var phi := PhiGraph.new()
		gn.add_child(phi)
		gn.set_meta("phi", phi)
	elif p_node.kind == "note_warp":
		var info := Label.new()
		info.modulate = Color(1, 1, 1, 0.7)
		info.text = "No noise of its own: x, y, z read\nnotes (coarser than the one it moves).\nPlayed notes cost nothing more."
		gn.add_child(info)
	elif p_node.kind == "mix":
		var total := Label.new()
		total.modulate = Color(1, 1, 1, 0.7)
		total.text = "height = the rows' sum\na row: volume x lambda/8 x its shaped note"
		gn.add_child(total)
	# The notes are the ground's: no x.
	if p_node.kind != "note":
		var x := Button.new()
		x.text = "x"
		x.flat = true
		x.tooltip_text = "Delete this node (or select it and press Delete)"
		x.pressed.connect(func() -> void: p_ctx.delete_nodes([p_id]))
		gn.get_titlebar_hbox().add_child(x)
	refresh(gn, p_node, p_ctx)
	return gn


# What the climate's warp is, in a note's warp tooltip: its octaves and their reads.
static func climate_warp_info(p_model: Dictionary) -> String:
	var cw := Kinds.climate_warp(p_model)
	var k := int(cw.k)
	var n := int(cw.octaves)
	return "The climate's warp, every biome's: %s .. %s, %d octaves x 3 axes = %d reads per point, once however many notes it moves." % [
		fmt_m(Octaves.wavelength_m(k - n + 1, Octaves.voxel_m)), fmt_m(Octaves.wavelength_m(k, Octaves.voxel_m)), n, 3 * n]


# What a node shows from its parameters (title, phi graph), after an edit.
static func refresh(p_gn: GraphNode, p_node: Dictionary, _p_ctx: Object) -> void:
	p_gn.title = title(p_node, Octaves.voxel_m)
	if p_gn.has_meta("phi"):
		var phi: PhiGraph = p_gn.get_meta("phi")
		phi.coeffs = Kinds.shape_coeffs(p_node)
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
static func _slider(p_value: float, p_min: float, p_max: float, p_step: float, p_set: Callable) -> Control:
	var row := HBoxContainer.new()
	var s := HSlider.new()
	s.min_value = p_min
	s.max_value = p_max
	s.step = p_step
	s.value = p_value
	s.custom_minimum_size.x = 110
	s.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	s.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	var sp := SpinBox.new()
	sp.min_value = p_min
	sp.max_value = p_max
	sp.step = p_step / 10.0
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


# A Mix row: its note's wavelength, volume, mute / solo / centre, and a button that takes the row out (its note stays).
static func _mix_row(p_id: String, p_node: Dictionary, p_i: int, p_ctx: Object, p_out: String) -> Control:
	var row := HBoxContainer.new()
	var r: Dictionary = Kinds.rows(p_node)[p_i]
	var set_row := func(p_key: String, p_value: Variant) -> void:
		var rows: Array = Kinds.rows(p_ctx.model.nodes[p_id]).duplicate(true)
		rows[p_i][p_key] = p_value
		p_ctx.set_params(p_id, {"rows": rows})
	var k := _row_note(p_ctx.model, p_id, p_i)
	var name := _label(fmt_m(Octaves.wavelength_m(k, Octaves.voxel_m)) if k >= 0 else "no note", 58)
	name.tooltip_text = "The wavelength of the note this row plays"
	row.add_child(name)
	row.add_child(_slider(float(r.get("vol", 1.0)), 0.0, 4.0, 0.01, func(v: float) -> void: set_row.call("vol", v)))
	row.add_child(_toggle("M", r.get("mute", false), "Mute this row", func(on: bool) -> void: set_row.call("mute", on)))
	row.add_child(_toggle("S", r.get("solo", false), "Solo: only the soloed rows sound", func(on: bool) -> void: set_row.call("solo", on)))
	row.add_child(_toggle("C", r.get("centre", true), "Centre: take the shape's mean out (the engine does; a shape then moves no average height)", func(on: bool) -> void: set_row.call("centre", on)))
	var x := Button.new()
	x.text = "x"
	x.flat = true
	x.tooltip_text = "Remove this row, and its shape if nothing else uses it (the note stays, silent)"
	x.pressed.connect(func() -> void: p_ctx.remove_mix_row(p_id, p_i))
	row.add_child(x)
	if p_out != "":
		var o := Label.new()
		o.text = p_out
		row.add_child(o)
	return row


# The octave of the note a Mix row plays (through its shape and multifractal), -1 for none.
static func _row_note(p_model: Dictionary, p_mix: String, p_row: int) -> int:
	var src = Kinds.input_of(p_model, p_mix, p_row)
	while src != null:
		var n: Dictionary = p_model.nodes[src.id]
		if n.kind == "note":
			return int(Kinds.param(n, "k"))
		src = Kinds.input_of(p_model, src.id, 0)
	return -1


static func _param_row(p_id: String, p_node: Dictionary, p_d: Dictionary, p_ctx: Object) -> Control:
	var row := HBoxContainer.new()
	row.add_child(_label(p_d.name))
	row.tooltip_text = p_d.get("hint", "")
	if p_node.kind == "note" and p_d.name == "warp":
		row.tooltip_text += "\n" + climate_warp_info(p_ctx.model)
	var v: Variant = Kinds.param(p_node, p_d.name)
	var name: String = p_d.name
	var set1 := func(p_value: Variant) -> void: p_ctx.set_params(p_id, {name: p_value})
	match p_d.type:
		"float":
			var c := _slider(float(v), p_d.min, p_d.max, p_d.get("step", 0.01), set1)
			c.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			row.add_child(c)
		"int":
			var sp := SpinBox.new()
			sp.min_value = p_d.get("min", 0)
			sp.max_value = p_d.get("max", 100)
			sp.allow_greater = true
			sp.rounded = true
			sp.value = int(v)
			sp.value_changed.connect(func(x: float) -> void: p_ctx.set_params(p_id, {name: int(x)}))
			row.add_child(sp)
		"bool":
			var cb := CheckBox.new()
			cb.button_pressed = bool(v)
			cb.toggled.connect(set1)
			row.add_child(cb)
		"enum":
			var ob := OptionButton.new()
			for o in p_d.options:
				ob.add_item(o)
			ob.select(p_d.options.find(v))
			# An option sets its option_params with it (a Shape preset its offset), and may show other rows.
			ob.item_selected.connect(func(i: int) -> void:
				var option: String = p_d.options[i]
				var sets := Kinds.option_params(p_node.kind, name, option)
				sets[name] = option
				p_ctx.set_params(p_id, sets, true))
			row.add_child(ob)
		"color":
			var cp := ColorPickerButton.new()
			cp.color = Color(v)
			cp.custom_minimum_size = Vector2(70, 24)
			cp.color_changed.connect(func(c: Color) -> void: p_ctx.set_params(p_id, {name: "#" + c.to_html(false)}))
			row.add_child(cp)
		"k":
			# Fixed: the note is the ground's octave k.
			row.add_child(_label("k %d (%s)" % [int(v), fmt_m(Octaves.wavelength_m(int(v), Octaves.voxel_m))], 0.0))
		"points":
			var box := VBoxContainer.new()
			var ce := CurveEdit.new()
			ce.set_points(v, p_ctx.curve_span(p_id))
			ce.points_changed.connect(func(p: Array) -> void: p_ctx.set_params(p_id, {name: p}))
			p_ctx.curve_ui[p_id] = ce
			var reset := Button.new()
			reset.text = "Reset to no remap"
			reset.pressed.connect(func() -> void: p_ctx.set_params(p_id, {name: Kinds.identity_curve(p_ctx.curve_span(p_id))}, true))
			box.add_child(ce)
			box.add_child(reset)
			row.add_child(box)
	return row
