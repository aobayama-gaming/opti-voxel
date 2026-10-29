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
	if p_node.kind == "note":
		var thumb := ColorRect.new()
		thumb.custom_minimum_size = Vector2(150, 64)
		thumb.material = ShaderMaterial.new()
		thumb.material.shader = p_ctx.thumb_shader
		thumb.tooltip_text = "The note alone, from above, over four wavelengths"
		gn.add_child(thumb)
		gn.set_meta("thumb", thumb)
	elif p_node.kind == "shape":
		var phi := PhiGraph.new()
		gn.add_child(phi)
		gn.set_meta("phi", phi)
	elif p_node.kind in ["warp", "note_warp"]:
		var info := Label.new()
		info.modulate = Color(1, 1, 1, 0.7)
		info.text = _warp_info(p_node.kind, p_ctx.model)
		gn.add_child(info)
	elif p_node.kind == "mix":
		var total := Label.new()
		total.modulate = Color(1, 1, 1, 0.7)
		gn.add_child(total)
		p_ctx.mix_ui.get_or_add(p_id, {"rows": [], "total": null}).total = total
	var x := Button.new()
	x.text = "x"
	x.flat = true
	x.tooltip_text = "Delete this node (or select it and press Delete)"
	x.pressed.connect(func() -> void: p_ctx.delete_nodes([p_id]))
	gn.get_titlebar_hbox().add_child(x)
	refresh(gn, p_node, p_ctx)
	return gn


# What a warp costs, under its ports: the climate's octaves and their reads, or none.
static func _warp_info(p_kind: String, p_model: Dictionary) -> String:
	if p_kind == "note_warp":
		return "No noise of its own: x, y, z read\nnotes (coarser than the one it moves).\nPlayed notes cost nothing more."
	var cw := Kinds.climate_warp(p_model)
	var k := int(cw.k)
	var n := int(cw.octaves)
	return "The climate's, every biome's:\n%s .. %s, %d octaves x 3 axes\n= %d reads per point, once." % [
		fmt_m(Octaves.wavelength_m(k - n + 1, p_model.voxel_m)), fmt_m(Octaves.wavelength_m(k, p_model.voxel_m)), n, 3 * n]


# What a node shows from its parameters (title, thumbnail, phi graph), after an edit.
static func refresh(p_gn: GraphNode, p_node: Dictionary, p_ctx: Object) -> void:
	var voxel_m: float = p_ctx.model.voxel_m
	p_gn.title = title(p_node, voxel_m)
	if p_gn.has_meta("thumb"):
		var k := int(Kinds.param(p_node, "k"))
		var w := Octaves.shader_words(Octaves.ground_octave(k, int(p_ctx.model.lods)), k, [0, 0, 0])
		var m: ShaderMaterial = p_gn.get_meta("thumb").material
		m.set_shader_parameter("u_base", w.base)
		m.set_shader_parameter("u_frac", w.frac)
		m.set_shader_parameter("u_salt", w.salt)
		m.set_shader_parameter("u_inv", pow(2.0, -k))
		m.set_shader_parameter("u_span", 4.0 * pow(2.0, k))
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


# A Mix row: its note, volume, mute / solo / centre, then what it adds (its swing in metres and its share of the
# Mix's, filled by the mixer after each compile: show_mix) and a button that takes the row and its note out.
static func _mix_row(p_id: String, p_node: Dictionary, p_i: int, p_ctx: Object, p_out: String) -> Control:
	var row := HBoxContainer.new()
	var r: Dictionary = Kinds.rows(p_node)[p_i]
	var set_row := func(p_key: String, p_value: Variant) -> void:
		var rows: Array = Kinds.rows(p_ctx.model.nodes[p_id]).duplicate(true)
		rows[p_i][p_key] = p_value
		p_ctx.set_params(p_id, {"rows": rows})
	var name := _label("", 58)
	name.tooltip_text = "The wavelength of the note this row plays"
	row.add_child(name)
	row.add_child(_slider(float(r.get("vol", 1.0)), 0.0, 4.0, 0.01, func(v: float) -> void: set_row.call("vol", v)))
	row.add_child(_toggle("M", r.get("mute", false), "Mute this row", func(on: bool) -> void: set_row.call("mute", on)))
	row.add_child(_toggle("S", r.get("solo", false), "Solo: only the soloed rows sound", func(on: bool) -> void: set_row.call("solo", on)))
	row.add_child(_toggle("C", r.get("centre", true), "Centre: take the shape's mean out (the engine does; a shape then moves no average height)", func(on: bool) -> void: set_row.call("centre", on)))
	var peak := _label("", 70)
	peak.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	row.add_child(peak)
	var bar := ProgressBar.new()
	bar.custom_minimum_size = Vector2(80, 18)
	bar.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	bar.tooltip_text = "This row's share of the Mix's relief"
	row.add_child(bar)
	var x := Button.new()
	x.text = "x"
	x.flat = true
	x.tooltip_text = "Remove this row, and its note and shape if nothing else uses them"
	x.pressed.connect(func() -> void: p_ctx.remove_mix_row(p_id, p_i))
	row.add_child(x)
	if p_out != "":
		var o := Label.new()
		o.text = p_out
		row.add_child(o)
	var ui: Dictionary = p_ctx.mix_ui.get_or_add(p_id, {"rows": [], "total": null})
	ui.rows.append({"name": name, "peak": peak, "bar": bar})
	return row


# What a Mix's rows add, after a compile (p_rows: compile.gd's mix_rows for it).
static func show_mix(p_ui: Dictionary, p_rows: Array) -> void:
	var total := 0.0
	for r in p_rows:
		total += r.peak if r.on else 0.0
	for i in mini(p_ui.rows.size(), p_rows.size()):
		var r: Dictionary = p_rows[i]
		var ui: Dictionary = p_ui.rows[i]
		if not is_instance_valid(ui.peak):
			continue
		ui.name.text = fmt_m(r.lam) if r.lam > 0.0 else ("-" if r.unit == 0.0 else "no note")
		ui.peak.text = ("+-%s" % fmt_m(r.peak)) if r.unit > 0.0 else ""
		ui.peak.modulate.a = 1.0 if r.on else 0.4
		ui.peak.tooltip_text = "Its swing: volume x %s (lambda / 8) x the shaped note's bound%s" % [fmt_m(r.unit), "" if r.on else " (muted)"]
		ui.bar.value = 100.0 * r.peak / total if r.on and total > 0.0 else 0.0
	if p_ui.total != null and is_instance_valid(p_ui.total):
		p_ui.total.text = "height = the rows' sum, within +-%s\na row: volume x lambda/8 x its shaped note" % fmt_m(total)


# A wavelength picker over the ground's notes (K_MIN .. k_top), its items' ids their k.
static func k_picker(p_k: int, p_model: Dictionary, p_show_k: bool, p_pick: Callable) -> OptionButton:
	var ob := OptionButton.new()
	var top := Octaves.k_top(int(p_model.lods))
	for k in range(Octaves.K_MIN, top + 1):
		var m := fmt_m(Octaves.wavelength_m(k, p_model.voxel_m))
		ob.add_item("%s (k %d)" % [m, k] if p_show_k else m, k)
	ob.select(clampi(p_k, Octaves.K_MIN, top) - Octaves.K_MIN)
	ob.item_selected.connect(func(i: int) -> void: p_pick.call(ob.get_item_id(i)))
	return ob


static func _param_row(p_id: String, p_node: Dictionary, p_d: Dictionary, p_ctx: Object) -> Control:
	var row := HBoxContainer.new()
	row.add_child(_label(p_d.name))
	row.tooltip_text = p_d.get("hint", "")
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
			row.add_child(k_picker(int(v), p_ctx.model, true, func(k: int) -> void: p_ctx.set_params(p_id, {name: k})))
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
