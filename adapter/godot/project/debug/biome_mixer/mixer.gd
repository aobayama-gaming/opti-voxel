extends Control
## The biome mixer: build one biome from the ground's octaves (notes) in a node graph, seen raymarched beside it.
## The graph (the model: nodes, links) is the truth; compile.gd turns it into the preview's shader, the structure into
## code (recompiled when it changes), every parameter into a uniform (a slider redraws, no recompile).
## Run: tools/biome_mixer.sh [-- flags]  (godot --path adapter/godot/project res://debug/biome_mixer/biome_mixer.tscn)
## Flags:
##   --open=PATH        a biome file (res://climate/biomes/*.biome.json); default: the autosave, else world.json's mountain
##   --import=NAME      a biome of res://climate/world.json
##   --set=ID.PARAM=V   set a node's parameter after loading (V as JSON: 1.5, true, "plain"), or view.KEY=V a view
##                      setting (view.shadows=false); repeatable
##   --shot=PATH        draw once, save the window to PATH (and the view alone to PATH_view.png), quit
##   --check            headless self-test: the noise pins, every template and world.json biome compiled, files
##                      round-tripped; prints MIXER: {json} and quits 0 if ok

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const Compile := preload("res://debug/biome_mixer/compile.gd")
const Nodes := preload("res://debug/biome_mixer/nodes.gd")
const ShaderLib := preload("res://debug/biome_mixer/shader_lib.gd")
const ImportTable := preload("res://debug/biome_mixer/import_table.gd")
const Preview := preload("res://debug/biome_mixer/preview.gd")

const WORLD := "res://climate/world.json"
const BIOMES_DIR := "res://climate/biomes"
const AUTOSAVE := "user://biome_mixer_autosave.biome.json"
const VIEW_DEFAULTS := {
	"quality": 1, "steps": 256, "step": 1.0, "shadows": true, "sun_az": 135.0, "sun_el": 35.0, "exag": 1.0,
	"water_on": false, "water": 0.0, "contour": 0.0,
}

var model := {}
var result := {}
var thumb_shader := Shader.new()
var mix_ui := {}  # Mix id -> {rows: [{name, peak, bar}], total}: what its rows add (nodes.gd fills it)
var curve_ui := {}  # Height curve id -> its editor, told its input's range after each compile

var graph := GraphEdit.new()
var preview := Preview.new()
var status := Label.new()
var budget := Label.new()
var name_edit := LineEdit.new()
var add_menu := PopupMenu.new()
var add_at := Vector2.ZERO
var autosave := Timer.new()
var view_box := HFlowContainer.new()
var compile_queued := false
var pushed := {}  # the uniforms the preview has
var pushed_code := ""
var world := {}  # res://climate/world.json, parsed once


func _ready() -> void:
	var args := OS.get_cmdline_user_args()
	if "--check" in args:
		_check()
		return
	thumb_shader.code = ShaderLib.thumb()
	var start := {}
	var sets := []
	var shot := ""
	for a in args:
		if a.begins_with("--shot="):
			shot = a.trim_prefix("--shot=")
	# The window redraws only on input or a change (an idle GPU); a shot needs every frame drawn.
	OS.low_processor_usage_mode = shot == ""
	var screen := DisplayServer.screen_get_usable_rect(get_window().current_screen)
	get_window().size = Vector2i(Vector2(screen.size) * 0.85)
	get_window().position = screen.position + (screen.size - get_window().size) / 2
	_build_ui()
	# A shot leaves the session's autosave alone (it may edit the graph with --set).
	if shot != "":
		autosave.timeout.disconnect(_autosave)
	var asked := ""
	for a in args:
		if a.begins_with("--open="):
			asked = a
			start = _read(a.trim_prefix("--open="))
		elif a.begins_with("--import="):
			asked = a
			start = _import(a.trim_prefix("--import="))
		elif a.begins_with("--set="):
			sets.append(a.trim_prefix("--set="))
	if asked != "" and start.is_empty():
		push_error("%s: no such biome file or world.json biome" % asked)
		if shot != "":
			get_tree().quit(1)
			return
	if start.is_empty() and shot == "":
		start = _read(AUTOSAVE)
	if start.is_empty():
		start = _import("mountain")
	set_model(start, true)
	for s in sets:
		_apply_set(s)
	if shot != "":
		await _shot(shot)


# --- the model ---------------------------------------------------------------------------------------------------

func set_model(p_model: Dictionary, p_frame: bool) -> void:
	model = p_model
	model.next_id = int(model.get("next_id", 1))
	model.voxel_m = float(model.get("voxel_m", Octaves.DEFAULT_VOXEL_M))
	model.lods = int(model.get("lods", Octaves.DEFAULT_LODS))
	if not model.has("nodes"):
		model.nodes = {}
	if not model.has("links"):
		model.links = []
	Kinds.drop_unknown(model)
	# The climate's warp is the model's (a file from before kept it on its warp node).
	model.climate_warp = Kinds.climate_warp(model)
	var view: Dictionary = VIEW_DEFAULTS.duplicate()
	view.merge(model.get("view", {}), true)
	model.view = view
	name_edit.text = model.get("name", "biome")
	_build_view_bar()
	rebuild_ui()
	recompile(false)
	if p_frame:
		preview.frame(result.hmax)
		_fit_graph()


# The graph's top-left corner in view, zoomed so its columns (warp, note, shape, multifractal, mix) fit the width. A
# frame later: the GraphEdit sizes its scroll range from its nodes then, and clamps any offset set before.
func _fit_graph() -> void:
	await get_tree().process_frame
	if model.nodes.is_empty():
		return
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for id in model.nodes:
		var p := Vector2(model.nodes[id].pos[0], model.nodes[id].pos[1])
		lo = lo.min(p)
		hi = hi.max(p)
	graph.zoom = clampf(maxf(graph.size.x, 600.0) / (hi.x - lo.x + 400.0), 0.35, 0.8)
	graph.scroll_offset = lo * graph.zoom - Vector2(30, 30)


# A node's parameters changed (p_rebuild: its rows changed too, rebuild its GraphNode). A slider sends several a frame:
# they compile once, at the frame's end.
func set_params(p_id: String, p_params: Dictionary, p_rebuild: bool = false) -> void:
	var node: Dictionary = model.nodes[p_id]
	node.params.merge(p_params, true)
	if p_rebuild:
		_rebuild_node(p_id)
	elif graph.has_node(NodePath(p_id)):
		Nodes.refresh(graph.get_node(NodePath(p_id)), node, self)
	if not compile_queued:
		compile_queued = true
		recompile.call_deferred(true)


func recompile(p_moving: bool) -> void:
	compile_queued = false
	result = Compile.new().run(model, float(model.view.exag))
	preview.set_code(result.code)
	# Only the uniforms that changed (a slider moves one or two of a few hundred); all of them after a recompile.
	var fresh := preview.code != pushed_code
	pushed_code = preview.code
	var changed := {}
	for key in result.uniforms:
		if fresh or not pushed.has(key) or pushed[key] != result.uniforms[key]:
			changed[key] = result.uniforms[key]
	pushed = result.uniforms
	preview.set_uniforms(changed)
	_apply_view()
	for id in mix_ui:
		Nodes.show_mix(mix_ui[id], result.mix_rows.get(id, []))
	for id in curve_ui:
		curve_ui[id].set_span(curve_span(id))
	_show_budget()
	var lines := PackedStringArray()
	for e in result.errors:
		lines.append("Error: " + e)
	var warned := {}
	for w in result.warnings:
		if not warned.has(w):
			warned[w] = true
			lines.append(w)
	lines.append("%d nodes. Right-click the graph: add a node; x or Delete removes one; a Mix row's x removes its note." % model.nodes.size())
	status.text = "  ".join(lines)
	status.add_theme_color_override("font_color", Color(1.0, 0.5, 0.45) if not result.ok else Color(0.8, 0.82, 0.86))
	preview.redraw(p_moving)
	autosave.start()


# The budget line over the view: what the biome reads per point, its relief and the row that moves the most, and
# how long the view took to sharpen (preview.sharp_ms, after its last pass).
func _show_budget() -> void:
	var b: Dictionary = result.budget
	var top := {}
	for id in result.mix_rows:
		for r in result.mix_rows[id]:
			if r.on and r.peak > top.get("peak", 0.0):
				top = r
	var parts := PackedStringArray([
		"Per point: %d noise reads = %d notes (%d shaped, %d multifractal, %d moved by the climate warp, %d by notes for free) + %d climate warp octaves; plain ground reads %d." % [
			b.reads, b.notes, b.shapes, b.multifractals, b.warped, b.note_warped, b.warp_octaves, Octaves.k_top(model.lods) - Octaves.K_MIN + 1],
		"Relief within +-%s%s." % [Nodes.fmt_m(result.hmax / float(model.view.exag)),
			", most from the %s note (+-%s)" % [Nodes.fmt_m(top.lam), Nodes.fmt_m(top.peak)] if top.get("lam", 0.0) > 0.0 else ""],
	])
	if preview.sharp_ms > 0.0:
		parts.append("View sharp in %d ms." % int(preview.sharp_ms))
	budget.text = "  ".join(parts)


# The input a Height curve's editor shows (metres either side of 0): its input's bound, else (a new curve, not linked
# yet) the ground's.
func curve_span(p_id: String = "") -> float:
	var curve_in: Dictionary = result.get("curve_in", {})
	if curve_in.has(p_id):
		return curve_in[p_id]
	return maxf(float(result.get("hmax", 1000.0)) / float(model.view.exag), 10.0)


func rebuild_ui() -> void:
	graph.clear_connections()
	for c in graph.get_children():
		if c is GraphNode:
			graph.remove_child(c)
			c.queue_free()
	mix_ui.clear()
	curve_ui.clear()
	for id in model.nodes:
		_add_graph_node(id)
	for l in model.links:
		graph.connect_node(l.from, int(l.from_port), l.to, int(l.to_port))


# One node's GraphNode built again (its rows changed: a Shape's preset, a Mix's new row) and its links put back.
func _rebuild_node(p_id: String) -> void:
	var old := graph.get_node_or_null(NodePath(p_id))
	if old != null:
		for l in model.links:
			if l.from == p_id or l.to == p_id:
				graph.disconnect_node(l.from, int(l.from_port), l.to, int(l.to_port))
		graph.remove_child(old)
		old.queue_free()
	mix_ui.erase(p_id)
	curve_ui.erase(p_id)
	_add_graph_node(p_id)
	for l in model.links:
		if l.from == p_id or l.to == p_id:
			graph.connect_node(l.from, int(l.from_port), l.to, int(l.to_port))


func _add_graph_node(p_id: String) -> void:
	var gn := Nodes.build(p_id, model.nodes[p_id], self)
	gn.position_offset_changed.connect(func() -> void:
		model.nodes[p_id].pos = [gn.position_offset.x, gn.position_offset.y]
		autosave.start())
	graph.add_child(gn)


func add_node(p_kind: String, p_pos: Vector2) -> String:
	# A new Height curve starts as no remap over the ground's span (its kind's default has no span to know).
	var id := Kinds.add(model, p_kind, p_pos, {"points": Kinds.identity_curve(curve_span())} if p_kind == "curve" else {})
	_add_graph_node(id)
	recompile(false)
	return id


# Whether p_to reaches p_from along the links (a link from p_from to p_to would close a loop).
func _reaches(p_to: String, p_from: String) -> bool:
	var stack := [p_to]
	var seen := {}
	while not stack.is_empty():
		var n: String = stack.pop_back()
		if n == p_from:
			return true
		if seen.has(n):
			continue
		seen[n] = true
		for l in model.links:
			if l.from == n:
				stack.append(l.to)
	return false


func _on_connection_request(p_from: StringName, p_from_port: int, p_to: StringName, p_to_port: int) -> void:
	if _reaches(p_to, p_from):
		status.text = "That link would make a loop."
		return
	for l in model.links.duplicate():
		if l.to == String(p_to) and int(l.to_port) == p_to_port:
			model.links.erase(l)
			graph.disconnect_node(l.from, int(l.from_port), l.to, p_to_port)
	model.links.append({"from": String(p_from), "from_port": p_from_port, "to": String(p_to), "to_port": p_to_port})
	var to: Dictionary = model.nodes[String(p_to)]
	if to.kind == "mix" and p_to_port >= Kinds.rows(to).size():
		while Kinds.rows(to).size() <= p_to_port:
			to.params.rows.append(Kinds.new_row())
		_rebuild_node(String(p_to))
	else:
		graph.connect_node(p_from, p_from_port, p_to, p_to_port)
	recompile(false)


func _on_disconnection_request(p_from: StringName, p_from_port: int, p_to: StringName, p_to_port: int) -> void:
	for l in model.links.duplicate():
		if l.from == String(p_from) and int(l.from_port) == p_from_port and l.to == String(p_to) and int(l.to_port) == p_to_port:
			model.links.erase(l)
	graph.disconnect_node(p_from, p_from_port, p_to, p_to_port)
	recompile(false)


func _on_delete_nodes_request(p_names: Array) -> void:
	delete_nodes(p_names.map(func(n): return String(n)))


func delete_nodes(p_ids: Array) -> void:
	Kinds.cut_nodes(model, p_ids)
	rebuild_ui()
	recompile(false)


func remove_mix_row(p_mix: String, p_row: int) -> void:
	Kinds.cut_row(model, p_mix, p_row)
	rebuild_ui()
	recompile(false)


func _apply_set(p_spec: String) -> void:
	var eq := p_spec.find("=")
	var dot := p_spec.find(".")
	var id := p_spec.substr(0, dot)
	if eq < 0 or dot < 0 or dot > eq or not (model.nodes.has(id) or id == "view"):
		push_error("--set=%s: expected ID.PARAM=VALUE with a node of the graph (or view.KEY=VALUE)" % p_spec)
		return
	var raw := p_spec.substr(eq + 1)
	var json := JSON.new()
	var v = json.data if json.parse(raw) == OK else null
	var key := p_spec.substr(dot + 1, eq - dot - 1)
	if id == "view":
		model.view[key] = v if v != null else raw
		recompile(false)
	else:
		set_params(id, {key: v if v != null else raw}, true)


# --- files -------------------------------------------------------------------------------------------------------

func _read(p_path: String) -> Dictionary:
	if not FileAccess.file_exists(p_path):
		return {}
	var d = JSON.parse_string(FileAccess.get_file_as_string(p_path))
	return d if d is Dictionary and d.has("nodes") else {}


func _world_table() -> Dictionary:
	if world.is_empty():
		var table = JSON.parse_string(FileAccess.get_file_as_string(WORLD))
		world = table if table is Dictionary else {}
	return world


func _import(p_name: String) -> Dictionary:
	if not p_name in ImportTable.biome_names(_world_table()):
		return {}
	var r: Dictionary = ImportTable.new(p_name).import_biome(world)
	for w in r.warnings:
		print("import %s: %s" % [p_name, w])
	return r.model


func _write(p_path: String) -> bool:
	var f := FileAccess.open(p_path, FileAccess.WRITE)
	if f == null:
		return false
	f.store_string(JSON.stringify(model, "\t", false, true))
	return true


func _autosave() -> void:
	_write(AUTOSAVE)


func _save() -> void:
	var file := String(model.get("name", "biome")).validate_filename().replace(" ", "_")
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(BIOMES_DIR))
	var path := "%s/%s.biome.json" % [BIOMES_DIR, file]
	status.text = ("Saved " + path) if _write(path) else ("Could not write " + path)


# --- the window --------------------------------------------------------------------------------------------------

func _build_ui() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	var root := VBoxContainer.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(root)
	var bar := HBoxContainer.new()
	root.add_child(bar)
	var title := Label.new()
	title.text = "  Biome mixer  "
	bar.add_child(title)
	name_edit.custom_minimum_size.x = 160
	name_edit.text_changed.connect(func(t: String) -> void: model.name = t)
	bar.add_child(name_edit)
	bar.add_child(_menu("New", ["Plain ground", "Empty"], func(i: int) -> void:
		if i == 0:
			set_model(ImportTable.new("plain ground").plain_ground(), true)
		else:
			var m := ImportTable.new("biome").model
			set_model(m, false)
			add_node("output", Vector2(600, 0))))
	var open := _menu("Open", [], func(_i: int) -> void: pass)
	open.about_to_popup.connect(_fill_open.bind(open))
	bar.add_child(open)
	var names: Array = ImportTable.biome_names(_world_table())
	bar.add_child(_menu("Import from world.json", names, func(i: int) -> void: set_model(_import(names[i]), true)))
	var save := Button.new()
	save.text = "Save"
	save.tooltip_text = "Writes %s/<name>.biome.json" % BIOMES_DIR
	save.pressed.connect(_save)
	bar.add_child(save)
	var titles := Kinds.MENU.map(func(k): return Kinds.KINDS[k]["title"])
	bar.add_child(_menu("Add node", titles, func(i: int) -> void:
		add_node(Kinds.MENU[i], (graph.scroll_offset + graph.size * 0.4) / graph.zoom)))
	var notes := Button.new()
	notes.text = "Add notes..."
	notes.tooltip_text = "A run of notes, each through a shape into a new row of the first Mix"
	notes.pressed.connect(_notes_dialog)
	bar.add_child(notes)
	var frame := Button.new()
	frame.text = "Frame"
	frame.pressed.connect(func() -> void: preview.frame(result.get("hmax", 1000.0)))
	bar.add_child(frame)

	var split := HSplitContainer.new()
	split.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(split)
	graph.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	graph.custom_minimum_size.x = 420
	graph.right_disconnects = true
	graph.minimap_enabled = false
	graph.zoom = 0.75
	graph.connection_request.connect(_on_connection_request)
	graph.disconnection_request.connect(_on_disconnection_request)
	graph.delete_nodes_request.connect(_on_delete_nodes_request)
	graph.popup_request.connect(func(at: Vector2) -> void:
		add_at = (graph.scroll_offset + at) / graph.zoom
		# The menu is a subwindow: embedded (the default), its position is the window's own; a native one's is the
		# screen's.
		var pos := Vector2i(get_global_mouse_position())
		if not get_viewport().gui_embed_subwindows:
			pos += get_window().position
		add_menu.popup(Rect2i(pos, Vector2i.ZERO)))
	split.add_child(graph)
	for k in Kinds.MENU:
		add_menu.add_item(Kinds.KINDS[k]["title"])
	add_menu.id_pressed.connect(func(i: int) -> void: add_node(Kinds.MENU[i], add_at))
	add_child(add_menu)
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right.custom_minimum_size.x = 420
	split.add_child(right)
	budget.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	budget.add_theme_color_override("font_color", Color(0.85, 0.88, 0.7))
	right.add_child(budget)
	preview.sharpened.connect(_show_budget)
	right.add_child(preview)
	right.add_child(view_box)
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status.custom_minimum_size.y = 22
	root.add_child(status)
	autosave.one_shot = true
	autosave.wait_time = 1.0
	autosave.timeout.connect(_autosave)
	add_child(autosave)


func _menu(p_text: String, p_items: Array, p_pressed: Callable) -> MenuButton:
	var m := MenuButton.new()
	m.text = p_text
	m.flat = false
	for it in p_items:
		m.get_popup().add_item(it)
	m.get_popup().id_pressed.connect(p_pressed)
	return m


func _fill_open(p_menu: MenuButton) -> void:
	var popup := p_menu.get_popup()
	popup.clear()
	for c in popup.id_pressed.get_connections():
		popup.id_pressed.disconnect(c.callable)
	# The folder only exists after the first save.
	var files: Array = []
	if DirAccess.dir_exists_absolute(BIOMES_DIR):
		files = Array(DirAccess.get_files_at(BIOMES_DIR)).filter(func(f): return f.ends_with(".biome.json"))
	for f in files:
		popup.add_item(f)
	if files.is_empty():
		popup.add_item("(nothing saved in %s yet)" % BIOMES_DIR)
		popup.set_item_disabled(0, true)
	popup.id_pressed.connect(func(i: int) -> void: set_model(_read("%s/%s" % [BIOMES_DIR, files[i]]), true))


func _notes_dialog() -> void:
	var d := ConfirmationDialog.new()
	d.title = "Add notes"
	var box := VBoxContainer.new()
	var pick := func(p_label: String, p_k: int) -> OptionButton:
		var row := HBoxContainer.new()
		row.add_child(Nodes._label(p_label, 120))
		var ob := Nodes.k_picker(p_k, model, false, func(_k: int) -> void: pass)
		row.add_child(ob)
		box.add_child(row)
		return ob
	var top: OptionButton = pick.call("from (coarsest)", 17)
	var bottom: OptionButton = pick.call("to (finest)", 7)
	var shape := OptionButton.new()
	for s in Kinds.SHAPES.slice(0, 4):
		shape.add_item(s)
	box.add_child(shape)
	d.add_child(box)
	d.confirmed.connect(func() -> void:
		var mix := Kinds.first_of(model, "mix")
		if mix == "":
			mix = Kinds.add(model, "mix", Vector2(ImportTable.COL.mix, 0.0))
		var y := 0.0
		for id in model.nodes:
			y = maxf(y, float(model.nodes[id].pos[1]) + ImportTable.ROW_H)
		var k_top := top.get_selected_id()
		var k_bottom := bottom.get_selected_id()
		ImportTable.add_notes(model, mix, maxi(k_top, k_bottom), mini(k_top, k_bottom), y, Kinds.SHAPES[shape.selected])
		rebuild_ui()
		recompile(false)
		d.queue_free())
	d.canceled.connect(d.queue_free)
	add_child(d)
	d.popup_centered()


# The view's options under the preview (saved in the biome file: its "view").
func _build_view_bar() -> void:
	for c in view_box.get_children():
		view_box.remove_child(c)
		c.queue_free()
	var view: Dictionary = model.view
	var set_view := func(p_key: String, p_value: Variant, p_recompile: bool = false) -> void:
		view[p_key] = p_value
		if p_recompile:
			recompile(true)
		else:
			_apply_view()
			preview.redraw(true)
		autosave.start()
	var q := OptionButton.new()
	for t in ["Full", "1/2", "1/3", "1/4"]:
		q.add_item(t)
	q.select(int(view.quality) - 1)
	q.tooltip_text = "Resolution once still (dragging draws at 1/3 at most)"
	q.item_selected.connect(func(i: int) -> void:
		preview.quality = i + 1
		set_view.call("quality", i + 1))
	preview.quality = int(view.quality)
	_view_item("Quality", q)
	_view_item("Steps", _spin(view.steps, 32, 2048, 1, func(v: float) -> void: set_view.call("steps", int(v))), "The most march steps per pixel")
	_view_item("Step", _spin(view.step, 0.1, 2.0, 0.05, func(v: float) -> void: set_view.call("step", v)), "The march step over its safe bound: lower if thin ridges show holes")
	var sh := CheckBox.new()
	sh.button_pressed = view.shadows
	sh.toggled.connect(func(on: bool) -> void: set_view.call("shadows", on))
	_view_item("Shadows", sh)
	_view_item("Sun", _spin(view.sun_az, 0.0, 360.0, 5.0, func(v: float) -> void: set_view.call("sun_az", v)), "Azimuth, degrees")
	_view_item("high", _spin(view.sun_el, 2.0, 90.0, 1.0, func(v: float) -> void: set_view.call("sun_el", v)), "Elevation, degrees")
	_view_item("Exaggerate", _spin(view.exag, 0.1, 10.0, 0.1, func(v: float) -> void: set_view.call("exag", v, true)), "Heights times this (the view only)")
	var w := CheckBox.new()
	w.button_pressed = view.water_on
	w.toggled.connect(func(on: bool) -> void: set_view.call("water_on", on))
	_view_item("Water", w)
	_view_item("at m", _spin(view.water, -5000.0, 5000.0, 1.0, func(v: float) -> void: set_view.call("water", v)))
	_view_item("Contours m", _spin(view.contour, 0.0, 1000.0, 1.0, func(v: float) -> void: set_view.call("contour", v)), "A line every this many metres (0: none)")
	var origin: Array = model.get("origin_m", [0.0, 0.0])
	model.origin_m = origin
	_view_item("Centre x m", _spin(origin[0], -1e7, 1e7, 100.0, func(v: float) -> void:
		model.origin_m[0] = v
		recompile(false)), "Where in the world the preview sits (the notes move under it)")
	_view_item("z m", _spin(origin[1], -1e7, 1e7, 100.0, func(v: float) -> void:
		model.origin_m[1] = v
		recompile(false)))


func _view_item(p_label: String, p_control: Control, p_tip: String = "") -> void:
	var row := HBoxContainer.new()
	var l := Label.new()
	l.text = p_label
	row.add_child(l)
	row.add_child(p_control)
	row.tooltip_text = p_tip
	view_box.add_child(row)


func _spin(p_value: float, p_min: float, p_max: float, p_step: float, p_set: Callable) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = p_min
	s.max_value = p_max
	s.step = p_step
	s.value = p_value
	s.custom_minimum_size.x = 80
	s.value_changed.connect(p_set)
	return s


func _apply_view() -> void:
	var v: Dictionary = model.view
	var az := deg_to_rad(float(v.sun_az))
	var el := deg_to_rad(float(v.sun_el))
	preview.set_uniforms({
		"u_max_steps": int(v.steps),
		# The slope bound is a worst case (every note at its steepest at once): twice its step still holds on
		# the imported biomes, at half the frame.
		"u_step": float(v.step) * 2.0,
		"u_shadows": bool(v.shadows),
		"u_sun": Vector3(cos(el) * sin(az), sin(el), cos(el) * cos(az)),
		"u_water_on": bool(v.water_on),
		"u_water": float(v.water),
		"u_contour": float(v.contour),
	})


func _shot(p_path: String) -> void:
	preview.quality = 1
	# Frames drawn with the whole view redrawn each one: their times are the view's cost (the first compiles the shader).
	var ms := []
	var t := Time.get_ticks_usec()
	for i in 6:
		preview.draw_full()
		await RenderingServer.frame_post_draw
		var now := Time.get_ticks_usec()
		ms.append(snappedf((now - t) / 1000.0, 0.1))
		t = now
	get_viewport().get_texture().get_image().save_png(p_path)
	preview.image().save_png(p_path.get_basename() + "_view.png")
	# Then one pass as the window draws it, the coarse layer and the strips: no frame should wait on the whole view.
	var pass_ms := []
	preview.redraw()
	t = Time.get_ticks_usec()
	while preview.strip_y >= 0 and pass_ms.size() < 200:
		await RenderingServer.frame_post_draw
		var now := Time.get_ticks_usec()
		pass_ms.append(snappedf((now - t) / 1000.0, 0.1))
		t = now
	print("MIXER_SHOT: ", JSON.stringify({"path": p_path, "view_px": preview.sharp.size, "frame_ms": ms, "pass_ms": pass_ms, "hmax": result.hmax, "lip": result.lip, "errors": result.errors}))
	get_tree().quit(0)


# --- the headless check ------------------------------------------------------------------------------------------

func _check() -> void:
	var report := {"pins": Octaves.check_pins(), "graphs": {}}
	var ok: bool = report.pins.is_empty()
	var graphs := {"plain ground": ImportTable.new("plain ground").plain_ground(), "every kind": _every_kind()}
	var table := _world_table()
	for name in ImportTable.biome_names(table):
		graphs["world.json " + name] = ImportTable.new(name).import_biome(table).model
	for g in graphs:
		var r := Compile.new().run(graphs[g])
		var again = JSON.parse_string(JSON.stringify(graphs[g], "", true, true))
		var r2 := Compile.new().run(again)
		# The shader gets float32: a double's last digits through the file don't count (a curve coefficient's 0 may come
		# back as 3e-14).
		var same := func(a, b) -> bool:
			if a is float and b is float:
				return absf(a - b) <= 1e-6 * maxf(1.0, absf(a))
			if a is PackedFloat32Array and b is PackedFloat32Array and a.size() == b.size():
				var scale := 1.0
				for x in a:
					scale = maxf(scale, absf(x))
				for i in a.size():
					if absf(a[i] - b[i]) > 1e-6 * scale:
						return false
				return true
			return a == b
		var differ: Array = r.uniforms.keys().filter(func(key): return not r2.uniforms.has(key) or not same.call(r.uniforms[key], r2.uniforms[key]))
		var entry := {"ok": r.ok, "errors": r.errors, "nodes": graphs[g].nodes.size(), "hmax": r.hmax, "same_after_file": r2.code == r.code and differ.is_empty()}
		# A Height curve's editor spans its input's bound; its points may reach past it (an import's span the table's
		# field range): [input bound, the points' widest |h|].
		for id in r.curve_in:
			var pts: Array = Kinds.param(graphs[g].nodes[id], "points")
			entry["curve_" + id] = [snappedf(r.curve_in[id], 0.1), pts.reduce(func(m, p): return maxf(m, absf(p[0])), 0.0)]
		if not differ.is_empty():
			entry["uniforms_differ"] = differ.slice(0, 5).map(func(key): return [key, r.uniforms[key], r2.uniforms.get(key)])
		if g == "every kind":
			var missing := []
			for needle in ["(F[1] * N[1])", "shape(u_", "w_n", "curve_n", "band", "F[0] * (shape("]:
				if not r.code.contains(needle):
					missing.append(needle)
			entry["missing"] = missing
			entry["warp_rows"] = r.uniforms.u_n_warp
			# The note warp's notes: earlier rows of the table than the note they move.
			entry["note_warped"] = r.budget.note_warped
			ok = ok and missing.is_empty() and r.uniforms.u_n_warp > 0 and r.budget.note_warped == 1 and Array(r.uniforms.u_jx).max() >= 0
		ok = ok and r.ok and entry.same_after_file
		report.graphs[g] = entry
	report["cuts"] = _check_cuts(table)
	report["old_kinds"] = _check_old_kinds()
	ok = ok and report.cuts.ok and report.old_kinds.ok
	report["ok"] = ok
	print("MIXER: ", JSON.stringify(report))
	# The window's parts, never added to the tree in a check.
	for c in [graph, preview, status, budget, name_edit, add_menu, autosave, view_box]:
		c.free()
	await get_tree().process_frame
	get_tree().quit(0 if ok else 1)


# A file saved with a kind since removed (a Terrace before the output) loads as the graph without it: the same nodes,
# the same code.
func _check_old_kinds() -> Dictionary:
	var want := ImportTable.new("plain ground").plain_ground()
	var m = JSON.parse_string(JSON.stringify(want))
	var into_out: Dictionary = m.links.filter(func(l): return l.to == Kinds.first_of(m, "output"))[0]
	m.nodes["old"] = {"kind": "terrace", "pos": [0, 0], "params": {"step_m": 40.0}}
	m.links.append({"from": into_out.from, "from_port": into_out.from_port, "to": "old", "to_port": 0})
	into_out.from = "old"
	into_out.from_port = 0
	Kinds.drop_unknown(m)
	var same: bool = Compile.new().run(m).code == Compile.new().run(want).code
	return {"ok": same and m.nodes.size() == want.nodes.size(), "nodes": [want.nodes.size(), m.nodes.size()], "same_code": same}


# Removing a Mix row (the mountain's first: its shape and note go, its multifractal stays for the chain) and a note
# that plays a row (plain ground's first: its row goes): one row less, the other rows' links one port up, no gaps.
func _check_cuts(p_table: Dictionary) -> Dictionary:
	var out := {}
	var cases := {
		"row": [ImportTable.new("mountain").import_biome(p_table).model, func(m: Dictionary, mix: String) -> void: Kinds.cut_row(m, mix, 0)],
		"note": [ImportTable.new("plain ground").plain_ground(), func(m: Dictionary, _mix: String) -> void: Kinds.cut_nodes(m, [Kinds.first_of(m, "note")])],
	}
	var ok := true
	for c in cases:
		var m: Dictionary = cases[c][0]
		var mix := Kinds.first_of(m, "mix")
		var rows := Kinds.rows(m.nodes[mix]).size()
		var nodes: int = m.nodes.size()
		cases[c][1].call(m, mix)
		var ports: Array = m.links.filter(func(l): return l.to == mix).map(func(l): return int(l.to_port))
		ports.sort()
		var r := Compile.new().run(m)
		var entry := {"rows": [rows, Kinds.rows(m.nodes[mix]).size()], "nodes": [nodes, m.nodes.size()], "compiles": r.ok}
		entry.ok = r.ok and entry.rows[1] == rows - 1 and entry.nodes[1] < nodes and ports == range(ports.size())
		ok = ok and entry.ok
		out[c] = entry
	out.ok = ok
	return out


# A graph with every kind of node, wired: two warped notes, shaped, chained, mixed, combined, curved, carved.
func _every_kind() -> Dictionary:
	var b := ImportTable.new("every kind")
	var warp := b.add("warp", Vector2.ZERO)
	var a := b.add("note", Vector2.ZERO, {"k": 15, "warp_m": 200.0})
	var c := b.add("note", Vector2.ZERO, {"k": 14})
	var d := b.add("note", Vector2.ZERO, {"k": 12, "warp_m": 30.0})
	var nw := b.add("note_warp", Vector2.ZERO)
	var sa := b.add("shape", Vector2.ZERO, {"preset": "ridged"})
	var sc := b.add("shape", Vector2.ZERO, {"preset": "coeffs", "c2": 1.0})
	var ma := b.add("multifractal", Vector2.ZERO)
	var mc := b.add("multifractal", Vector2.ZERO)
	var mix := b.add("mix", Vector2.ZERO, {"rows": [Kinds.new_row(), Kinds.new_row(0.5)]})
	var k := b.add("constant", Vector2.ZERO, {"value": 2.0})
	var mul := b.add("multiply", Vector2.ZERO)
	var add := b.add("add", Vector2.ZERO)
	var cv := b.add("curve", Vector2.ZERO, {"points": [[-500.0, -300.0], [0.0, 50.0], [500.0, 900.0]]})
	var carve := b.add("carve", Vector2.ZERO)
	var out := b.add("output", Vector2.ZERO)
	b.link(warp, 0, a, 0)
	b.link(a, 0, nw, 0)
	b.link(c, 0, nw, 2)
	b.link(nw, 0, d, 0)
	b.link(d, 0, add, 1)
	b.link(a, 0, sa, 0)
	b.link(c, 0, sc, 0)
	b.link(sa, 0, ma, 0)
	b.link(sc, 0, mc, 0)
	b.link(ma, 1, mc, 1)
	b.link(ma, 0, mix, 0)
	b.link(mc, 0, mix, 1)
	b.link(mix, 0, mul, 0)
	b.link(k, 0, mul, 1)
	b.link(mul, 0, add, 0)
	b.link(add, 0, cv, 0)
	b.link(cv, 0, out, 0)
	b.link(c, 0, carve, 0)
	b.link(carve, 0, out, 1)
	return b.model
