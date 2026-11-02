extends Control
## The biome mixer: build a world's biomes from the ground's octaves (notes) in a node graph, seen through the engine
## beside it. Two modes: Biome (one biome's notes in the graph, the view that biome alone) and World (the climate chart:
## every biome's place in the fields' space, the view the whole world, its biomes blended).
## The graph (the model: nodes, links) is the truth. Every edit exports it as a world file's text (export_table.gd, the
## same as the World menu writes) and the engine draws it (engine_view.gd: ClimateReportNode.preview, the game's own
## cascade); "Open in game" runs the game on it. The mixer computes no terrain of its own.
## Run: tools/biome_mixer.sh [-- flags]  (godot --path adapter/godot/project res://debug/biome_mixer/biome_mixer.tscn)
## Flags:
##   --open=PATH        a biome file (res://climate/biomes/*.biome.json) into the world's biome of its name (else the
##                      current one); the world: the autosave, else res://climate/world.json
##   --import=NAME      res://climate/world.json, its biome NAME in the graph
##   --world            start in the World mode
##   --set=ID.PARAM=V   set a node's parameter after loading (V as JSON: 1.5, true, "plain"), or view.KEY=V a view
##                      setting (view.extent_km=40); repeatable
##   --shot=PATH        draw once (the sharp map), save the window to PATH (and the view alone to PATH_view.png), quit
##   --check            headless self-test: the templates and every world.json biome exported and drawn by the engine,
##                      files round-tripped, world.json exported back; prints MIXER: {json} and quits 0 if ok
## The World menu exports the world over res://climate/world.json (and climate.json), the game's climate (export_table.gd).

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const Nodes := preload("res://debug/biome_mixer/nodes.gd")
const ImportTable := preload("res://debug/biome_mixer/import_table.gd")
const EngineView := preload("res://debug/biome_mixer/engine_view.gd")
const World := preload("res://debug/biome_mixer/world.gd")
const WorldPanel := preload("res://debug/biome_mixer/world_panel.gd")
const Checks := preload("res://debug/biome_mixer/checks.gd")
const ExportTable := preload("res://debug/biome_mixer/export_table.gd")

const WORLD := Octaves.WORLD
const BIOMES_DIR := "res://climate/biomes"
const WORLDS_DIR := "res://climate/worlds"
const AUTOSAVE := "user://biome_mixer_autosave.world.json"
# The autosave of one biome, before the World mode: read once when there is no world autosave yet.
const OLD_AUTOSAVE := "user://biome_mixer_autosave.biome.json"
# Pixels between an imported graph's columns and rows.
const LAYOUT_GAP := 60.0
# A biome's view: the map's extent (the World mode's is the world's extent_m).
const VIEW_DEFAULTS := {"extent_km": 20.0}
# The world "Open in game" runs.
const GAME_WORLD := "user://biome_mixer_game.world.json"

var model := {}
var curve_ui := {}  # Height curve id -> its editor, told the relief's range after each map

var graph := GraphEdit.new()
var preview := EngineView.new()
var status := Label.new()
var name_edit := LineEdit.new()
var add_menu := PopupMenu.new()
var add_at := Vector2.ZERO
var autosave := Timer.new()
var refit := false  # the next World mode resample fits the chart's axes to the samples
var compile_queued := false
var table := {}  # res://climate/world.json, parsed once
var world := {}  # the world (world.gd): every biome, its place and its graph; model is its current biome's graph
var world_panel := WorldPanel.new()
var biome_pick := OptionButton.new()
var mode_pick := OptionButton.new()


func _ready() -> void:
	var args := OS.get_cmdline_user_args()
	if "--check" in args:
		_check()
		return
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
	var pick := ""
	var mode := ""
	for a in args:
		if a.begins_with("--open="):
			asked = a
			start = _read(a.trim_prefix("--open="))
		elif a.begins_with("--import="):
			asked = a
			pick = a.trim_prefix("--import=")
		elif a == "--world":
			mode = "world"
		elif a.begins_with("--set="):
			sets.append(a.trim_prefix("--set="))
	var w := _read(AUTOSAVE, true) if shot == "" and pick == "" else {}
	if w.is_empty() and shot == "" and pick == "" and start.is_empty():
		start = _read(OLD_AUTOSAVE)
	if w.is_empty():
		w = _fresh_world()
	var names: Array = w.biomes.map(func(b): return b.name)
	if pick != "":
		w.current = names.find(pick)
	if not start.is_empty():
		var found := names.find(String(start.get("name", "")))
		w.current = found if found >= 0 else int(w.get("current", 0))
		w.biomes[w.current].model = start
	if asked != "" and (int(w.current) < 0 or (asked.begins_with("--open=") and start.is_empty())):
		push_error("%s: no such biome file or world.json biome" % asked)
		if shot != "":
			get_tree().quit(1)
			return
		w.current = 0
	if mode != "":
		w.mode = mode
	elif pick != "" or not start.is_empty():
		w.mode = "biome"
	set_world(w, true)
	for s in sets:
		_apply_set(s)
	if shot != "":
		await _shot(shot)


# --- the model ---------------------------------------------------------------------------------------------------

# The world (world.gd) and its mode: every biome's place on the chart, the current one's graph.
func set_world(p_world: Dictionary, p_frame: bool) -> void:
	world = p_world
	model = {}
	world.erase("warnings")
	world.current = clampi(int(world.get("current", 0)), 0, world.biomes.size() - 1)
	# One view for the World mode: the Biome mode's settings and the World mode's own (the map, the chart).
	var view: Dictionary = VIEW_DEFAULTS.duplicate()
	view.merge(World.WORLD_VIEW)
	view.merge(world.get("view", {}), true)
	world.view = view
	# Every biome as its graph would load it alone: run_world compiles them all, not only the one in the graph.
	for b in world.biomes:
		Kinds.normalise(b.model)
	world_panel.set_world(world)
	biome_pick.clear()
	for b in world.biomes:
		biome_pick.add_item(b.name)
	biome_pick.select(world.current)
	_enter(String(world.get("mode", "biome")), p_frame)


# The Biome mode (the graph of the current biome, the view of it alone) or the World mode (the chart, the world view).
func _enter(p_mode: String, p_frame: bool) -> void:
	world.mode = p_mode
	mode_pick.select(0 if p_mode == "biome" else 1)
	graph.visible = p_mode == "biome"
	world_panel.visible = p_mode == "world"
	biome_pick.disabled = p_mode == "world"
	if p_mode == "biome" or model.is_empty():
		# The graph (hidden in the World mode) always holds the current biome: the budget line and --set read it.
		set_model(world.biomes[world.current].model, p_frame and p_mode == "biome", p_mode == "biome")
	if p_mode == "biome":
		return
	preview.set_extent(float(world.view.extent_m) / 1000.0)
	refit = true
	recompile()


func world_mode() -> bool:
	return world.get("mode", "biome") == "world"


# The view settings of what the preview shows: the world's in the World mode, the biome's in the Biome mode.
func view_settings() -> Dictionary:
	return world.view if world_mode() else model.view


# p_compile false: the World mode's entry, which compiles the world itself.
func set_model(p_model: Dictionary, p_frame: bool, p_compile: bool = true) -> void:
	model = p_model
	if not world.is_empty():
		world.biomes[world.current].model = model
	Kinds.normalise(model)
	var view: Dictionary = VIEW_DEFAULTS.duplicate()
	view.merge(model.get("view", {}), true)
	model.view = view
	name_edit.text = model.get("name", "biome")
	if not world_mode():
		preview.set_extent(float(view.extent_km))
	rebuild_ui()
	if model.get("layout", "") == "grid":
		model.erase("layout")
		_spread_layout()
	if p_compile:
		recompile()
	if p_frame and p_compile:
		_fit_graph()


# An importer's grid (ImportTable's columns, ROW_H rows) spread to the built nodes' sizes: each column as wide as its
# widest node, the rows as tall as the tallest note, shape or multifractal, LAYOUT_GAP between them.
func _spread_layout() -> void:
	var widths := {}
	var row_h := 0.0
	for id in model.nodes:
		var size := (graph.get_node(NodePath(id)) as GraphNode).get_combined_minimum_size()
		var x := float(model.nodes[id].pos[0])
		widths[x] = maxf(widths.get(x, 0.0), size.x)
		if model.nodes[id].kind in ["note", "shape", "multifractal"]:
			row_h = maxf(row_h, size.y)
	var xs := widths.keys()
	xs.sort()
	var at := {}
	var left := 0.0
	for x in xs:
		at[x] = left
		left += widths[x] + LAYOUT_GAP
	var scale := (row_h + LAYOUT_GAP) / ImportTable.ROW_H
	for id in model.nodes:
		var p: Array = model.nodes[id].pos
		var to := Vector2(at[float(p[0])], float(p[1]) * scale)
		model.nodes[id].pos = [to.x, to.y]
		(graph.get_node(NodePath(id)) as GraphNode).position_offset = to


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
		Nodes.refresh(graph.get_node(NodePath(p_id)), node)
	queue_recompile()


# Several edits in a frame (a slider, a chart drag's motion) compile once, at the frame's end.
func queue_recompile() -> void:
	if not compile_queued:
		compile_queued = true
		recompile.call_deferred()


func recompile() -> void:
	compile_queued = false
	var r := _preview_world()
	var lines := PackedStringArray()
	for e in r.errors:
		lines.append("Error: " + e)
	var warned := {}
	for w in r.warnings:
		if not warned.has(w):
			warned[w] = true
			lines.append(w)
	if r.errors.is_empty():
		var map := preview.show_world(JSON.stringify(r.world), world_mode())
		if String(map.get("error", "")) != "":
			lines.append("Error: the engine refuses it: " + String(map.error))
	if world_mode():
		resample(refit)
		refit = false
	var span := curve_span()
	for id in curve_ui:
		curve_ui[id].set_span(span)
	var ok := lines.is_empty() or not lines[0].begins_with("Error")
	if world_mode():
		lines.append("%d biomes over %d fields. The chart: drag a biome's box to move it, its edge to resize; double-click a biome in the list to edit its notes." % [world.biomes.size(), world.fields.size()])
	else:
		lines.append("%d nodes. Right-click the graph: add a node; x or Delete removes one (not a note: the ground's notes are fixed); a note's warp switch moves it by the climate's warp." % model.nodes.size())
	status.text = "  ".join(lines)
	status.add_theme_color_override("font_color", Color(1.0, 0.5, 0.45) if not ok else Color(0.8, 0.82, 0.86))
	autosave.start()


# What the view draws, as the engine reads it (ExportTable.preview_world): the world, or the current biome alone.
func _preview_world() -> Dictionary:
	return ExportTable.preview_world(world, _world_table(), -1 if world_mode() else int(world.current))


# The input a Height curve's editor shows (metres either side of 0): the relief the view last saw.
func curve_span() -> float:
	return maxf(float(preview.last.get("relief_m", 0.0)), 10.0)


func rebuild_ui() -> void:
	graph.clear_connections()
	for c in graph.get_children():
		if c is GraphNode:
			graph.remove_child(c)
			c.queue_free()
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
	recompile()
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
	recompile()


func _on_disconnection_request(p_from: StringName, p_from_port: int, p_to: StringName, p_to_port: int) -> void:
	for l in model.links.duplicate():
		if l.from == String(p_from) and int(l.from_port) == p_from_port and l.to == String(p_to) and int(l.to_port) == p_to_port:
			model.links.erase(l)
	graph.disconnect_node(p_from, p_from_port, p_to, p_to_port)
	recompile()


func _on_delete_nodes_request(p_names: Array) -> void:
	delete_nodes(p_names.map(func(n): return String(n)))


func delete_nodes(p_ids: Array) -> void:
	Kinds.cut_nodes(model, p_ids)
	rebuild_ui()
	recompile()
	if p_ids.any(func(id): return model.nodes.has(id) and model.nodes[id].kind == "note"):
		status.text = "The notes are the ground's: they stay (a Mix row's volume 0 or M silences one).  " + status.text


func remove_mix_row(p_mix: String, p_row: int) -> void:
	Kinds.cut_row(model, p_mix, p_row)
	rebuild_ui()
	recompile()


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
		view_settings()[key] = v if v != null else raw
		# The World mode applies the map and the chart's extent on entering it.
		if world_mode():
			_enter("world", true)
		else:
			recompile()
	else:
		set_params(id, {key: v if v != null else raw}, true)


# --- files -------------------------------------------------------------------------------------------------------

# A biome file (p_world false) or a world file, {} if it isn't one.
func _read(p_path: String, p_world: bool = false) -> Dictionary:
	if not FileAccess.file_exists(p_path):
		return {}
	var d = JSON.parse_string(FileAccess.get_file_as_string(p_path))
	if not d is Dictionary:
		return {}
	var ok: bool = (d.get("kind", "") == "world" and not d.get("biomes", []).is_empty()) if p_world else d.has("nodes")
	return d if ok else {}


func _world_table() -> Dictionary:
	if table.is_empty():
		table = World.read_table(WORLD)
	return table


# res://climate/world.json as a world (each biome imported).
func _fresh_world() -> Dictionary:
	var w := World.from_table(_world_table())
	for m in w.warnings:
		print("import: ", m)
	return w


func _import(p_name: String) -> Dictionary:
	if not p_name in ImportTable.biome_names(_world_table()):
		return {}
	var r: Dictionary = ImportTable.new(p_name).import_biome(table)
	for w in r.warnings:
		print("import %s: %s" % [p_name, w])
	return r.model


func _write(p_path: String, p_data: Dictionary) -> bool:
	return ExportTable.store(p_path, JSON.stringify(p_data, "\t", false, true))


func _autosave() -> void:
	_write(AUTOSAVE, world)


func _save() -> void:
	var file := String(model.get("name", "biome")).validate_filename().replace(" ", "_")
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(BIOMES_DIR))
	var path := "%s/%s.biome.json" % [BIOMES_DIR, file]
	status.text = ("Saved " + path) if _write(path, model) else ("Could not write " + path)


func _save_world() -> void:
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(WORLDS_DIR))
	var path := "%s/%s.world.json" % [WORLDS_DIR, String(world.get("name", "world")).validate_filename()]
	status.text = ("Saved " + path) if _write(path, world) else ("Could not write " + path)


# The world over WORLD, the game's climate table (ExportTable.write: the engine checks it first).
func _export_world() -> void:
	var r := ExportTable.write(world, _world_table(), WORLD)
	for m in r.warnings:
		print("export: ", m)
	if r.written:
		table = r.table
	status.text = r.message


# The chart's cloud: the world's fields on its two axes at the view's points (the engine's samples).
func resample(p_fit: bool) -> void:
	if not world_mode():
		return
	var c := world_panel.chart
	c.set_samples(preview.samples(c.fx, c.fy), p_fit)


# The game on this world (its biomes as the view shows them, the rest of the game world as it is): a world file next
# to the user's data, run by the same Godot.
func _open_game() -> void:
	var r := ExportTable.preview_world(world, _world_table())
	if not r.errors.is_empty():
		status.text = "Not opened: " + "  ".join(r.errors)
		return
	if not ExportTable.store(GAME_WORLD, ExportTable.text(r.world)):
		status.text = "Could not write " + GAME_WORLD
		return
	var path := ProjectSettings.globalize_path(GAME_WORLD)
	var pid := OS.create_process(OS.get_executable_path(), ["--path", ProjectSettings.globalize_path("res://"), "--", "--world=" + path])
	status.text = ("The game runs on %s (pid %d)." % [path, pid]) if pid > 0 else "Could not start the game."


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
	mode_pick.add_item("Biome")
	mode_pick.add_item("World")
	mode_pick.tooltip_text = "Biome: the current biome's notes and it alone in the view. World: every biome's place on the climate chart, the whole world in the view."
	mode_pick.item_selected.connect(func(i: int) -> void: _enter("biome" if i == 0 else "world", true))
	bar.add_child(mode_pick)
	biome_pick.tooltip_text = "The biome the graph edits"
	biome_pick.item_selected.connect(func(i: int) -> void:
		world.current = i
		world_panel.select(i)
		set_model(world.biomes[i].model, true))
	bar.add_child(biome_pick)
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
	bar.add_child(_menu("Import from world.json", names, func(i: int) -> void: _load_biome(_import(names[i]), names[i])))
	var worlds := _menu("World", [], func(_i: int) -> void: pass)
	worlds.about_to_popup.connect(_fill_worlds.bind(worlds))
	bar.add_child(worlds)
	var save := Button.new()
	save.text = "Save"
	save.tooltip_text = "Writes %s/<name>.biome.json" % BIOMES_DIR
	save.pressed.connect(_save)
	bar.add_child(save)
	var titles := Kinds.MENU.map(func(k): return Kinds.KINDS[k]["title"])
	bar.add_child(_menu("Add node", titles, func(i: int) -> void:
		add_node(Kinds.MENU[i], (graph.scroll_offset + graph.size * 0.4) / graph.zoom)))

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
	var left := VBoxContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	graph.size_flags_vertical = Control.SIZE_EXPAND_FILL
	left.add_child(graph)
	left.add_child(world_panel)
	world_panel.visible = false
	world_panel.placed.connect(queue_recompile)
	world_panel.edit_notes.connect(func(b: int) -> void:
		world.current = b
		biome_pick.select(b)
		_enter("biome", true))
	world_panel.axes_changed.connect(func() -> void: resample(true))
	split.add_child(left)
	for k in Kinds.MENU:
		add_menu.add_item(Kinds.KINDS[k]["title"])
	add_menu.id_pressed.connect(func(i: int) -> void: add_node(Kinds.MENU[i], add_at))
	add_child(add_menu)
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right.custom_minimum_size.x = 420
	split.add_child(right)
	preview.view_changed.connect(func() -> void:
		view_settings()["extent_m" if world_mode() else "extent_km"] = preview.extent_km * 1000.0 if world_mode() else preview.extent_km
		refit = world_mode()
		preview.redraw()
		if world_mode():
			resample(refit)
			refit = false)
	preview.open_game.connect(_open_game)
	right.add_child(preview)
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


# The World menu: world.json again, or a saved world; save this one.
func _fill_worlds(p_menu: MenuButton) -> void:
	var popup := p_menu.get_popup()
	popup.clear()
	for c in popup.id_pressed.get_connections():
		popup.id_pressed.disconnect(c.callable)
	var actions := [
		["Reload %s" % WORLD, func() -> void: set_world(_fresh_world(), true)],
		["Save this world to %s" % WORLDS_DIR, _save_world],
		["Export this world to %s (the game's)" % WORLD, _export_world],
	]
	for a in actions:
		popup.add_item(a[0])
	var files: Array = []
	if DirAccess.dir_exists_absolute(WORLDS_DIR):
		files = Array(DirAccess.get_files_at(WORLDS_DIR)).filter(func(f): return f.ends_with(".world.json"))
	for f in files:
		popup.add_item("Open " + f)
	popup.id_pressed.connect(func(i: int) -> void:
		if i < actions.size():
			actions[i][1].call()
		else:
			var w := _read("%s/%s" % [WORLDS_DIR, files[i - actions.size()]], true)
			if not w.is_empty():
				set_world(w, true))


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
	popup.id_pressed.connect(func(i: int) -> void: _load_biome(_read("%s/%s" % [BIOMES_DIR, files[i]]), files[i]))


# A biome's graph from a file or world.json into the world: into the world biome of its name (else the current one),
# shown in the Biome mode.
func _load_biome(p_model: Dictionary, p_what: String) -> void:
	if p_model.is_empty():
		status.text = "%s: not a biome file." % p_what
		return
	var found: int = world.biomes.map(func(b): return b.name).find(String(p_model.get("name", "")))
	if found >= 0:
		world.current = found
		biome_pick.select(found)
		world_panel.select(found)
	world.biomes[world.current].model = p_model
	model = {}
	_enter("biome", true)


func _shot(p_path: String) -> void:
	var r := _preview_world()
	var t := Time.get_ticks_usec()
	var map := preview.show_world(JSON.stringify(r.world), world_mode(), true)
	var ms := snappedf((Time.get_ticks_usec() - t) / 1000.0, 0.1)
	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw
	get_viewport().get_texture().get_image().save_png(p_path)
	preview.image().save_png(p_path.get_basename() + "_view.png")
	print("MIXER_SHOT: ", JSON.stringify({"path": p_path, "map_ms": ms, "us_per_point": map.get("us_per_point", 0.0), "error": map.get("error", ""), "errors": r.errors}))
	get_tree().quit(0)


# --- the headless check ------------------------------------------------------------------------------------------

func _check() -> void:
	var report := Checks.run(_world_table())
	print("MIXER: ", JSON.stringify(report))
	# The window's parts, never added to the tree in a check.
	for c in [graph, preview, status, name_edit, add_menu, autosave, world_panel, biome_pick, mode_pick]:
		c.free()
	await get_tree().process_frame
	get_tree().quit(0 if report.ok else 1)
