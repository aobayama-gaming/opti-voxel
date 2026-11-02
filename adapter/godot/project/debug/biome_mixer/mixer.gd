extends Control
## The biome mixer: a world's biomes edited as the engine reads them, seen through the engine beside them. Two modes:
## Biome (the current biome's graph, the view that biome alone) and World (the climate chart: every biome's place in the
## fields' space, the view the whole world, its biomes blended).
## The world file is the truth (world.gd): a biome's graph is its "graph" in world.json, node for node (graph.gd), its
## nodes' kinds, inputs and parameters the engine's (EngineInfo.kinds). Every edit sends the world's text to the engine,
## which draws it (engine_view.gd: ClimateReportNode.preview, the game's own cascade) or says why not; "Open in game" runs
## the game on it. The mixer computes no terrain and keeps no model of its own.
## Run: tools/biome_mixer.sh [-- flags]  (godot --path adapter/godot/project res://debug/biome_mixer/biome_mixer.tscn)
## Flags:
##   --biome=NAME       start on the world's biome NAME (the world: the autosave, else res://climate/world.json)
##   --world            start in the World mode
##   --set=ID.KEY=V     set a node's key in the current biome's graph after loading (V as JSON: 1.5, "ridged"), or
##                      view.KEY=V a view setting (view.extent_km=40); repeatable
##   --shot=PATH        draw once (the sharp map; in 3D, the terrain's load then two restarts, timed), save the window to
##                      PATH (and the view alone to PATH_view.png), quit
##   --check            headless self-test (checks.gd); prints MIXER: {json} and quits 0 if ok
## The World menu exports the world over res://climate/world.json, the game's (the engine checks it first).

const EngineInfo := preload("res://debug/biome_mixer/engine_info.gd")
const Graph := preload("res://debug/biome_mixer/graph.gd")
const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Nodes := preload("res://debug/biome_mixer/nodes.gd")
const EngineView := preload("res://debug/biome_mixer/engine_view.gd")
const World := preload("res://debug/biome_mixer/world.gd")
const WorldPanel := preload("res://debug/biome_mixer/world_panel.gd")
const Checks := preload("res://debug/biome_mixer/checks.gd")

const WORLD := EngineInfo.WORLD
const WORLDS_DIR := "res://climate/worlds"
const AUTOSAVE := "user://biome_mixer_autosave.world.json"
# The Biome mode's view (the World mode's is WorldView's).
const VIEW_DEFAULTS := {"extent_km": 20.0}
# The world "Open in game" runs.
const GAME_WORLD := "user://biome_mixer_game.world.json"

var world := {}  # the world (world.gd): the world file and the mixer's keys
var model := {}  # the current biome: world.biomes[world.current], its "graph" the one shown
var curve_ui := {}  # Height curve id -> its editor, told the relief's range after each map
# Mute and solo of the mix rows, the view's only: "biome/mix/row" -> {mute, solo}.
var row_flags := {}

var graph := GraphEdit.new()
var preview := EngineView.new()
var status := Label.new()
var name_edit := LineEdit.new()
var add_menu := PopupMenu.new()
var add_at := Vector2.ZERO
var autosave := Timer.new()
var refit := false  # the next World mode resample fits the chart's axes to the samples
var compile_queued := false
var world_panel := WorldPanel.new()
var biome_pick := OptionButton.new()
var mode_pick := OptionButton.new()


func _ready() -> void:
	var args := OS.get_cmdline_user_args()
	if "--check" in args:
		_check()
		return
	var sets := []
	var shot := ""
	var pick := ""
	var mode := ""
	for a in args:
		if a.begins_with("--shot="):
			shot = a.trim_prefix("--shot=")
		elif a.begins_with("--biome="):
			pick = a.trim_prefix("--biome=")
		elif a == "--world":
			mode = "world"
		elif a.begins_with("--set="):
			sets.append(a.trim_prefix("--set="))
	# The window redraws only on input or a change (an idle GPU); a shot needs every frame drawn.
	OS.low_processor_usage_mode = shot == ""
	var screen := DisplayServer.screen_get_usable_rect(get_window().current_screen)
	get_window().size = Vector2i(Vector2(screen.size) * 0.85)
	get_window().position = screen.position + (screen.size - get_window().size) / 2
	_build_ui()
	# A shot leaves the session's autosave alone (it may edit the graph with --set).
	if shot != "":
		autosave.timeout.disconnect(_autosave)
	var w := World.read(AUTOSAVE) if shot == "" else {}
	if w.is_empty():
		w = World.read(WORLD)
	if pick != "":
		var found: int = w.biomes.map(func(b): return b.name).find(pick)
		if found < 0:
			push_error("--biome=%s: no such biome" % pick)
			if shot != "":
				get_tree().quit(1)
				return
		else:
			w.current = found
			w.mode = "biome"
	if mode != "":
		w.mode = mode
	set_world(w, true)
	for s in sets:
		_apply_set(s)
	if shot != "":
		await _shot(shot)


# --- the world ---------------------------------------------------------------------------------------------------

func set_world(p_world: Dictionary, p_frame: bool) -> void:
	world = p_world
	model = {}
	row_flags.clear()
	var view: Dictionary = VIEW_DEFAULTS.duplicate()
	view.merge(World.WORLD_VIEW)
	view.merge(world.get("view", {}), true)
	world.view = view
	world_panel.set_world(world)
	_fill_biome_pick()
	_enter(String(world.get("mode", "biome")), p_frame)


func _fill_biome_pick() -> void:
	biome_pick.clear()
	for b in world.biomes:
		biome_pick.add_item(b.name)
	biome_pick.select(world.current)


# The Biome mode (the graph of the current biome, the view of it alone) or the World mode (the chart, the world view).
func _enter(p_mode: String, p_frame: bool) -> void:
	world.mode = p_mode
	mode_pick.select(0 if p_mode == "biome" else 1)
	graph.visible = p_mode == "biome"
	world_panel.visible = p_mode == "world"
	biome_pick.disabled = p_mode == "world"
	# The graph (hidden in the World mode) always holds the current biome: --set reads it.
	set_model(world.biomes[world.current], p_frame and p_mode == "biome", false)
	preview.set_extent(float(world.view.extent_m) / 1000.0 if p_mode == "world" else float(world.view.extent_km))
	refit = p_mode == "world"
	recompile()


func world_mode() -> bool:
	return world.get("mode", "biome") == "world"


func set_model(p_biome: Dictionary, p_frame: bool, p_compile: bool = true) -> void:
	model = p_biome
	if not model.has("graph"):
		model.graph = {"nodes": []}
	# A graph without positions (world.json's, the template's) laid out once its nodes' sizes are known.
	var fresh := not Graph.laid_out(model.graph)
	Graph.layout(model.graph)
	name_edit.text = String(model.get("name", "biome"))
	rebuild_ui()
	if fresh:
		var sizes := {}
		for c in graph.get_children():
			if c is GraphNode:
				sizes[String(c.name)] = (c as GraphNode).get_combined_minimum_size()
		Graph.layout(model.graph, sizes, true)
		for n in Graph.nodes(model.graph):
			(graph.get_node(NodePath(String(n.id))) as GraphNode).position_offset = Vector2(n.pos[0], n.pos[1])
	if p_compile:
		recompile()
	if p_frame:
		_fit_graph()


# The graph's top-left corner in view, zoomed so its columns fit the width. A frame later: the GraphEdit sizes its
# scroll range from its nodes then, and clamps any offset set before.
func _fit_graph() -> void:
	await get_tree().process_frame
	var ns := Graph.nodes(model.graph)
	if ns.is_empty():
		return
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for n in ns:
		var p := Vector2(n.pos[0], n.pos[1])
		lo = lo.min(p)
		hi = hi.max(p)
	graph.zoom = clampf(maxf(graph.size.x, 600.0) / (hi.x - lo.x + 400.0), 0.35, 0.8)
	graph.scroll_offset = lo * graph.zoom - Vector2(30, 30)


func model_node(p_id: String) -> Dictionary:
	return Graph.node(model.graph, p_id)


# A node's keys changed (null: taken out; p_rebuild: its rows changed too, rebuild its GraphNode). A slider sends
# several a frame: they compile once, at the frame's end.
func set_params(p_id: String, p_params: Dictionary, p_rebuild: bool = false) -> void:
	var node := model_node(p_id)
	for key in p_params:
		if p_params[key] == null:
			node.erase(key)
		else:
			node[key] = p_params[key]
	if p_rebuild:
		rebuild_ui()
	elif graph.has_node(NodePath(p_id)):
		Nodes.refresh(graph.get_node(NodePath(p_id)), node, self)
	queue_recompile()


func set_surface(p_key: String, p_hex: String) -> void:
	if not model.has("surface"):
		model.surface = {}
	model.surface[p_key] = p_hex
	queue_recompile()


func row_flag(p_mix: String, p_row: int, p_flag: String) -> bool:
	return row_flags.get("%s/%s/%d" % [model.name, p_mix, p_row], {}).get(p_flag, false)


func set_row_flag(p_mix: String, p_row: int, p_flag: String, p_on: bool) -> void:
	var key := "%s/%s/%d" % [model.name, p_mix, p_row]
	var f: Dictionary = row_flags.get(key, {})
	f[p_flag] = p_on
	row_flags[key] = f
	queue_recompile()


# Several edits in a frame (a slider, a chart drag's motion) compile once, at the frame's end. None without the window
# (the headless check edits through the mixer, then frees its parts).
func queue_recompile() -> void:
	if not compile_queued and preview.is_inside_tree():
		compile_queued = true
		recompile.call_deferred()


func recompile() -> void:
	compile_queued = false
	var lines := PackedStringArray()
	var map := preview.show_world(JSON.stringify(_heard(World.preview_file(world, -1 if world_mode() else int(world.current)))), world_mode())
	if String(map.get("error", "")) != "":
		lines.append("Error: the engine refuses it: " + String(map.error))
	if world_mode():
		resample(refit)
		refit = false
	var span := curve_span()
	for id in curve_ui:
		curve_ui[id].set_span(span)
	var ok := lines.is_empty()
	if world_mode():
		lines.append("%d biomes over %d fields. The chart: drag a biome's box to move it, its edge to resize; double-click a biome in the list to edit its graph." % [world.biomes.size(), world.fields.size()])
	else:
		lines.append("%d nodes. Right-click the graph: add a node; x or Delete removes one; drag from an output to an input to link, from an input away to unlink." % Graph.nodes(model.graph).size())
	status.text = "  ".join(lines)
	status.add_theme_color_override("font_color", Color(1.0, 0.5, 0.45) if not ok else Color(0.8, 0.82, 0.86))
	autosave.start()


# p_file with its mix rows as heard (row_flags: a muted row, or one not soloed where its mix has a solo, at volume 0).
func _heard(p_file: Dictionary) -> Dictionary:
	for b in p_file.biomes:
		for n in b.get("graph", {}).get("nodes", []):
			if n.get("kind") != "mix":
				continue
			var key := "%s/%s/" % [b.name, n.id]
			var rows: Array = n.get("rows", [])
			var solo := range(rows.size()).any(func(i): return row_flags.get(key + str(i), {}).get("solo", false))
			for i in rows.size():
				var f: Dictionary = row_flags.get(key + str(i), {})
				if f.get("mute", false) or (solo and not f.get("solo", false)):
					rows[i].x = 0.0
	return p_file


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
	for n in Graph.nodes(model.graph):
		_add_graph_node(n)
	for l in Graph.links(model.graph):
		graph.connect_node(l.from, 0, l.to, int(l.to_port))


func _add_graph_node(p_node: Dictionary) -> void:
	var gn := Nodes.build(p_node, self)
	gn.position_offset_changed.connect(func() -> void:
		p_node.pos = [gn.position_offset.x, gn.position_offset.y]
		autosave.start())
	graph.add_child(gn)


func add_node(p_kind: String, p_pos: Vector2) -> String:
	# A new Height curve starts as no remap over the ground's span.
	var id := Graph.add(model.graph, p_kind, p_pos, world.fields, {"points": Kinds.identity_curve(curve_span())} if p_kind == "curve" else {})
	rebuild_ui()
	recompile()
	return id


func _on_connection_request(p_from: StringName, _p_from_port: int, p_to: StringName, p_to_port: int) -> void:
	if Graph.reads(model.graph, String(p_from), String(p_to)):
		status.text = "That link would make a loop."
		return
	Graph.link(model.graph, String(p_from), String(p_to), p_to_port)
	rebuild_ui()
	recompile()


func _on_disconnection_request(_p_from: StringName, _p_from_port: int, p_to: StringName, p_to_port: int) -> void:
	Graph.unlink(model.graph, String(p_to), p_to_port)
	_clear_flags()
	rebuild_ui()
	recompile()


func _on_delete_nodes_request(p_names: Array) -> void:
	delete_nodes(p_names.map(func(n): return String(n)))


func delete_nodes(p_ids: Array) -> void:
	Graph.delete(model.graph, p_ids)
	_clear_flags()
	rebuild_ui()
	recompile()


func remove_mix_row(p_mix: String, p_row: int) -> void:
	Graph.unlink(model.graph, p_mix, p_row)
	_clear_flags()
	rebuild_ui()
	recompile()


# The current biome's mute and solo (its rows moved).
func _clear_flags() -> void:
	for key in row_flags.keys():
		if key.begins_with(String(model.name) + "/"):
			row_flags.erase(key)


func _apply_set(p_spec: String) -> void:
	var eq := p_spec.find("=")
	var dot := p_spec.find(".")
	var id := p_spec.substr(0, dot)
	if eq < 0 or dot < 0 or dot > eq or not (not model_node(id).is_empty() or id == "view"):
		push_error("--set=%s: expected ID.KEY=VALUE with a node of the graph (or view.KEY=VALUE)" % p_spec)
		return
	var raw := p_spec.substr(eq + 1)
	var json := JSON.new()
	var v = json.data if json.parse(raw) == OK else raw
	var key := p_spec.substr(dot + 1, eq - dot - 1)
	if id == "view":
		world.view[key] = v
		_enter(String(world.mode), true)
	else:
		set_params(id, {key: v}, true)


# --- files -------------------------------------------------------------------------------------------------------

func _autosave() -> void:
	World.store(AUTOSAVE, JSON.stringify(world, "\t", false, true))


# The biome's graph as the engine's template, or an output alone.
func _new_graph(p_template: bool) -> void:
	model.graph = EngineInfo.template_biome(String(model.name)).graph if p_template else {"nodes": [{"id": "out", "kind": "output"}]}
	_clear_flags()
	set_model(model, true)


func _save_world() -> void:
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(WORLDS_DIR))
	var path := "%s/%s.world.json" % [WORLDS_DIR, String(world.get("name", "world")).validate_filename()]
	status.text = ("Saved " + path) if World.store(path, JSON.stringify(world, "\t", false, true)) else ("Could not write " + path)


func _export_world() -> void:
	status.text = World.write(world, WORLD).message


# The chart's cloud: the world's fields on its two axes at the view's points (the engine's samples).
func resample(p_fit: bool) -> void:
	if not world_mode():
		return
	var c := world_panel.chart
	c.set_samples(preview.samples(c.fx, c.fy), p_fit)


# The game on this world, as heard: a world file next to the user's data, run by the same Godot.
func _open_game() -> void:
	if not World.store(GAME_WORLD, World.text(_heard(World.preview_file(world)))):
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
	mode_pick.tooltip_text = "Biome: the current biome's graph and it alone in the view. World: every biome's place on the climate chart, the whole world in the view."
	mode_pick.item_selected.connect(func(i: int) -> void: _enter("biome" if i == 0 else "world", true))
	bar.add_child(mode_pick)
	biome_pick.tooltip_text = "The biome the graph edits"
	biome_pick.item_selected.connect(func(i: int) -> void:
		world.current = i
		world_panel.select(i)
		set_model(world.biomes[i], true))
	bar.add_child(biome_pick)
	name_edit.custom_minimum_size.x = 160
	name_edit.tooltip_text = "The biome's name"
	name_edit.text_submitted.connect(func(t: String) -> void:
		_clear_flags()
		model.name = t
		_fill_biome_pick()
		world_panel.set_world(world)
		queue_recompile())
	bar.add_child(name_edit)
	bar.add_child(_menu("New graph", ["The engine's template ground", "An output alone"], func(i: int) -> void: _new_graph(i == 0)))
	var worlds := _menu("World", [], func(_i: int) -> void: pass)
	worlds.about_to_popup.connect(_fill_worlds.bind(worlds))
	bar.add_child(worlds)
	var kinds: Array = EngineInfo.kinds.map(func(k): return String(k.kind))
	bar.add_child(_menu("Add node", kinds.map(func(k): return Kinds.title(k)), func(i: int) -> void:
		add_node(kinds[i], (graph.scroll_offset + graph.size * 0.4) / graph.zoom)))

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
	for k in kinds:
		add_menu.add_item(Kinds.title(k))
	add_menu.id_pressed.connect(func(i: int) -> void: add_node(kinds[i], add_at))
	add_child(add_menu)
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right.custom_minimum_size.x = 420
	split.add_child(right)
	preview.view_changed.connect(func() -> void:
		if world_mode():
			world.view.extent_m = preview.extent_km * 1000.0
		else:
			world.view.extent_km = preview.extent_km
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


# The World menu: world.json again, or a saved world; save this one, export it.
func _fill_worlds(p_menu: MenuButton) -> void:
	var popup := p_menu.get_popup()
	popup.clear()
	for c in popup.id_pressed.get_connections():
		popup.id_pressed.disconnect(c.callable)
	var actions := [
		["Reload %s" % WORLD, func() -> void: set_world(World.read(WORLD), true)],
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
			var w := World.read("%s/%s" % [WORLDS_DIR, files[i - actions.size()]])
			if not w.is_empty():
				set_world(w, true))


func _shot(p_path: String) -> void:
	var t := Time.get_ticks_usec()
	var map := preview.show_world(JSON.stringify(_heard(World.preview_file(world, -1 if world_mode() else int(world.current)))), world_mode(), true)
	var ms := snappedf((Time.get_ticks_usec() - t) / 1000.0, 0.1)
	# The 3D view: the first load (its shaders compiled), then two restarts on the same world (an edit's path).
	var loads: Array = await preview.wait_terrain(120.0, 2) if preview.is_3d() else []
	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw
	get_viewport().get_texture().get_image().save_png(p_path)
	preview.image().save_png(p_path.get_basename() + "_view.png")
	print("MIXER_SHOT: ", JSON.stringify({"path": p_path, "map_ms": ms, "us_per_point": map.get("us_per_point", 0.0), "terrain_loads_ms": loads, "error": map.get("error", "")}))
	get_tree().quit(0)


# --- the headless check ------------------------------------------------------------------------------------------

func _check() -> void:
	var report := Checks.run(self)
	print("MIXER: ", JSON.stringify(report))
	# The window's parts, never added to the tree in a check.
	for c in [graph, preview, status, name_edit, add_menu, autosave, world_panel, biome_pick, mode_pick]:
		c.free()
	await get_tree().process_frame
	get_tree().quit(0 if report.ok else 1)
