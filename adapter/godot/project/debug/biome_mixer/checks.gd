extends RefCounted
## The biome mixer's headless self-test (mixer.gd --check): every engine kind has the mixer's UI; world.json read as the
## mixer's world is the same file; every biome, and a graph of every kind, drawn by the engine and every node's GraphNode
## built; edits (a row out, a node out, a loop the engine refuses, mute) still drawn; the export written, read back the same, and
## refused when the engine refuses; the blank planet world drawn over its planet, its fields editable, a ground under its
## centre refused. run() returns the report (its "ok" the verdict).

const EngineInfo := preload("res://debug/biome_mixer/engine_info.gd")
const Graph := preload("res://debug/biome_mixer/graph.gd")
const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Nodes := preload("res://debug/biome_mixer/nodes.gd")
const World := preload("res://debug/biome_mixer/world.gd")
const EngineView := preload("res://debug/biome_mixer/engine_view.gd")

# Where the check writes world.json's export.
const EXPORT_PATH := "user://biome_mixer_export_check.json"
# The blank planet world (tools/biome_mixer.sh --open=... starts on it).
const PLANET := "res://worlds/tiny_biomes.world.json"
# The engine's maps the check asks for (points a side): small, the check is about the bits reaching the engine.
const SIZE := 16


static func run(p_mixer: Object) -> Dictionary:
	var report := {}
	var w := World.read(EngineInfo.WORLD)
	var missing: Array = EngineInfo.kinds.map(func(k): return String(k.kind)).filter(func(k): return not Kinds.KINDS.has(k))
	report.kinds = {"engine": EngineInfo.kinds.size(), "without_ui": missing, "ok": EngineInfo.kinds.size() > 0 and missing.is_empty()}
	if w.is_empty():
		report.ok = false
		return report
	report.same_file = same_file(w)
	report.biomes = {}
	for i in w.biomes.size():
		report.biomes[w.biomes[i].name] = drawn(p_mixer, w, i)
	var every := every_kind(w)
	report.every_kind = drawn(p_mixer, every, every.biomes.size() - 1)
	report.world = whole(w)
	report.edits = edits(p_mixer)
	report.export = export_world(w)
	report.planet = planet()
	var ok: bool = report.kinds.ok and report.same_file.ok and report.every_kind.ok and report.world.ok and report.edits.ok and report.export.ok and report.planet.ok
	for b in report.biomes:
		ok = ok and report.biomes[b].ok
	report.ok = ok
	return report


# Biome p_b of p_world alone, drawn by the engine (as the mixer hears it), and every node's GraphNode built with the
# mixer as its context: {engine (its error), nodes, built, ok}.
static func drawn(p_mixer: Object, p_world: Dictionary, p_b: int) -> Dictionary:
	var map := EngineView.draw(JSON.stringify(p_mixer._heard(World.preview_file(p_world, p_b))), 20.0, SIZE)
	p_mixer.world = p_world
	p_mixer.model = p_world.biomes[p_b]
	var built := 0
	var ns := Graph.nodes(p_world.biomes[p_b].graph)
	for n in ns:
		var gn := Nodes.build(n, p_mixer)
		if gn != null and gn.get_child_count() > 0 and gn.title != "":
			built += 1
		if gn != null:
			gn.free()
	p_mixer.curve_ui.clear()
	return {"engine": String(map.error), "nodes": ns.size(), "built": built, "ok": String(map.error) == "" and built == ns.size() and built > 0}


# world.json read as the mixer's world and written back is the same file, but for the graphs' positions (the mixer's,
# the engine ignores them).
static func same_file(p_world: Dictionary) -> Dictionary:
	var want = JSON.parse_string(FileAccess.get_file_as_string(EngineInfo.WORLD))
	var got = JSON.parse_string(World.text(World.file_of(p_world)))
	for b in got.biomes:
		for n in b.get("graph", {}).get("nodes", []):
			n.erase("pos")
	return {"ok": _close(want, got)}


# The whole world drawn by the engine across p_km: every biome named, the chart's samples one per point (every field each).
static func whole(p_world: Dictionary, p_km: float = 200.0) -> Dictionary:
	var map := _map(p_world, p_km)
	var names: PackedStringArray = map.get("biome_names", PackedStringArray())
	var fields: PackedStringArray = map.get("field_names", PackedStringArray())
	var samples: int = map.get("fields", PackedFloat32Array()).size() / maxi(fields.size(), 1)
	return {"engine": String(map.error), "biomes": names.size(), "fields": fields.size(), "samples": samples, "us_per_point": map.get("us_per_point", 0.0),
		"ok": String(map.error) == "" and names.size() == p_world.biomes.size() and fields.size() == p_world.fields.size() and samples == SIZE * SIZE}


# p_world with one biome more, the last (the default): a graph of every kind of node the engine has, wired as it reads
# them (notes, one warped by a coarser one, one by a warp's x, y, z ports, shapes, a multifractal of a shaped parent, a
# mix, then every other op, the sea's level (a sea added to the world), a curve, the output).
static func every_kind(p_world: Dictionary) -> Dictionary:
	var w: Dictionary = p_world.duplicate(true)
	var g := {"nodes": []}
	var f: Array = w.fields
	var add := func(p_kind: String, p_params: Dictionary) -> String: return Graph.add(g, p_kind, Vector2.ZERO, f, p_params)
	var n0: String = add.call("note", {"wavelength_m": 1638.4, "seed": 5, "amp_m": 0.5})
	var wp: String = add.call("warp", {"in": n0, "amount_m": 30.0})
	var n1: String = add.call("note", {"wavelength_m": 409.6, "seed": 1, "amp_m": 20.0, "warp": wp})
	var r0: String = add.call("shape", {"in": n0, "shape": "ridged"})
	var xyz: String = add.call("warp", {"x": n0, "y": r0, "z": n1, "amount_m": 4.0})
	var n2: String = add.call("note", {"wavelength_m": 102.4, "seed": 2, "amp_m": 5.0, "stretch_y": 2, "warp": xyz})
	var s1: String = add.call("shape", {"in": n1, "shape": "ridged"})
	var mf: String = add.call("multifractal", {"in": s1, "parent": r0, "low_m": 0.0, "high_m": 0.5})
	var s2: String = add.call("shape", {"in": n2, "coeffs": [0.0, 0.0, 1.0, 0.0]})
	var mix: String = add.call("mix", {"rows": [{"in": mf, "x": 1.0}, {"in": s2, "x": 0.5}]})
	var c: String = add.call("constant", {"value_m": 10.0})
	var fd: String = add.call("field", {})
	# A cave's form: smin of the ground and up minus a depth.
	var cave: String = add.call("subtract", {"a": add.call("up", {}), "b": c})
	# An ocean floor's: the sea's level (the world then has a sea).
	var sea: String = add.call("sea", {})
	w.sea = {"level_m": 0.0}
	var last := mix
	for op in ["add", "subtract", "multiply", "min", "max", "smin", "smax"]:
		var b := fd if op == "multiply" else cave if op == "smin" else sea if op == "max" else c
		last = add.call(op, {"a": last, "b": b, "k_m": 5.0} if op.begins_with("s") and op != "subtract" else {"a": last, "b": b})
	var cv: String = add.call("curve", {"in": last, "points": Kinds.identity_curve(500.0)})
	add.call("output", {"height": cv})
	Graph.layout(g)
	var used := {}
	for n in g.nodes:
		used[n.kind] = true
	if used.size() != EngineInfo.kinds.size():
		push_error("checks.gd every_kind: %d kinds of the engine's %d" % [used.size(), EngineInfo.kinds.size()])
	w.biomes.append({"name": "every kind", "at": {}, "graph": g})
	return w


# Edits on world.json's biomes, each still drawn by the engine: the mountain's first mix row out (one row less, its note
# stays), a shape node out (what read it reads its note), a loop linked and refused by the engine then taken out, a
# muted row heard at volume 0, a weather key set.
static func edits(p_mixer: Object) -> Dictionary:
	var w := World.read(EngineInfo.WORLD)
	var i: int = w.biomes.map(func(b): return b.name).find("warped")
	var g: Dictionary = w.biomes[i].graph
	var mix := ""
	var shape := ""
	for n in g.nodes:
		if n.kind == "mix":
			mix = n.id
		elif n.kind == "shape" and shape == "":
			shape = n.id
	var r := {}
	var rows := Graph.rows(Graph.node(g, mix)).size()
	var kept := Graph.nodes(g).size()
	Graph.unlink(g, mix, 0)
	r.row = Graph.rows(Graph.node(g, mix)).size() == rows - 1 and Graph.nodes(g).size() == kept
	var under := String(Graph.node(g, shape).get("in", ""))
	Graph.delete(g, [shape])
	r.node = Graph.node(g, shape).is_empty() and not Graph.links(g).any(func(l): return l.from == shape) and Graph.links(g).any(func(l): return l.from == under)
	var sum := Graph.add(g, "add", Vector2.ZERO, w.fields, {"a": mix, "b": under})
	Graph.link(g, sum, mix, Graph.rows(Graph.node(g, mix)).size())
	var looped := drawn(p_mixer, w, i)
	r.loop_refused = not looped.ok and String(looped.engine).contains("loops")
	r.loop_message = looped.engine
	Graph.delete(g, [sum])
	r.drawn = drawn(p_mixer, w, i).ok
	p_mixer.model = w.biomes[i]
	p_mixer.set_row_flag(mix, 0, "mute", true)
	var heard: Dictionary = p_mixer._heard(World.preview_file(w, i))
	var hn: Dictionary = Graph.node(heard.biomes[0].graph, mix)
	r.muted = float(hn.rows[0].get("x", 1.0)) == 0.0 and float(Graph.rows(Graph.node(g, mix))[0].get("x", 1.0)) != 0.0
	p_mixer.row_flags.clear()
	# A weather key set through the mixer's row: in the biome's "weather" block, the file the engine takes.
	p_mixer.set_weather("wetness", 0.25)
	var sent: Dictionary = World.preview_file(w, i)
	r.weather = float(sent.biomes[0].get("weather", {}).get("wetness", -1.0)) == 0.25 and World.check(w) == ""
	r.ok = r.row and r.node and r.loop_refused and r.drawn and r.muted and r.weather
	return r


# The world exported (World.write: the engine checks it first) and read back: the same biomes; written again, the same
# text. A span of 0 refused, nothing written.
static func export_world(p_world: Dictionary) -> Dictionary:
	DirAccess.remove_absolute(ProjectSettings.globalize_path(EXPORT_PATH))
	var wr := World.write(p_world, EXPORT_PATH)
	var r := {"written": wr.written, "message": wr.message}
	var text := FileAccess.get_file_as_string(EXPORT_PATH) if wr.written else ""
	var back := World.read(EXPORT_PATH) if wr.written else {}
	r.same = not back.is_empty() and _close(World.file_of(back).biomes, World.file_of(p_world).biomes)
	r.stable = not back.is_empty() and World.text(World.file_of(back)) == text
	var zero: Dictionary = p_world.duplicate(true)
	var at: Dictionary = zero.biomes[0].at
	at[at.keys()[0]][1] = 0.0
	var zero_path := EXPORT_PATH + ".zero"
	DirAccess.remove_absolute(ProjectSettings.globalize_path(zero_path))
	var wz := World.write(zero, zero_path)
	r.refusal = wz.message
	r.refused = not wz.written and not FileAccess.file_exists(zero_path)
	r.ok = wr.written and r.same and r.stable and r.refused
	return r


# The blank planet world: drawn whole by the engine across the planet; a field's band edited (the fields panel's path,
# World.preview_file) changes the field the engine samples; a ground 2.5 km down refused by the export check (it would
# cross the 2 km planet's centre), taken by it on the flat world.
static func planet() -> Dictionary:
	var w := World.read(PLANET)
	if w.is_empty():
		return {"ok": false, "error": PLANET + " is not a world file with biomes"}
	var km := World.planet_extent_m(w) / 1000.0
	var r := {"world": whole(w, km)}
	var before: PackedFloat32Array = _map(w, km).get("fields", PackedFloat32Array())
	w.fields[0].bands[0].wavelength_m = float(w.fields[0].bands[0].wavelength_m) * 2.0
	r.field_edit = not before.is_empty() and _map(w, km).get("fields", PackedFloat32Array()) != before
	var deep: Dictionary = w.duplicate(true)
	deep.biomes[0].graph = {"nodes": [{"id": "c", "kind": "constant", "value_m": -2500.0}, {"id": "out", "kind": "output", "height": "c"}]}
	r.refusal = World.check(deep)
	var flat: Dictionary = deep.duplicate(true)
	flat.shape = {"kind": "flat"}
	r.flat = World.check(flat)
	r.ok = r.world.ok and r.field_edit and r.refusal.contains("past the planet's centre") and r.flat == ""
	return r


# p_world drawn whole by the engine across p_km (its maps, as the mixer asks for them).
static func _map(p_world: Dictionary, p_km: float) -> Dictionary:
	return EngineView.draw(JSON.stringify(World.preview_file(p_world)), p_km, SIZE)


# Equal, numbers within 1e-6 (the file keeps World.DIGITS digits).
static func _close(p_a: Variant, p_b: Variant) -> bool:
	if (p_a is float or p_a is int) and (p_b is float or p_b is int):
		return absf(float(p_a) - float(p_b)) <= 1e-6 * maxf(1.0, absf(float(p_a)))
	if p_a is Array and p_b is Array:
		if p_a.size() != p_b.size():
			return false
		for i in p_a.size():
			if not _close(p_a[i], p_b[i]):
				return false
		return true
	if p_a is Dictionary and p_b is Dictionary:
		if p_a.size() != p_b.size():
			return false
		for key in p_a:
			if not p_b.has(key) or not _close(p_a[key], p_b[key]):
				return false
		return true
	return p_a == p_b
