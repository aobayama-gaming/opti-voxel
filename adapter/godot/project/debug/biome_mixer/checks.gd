extends RefCounted
## The biome mixer's headless self-test (mixer.gd --check): every template and world.json biome exported and drawn by
## the engine (ClimateReportNode.preview) and round-tripped through a file, the fixed notes, the cuts, old files, the
## whole world drawn with its chart's samples, the export back to a table (instances and features kept). run() returns
## the report (its "ok" the verdict).

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const ImportTable := preload("res://debug/biome_mixer/import_table.gd")
const World := preload("res://debug/biome_mixer/world.gd")
const ExportTable := preload("res://debug/biome_mixer/export_table.gd")
const EngineView := preload("res://debug/biome_mixer/engine_view.gd")

# Where the check writes world.json's export for the engine to parse.
const EXPORT_PATH := "user://biome_mixer_export_check.json"
# The engine's maps the check asks for (points a side): small, the check is about the bits reaching the engine.
const SIZE := 16


static func run(p_table: Dictionary) -> Dictionary:
	var report := {"graphs": {}}
	var ok := true
	var graphs := {"plain ground": ImportTable.new("plain ground").plain_ground(), "every kind": every_kind()}
	for name in ImportTable.biome_names(p_table):
		graphs["world.json " + name] = ImportTable.new(name).import_biome(p_table).model
	for g in graphs:
		var r := drawn(graphs[g], p_table)
		var again = JSON.parse_string(JSON.stringify(graphs[g], "", true, true))
		var entry := {"exported": r.errors.is_empty(), "errors": r.errors, "nodes": graphs[g].nodes.size(), "engine": r.error,
			"same_after_file": graph_text(again, p_table) == graph_text(graphs[g], p_table)}
		# The ground's notes, fixed: one per k, K_MIN .. ground_top.
		entry["notes_fixed"] = note_ks(graphs[g]) == range(Octaves.K_MIN, Octaves.ground_top() + 1)
		if g == "every kind":
			# What only the preview had (Carve, Note warp, Add / Multiply / Constant) the table can't hold: refused.
			entry.ok = not entry.exported and entry.notes_fixed
		else:
			entry.ok = entry.exported and r.error == "" and entry.same_after_file and entry.notes_fixed
		ok = ok and entry.ok
		report.graphs[g] = entry
	report["cuts"] = cuts(p_table)
	report["old_kinds"] = old_kinds(p_table)
	# The template ground's notes (the engine's at world.json's engine, 5 cm and 25 LODs): 0.4 m .. 26 km.
	report["notes"] = [Octaves.K_MIN, Octaves.ground_top()]
	ok = ok and report.cuts.ok and report.old_kinds.ok and report.notes == [3, 19]
	report["world"] = world(p_table)
	ok = ok and report.world.ok
	report["export"] = export_world(p_table)
	ok = ok and report.export.ok
	report["ok"] = ok
	return report


# p_model as a world of its own (the biome alone, world.json's fields), drawn by the engine: {errors (the export's),
# error (the engine's), map}.
static func drawn(p_model: Dictionary, p_table: Dictionary) -> Dictionary:
	var w := World.from_table(p_table)
	w.biomes = [{"name": String(p_model.get("name", "biome")), "at": {}, "model": p_model}]
	var r := ExportTable.preview_world(w, p_table)
	if not r.errors.is_empty():
		return {"errors": r.errors, "error": "", "map": {}}
	var map := EngineView.draw(JSON.stringify(r.world), 20.0, SIZE)
	return {"errors": r.errors, "error": map.error, "map": map}


# p_model's graph as the table holds it (ExportTable): what the engine reads of it.
static func graph_text(p_model: Dictionary, p_table: Dictionary) -> String:
	return JSON.stringify(ExportTable._biome(p_model, World.from_table(p_table), []).graph)


# A file saved with kinds since removed (a Terrace before the output, a Climate warp node linked into a note) loads as
# the graph without them, the note's warp on: the same nodes, the same table graph. A file missing a note gets it back.
static func old_kinds(p_table: Dictionary) -> Dictionary:
	var want := ImportTable.new("plain ground").plain_ground()
	var note := Kinds.first_of(want, "note")
	var m = JSON.parse_string(JSON.stringify(want))
	want.nodes[note].params.warp = true
	var into_out: Dictionary = m.links.filter(func(l): return l.to == Kinds.first_of(m, "output"))[0]
	m.nodes["old"] = {"kind": "terrace", "pos": [0, 0], "params": {"step_m": 40.0}}
	m.links.append({"from": into_out.from, "from_port": into_out.from_port, "to": "old", "to_port": 0})
	into_out.from = "old"
	into_out.from_port = 0
	m.nodes["old_warp"] = {"kind": "warp", "pos": [0, 0], "params": {}}
	m.links.append({"from": "old_warp", "from_port": 0, "to": note, "to_port": 0})
	Kinds.fix_notes(m)
	Kinds.drop_unknown(m)
	var same: bool = graph_text(m, p_table) == graph_text(want, p_table)
	var gone = JSON.parse_string(JSON.stringify(want))
	gone.nodes.erase(note)
	gone.links = gone.links.filter(func(l): return l.from != note)
	Kinds.fix_notes(gone)
	var back: bool = note_ks(gone) == note_ks(want)
	return {"ok": same and back and m.nodes.size() == want.nodes.size(), "nodes": [want.nodes.size(), m.nodes.size()], "same_graph": same, "note_back": back}


# A model's notes' k, sorted.
static func note_ks(p_model: Dictionary) -> Array:
	var ks: Array = p_model.nodes.values().filter(func(n): return n.kind == "note").map(func(n): return int(Kinds.param(n, "k")))
	ks.sort()
	return ks


# Removing a Mix row (the mountain's first: one row less, the other rows' links one port up, no gaps, its note stays)
# and deleting a note (plain ground's first: nothing changes, the notes are fixed).
static func cuts(p_table: Dictionary) -> Dictionary:
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
		var r := drawn(m, p_table)
		var drew: bool = r.errors.is_empty() and r.error == ""
		var entry := {"rows": [rows, Kinds.rows(m.nodes[mix]).size()], "nodes": [nodes, m.nodes.size()], "drawn": drew}
		var want_rows: int = rows - 1 if c == "row" else rows
		entry.ok = drew and entry.rows[1] == want_rows and (c == "row" or entry.nodes[1] == nodes) and ports == range(ports.size())
		entry.ok = entry.ok and note_ks(m) == range(Octaves.K_MIN, Octaves.ground_top() + 1)
		ok = ok and entry.ok
		out[c] = entry
	out.ok = ok
	return out


# A graph with every kind of node, wired: two warped notes (the climate's warp, a Note warp), shaped, chained, mixed,
# combined, curved, carved; the other notes silent rows.
static func every_kind() -> Dictionary:
	var b := ImportTable.new("every kind")
	var a := b.add("note", Vector2.ZERO, {"k": 15, "warp": true, "warp_m": 200.0})
	var c := b.add("note", Vector2.ZERO, {"k": 14})
	var d := b.add("note", Vector2.ZERO, {"k": 12, "warp": true, "warp_m": 30.0})
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
	Kinds.fix_notes(b.model)
	return b.model


# The whole world (world.json's biomes) drawn by the engine: every biome named, the chart's samples one per point (two
# fields each), the same file after a round trip.
static func world(p_table: Dictionary) -> Dictionary:
	var w := World.from_table(p_table)
	var r := ExportTable.preview_world(w, p_table)
	var again = JSON.parse_string(JSON.stringify(w, "", true, true))
	var r2 := ExportTable.preview_world(again, p_table)
	var map: Dictionary = EngineView.draw(JSON.stringify(r.world), 200.0, SIZE) if r.errors.is_empty() else {}
	var names: PackedStringArray = map.get("biome_names", PackedStringArray())
	var fields: PackedStringArray = map.get("field_names", PackedStringArray())
	var samples: int = map.get("fields", PackedFloat32Array()).size() / maxi(fields.size(), 1)
	var entry := {
		"exported": r.errors.is_empty(), "errors": r.errors, "engine": map.get("error", "not drawn"), "biomes": names.size(), "fields": fields.size(),
		"samples": samples, "us_per_point": map.get("us_per_point", 0.0), "same_after_file": JSON.stringify(r2.world) == JSON.stringify(r.world),
	}
	entry.ok = entry.exported and String(entry.engine) == "" and names.size() == w.biomes.size() and fields.size() == w.fields.size() and samples == SIZE * SIZE and entry.same_after_file
	return entry


# world.json as a world exported back (ExportTable): every biome plays the same notes as the table's (each octave's
# volume, shape, multifractal and warp as the engine reads them), the same places, curve, surface and clouds; the world
# exported again is the same file. The mixer's export path writes it, the engine accepting it (ClimateReportNode.check,
# when the extension is loaded), and refuses a graph the table can't hold (every kind's) and a span of 0 (nothing
# written).
static func export_world(p_table: Dictionary) -> Dictionary:
	var w := World.from_table(p_table)
	var wr := ExportTable.write(w, p_table, EXPORT_PATH)
	var entry := {"written": wr.written, "message": wr.message, "warnings": wr.warnings}
	var text := FileAccess.get_file_as_string(EXPORT_PATH) if wr.written else ""
	var back := World.read_table(EXPORT_PATH) if wr.written else {}
	entry.lines = text.count("\n")
	if back.is_empty():
		entry.ok = false
		return entry
	var differ := []
	var want := notes_of(p_table, w)
	var got := notes_of(back, w)
	for b in want:
		if not got.has(b) or not _close(want[b], got[b]):
			differ.append(b)
	for i in w.biomes.size():
		var a: Dictionary = p_table.biomes[i]
		var e: Dictionary = back.biomes[i]
		for key in ["at", "surface", "clouds", "debug_color", "instances", "features"]:
			if not _close(a.get(key), e.get(key)):
				differ.append("%s.%s" % [a.name, key])
		if not _close(ImportTable.graph_rows(a).curve, ImportTable.graph_rows(e).curve):
			differ.append("%s.curve" % a.name)
	for key in ["comment", "fields"]:
		if not _close(p_table.get(key), back.get(key)):
			differ.append(key)
	entry.differ = differ
	# The export is the table's notes in their plainest form (a plain shape is no Shape): exporting its world again
	# gives the same file (so the same terrain).
	entry.stable = ExportTable.text(ExportTable.split(ExportTable.table(World.from_table(back), back).table, back).world) == text
	entry.refused = not ExportTable._biome(every_kind(), w, []).errors.is_empty()
	var placed: Array = w.biomes.filter(func(b): return not b.at.is_empty())
	if ClassDB.class_exists("ClimateReportNode") and not placed.is_empty():
		var zero_path := EXPORT_PATH + ".zero"
		DirAccess.remove_absolute(ProjectSettings.globalize_path(zero_path))
		var at: Dictionary = placed[0].at
		at[at.keys()[0]][1] = 0.0
		var wz := ExportTable.write(w, p_table, zero_path)
		entry.engine_refusal = wz.message
		entry.refused = entry.refused and not wz.written and not FileAccess.file_exists(zero_path)
	entry.ok = wr.written and differ.is_empty() and entry.stable and entry.refused
	return entry


# Each biome's notes as the engine reads its graph (ImportTable.graph_rows): {biome: {k: [x, coeffs, multifractal,
# warp_m, warp_fields, stretch_y]}} over the ground's octaves (a note no row mixes: volume 0). The constant coefficient
# only counts in a multifractal (elsewhere the band goes in without its mean).
static func notes_of(p_table: Dictionary, p_world: Dictionary) -> Dictionary:
	var out := {}
	var top := Octaves.ground_top()
	for b in p_table.get("biomes", []):
		var g := ImportTable.graph_rows(b)
		var per := {}
		for k in range(Octaves.K_MIN, top + 1):
			var a: Dictionary = g.per_k.get(k, {"x": 0.0})
			var mf := float(a.get("multifractal", 0.0))
			var c := Kinds.shape_coeffs({"kind": "shape", "params": ImportTable.shape_params(a)}).map(func(v): return float(v))
			if mf == 0.0:
				c[0] = 0.0
			var warp := float(a.get("warp_m", 0.0))
			per[k] = [float(a.get("x", 1.0)), c, mf, warp, a.get("warp_fields", []) if warp > 0.0 else [], g.stretch.get(k, 1)]
		out[b.name] = per
	return out


# Equal, numbers within 1e-6 (the file keeps ExportTable.DECIMALS decimals).
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
