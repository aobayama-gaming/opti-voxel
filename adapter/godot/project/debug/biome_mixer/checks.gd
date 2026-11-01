extends RefCounted
## The biome mixer's headless self-test (mixer.gd --check): the noise pins, every template and world.json biome
## compiled and round-tripped through a file, the fixed notes, the cuts, old files, the world shader. run() returns the
## report (its "ok" the verdict).

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const Compile := preload("res://debug/biome_mixer/compile.gd")
const ImportTable := preload("res://debug/biome_mixer/import_table.gd")
const World := preload("res://debug/biome_mixer/world.gd")


static func run(p_table: Dictionary) -> Dictionary:
	var report := {"pins": Octaves.check_pins(), "graphs": {}}
	var ok: bool = report.pins.is_empty()
	var graphs := {"plain ground": ImportTable.new("plain ground").plain_ground(), "every kind": every_kind()}
	for name in ImportTable.biome_names(p_table):
		graphs["world.json " + name] = ImportTable.new(name).import_biome(p_table).model
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
		# The ground's notes, fixed: one per k, K_MIN .. ground_top.
		entry["notes_fixed"] = note_ks(graphs[g]) == range(Octaves.K_MIN, Octaves.ground_top(25, 0.05) + 1)
		ok = ok and r.ok and entry.same_after_file and entry.notes_fixed
		report.graphs[g] = entry
	report["cuts"] = cuts(p_table)
	report["old_kinds"] = old_kinds()
	# SDFCascade::micro_top_k's pins (test_sdf_cascade.cpp): the notes stop where the engine's ground does.
	report["ground_top"] = [Octaves.ground_top(25, 0.05), Octaves.ground_top(25, 0.5)]
	ok = ok and report.cuts.ok and report.old_kinds.ok and report.ground_top == [19, 16]
	report["world"] = world(p_table)
	ok = ok and report.world.ok
	report["ok"] = ok
	return report


# A file saved with kinds since removed (a Terrace before the output, a Climate warp node linked into a note) loads as
# the graph without them, the note's warp on: the same nodes, the same code. A file missing a note gets it back.
static func old_kinds() -> Dictionary:
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
	var same: bool = Compile.new().run(m).code == Compile.new().run(want).code
	var gone = JSON.parse_string(JSON.stringify(want))
	gone.nodes.erase(note)
	gone.links = gone.links.filter(func(l): return l.from != note)
	Kinds.fix_notes(gone)
	var back: bool = note_ks(gone) == note_ks(want)
	return {"ok": same and back and m.nodes.size() == want.nodes.size(), "nodes": [want.nodes.size(), m.nodes.size()], "same_code": same, "note_back": back}


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
		var r := Compile.new().run(m)
		var entry := {"rows": [rows, Kinds.rows(m.nodes[mix]).size()], "nodes": [nodes, m.nodes.size()], "compiles": r.ok}
		var want_rows: int = rows - 1 if c == "row" else rows
		entry.ok = r.ok and entry.rows[1] == want_rows and (c == "row" or entry.nodes[1] == nodes) and ports == range(ports.size())
		entry.ok = entry.ok and note_ks(m) == range(Octaves.K_MIN, Octaves.ground_top(25, 0.05) + 1)
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


# A world (world.json's biomes in one shader): it compiles, its notes are shared (fewer rows than the biomes' notes
# summed), its places are the uniforms u_atc / u_ati, it comes back the same from a file; and the weights' port
# (World.weights) has the engine's properties on a two-biome world: whole at a centre, the default far from both, a
# blend between them summing to 1.
static func world(p_table: Dictionary) -> Dictionary:
	var w := World.from_table(p_table)
	var r := Compile.new().run_world(w)
	var notes := 0
	for b in w.biomes:
		notes += note_ks(b.model).size()
	var again = JSON.parse_string(JSON.stringify(w, "", true, true))
	var r2 := Compile.new().run_world(again)
	var names := World.field_names(w)
	var b0: Dictionary = w.biomes[0]
	var f0: String = b0.at.keys()[0]
	var i0 := names.find(f0)
	var places_ok: bool = absf(r.uniforms.u_atc[i0] - b0.at[f0][0]) < 1e-6 and absf(r.uniforms.u_ati[i0] - 1.0 / b0.at[f0][1]) < 1e-4
	var needles := []
	for needle in ["kfac(FL[", "wk1_%d" % (w.biomes.size() - 1), "if (wk1_0 > 0.0)", "c_map = "]:
		if not r.code.contains(needle):
			needles.append(needle)
	var two := {"fields": [{"name": "t"}], "biomes": [{"name": "hot", "at": {"t": [0.5, 0.75]}}, {"name": "cold", "at": {"t": [-0.5, 0.75]}}, {"name": "rest", "at": {}}]}
	var w0 := World.weights(two, [0.5])
	var wfar := World.weights(two, [1.5])
	var wmid := World.weights(two, [0.2])
	var props: bool = absf(w0[0] - 1.0) < 1e-9 and absf(wfar[2] - 1.0) < 1e-9 and wmid[0] > wmid[1] and wmid[1] > 0.0 and absf(wmid[0] + wmid[1] + wmid[2] - 1.0) < 1e-9
	var entry := {
		"compiles": r.ok, "errors": r.errors, "biomes": w.biomes.size(), "fields": names.size(),
		"note_rows": [r.budget.notes, notes], "field_rows": r.uniforms.u_n_warp, "places": places_ok, "missing": needles,
		"same_after_file": r2.code == r.code, "weights": props, "uniform_kb": r.uniform_bytes / 1024,
	}
	entry.ok = r.ok and r.drawable and r.budget.notes < notes and places_ok and needles.is_empty() and entry.same_after_file and props
	return entry
