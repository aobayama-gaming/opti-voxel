extends RefCounted
## Biome graphs to start from: the plain ground, or one biome of a climate table (read only). Each of the biome's
## ground octaves becomes a Note (warped by a Warp field built from the table's warp fields), its shape a Shape, its
## multifractal a Multifractal chained to the next finer one, its multiplier a Mix row's volume; its curve a Height
## curve (the same points), its surface the Output's colours.

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")

const COL := {"warp": -360.0, "note": 0.0, "shape": 300.0, "multifractal": 580.0, "mix": 880.0}
# How wide a Mix is (its rows carry their swing, share and x): the column after it starts past it.
const MIX_W := 640.0
const ROW_H := 240.0

var model := {}
var warnings := PackedStringArray()
var voxel_m := Octaves.DEFAULT_VOXEL_M
var lods := Octaves.DEFAULT_LODS


func _init(p_name: String, p_voxel_m: float = Octaves.DEFAULT_VOXEL_M, p_lods: int = Octaves.DEFAULT_LODS) -> void:
	voxel_m = p_voxel_m
	lods = p_lods
	model = {"version": 1, "name": p_name, "voxel_m": voxel_m, "lods": lods, "origin_m": [0.0, 0.0], "next_id": 1, "nodes": {}, "links": []}


func add(p_kind: String, p_pos: Vector2, p_params: Dictionary = {}) -> String:
	return Kinds.add(model, p_kind, p_pos, p_params)


func link(p_from: String, p_from_port: int, p_to: String, p_to_port: int) -> void:
	link_in(model, p_from, p_from_port, p_to, p_to_port)


static func link_in(p_model: Dictionary, p_from: String, p_from_port: int, p_to: String, p_to_port: int) -> void:
	p_model.links.append({"from": p_from, "from_port": p_from_port, "to": p_to, "to_port": p_to_port})


func k_of(p_m: float) -> int:
	return int(round(log(p_m / voxel_m) / log(2.0)))


# Notes k_top down to k_bottom in p_model, each through a shape (none: plain) into a new row of mix p_mix, laid out from
# p_y down; returns their Note ids.
static func add_notes(p_model: Dictionary, p_mix: String, p_k_top: int, p_k_bottom: int, p_y: float, p_preset: String = "plain") -> Array:
	var ids := []
	var rows: Array = p_model.nodes[p_mix].params.rows
	for k in range(p_k_top, p_k_bottom - 1, -1):
		var y := p_y + (p_k_top - k) * ROW_H
		var note := Kinds.add(p_model, "note", Vector2(COL.note, y), {"k": k})
		var last := note
		if p_preset != "plain":
			last = Kinds.add(p_model, "shape", Vector2(COL.shape, y), {"preset": p_preset, "offset": Kinds.preset_offset(p_preset)})
			link_in(p_model, note, 0, last, 0)
		link_in(p_model, last, 0, p_mix, rows.size())
		rows.append(Kinds.new_row())
		ids.append(note)
	return ids


# The ground as it is: notes 6.4 m .. 6.5 km, plain, every one at volume 1 (ImportTable.new("plain ground").plain_ground()).
func plain_ground() -> Dictionary:
	var mix := add("mix", Vector2(COL.mix, 0.0))
	var out := add("output", Vector2(COL.mix + MIX_W, 0.0))
	link(mix, 0, out, 0)
	add_notes(model, mix, k_of(6553.6), k_of(6.4), 0.0)
	model.nodes[mix].pos = [COL.mix, model.nodes[mix].params.rows.size() * ROW_H * 0.5 - ROW_H]
	model.nodes[out].pos = [COL.mix + MIX_W, model.nodes[mix].pos[1]]
	return model


static func biome_names(p_table: Dictionary) -> Array:
	return p_table.get("biomes", []).map(func(b): return b.get("name", "?"))


# The biome named as this importer (ImportTable.new(name).import_biome(table)): {model, warnings}.
func import_biome(p_table: Dictionary) -> Dictionary:
	var p_name: String = model.name
	model.source = "climate table, biome %s" % p_name
	_import(p_table, p_name)
	return {"model": model, "warnings": warnings}


func _import(p_table: Dictionary, p_name: String) -> void:
	var biome := {}
	for d in p_table.get("biomes", []):
		if d.get("name", "") == p_name:
			biome = d
	if biome.is_empty():
		warnings.append("No biome %s in the table." % p_name)
		return
	var fields := {}
	for f in p_table.get("fields", []):
		fields[f.name] = f
	# The biome's entry per ground octave (none: the ground's own, volume 1).
	var k_top := Octaves.ground_top(lods, voxel_m)
	var per_k := {}
	for a in biome.get("amp", []):
		if a.get("field", "ground") != "ground":
			warnings.append("Multipliers of the climate field %s left out: the mixer has the ground's notes only." % a.field)
			continue
		for k in _ks(a):
			per_k[k] = a
	# The ground's stretched octaves (the table's "ground_stretch": a stretch_y per wavelength range).
	var stretch := {}
	for s in p_table.get("ground_stretch", []):
		for k in _ks(s):
			stretch[k] = int(s.get("stretch_y", 1))
	var mix := add("mix", Vector2(COL.mix, 0.0))
	var rows: Array = model.nodes[mix].params.rows
	var warps := {}
	var chain := ""
	var row := 0
	for k in range(k_top, Octaves.K_MIN - 1, -1):
		var a: Dictionary = per_k.get(k, {})
		var x: float = a.get("x", 1.0)
		if x == 0.0:
			continue
		var y := row * ROW_H
		var note := add("note", Vector2(COL.note, y), {"k": k, "warp_m": float(a.get("warp_m", 0.0)), "stretch_y": stretch.get(k, 1)})
		var last := note
		if a.has("warp_m"):
			var key := ",".join(a.get("warp_fields", []))
			if not warps.has(key):
				warps[key] = _warp(a.get("warp_fields", []), fields, y)
			if warps[key] != "":
				link(warps[key], 0, note, 0)
		if a.has("shape") or a.has("scale") or a.has("offset") or a.has("coeffs"):
			last = add("shape", Vector2(COL.shape, y), _shape(a))
			link(note, 0, last, 0)
		if a.get("multifractal", 0.0) > 0.0:
			var mf := add("multifractal", Vector2(COL.multifractal, y), {"gain": float(a.multifractal)})
			link(last, 0, mf, 0)
			if chain != "":
				link(chain, 1, mf, 1)
			chain = mf
			last = mf
		link(last, 0, mix, rows.size())
		rows.append(Kinds.new_row(x))
		row += 1
	model.nodes[mix].pos = [COL.mix, row * ROW_H * 0.5 - ROW_H]
	var last_id := mix
	var x_next: float = COL.mix + MIX_W
	if biome.has("curve"):
		var cv := add("curve", Vector2(x_next, model.nodes[mix].pos[1]), {"points": biome.curve})
		link(last_id, 0, cv, 0)
		last_id = cv
		x_next += 340.0
	var surface: Dictionary = biome.get("surface", {})
	var colours := {}
	for key in ["flat", "steep"]:
		if surface.get(key) is String:
			colours[key] = surface[key]
	var out := add("output", Vector2(x_next, model.nodes[mix].pos[1]), colours)
	link(last_id, 0, out, 0)


func _shape(p_a: Dictionary) -> Dictionary:
	if p_a.has("coeffs"):
		var c: Array = p_a.coeffs
		return {"preset": "coeffs", "c0": float(c[0]), "c1": float(c[1]), "c2": float(c[2]), "c3": float(c[3])}
	var preset: String = p_a.get("shape", "plain")
	return {"preset": preset, "scale": float(p_a.get("scale", 1.0)), "offset": float(p_a.get("offset", Kinds.preset_offset(preset)))}


# A Climate warp node from the table's three warp fields (x, y, z: seeds s, s + 1, s + 2 and one band range), which
# become the model's climate_warp, or "". The mixer has one climate warp: a second set of fields is left out.
func _warp(p_names: Array, p_fields: Dictionary, p_y: float) -> String:
	if model.has("climate_warp"):
		warnings.append("Warp fields %s left out: the mixer has one climate warp (the first set's)." % [p_names])
		return ""
	if p_names.size() != 3 or not p_fields.has(p_names[0]):
		warnings.append("Warp fields %s not found: those notes are unwarped." % [p_names])
		return ""
	var f: Dictionary = p_fields[p_names[0]]
	var seed: int = f.get("seed", 0)
	for i in 3:
		if p_fields.get(p_names[i], {}).get("seed", -1) != seed + i:
			warnings.append("Warp fields %s: seeds aren't s, s + 1, s + 2; the mixer uses %d, %d, %d." % [p_names, seed, seed + 1, seed + 2])
			break
	var band: Dictionary = f.get("bands", [{}])[0]
	var ks := _ks(band)
	if ks.is_empty():
		warnings.append("Warp fields %s have no octave in range: those notes are unwarped." % [p_names])
		return ""
	model.climate_warp = {"k": ks.max(), "octaves": ks.max() - ks.min() + 1, "amp": float(band.get("amp", 0.5)), "persistence": float(band.get("persistence", 0.5)), "seed": seed}
	return add("warp", Vector2(COL.warp, p_y))


# The octaves a table entry covers: its wavelength_m's, or every one in from_m .. to_m (the engine's band ranges).
func _ks(p_entry: Dictionary) -> Array:
	if p_entry.has("wavelength_m"):
		return [k_of(p_entry.wavelength_m)]
	var ks := []
	for k in range(Octaves.K_MIN, Octaves.k_top(lods) + 1):
		var lam := Octaves.wavelength_m(k, voxel_m)
		if lam >= p_entry.get("from_m", INF) * 0.999 and lam <= p_entry.get("to_m", -INF) * 1.001:
			ks.append(k)
	return ks
