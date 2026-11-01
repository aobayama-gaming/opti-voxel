extends RefCounted
## A world back to a world file, the engine's (res://climate/world.json and its climate.json; src/world_file.hpp and
## climate_parse.cpp describe the format):
## ImportTable read backwards. A biome's height is its Mix, through an optional Height curve, into the Output; each Mix
## row plays a Note through an optional Shape then an optional Multifractal. Each note becomes an "amp" entry (a run
## of equal ones one range, a plain note at volume 1 none: the engine's default), the curve its "curve", the Output's
## colours its surface. What the table can't hold stops the export (errors); what only the preview has (Carve, Note
## warp, an uncentred row, a note's own stretch) is left out with a warning. Mute and solo export as heard: a silent
## row is volume 0.

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const ImportTable := preload("res://debug/biome_mixer/import_table.gd")
const World := preload("res://debug/biome_mixer/world.gd")
const Nodes := preload("res://debug/biome_mixer/nodes.gd")

# Significant digits a number keeps in the file (a float32's).
const DIGITS := 7
# A plain note at volume 1 as written: the engine's default, no entry.
const PLAIN := "{\"x\": 1}"
# A value fits on one line of the file up to this width, else it opens one line per item.
const LINE := 120
# The climate warp's fields when the world has none yet (the engine's wx, wy, wz: seeds s, s + 1, s + 2).
const WARP_NAMES := ["wx", "wy", "wz"]


# {table, errors, warnings}: the world as a table, p_base the table it came from (its comment, the biomes' surface keys
# the mixer doesn't edit: roughness). The table only counts when errors is empty.
static func table(p_world: Dictionary, p_base: Dictionary = {}) -> Dictionary:
	var errors := PackedStringArray()
	var warnings := PackedStringArray()
	var fields: Array = p_world.fields.duplicate(true)
	var warp := _warp_fields(p_world, fields)
	var names := World.field_names(p_world)
	var base := {}
	for b in p_base.get("biomes", []):
		base[b.get("name", "")] = b
	var biomes := []
	var seen := {}
	var n: int = p_world.biomes.size()
	var warped := false
	var stretch := ImportTable.stretch_by_k(p_world)
	for i in n:
		var wb: Dictionary = p_world.biomes[i]
		var bname := String(wb.name)
		if seen.has(bname):
			errors.append("Two biomes are named %s." % bname)
		seen[bname] = true
		var r := _biome(wb.model, p_world, warp.names, stretch)
		for e in r.errors:
			errors.append("%s: %s" % [bname, e])
		for e in r.warnings:
			warnings.append("%s: %s" % [bname, e])
		warped = warped or r.warped
		var b := {"name": bname}
		if i < n - 1:
			b.at = {}
			for f in names:
				if wb.at.has(f):
					b.at[f] = [float(wb.at[f][0]), float(wb.at[f][1])]
			for f in wb.at:
				if not f in names:
					warnings.append("%s: its place on %s, a field the world doesn't have, is left out." % [bname, f])
		elif not wb.get("at", {}).is_empty():
			warnings.append("%s is the default (the last biome): it fills what the others leave, its place is left out." % bname)
		b.amp = r.amp
		if r.has("curve"):
			b.curve = r.curve
		var surface: Dictionary = base.get(bname, {}).get("surface", {}).duplicate(true)
		surface.merge(r.surface, true)
		b.surface = surface
		if not wb.get("clouds", {}).is_empty():
			b.clouds = wb.clouds.duplicate(true)
		if String(wb.get("debug_color", "")) != "":
			b.debug_color = wb.debug_color
		biomes.append(b)
	if warped and warp.missing:
		fields.append_array(warp.fields)
	var t := {}
	if p_base.has("comment"):
		t.comment = p_base.comment
	t.fields = fields
	if not p_world.get("ground_stretch", []).is_empty():
		t.ground_stretch = p_world.ground_stretch.duplicate(true)
	t.biomes = biomes
	return {"table": t, "errors": errors, "warnings": warnings}


# The climate warp (Kinds.climate_warp: an fBm of `octaves` octaves from k down, seeds s, s + 1, s + 2) as the table's
# three warp fields: the world's fields of those seeds, their band set to it in place (p_fields) when it differs, else
# new ones (missing: added only if a note is warped). {names, missing, fields}.
static func _warp_fields(p_world: Dictionary, p_fields: Array) -> Dictionary:
	var cw := Kinds.climate_warp(p_world)
	var vox: float = Octaves.voxel_m
	var lo := int(cw.k) - int(cw.octaves) + 1
	var band := {
		"from_m": Octaves.wavelength_m(lo, vox), "to_m": Octaves.wavelength_m(int(cw.k), vox),
		"amp": float(cw.amp), "persistence": float(cw.persistence),
	}
	var ks := range(lo, int(cw.k) + 1)
	var found := []
	for i in 3:
		var at := -1
		for j in p_fields.size():
			if World.field_seed(p_fields[j]) == int(cw.seed) + i:
				at = j
		found.append(at)
	if not -1 in found:
		for j in found:
			var bands: Array = p_fields[j].get("bands", [])
			var same: bool = bands.size() == 1 and ImportTable.band_ks(bands[0]) == ks \
				and is_equal_approx(float(bands[0].get("amp", 0.0)), band.amp) \
				and is_equal_approx(float(bands[0].get("persistence", 1.0)), band.persistence)
			if not same:
				p_fields[j].bands = [band.duplicate()]
		return {"names": found.map(func(j): return String(p_fields[j].name)), "missing": false, "fields": []}
	var taken := World.field_names(p_world)
	var names := []
	var made := []
	for i in 3:
		var fname: String = WARP_NAMES[i]
		while fname in taken:
			fname += "_"
		names.append(fname)
		made.append({"name": fname, "seed": int(cw.seed) + i, "bands": [band.duplicate()]})
	return {"names": names, "missing": true, "fields": made}


# One biome's graph as its table entries: {amp, curve (if any), surface, warped (a note reads the climate warp),
# errors, warnings}. p_stretch: the world's stretched octaves (ImportTable.stretch_by_k).
static func _biome(p_model: Dictionary, p_world: Dictionary, p_warp: Array, p_stretch: Dictionary) -> Dictionary:
	var r := {"amp": [], "surface": {}, "warped": false, "errors": PackedStringArray(), "warnings": PackedStringArray()}
	var out := Kinds.first_of(p_model, "output")
	if out == "":
		r.errors.append("no Output.")
		return r
	for key in ["flat", "steep"]:
		r.surface[key] = String(Kinds.param(p_model.nodes[out], key))
	if Kinds.input_of(p_model, out, 1) != null:
		r.warnings.append("its Carve is left out: the engine has no holes yet.")
	var src = Kinds.input_of(p_model, out, 0)
	if src != null and p_model.nodes[src.id].kind == "curve":
		r.curve = Kinds.param(p_model.nodes[src.id], "points").map(func(p): return [float(p[0]), float(p[1])])
		src = Kinds.input_of(p_model, src.id, 0)
	if src == null or p_model.nodes[src.id].kind != "mix":
		r.errors.append("its height isn't a Mix (through a Height curve or not) into the Output: the table has no other.")
		return r
	var per_k := _rows(p_model, src.id, p_world, p_warp, p_stretch, r)
	if not r.errors.is_empty():
		return r
	var entries := []
	for k in range(Octaves.ground_top(Octaves.lods, Octaves.voxel_m), Octaves.K_MIN - 1, -1):
		entries.append({"k": k, "e": per_k.get(k, {"x": 0.0})})
	r.amp = _ranges(entries, Octaves.voxel_m)
	return r


# The Mix's rows as {k: entry} (a note in no row is silent); into p_r its errors and warnings.
static func _rows(p_model: Dictionary, p_mix: String, p_world: Dictionary, p_warp: Array, p_stretch: Dictionary, p_r: Dictionary) -> Dictionary:
	var rows := Kinds.rows(p_model.nodes[p_mix])
	var solo := rows.any(func(row): return row.get("solo", false))
	var per_k := {}
	var chains := {}  # k -> the multifractal its chain in comes from ("" for none)
	var mfs := {}  # k -> its multifractal's id
	for i in rows.size():
		var row: Dictionary = rows[i]
		var from = Kinds.input_of(p_model, p_mix, i)
		if from == null:
			continue
		var mf := ""
		var shape := ""
		var id: String = from.id
		if p_model.nodes[id].kind == "multifractal" and int(from.port) == 0:
			mf = id
			var up = Kinds.input_of(p_model, id, 0)
			id = up.id if up != null else ""
		if id != "" and p_model.nodes[id].kind == "shape":
			shape = id
			var up = Kinds.input_of(p_model, id, 0)
			id = up.id if up != null else ""
		if id == "" or p_model.nodes[id].kind != "note":
			p_r.errors.append("row %d isn't a Note through a Shape then a Multifractal (each optional): the table has no other." % (i + 1))
			continue
		var note: Dictionary = p_model.nodes[id]
		var k := int(Kinds.param(note, "k"))
		if per_k.has(k):
			p_r.errors.append("the %s note plays in two rows." % _m(k, p_world))
			continue
		if int(Kinds.param(note, "stretch_y")) != p_stretch.get(k, 1):
			p_r.warnings.append("the %s note's stretch is the world's (ground_stretch, %d), not its own." % [_m(k, p_world), p_stretch.get(k, 1)])
		var heard: bool = not row.get("mute", false) and (not solo or row.get("solo", false))
		if not heard:
			p_r.warnings.append("row %d is silent (muted or not soloed): exported at volume 0." % (i + 1))
		if not row.get("centre", true):
			p_r.warnings.append("row %d isn't centred: the engine always takes a note's mean out." % (i + 1))
		var e := {"x": float(row.get("vol", 1.0)) if heard else 0.0}
		if shape != "":
			e.merge(_shape_keys(p_model.nodes[shape], mf != ""))
		if mf != "":
			e.multifractal = float(Kinds.param(p_model.nodes[mf], "gain"))
			var c = Kinds.input_of(p_model, mf, 1)
			chains[k] = c.id if c != null else ""
			mfs[k] = mf
		var warp_m := float(Kinds.param(note, "warp_m"))
		if bool(Kinds.param(note, "warp")) and warp_m > 0.0:
			if Kinds.input_of(p_model, id, 0) != null:
				p_r.warnings.append("the %s note's Note warp is preview-only: it exports reading the climate warp." % _m(k, p_world))
			e.warp_m = warp_m
			e.warp_fields = p_warp.duplicate()
			p_r.warped = true
		per_k[k] = e
	# The engine chains each multifractal to the next coarser one.
	var ks := mfs.keys()
	ks.sort()
	ks.reverse()
	for j in ks.size():
		var want: String = mfs[ks[j - 1]] if j > 0 else ""
		if chains[ks[j]] != want:
			p_r.warnings.append("the %s note's multifractal chains to the next coarser one in the engine, whatever its link." % _m(ks[j], p_world))
	return per_k


# A Shape node as an entry's keys: the preset's name, scale and offset where they count, or coeffs. An offset (a
# constant term) only counts in a multifractal or on a ridged shape (elsewhere the band goes in without its mean, the
# engine refuses it): it is dropped there, as the preview's centred row ignores it.
static func _shape_keys(p_node: Dictionary, p_mf: bool) -> Dictionary:
	var preset := String(Kinds.param(p_node, "preset"))
	if preset == "coeffs":
		var c := Kinds.shape_coeffs(p_node).map(func(v): return float(v))
		if not p_mf:
			c[0] = 0.0
		return {"coeffs": c}
	var e := {}
	if preset != "plain":
		e.shape = preset
	var a := float(Kinds.param(p_node, "scale"))
	if not is_equal_approx(a, 1.0):
		e.scale = a
	var o := float(Kinds.param(p_node, "offset"))
	if (p_mf or preset == "ridged") and not is_equal_approx(o, Kinds.preset_offset(preset)):
		e.offset = o
	return e


# The note entries (coarsest first, {k, e}) as "amp": a run of equal entries one range, a plain note at volume 1 none.
static func _ranges(p_entries: Array, p_voxel_m: float) -> Array:
	var out := []
	var i := 0
	while i < p_entries.size():
		# Equal as written (_inline keeps DIGITS digits).
		var j := i
		var written := _inline(p_entries[i].e)
		while j + 1 < p_entries.size() and _inline(p_entries[j + 1].e) == written:
			j += 1
		if written != PLAIN:
			# A range even for one octave: the engine trims a range to its LOD count's octaves, it refuses a single
			# wavelength past them.
			var entry := {"from_m": Octaves.wavelength_m(p_entries[j].k, p_voxel_m), "to_m": Octaves.wavelength_m(p_entries[i].k, p_voxel_m)}
			entry.merge(p_entries[i].e)
			out.append(entry)
		i = j + 1
	return out


# A note's wavelength as the graph's titles show it.
static func _m(p_k: int, p_world: Dictionary) -> String:
	return Nodes.fmt_m(Octaves.wavelength_m(p_k, Octaves.voxel_m))


# r_table split back into its files: {world (the world file's keys: p_base's own, the fields left to the climate), fields}.
static func split(p_table: Dictionary, p_base: Dictionary) -> Dictionary:
	var world := {}
	for key in p_base:
		if not key in ["fields", "ground_stretch", "biomes"]:
			world[key] = p_base[key]
	for key in ["ground_stretch", "biomes"]:
		if p_table.has(key):
			world[key] = p_table[key]
	if p_base.get("climate") is Dictionary:
		world.climate = {"fields": p_table.fields}
	return {"world": world, "fields": p_table.fields}


# The climate file p_world's "climate" names next to p_path, "" when it's inline.
static func climate_path(p_world: Dictionary, p_path: String) -> String:
	var climate = p_world.get("climate", "")
	return Octaves.section_path(climate, p_path) if climate is String and climate != "" else ""


# The world over p_path (res://climate/world.json) and, when the warp changed its fields, its climate file, once the
# engine accepts the merged table (ClimateReportNode.check: parsed against the cascade of the mixer's engine; unchecked
# without the extension). {written, message (for the status line), table (read_table's form), warnings}.
static func write(p_world: Dictionary, p_base: Dictionary, p_path: String) -> Dictionary:
	var r := table(p_world, p_base)
	var out := {"written": false, "table": r.table, "warnings": r.warnings}
	if not r.errors.is_empty():
		out.message = "Not exported: " + "  ".join(r.errors)
		return out
	var body := text(r.table)
	var checked := ClassDB.class_exists("ClimateReportNode")
	if checked:
		var err: String = ClassDB.class_call_static("ClimateReportNode", "check", body, Octaves.voxel_m, Octaves.lods)
		if err != "":
			out.message = "Not exported, the engine refuses it: " + err
			return out
	var files := split(r.table, p_base)
	var climate := climate_path(files.world, p_path)
	if climate != "":
		var had = JSON.parse_string(FileAccess.get_file_as_string(climate)) if FileAccess.file_exists(climate) else null
		if (not had is Dictionary or had.get("fields") != files.fields) and not store(climate, text({"fields": files.fields})):
			out.message = "Could not write " + climate
			return out
	if not store(p_path, text(files.world)):
		out.message = "Could not write " + p_path
		return out
	out.written = true
	out.table = files.world.duplicate()
	out.table.fields = files.fields
	out.message = "Exported %d biomes to %s%s%s" % [r.table.biomes.size(), p_path, "" if checked else " (unchecked: no extension)",
		"; %d things left out (the log lists them)" % r.warnings.size() if not r.warnings.is_empty() else ""]
	return out


# p_text into the file p_path (the mixer's saves too): false if it can't be opened.
static func store(p_path: String, p_text: String) -> bool:
	var f := FileAccess.open(p_path, FileAccess.WRITE)
	if f == null:
		return false
	f.store_string(p_text)
	f.close()
	return true


# --- the file ----------------------------------------------------------------------------------------------------

# The table as world.json's text: a value on one line when it fits LINE, else one line per item (one-space indent).
static func text(p_table: Dictionary) -> String:
	return _text(p_table, "") + "\n"


# p_v at p_indent, after p_lead characters on its line (a dictionary item's key).
static func _text(p_v: Variant, p_indent: String, p_lead: int = 0) -> String:
	var one := _inline(p_v)
	# One character for the comma after an item.
	if p_indent.length() + p_lead + one.length() + 1 <= LINE or not (p_v is Dictionary or p_v is Array) or p_v.is_empty():
		return one
	var inner := p_indent + " "
	var items := PackedStringArray()
	if p_v is Dictionary:
		for key in p_v:
			var lead := JSON.stringify(String(key)) + ": "
			items.append(inner + lead + _text(p_v[key], inner, lead.length()))
		return "{\n" + ",\n".join(items) + "\n" + p_indent + "}"
	for x in p_v:
		items.append(inner + _text(x, inner))
	return "[\n" + ",\n".join(items) + "\n" + p_indent + "]"


static func _inline(p_v: Variant) -> String:
	if p_v is Dictionary:
		var items := PackedStringArray()
		for key in p_v:
			items.append(JSON.stringify(String(key)) + ": " + _inline(p_v[key]))
		return "{" + ", ".join(items) + "}"
	if p_v is Array:
		return "[" + ", ".join(PackedStringArray(p_v.map(_inline))) + "]"
	if p_v is float or p_v is int:
		return _num(float(p_v))
	return JSON.stringify(p_v)


# A number to DIGITS significant digits, no trailing zeros (seeds stay integers; a small term keeps its digits).
static func _num(p_v: float) -> String:
	if p_v == roundf(p_v) and absf(p_v) < 1e15:
		return str(int(p_v))
	var decimals := DIGITS - 1 - int(floor(log(absf(p_v)) / log(10.0)))
	if decimals > 15:
		return String.num_scientific(p_v)
	var v := snappedf(p_v, pow(10.0, -maxi(decimals, 0)))
	return str(int(v)) if v == roundf(v) else String.num(v, maxi(decimals, 0))
