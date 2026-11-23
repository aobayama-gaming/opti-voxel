extends RefCounted
## The mixer's world: the world file itself (res://climate/world.json, or the file --open names: its biomes, their
## graphs, places, surfaces, instances and features, its shape, seed and sea, every other key as it is), with the mixer's
## keys beside them: "fields" (its climate's, each with its "mean" and "poles", edited by the fields panel), "path" (the
## file it came from and exports to), "current" (the biome the graph edits), "mode" ("biome" or "world"), "view",
## "mixer_version" and "sources" (the files its biomes and their list came from, as read: World.file_of names a file
## again where its biome is unchanged). A biome's "at" is kept as {field: [centre, span]} (the default, the last, has
## none). The weights and the maps are the engine's (ClimateReportNode.preview); the file it writes is the world minus the
## mixer's keys, checked by the engine first.

const EngineInfo := preload("res://debug/biome_mixer/engine_info.gd")

# The world's keys that are the mixer's, not the file's.
const MIXER_KEYS := ["fields", "path", "current", "mode", "view", "mixer_version", "name", "sources"]
# The world file's keys that may name a file (WorldFile: relative to the world file). "biomes" may also list files, and
# an inline climate's "fields" name one (_absolute); "shape" and "sea" never do.
const SECTIONS := ["engine", "climate", "look", "instances", "features", "species", "biomes", "edits"]
# A field's offsets (WorldFile: the climate's "<field>": {mean, poles}), kept on the mixer's fields.
const OFFSETS := ["mean", "poles"]
# The format the mixer writes (3: single-octave notes, warp nodes; 4: multifractals with a parent).
const VERSION := 4
# The World mode's view settings, beside the Biome mode's (the mixer's VIEW_DEFAULTS) in the world's "view".
const WORLD_VIEW := {"extent_m": 100000.0, "chart_x": 0, "chart_y": 1}
# Significant digits a number keeps in the file (a float32's).
const DIGITS := 7
# A value fits on one line of the file up to this width, else it opens one line per item.
const LINE := 120


# The world file p_path as the mixer's world ({} if it isn't one): its climate's fields beside it (their offsets on
# them), its biomes inline (a list file's, a biome file's: their names kept in "sources"), the default moved last, every
# biome's place as [centre, span] (a biome without a graph gets the template's). p_file: the world file a saved world
# without a "path" came from (an autosave older than the key: world.json), else p_path.
static func read(p_path: String, p_file: String = "") -> Dictionary:
	var w = JSON.parse_string(FileAccess.get_file_as_string(p_path)) if FileAccess.file_exists(p_path) else null
	if not w is Dictionary:
		return {}
	# A saved world (the autosave, Save) keeps the file it came from: the files it names are relative to that one.
	w.path = String(w.get("path", p_file if p_file != "" else p_path))
	if not w.get("sources") is Dictionary:
		w.sources = {}
	var named := _read_biomes(w)
	if named.is_empty():
		return {}
	var biomes: Array = named.map(func(e): return e[0])
	var n: int = biomes.size()
	# The default (the one without "at") last, as the engine reads it.
	var d := range(n - 1).filter(func(i): return not biomes[i].has("at"))
	if not d.is_empty():
		named.append(named.pop_at(d[0]))
		biomes.append(biomes.pop_at(d[0]))
	for i in n:
		var b: Dictionary = biomes[i]
		b.at = at_of(b.get("at", {})) if i < n - 1 else {}
		if not b.has("graph"):
			b.graph = EngineInfo.template_biome(String(b.get("name", "biome"))).graph
	w.biomes = biomes
	for e in named:
		if e[1] != "":
			w.sources.get_or_add("biome_files", {})[e[1]] = _form(e[0])
	# As read from the list file (an autosave keeps the forms it was read with, not its edits).
	if w.sources.has("biome_list") and not w.sources.biome_list.has("forms"):
		w.sources.biome_list.forms = biomes.map(_form)
	if not w.has("fields"):
		w.fields = _read_fields(w)
	w.mixer_version = VERSION
	w.current = clampi(int(w.get("current", 0)), 0, n - 1)
	w.mode = String(w.get("mode", "biome"))
	return w


# p_world's biomes as [biome, the name of the file it came from ("" inline)]: a list file's read (its name in
# sources.biome_list), a biome file's (relative to the file naming it). [] if one can't be read.
static func _read_biomes(r_world: Dictionary) -> Array:
	var listed = r_world.get("biomes", [])
	var dir_of := String(r_world.path)
	if listed is String:
		r_world.sources.biome_list = {"path": listed}
		dir_of = EngineInfo.section_path(listed, dir_of)
		listed = _json_at(dir_of)
	var out := []
	if not listed is Array:
		return out
	for v in listed:
		if v is Dictionary:
			out.append([v, ""])
			continue
		var b = _json_at(EngineInfo.section_path(String(v), dir_of)) if v is String else null
		if not b is Dictionary:
			return []
		# A list file's biome files are relative to it: the world names them absolute.
		out.append([b, String(v) if dir_of == String(r_world.path) else EngineInfo.section_path(String(v), dir_of)])
	return out


# The climate's fields (a list or the file holding it) with their offsets on them (the climate's "<field>" keys).
static func _read_fields(p_world: Dictionary) -> Array:
	var sec := EngineInfo.section(p_world, "climate", String(p_world.path))
	var fields = sec.get("fields", [])
	if fields is String:
		fields = _json_at(EngineInfo.section_path(fields, _climate_path(p_world)))
	if not fields is Array:
		return []
	fields = fields.duplicate(true)
	for f in fields:
		var o = sec.get(String(f.get("name", "")), {})
		if o is Dictionary:
			for k in OFFSETS:
				if o.has(k):
					f[k] = float(o[k])
	return fields


# The file the climate's "fields" name is relative to: the climate file, or the world's.
static func _climate_path(p_world: Dictionary) -> String:
	var c = p_world.get("climate")
	return EngineInfo.section_path(c, String(p_world.path)) if c is String else String(p_world.path)


static func _json_at(p_path: String) -> Variant:
	return JSON.parse_string(FileAccess.get_file_as_string(p_path)) if FileAccess.file_exists(p_path) else null


# A biome as its file holds it: the default without a place.
static func _form(p_biome: Dictionary) -> Dictionary:
	var b: Dictionary = p_biome.duplicate(true)
	if b.get("at", {}).is_empty():
		b.erase("at")
	return b


# Two biomes the same but for their graphs' positions (the mixer's layout, the engine ignores it).
static func _same(p_a: Dictionary, p_b: Dictionary) -> bool:
	return _unplaced(p_a) == _unplaced(p_b)


static func _unplaced(p_biome: Dictionary) -> Dictionary:
	var b: Dictionary = p_biome.duplicate(true)
	for n in b.get("graph", {}).get("nodes", []):
		n.erase("pos")
	return b


# A biome's "at" as {field: [centre, span]} ({min, max}: their middle and half their gap).
static func at_of(p_at: Dictionary) -> Dictionary:
	var at := {}
	for f in p_at:
		var v = p_at[f]
		if v is Array and v.size() == 2:
			at[f] = [float(v[0]), float(v[1])]
		elif v is Dictionary and v.has("min") and v.has("max"):
			at[f] = [0.5 * (float(v.min) + float(v.max)), 0.5 * (float(v.max) - float(v.min))]
	return at


static func field_names(p_world: Dictionary) -> Array:
	return p_world.fields.map(func(f): return String(f.get("name", "?")))


static func debug_color(p_biome: Dictionary) -> Color:
	return EngineInfo.debug_color(p_biome)


# The world's "shape" kind (WorldFile: flat without the key).
static func shape_kind(p_world: Dictionary) -> String:
	var s = p_world.get("shape", {})
	return String(s.get("kind", "flat")) if s is Dictionary else "flat"


# The span of the views that frames a planet (metres; 0 on the flat world): its diameter, a torus's outer one. The maps
# are the base's tangent plane traced down to the ground, so wider shows nothing more of the planet.
static func planet_extent_m(p_world: Dictionary) -> float:
	if shape_kind(p_world) == "flat":
		return 0.0
	var shape: Dictionary = p_world.shape
	return 2.0 * (float(shape.get("radius_m", 0.0)) + float(shape.get("ring_m", 0.0)))


# The world as its file: a copy without the mixer's keys, the default biome without a place, its fields (edited or not)
# and their offsets in its climate. p_keep: the files it named kept where their content is unchanged (a biome file, the
# biomes' list file, the climate file, its fields file), else inline (the mixer never writes a file another world may
# name). p_keep false: everything the mixer edits inline (the preview, the frozen file).
static func file_of(p_world: Dictionary, p_keep: bool = true) -> Dictionary:
	var f: Dictionary = p_world.duplicate(true)
	for key in MIXER_KEYS:
		f.erase(key)
	var forms: Array = p_world.biomes.map(_form)
	f.biomes = _biome_names(p_world.get("sources", {}), forms) if p_keep else forms
	_climate_into(p_world, f, p_keep)
	return f


# p_forms with each biome a file holds unchanged named by it, or the list file's name if none changed.
static func _biome_names(p_sources: Dictionary, p_forms: Array) -> Variant:
	var list: Dictionary = p_sources.get("biome_list", {})
	var forms: Array = list.get("forms", [])
	if not forms.is_empty() and forms.size() == p_forms.size() and range(forms.size()).all(func(i): return _same(forms[i], p_forms[i])):
		return list.path
	var files: Dictionary = p_sources.get("biome_files", {})
	return p_forms.map(func(b):
		for p in files:
			if _same(files[p], b):
				return p
		return b)


# p_fields as [the fields without their offsets, {field name: {mean, poles}} (the nonzero ones)].
static func _split(p_fields: Array) -> Array:
	var plain := []
	var offsets := {}
	for f in p_fields:
		var c: Dictionary = f.duplicate(true)
		var o := {}
		for k in OFFSETS:
			if float(c.get(k, 0.0)) != 0.0:
				o[k] = float(c[k])
			c.erase(k)
		plain.append(c)
		if not o.is_empty():
			offsets[String(c.get("name", ""))] = o
	return [plain, offsets]


# r_file's climate from p_world's fields: its file kept (p_keep) when they are its fields and offsets, else inline (an
# inline climate's other keys kept, its fields file kept when they are its fields); none for no fields and no climate.
static func _climate_into(p_world: Dictionary, r_file: Dictionary, p_keep: bool) -> void:
	var ours := _split(p_world.fields)
	var given = p_world.get("climate")
	var c := {}
	if given is String:
		if p_keep and _split(_read_fields(p_world)) == ours:
			return
	elif given is Dictionary:
		for k in given:
			if k == "fields" or k == "comment":
				c[k] = given[k]
	elif ours[0].is_empty() and ours[1].is_empty():
		return
	var named = c.get("fields")
	if not (p_keep and named is String and _split(_read_fields(p_world))[0] == ours[0]):
		c.fields = ours[0]
	c.merge(ours[1], true)
	r_file.climate = c


# What the engine draws for p_world (ClimateReportNode.preview, the mixer's view and "Open in game"): its file with
# everything the mixer edits inline and the sections it names absolute paths (WorldFile reads them relative to the world
# file); p_alone >= 0: that biome alone (the default).
static func preview_file(p_world: Dictionary, p_alone: int = -1) -> Dictionary:
	var f := _absolute(file_of(p_world, false), String(p_world.path))
	if p_alone >= 0:
		var b: Dictionary = f.biomes[p_alone]
		b.erase("at")
		f.biomes = [b]
	return f


# p_file with the files it names relative to the world file p_path absolute.
static func _absolute(p_file: Dictionary, p_path: String) -> Dictionary:
	for key in SECTIONS:
		if p_file.get(key) is String:
			p_file[key] = EngineInfo.section_path(p_file[key], p_path)
	if p_file.get("biomes") is Array:
		p_file.biomes = p_file.biomes.map(func(b): return EngineInfo.section_path(b, p_path) if b is String else b)
	if p_file.get("climate") is Dictionary and p_file.climate.get("fields") is String:
		p_file.climate.fields = EngineInfo.section_path(p_file.climate.fields, p_path)
	return p_file


# p_file with every section it names read in its place (relative to the world file p_path): one file, nothing named.
static func _inlined(p_file: Dictionary, p_path: String) -> Dictionary:
	for key in SECTIONS:
		if p_file.get(key) is String:
			var v = _json_at(EngineInfo.section_path(p_file[key], p_path))
			if v != null:
				p_file[key] = v
	return p_file


# The engine's verdict on p_world (ClimateReportNode.check: the whole file, at its own engine and over its own planet,
# which refuses a ground under its centre the flat world takes): "" when it accepts.
static func check(p_world: Dictionary) -> String:
	return ClassDB.class_call_static("ClimateReportNode", "check", JSON.stringify(preview_file(p_world)), String(p_world.path).get_base_dir())


# The world over its own file (p_world.path, or p_path: elsewhere, the files it names made absolute so it loads from
# there; the files it named kept where unchanged, file_of) once the engine accepts it: {written, message (for the status
# line)}. p_inline: everything in the one file (the biomes, the climate, every section; "Freeze").
static func write(p_world: Dictionary, p_path: String = "", p_inline: bool = false) -> Dictionary:
	var path := p_path if p_path != "" else String(p_world.path)
	var err := check(p_world)
	if err != "":
		return {"written": false, "message": "Not exported, the engine refuses it: " + err}
	var f := file_of(p_world, not p_inline)
	if p_inline:
		_inlined(f, String(p_world.path))
	elif path.get_base_dir() != String(p_world.path).get_base_dir():
		_absolute(f, String(p_world.path))
	if not store(path, text(f)):
		return {"written": false, "message": "Could not write " + path}
	if p_inline:
		return {"written": true, "message": "Froze %d biomes, the climate and every section into %s" % [p_world.biomes.size(), path]}
	var inlined: bool = f.get("climate") is Dictionary and p_world.get("climate") is String
	return {"written": true, "message": "Exported %d biomes to %s%s" % [p_world.biomes.size(), path, " (its edited fields inline, the climate file untouched)" if inlined else ""]}


# p_text into the file p_path (the mixer's saves too): false if it can't be opened.
static func store(p_path: String, p_text: String) -> bool:
	var f := FileAccess.open(p_path, FileAccess.WRITE)
	if f == null:
		return false
	f.store_string(p_text)
	f.close()
	return true


# --- the file's text ---------------------------------------------------------------------------------------------

# A value as world.json's text: on one line when it fits LINE, else one line per item (one-space indent).
static func text(p_value: Dictionary) -> String:
	return _text(p_value, "") + "\n"


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
