extends RefCounted
## The mixer's world: the world file itself (res://climate/world.json, or the file --open names: its biomes, their
## graphs, places, surfaces, instances and features, its planet, every other key as it is), with the mixer's keys beside
## them: "fields" (its climate's, edited by the fields panel), "path" (the file it came from and exports to), "current"
## (the biome the graph edits), "mode" ("biome" or "world"), "view" and "mixer_version". A biome's "at" is kept as
## {field: [centre, span]} (the default, the last, has none). The weights and the maps are the engine's
## (ClimateReportNode.preview); the file it writes is the world minus the mixer's keys, checked by the engine first.

const EngineInfo := preload("res://debug/biome_mixer/engine_info.gd")

# The world's keys that are the mixer's, not the file's.
const MIXER_KEYS := ["fields", "path", "current", "mode", "view", "mixer_version", "name"]
# The world file's keys that may name a file (WorldFile: relative to the world file).
const SECTIONS := ["engine", "climate", "look", "zone", "instances", "features", "species"]
# The format the mixer writes (3: single-octave notes, warp nodes; 4: multifractals with a parent).
const VERSION := 4
# The World mode's view settings, beside the Biome mode's (the mixer's VIEW_DEFAULTS) in the world's "view".
const WORLD_VIEW := {"extent_m": 100000.0, "chart_x": 0, "chart_y": 1}
# Significant digits a number keeps in the file (a float32's).
const DIGITS := 7
# A value fits on one line of the file up to this width, else it opens one line per item.
const LINE := 120


# The world file p_path as the mixer's world ({} if it isn't one): its climate's fields beside it, every biome's place as
# [centre, span] (a biome without a graph gets the template's). p_file: the world file a saved world without a "path"
# came from (an autosave older than the key: world.json), else p_path.
static func read(p_path: String, p_file: String = "") -> Dictionary:
	var w = JSON.parse_string(FileAccess.get_file_as_string(p_path)) if FileAccess.file_exists(p_path) else null
	# Biomes named by a file (WorldFile: the list it holds, ground.biomes.json): edited here, exported inline.
	if w is Dictionary and w.get("biomes") is String:
		var listed = JSON.parse_string(FileAccess.get_file_as_string(EngineInfo.section_path(w.biomes, p_path)))
		w.biomes = listed if listed is Array else []
	if not w is Dictionary or w.get("biomes", []).is_empty():
		return {}
	# A saved world (the autosave, Save) keeps the file it came from.
	w.path = String(w.get("path", p_file if p_file != "" else p_path))
	if not w.has("fields"):
		w.fields = EngineInfo.section(w, "climate", w.path).get("fields", [])
	var n: int = w.biomes.size()
	for i in n:
		var b: Dictionary = w.biomes[i]
		b.at = at_of(b.get("at", {})) if i < n - 1 else {}
		if not b.has("graph"):
			b.graph = EngineInfo.template_biome(String(b.get("name", "biome"))).graph
	w.mixer_version = VERSION
	w.current = clampi(int(w.get("current", 0)), 0, n - 1)
	w.mode = String(w.get("mode", "biome"))
	return w


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


# The span of the views that frames a planet (metres; 0 on the flat world): its diameter, a torus's outer one. The maps
# are the base's tangent plane traced down to the ground, so wider shows nothing more of the planet.
static func planet_extent_m(p_world: Dictionary) -> float:
	var planet: Dictionary = p_world.get("planet", {})
	return 2.0 * (float(planet.get("radius_m", 0.0)) + float(planet.get("ring_m", 0.0)))


# The world as its file: a copy without the mixer's keys, the default biome without a place, its fields (edited or not)
# in its climate: inline where it was inline; a climate file the world names kept unless the fields differ from it,
# then inline (the mixer never writes a climate file another world may name). p_climate false: that comparison (a file
# read) skipped, the climate as it is (preview_file inlines the fields anyway, on every edit).
static func file_of(p_world: Dictionary, p_climate: bool = true) -> Dictionary:
	var f: Dictionary = p_world.duplicate(true)
	for key in MIXER_KEYS:
		f.erase(key)
	for b in f.biomes:
		if b.get("at", {}).is_empty():
			b.erase("at")
	if f.get("climate") is Dictionary:
		f.climate.fields = p_world.fields.duplicate(true)
	elif p_climate and EngineInfo.section(p_world, "climate", String(p_world.path)).get("fields", []) != p_world.fields:
		f.climate = {"fields": p_world.fields.duplicate(true)}
	return f


# What the engine draws for p_world (ClimateReportNode.preview, the mixer's view and "Open in game"): its file with
# the sections it names absolute paths (WorldFile reads them relative to the world file) and its fields inline;
# p_alone >= 0: that biome alone (the default).
static func preview_file(p_world: Dictionary, p_alone: int = -1) -> Dictionary:
	var f := _absolute(file_of(p_world, false), String(p_world.path))
	f.climate = {"fields": p_world.fields.duplicate(true)}
	if p_alone >= 0:
		var b: Dictionary = f.biomes[p_alone]
		b.erase("at")
		f.biomes = [b]
	return f


# p_file with the sections it names relative to the world file p_path absolute.
static func _absolute(p_file: Dictionary, p_path: String) -> Dictionary:
	for key in SECTIONS:
		if p_file.get(key) is String:
			p_file[key] = EngineInfo.section_path(p_file[key], p_path)
	return p_file


# The engine's verdict on p_world (ClimateReportNode.check: the whole file, at its own engine and over its own planet,
# which refuses a ground under its centre the flat world takes): "" when it accepts.
static func check(p_world: Dictionary) -> String:
	return ClassDB.class_call_static("ClimateReportNode", "check", JSON.stringify(preview_file(p_world)), String(p_world.path).get_base_dir())


# The world over its own file (p_world.path, or p_path: elsewhere, the sections it names made absolute so it loads from
# there; its climate file untouched, file_of) once the engine accepts it: {written, message (for the status line)}.
static func write(p_world: Dictionary, p_path: String = "") -> Dictionary:
	var path := p_path if p_path != "" else String(p_world.path)
	var err := check(p_world)
	if err != "":
		return {"written": false, "message": "Not exported, the engine refuses it: " + err}
	var f := file_of(p_world)
	if path.get_base_dir() != String(p_world.path).get_base_dir():
		_absolute(f, String(p_world.path))
	if not store(path, text(f)):
		return {"written": false, "message": "Could not write " + path}
	var inlined: bool = f.get("climate") is Dictionary and not p_world.get("climate") is Dictionary
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
