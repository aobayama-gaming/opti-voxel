extends RefCounted
## The mixer's world: the whole climate as one document, so a biome is edited inside its world (its notes in the
## graph, its place among the others on the climate chart, its neighbours in the world view). A world is
## {fields (the table's, as JSON), climate_warp, biomes: [{name, at: {field: [centre, span]}, model:
## the biome's graph, debug_color, clouds}] (the last the default: no "at", it fills what the others leave),
## origin_m, view, current (the biome the graph edits), mode ("biome" or "world")}. Its biomes' keys the mixer doesn't
## edit (instances, features, ...) ride along in "keep" from import to export. The weights and the maps are the
## engine's (ClimateReportNode.preview).

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const ImportTable := preload("res://debug/biome_mixer/import_table.gd")

# Where a biome is whole on a field (Climate::KERNEL_FADE 2: within 1 / sqrt(2) of its span): the chart draws it.
const CORE := sqrt(0.5)
# The World mode's view settings, beside the Biome mode's (the mixer's VIEW_DEFAULTS) in the world's "view".
# A table biome's keys the mixer edits (the rest ride along: "keep").
const EDITED := ["name", "at", "graph", "surface", "clouds", "debug_color"]
const WORLD_VIEW := {"extent_m": 400000.0, "map": true, "chart_x": 0, "chart_y": 1}


# The world file p_path as one table: its keys and its climate's fields (the climate a path next to it, or inline).
static func read_table(p_path: String) -> Dictionary:
	var t = JSON.parse_string(FileAccess.get_file_as_string(p_path))
	if not t is Dictionary:
		return {}
	t.fields = Octaves.section(t, "climate", p_path).get("fields", [])
	return t


# A world from a climate table (read_table): each biome imported as a graph (ImportTable), its
# place as [centre, span] per field.
static func from_table(p_table: Dictionary) -> Dictionary:
	var w := {
		"version": 1, "kind": "world", "name": "world", "origin_m": [0.0, 0.0],
		"fields": p_table.get("fields", []).duplicate(true),
		"biomes": [], "current": 0, "mode": "biome", "warnings": [],
	}
	for b in p_table.get("biomes", []):
		var r: Dictionary = ImportTable.new(b.get("name", "?")).import_biome(p_table)
		if r.model.has("climate_warp") and not w.has("climate_warp"):
			w.climate_warp = r.model.climate_warp
		w.warnings.append_array(r.warnings)
		var keep := {}
		for key in b:
			if not key in EDITED:
				keep[key] = b[key]
		w.biomes.append({"name": b.get("name", "?"), "at": at_of(b.get("at", {})), "model": r.model, "debug_color": b.get("debug_color", ""), "clouds": b.get("clouds", {}), "keep": keep})
	w.climate_warp = Kinds.climate_warp(w)
	return w


# A table biome's "at" as {field: [centre, span]} ({min, max}: their middle and half their gap).
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


static func field_index(p_world: Dictionary, p_name: String) -> int:
	return field_names(p_world).find(p_name)


# FNV-1a of a field's name: its seed without one (Climate's parser, name_hash).
static func name_hash(p_name: String) -> int:
	var h := 2166136261
	for byte in p_name.to_utf8_buffer():
		h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
	return h


static func field_seed(p_field: Dictionary) -> int:
	return int(p_field.seed) if p_field.has("seed") else name_hash(String(p_field.get("name", "")))


# A biome's debug colour (#rrggbb), else one from its name's hash (Climate's parser: name_hash & 0xFFFFFF).
static func debug_color(p_biome: Dictionary) -> Color:
	var hex := String(p_biome.get("debug_color", ""))
	return Color(hex) if hex.begins_with("#") else Color.hex((name_hash(String(p_biome.name)) & 0xFFFFFF) << 8 | 0xFF)
