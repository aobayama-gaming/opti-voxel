extends RefCounted
## The mixer's world: the whole climate as one document, so a biome is edited inside its world (its notes in the
## graph, its place among the others on the climate chart, its neighbours in the world view). A world is
## {fields (the table's, as JSON), climate_warp, biomes: [{name, at: {field: [centre, span]}, model:
## the biome's graph, debug_color, clouds}] (the last the default: no "at", it fills what the others leave),
## origin_m, view, current (the biome the graph edits), mode ("biome" or "world")}.
## The weights are the engine's (Climate::weights, src/climate.cpp) in float: a biome's kernel is the product over the
## fields it names of the smoothstep of clamp((1 - u^2) KERNEL_FADE, 0, 1), u = (f - centre) / span; the default's
## (1 - sum)^2; each over their sum.

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const ImportTable := preload("res://debug/biome_mixer/import_table.gd")

# Climate::KERNEL_FADE: on each field a biome is whole within 1 / sqrt(2) of its span.
const KERNEL_FADE := 2.0
const CORE := sqrt(1.0 - 1.0 / KERNEL_FADE)
# The World mode's view settings, beside the Biome mode's (the mixer's VIEW_DEFAULTS) in the world's "view".
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
		w.biomes.append({"name": b.get("name", "?"), "at": at_of(b.get("at", {})), "model": r.model, "debug_color": b.get("debug_color", ""), "clouds": b.get("clouds", {})})
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
		h = ((h ^ byte) * 16777619) & Octaves.MASK
	return h


static func field_seed(p_field: Dictionary) -> int:
	return int(p_field.seed) if p_field.has("seed") else name_hash(String(p_field.get("name", "")))


# A field's octaves as the engine inserts them ([{k, amp}], coarsest first; Climate's parser, bands): a band's
# wavelength_m, or every octave of its range from_m .. to_m within K_MIN .. k_top, amp at the range's top (to_m)
# times persistence per finer octave.
static func field_octaves(p_world: Dictionary, p_field: Dictionary) -> Array:
	var out := []
	for band in p_field.get("bands", []):
		var amp := float(band.get("amp", 0.0))
		if band.has("wavelength_m"):
			out.append({"k": ImportTable.band_ks(band)[0], "amp": amp})
			continue
		var hi := Octaves.lattice_k(float(band.to_m), Octaves.voxel_m)
		for k in ImportTable.band_ks(band):
			out.append({"k": k, "amp": amp * pow(float(band.get("persistence", 1.0)), hi - k)})
	out.sort_custom(func(a, b): return a.k > b.k)
	return out


# A field's terms as [{source: -1 for the ground or an earlier field's index, coef: per metre or x}].
static func field_terms(p_world: Dictionary, p_i: int) -> Array:
	var out := []
	for t in p_world.fields[p_i].get("terms", []):
		var src := String(t.get("source", "ground"))
		if src == "ground":
			out.append({"source": -1, "coef": float(t.get("per_m", 0.0))})
		else:
			out.append({"source": field_index(p_world, src), "coef": float(t.get("x", 0.0))})
	return out


static func has_ground_terms(p_world: Dictionary) -> bool:
	for i in p_world.fields.size():
		if field_terms(p_world, i).any(func(t): return t.source < 0):
			return true
	return false


# One field's factor of a kernel (the engine's axis_factor, in double).
static func axis_factor(p_value: float, p_centre: float, p_span: float) -> float:
	var u := (p_value - p_centre) / p_span
	var s := clampf((1.0 - u * u) * KERNEL_FADE, 0.0, 1.0)
	return s * s * (3.0 - 2.0 * s)


# The biomes' weights at the field values p_values (one per field, the world's order).
static func weights(p_world: Dictionary, p_values: Array) -> Array:
	var names := field_names(p_world)
	var n: int = p_world.biomes.size()
	var w := []
	w.resize(n)
	w.fill(0.0)
	var sum := 0.0
	for b in n - 1:
		var k := 1.0
		var at: Dictionary = p_world.biomes[b].at
		for f in at:
			var i := names.find(f)
			if i >= 0:
				k *= axis_factor(p_values[i], at[f][0], at[f][1])
		w[b] = k
		sum += k
	w[n - 1] = (1.0 - sum) * (1.0 - sum) if sum < 1.0 else 0.0
	var total: float = sum + w[n - 1]
	for b in n:
		w[b] /= total
	return w


# A biome's debug colour (#rrggbb), else one from its name's hash (Climate's parser: name_hash & 0xFFFFFF).
static func debug_color(p_biome: Dictionary) -> Color:
	var hex := String(p_biome.get("debug_color", ""))
	return Color(hex) if hex.begins_with("#") else Color.hex((name_hash(String(p_biome.name)) & 0xFFFFFF) << 8 | 0xFF)
