extends RefCounted
## The mixer's node kinds: their ports and parameters (data only; nodes.gd builds their UI, compile.gd their shader).
## A port is [label, type]; a node connects an output to an input of the same type, one link per input.

enum { FIELD, WARP, CHAIN, CARVE }

const PORT_COLORS := {
	FIELD: Color(0.55, 0.8, 1.0),
	WARP: Color(0.95, 0.6, 0.95),
	CHAIN: Color(1.0, 0.8, 0.35),
	CARVE: Color(0.95, 0.45, 0.4),
}

# The plots' background (the phi graph, the curve editor).
const PLOT_BG := Color(0.1, 0.11, 0.13)

const SHAPES := ["plain", "billow", "ridged", "square", "coeffs"]
# The engine's shape constants: |n| smoothed as sqrt(n^2 + E^2) - E (Cascade::SMOOTH_ABS_E), and E[sabs(n)], E[n^2]
# of its noise (Cascade::NOISE_ABS_MEAN, NOISE_SQ_MEAN).
const SMOOTH_ABS_E := 0.1
const NOISE_ABS_MEAN := 0.1527
const NOISE_SQ_MEAN := 0.0728

# The climate's warp fields when a biome has none of its own (a model's "climate_warp"): world.json's wx, wy, wz, an
# fBm from note k down (seeds seed, seed + 1, seed + 2 for x, y, z).
const CLIMATE_WARP := {"k": 17, "octaves": 4, "amp": 0.5, "persistence": 0.6, "seed": 905}

# A parameter: name, type (float, int, bool, enum, color, k: a wavelength, points: the curve's, rows: the mix's),
# default and, for numbers, the slider's range (the number box takes any value) and a tooltip.
const KINDS := {
	"note": {
		"title": "Note",
		"hint": "One octave of the ground: a 3D noise in [-1, 1] at one wavelength. Every biome of the climate plays the same notes, the ground's: a biome only shapes and mixes them.",
		"in": [["warp", WARP]],
		"out": [["n", FIELD]],
		"params": [
			{"name": "k", "type": "k", "default": 12},
			{"name": "warp_m", "type": "float", "default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.1, "hint": "How far the warp moves the point the note is read at: metres per unit of the warp (a warp's value is within about +-1)"},
			{"name": "stretch_y", "type": "int", "default": 1, "min": 1, "max": 64, "hint": "The engine's stretched octave (ground_stretch, a power of two): the noise read at y / stretch from the datum, a height field where it dominates"},
		],
	},
	"warp": {
		"title": "Climate warp",
		"hint": "The climate's warp fields (wx, wy, wz): three fBms every biome shares, read once per point however many notes they move. Their octaves are the climate's, not the biome's (CLIMATE_WARP; an import takes its table's).",
		"in": [],
		"out": [["xyz", WARP]],
		"params": [],
	},
	"note_warp": {
		"title": "Note warp",
		"hint": "The cheap warp: moves a note by the values of coarser notes (x, y, z: link Notes' n). Notes the biome already plays cost no read. Preview only: the engine's warps are climate fields.",
		"in": [["x", FIELD], ["y", FIELD], ["z", FIELD]],
		"out": [["xyz", WARP]],
		"params": [],
	},
	"shape": {
		"title": "Shape",
		"hint": "phi(n) = c0 + c1 n + c2 |n| + c3 n^2, the engine's band shapes (|n| smoothed below 0.1).",
		"in": [["n", FIELD]],
		"out": [["phi", FIELD]],
		"params": [
			{"name": "preset", "type": "enum", "options": SHAPES, "default": "ridged",
				"option_params": {"offset": {"plain": 0.0, "billow": 0.0, "ridged": 1.0, "square": 0.0, "coeffs": 0.0}}},
			{"name": "scale", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "hint": "a: the shape reads a n"},
			{"name": "offset", "type": "float", "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "hint": "o: plain a n + o, billow 2|a n| - 1 + o, ridged (o - |a n|)^2, square (a n)^2 + o"},
			{"name": "c0", "type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
			{"name": "c1", "type": "float", "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01},
			{"name": "c2", "type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
			{"name": "c3", "type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
		],
	},
	"multifractal": {
		"title": "Multifractal",
		"hint": "w = clamp(gain * chain in): the signal times w. Chain out (w * signal) feeds the next finer note: detail only where the coarser one was high (Musgrave).",
		"in": [["signal", FIELD], ["chain in", CHAIN]],
		"out": [["signal", FIELD], ["chain out", CHAIN]],
		"params": [
			{"name": "gain", "type": "float", "default": 2.0, "min": 0.0, "max": 4.0, "step": 0.01},
		],
	},
	"mix": {
		"title": "Mix",
		"hint": "The notes to metres: each row's volume x the ground's amplitude of its note (lambda / 8). M mutes, S solos, C centres (takes the shape's mean out, as the engine).",
		"in": [],
		"out": [["height m", FIELD]],
		"params": [
			{"name": "rows", "type": "rows", "default": []},
		],
	},
	"add": {
		"title": "Add",
		"in": [["a", FIELD], ["b", FIELD]],
		"out": [["a + b", FIELD]],
		"params": [],
	},
	"multiply": {
		"title": "Multiply",
		"in": [["a", FIELD], ["b", FIELD]],
		"out": [["a x b", FIELD]],
		"params": [],
	},
	"constant": {
		"title": "Constant",
		"in": [],
		"out": [["c", FIELD]],
		"params": [
			{"name": "value", "type": "float", "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
		],
	},
	"curve": {
		"title": "Height curve",
		"hint": "A remap of the height (metres in, metres out) through its points: the engine's height + height_m * curve(ground). Past the ends it follows the height again. Drag points, double-click to add, right-click to remove.",
		"in": [["h", FIELD]],
		"out": [["R(h)", FIELD]],
		"params": [
			{"name": "points", "type": "points", "default": []},
		],
	},
	"terrace": {
		"title": "Terrace",
		"hint": "libnoise's Terrace every step_m, the engine's biome terraces; amount blends it in.",
		"in": [["h", FIELD]],
		"out": [["h", FIELD]],
		"params": [
			{"name": "step_m", "type": "float", "default": 40.0, "min": 0.5, "max": 500.0, "step": 0.5},
			{"name": "invert", "type": "bool", "default": false},
			{"name": "amount", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
		],
	},
	"carve": {
		"title": "Carve (preview only)",
		"hint": "Air where the field is above the threshold, between top_m and bottom_m below the surface: caves. The engine has no carve yet.",
		"in": [["field", FIELD]],
		"out": [["holes", CARVE]],
		"params": [
			{"name": "threshold", "type": "float", "default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01},
			{"name": "top_m", "type": "float", "default": 2.0, "min": -50.0, "max": 200.0, "step": 0.5},
			{"name": "bottom_m", "type": "float", "default": 200.0, "min": 0.0, "max": 2000.0, "step": 1.0},
			{"name": "fade_m", "type": "float", "default": 10.0, "min": 0.1, "max": 200.0, "step": 0.1},
		],
	},
	"output": {
		"title": "Output",
		"hint": "The biome: the ground's height (metres above the datum) and its holes; its ground colours, flat and steep (the engine's slope blend).",
		"in": [["height m", FIELD], ["holes", CARVE]],
		"out": [],
		"params": [
			{"name": "flat", "type": "color", "default": "#5c7a3a"},
			{"name": "steep", "type": "color", "default": "#6b625a"},
		],
	},
}

# The kinds the Add menu lists, in order.
const MENU := ["note", "shape", "multifractal", "warp", "note_warp", "mix", "add", "multiply", "constant", "curve", "terrace", "carve", "output"]


static func inputs(p_node: Dictionary) -> Array:
	if p_node.kind == "mix":
		var ins := []
		for i in rows(p_node).size() + 1:
			ins.append(["row %d" % (i + 1), FIELD])
		return ins
	return KINDS[p_node.kind]["in"]


static func outputs(p_node: Dictionary) -> Array:
	return KINDS[p_node.kind]["out"]


static func param_def(p_kind: String, p_name: String) -> Dictionary:
	for d in KINDS[p_kind]["params"]:
		if d.name == p_name:
			return d
	return {}


# A node's parameter, its kind's default when the file has none.
static func param(p_node: Dictionary, p_name: String) -> Variant:
	var params: Dictionary = p_node.get("params", {})
	if params.has(p_name):
		return params[p_name]
	return param_def(p_node.kind, p_name).get("default")


# The climate's warp fields a model's Climate warp reads: its "climate_warp", else (a file from before it) the
# settings its first warp node carried, else CLIMATE_WARP.
static func climate_warp(p_model: Dictionary) -> Dictionary:
	var w: Dictionary = CLIMATE_WARP.duplicate()
	if p_model.has("climate_warp"):
		w.merge(p_model.climate_warp, true)
		return w
	for id in p_model.get("nodes", {}):
		var node: Dictionary = p_model.nodes[id]
		if node.kind == "warp" and node.get("params", {}).has("k"):
			for key in w:
				w[key] = node.params.get(key, w[key])
			break
	return w


static func rows(p_node: Dictionary) -> Array:
	return p_node.get("params", {}).get("rows", [])


static func new_row(p_vol: float = 1.0) -> Dictionary:
	return {"vol": p_vol, "mute": false, "solo": false, "centre": true}


static func new_node(p_kind: String, p_pos: Vector2) -> Dictionary:
	var params := {}
	for d in KINDS[p_kind]["params"]:
		params[d.name] = d.default.duplicate(true) if d.default is Array else d.default
	return {"kind": p_kind, "pos": [p_pos.x, p_pos.y], "params": params}


# A new node in a biome model (its id: "n" + the model's next_id), its kind's defaults merged with p_params.
static func add(p_model: Dictionary, p_kind: String, p_pos: Vector2, p_params: Dictionary = {}) -> String:
	var id := "n%d" % int(p_model.next_id)
	p_model.next_id = int(p_model.next_id) + 1
	var node := new_node(p_kind, p_pos)
	node.params.merge(p_params, true)
	p_model.nodes[id] = node
	return id


# Nodes out of a biome model, with their links; a Mix row a deleted node played goes too (no empty row left behind).
static func cut_nodes(p_model: Dictionary, p_ids: Array) -> void:
	for id in p_ids:
		if not p_model.nodes.has(id):
			continue
		while true:
			var into_mix: Array = p_model.links.filter(func(l): return l.from == id and p_model.nodes[l.to].kind == "mix" and not l.to in p_ids)
			if into_mix.is_empty():
				break
			_drop_row(p_model, into_mix[0].to, int(into_mix[0].to_port))
		p_model.nodes.erase(id)
		p_model.links = p_model.links.filter(func(l): return l.from != id and l.to != id)


# A Mix row out (its x), with what played it if nothing else uses it: its shape, its note, a warp field only it read
# (walking up from the row, every node left with no output link goes).
static func cut_row(p_model: Dictionary, p_mix: String, p_row: int) -> void:
	var up := []
	for l in p_model.links:
		if l.to == p_mix and int(l.to_port) == p_row:
			up.append(l.from)
	_drop_row(p_model, p_mix, p_row)
	while not up.is_empty():
		var id: String = up.pop_back()
		if not p_model.nodes.has(id) or p_model.nodes[id].kind == "output":
			continue
		if p_model.links.any(func(l): return l.from == id):
			continue
		for l in p_model.links:
			if l.to == id:
				up.append(l.from)
		p_model.nodes.erase(id)
		p_model.links = p_model.links.filter(func(l): return l.to != id)


# Row p_row out of Mix p_mix: its link goes, the rows under it move up a port.
static func _drop_row(p_model: Dictionary, p_mix: String, p_row: int) -> void:
	var r: Array = rows(p_model.nodes[p_mix]).duplicate(true)
	if p_row < r.size():
		r.remove_at(p_row)
	p_model.nodes[p_mix].params.rows = r
	p_model.links = p_model.links.filter(func(l): return not (l.to == p_mix and int(l.to_port) == p_row))
	for l in p_model.links:
		if l.to == p_mix and int(l.to_port) > p_row:
			l.to_port = int(l.to_port) - 1


# The id of the model's first node of a kind, or "".
static func first_of(p_model: Dictionary, p_kind: String) -> String:
	for id in p_model.nodes:
		if p_model.nodes[id].kind == p_kind:
			return id
	return ""


# A Height curve that remaps nothing over [-span, span].
static func identity_curve(p_span: float) -> Array:
	return [[-p_span, -p_span], [0.0, 0.0], [p_span, p_span]]


# The parameters an enum's option sets with it (its definition's "option_params"): a Shape preset's offset.
static func option_params(p_kind: String, p_name: String, p_option: String) -> Dictionary:
	var sets := {}
	var per: Dictionary = param_def(p_kind, p_name).get("option_params", {})
	for key in per:
		sets[key] = per[key][p_option]
	return sets


static func preset_offset(p_preset: String) -> float:
	return option_params("shape", "preset", p_preset).offset


# phi(n) of a shape's coefficients with the engine's smoothed |n| (shader_lib's GLSL shape(), from the same E), and
# its derivative.
static func phi(p_c: Array, p_n: float) -> float:
	return p_c[0] + p_c[1] * p_n + p_c[2] * (sqrt(p_n * p_n + SMOOTH_ABS_E * SMOOTH_ABS_E) - SMOOTH_ABS_E) + p_c[3] * p_n * p_n


static func dphi(p_c: Array, p_n: float) -> float:
	return p_c[1] + p_c[2] * p_n / sqrt(p_n * p_n + SMOOTH_ABS_E * SMOOTH_ABS_E) + 2.0 * p_c[3] * p_n


# E[phi(n)] over the noise (the engine inserts a band without it).
static func phi_mean(p_c: Array) -> float:
	return p_c[0] + p_c[2] * NOISE_ABS_MEAN + p_c[3] * NOISE_SQ_MEAN


# A Shape node's [c0, c1, c2, c3] on (1, n, |n|, n^2), as Climate's parser turns shape, scale and offset into them.
static func shape_coeffs(p_node: Dictionary) -> Array:
	var preset: String = param(p_node, "preset")
	if preset == "coeffs":
		return [param(p_node, "c0"), param(p_node, "c1"), param(p_node, "c2"), param(p_node, "c3")]
	var a: float = param(p_node, "scale")
	var o: float = param(p_node, "offset")
	match preset:
		"billow":
			return [o - 1.0, 0.0, 2.0 * absf(a), 0.0]
		"ridged":
			return [o * o, 0.0, -2.0 * o * absf(a), a * a]
		"square":
			return [o, 0.0, 0.0, a * a]
	return [o, a, 0.0, 0.0]
