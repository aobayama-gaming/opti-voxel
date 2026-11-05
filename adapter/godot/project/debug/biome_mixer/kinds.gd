extends RefCounted
## How the mixer shows the engine's node kinds (EngineInfo.kinds: what a kind reads and takes is the engine's, here only
## its title, its tooltip and its sliders' ranges). checks.gd checks every engine kind has an entry.

# The plots' background (the phi graph, the curve editor, the chart) and the links' colour.
const PLOT_BG := Color(0.1, 0.11, 0.13)
const LINK := Color(0.55, 0.8, 1.0)

# A slider's range when a parameter has none below: [min, max, step] (the number box takes any value).
const RANGE := [0.0, 10.0, 0.01]

# kind -> {title, hint, ranges: {param: [min, max, step]}}.
const KINDS := {
	"note": {
		"title": "Note",
		"hint": "A 3D noise of one octave: wavelength_m (an octave: 2^k voxels from 8, the lattice the cascade inserts it on; the menu lists them), amp_m. stretch_y squeezes it along the vertical. Its warp input (a Warp node) reads it displaced. A note the LOD count doesn't have plays nothing. A note is one node whatever reads it: the same note in several biomes is read once.",
		"ranges": {"wavelength_m": [0.4, 30000.0, 0.1], "amp_m": [0.0, 5000.0, 0.1]},
	},
	"warp": {
		"title": "Warp",
		"hint": "A note's warp input: the note is read displaced by amount_m times (x, y, z): in read with its notes' seed, seed + 1 and seed + 2, or instead the ports x, y and z, a node each. Any node, strictly coarser than the note it warps (the engine says which isn't).",
		"ranges": {"amount_m": [0.0, 2000.0, 0.1]},
	},
	"shape": {
		"title": "Shape",
		"hint": "phi of its input: plain a n + o, billow 2|a n| - 1 + o, ridged (o - |a n|)^2, square (a n)^2 + o, or coeffs on (1, n, |n|, n^2). On a note it shapes the noise without its mean (an offset doesn't count there, but ridged's); on any other node its value, the offset counting.",
		"ranges": {"scale": [0.0, 4.0, 0.01], "offset": [-2.0, 2.0, 0.01]},
	},
	"multifractal": {
		"title": "Multifractal",
		"hint": "Weights its input (in) by its parent's value, any node (a coarser one is Musgrave's chain): 0 at or under low_m, 1 at or over high_m, linear between. low_m above high_m inverts it: detail in the parent's valleys.",
		"ranges": {"low_m": [-2000.0, 2000.0, 0.1], "high_m": [-2000.0, 2000.0, 0.1]},
	},
	"mix": {
		"title": "Mix",
		"hint": "The rows' sum: each row any node at volume x. M mutes, S solos a row in the view only (the file keeps x).",
		"ranges": {"x": [0.0, 4.0, 0.01]},
	},
	"up": {"title": "Up", "hint": "The sample's height above the base, in metres: a cave is a smooth min of the mix and up minus the cave's notes.", "ranges": {}},
	"constant": {"title": "Constant", "hint": "A height in metres.", "ranges": {"value_m": [-2000.0, 2000.0, 0.1]}},
	"field": {"title": "Field", "hint": "A climate field's value at the sample (temperature, humidity, ...).", "ranges": {}},
	"add": {"title": "Add", "hint": "a + b.", "ranges": {}},
	"subtract": {"title": "Subtract", "hint": "a - b.", "ranges": {}},
	"multiply": {"title": "Multiply", "hint": "a x b.", "ranges": {}},
	"min": {"title": "Min", "hint": "min(a, b).", "ranges": {}},
	"max": {"title": "Max", "hint": "max(a, b).", "ranges": {}},
	"smin": {"title": "Smooth min", "hint": "min(a, b) rounded over k_m metres.", "ranges": {"k_m": [0.0, 500.0, 0.1]}},
	"smax": {"title": "Smooth max", "hint": "max(a, b) rounded over k_m metres.", "ranges": {"k_m": [0.0, 500.0, 0.1]}},
	"curve": {
		"title": "Height curve",
		"hint": "Remaps a height (metres in, metres out) through its points, flat past the ends. Drag points, double-click to add, right-click to remove.",
		"ranges": {},
	},
	"output": {"title": "Output", "hint": "The biome's ground: its height, and its surface colours on gentle and steep slopes.", "ranges": {}},
}


static func title(p_kind: String) -> String:
	return KINDS.get(p_kind, {}).get("title", p_kind)


static func hint(p_kind: String) -> String:
	return KINDS.get(p_kind, {}).get("hint", "")


static func range_of(p_kind: String, p_name: String) -> Array:
	return KINDS.get(p_kind, {}).get("ranges", {}).get(p_name, RANGE)


# A Height curve that remaps nothing over [-span, span].
static func identity_curve(p_span: float) -> Array:
	return [[-p_span, -p_span], [0.0, 0.0], [p_span, p_span]]
