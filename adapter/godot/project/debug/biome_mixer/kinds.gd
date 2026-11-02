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
		"hint": "A 3D noise, one octave (wavelength_m) or every octave from from_m to to_m that the LOD count has. Its amplitude: steepness times each octave's wavelength, or amp_m (the coarsest) times persistence per finer octave, capped at max_amp_m. stretch_y squeezes it along the vertical. Notes with the same seed and stretch are read once, whichever biomes play them.",
		"ranges": {"wavelength_m": [1.0, 30000.0, 0.1], "from_m": [0.4, 30000.0, 0.1], "to_m": [0.4, 30000.0, 0.1], "steepness": [0.0, 0.5, 0.001],
			"amp_m": [0.0, 5000.0, 0.1], "persistence": [0.0, 2.0, 0.01], "max_amp_m": [0.0, 10000.0, 1.0]},
	},
	"warp": {
		"title": "Warp",
		"hint": "Reads its note displaced by amount_m metres times three climate fields (x, y, z). One read per note whatever the biomes: where biomes blend, the note is read at their weights' mix of displacements.",
		"ranges": {"amount_m": [0.0, 2000.0, 0.1]},
	},
	"shape": {
		"title": "Shape",
		"hint": "phi(n) on a note: plain a n + o, billow 2|a n| - 1 + o, ridged (o - |a n|)^2, square (a n)^2 + o, or coeffs on (1, n, |n|, n^2). A band goes in without its mean: an offset only counts in a multifractal (or ridged).",
		"ranges": {"scale": [0.0, 4.0, 0.01], "offset": [-2.0, 2.0, 0.01]},
	},
	"multifractal": {
		"title": "Multifractal",
		"hint": "Weights its note by clamp(gain (1 - c)), c from the next coarser multifractal note's signal: detail where the coarser one was high (Musgrave).",
		"ranges": {"gain": [0.0, 4.0, 0.01]},
	},
	"mix": {
		"title": "Mix",
		"hint": "The notes' sum: each row a note (through a warp, a shape and a multifractal, each optional) at volume x. A biome mixes its notes once. M mutes, S solos a row in the view only (the file keeps x).",
		"ranges": {"x": [0.0, 4.0, 0.01]},
	},
	"constant": {"title": "Constant", "hint": "A height in metres.", "ranges": {"value_m": [-2000.0, 2000.0, 0.1]}},
	"field": {"title": "Field", "hint": "A climate field's value at the sample (temperature, humidity, ...).", "ranges": {}},
	"add": {"title": "Add", "hint": "a + b, at the samples.", "ranges": {}},
	"subtract": {"title": "Subtract", "hint": "a - b, at the samples.", "ranges": {}},
	"multiply": {"title": "Multiply", "hint": "a x b, at the samples.", "ranges": {}},
	"min": {"title": "Min", "hint": "min(a, b), at the samples.", "ranges": {}},
	"max": {"title": "Max", "hint": "max(a, b), at the samples.", "ranges": {}},
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
