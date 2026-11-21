extends RefCounted
## What the engine says, once (ClimateReportNode's static bindings): the game's world and its engine, the biome graph's
## node kinds (Climate::graph_kinds: their inputs and parameters, what the mixer builds its nodes from), the template
## biome, a biome's debug colour, a shape's phi, where a biome is whole. Nothing here computes the terrain.

# The game's world, and the master variables every note and field is read at: the edited world's engine (use_world),
# world.json's until a world is set.
const WORLD := "res://climate/world.json"
static var voxel_m: float
static var lods: int
# Where a biome is whole on a field (within core of its span: the chart's filled box), the engine's kernel.
static var core := 0.7
# The node kinds in the format's order, and by name: {kind, inputs, params: [{name, type, required, default, options}]}.
static var kinds: Array = []
static var _by_kind := {}


static func _static_init() -> void:
	var world = JSON.parse_string(FileAccess.get_file_as_string(WORLD))
	use_world(world if world is Dictionary else {})
	if not ClassDB.class_exists("ClimateReportNode"):
		push_error("biome mixer: the engine's extension isn't loaded (ClimateReportNode): run the import, then the mixer.")
		return
	core = float(ClassDB.class_call_static("ClimateReportNode", "kernel_core"))
	kinds = ClassDB.class_call_static("ClimateReportNode", "graph_kinds")
	for k in kinds:
		_by_kind[k.kind] = k


# The master variables of the world being edited (its engine section, else WorldFile's defaults): a new biome's
# template and the fields panel's octaves read them.
static func use_world(p_world: Dictionary) -> void:
	var engine := section(p_world, "engine", String(p_world.get("path", WORLD)))
	voxel_m = float(engine.get("voxel_m", 0.5))
	lods = int(engine.get("lods", 15))


# The octaves a note or a field band may sit on (k: 2^k voxels), from the cascade's finest (3) to the top note of the
# LOD count (lods + 4: 26 km at 5 cm and 15 LODs), and the one the engine reads a wavelength of p_m metres as.
static func octave_range() -> Vector2i:
	return Vector2i(3, lods + 4)


static func octave_of(p_m: float) -> int:
	var r := octave_range()
	return clampi(roundi(log(maxf(p_m, 1e-6) / voxel_m) / log(2.0)), r.x, r.y)


static func kind(p_kind: String) -> Dictionary:
	return _by_kind.get(p_kind, {})


static func param(p_kind: String, p_name: String) -> Dictionary:
	for p in kind(p_kind).get("params", []):
		if p.name == p_name:
			return p
	return {}


# A new biome named p_name: the engine's template (its world file, SDFCascade::template_path).
static func template_biome(p_name: String) -> Dictionary:
	return ClassDB.class_call_static("ClimateReportNode", "template_biome", voxel_m, p_name)


# A biome's debug colour: its own (#rrggbb), else its name's hash (the parser's).
static func debug_color(p_biome: Dictionary) -> Color:
	var hex := String(p_biome.get("debug_color", ""))
	return Color(hex) if hex.begins_with("#") else ClassDB.class_call_static("ClimateReportNode", "debug_color", String(p_biome.get("name", "")))


# A shape node's phi at p_ns, as a note's (its constant term doesn't count): {error, phi}.
static func shape_phi(p_node: Dictionary, p_ns: PackedFloat32Array) -> Dictionary:
	return ClassDB.class_call_static("ClimateReportNode", "shape", p_node, p_ns)


# p_world[p_key]: inline, or the file its string names (relative to the world file p_path), as WorldFile reads it.
static func section(p_world: Dictionary, p_key: String, p_path: String) -> Dictionary:
	var v = p_world.get(p_key, {})
	if v is String:
		v = JSON.parse_string(FileAccess.get_file_as_string(section_path(v, p_path)))
	return v if v is Dictionary else {}


static func section_path(p_name: String, p_path: String) -> String:
	return p_name if p_name.is_absolute_path() else p_path.get_base_dir().path_join(p_name)
