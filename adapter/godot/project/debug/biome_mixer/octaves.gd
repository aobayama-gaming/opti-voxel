extends RefCounted
## The game's world and the octaves its notes are on, as the engine says (ClimateReportNode.template_notes: the template
## biome's notes, Climate::template_graph): the mixer edits one note per octave from K_MIN to ground_top, each of the
## template's seed and amplitude rule. Nothing here computes the terrain: the previews are the engine's.

# The game's world, and the master variables every note and field is read at: its engine.
const WORLD := "res://climate/world.json"
static var voxel_m: float
static var lods: int
# The template's notes (Climate::TEMPLATE_*): its seed, its amplitude (steepness times the wavelength, capped), its
# finest octave (8 voxels: LOD 0's first) and its coarsest at this engine.
static var GROUND_SEED: int
static var STEEPNESS: float
static var MAX_AMP_M: float
static var K_MIN: int
static var _ground_top: int
# Where a biome is whole on a field (within core of its span: the chart's filled box), the engine's kernel.
static var core := 0.7
static var _octave_top: int


static func _static_init() -> void:
	var world = JSON.parse_string(FileAccess.get_file_as_string(WORLD))
	var engine := section(world if world is Dictionary else {}, "engine", WORLD)
	# WorldFile's defaults.
	voxel_m = float(engine.get("voxel_m", 0.5))
	lods = int(engine.get("lods", 15))
	var t: Dictionary = ClassDB.class_call_static("ClimateReportNode", "template_notes", voxel_m, lods) if ClassDB.class_exists("ClimateReportNode") else {}
	if t.is_empty():
		push_error("biome mixer: the engine's extension isn't loaded (ClimateReportNode): run the import, then the mixer.")
		return
	GROUND_SEED = int(t.seed)
	STEEPNESS = float(t.steepness)
	MAX_AMP_M = float(t.max_amp_m)
	K_MIN = int(t.k_min)
	_ground_top = int(t.k_top)
	_octave_top = int(t.octave_top)
	core = float(t.core)


# p_world[p_key]: inline, or the file its string names (relative to the world file p_path), as WorldFile reads it.
static func section(p_world: Dictionary, p_key: String, p_path: String) -> Dictionary:
	var v = p_world.get(p_key, {})
	if v is String:
		v = JSON.parse_string(FileAccess.get_file_as_string(section_path(v, p_path)))
	return v if v is Dictionary else {}


static func section_path(p_name: String, p_path: String) -> String:
	return p_name if p_name.is_absolute_path() else p_path.get_base_dir().path_join(p_name)


# The engine's coarsest octave of a note or a field (SDFCascade::top_octave), and the ground's (the template's).
static func k_top() -> int:
	return _octave_top


static func ground_top() -> int:
	return _ground_top


# Octave k's wavelength: 2^k voxels.
static func wavelength_m(p_k: int, p_voxel_m: float) -> float:
	return p_voxel_m * pow(2.0, p_k)


# The octave of a wavelength (log2 of it in voxels, rounded: ExactNoise::lattice_k): a table's wavelength_m, and a
# range's from_m and to_m ends.
static func lattice_k(p_m: float, p_voxel_m: float) -> int:
	return int(round(log(p_m / p_voxel_m) / log(2.0)))
