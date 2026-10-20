## Named argument sets shared by bench.gd and player.gd: `--preset=NAME` puts the preset's arguments
## before the command line's, so an explicit argument overrides the preset's.

const PRESETS := {
	# The perf target: a 5 cm forest at lod_error 2. The camera args place the bench camera 5 m above
	# the ground at the main_scene spawn, looking along the fly path (+X); the player ignores them and
	# starts --altitude metres above the ground (the bench ignores that one).
	"forest5cm": ["--trees=4000", "--tree_area=316", "--voxel=0.05", "--lod_error=2", "--altitude=20",
		"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90", "--fly_dist=150", "--fly_seconds=6"],
	# Biome features (lava lakes, floating islands) within 2 km of the scene centre. The bench camera looks at a
	# lava lake from 60 m up, two floating islands behind it.
	"biomes": ["--biomes=2", "--cam_pos=-560,2650,850", "--above_ground=60", "--yaw=45", "--pitch=-18"],
}


# --grade=0: no colour grading (the scene's LUT): adjustments off on the WorldEnvironment under p_scene_root.
static func ungrade(p_scene_root: Node) -> void:
	var world_env := p_scene_root.get_node_or_null("WorldEnvironment") as WorldEnvironment
	if world_env:
		world_env.environment.adjustment_enabled = false


static func user_args() -> PackedStringArray:
	var given := OS.get_cmdline_user_args()
	var out := PackedStringArray()
	for a in given:
		if a.begins_with("--preset="):
			var name := a.trim_prefix("--preset=")
			if PRESETS.has(name):
				out.append_array(PackedStringArray(PRESETS[name]))
			else:
				push_error("unknown --preset=%s (expected one of %s)" % [name, PRESETS.keys()])
	out.append_array(given)
	return out
