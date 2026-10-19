## Named argument sets shared by bench.gd and player.gd: `--preset=NAME` puts the preset's arguments
## before the command line's, so an explicit argument overrides the preset's.

const PRESETS := {
	# The perf target: a 5 cm forest at lod_error 2. The camera args place the bench camera 5 m above
	# the ground at the main_scene spawn, looking along the fly path (+X); the player ignores them.
	"forest5cm": ["--terrain=archetypes", "--trees=4000", "--tree_area=316", "--voxel=0.05", "--lod_error=2",
		"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90", "--fly_dist=150", "--fly_seconds=6"],
}


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
