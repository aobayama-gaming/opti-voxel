## Named argument sets shared by bench.gd and player.gd: `--preset=NAME` puts the preset's arguments
## before the command line's, so an explicit argument overrides the preset's.

const PRESETS := {
	# The perf target: a 5 cm forest at lod_error 2. The camera args place the bench camera 5 m above
	# the ground at the main_scene spawn, looking along the fly path (+X); the player ignores them and
	# starts --altitude metres above the ground (the bench ignores that one).
	"forest5cm": ["--trees=4000", "--tree_area=316", "--voxel=0.05", "--lod_error=2", "--altitude=20",
		"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90", "--fly_dist=150", "--fly_seconds=6"],
	# forest5cm ten times wider in area at the same density (one tree per 25 m^2): 40,000 trees on 1 km.
	"forest5cm_x10": ["--trees=40000", "--tree_area=1000", "--voxel=0.05", "--lod_error=2", "--altitude=20",
		"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90", "--fly_dist=150", "--fly_seconds=6"],
	# The streamed forest at forest5cm's density on 5 x 5 km (1M trees) and 32 x 32 km (41M): only the keys the view asks
	# for are computed, so the same configure time and memory as forest5cm.
	"forest5cm_1m": ["--trees=1000000", "--tree_area=5000", "--voxel=0.05", "--lod_error=2", "--altitude=20",
		"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90", "--fly_dist=150", "--fly_seconds=6"],
	"forest5cm_41m": ["--trees=41000000", "--tree_area=32000", "--voxel=0.05", "--lod_error=2", "--altitude=20",
		"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90", "--fly_dist=150", "--fly_seconds=6"],
	# The look for videos (player only): forest5cm_1m on the world climate, at error 2 front and back, FSR 2 from 50% then 2x2 pixel blocks, the flashlight, no mesher budget
	# and the frame-time overlay, no SSAO and the sky's ambient light x1.5 (lighter under the trees; the LUT is kept). Run
	# with `--render-thread safe` (FSR 2).
	"video": ["--trees=1000000", "--tree_area=5000", "--voxel=0.05", "--lod_error=2", "--altitude=20",
		"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90", "--lods=25",
		"--climate=res://climate/world.json", "--lod_error_back=2", "--upscale=fsr2", "--render_scale=0.5",
		"--pixelate=2", "--flashlight=1", "--mesher_budget=0", "--overlay=video", "--ssao=0", "--ambient=1.5"],
	# Biome features (lava lakes, floating islands) within 2 km of the scene centre. The bench camera looks at a
	# lava lake from 60 m up, two floating islands behind it.
	"biomes": ["--biomes=2", "--cam_pos=-560,2650,850", "--above_ground=60", "--yaw=45", "--pitch=-18"],
}


# The Environment of the WorldEnvironment under p_scene_root, null without one.
static func environment(p_scene_root: Node) -> Environment:
	var world_env := p_scene_root.get_node_or_null("WorldEnvironment") as WorldEnvironment
	return world_env.environment if world_env else null


# --grade=0: no colour grading (the scene's LUT): adjustments off.
static func ungrade(p_scene_root: Node) -> void:
	var env := environment(p_scene_root)
	if env:
		env.adjustment_enabled = false


# The camera's clip planes, called every frame by the player and the bench: near = altitude / 1000 (altitude = |SDF|
# at the camera, at least p_base_near, the scene's), far = near * 2^21 capped at the discoverer's load range, so the
# ground stays in view from any height. Godot's float projection loses its far plane past far/near ~2^24 (the light
# culler then ERRORs every frame), so far/near stays at 2^21.
static func clip_planes(p_camera: Camera3D, p_terrain: Node, p_base_near: float) -> void:
	var altitude: float = absf(p_terrain.get_sdf_at(p_camera.global_position))
	p_camera.near = maxf(p_base_near, altitude / 1000.0)
	p_camera.far = minf(p_terrain.get_load_radius(), p_camera.near * 2097152.0)


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
