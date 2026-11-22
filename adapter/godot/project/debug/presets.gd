## Named argument sets shared by bench.gd and player.gd: `--preset=NAME` puts the preset's arguments
## before the command line's, so an explicit argument overrides the preset's.

const FOREST5CM := ["--world=res://worlds/forest5cm.world.json", "--lod_error=2", "--altitude=20",
	"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90", "--fly_dist=150", "--fly_seconds=6"]

const PRESETS := {
	# The perf target: a 5 cm forest at lod_error 2, one tree per 25 m^2 over the whole ground (streamed: only the keys the
	# view asks for are computed). The camera args place the bench camera 5 m above the ground at the main_scene spawn,
	# looking along the fly path (+X); the player ignores them and starts --altitude metres above the ground (the bench
	# ignores that one).
	"forest5cm": FOREST5CM,
	# The 1M viewer's name from when the forest stood on a 5 km square: the same forest now.
	"forest5cm_1m": FOREST5CM,
	# The look for videos (player only): the 1M forest of the climate's world, at error 2 front and back, FSR 2 from 50% then 2x2 pixel blocks, the flashlight, no mesher budget
	# and the frame-time overlay, no SSAO and the sky's ambient light x1.5 (lighter under the trees; the LUT is kept). Run
	# with `--render-thread safe` (FSR 2).
	"video": ["--world=res://climate/world.json", "--lod_error=2", "--altitude=20",
		"--cam_pos=-52,320,-95", "--above_ground=5", "--pitch=-5", "--yaw=-90",
		"--lod_error_back=2", "--upscale=fsr2", "--render_scale=0.5",
		"--pixelate=2", "--flashlight=1", "--mesher_budget=0", "--overlay=video", "--ssao=0", "--ambient=1.5"],
	# Biome features (lava lakes, floating islands) within 2 km of the scene centre (the world's feature_region). The bench camera looks at a
	# lava lake from 60 m up, two floating islands behind it.
	"biomes": ["--world=res://worlds/biomes.world.json", "--cam_pos=-560,2650,850", "--above_ground=60", "--yaw=45", "--pitch=-18"],
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


# The command line's VoxelTerrain flags (docs/testing.md), one parser for bench.gd and player.gd: flag -> property.
const FLOAT_FLAGS := {"lod_error": "lod_pixel_error", "lod_error_back": "lod_pixel_error_back",
	"lod_view_margin": "lod_view_margin_deg", "lod_view_memory": "lod_view_memory_m", "mesher_budget": "direct_gpu_budget_ms",
	"skirt": "skirt_depth", "compact_extent": "direct_compact_max_extent", "rebase_m": "rebase_distance_m"}
const INT_FLAGS := {"view": "debug_view", "batch": "direct_batch_size", "pages": "direct_pages_per_batch",
	"cap": "direct_page_capacity", "batches_per_frame": "direct_batches_per_frame", "max_in_flight": "direct_max_in_flight"}
# 0 off, anything else on.
const BOOL_FLAGS := {"lod_far_limit": "lod_far_limit", "stream": "archetype_stream", "walk_gpu": "walk_gpu",
	"compact": "direct_compaction", "destroy_verify": "destroy_verify", "walk_verify": "walk_verify",
	"discovery_compare": "discovery_compare"}
# --terrain=NAME: a world by name (the shapes gallery's), as --world= would name it.
const TERRAINS := {"shapes": "res://worlds/shapes.world.json"}
const DISCOVERERS := {"cpu": 0, "gpu": 1}


# Sets p_terrain from p_args' flags (the tables above, --world=, --terrain=, --discoverer=, --lod_errors=, --origin=;
# --grade=0 ungrades p_scene_root), before the terrain enters the tree: the world checks the values' ranges when it
# starts (World::Settings::validate, the terrain's error). A later flag overrides an earlier one (a preset's). False (an
# error printed) on a name no setting takes.
static func apply_args(p_terrain: Node, p_scene_root: Node, p_args: PackedStringArray) -> bool:
	var args := {}
	for a: String in p_args:
		if a.begins_with("--"):
			var kv: PackedStringArray = a.substr(2).split("=", true, 1)
			args[kv[0]] = kv[1] if kv.size() > 1 else "true"
	for key: String in args:
		var value: String = args[key]
		if FLOAT_FLAGS.has(key):
			p_terrain.set(FLOAT_FLAGS[key], value.to_float())
		elif INT_FLAGS.has(key):
			p_terrain.set(INT_FLAGS[key], value.to_int())
		elif BOOL_FLAGS.has(key):
			p_terrain.set(BOOL_FLAGS[key], value != "0")
	if args.has("world"):
		p_terrain.world_path = args["world"]
	if args.has("terrain"):
		if not TERRAINS.has(args["terrain"]):
			push_error("unknown --terrain=%s (expected shapes; any world: --world=PATH)" % args["terrain"])
			return false
		p_terrain.world_path = TERRAINS[args["terrain"]]
	if args.has("discoverer"):
		if not DISCOVERERS.has(args["discoverer"]):
			push_error("unknown --discoverer=%s (expected cpu or gpu)" % args["discoverer"])
			return false
		p_terrain.discoverer_mode = DISCOVERERS[args["discoverer"]]
	if args.has("lod_errors"):
		var errors: PackedStringArray = String(args["lod_errors"]).split(",")
		p_terrain.lod_pixel_errors = PackedFloat32Array(Array(errors).map(func(e): return e.to_float()))
	if args.has("origin"):
		var o: PackedStringArray = String(args["origin"]).split(",")
		if o.size() != 3:
			push_error("--origin=%s: expected X,Y,Z in LOD0 voxels" % args["origin"])
			return false
		p_terrain.world_origin_x = o[0].to_int()
		p_terrain.world_origin_y = o[1].to_int()
		p_terrain.world_origin_z = o[2].to_int()
	if args.get("grade", "1") == "0":
		ungrade(p_scene_root)
	return true


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
