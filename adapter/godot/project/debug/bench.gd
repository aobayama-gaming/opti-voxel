extends Node3D
## Scripted benchmark / debug driver. Run from the CLI, e.g.:
##   godot --path project --resolution 1280x720 res://debug/bench.tscn -- \
##       --mode=legacy --scenario=load --out=/tmp/stats.json --shot=loaded:/tmp/a.png
##
## Args (all optional):
##   --mode=legacy|direct         pipeline (default legacy)
##   --preset=NAME                a named argument set (debug/presets.gd; forest5cm = the perf target),
##                                explicit arguments override it
##   --scenario=load|fly|turn|turn_walk|edit|dig|rise|approach load: load + steady phase. fly: load, fly out and back, settle, steady.
##                                turn: load, --turns=N full turns of the camera's yaw (default 1), --turn_seconds
##                                each (default 8), settle, steady
##                                turn_walk: turn, then walk --walk=M metres (default 300) straight ahead,
##                                level, in --walk_seconds (default 6), settle, steady
##                                edit: load, then N instance edits (add / move / remove) then settle, steady
##                                dig: load, then K dig holes along the view, a scripted drag, settle, steady
##                                rise: load, then one phase per --altitudes step (rise_<i>): the camera at that
##                                height above the terrain under the start position (same view), until loaded
##                                approach: load at a far pose (--approach_from above and --approach_out behind the start,
##                                against the fly direction, looking at the start), fly straight to the start at constant
##                                speed in --approach_seconds without waiting for loads (unlike rise), then the start's
##                                pitch / yaw, settle, steady
##   --approach_from=M            approach scenario: metres above the start pose (default 1500)
##   --approach_out=M             approach scenario: metres from the start along -X, the fly's -direction (default 2000)
##   --approach_seconds=S         approach scenario: wall-clock duration of the flight (default 20)
##   --approach_shots=DIR         approach scenario: a screenshot every --approach_shot_every seconds (default 1),
##                                DIR/approach_<i>_<S>s.png (relative to the run dir), in the --view of the run: run it
##                                twice (--view=0, --view=2) for shaded and LOD sheets, as rise does
##   --altitudes=M,M,...          rise scenario: metres above the terrain (default 20,50,100,200,400,800,1600,3200,6400,13000)
##   --rise_shots=DIR             rise scenario: a screenshot per step, DIR/rise_<i>_<M>m.png (relative to the run dir)
##   --edits=N                    edit scenario: number of edits (default 100)
##   --digs=K                     dig scenario: number of scripted right-click holes (default 8)
##   --dig_ahead=M                dig scenario: metres from the camera to the first hole (default 20)
##   --drag_s=S                   dig scenario: wall-clock seconds of scripted drag (tube re-placed every frame, default 2)
##   --steady=N                   frames measured after load completes (default 300)
##   --fly_seconds=S              wall-clock duration of the outbound leg (same again for return, default 10)
##   --fly_dist=M                 metres travelled on the outbound leg (default 1500)
##                                Movement is time-based (like a player), so a faster pipeline is not
##                                penalised by flying further per second.
##   --max_frames=N               hard cap; run is marked timed_out (default 20000)
##   --pitch=DEG                  camera pitch (default -20)
##   --yaw=DEG                    camera yaw (default 0, looking down -Z); player.gd hole marks print both
##   --shot=<when>:<path.png>     repeatable. when = loaded | steady_end | fly_end | approach_end | <frame number>
##   --out=<path.json>            stats output
##   --log=N                      VoxelTerrain.log_level
##   --profile=1                  per-frame profile to profile.json next to --out: main thread (frame time, each
##                                VoxelTerrain::_process section) and, per captured frame, every
##                                renderer pass's GPU and render-thread CPU time. Per-pass timestamps need Godot's
##                                visual profiler, which needs a debugger: run through `tools/bench.sh --profile=1`.
##   --trace=1                    engine trace from the terrain's creation to the end: trace.json next to --out (every
##                                thread's spans and hand-offs, the GPU's passes; ui.perfetto.dev) and the phases'
##                                start times in trace_phases.json; tools/engine_map.py <run> draws the map
##   --trace_max=N                the trace's event cap (default 2000000, ~120 B of JSON each), the rest dropped
##   --audit=1                   direct: after steady, classify every drawn quad slot of every page;
##                                writes per-chunk mesh digests to chunks.json next to --out
##   --validate=N                 direct: after steady, compare N chunks against the legacy mesher
##   --batch=N                    legacy batch size
##   --trim=1                     legacy: trim the zero-padded index buffers (pre-existing bug)
##   --dual_contouring=1          both meshers: DC-lite vertex placement (sharp UNION/SUBTRACT creases)
##   --batch=N --pages=N --cap=N  direct: batch size, pages per batch, page capacity (quads)
##   --batches_per_frame=N        direct: mesher batches dispatched per frame at most (default 4, at most 16)
##   --mesher_budget=MS           direct: estimated GPU ms of mesh batches per frame (default 0 = whole batches)
##   --max_in_flight=N            direct: mesher batches waiting for their directory at most (default 3 per batch per frame)
##   --voxel=M                    LOD 0 voxel size in metres (default 0.5)
##   --terrain=cascade|continents|shapes
##                                VoxelTerrain.terrain_mode (default cascade: the Hermite cascade terrain,
##                                SDFCascade; continents: its continents preset; shapes: its flat preset under the
##                                hard-CSG shapes gallery, SDFTable::Scene::SHAPES)
##   Instances (around x -50 z -150 m, on the ground):
##   --instances=N                scatter N rocks/pillars (default 0)
##   --instance_area=M            side of the scatter square, metres (default 200)
##   --instance_band=0            keep every instance at every LOD (no band rule; the reference)
##   --stream=0                   place the scatter and the forest whole at configure instead of streaming them
##   --stream_part_sides=N        streamed moment layers read the records while their cell is under N source cells,
##                                proxies above (default 1; 3 = the whole build's blobs)
##   --moments=0                  leave out the far field of the dropped instances (their moments' blobs)
##   --instance_erode=F           the band rule's erosion, cells per LOD (default 1): the skirt, the fastest a part shrinks
##   --instance_resolve=F         a part is kept while its half thickness holds F cells (default 1); it erodes just before its drop
##   --trees=N                    a forest of about N grown trees (default 0)
##   --tree_area=M                side of the forest square, metres (default 60)
##   --biomes=KM                  lava lakes and floating islands of the biome cells within KM km (default 0)
##   --lods=N                     VoxelTerrain.lod_numbers (default 15, max 31; the cascade's octaves reach k = N + 4).
##                                The clip planes follow the altitude at every LOD count, as in the player
##                                (Presets.clip_planes: far up to the discoverer's load range)
##   --lod_error=F                LOD screen-space error target (default 2.0; higher = coarser)
##   --lod_error_back=F           error outside the view cone, reached behind the camera (default 0 = lod_error)
##   --lod_view_margin=DEG        view cone = the frustum's diagonal half-FOV + DEG (default 20)
##   --lod_view_memory=M          a view keeps the front error until the camera is M metres from where it
##                                had it (default -1: where LOD 2 starts, 0 = off)
##   --lod_far_limit=0            scan past the camera's far plane too (VoxelTerrain.lod_far_limit, default 1)
##   --skirt=F                    skirt depth in voxels of each chunk's LOD (default 1.0, 0 = off)
##   --origin=X,Y,Z               integer world origin in LOD0 voxels (int64, snapped to the root chunk)
##   --compact=0|1                direct: page compaction (default 1)
##   --compact_extent=F           direct: max width of a re-packed page, in its smallest chunks (default 8)
##   --view=N                     direct: debug view (0 shaded 1 normals 2 lod 3 page 4 wire 5 chunk ... 8 faceted 9 material 12 biome)
##   --grade=0                    no colour grading (the scene's LUT, res://grade_lut.png, tools/make_lut.py)
##   --probe=PREFIX               direct: once loaded, PREFIX_shaded.png, then the bark seam probe views 10 and 11
##                                as PREFIX_10.png and PREFIX_11.png, raw (tools/bark_seams.sh)
##   --cam_pos=X,Y,Z              camera start position in metres (Godot space; default: the scene's)
##   --above_ground=M             move the camera vertically to M metres above the terrain under it (SDF
##                                distance, iterated; negative: below it). Default 50 on the cascade terrains,
##                                whose ground height varies by kilometres, unless --cam_pos gives the height
##   --camera_voxel=X,Y,Z         place the scene (camera, lights, --cam_pos) relative to this absolute
##                                LOD0 voxel instead of the world origin (origin must be a root-span multiple)
##   --discovery_cull=0|1         discoverer cull mode (0 none, 1 legacy corners; default 1)
##   --discovery_cache=0|1        discoverer: remember node verdicts across scans (default 1)
##   --discovery_verify=1         discoverer: recompute every cached verdict, ERROR on a mismatch
##   --destroy_verify=1           check every scan delta and incremental destroy check against the full recompute, ERROR on a mismatch
##   --discoverer=cpu|gpu         VoxelTerrain.discoverer_mode: where the cull test's corners are evaluated
##                                (default: gpu for the cascade terrain, cpu otherwise)
##   --discovery_compare=1        discoverer: diff every scan against a fresh CPU scan, ERROR on a difference
##   --walk_gpu=0|1               VoxelTerrain.walk_gpu: the streamed sources' site walks read the ground on the GPU (default 1)
##   --walk_batch=N               site walks: walk keys a drain task walks together on the GPU field (default 64)
##   --walk_verify=1              site walks: every batch of the GPU's readings read again on the CPU, ERROR on a difference
##   --discovery_audit=1          direct: after steady, rebuild the scan without culling and count the
##                                quads of every culled chunk on the GPU (any quad = a miss = a hole)
##   --depth_grab=1               after steady, read one frame's scene depth and its camera back to depth.f32 +
##                                depth.json next to --out (DepthGrab); with --audit=1, tools/occlusion_census.py
##                                <run> tells which loaded chunks that depth hides
##
## Checks after steady run in this order: discovery_audit, depth_grab, audit, validate.
##
## Render GPU/CPU time is only read during the "steady" phase: reading it forces a
## render-thread sync, which would distort the load/fly frame timings.

const MODES := {"legacy": 1, "direct": 2}
const Presets = preload("res://debug/presets.gd")

var args := {}
var shots := []  # [[when, path], ...]
var terrain
var camera: Camera3D
var base_near := 0.05  # the scene's near plane, the clip planes' floor
var start_pos: Vector3
var walk_from: Vector3  # turn_walk: where the walk starts
var start_rot: Vector3  # approach: the start pose's rotation (degrees), restored on arrival
var approach_from_pos: Vector3  # approach: where the flight starts
var approach_shot_next := 0  # approach: the next screenshot's index
var vp_rid: RID

var frame := 0
var phase := "load"  # load -> (fly_out -> fly_back -> settle) -> steady -> done
var edit_ids: Array[int] = []  # edit scenario: the live instance ids
var edits_done := 0
var digs_done := 0  # dig scenario
var drag_frames := 0
var drag_start_usec := 0
var script_drag_id := -1
var phase_frame := 0
var phase_start_usec := 0
var samples := {}  # phase -> {frame_ms: [], gpu_ms: [], cpu_render_ms: [], pending: []}
var phase_results := {}
var timed_out := false
var validation_report := {}
var audit_report := {}
var discovery_audit_report := {}
const DepthGrabScript := preload("res://debug/depth_grab.gd")
var depth_grab = null  # --depth_grab=1: a DepthGrab
var checks := []  # remaining post-steady checks
var profile_main := []  # --profile=1: one entry per frame
var profile_gpu := []  # --profile=1: one entry per captured renderer frame
var profile_last_process := {}
var profile_last_usec := 0
var trace_phases := []  # --trace=1: {phase, start_usec}, on the trace's clock


func _ready() -> void:
	_parse_args()
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0

	camera = $Camera3D
	camera.rotation_degrees.x = float(args.get("pitch", "-20"))
	camera.rotation_degrees.y = float(args.get("yaw", "0"))
	if args.has("cam_pos"):
		var c: PackedStringArray = args["cam_pos"].split(",")
		camera.global_position = Vector3(float(c[0]), float(c[1]), float(c[2]))
	start_pos = camera.global_position
	if args.has("depth_grab"):
		depth_grab = DepthGrabScript.new()
		if camera.compositor == null:
			camera.compositor = Compositor.new()
		var effects := camera.compositor.compositor_effects
		effects.append(depth_grab)
		camera.compositor.compositor_effects = effects

	vp_rid = get_viewport().get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(vp_rid, true)

	terrain = ClassDB.instantiate("VoxelTerrain")
	if terrain == null:
		# The extension failed to load (see the errors above): fail now instead of timing out.
		push_error("BENCH: VoxelTerrain is not available")
		set_process(false)
		get_tree().quit(1)
		return
	terrain.log_level = int(args.get("log", "1"))
	terrain.debug_gpu_profile = true
	if args.get("profile", "0") == "1":
		terrain.debug_gpu_profile_frames = true
		if EngineDebugger.has_profiler("visual"):
			EngineDebugger.profiler_enable("visual", true)
		else:
			push_warning("BENCH: --profile=1 without a debugger: the renderer only times the viewport (use bench.sh --profile=1)")
	if args.get("trace", "0") == "1":
		terrain.debug_trace_start(int(args.get("trace_max", "2000000")))
	var mode: String = args.get("mode", "legacy")
	if not MODES.has(mode):
		push_error("BENCH: unknown --mode=%s (expected legacy or direct)" % mode)
		set_process(false)
		get_tree().quit(1)
		return
	terrain.pipeline_mode = MODES[mode]
	if args.has("batch"):
		terrain.legacy_batch_size = int(args["batch"])
		terrain.direct_batch_size = int(args["batch"])
	if args.has("pages"):
		terrain.direct_pages_per_batch = int(args["pages"])
	if args.has("cap"):
		terrain.direct_page_capacity = int(args["cap"])
	if args.has("batches_per_frame"):
		terrain.direct_batches_per_frame = int(args["batches_per_frame"])
	if args.has("mesher_budget"):
		terrain.direct_gpu_budget_ms = float(args["mesher_budget"])
	if args.has("max_in_flight"):
		terrain.direct_max_in_flight = int(args["max_in_flight"])
	terrain.debug_view = int(args.get("view", "0"))
	if args.get("grade", "1") == "0":
		Presets.ungrade(self)
	terrain.legacy_trim_indices = args.get("trim", "0") == "1"
	terrain.dual_contouring = args.get("dual_contouring", "0") == "1"
	if args.has("voxel"):
		terrain.voxel_size = float(args["voxel"])
	if args.has("lods"):
		terrain.lod_numbers = int(args["lods"])
	var terrains := {"cascade": 1, "continents": 2, "shapes": 4}
	var terrain_arg: String = args.get("terrain", "cascade")
	if not terrains.has(terrain_arg):
		push_error("BENCH: unknown --terrain=%s (expected cascade, continents or shapes)" % terrain_arg)
		set_process(false)
		get_tree().quit(1)
		return
	if args.has("instances"):
		terrain.archetype_instances = int(args["instances"])
	if args.has("instance_area"):
		terrain.archetype_instance_area = float(args["instance_area"])
	if args.has("moments"):
		terrain.archetype_moments = args["moments"] != "0"
	if args.has("instance_band"):
		terrain.archetype_band = args["instance_band"] != "0"
	if args.has("stream"):
		terrain.archetype_stream = args["stream"] != "0"
	if args.has("stream_part_sides"):
		terrain.archetype_stream_part_sides = int(args["stream_part_sides"])
	if args.has("instance_erode"):
		terrain.archetype_erode = float(args["instance_erode"])
	if args.has("instance_resolve"):
		terrain.archetype_resolve = float(args["instance_resolve"])
	if args.has("trees"):
		terrain.archetype_trees = int(args["trees"])
	if args.has("tree_area"):
		terrain.archetype_tree_area = float(args["tree_area"])
	if args.has("biomes"):
		terrain.biome_radius = 1000.0 * float(args["biomes"])
	if args.has("climate"):
		terrain.climate_path = args["climate"]
	terrain.terrain_mode = terrains[terrain_arg]
	if args.has("compact"):
		terrain.direct_compaction = args["compact"] == "1"
	if args.has("compact_extent"):
		terrain.direct_compact_max_extent = float(args["compact_extent"])
	if args.has("lod_error"):
		terrain.lod_pixel_error = float(args["lod_error"])
	if args.has("lod_error_back"):
		terrain.lod_pixel_error_back = float(args["lod_error_back"])
	if args.has("lod_view_margin"):
		terrain.lod_view_margin_deg = float(args["lod_view_margin"])
	if args.has("lod_view_memory"):
		terrain.lod_view_memory_m = float(args["lod_view_memory"])
	if args.has("lod_far_limit"):
		terrain.lod_far_limit = args["lod_far_limit"] != "0"
	if args.has("skirt"):
		terrain.skirt_depth = float(args["skirt"])
	if args.has("discovery_cull"):
		terrain.discovery_cull_mode = int(args["discovery_cull"])
	if args.has("discovery_cache"):
		terrain.discovery_cache = args["discovery_cache"] == "1"
	terrain.discovery_verify = args.get("discovery_verify", "0") == "1"
	terrain.destroy_verify = args.get("destroy_verify", "0") == "1"
	if args.has("walk_gpu"):
		terrain.walk_gpu = args["walk_gpu"] == "1"
	terrain.walk_verify = args.get("walk_verify", "0") == "1"
	if args.has("walk_batch"):
		terrain.walk_batch = int(args["walk_batch"])
	if args.has("discoverer"):
		var discoverers := {"cpu": 0, "gpu": 1}
		if not discoverers.has(args["discoverer"]):
			push_error("BENCH: unknown --discoverer=%s (expected cpu or gpu)" % args["discoverer"])
			set_process(false)
			get_tree().quit(1)
			return
		terrain.discoverer_mode = discoverers[args["discoverer"]]
	terrain.discovery_compare = args.get("discovery_compare", "0") == "1"
	if args.has("origin"):
		var o: PackedStringArray = args["origin"].split(",")
		terrain.world_origin_x = o[0].to_int()
		terrain.world_origin_y = o[1].to_int()
		terrain.world_origin_z = o[2].to_int()
	if args.has("camera_voxel"):
		# The scene (camera, lights) is placed relative to this absolute voxel instead of the world
		# origin: the same world point seen from another origin. Origins must be multiples of the root
		# chunk span (no snapping), as in tools/origin_parity.sh.
		var cv: PackedStringArray = args["camera_voxel"].split(",")
		var vs: float = terrain.voxel_size
		var shift := Vector3(float(cv[0].to_int() - terrain.world_origin_x) * vs,
			float(cv[1].to_int() - terrain.world_origin_y) * vs, float(cv[2].to_int() - terrain.world_origin_z) * vs)
		for child in get_children():
			if child is Node3D:
				child.global_position += shift
		start_pos = camera.global_position
	add_child(terrain)
	var above_ground: String = args.get("above_ground", "" if terrain.terrain_mode == 4 or args.has("cam_pos") else "50")
	if above_ground != "":
		_place_above_ground(float(above_ground))  # the fly path starts there too
		start_pos = camera.global_position
	if args.get("scenario", "load") == "approach":
		_place_for_approach()
	base_near = camera.near
	Presets.clip_planes(camera, terrain, base_near)


	_begin_phase("load")
	print("BENCH: start mode=%s scenario=%s" % [args.get("mode", "legacy"), args.get("scenario", "load")])


func _parse_args() -> void:
	for a in Presets.user_args():
		if not a.begins_with("--"):
			continue
		var kv: PackedStringArray = a.substr(2).split("=", true, 1)
		var key := kv[0]
		var value := kv[1] if kv.size() > 1 else "true"
		if key == "shot":
			var parts := value.split(":", true, 1)
			shots.append([parts[0], parts[1]])
		else:
			args[key] = value


func _begin_phase(p_name: String) -> void:
	phase = p_name
	phase_frame = 0
	phase_start_usec = Time.get_ticks_usec()
	trace_phases.append({"phase": p_name, "start_usec": phase_start_usec})
	samples[p_name] = {"frame_ms": [], "gpu_ms": [], "cpu_render_ms": [], "pending": [],
		"draw_calls": [], "primitives": [], "draw_calls_view": [], "draw_calls_shadow": []}


func _end_phase() -> void:
	var s: Dictionary = samples[phase]
	var r := {
		"frames": phase_frame,
		"wall_ms": (Time.get_ticks_usec() - phase_start_usec) / 1000.0,
		"frame_ms": _summary(s["frame_ms"]),
		"frame_ms_slowest": _slowest(s["frame_ms"], 5),
		"render_gpu_ms": _summary(s["gpu_ms"]),
		"render_cpu_ms": _summary(s["cpu_render_ms"]),
		"pending_chunks": _summary(s["pending"]),
		"draw_calls": _summary(s["draw_calls"]),
		"primitives": _summary(s["primitives"]),
		"draw_calls_view": _summary(s["draw_calls_view"]),
		"draw_calls_shadow": _summary(s["draw_calls_shadow"]),
		"terrain": terrain.get_debug_stats(),
		"memory": _memory(),
	}
	phase_results[phase] = r
	print("BENCH: phase %s done: %d frames, %.0f ms, frame p50 %.2f p99 %.2f max %.2f ms" % [
		phase, phase_frame, r["wall_ms"], r["frame_ms"]["p50"], r["frame_ms"]["p99"], r["frame_ms"]["max"]])
	_take_shots(phase + "_end")


## Approach scenario: the camera loads at the far pose, looking at the start pose it will fly to (the direction from
## any point of the straight flight to its end is the same: the orientation stays).
func _place_for_approach() -> void:
	start_rot = camera.rotation_degrees
	approach_from_pos = start_pos + Vector3(-float(args.get("approach_out", "2000")), float(args.get("approach_from", "1500")), 0)
	camera.global_position = approach_from_pos
	camera.look_at(start_pos, Vector3.UP)
	print("BENCH: approach from %s to %s" % [approach_from_pos, start_pos])


## Approach scenario, one frame: the camera on the straight line at constant speed, a screenshot every
## --approach_shot_every seconds; true on arrival (the start pose's position and rotation set).
func _run_approach() -> bool:
	var elapsed := (Time.get_ticks_usec() - phase_start_usec) * 1e-6
	var t := elapsed / float(args.get("approach_seconds", "20"))
	camera.global_position = approach_from_pos.lerp(start_pos, minf(t, 1.0))
	var every := float(args.get("approach_shot_every", "1"))
	if args.has("approach_shots") and elapsed >= approach_shot_next * every:
		_save_shot("%s/approach_%02d_%ds.png" % [args["approach_shots"], approach_shot_next, int(approach_shot_next * every)])
		approach_shot_next += 1
	if t < 1.0:
		return false
	camera.global_position = start_pos
	camera.rotation_degrees = start_rot
	return true


## Raise or lower the camera to p_height metres above the terrain under it (SDF distance, iterated).
func _place_above_ground(p_height: float) -> void:
	for i in 32:
		var d: float = terrain.get_sdf_at(camera.global_position)
		if absf(d - p_height) < maxf(1.0, p_height * 0.05):
			break
		camera.global_position.y += p_height - d
	print("BENCH: camera at %s (%.1f m above the terrain)" % [camera.global_position, terrain.get_sdf_at(camera.global_position)])


func _altitudes() -> PackedFloat64Array:
	var out := PackedFloat64Array()
	for a in args.get("altitudes", "20,50,100,200,400,800,1600,3200,6400,13000").split(","):
		out.append(float(a))
	return out


## LOD0 voxel of a render-space position: the int64 world origin plus a small rounded offset (the origin
## never goes through a float).
func _voxel_of(pos: Vector3, vt: float) -> Array:
	return [terrain.world_origin_x + int(round(pos.x / vt)), terrain.world_origin_y + int(round(pos.y / vt)),
			terrain.world_origin_z + int(round(pos.z / vt))]


## Edit scenario: a few instance edits per frame in front of the camera (alternating add / move /
## remove). Anchors are LOD0 voxels = world origin + render position / voxel size.
func _run_edits() -> void:
	var total := int(args.get("edits", "100"))
	var vt: float = terrain.get_voxel_size()
	for k in range(4):
		if edits_done >= total:
			return
		var fwd := -camera.global_transform.basis.z
		var pos := camera.global_position + fwd * 30.0 \
			+ Vector3(randf_range(-25, 25), randf_range(-15, 15), randf_range(-25, 25))
		var vox := _voxel_of(pos, vt)
		if edits_done % 3 == 0 and edit_ids.size() > 0:
			var idx: int = edit_ids[randi() % edit_ids.size()]
			if edits_done % 6 == 0:
				terrain.remove_instance(idx)
				edit_ids.erase(idx)
			else:
				terrain.move_instance(idx, vox[0], vox[1], vox[2])
		else:
			var id: int = terrain.add_instance(0, vox[0], vox[1], vox[2],
				PackedFloat32Array(), PackedFloat32Array([4.0 / vt]), 3, 1.0 / vt)
			if id >= 0:
				edit_ids.append(id)
		edits_done += 1


# Dig scenario, the player's tools scripted (player.gd): --digs K smooth-subtract capsules (6 m long,
# 1.5 m radius) starting along the view, one per frame, then the left-click tube for --drag_s seconds: a
# 20 m SUBTRACT capsule from 0.5 m ahead of the camera whose direction sweeps 10 degrees either side
# (re-placed every frame, as when the player turns), then removed.
const CAPSULE_TYPE := 1  # SDFTable's library: 0 sphere, 1 capsule, 2 rock (build_table)


func _dig_capsule(vt: float, pos: Vector3, fwd: Vector3, length: float, radius: float, op: int, id: int) -> int:
	var up := Vector3.UP
	if absf(fwd.dot(up)) > 0.99:
		up = Vector3.RIGHT
	var right := fwd.cross(up).normalized()
	var u2 := right.cross(fwd).normalized()
	var rot := PackedFloat32Array([right.x, right.y, right.z, fwd.x, fwd.y, fwd.z, u2.x, u2.y, u2.z])
	var v := _voxel_of(pos, vt)
	if id < 0:
		var params := PackedFloat32Array([length / vt, radius / vt])
		return terrain.add_instance(CAPSULE_TYPE, v[0], v[1], v[2], rot, params, op, 0.3 / vt)
	terrain.move_instance(id, v[0], v[1], v[2], rot)
	return id


func _run_dig() -> void:
	var vt: float = terrain.get_voxel_size()
	var fwd := -camera.global_transform.basis.z
	var digs := int(args.get("digs", "8"))
	if digs_done < digs:
		_dig_capsule(vt, camera.global_position + fwd * (float(args.get("dig_ahead", "20")) + digs_done * 6.0), fwd, 6.0, 1.5, 3, -1)
		digs_done += 1
		drag_start_usec = Time.get_ticks_usec()
		return
	var t := float(Time.get_ticks_usec() - drag_start_usec) * 1e-6
	var dir := fwd.rotated(camera.global_transform.basis.y.normalized(), deg_to_rad(10.0) * sin(t * 3.0))
	script_drag_id = _dig_capsule(vt, camera.global_position + dir * 0.5, dir, 20.0, 1.0, 2, script_drag_id)
	drag_frames += 1


func _process(delta: float) -> void:
	frame += 1
	phase_frame += 1

	var s: Dictionary = samples[phase]
	s["frame_ms"].append(delta * 1000.0)
	if phase == "steady":
		s["gpu_ms"].append(RenderingServer.viewport_get_measured_render_time_gpu(vp_rid))
		s["cpu_render_ms"].append(RenderingServer.viewport_get_measured_render_time_cpu(vp_rid)
			+ RenderingServer.get_frame_setup_time_cpu())
	s["pending"].append(terrain.get_pending_chunk_count())
	s["draw_calls"].append(RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME))
	s["primitives"].append(RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME))
	# The totals mix the camera pass with the shadow pass (which draws pages behind the camera).
	# GPU Direct pages draw indirectly, so their primitive counts are page capacities, not real
	# triangles: draw calls per pass are the meaningful culling numbers.
	s["draw_calls_view"].append(RenderingServer.viewport_get_render_info(vp_rid,
		RenderingServer.VIEWPORT_RENDER_INFO_TYPE_VISIBLE, RenderingServer.VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME))
	s["draw_calls_shadow"].append(RenderingServer.viewport_get_render_info(vp_rid,
		RenderingServer.VIEWPORT_RENDER_INFO_TYPE_SHADOW, RenderingServer.VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME))

	if terrain.debug_gpu_profile_frames:
		_record_profile(delta)
	Presets.clip_planes(camera, terrain, base_near)

	_take_shots(str(frame))

	var steady := int(args.get("steady", "300"))
	var fly_usec := float(args.get("fly_seconds", "10")) * 1e6
	var leg := (Time.get_ticks_usec() - phase_start_usec) / fly_usec
	var fly_dist := float(args.get("fly_dist", "1500"))

	match phase:
		"load":
			if terrain.is_load_complete():
				_end_phase()
				_take_shots("loaded")
				if args.has("probe"):
					_take_probe(args["probe"])
				if args.get("scenario", "load") == "fly":
					_begin_phase("fly_out")
				elif args.get("scenario", "load") in ["turn", "turn_walk"]:
					_begin_phase("turn")
				elif args.get("scenario", "load") == "edit":
					seed(0x5EED)  # the same edits every run
					_begin_phase("edit")
				elif args.get("scenario", "load") == "dig":
					_begin_phase("dig")
				elif args.get("scenario", "load") == "rise":
					_begin_phase("rise_0")
				elif args.get("scenario", "load") == "approach":
					_begin_phase("approach")
				else:
					_begin_phase("steady")
		"dig":
			_run_dig()
			var digs := int(args.get("digs", "8"))
			var drag_us := int(float(args.get("drag_s", "2")) * 1e6)
			if digs_done >= digs and drag_frames > 0 and Time.get_ticks_usec() - drag_start_usec >= drag_us:
				if script_drag_id >= 0:
					terrain.remove_instance(script_drag_id)
					script_drag_id = -1
				_end_phase()
				_take_shots("dig_end")
				_begin_phase("settle")
		"edit":
			_run_edits()
			if edits_done >= int(args.get("edits", "100")):
				_end_phase()
				_take_shots("edit_end")
				_begin_phase("settle")
		"turn":
			var turns := float(args.get("turns", "1"))
			var turn := (Time.get_ticks_usec() - phase_start_usec) / (float(args.get("turn_seconds", "8")) * 1e6)
			camera.rotation_degrees.y = float(args.get("yaw", "0")) + 360.0 * minf(turn, turns)
			if turn >= turns:
				_end_phase()
				_take_shots("turn_end")
				walk_from = camera.global_position
				_begin_phase("walk" if args.get("scenario", "load") == "turn_walk" else "settle")
		"walk":
			var walked := (Time.get_ticks_usec() - phase_start_usec) / (float(args.get("walk_seconds", "6")) * 1e6)
			var ahead := Vector3.FORWARD.rotated(Vector3.UP, camera.global_rotation.y)  # level, whatever the pitch
			camera.global_position = walk_from + ahead * float(args.get("walk", "300")) * minf(walked, 1.0)
			if walked >= 1.0:
				_end_phase()
				_take_shots("walk_end")
				_begin_phase("settle")
		"approach":
			if _run_approach():
				_end_phase()  # its shots: approach_end
				_begin_phase("settle")
		"fly_out":
			camera.global_position = start_pos + Vector3(fly_dist, 0, 0) * minf(leg, 1.0)
			if leg >= 1.0:
				_end_phase()
				_begin_phase("fly_back")
		"fly_back":
			camera.global_position = start_pos + Vector3(fly_dist, 0, 0) * maxf(1.0 - leg, 0.0)
			if leg >= 1.0:
				camera.global_position = start_pos
				_end_phase()
				_take_shots("fly_end")
				_begin_phase("settle")
		var p when p.begins_with("rise_"):
			# The camera moves on the step's first frame, once the last step's screenshot is drawn; a step
			# ends loaded (a few frames in, so the scan is the new pose's).
			var step := int(p.trim_prefix("rise_"))
			var altitudes := _altitudes()
			if phase_frame == 1:
				camera.global_position = start_pos
				_place_above_ground(altitudes[step])
			elif phase_frame > 4 and terrain.is_load_complete():
				_end_phase()
				if args.has("rise_shots"):
					_save_shot("%s/rise_%d_%dm.png" % [args["rise_shots"], step, int(altitudes[step])])
				_begin_phase("rise_%d" % (step + 1) if step + 1 < altitudes.size() else "steady")
		"settle":
			if terrain.is_load_complete():
				_end_phase()
				_begin_phase("steady")
		"steady":
			if phase_frame >= steady:
				_end_phase()
				for c in ["discovery_audit", "depth_grab", "audit", "validate"]:
					if args.has(c):
						checks.append(c)
				_next_check()
				return
		"discovery_audit":
			var report: Dictionary = terrain.get_discovery_audit_report()
			if not report.is_empty():
				discovery_audit_report = report
				print("BENCH: discovery_audit ", JSON.stringify(report))
				_next_check()
				return
		"depth_grab":
			if depth_grab.write(args["out"].get_base_dir() if args.has("out") else "user://", camera, terrain):
				_next_check()
				return
		"audit":
			var report: Dictionary = terrain.get_audit_report()
			if not report.is_empty():
				audit_report = report
				var shown := report.duplicate()
				shown.erase("chunk_digests")
				print("BENCH: audit ", JSON.stringify(shown))
				_next_check()
				return
		"validate":
			var report: Dictionary = terrain.get_validation_report()
			if not report.is_empty():
				validation_report = report
				print("BENCH: validation ok=%s matched=%s/%s legacy_unmatched=%s direct_unmatched=%s flipped=%s max_pos=%s cells max_normal=%s deg" % [
					report.get("ok"), report.get("matched_tris"), report.get("legacy_tris"), report.get("legacy_unmatched_tris"),
					report.get("direct_unmatched_tris"), report.get("winding_flipped_tris"),
					report.get("max_position_error_cells"), report.get("max_normal_error_deg")])
				_next_check()
				return

	if frame >= int(args.get("max_frames", "20000")):
		timed_out = true
		_end_phase()
		_finish()


func _next_check() -> void:
	if checks.is_empty():
		_finish()
		return
	var c: String = checks.pop_front()
	_begin_phase(c)
	match c:
		"discovery_audit":
			terrain.debug_audit_discovery()
		"depth_grab":
			depth_grab.request()
		"audit":
			terrain.debug_audit_pages()
		"validate":
			terrain.debug_validate(int(args["validate"]), args["out"].get_base_dir() if args.has("out") else "")


func _take_shots(when: String) -> void:
	for shot in shots:
		if shot[0] == when:
			_save_shot(shot[1])


func _save_shot(p_path: String) -> void:
	await RenderingServer.frame_post_draw
	DirAccess.make_dir_recursive_absolute(p_path.get_base_dir())
	get_viewport().get_texture().get_image().save_png(p_path)
	print("BENCH: screenshot -> %s" % p_path)


## --probe=PREFIX: a shaded shot, then the bark seam probe views (the page shader's 10 and 11) with the environment
## made raw (linear tonemapper, no grade, glow or fog), so each pixel reads back as the bytes the shader wrote.
func _take_probe(p_prefix: String) -> void:
	await RenderingServer.frame_post_draw
	get_viewport().get_texture().get_image().save_png(p_prefix + "_shaded.png")
	var env: Environment = ($WorldEnvironment as WorldEnvironment).environment
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	env.tonemap_exposure = 1.0
	env.adjustment_enabled = false
	env.glow_enabled = false
	env.fog_enabled = false
	env.volumetric_fog_enabled = false
	for view in [10, 11]:
		terrain.debug_view = view
		for i in 3:
			await RenderingServer.frame_post_draw
		var path := "%s_%d.png" % [p_prefix, view]
		get_viewport().get_texture().get_image().save_png(path)
		print("BENCH: probe view %d -> %s" % [view, path])


## --profile=1: this frame's main-thread times (VoxelTerrain sections as the difference of their running totals)
## and the renderer frames captured since the last call, tagged with the current phase.
func _record_profile(delta: float) -> void:
	var totals: Dictionary = terrain.get_process_times_ms()
	var sections := {}
	for k in totals:
		sections[k] = totals[k] - profile_last_process.get(k, 0.0)
	profile_last_process = totals
	var now := Time.get_ticks_usec()
	var wall_ms := (now - profile_last_usec) / 1000.0 if profile_last_usec > 0 else delta * 1000.0
	profile_last_usec = now
	profile_main.append({
		"frame": frame,
		"engine_frame": Engine.get_frames_drawn(),
		"phase": phase,
		"frame_ms": delta * 1000.0,  # Godot's delta (may be smoothed)
		"wall_ms": wall_ms,  # wall clock since the previous frame's _process
		"terrain": sections,
		"pending": terrain.get_pending_chunk_count(),
		"draw_calls_view": samples[phase]["draw_calls_view"][-1],
		"draw_calls_shadow": samples[phase]["draw_calls_shadow"][-1],
	})
	for g in terrain.take_gpu_profile_frames():
		g["phase"] = phase
		profile_gpu.append(g)


func _finish() -> void:
	set_process(false)
	if terrain.debug_gpu_profile_frames and args.has("out"):
		var fp := FileAccess.open(args["out"].get_base_dir().path_join("profile.json"), FileAccess.WRITE)
		fp.store_string(JSON.stringify({"main": profile_main, "gpu": profile_gpu,
			"window": DisplayServer.window_get_size(), "phases": phase_results.keys()}))
		fp.close()
	if args.get("trace", "0") == "1" and args.has("out"):
		var dir: String = args["out"].get_base_dir()
		var written: Dictionary = terrain.debug_trace_write(dir.path_join("trace.json"))
		print("BENCH: trace: %s" % JSON.stringify(written))
		if int(written["dropped"]) > 0:
			push_warning("BENCH: the trace dropped %d events past --trace_max" % int(written["dropped"]))
		var ft := FileAccess.open(dir.path_join("trace_phases.json"), FileAccess.WRITE)
		ft.store_string(JSON.stringify({"phases": trace_phases, "end_usec": Time.get_ticks_usec()}))
		ft.close()
	# Per-chunk mesh digests of the page audit go to their own file (thousands of entries).
	if audit_report.has("chunk_digests"):
		if args.has("out"):
			var fd := FileAccess.open(args["out"].get_base_dir().path_join("chunks.json"), FileAccess.WRITE)
			fd.store_string(JSON.stringify(audit_report["chunk_digests"]))
			fd.close()
		audit_report.erase("chunk_digests")
	var final_terrain: Dictionary = terrain.get_debug_stats()
	var result := {
		"args": args,
		"timed_out": timed_out,
		"total_frames": frame,
		"phases": phase_results,
		"validation": validation_report,
		"audit": audit_report,
		"discovery_audit": discovery_audit_report,
		"final_terrain": final_terrain,
		# Mode, first / last scan with their waves (GPU ms, corners, dedup, readback), compare counts.
		"discoverer": final_terrain.get("discoverer", {}),
		"memory": _memory(),
	}
	if args.has("out"):
		var f := FileAccess.open(args["out"], FileAccess.WRITE)
		f.store_string(JSON.stringify(result, "  "))
		f.close()
		print("BENCH: wrote ", args["out"])
	print("BENCH: done timed_out=%s frames=%d" % [timed_out, frame])
	# Let pending screenshots finish, then free the terrain before quitting so teardown is exercised.
	await RenderingServer.frame_post_draw
	terrain.queue_free()
	await get_tree().process_frame
	get_tree().quit(1 if timed_out else 0)


func _memory() -> Dictionary:
	# procfs files report size 0, so FileAccess reads nothing; shell out instead (phase ends only).
	var rss_kb := -1
	var out := []
	OS.execute("grep", ["VmRSS", "/proc/%d/status" % OS.get_process_id()], out)
	if not out.is_empty():
		var parts: PackedStringArray = out[0].split(":")
		if parts.size() > 1:
			rss_kb = int(parts[1].strip_edges().split(" ")[0])
	return {
		"rss_mb": rss_kb / 1024.0,
		"static_mb": Performance.get_monitor(Performance.MEMORY_STATIC) / 1048576.0,
		"video_mem_mb": RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_VIDEO_MEM_USED) / 1048576.0,
		"buffer_mem_mb": RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_BUFFER_MEM_USED) / 1048576.0,
	}


# The p_count largest values as [index in the phase, value], largest first: tells startup frames from stutter.
static func _slowest(values: Array, p_count: int) -> Array:
	var order := range(values.size())
	order.sort_custom(func(a, b): return values[a] > values[b])
	var out := []
	for i in order.slice(0, p_count):
		out.append([i, values[i]])
	return out


static func _summary(values: Array) -> Dictionary:
	if values.is_empty():
		return {"n": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
	var v := values.duplicate()
	v.sort()
	var total := 0.0
	for x in v:
		total += x
	return {
		"n": v.size(),
		"mean": total / v.size(),
		"p50": v[int(0.50 * (v.size() - 1))],
		"p95": v[int(0.95 * (v.size() - 1))],
		"p99": v[int(0.99 * (v.size() - 1))],
		"max": v[v.size() - 1],
	}


