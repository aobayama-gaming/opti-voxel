extends Node3D
## Scripted benchmark / debug driver. Run from the CLI, e.g.:
##   godot --path project --resolution 1280x720 res://debug/bench.tscn -- \
##       --mode=legacy --scenario=load --out=/tmp/stats.json --shot=loaded:/tmp/a.png
##
## Args (all optional):
##   --mode=legacy|direct         pipeline (default legacy)
##   --scenario=load|fly          load: load + steady phase. fly: load, fly out and back, settle, steady.
##   --steady=N                   frames measured after load completes (default 300)
##   --fly_seconds=S              wall-clock duration of the outbound leg (same again for return, default 10)
##   --fly_dist=M                 metres travelled on the outbound leg (default 1500)
##                                Movement is time-based (like a player), so a faster pipeline is not
##                                penalised by flying further per second.
##   --max_frames=N               hard cap; run is marked timed_out (default 20000)
##   --pitch=DEG                  camera pitch (default -20)
##   --yaw=DEG                    camera yaw (default 0, looking down -Z); player.gd hole marks print both
##   --shot=<when>:<path.png>     repeatable. when = loaded | steady_end | fly_end | <frame number>
##   --out=<path.json>            stats output
##   --log=N                      VoxelTerrain.log_level
##   --audit=1                    direct: after steady, classify every drawn quad slot of every page;
##                                writes per-chunk mesh digests to chunks.json next to --out
##   --validate=N                 direct: after steady, compare N chunks against the legacy mesher
##   --batch=N                    legacy batch size
##   --trim=1                     legacy: trim the zero-padded index buffers (pre-existing bug)
##   --dual_contouring=1          both meshers: DC-lite vertex placement (sharp UNION/SUBTRACT creases)
##   --batch=N --pages=N --cap=N  direct: batch size, pages per batch, page capacity (quads)
##   --batches_per_frame=N        direct: mesher batches dispatched per frame at most (default 4, at most 16)
##   --max_in_flight=N            direct: mesher batches waiting for their directory at most (default 3 per batch per frame)
##   --voxel=M                    LOD 0 voxel size in metres (default 0.5)
##   --terrain=cascade|continents|archetypes|shapes
##                                VoxelTerrain.terrain_mode (default cascade: the Hermite cascade terrain,
##                                SDFCascade; continents: its preset; archetypes: the archetype IR demo, SDFArchetype;
##                                shapes: its hard-CSG gallery on a flat ground, SDFArchetype::shapes)
##   --lods=N                     VoxelTerrain.lod_numbers (default 15, max 31). Above 15 the far plane
##                                covers the discoverer's load range (the cascade's octaves reach k = N + 4)
##   --lod_error=F                LOD screen-space error target (default 2.0; higher = coarser)
##   --skirt=F                    skirt depth in voxels of each chunk's LOD (default 1.0, 0 = off)
##   --origin=X,Y,Z               integer world origin in LOD0 voxels (int64, snapped to the root chunk)
##   --compact=0|1                direct: page compaction (default 1)
##   --compact_extent=F           direct: max width of a re-packed page, in its smallest chunks (default 8)
##   --view=N                     direct: debug view (0 shaded 1 normals 2 lod 3 page 4 wire 5 chunk)
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
##   --discovery_audit=1          direct: after steady, rebuild the scan without culling and count the
##                                quads of every culled chunk on the GPU (any quad = a miss = a hole)
##
## Checks after steady run in this order: discovery_audit, audit, validate.
##
## Render GPU/CPU time is only read during the "steady" phase: reading it forces a
## render-thread sync, which would distort the load/fly frame timings.

const MODES := {"legacy": 1, "direct": 2}

var args := {}
var shots := []  # [[when, path], ...]
var terrain
var camera: Camera3D
var start_pos: Vector3
var vp_rid: RID

var frame := 0
var phase := "load"  # load -> (fly_out -> fly_back -> settle) -> steady -> done
var phase_frame := 0
var phase_start_usec := 0
var samples := {}  # phase -> {frame_ms: [], gpu_ms: [], cpu_render_ms: [], pending: []}
var phase_results := {}
var timed_out := false
var validation_report := {}
var audit_report := {}
var discovery_audit_report := {}
var checks := []  # remaining post-steady checks


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
	if args.has("max_in_flight"):
		terrain.direct_max_in_flight = int(args["max_in_flight"])
	terrain.debug_view = int(args.get("view", "0"))
	terrain.legacy_trim_indices = args.get("trim", "0") == "1"
	terrain.dual_contouring = args.get("dual_contouring", "0") == "1"
	if args.has("voxel"):
		terrain.voxel_size = float(args["voxel"])
	if args.has("lods"):
		terrain.lod_numbers = int(args["lods"])
	var terrains := {"cascade": 1, "continents": 2, "archetypes": 3, "shapes": 4}
	var terrain_arg: String = args.get("terrain", "cascade")
	if not terrains.has(terrain_arg):
		push_error("BENCH: unknown --terrain=%s (expected cascade, continents, archetypes or shapes)" % terrain_arg)
		set_process(false)
		get_tree().quit(1)
		return
	terrain.terrain_mode = terrains[terrain_arg]
	if args.has("compact"):
		terrain.direct_compaction = args["compact"] == "1"
	if args.has("compact_extent"):
		terrain.direct_compact_max_extent = float(args["compact_extent"])
	if args.has("lod_error"):
		terrain.lod_pixel_error = float(args["lod_error"])
	if args.has("skirt"):
		terrain.skirt_depth = float(args["skirt"])
	if args.has("discovery_cull"):
		terrain.discovery_cull_mode = int(args["discovery_cull"])
	if args.has("discovery_cache"):
		terrain.discovery_cache = args["discovery_cache"] == "1"
	terrain.discovery_verify = args.get("discovery_verify", "0") == "1"
	terrain.destroy_verify = args.get("destroy_verify", "0") == "1"
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
	var above_ground: String = args.get("above_ground", "" if terrain.terrain_mode >= 3 or args.has("cam_pos") else "50")
	if above_ground != "":
		# Raise or lower the camera (and the fly path) to this height above the terrain under it.
		var h := float(above_ground)
		for i in 32:
			var d: float = terrain.get_sdf_at(camera.global_position)
			if absf(d - h) < maxf(1.0, h * 0.05):
				break
			camera.global_position.y += h - d
		start_pos = camera.global_position
		print("BENCH: camera at %s (%.1f m above the terrain)" % [camera.global_position, terrain.get_sdf_at(camera.global_position)])
	# The scene's clip planes at the default LOD count, so pictures don't change; with more LODs they
	# follow the camera's altitude (see _clip_planes), set once at the pose.
	if terrain.lod_numbers > 15:
		_clip_planes(camera, camera.near)


	_begin_phase("load")
	print("BENCH: start mode=%s scenario=%s" % [args.get("mode", "legacy"), args.get("scenario", "load")])


func _parse_args() -> void:
	for a in OS.get_cmdline_user_args():
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
				if args.get("scenario", "load") == "fly":
					_begin_phase("fly_out")
				else:
					_begin_phase("steady")
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
		"settle":
			if terrain.is_load_complete():
				_end_phase()
				_begin_phase("steady")
		"steady":
			if phase_frame >= steady:
				_end_phase()
				for c in ["discovery_audit", "audit", "validate"]:
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
		"audit":
			terrain.debug_audit_pages()
		"validate":
			terrain.debug_validate(int(args["validate"]), args["out"].get_base_dir() if args.has("out") else "")


func _take_shots(when: String) -> void:
	for shot in shots:
		if shot[0] == when:
			await RenderingServer.frame_post_draw
			var img := get_viewport().get_texture().get_image()
			img.save_png(shot[1])
			print("BENCH: screenshot %s -> %s" % [when, shot[1]])


func _finish() -> void:
	set_process(false)
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


# Above 15 LODs: near = altitude / 1000 (altitude = |SDF| at the camera, at least the scene's near),
# far = near * 2^21 capped at the discoverer's load range. Godot's float projection loses its far
# plane past far/near ~2^24 (the light culler then ERRORs every frame), so far/near stays at 2^21.
func _clip_planes(cam: Camera3D, base_near: float) -> void:
	var altitude: float = absf(terrain.get_sdf_at(cam.global_position))
	cam.near = maxf(base_near, altitude / 1000.0)
	cam.far = minf(terrain.get_load_radius(), cam.near * 2097152.0)
