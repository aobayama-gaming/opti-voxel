extends Node3D
## Spike test for the GPU -> MultiMesh page path (no terrain). Run:
##   godot --path project --resolution 1280x720 res://debug/spike.tscn -- --out=/tmp/spike.json --shots=/tmp/spike
## Args: --extra_pages=N (pool RAM measurement), --page_capacity=N, --side=N, --out=, --shots=<prefix>

var args := {}
var spike
var frame := 0
var rss_before := 0.0
var rss_after_pool := 0.0


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv: PackedStringArray = a.substr(2).split("=", true, 1)
			args[kv[0]] = kv[1] if kv.size() > 1 else "true"
	$Camera3D.rotation_degrees.x = -20.0
	rss_before = _rss_mb()
	spike = ClassDB.instantiate("QuadPoolSpike")
	spike.extra_pages = int(args.get("extra_pages", "0"))
	spike.page_capacity = int(args.get("page_capacity", "16384"))
	spike.side = int(args.get("side", "64"))
	add_child(spike)
	rss_after_pool = _rss_mb()


func _process(_delta: float) -> void:
	frame += 1
	if frame > 900:
		_finish({"ok": false, "error": "spike did not finish in 900 frames", "stage": spike.get_stage()})
		return
	if spike.get_stage() != 1:
		return
	# Write verified: screenshot the full sheet, then run the kill test.
	set_process(false)
	var prefix: String = args.get("shots", "")
	if prefix != "":
		if args.get("top", "0") == "1":
			# Straight down onto the sheet centre (sheet: origin (-72, 342, -150), 40 m).
			$Camera3D.global_position = Vector3(-52.0, 400.0, -130.0)
			$Camera3D.rotation_degrees = Vector3(-90.0, 0.0, 0.0)
		for view in [[0, "shaded"], [4, "wire"], [1, "normals"], [6, "group"], [7, "quad"]]:
			spike.set_debug_view(view[0])
			for i in 3:
				await RenderingServer.frame_post_draw
			get_viewport().get_texture().get_image().save_png("%s_%s.png" % [prefix, view[1]])
	spike.run_kill_test()
	while spike.get_stage() != 3:
		await get_tree().process_frame
	var report: Dictionary = spike.get_report()
	var slots: int = (1 + spike.extra_pages) * spike.page_capacity
	report["rss_before_mb"] = rss_before
	# Pages are allocated on the render thread, possibly after add_child returned: measure now.
	var rss_now := _rss_mb()
	report["rss_at_report_mb"] = rss_now
	report["pool_slots"] = slots
	report["ram_bytes_per_slot"] = (rss_now - rss_before) * 1048576.0 / slots
	_finish(report)


func _finish(report: Dictionary) -> void:
	print("SPIKE: ", JSON.stringify(report))
	if args.has("out"):
		var f := FileAccess.open(args["out"], FileAccess.WRITE)
		f.store_string(JSON.stringify(report, "  "))
		f.close()
	print("BENCH: done")
	spike.queue_free()
	await get_tree().process_frame
	await get_tree().process_frame
	get_tree().quit(0 if report.get("ok", false) else 1)


func _rss_mb() -> float:
	var out := []
	OS.execute("grep", ["VmRSS", "/proc/%d/status" % OS.get_process_id()], out)
	if out.is_empty():
		return -1.0
	return int(out[0].split(":")[1].strip_edges().split(" ")[0]) / 1024.0
