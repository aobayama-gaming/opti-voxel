extends Node
## Integer-frame harness (FrameSpike, src/frame_spike.cpp): GPU WorldPos / SDF / probes vs the CPU
## mirrors, around far world origins and 2^32-voxel boundaries. Needs a GPU (not --headless). Run:
##   tools/frame_test.sh            (or: godot --path project res://debug/frame_spike.tscn -- --out=x.json)
## Args: --voxel=<metres> (default 0.5), --out=<json>

var args := {}


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv: PackedStringArray = a.substr(2).split("=", true, 1)
			args[kv[0]] = kv[1] if kv.size() > 1 else "true"
	# One frame first, so the engine has finished starting up.
	await get_tree().process_frame
	var spike = ClassDB.instantiate("FrameSpike")
	spike.voxel_size = float(args.get("voxel", "0.5"))
	add_child(spike)
	var report: Dictionary = spike.run()
	print("FRAME: ", JSON.stringify(report))
	if args.has("out"):
		var f := FileAccess.open(args["out"], FileAccess.WRITE)
		f.store_string(JSON.stringify(report, "  "))
		f.close()
	print("BENCH: done")
	spike.queue_free()
	await get_tree().process_frame
	get_tree().quit(0 if report.get("ok", false) else 1)
