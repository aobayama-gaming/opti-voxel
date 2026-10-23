extends Node
## GPU cascade harness (CascadeSpike, src/cascade_spike.cpp): the cascade passes of the meshers against the
## CPU oracle (Cascade::run), bit for bit, every level of chunks of every LOD, plus their GPU timing.
## Needs a GPU (not --headless). Run:
##   tools/cascade_test.sh       (or: godot --path project res://debug/cascade_spike.tscn -- --out=x.json)
## Args: --lods=N (25), --voxel=<metres> (0.5), --check=0|1, --points=0|1, --timing=0|1, --runs=N (10), --sets=8,4,4c (4c: the example climate, 4f: its fields only),
## --out=<json>

var args := {}


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv: PackedStringArray = a.substr(2).split("=", true, 1)
			args[kv[0]] = kv[1] if kv.size() > 1 else "true"
	# One frame first, so the engine has finished starting up.
	await get_tree().process_frame
	var spike = ClassDB.instantiate("CascadeSpike")
	add_child(spike)
	var options := {
		"lods": int(args.get("lods", "25")),
		"voxel": float(args.get("voxel", "0.5")),
		"check": args.get("check", "1") == "1",
		"points": args.get("points", "1") == "1",
		"table": args.get("table", "1") == "1",
		"biomes": args.get("biomes", "1") == "1",
		"timing": args.get("timing", "1") == "1",
		"runs": int(args.get("runs", "10")),
		"sets": args.get("sets", "8,4,4c"),
	}
	var report: Dictionary = spike.run(options)
	print("CASCADE: ", JSON.stringify(report))
	if args.has("out"):
		var f := FileAccess.open(args["out"], FileAccess.WRITE)
		f.store_string(JSON.stringify(report, "  "))
		f.close()
	print("BENCH: done")
	spike.queue_free()
	await get_tree().process_frame
	get_tree().quit(0 if report.get("ok", false) else 1)
