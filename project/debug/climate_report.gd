extends Node
## Climate report (ClimateReport, src/climate_report.cpp): biome and field maps of a climate table, pure CPU,
## works with --headless. Run:
##   tools/climate_report.sh   (or: godot --headless --path project res://debug/climate_report.tscn -- --out=DIR)
## Args: --out=<dir> (required), --climate=<res:// or absolute path> (default res://climate/example.json),
## --voxel=<m> (0.05), --lods=<n> (15), --extent_km=<km> (60), --size=<points per side> (512),
## --surface=<0|1> (1: each point of the base's tangent plane traced down to its ground; 0: the plane itself), --center=<u,v> (tangent metres, 0,0)


func _center(p_text: String) -> Vector2:
	var parts := p_text.split(",")
	return Vector2(float(parts[0]), float(parts[1])) if parts.size() == 2 else Vector2.ZERO


func _ready() -> void:
	var args := {}
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv: PackedStringArray = a.substr(2).split("=", true, 1)
			args[kv[0]] = kv[1] if kv.size() > 1 else "true"
	var report_node = ClassDB.instantiate("ClimateReport")
	add_child(report_node)
	var report: Dictionary = report_node.run(args.get("out", "user://climate"), args.get("climate", "res://climate/example.json"),
			float(args.get("voxel", "0.05")), int(args.get("lods", "15")), float(args.get("extent_km", "60")),
			int(args.get("size", "512")), args.get("surface", "1") != "0", _center(args.get("center", "0,0")),
			int(args.get("lattice", "-1")))
	print("CLIMATE: ", JSON.stringify(report))
	report_node.queue_free()
	await get_tree().process_frame
	get_tree().quit(0 if report.get("ok", false) else 1)
