extends Node
## Hermite cascade report (CascadeReportNode -> CascadeReport, src/cascade_report.cpp): fidelity, cost and PNG slices of the
## CPU reference. Pure CPU, works with --headless. Run:
##   tools/cascade_report.sh        (or: godot --headless --path project res://debug/cascade_report.tscn -- --out=DIR)
## Args: --out=<dir> (required), --scale=<n> (sample size multiplier, default 1)


func _ready() -> void:
	var args := {}
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv: PackedStringArray = a.substr(2).split("=", true, 1)
			args[kv[0]] = kv[1] if kv.size() > 1 else "true"
	var report_node = ClassDB.instantiate("CascadeReportNode")
	add_child(report_node)
	var report: Dictionary = report_node.run(args.get("out", "user://cascade"), int(args.get("scale", "1")))
	print("CASCADE: ", JSON.stringify(report))
	report_node.queue_free()
	await get_tree().process_frame
	get_tree().quit(0 if report.get("ok", false) else 1)
