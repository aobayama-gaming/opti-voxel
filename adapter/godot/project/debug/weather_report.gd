extends Node
## Weather report (ClimateReportNode.weather -> WeatherReport, src/weather_report.cpp): a world's weather run on the CPU
## alone, mapped at game minutes; works with --headless. Run:
##   tools/weather_report.sh   (or: godot --headless --path project res://debug/weather_report.tscn -- --out=DIR)
## Args: --out=<dir> (required), --world=<res:// or absolute path> (default res://worlds/ocean_test.planet.json),
## --minutes=<a,b,c> (game minutes after the spin-up, 0,1,10), --width=<pixels> (512)


func _ready() -> void:
	var args := {}
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--"):
			var kv: PackedStringArray = a.substr(2).split("=", true, 1)
			args[kv[0]] = kv[1] if kv.size() > 1 else "true"
	var minutes := PackedFloat32Array()
	for m in String(args.get("minutes", "0,1,10")).split(","):
		minutes.append(float(m))
	var report: Dictionary = ClassDB.class_call_static("ClimateReportNode", "weather", args.get("out", "user://weather"),
			args.get("world", "res://worlds/ocean_test.planet.json"), minutes, int(args.get("width", "512")))
	print("WEATHER: ", JSON.stringify(report))
	await get_tree().process_frame
	get_tree().quit(0 if report.get("ok", false) else 1)
