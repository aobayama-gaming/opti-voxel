extends Node
## C++ unit tests (UnitTests, src/tests/): runs them, prints the totals and quits. Pure CPU, works
## with --headless. Run:
##   tools/unit_test.sh [filter]     (or: godot --headless --path project res://debug/unit_tests.tscn -- --filter=X)
## Args: --filter=<doctest wildcard on test case names>


func _ready() -> void:
	var filter := ""
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--filter="):
			filter = a.substr(len("--filter="))
	if not ClassDB.class_exists("UnitTests"):
		# Release builds don't contain the tests.
		print("UNIT: ", JSON.stringify({"ok": false, "error": "UnitTests not registered (release build?)"}))
		get_tree().quit(1)
		return
	var tests = ClassDB.instantiate("UnitTests")
	add_child(tests)
	var report: Dictionary = tests.run(filter)
	print("UNIT: ", JSON.stringify(report))
	tests.queue_free()
	await get_tree().process_frame
	get_tree().quit(0 if report.get("ok", false) else 1)
