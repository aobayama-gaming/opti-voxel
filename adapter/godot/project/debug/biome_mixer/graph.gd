extends RefCounted
## A biome's graph as the engine reads it (world.json's "graph": {"nodes": [...]}, climate_parse.cpp's format), edited in
## place: the mixer keeps no model of its own. A link is an input key naming another node's id: input port i of a node
## is its kind's input i (EngineInfo.kind), a mix's port i its row i's "in" (one port more: a new row). Every node but
## the output has one output, port 0. "pos" is the mixer's (the engine ignores it).

const EngineInfo := preload("res://debug/biome_mixer/engine_info.gd")

# Pixels between layout columns and rows (layout).
const COL_W := 340.0
const ROW_H := 200.0
# A mix row's height in the layout.
const MIX_ROW_H := 34.0


static func nodes(p_graph: Dictionary) -> Array:
	if not p_graph.has("nodes"):
		p_graph.nodes = []
	return p_graph.nodes


static func node(p_graph: Dictionary, p_id: String) -> Dictionary:
	for n in nodes(p_graph):
		if String(n.get("id", "")) == p_id:
			return n
	return {}


static func rows(p_node: Dictionary) -> Array:
	if p_node.kind == "mix" and not p_node.has("rows"):
		p_node.rows = []
	return p_node.get("rows", [])


static func input_count(p_node: Dictionary) -> int:
	if p_node.kind == "mix":
		return rows(p_node).size() + 1
	return EngineInfo.kind(p_node.kind).get("inputs", PackedStringArray()).size()


static func input_name(p_node: Dictionary, p_port: int) -> String:
	if p_node.kind == "mix":
		return "row %d" % (p_port + 1) if p_port < rows(p_node).size() else "+ link here: a new row"
	return EngineInfo.kind(p_node.kind).inputs[p_port]


static func has_output(p_node: Dictionary) -> bool:
	return p_node.kind != "output"


# The id input p_port of p_node reads, "" for none.
static func input_of(p_node: Dictionary, p_port: int) -> String:
	if p_node.kind == "mix":
		var r := rows(p_node)
		return String(r[p_port].get("in", "")) if p_port < r.size() else ""
	var inputs: PackedStringArray = EngineInfo.kind(p_node.kind).get("inputs", PackedStringArray())
	return String(p_node.get(inputs[p_port], "")) if p_port < inputs.size() else ""


# Every link: {from, to, to_port} (an input naming a node the graph has).
static func links(p_graph: Dictionary) -> Array:
	var out := []
	for n in nodes(p_graph):
		for port in input_count(n):
			var from := input_of(n, port)
			if from != "" and not node(p_graph, from).is_empty():
				out.append({"from": from, "to": String(n.id), "to_port": port})
	return out


# p_to's input p_port reads p_from (a mix's port past its rows: a new row at volume 1).
static func link(p_graph: Dictionary, p_from: String, p_to: String, p_port: int) -> void:
	var n := node(p_graph, p_to)
	if n.kind == "mix":
		var r := rows(n)
		if p_port < r.size():
			r[p_port].in = p_from
		else:
			r.append({"in": p_from, "x": 1.0})
		return
	n[EngineInfo.kind(n.kind).inputs[p_port]] = p_from


# p_to's input p_port reads nothing (a mix's row goes, the rows under it move up a port).
static func unlink(p_graph: Dictionary, p_to: String, p_port: int) -> void:
	var n := node(p_graph, p_to)
	if n.kind == "mix":
		if p_port < rows(n).size():
			rows(n).remove_at(p_port)
		return
	n.erase(EngineInfo.kind(n.kind).inputs[p_port])


# A parameter's value for a new node: its default, else one of its type's.
static func default_value(p_param: Dictionary, p_fields: Array) -> Variant:
	var d = p_param.get("default")
	if d != null:
		return int(d) if p_param.type in ["seed", "pow2"] else d
	var names: Array = p_fields.map(func(f): return String(f.get("name", "")))
	match String(p_param.type):
		"seed", "pow2":
			return 1
		"field":
			return names[0] if not names.is_empty() else ""
		"shape":
			return p_param.options[0]
		"coeffs":
			return [0.0, 1.0, 0.0, 0.0]
		"points":
			return [[-100.0, -100.0], [0.0, 0.0], [100.0, 100.0]]
		"rows":
			return []
	return 1.0


# A new node of p_kind at p_pos, its required parameters at their defaults merged with p_params: its id.
static func add(p_graph: Dictionary, p_kind: String, p_pos: Vector2, p_fields: Array, p_params: Dictionary = {}) -> String:
	var i := 1
	while not node(p_graph, "%s%d" % [p_kind, i]).is_empty():
		i += 1
	var n := {"id": "%s%d" % [p_kind, i], "kind": p_kind}
	for p in EngineInfo.kind(p_kind).get("params", []):
		if p.required:
			n[p.name] = default_value(p, p_fields)
	n.merge(p_params, true)
	n.pos = [p_pos.x, p_pos.y]
	nodes(p_graph).append(n)
	return n.id


# Nodes out of the graph. What read one reads what it read ("in"), else nothing (a mix row that played it goes; a note
# it warped reads no warp: a note's warp input is a warp node).
static func delete(p_graph: Dictionary, p_ids: Array) -> void:
	for id in p_ids:
		var gone := node(p_graph, id)
		if gone.is_empty():
			continue
		var through := String(gone.get("in", ""))
		for n in nodes(p_graph):
			for port in range(input_count(n) - 1, -1, -1):
				if input_of(n, port) != id:
					continue
				if through != "" and input_name(n, port) != "warp":
					link(p_graph, through, String(n.id), port)
				else:
					unlink(p_graph, String(n.id), port)
		nodes(p_graph).erase(gone)


static func laid_out(p_graph: Dictionary) -> bool:
	return nodes(p_graph).all(func(n): return n.has("pos"))


# Positions for a graph read without them (world.json's, the template's), or again (p_force): columns by how far a
# node is from the output (the output rightmost), rows in the order the nodes come, spaced by p_sizes (id -> its
# GraphNode's size, once built) or by COL_W and ROW_H.
static func layout(p_graph: Dictionary, p_sizes: Dictionary = {}, p_force: bool = false) -> void:
	if laid_out(p_graph) and not p_force:
		return
	var depth := {}
	var out := ""
	for n in nodes(p_graph):
		if n.kind == "output":
			out = String(n.id)
	if out != "":
		var stack := [[out, 0]]
		while not stack.is_empty():
			var top: Array = stack.pop_back()
			if depth.get(top[0], -1) >= int(top[1]) or int(top[1]) > nodes(p_graph).size():
				continue
			depth[top[0]] = int(top[1])
			var n := node(p_graph, top[0])
			for port in input_count(n):
				var from := input_of(n, port)
				if from != "" and not node(p_graph, from).is_empty():
					stack.append([from, int(top[1]) + 1])
	var deepest := 0
	for id in depth:
		deepest = maxi(deepest, depth[id])
	var gap := COL_W - 300.0
	var col_of := {}
	var widths := {}
	for n in nodes(p_graph):
		var col: int = deepest + 1 - depth.get(String(n.id), deepest + 1)
		col_of[n.id] = col
		var size: Vector2 = p_sizes.get(String(n.id), Vector2(COL_W - gap, ROW_H - gap + (rows(n).size() * MIX_ROW_H if n.kind == "mix" else 0.0)))
		widths[col] = maxf(widths.get(col, 0.0), size.x)
	var x := {}
	var left := 0.0
	for col in range(deepest + 2):
		x[col] = left
		left += widths.get(col, 0.0) + gap
	var y := {}
	for n in nodes(p_graph):
		var col: int = col_of[n.id]
		var size: Vector2 = p_sizes.get(String(n.id), Vector2(COL_W - gap, ROW_H - gap + (rows(n).size() * MIX_ROW_H if n.kind == "mix" else 0.0)))
		n.pos = [x[col], y.get(col, 0.0)]
		y[col] = y.get(col, 0.0) + size.y + gap
