extends RefCounted
## A biome graph to its preview shader. run() walks the graph from its Output node and returns the shader's code
## (which depends only on the graph's structure: the mixer recompiles when it changes) and the uniforms' values (which
## carry every parameter: a slider sets uniforms, no recompile), with the bounds the march needs.
##
## A field port's record (_rec): e (the GLSL expression), m (its mean's expression, for a Mix's centring), mb (a bound
## of |mean|), amp (a bound of |value|), dev (a bound of |value - mean|), slope (|gradient| per metre: sums of
## independent notes add in quadrature, their typical size, not the worst case, which never happens) and lam (the
## coarsest note's wavelength upstream, metres; 0 if none). The bounds start the march under the highest ground
## (u_hmax) and size its step (u_lip). A warp port's record is its W slot and slope; a carve port's, its code.

const Kinds := preload("res://debug/biome_mixer/kinds.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")
const ShaderLib := preload("res://debug/biome_mixer/shader_lib.gd")
const World := preload("res://debug/biome_mixer/world.gd")

# |grad n| per wavelength of a note: about its typical size, not its worst (~2.2 in rare corners): the worst case summed
# over a biome's notes made the march crawl. A rare overshoot lands inside the ground and is bisected back; the view's
# Step lowers it where thin ridges show holes.
const NOISE_SLOPE := 1.6
const CURVE_POINTS := 64
const CARVE_WARNING := "Carve is preview-only: the engine has no holes yet."
const NOTE_WARP_WARNING := "Note warp is preview-only: the engine's warps are climate fields."
# A row's warp notes (x, y, z), none.
const NO_SRC := [-1, -1, -1]

var model: Dictionary
var voxel_m: float
var lods: int
var origin: Array  # the preview's centre, absolute voxels (the datum's height)
var exaggeration := 1.0

var incoming := {}  # "to:port" -> [from, from_port]
var done := {}  # id -> its output records
var visiting := {}
var decls := PackedStringArray()
var functions := PackedStringArray()
var body := PackedStringArray()
var uniforms := {}
var errors := PackedStringArray()
var warnings := PackedStringArray()
var mix_rows := {}  # Mix id -> its rows: {unit (metres at volume 1), lam (its note's wavelength), peak (metres), on}
var warp_rows := []  # the octave table's rows: {octave, k, slot, src, axis, amp, wm, st}
var note_rows := []
var warps := 0
var climate_rec := {}  # the climate warp's record, once a note reads it
var climate_cfg := {}  # the climate warp's octaves (a world's; else the model's, Kinds.climate_warp)
var curve_in := {}  # Height curve id -> the bound of its input (metres): what its editor shows
var kinds := {}  # node id -> its kind, for the budget (a world compiles several models)
var row_of := {}  # a note row's key -> its index: biomes reading the same note share its row
var place_count := 0  # a world's biomes x fields: the length of u_atc / u_ati


# The preview's centre (a model's or a world's origin_m) in absolute voxels, at the datum's height.
func _set_origin(p_doc: Dictionary) -> void:
	var o: Array = p_doc.get("origin_m", [0.0, 0.0])
	origin = [int(round(o[0] / voxel_m)), int(round(Octaves.DATUM_M / voxel_m)), int(round(o[1] / voxel_m))]


# Compile.new().run(model): one build per instance.
func run(p_model: Dictionary, p_exaggeration: float = 1.0) -> Dictionary:
	model = p_model
	exaggeration = p_exaggeration
	voxel_m = model.get("voxel_m", Octaves.DEFAULT_VOXEL_M)
	lods = model.get("lods", Octaves.DEFAULT_LODS)
	_set_origin(model)
	for l in model.get("links", []):
		incoming["%s:%d" % [l.to, int(l.to_port)]] = [l.from, int(l.from_port)]
	var out_id := Kinds.first_of(model, "output")
	var h := _zero()
	var holes = null
	if out_id == "":
		errors.append("No Output node: add one and connect a height to it.")
	else:
		var node: Dictionary = model.nodes[out_id]
		h = _input_or_zero(out_id, 0, "The Output's height is not connected.")
		holes = _input(out_id, 1)
		uniforms["u_flat"] = _linear(Kinds.param(node, "flat"))
		uniforms["u_steep"] = _linear(Kinds.param(node, "steep"))
	body.append("\tfloat h = (%s) * u_exag;" % h.e)
	body.append("\tc_flat = u_flat;\n\tc_steep = u_steep;\n\tc_map = u_flat;\n\tc_fields = vec4(0.0, 0.0, h, 0.0);")
	body.append("\tfloat d = p.y - h;")
	if holes != null:
		body.append(holes.code)
	body.append("\treturn d;")
	_table()
	return _result(exaggeration * h.amp + 1.0, sqrt(1.0 + pow(exaggeration * h.slope, 2.0)))


# The shader and its uniforms, with the march's bounds: the highest ground p_hmax and the slope bound p_lip.
func _result(p_hmax: float, p_lip: float) -> Dictionary:
	uniforms["u_hmax"] = p_hmax
	uniforms["u_lip"] = p_lip
	uniforms["u_voxel"] = voxel_m
	uniforms["u_exag"] = exaggeration
	var code := ShaderLib.preview("\n".join(decls) + "\n", "\n".join(functions) + "\n", "\n".join(body) + "\n", note_rows.size(), warps, int(uniforms.u_n_oct), place_count)
	var bytes := ShaderLib.uniform_bytes(code)
	if bytes > ShaderLib.UNIFORM_BYTES:
		errors.append("The shader's uniforms take %d KB: past %d KB the GPU hangs, the view keeps the last shader (fewer Height curves or biomes)." % [bytes / 1024, ShaderLib.UNIFORM_BYTES / 1024])
	return {
		"drawable": bytes <= ShaderLib.UNIFORM_BYTES,
		"uniform_bytes": bytes,
		"ok": errors.is_empty(),
		"code": code,
		"uniforms": uniforms,
		"errors": errors,
		"warnings": warnings,
		"hmax": p_hmax,
		"lip": p_lip,
		"mix_rows": mix_rows,
		"curve_in": curve_in,
		"budget": _budget(),
	}


# What the graph spends per point: the noise it reads (each note once, each warp field's octaves once per axis) and
# the work on top of them.
func _budget() -> Dictionary:
	var counts := {}
	for id in done:
		var kind: String = kinds.get(id, "")
		counts[kind] = counts.get(kind, 0) + 1
	return {
		"reads": note_rows.size() + warp_rows.size(),
		"notes": note_rows.size(),
		"warped": note_rows.filter(func(r): return r.slot >= 0).size(),
		"note_warped": note_rows.filter(func(r): return r.src != NO_SRC).size(),
		"warp_octaves": warp_rows.size(),
		"shapes": counts.get("shape", 0),
		"multifractals": counts.get("multifractal", 0),
	}


static func _rec(p_e: String, p_m: String, p_mb: float, p_amp: float, p_dev: float, p_slope: float, p_lam: float) -> Dictionary:
	return {"e": p_e, "m": p_m, "mb": p_mb, "amp": p_amp, "dev": p_dev, "slope": p_slope, "lam": p_lam}


static func _zero() -> Dictionary:
	return _rec("0.0", "0.0", 0.0, 0.0, 0.0, 0.0, 0.0)


# The record of the link into (id, port), or null.
func _input(p_id: String, p_port: int) -> Variant:
	var key := "%s:%d" % [p_id, p_port]
	if not incoming.has(key):
		return null
	var src: Array = incoming[key]
	if not model.nodes.has(src[0]):
		return null
	var recs: Array = _node(src[0])
	return recs[src[1]] if src[1] < recs.size() else null


# The record into (id, port), or zero (with p_warning if one is given).
func _input_or_zero(p_id: String, p_port: int, p_warning: String = "") -> Dictionary:
	var r = _input(p_id, p_port)
	if r != null:
		return r
	if p_warning != "":
		warnings.append(p_warning)
	return _zero()


func _node(p_id: String) -> Array:
	if done.has(p_id):
		return done[p_id]
	if visiting.has(p_id):
		errors.append("A loop through %s: a node can't feed itself." % p_id)
		return []
	visiting[p_id] = true
	var node: Dictionary = model.nodes[p_id]
	kinds[p_id] = node.kind
	var recs: Array = call("_emit_" + node.kind, p_id, node)
	visiting.erase(p_id)
	done[p_id] = recs
	return recs


func _uniform(p_type: String, p_name: String, p_value: Variant) -> String:
	decls.append("uniform %s %s;" % [p_type, p_name])
	uniforms[p_name] = p_value
	return p_name


func _var(p_type: String, p_name: String, p_expr: String) -> String:
	body.append("\t%s %s = %s;" % [p_type, p_name, p_expr])
	return p_name


func _k(p_node: Dictionary) -> int:
	return clampi(int(Kinds.param(p_node, "k")), Octaves.K_MIN, Octaves.ground_top(lods, voxel_m))


# A note reads N[index] of the octave table (shader_lib's density loop) times its fade F[index]; with its warp on, at
# the point moved by W[slot] (the climate's) or by the notes of the Note warp linked into its port. The record also
# carries the unfaded noise and the fade, so a Shape fades around its mean (a faded note moves nothing).
func _emit_note(p_id: String, p_node: Dictionary) -> Array:
	var k := _k(p_node)
	var lam := Octaves.wavelength_m(k, voxel_m)
	var warp = null
	if Kinds.param(p_node, "warp"):
		warp = _input(p_id, 0)
		if warp == null:
			warp = _climate_warp()
	var slope := NOISE_SLOPE / lam
	var stretch := maxf(float(Kinds.param(p_node, "stretch_y")), 1.0)
	var row := {"octave": Octaves.ground_octave(k, lods, voxel_m), "k": k, "slot": -1, "src": NO_SRC, "axis": 0, "amp": 1.0, "wm": 0.0, "st": 1.0 / stretch}
	if warp != null:
		var wm := float(Kinds.param(p_node, "warp_m"))
		row.slot = warp.get("slot", -1)
		row.src = warp.get("src", NO_SRC)
		row.wm = wm / voxel_m
		slope *= 1.0 + wm * warp.slope
		if warp.has("src") and warp.lam <= lam:
			warnings.append("A Note warp moves the %.0f m note by a note as fine or finer: the engine reads notes coarse to fine, so a warp's notes are coarser." % lam)
	var key := str([k, row.slot, row.src, row.wm, row.st])
	if not row_of.has(key):
		row_of[key] = note_rows.size()
		note_rows.append(row)
	var i: int = row_of[key]
	var rec := _rec("(F[%d] * N[%d])" % [i, i], "0.0", 0.0, 1.0, 1.0, slope, lam)
	rec.raw = "N[%d]" % i
	rec.fade = "F[%d]" % i
	rec.row = i
	return [rec]


# The climate's warp fields: three fBms (the model's climate_warp), rows of the octave table summed into W[slot] (x, y,
# z). One slot however many notes it moves: the engine reads its climate once per point.
func _climate_warp() -> Dictionary:
	if climate_rec.is_empty():
		var cw := climate_cfg if not climate_cfg.is_empty() else Kinds.climate_warp(model)
		var k := clampi(int(cw.k), Octaves.K_MIN, Octaves.k_top(lods))
		var slot := warps
		warps += 1
		var slope := 0.0
		for i in mini(int(cw.octaves), k - Octaves.K_MIN + 1):
			var ki := k - i
			var ai: float = cw.amp * pow(cw.persistence, i)
			for axis in 3:
				warp_rows.append({"octave": Octaves.field_octave(int(cw.seed) + axis, ki), "k": ki, "slot": slot, "src": NO_SRC, "axis": axis, "amp": ai, "wm": 0.0, "st": 1.0})
			slope += ai * NOISE_SLOPE / Octaves.wavelength_m(ki, voxel_m)
		climate_rec = {"slot": slot, "slope": slope}
	return climate_rec


# The cheap warp: x, y, z are notes read earlier in the table (their row, F * N), no noise of its own. Its record: the
# rows (NO_SRC's -1 for an axis left open), its slope and its finest note's wavelength.
func _emit_note_warp(p_id: String, _node: Dictionary) -> Array:
	if not NOTE_WARP_WARNING in warnings:
		warnings.append(NOTE_WARP_WARNING)
	var src := NO_SRC.duplicate()
	var slope := 0.0
	var lam := INF
	for axis in 3:
		var r = _input(p_id, axis)
		if r == null:
			continue
		if not r.has("row"):
			errors.append("A Note warp's %s takes a Note's n directly (not a shaped or mixed signal)." % "xyz"[axis])
			continue
		src[axis] = r.row
		slope += r.slope * r.slope
		lam = minf(lam, r.lam)
	return [{"src": src, "slope": sqrt(slope), "lam": lam}]


# The octave table's uniforms (ShaderLib.OCTAVE_TABLE's columns): the warps' rows first (their W is complete before a
# note reads it), then the notes'.
func _table() -> void:
	var rows: Array = warp_rows + note_rows
	if warps > ShaderLib.MAX_WARPS:
		errors.append("%d warp fields: the preview takes %d." % [warps, ShaderLib.MAX_WARPS])
	if rows.size() > ShaderLib.MAX_OCTAVES:
		errors.append("%d octaves (notes, and 3 per warp octave): the preview takes %d." % [rows.size(), ShaderLib.MAX_OCTAVES])
		rows = rows.slice(0, ShaderLib.MAX_OCTAVES)
	var cols := {}
	for c in ShaderLib.OCTAVE_TABLE:
		match c[1]:
			"int":
				cols[c[0]] = PackedInt32Array()
			"float":
				cols[c[0]] = PackedFloat32Array()
			"vec3":
				cols[c[0]] = PackedVector3Array()
		cols[c[0]].resize(ShaderLib.table_size(rows.size()))
	for i in rows.size():
		var r: Dictionary = rows[i]
		var w := Octaves.shader_words(r.octave, r.k, origin)
		var row := {
			"u_bx": w.base.x, "u_by": w.base.y, "u_bz": w.base.z, "u_sx": w.salt.x, "u_sy": w.salt.y, "u_sz": w.salt.z,
			"u_fr": w.frac, "u_inv": pow(2.0, -r.k), "u_lam": Octaves.wavelength_m(r.k, voxel_m), "u_amp": r.amp,
			"u_wm": r.wm, "u_st": r.st, "u_slot": r.slot, "u_axis": r.axis, "u_jx": r.src[0], "u_jy": r.src[1], "u_jz": r.src[2],
		}
		for key in cols:
			cols[key][i] = row[key]
	uniforms.merge(cols, true)
	uniforms["u_n_oct"] = rows.size()
	uniforms["u_n_warp"] = mini(warp_rows.size(), rows.size())


func _emit_shape(p_id: String, p_node: Dictionary) -> Array:
	var n := _input_or_zero(p_id, 0, "A Shape without input reads 0.")
	var c := Kinds.shape_coeffs(p_node)
	var cu := _uniform("vec4", "u_%s_c" % p_id, Vector4(c[0], c[1], c[2], c[3]))
	var mean := Kinds.phi_mean(c)
	var mu := _uniform("float", "u_%s_mean" % p_id, mean)
	# phi over the input's range, sampled (the engine's shape_bound): its largest value, distance to its mean, slope.
	var na: float = maxf(n.amp, 1e-6)
	var amp := 0.0
	var dev := 0.0
	var dphi := 0.0
	for j in 129:
		var x := lerpf(-na, na, j / 128.0)
		var phi := Kinds.phi(c, x)
		amp = maxf(amp, absf(phi))
		dev = maxf(dev, absf(phi - mean))
		dphi = maxf(dphi, absf(Kinds.dphi(c, x)))
	# A note's shape fades around its mean: a far note's centred row then adds nothing (phi(0) would).
	var expr := "shape(%s, %s)" % [cu, n.e]
	if n.has("fade"):
		expr = "%s + %s * (shape(%s, %s) - %s)" % [mu, n.fade, cu, n.raw, mu]
	var v := _var("float", "v_%s" % p_id, expr)
	return [_rec(v, mu, absf(mean), amp, dev, dphi * n.slope, n.lam)]


func _emit_multifractal(p_id: String, p_node: Dictionary) -> Array:
	var s := _input_or_zero(p_id, 0, "A Multifractal without signal reads 0.")
	var chain = _input(p_id, 1)
	var gain: float = Kinds.param(p_node, "gain")
	var gu := _uniform("float", "u_%s_gain" % p_id, gain)
	var w := _var("float", "w_%s" % p_id, "clamp(%s * %s, 0.0, 1.0)" % [gu, chain.e if chain != null else "1.0"])
	var v := _var("float", "v_%s" % p_id, "%s * %s" % [w, s.e])
	var slope: float = sqrt(pow(s.slope, 2.0) + (pow(s.amp * gain * chain.slope, 2.0) if chain != null else 0.0))
	var rec := _rec(v, "(%s * %s)" % [w, s.m], s.mb, s.amp, s.dev, slope, s.lam)
	return [rec, rec]


func _emit_mix(p_id: String, p_node: Dictionary) -> Array:
	var rows := Kinds.rows(p_node)
	var solo := rows.any(func(r): return r.get("solo", false))
	var terms := PackedStringArray()
	var means := PackedStringArray()
	var units := []
	var amp := 0.0
	var slope := 0.0
	var mb := 0.0
	var lam := 0.0
	for i in rows.size():
		var r: Dictionary = rows[i]
		var rec = _input(p_id, i)
		if rec == null:
			units.append({"unit": 0.0, "lam": 0.0, "peak": 0.0, "on": false})
			continue
		# A row's metres at volume 1: the ground's amplitude of its note (a note-less row: 1 m).
		var unit: float = Octaves.ground_amp_m(rec.lam) if rec.lam > 0.0 else 1.0
		var on: bool = not r.get("mute", false) and (not solo or r.get("solo", false))
		var a: float = float(r.get("vol", 1.0)) * unit if on else 0.0
		var centre := 1.0 if r.get("centre", true) else 0.0
		# Its swing when it plays: volume x unit x the shaped note's bound (about its mean when centred).
		var peak: float = absf(float(r.get("vol", 1.0)) * unit) * (rec.dev if centre > 0.0 else rec.amp)
		units.append({"unit": unit, "lam": rec.lam, "peak": peak, "on": on})
		var au := _uniform("float", "u_%s_a%d" % [p_id, i], a)
		var cu := _uniform("float", "u_%s_c%d" % [p_id, i], centre)
		terms.append("%s * (%s - %s * %s)" % [au, rec.e, cu, rec.m])
		means.append("%s * (1.0 - %s) * %s" % [au, cu, rec.m])
		amp += absf(a) * (rec.dev if centre > 0.0 else rec.amp)
		slope += pow(a * rec.slope, 2.0)
		mb += absf(a) * (1.0 - centre) * rec.mb
		lam = maxf(lam, rec.lam)
	mix_rows[p_id] = units
	var v := _var("float", "v_%s" % p_id, " + ".join(terms) if not terms.is_empty() else "0.0")
	var m := " + ".join(means) if not means.is_empty() else "0.0"
	return [_rec(v, "(%s)" % m, mb, amp, amp + mb, sqrt(slope), lam)]


func _emit_add(p_id: String, _node: Dictionary) -> Array:
	var a := _input_or_zero(p_id, 0)
	var b := _input_or_zero(p_id, 1)
	var v := _var("float", "v_%s" % p_id, "%s + %s" % [a.e, b.e])
	return [_rec(v, "(%s + %s)" % [a.m, b.m], a.mb + b.mb, a.amp + b.amp, a.dev + b.dev, sqrt(a.slope * a.slope + b.slope * b.slope), maxf(a.lam, b.lam))]


func _emit_multiply(p_id: String, _node: Dictionary) -> Array:
	var a := _input_or_zero(p_id, 0)
	var b := _input_or_zero(p_id, 1)
	var v := _var("float", "v_%s" % p_id, "%s * %s" % [a.e, b.e])
	return [_rec(v, "(%s * %s)" % [a.m, b.m], a.mb * b.mb, a.amp * b.amp, a.amp * b.amp + a.mb * b.mb, a.amp * b.slope + b.amp * a.slope, maxf(a.lam, b.lam))]


func _emit_constant(p_id: String, p_node: Dictionary) -> Array:
	var value: float = Kinds.param(p_node, "value")
	var u := _uniform("float", "u_%s_v" % p_id, value)
	return [_rec(u, u, absf(value), absf(value), 0.0, 0.0, 0.0)]


# One segment of libnoise's cubic at a in [0, 1] (the GLSL curve_<id> writes the same).
static func _cubic(p_p: float, p_q: float, p_r: float, p_s: float, p_a: float) -> float:
	return ((p_p * p_a + p_q) * p_a + p_r) * p_a + p_s


# The curve's points [[h, R(h)], ...] (h increasing) as the engine's: C = R - h through libnoise's cubic (make_curve,
# src/climate_shape.cpp), constant past the ends, so R follows h there.
static func curve_segments(p_points: Array) -> Dictionary:
	var pts := p_points.duplicate(true)
	pts.sort_custom(func(a, b): return a[0] < b[0])
	var n := mini(pts.size(), CURVE_POINTS)
	var x := PackedFloat32Array()
	var segs := {"p": PackedFloat32Array(), "q": PackedFloat32Array(), "r": PackedFloat32Array(), "s": PackedFloat32Array(), "inv": PackedFloat32Array()}
	x.resize(CURVE_POINTS)
	for key in segs:
		segs[key].resize(CURVE_POINTS)
	var y := func(i: int) -> float:
		var pt: Array = pts[clampi(i, 0, n - 1)]
		return pt[1] - pt[0]
	for i in n:
		x[i] = pts[i][0]
	for i in n - 1:
		var n0: float = y.call(i - 1)
		var n1: float = y.call(i)
		var n2: float = y.call(i + 1)
		var n3: float = y.call(i + 2)
		var p := (n3 - n2) - (n0 - n1)
		var q := (n0 - n1) - p
		var inv := 1.0 / maxf(x[i + 1] - x[i], 1e-3)
		segs.p[i] = p
		segs.q[i] = q
		segs.r[i] = n2 - n0
		segs.s[i] = n1
		segs.inv[i] = inv
	return {"n": n, "x": x, "segs": segs, "first": y.call(0) if n > 0 else 0.0, "last": y.call(n - 1) if n > 0 else 0.0}


# R(h) of curve_segments' result, as the shader's curve_<id> (the editor draws it).
static func curve_eval(p_c: Dictionary, p_h: float) -> float:
	if p_c.n < 2:
		return p_h
	if p_h <= p_c.x[0]:
		return p_h + p_c.first
	for i in p_c.n - 1:
		if p_h < p_c.x[i + 1]:
			var a: float = (p_h - p_c.x[i]) * p_c.segs.inv[i]
			return p_h + _cubic(p_c.segs.p[i], p_c.segs.q[i], p_c.segs.r[i], p_c.segs.s[i], a)
	return p_h + p_c.last


func _emit_curve(p_id: String, p_node: Dictionary) -> Array:
	var h := _input_or_zero(p_id, 0)
	var rec := _curve_function(p_id, p_node, h)
	var v := _var("float", "v_%s" % p_id, "curve_%s(%s)" % [p_id, h.e])
	rec.e = v
	return [rec]


# A Height curve's GLSL function curve_<id>(h) and uniforms, and the record of its output on input p_h (its bounds;
# e: the call on p_h's expression).
func _curve_function(p_id: String, p_node: Dictionary, p_h: Dictionary) -> Dictionary:
	var h := p_h
	var points: Array = Kinds.param(p_node, "points")
	if points.size() > CURVE_POINTS:
		warnings.append("A Height curve has %d points: the preview takes the first %d." % [points.size(), CURVE_POINTS])
	var c := curve_segments(points)
	var pre := "u_%s" % p_id
	_uniform("int", pre + "_n", c.n)
	decls.append("uniform float %s_x[%d];" % [pre, CURVE_POINTS])
	uniforms[pre + "_x"] = c.x
	for key in c.segs:
		decls.append("uniform float %s_%s[%d];" % [pre, key, CURVE_POINTS])
		uniforms["%s_%s" % [pre, key]] = c.segs[key]
	_uniform("float", pre + "_first", c.first)
	_uniform("float", pre + "_last", c.last)
	functions.append((
		"float curve_{id}(float h) {\n"
		+ "\tif ({u}_n < 2) {\n\t\treturn h;\n\t}\n"
		+ "\tif (h <= {u}_x[0]) {\n\t\treturn h + {u}_first;\n\t}\n"
		+ "\tfor (int i = 0; i + 1 < {u}_n; i++) {\n"
		+ "\t\tif (h < {u}_x[i + 1]) {\n"
		+ "\t\t\tfloat a = (h - {u}_x[i]) * {u}_inv[i];\n"
		+ "\t\t\treturn h + ((({u}_p[i] * a + {u}_q[i]) * a + {u}_r[i]) * a + {u}_s[i]);\n"
		+ "\t\t}\n\t}\n"
		+ "\treturn h + {u}_last;\n}\n").format({"id": p_id, "u": pre}))
	# R only meets the heights its input reaches (within +-h.amp): its bound and slope there, not past (an imported
	# curve's points span the table's whole field range, kilometres beyond the biome's relief).
	var hi := maxf(h.amp, 1.0)
	curve_in[p_id] = hi
	var bound := 0.0
	var slope := 0.0
	var prev := 0.0
	for j in 257:
		var x := lerpf(-hi, hi, j / 256.0)
		var dr := curve_eval(c, x) - x
		bound = maxf(bound, absf(dr))
		if j > 0:
			slope = maxf(slope, absf(dr - prev) / (2.0 * hi / 256.0))
		prev = dr
	return _rec("curve_%s(%s)" % [p_id, h.e], h.m, h.mb + bound, h.amp + bound, h.dev + 2.0 * bound, h.slope * (1.0 + slope), h.lam)


# Holes: air where the field is above the threshold, inside the depth band below the surface. The field is scaled to
# metres (its wavelength / NOISE_SLOPE, so its slope is about 1) and pushed below the surface outside the band.
func _emit_carve(p_id: String, p_node: Dictionary) -> Array:
	var f := _input_or_zero(p_id, 0)
	if not CARVE_WARNING in warnings:
		warnings.append(CARVE_WARNING)
	var t: float = Kinds.param(p_node, "threshold")
	var scale: float = maxf(f.lam, 1.0) / NOISE_SLOPE / maxf(f.amp, 1e-3)
	var pre := "u_%s" % p_id
	_uniform("float", pre + "_t", t)
	_uniform("float", pre + "_s", scale)
	_uniform("float", pre + "_top", Kinds.param(p_node, "top_m"))
	_uniform("float", pre + "_bottom", Kinds.param(p_node, "bottom_m"))
	_uniform("float", pre + "_fade", Kinds.param(p_node, "fade_m"))
	_uniform("float", pre + "_big", 2.0 * scale * (f.amp + absf(t)) + 1.0)
	var code := (
		"\t{ float depth = h - p.y;\n"
		+ "\t  float band = smoothstep({u}_top - {u}_fade, {u}_top, depth) * (1.0 - smoothstep({u}_bottom, {u}_bottom + {u}_fade, depth));\n"
		+ "\t  d = max(d, ({e} - {u}_t) * {u}_s - (1.0 - band) * {u}_big); }").format({"u": pre, "e": f.e})
	return [{"code": code}]


# A colour (#rrggbb or a Color) as linear RGB.
static func _linear(p_color: Variant) -> Vector3:
	var c: Color = (p_color if p_color is Color else Color(String(p_color))).srgb_to_linear()
	return Vector3(c.r, c.g, c.b)


# --- the world: every biome in one shader ------------------------------------------------------------------------

# Compile.new().run_world(world): the climate's fields (their octaves rows of the warp table, three to a W slot), the
# weights (World.weights in float: the kernels' centres and spans uniforms, so the chart moves them without a
# recompile), each biome's graph as its height before its curve (its node ids prefixed b<i>_, notes shared), then the
# engine's blend: H = sum w_b H_b, h = H + sum w_b (R_b(H) - H). With a ground term in the fields (colder higher up),
# the weights are read twice: without it, then with it on that H (the engine reads them from the coarser octaves). A
# biome's graph is skipped where its weight is 0 (without ground terms). The colours blend by weight.
func run_world(p_world: Dictionary, p_exaggeration: float = 1.0) -> Dictionary:
	exaggeration = p_exaggeration
	voxel_m = p_world.voxel_m
	lods = p_world.lods
	climate_cfg = Kinds.climate_warp(p_world)
	_set_origin(p_world)
	var nf: int = p_world.fields.size()
	if nf > ShaderLib.MAX_FIELDS:
		errors.append("%d fields: the preview takes %d." % [nf, ShaderLib.MAX_FIELDS])
	var n: int = p_world.biomes.size()
	if n == 0:
		errors.append("The world has no biome.")
	elif n > ShaderLib.MAX_BIOMES:
		errors.append("%d biomes: the preview takes %d." % [n, ShaderLib.MAX_BIOMES])
		n = ShaderLib.MAX_BIOMES
	place_count = n * nf
	warps = (nf + 2) / 3
	for i in nf:
		var seed := World.field_seed(p_world.fields[i])
		for oc in World.field_octaves(p_world, p_world.fields[i]):
			warp_rows.append({"octave": Octaves.field_octave(seed, oc.k), "k": oc.k, "slot": i / 3, "src": NO_SRC, "axis": i % 3, "amp": oc.amp, "wm": 0.0, "st": 1.0})
	var ground_terms := World.has_ground_terms(p_world)
	body.append("\tfloat FL[%d];" % maxi(nf, 1))
	_world_fields(p_world, "0.0")
	_world_weights(p_world, n, "1")
	var hs := []
	var h_amp := 0.0
	var h_slope := 0.0
	for b in n:
		var r := _world_biome(p_world, b, not ground_terms)
		hs.append(r)
		h_amp = maxf(h_amp, r.pre.amp)
		h_slope = maxf(h_slope, r.pre.slope)
	# H is a weighted mean of the biomes' heights (within the largest's bound): a biome's curve meets all of it.
	var hmax := h_amp
	var slope := h_slope
	for r in hs:
		if r.curve != "":
			var at_h: Dictionary = r.pre.duplicate()
			at_h.amp = h_amp
			at_h.slope = h_slope
			var rec := _curve_function(r.curve_id, r.curve_node, at_h)
			hmax = maxf(hmax, rec.amp)
			slope = maxf(slope, rec.slope)
	var sum := func(p_w: String, p_term: Callable) -> String:
		var terms := PackedStringArray()
		for b in n:
			terms.append("%s%d * %s" % [p_w, b, p_term.call(b)])
		return " + ".join(terms) if not terms.is_empty() else "0.0"
	body.append("\tfloat H = %s;" % sum.call("wk1_", func(b): return "mb_%d" % b))
	var w := "wk1_"
	if ground_terms:
		_world_fields(p_world, "H")
		_world_weights(p_world, n, "2")
		body.append("\tH = %s;" % sum.call("wk2_", func(b): return "mb_%d" % b))
		w = "wk2_"
	var curves := PackedStringArray()
	for b in n:
		if hs[b].curve != "":
			curves.append("%s%d * (%s(H) - H)" % [w, b, hs[b].curve])
	body.append("\tfloat h = (H%s) * u_exag;" % ("" if curves.is_empty() else " + " + " + ".join(curves)))
	body.append("\tc_flat = %s;" % sum.call(w, func(b): return "u_b%d_flat" % b))
	body.append("\tc_steep = %s;" % sum.call(w, func(b): return "u_b%d_steep" % b))
	body.append("\tc_map = %s;" % sum.call(w, func(b): return "u_b%d_dbg" % b))
	body.append("\tc_fields = vec4(%s, %s, h, 0.0);" % (["FL[u_cx]", "FL[u_cy]"] if nf > 0 else ["0.0", "0.0"]))
	body.append("\treturn p.y - h;")
	_table()
	uniforms.merge(places(p_world), true)
	# The weights change over the fields' wavelengths (100s of km): a few percent on the biomes' own slopes.
	var hm: float = exaggeration * hmax + 1.0
	return _result(hm, sqrt(1.0 + pow(exaggeration * slope * 1.1, 2.0)))


# The fields' values FL[i] from their W slots, then their terms in list order (a ground term on p_ground, metres).
func _world_fields(p_world: Dictionary, p_ground: String) -> void:
	for i in p_world.fields.size():
		body.append("\tFL[%d] = W[%d].%s;" % [i, i / 3, "xyz"[i % 3]])
		for t in World.field_terms(p_world, i):
			if t.source < 0 and p_ground == "0.0":
				continue
			body.append("\tFL[%d] += %s * %s;" % [i, _num(t.coef), p_ground if t.source < 0 else "FL[%d]" % t.source])


# The weights wk<pass>_<b>: each biome's kernel over the fields it names, the default's (1 - sum)^2, over their sum.
func _world_weights(p_world: Dictionary, p_n: int, p_pass: String) -> void:
	var nf: int = p_world.fields.size()
	var names := World.field_names(p_world)
	body.append("\tfloat ks%s = 0.0;" % p_pass)
	for b in p_n - 1:
		# In field order, as the engine multiplies them (its axes sorted by field).
		var factors := PackedStringArray(["1.0"])
		for i in names.size():
			if p_world.biomes[b].at.has(names[i]):
				factors.append("kfac(FL[%d], u_atc[%d], u_ati[%d])" % [i, b * nf + i, b * nf + i])
		body.append("\tfloat wk%s_%d = %s;\n\tks%s += wk%s_%d;" % [p_pass, b, " * ".join(factors), p_pass, p_pass, b])
	var d := p_n - 1
	body.append("\tfloat wk%s_%d = ks%s < 1.0 ? (1.0 - ks%s) * (1.0 - ks%s) : 0.0;" % [p_pass, d, p_pass, p_pass, p_pass])
	body.append("\tfloat wt%s = ks%s + wk%s_%d;" % [p_pass, p_pass, p_pass, d])
	for b in p_n:
		body.append("\twk%s_%d /= wt%s;" % [p_pass, b, p_pass])


# Biome p_b's graph, its node ids prefixed: its height before its curve as mb_<b> (inside an if on its first weight when
# p_gate), its colours' uniforms; returns {curve (its curve's function name, or ""), curve_id, curve_node, pre (the
# record of its height before the curve)}: run_world writes the curve, bounded over the blended height.
func _world_biome(p_world: Dictionary, p_b: int, p_gate: bool) -> Dictionary:
	var biome: Dictionary = p_world.biomes[p_b]
	model = _prefixed(biome.model, "b%d_" % p_b)
	for l in model.links:
		incoming["%s:%d" % [l.to, int(l.to_port)]] = [l.from, int(l.from_port)]
	var out_id := Kinds.first_of(model, "output")
	body.append("\tfloat mb_%d = 0.0;" % p_b)
	var start := body.size()
	var pre := _zero()
	var curve_id := ""
	var flat := "#5c7a3a"
	var steep := "#6b625a"
	if out_id == "":
		warnings.append("Biome %s has no Output: flat." % biome.name)
	else:
		flat = Kinds.param(model.nodes[out_id], "flat")
		steep = Kinds.param(model.nodes[out_id], "steep")
		var src = incoming.get("%s:0" % out_id)
		if src != null and model.nodes.has(src[0]) and model.nodes[src[0]].kind == "curve":
			pre = _input_or_zero(src[0], 0)
			curve_id = src[0]
		else:
			pre = _input_or_zero(out_id, 0)
	body.append("\tmb_%d = %s;" % [p_b, pre.e])
	if p_gate:
		var inner := body.slice(start)
		body.resize(start)
		body.append("\tif (wk1_%d > 0.0) {" % p_b)
		for line in inner:
			body.append("\t" + line)
		body.append("\t}")
	_uniform("vec3", "u_b%d_flat" % p_b, _linear(flat))
	_uniform("vec3", "u_b%d_steep" % p_b, _linear(steep))
	_uniform("vec3", "u_b%d_dbg" % p_b, _linear(World.debug_color(biome)))
	return {"curve": "curve_%s" % curve_id if curve_id != "" else "", "curve_id": curve_id, "curve_node": model.nodes.get(curve_id, {}), "pre": pre}


# The biomes' places (centre, 1 / span per field, index biome * fields + field) as the uniforms u_atc / u_ati: what a
# drag on the chart changes (Compile.places(world) without a recompile).
static func places(p_world: Dictionary) -> Dictionary:
	var c := PackedFloat32Array()
	var inv := PackedFloat32Array()
	var nf: int = p_world.fields.size()
	var n := mini(p_world.biomes.size(), ShaderLib.MAX_BIOMES)
	c.resize(maxi(n * nf, 1))
	inv.resize(maxi(n * nf, 1))
	var names := World.field_names(p_world)
	for b in n:
		var at: Dictionary = p_world.biomes[b].at
		for f in at:
			var i := names.find(f)
			if i >= 0:
				c[b * nf + i] = at[f][0]
				inv[b * nf + i] = 1.0 / maxf(at[f][1], 1e-6)
	return {"u_atc": c, "u_ati": inv}


# A copy of a biome model with its node ids prefixed (several biomes in one shader).
static func _prefixed(p_model: Dictionary, p_pre: String) -> Dictionary:
	var m := p_model.duplicate()
	m.nodes = {}
	for id in p_model.get("nodes", {}):
		m.nodes[p_pre + id] = p_model.nodes[id]
	m.links = p_model.get("links", []).map(func(l): return {"from": p_pre + l.from, "from_port": l.from_port, "to": p_pre + l.to, "to_port": l.to_port})
	return m


static func _num(p_v: float) -> String:
	var t := String.num(p_v, 9)
	return t if "." in t or "e" in t else t + ".0"
