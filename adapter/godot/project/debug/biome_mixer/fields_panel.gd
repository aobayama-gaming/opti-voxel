extends VBoxContainer
## The World mode's climate and planet: each field's bands (an octave's wavelength and amplitude, or a range's from, to,
## amplitude and persistence: the format's, climate_parse.cpp) and its mean and poles (a sphere's), fields added and
## removed, a planet's radius, its seed, its sea (on or off, its level). It edits the world's "fields" (the climate the
## file gets, World.file_of), its "shape", "seed" and "sea"; the engine snaps a band to its octave (a
## power of two voxels of the world's engine), shown beside it. A field's wavelengths set how far apart its biomes lie.

const World := preload("res://debug/biome_mixer/world.gd")
const EngineInfo := preload("res://debug/biome_mixer/engine_info.gd")

signal changed(p_names: bool)  # an edit; p_names: a field added or removed (the chart's axes and the places change)
signal planet_changed  # the planet's radius, seed, sea or a field's offsets, debounced (the views frame it again)

# A new field's band: an octave of a few hundred metres at 5 cm (k = 14), biomes a few minutes' run apart.
const NEW_BAND := {"wavelength_m": 819.2, "amp": 1.0}
# A band's keys in the order they show, and each one's spin range.
const BAND_KEYS := {"wavelength_m": [0.1, 1e6, 0.1], "from_m": [0.1, 1e6, 0.1], "to_m": [0.1, 1e6, 0.1], "amp": [-10.0, 10.0, 0.01], "persistence": [0.0, 2.0, 0.01]}

var world := {}
var box := VBoxContainer.new()
# A planet edit (radius, seed, sea) applies once the spin rests: each one recompiles the shaders (held arrows would
# queue one per step).
var planet_wait := Timer.new()
# The sea last switched off (its water and shore), back when it is switched on again.
var sea_off := {"level_m": 0.0}


func _init() -> void:
	planet_wait.one_shot = true
	planet_wait.wait_time = 0.5
	planet_wait.timeout.connect(func() -> void: planet_changed.emit())
	add_child(planet_wait)
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.custom_minimum_size.y = 140
	box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(box)
	add_child(scroll)


func set_world(p_world: Dictionary) -> void:
	world = p_world
	_build()


func _build() -> void:
	for c in box.get_children():
		box.remove_child(c)
		c.queue_free()
	var row := HBoxContainer.new()
	if World.shape_kind(world) != "flat":
		row.add_child(_label("Radius m"))
		var r := _spin(float(world.shape.get("radius_m", 2000.0)), [10.0, 1e8, 1.0])
		r.tooltip_text = "The planet's radius (its tube's for a torus). A new radius compiles the terrain's shaders again: a pause of a second or two."
		r.value_changed.connect(func(v: float) -> void:
			world.shape.radius_m = v
			planet_wait.start())
		row.add_child(r)
	row.add_child(_label("Seed"))
	var s := _spin(float(world.get("seed", 0)), [0.0, 4294967295.0, 1.0])
	s.allow_greater = false
	s.tooltip_text = "The planet's seed: every note and field mixed with it (0: the table's own noises). Another seed, another planet from the same biomes."
	s.value_changed.connect(func(v: float) -> void:
		world.seed = int(v)
		planet_wait.start())
	row.add_child(s)
	box.add_child(row)
	_sea_row()
	for f in world.fields:
		_field_rows(f)
	var add := Button.new()
	add.text = "Add field"
	add.tooltip_text = "A new climate field (one band of %s m): the biomes' places can then name it" % NEW_BAND.wavelength_m
	add.pressed.connect(_add_field)
	box.add_child(add)


# The world's sea: on or off (WorldFile's "sea"; a graph's "sea" node needs one), its level above the base.
func _sea_row() -> void:
	var row := HBoxContainer.new()
	var on := CheckBox.new()
	on.text = "Sea"
	on.button_pressed = world.get("sea") is Dictionary
	on.tooltip_text = "A sea over the whole planet: what a graph's Sea level node reads, and no tree or rock grows under it (or on its shore)."
	on.toggled.connect(func(p_on: bool) -> void:
		if p_on:
			world.sea = sea_off.duplicate(true)
		else:
			sea_off = world.sea
			world.erase("sea")
		_build()
		planet_wait.start())
	row.add_child(on)
	if world.get("sea") is Dictionary:
		row.add_child(_label("level m"))
		var l := _spin(float(world.sea.get("level_m", 0.0)), [-1e5, 1e5, 0.1])
		l.allow_lesser = true
		l.tooltip_text = "The sea's level above the base (metres)."
		l.value_changed.connect(func(v: float) -> void:
			world.sea.level_m = v
			planet_wait.start())
		row.add_child(l)
	box.add_child(row)


func _field_rows(p_field: Dictionary) -> void:
	var head := HBoxContainer.new()
	var title := _label(String(p_field.get("name", "?")))
	title.custom_minimum_size.x = 110
	head.add_child(title)
	# Its offsets (the world's climate "<field>": {mean, poles}): poles only on a sphere (the engine refuses them elsewhere).
	for key in World.OFFSETS:
		if key == "poles" and World.shape_kind(world) != "sphere":
			continue
		head.add_child(_label(key))
		var o := _spin(float(p_field.get(key, 0.0)), [-10.0, 10.0, 0.01])
		o.allow_lesser = true
		o.tooltip_text = "Added to the field everywhere." if key == "mean" else "Times the latitude (0 at the equator, 1 at the poles): negative, colder poles."
		o.value_changed.connect(func(v: float) -> void:
			if v == 0.0:
				p_field.erase(key)
			else:
				p_field[key] = v
			planet_wait.start())
		head.add_child(o)
	var remove := Button.new()
	remove.text = "Remove field"
	remove.disabled = world.fields.size() <= 1
	remove.pressed.connect(_remove_field.bind(p_field))
	head.add_child(remove)
	box.add_child(head)
	var bands: Array = p_field.get("bands", [])
	for band in bands:
		var row := HBoxContainer.new()
		row.add_child(_label("    "))
		for key in BAND_KEYS:
			if not band.has(key):
				continue
			row.add_child(_label(key.trim_suffix("_m").replace("wavelength", "λ") + (" m" if key.ends_with("_m") else "")))
			var octave := _label(_octave(band, key))
			var s := _spin(float(band[key]), BAND_KEYS[key])
			s.value_changed.connect(func(v: float) -> void:
				band[key] = v
				octave.text = _octave(band, key)
				changed.emit(false))
			row.add_child(s)
			row.add_child(octave)
		var x := Button.new()
		x.text = "x"
		x.tooltip_text = "Remove this band"
		x.disabled = bands.size() <= 1
		x.pressed.connect(func() -> void:
			bands.erase(band)
			_build()
			changed.emit(false))
		row.add_child(x)
		box.add_child(row)
	var add := Button.new()
	add.text = "Add band"
	add.pressed.connect(func() -> void:
		bands.append(NEW_BAND.duplicate())
		p_field.bands = bands
		_build()
		changed.emit(false))
	var add_row := HBoxContainer.new()
	add_row.add_child(_label("    "))
	add_row.add_child(add)
	box.add_child(add_row)


# The octave the engine reads a wavelength key as: its nearest power of two voxels of the world's engine.
static func _octave(p_band: Dictionary, p_key: String) -> String:
	if not p_key.ends_with("_m"):
		return ""
	var k := EngineInfo.octave_of(float(p_band[p_key]))
	return "(k %d: %s m)" % [k, String.num(EngineInfo.voxel_m * pow(2.0, k), 1)]


func _add_field() -> void:
	var names := World.field_names(world)
	var i := names.size()
	while names.has("field%d" % i):
		i += 1
	world.fields.append({"name": "field%d" % i, "bands": [NEW_BAND.duplicate()]})
	_build()
	changed.emit(true)


# p_field out of the climate and of every place; a biome left with no field (not the default) takes the first one.
func _remove_field(p_field: Dictionary) -> void:
	var i: int = world.fields.find(p_field)
	world.fields.erase(p_field)
	# The later fields' terms that read it go (a term names an earlier field by name; the format's source).
	for f in world.fields.slice(i):
		if f.has("terms"):
			f.terms = f.terms.filter(func(t): return String(t.get("source", "")) != String(p_field.get("name", "")))
			if f.terms.is_empty():
				f.erase("terms")
	var first: String = World.field_names(world)[0]
	for b in world.biomes.size() - 1:
		var at: Dictionary = world.biomes[b].at
		at.erase(String(p_field.get("name", "")))
		if at.is_empty():
			at[first] = [0.0, 0.3]
	_build()
	changed.emit(true)


static func _label(p_text: String) -> Label:
	var l := Label.new()
	l.text = p_text
	return l


static func _spin(p_value: float, p_range: Array) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = p_range[0]
	s.max_value = p_range[1]
	s.step = p_range[2]
	s.allow_greater = true
	s.value = p_value
	s.custom_minimum_size.x = 84
	return s
