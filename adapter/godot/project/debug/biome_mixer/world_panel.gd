extends VBoxContainer
## The world mode's side: the climate chart (two fields as axes, the biomes' boxes), the biome list and the selected
## biome's place: for each field, whether it names it, its centre and its span. The world is the truth; this only
## shows and edits its biomes' "at".

const World := preload("res://debug/biome_mixer/world.gd")
const Chart := preload("res://debug/biome_mixer/chart.gd")
const Octaves := preload("res://debug/biome_mixer/octaves.gd")

signal placed  # a place changed (a biome's, or a field named or dropped)
signal edit_notes(p_biome: int)  # open the biome's notes (the biome mode)
signal axes_changed  # the chart's fields changed: new samples

var world := {}
var chart := Chart.new()
var list := ItemList.new()
var axis_x := OptionButton.new()
var axis_y := OptionButton.new()
var place_box := VBoxContainer.new()
var spins := {}  # field -> [centre, span] SpinBoxes of the shown biome
var hint := Label.new()


func _init() -> void:
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	var axes := HBoxContainer.new()
	axes.add_child(_label("x"))
	axes.add_child(axis_x)
	axes.add_child(_label("  y"))
	axes.add_child(axis_y)
	add_child(axes)
	for ob in [axis_x, axis_y]:
		ob.item_selected.connect(func(_i: int) -> void: _axes())
	chart.size_flags_stretch_ratio = 2.0
	add_child(chart)
	var below := HBoxContainer.new()
	below.size_flags_vertical = Control.SIZE_EXPAND_FILL
	add_child(below)
	list.custom_minimum_size = Vector2(160, 120)
	list.size_flags_vertical = Control.SIZE_EXPAND_FILL
	list.item_selected.connect(func(i: int) -> void: select(i))
	list.item_activated.connect(func(i: int) -> void: edit_notes.emit(i))
	list.tooltip_text = "Click: select its box; double-click: edit its notes"
	below.add_child(list)
	var scroll := ScrollContainer.new()
	scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(place_box)
	place_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	below.add_child(scroll)
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.modulate = Color(1, 1, 1, 0.7)
	hint.text = ("A biome is whole within %.2f of its span on each field it names (the filled box), gone at its span (the outline); overlaps blend by themselves, the last biome fills the rest. Drag a box to move it, its edge to resize.") % Octaves.core
	add_child(hint)
	chart.placed.connect(func(b: int) -> void:
		if b == int(world.get("current", -1)):
			_sync_spins(b)
		placed.emit())
	chart.selected.connect(func(b: int) -> void: select(b))


func set_world(p_world: Dictionary) -> void:
	world = p_world
	chart.set_world(world)
	var v: Dictionary = world.view
	var names := World.field_names(world)
	for ob in [axis_x, axis_y]:
		ob.clear()
		for n in names:
			ob.add_item(n)
	if not names.is_empty():
		chart.fx = clampi(int(v.chart_x), 0, names.size() - 1)
		chart.fy = clampi(int(v.chart_y), 0, names.size() - 1)
		axis_x.select(chart.fx)
		axis_y.select(chart.fy)
	list.clear()
	for b in world.biomes.size():
		var biome: Dictionary = world.biomes[b]
		var i := list.add_item(biome.name + ("  (default)" if b == world.biomes.size() - 1 else ""))
		list.set_item_custom_fg_color(i, World.debug_color(biome).lightened(0.3))
	select(clampi(int(world.get("current", 0)), 0, world.biomes.size() - 1))


func select(p_b: int) -> void:
	if p_b < 0 or p_b >= world.biomes.size():
		return
	world.current = p_b
	list.select(p_b)
	chart.current = p_b
	chart.queue_redraw()
	_show_place(p_b)


func _axes() -> void:
	chart.fx = axis_x.selected
	chart.fy = axis_y.selected
	var v: Dictionary = world.view
	v.chart_x = chart.fx
	v.chart_y = chart.fy
	chart.queue_redraw()
	axes_changed.emit()


# Biome p_b's place, a row per field: names it, centre, span.
func _show_place(p_b: int) -> void:
	for c in place_box.get_children():
		place_box.remove_child(c)
		c.queue_free()
	var biome: Dictionary = world.biomes[p_b]
	var title := Label.new()
	title.text = biome.name
	place_box.add_child(title)
	if p_b == world.biomes.size() - 1:
		var l := _label("The default: it has no place, it fills what the other biomes leave (the last in the list).")
		l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		place_box.add_child(l)
		return
	var at: Dictionary = biome.at
	spins.clear()
	for f in World.field_names(world):
		var row := HBoxContainer.new()
		var on := CheckBox.new()
		on.text = f
		on.button_pressed = at.has(f)
		on.custom_minimum_size.x = 120
		on.tooltip_text = "Whether the biome cares about %s (unticked: any %s)" % [f, f]
		row.add_child(on)
		var c := _spin(at[f][0] if at.has(f) else 0.0, -2.0, 2.0)
		var s := _spin(at[f][1] if at.has(f) else 0.3, 0.01, 4.0, false)
		c.editable = at.has(f)
		s.editable = at.has(f)
		spins[f] = [c, s]
		row.add_child(_label("centre"))
		row.add_child(c)
		row.add_child(_label("span"))
		row.add_child(s)
		on.toggled.connect(func(p_on: bool) -> void:
			if p_on:
				at[f] = [c.value, s.value]
			elif at.size() == 1:
				# Only the default has no place: a biome without one would be whole everywhere.
				on.set_pressed_no_signal(true)
				return
			else:
				at.erase(f)
			c.editable = p_on
			s.editable = p_on
			chart.queue_redraw()
			placed.emit())
		var set_at := func(_v: float) -> void:
			if at.has(f):
				at[f] = [c.value, s.value]
				chart.queue_redraw()
				placed.emit()
		c.value_changed.connect(set_at)
		s.value_changed.connect(set_at)
		place_box.add_child(row)
	var notes := Button.new()
	notes.text = "Edit its notes"
	notes.pressed.connect(func() -> void: edit_notes.emit(p_b))
	place_box.add_child(notes)


# The shown biome's spin boxes to its place (a drag on the chart moved it), without their signals.
func _sync_spins(p_b: int) -> void:
	var at: Dictionary = world.biomes[p_b].at
	for f in spins:
		if at.has(f):
			spins[f][0].set_value_no_signal(at[f][0])
			spins[f][1].set_value_no_signal(at[f][1])


static func _label(p_text: String) -> Label:
	var l := Label.new()
	l.text = p_text
	return l


static func _spin(p_value: float, p_min: float, p_max: float, p_below: bool = true) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = p_min
	s.max_value = p_max
	s.step = 0.001
	s.allow_greater = true
	s.allow_lesser = p_below
	s.value = p_value
	s.custom_minimum_size.x = 84
	return s
