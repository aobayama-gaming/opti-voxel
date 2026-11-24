# Climate

The climate in three parts: the **model** (what the ground is at a point, as math), the **implementation** (how the
cascade computes it, CPU and GPU), and the **measurements**. Then the biome mixer, the tool that designs a world's
biomes. The evaluation is cascade v2 (`docs/cascade-v2.md`): one graph, each node computed at its level.

## Model

A climate (`src/climate.hpp`, a table: format above `Climate::parse_layout` in `src/climate_parse.cpp`; three files
give it: `engine.json` the voxel and LODs, `climate.json` the fields that place biomes, `world.json` the biomes, merged
by `WorldFile` into the table `Terrain::compile` reads, `docs/terrain.md`) shapes **one ground**. Each biome is a
**graph**: the notes it mixes (each one octave k with its own seed, amplitude and stretch, read displaced by its warp),
and any nodes over them (shapes, multifractals, mixes, curves, min / max, smooth min / max, constants, fields, `up`). A
biome lists exactly what it uses: no engine default notes. The compiler (`src/graph_compile.{hpp,cpp}`) turns the
fields, the weights and every biome's graph into one `Terrain::LevelProgram`: a note is one node whatever biome mixes
it (identity: octave, seed, stretch, warp), so a biome more costs only the nodes it adds. A world names its biomes
(without, the cascade refuses it); the bare worlds name the **template ground** (`worlds/ground.biomes.json`,
`WorldFile::GROUND_BIOMES`, the mixer's start for a new biome too): seed 0xEA27, a note per octave from 0.4 m (k = 3 at
5 cm) to 26.2 km (k = 19 at 5 cm), each 1/8 of its wavelength, mixed as "ground"; one file at every voxel size (at
0.5 m its notes take k = 0 .. 16, the three finer than 8 voxels playing nothing). Notes and fields play from k = 3 (8
voxels) to lod_count + 4 (`Ground::k_max`, the cascade's top): one outside plays nothing, so a world reads at any voxel
and LOD count; the fields
placing the biomes (temperature, humidity, wind: 6.5-26 km, under the top LOD like the ground: a world to walk, not
continents) are macro inputs until the planned macro engine computes them.

1. **Fields** (`Field`): a few slow scalars, each a sum of plain perlin octaves of its own seed, plus linear terms on
   earlier fields (`coef * an earlier field`, in list order): temperature, humidity, wind. The v1 term on the ground
   is gone: the ground depends on the weights, which depend on the fields (the parser says so). A field's `mean` adds a
   constant (a term only when it isn't 0, so the pins don't move) and its `poles` times the latitude: the op `POLE`,
   n_y² of a sphere's up (0 on the equator, 1 at the poles; `BaseSurface::pole_form`, GLSL `base_pole`, op for op).
   `POLE` sits on a fixed level, the field's, or the coarsest level where the base's radius form is exact
   (`Ground::pole_level`, `BaseSurface::pole_lod`) if that is finer, so a point gets the same bits from every LOD. Poles
   need a sphere: the flat base has none and a torus's latitude (the tube's angle) isn't written yet (refused). A planet
   file writes both as per-field offsets (`"climate": {"fields": [...], "temperature": {"mean", "poles"}}`,
   `docs/terrain.md`).
   **Seeds:** the planet's `seed` (`Ground::seed`) mixes every note's and field's seed (`Climate::mix_seed`:
   `hash3(seed, planet, PLANET_SEED_SALT)`), 0 the identity, so one biome file grows another ground on another planet
   and every world without a seed keeps its bits. Instance scatter seeds aren't mixed yet (a planet's trees stand where
   another's would). **The `sea` node:** the world's sea level (`sea.level_m`) as a constant of LOD-0 voxels, for a
   graph to stand under; a world without a sea refuses it.
2. **Weights**: a biome is a place in the fields' space, `"at"`: a centre and a span on each field it names
   (`[centre, span]`, or `{min, max}`: their middle and half their gap). Its kernel `k_b` is the product over those
   fields of `smoothstep(clamp((1 - u^2) * 2, 0, 1))`, `u = (f - centre) / span` (`Climate::axis_factor`): on each
   field whole within 0.71 of its span, gone at its span, each field's span its own (naming one more field doesn't
   shrink the others). The last biome, the default, has no place: `k_d = (1 - sum k)^2` where the kernels sum under 1
   (`Climate::fill`). The weights are `w_b = k_b / S`, `S = sum k + k_d >= 3/4` (`blend_rcp`, `blend_weight`):
   overlapping biomes blend by themselves, no borders to match and no priority. `1 / S` is `det_rsqrt(S)^2` and one
   Newton step (exact 1 at S = 1: a biome alone at its centre is whole), each weight capped at 1.
3. **The ground** at a point is `sum_b w_b * out_b`, `out_b` biome b's graph output; each graph reads its own notes:
   - a note `n_k` times its `"amp_m"` (as written, on any base), read at the point moved by its warp: a warp node
     whose input is any node strictly coarser than the note (the gate), read with its notes' seed, seed + 1 and seed + 2
     for x, y and z (libnoise's turbulence), or its `x`, `y`, `z` ports; a `"stretch_y"` reads it squeezed along the
     base's up (at most 2^(k - 3) on octave k: more is refused, never clamped);
   - a shape `phi(n) = c0 + c1 n + c2 |n| + c3 n^2` (plain, billow, ridged, square) on any node, inserted without its
     mean on a note (`|n|` is `Cascade::smooth_abs`, `sqrt(n^2 + 0.01) - 0.1`: with the bare kink the Hermite
     subdivision rang along the lattice axes, ridged gradient channels 21% off the values' finite differences against
     2.5% plain);
   - a multifractal `{in, parent, low_m, high_m}` = `multiply(in, ramp(parent))`, the ramp a two-point clamped curve
     from (low_m, 0) to (high_m, 1) (`low_m` above `high_m` inverts it); the parent is any node (Musgrave's chain: each
     octave with its next coarser one);
   - a mix (its rows' inputs at their volumes), add, subtract, multiply, min, max, smooth min / max (Quilez's
     polynomial, over `k_m`), curves (`R(h)` through the points `[h, R(h)]`: libnoise's cubic of `C = R - h`, constant
     past the ends), constants, fields and `up` (the height above the base: `up_value`, exact; a cave is
     `smin(mix, up - C)`, `docs/caves.md`).
   Blend factoring (specialised only) takes a node every biome's output has alike out of the weights' products:
   the same ground up to the weights' normalisation error.
4. **The pick**: a biome drawn by the weights with a hash of the absolute voxel (`Climate::pick_of`, a dithered
   pick) gives the sample its material (`biome_material`: the procedural ground in the biome's colours). Where two
   biomes blend 70 / 30, 70% of the samples are the first: the ground's colours mix over the blend like grass
   thinning into sand, at the cost of one material per sample (smooth colour blends measured +2.2 ms,
   `docs/materials.md`). Was the largest weight, which cut every colour border sharp even where the heights blend.

The **LOD rule**: a node's level is the level of the finest note it reads (note k on Λ_{k-3}, 8 samples per
wavelength); it is computed on that lattice from its inputs there and Hermite-subdivided below, the pick copied from
the floor parent. So a nonlinear node is defined at its level (a crease on Λ_m rounded at its spacing), the weights
are computed on their fields' coarse lattices and carried down, and every value is a pure function of (point, LOD):
the same bits from any chunk or point query.

What the model doesn't have, on purpose: per-landform noise fields, regions, generic height terms on fields, terraces,
cell noise (the landform fitting's machinery, removed in the climate declutter: one way to shape the ground, the
biomes playing its notes), and anything macro (continents, the climate axes as smooth per-region values: the planned
macro layer in double on the CPU, `BaseSurface`'s region machinery).

## Implementation

- **Parse** (`climate_parse.cpp`, `Climate::parse_layout`): metres to octaves (a field band or a note is one octave,
  its wavelength's nearest power of two of voxels: a note or a field octave finer than 8 voxels or past the LOD
  count's top plays nothing, a single `wavelength_m` or a `from_m` / `to_m` range alike, so one table covers 5 cm and
  0.5 m, 15 and 25 LODs; past the noise's k = 62 is refused), the fields and biomes (places, looks, contents) validated: at most
  `MAX_FIELDS` (24) fields and `MAX_BIOMES` (16) biomes, names unique identifiers, a field's octave listed once and its
  amplitudes in the noise's exact range, terms on earlier fields, the default biome last and alone without a place.
  Each biome's graph is handed to the compiler as JSON; `graph_kinds` (the node catalogue: each kind's keys) is the
  parser's and the mixer's.
- **Compile** (`graph_compile.cpp`, `Terrain::compile`): the graph read back from each output (acyclic check, the
  warp's gate, the sugar expanded), the fields as LINs of their bands and terms, the weights (`weight_nodes`: each
  kernel the product of its `AXIS` factors, `FILL`, `RCP`, `NORM`), the blend and the pick; then constants folded,
  nested LINs merged, linear crossings carried as one channel (a plain fBm is one channel top to bottom) and, specialised,
  blend factoring. Then the bounds (`bound_program`: each node's value range, the output's step per LOD, the skirts'
  depth) and the validation: the output within the base's limits (`Ground::reach_cap`: `BaseSurface::ALT_LIMIT` either
  side, a carve's top only; `Ground::floor`: on a planet its low side above the centre; the blend's range is the hull of its biomes' outputs, its weights summing to 1, not their sum), a curve's slope within `MAX_CURVE_SLOPE` (4), a warp within `WARP_LIMIT`. Refusals name the biome and node.
- **CPU** (`level_program.cpp`): `Terrain::run` / `Terrain::point` on the op table's CPU forms, through
  `SDFCascade::channels_at` on every path (chunk samples, point items, `ground_along`, the report); BiomePick,
  CloudClimate and ClimateReport read the program's `fields` and `weights` nodes (`Terrain::point_nodes`, which
  subdivides a field the program carries no lower than its own level).
- **GPU** (`level_glsl.cpp`): the same ops' GLSL forms, op for op (`precise`, the climate's flushed dual ops `cl_*`,
  `Climate::shared_helpers_glsl`).
- **Generic or specialised** (`SDFCascade::specialize`, `World::Settings::specialize_shaders`, the `VoxelTerrain`
  property `specialize_shaders`; the game specialises, the mixer's 3D view doesn't): *specialised*, the program's
  value-dependent folds run (a silent note dropped, blend factoring) and each level's code is straight GLSL;
  *generic* (`Terrain::Mode::GENERIC`), the graph's structure is kept and the GPU text is a fixed interpreter over the
  program's records (`level_glsl.hpp`): its text depends on the op set and the caps only, so any edit, a node added or
  deleted included, compiles nothing (`cascade_test.sh --generic=1` checks the GPU's generic text against the CPU's
  generic program). The flat table is specialised either way: nothing edits it.
- **Stretched notes** (a note's `"stretch_y"`; `Band::stretch`, at most k − 3): through the base
  (`BaseSurface::stretched` / `stretch_gradient`), the noise is read at p − (1 − 2^-j) d n and its gradient's up
  component scaled by 2^−j. Flat: datum + (y − datum) / 2^j exactly (GLSL `base_stretched`); sphere: CPU and GLSL with
  FrameSpike's `stretched_sphere`, but its GPU path needs the regions of the band's insertion LOD, which `BaseRegions`
  doesn't upload yet, so `SDFCascade` refuses such a climate on a sphere. Why: the ground is 3D, and moving up a column
  meets other noise (needles, folds where dh/dy passes 1); on the Mont Blanc match a stretch of 4 from 100 m up took the
  100 m octave's amplitude from 8.7 m to 2.3 m (real: 2.6 m) and the slopes over 75° from 0.4% to none.
- **Clouds** (a biome's `"clouds": {"cover": 0..1, "type": 0..1}`, `Climate::BiomeClouds`): the volumetric clouds' cover
  and type over the biome, blended by the weights on a coarse map round the camera (`CloudClimate`, `docs/look.md`).
  They touch no cascade pass or shader.
- **Worlds** (`adapter/godot/project/climate/`): `world.json` with `engine.json` and `climate.json` (eight biomes on
  temperature x humidity x wind, 6.5-26 km octaves, 15 LODs: the top, LOD 14, is the ground's largest note; each
  biome's notes once fitted to a real Copernicus heightmap, see git history of `tools/landforms/`; dunes and mountain
  carry their height curve), `example.json` (continent and terrain-type fields choosing mountains, hills, badlands and
  plains, two curves), `shaped_test.json` (every operation, CascadeSpike `4s`), `plain.json` (amplitudes only),
  `fieldless_test.json` (one biome, no field: CascadeSpike `4d`) and the other `*_test.json`. The test tables are world
  files with an inline engine and climate, each pinned by a hash of its program's bits (`test_level_program.cpp`).
- **Checks**: `tools/unit_test.sh "Climate*"` and `"Terrain*"`; `tools/climate_report.sh` (pure CPU: biome maps, the
  pick drift, where biomes meet with a `--spawn`, relief per biome, gradient error, field maps) before any GPU run;
  `cascade_test.sh --sets=4,4c,4f,4p,4s,4d,4w` (G1 every channel of every level, G3, points, G10 the pick);
  `discovery_audit.sh -- --world=... --discovery_compare=1`; `origin_parity.sh --world=...`.

## Measurements

Historic numbers of the v1 climate (before cascade v2: bands modulated by the biomes, the climate carried to the
samples), each on the tables of its time; v2's own are PR G's (`docs/cascade-v2.md`, Stages):

- **The carry** (CascadeSpike, 128-chunk batches, 25 LODs): the samples' level was 11.7 of the 16.2 ms of a world.json
  batch at LOD 0 with the weights evaluated three times per sample (the modulation 4.0 ms, the height terms 2.6,
  carrying 72 floats 3.7). With the carry: world.json -42 to -55% at every LOD; deriving from the first evaluation (one
  climate per point) took world.json another -39 to -46% (the level above the samples 5.7 -> 2.1 ms at LOD 0), the
  example -15 to -36%, point items +1 to +4%. Carrying the climate from every level to the next instead made the
  levels above the samples 2-5x dearer than what it saved. v2 computes the weights at their fields' level and carries
  them, the same idea as a rule.
- **Cost of a climate** (before the carry, the example table, 128-chunk batches): its passes 4.2-4.5x the bare cascade
  at LODs 0-8, 5.6-6.5x at 12-24; fields without multipliers 1.00x; the bigger `Ch` alone made one shader 1.4x
  (registers, not work). Forest5cm: steady +2%, tree placement +190 ms (the CPU traces carry the fields), a new
  table's first run pays its shaders' compile.
- **Generic vs specialised** (v1, `cascade_test.sh` timing, µs per chunk, 2026-11-03): generic +7..9% on 4w, +12..21%
  on 4s, +9..13% on 4c, the same on the bare ground; a mixer edit restarted in ~15 ms.
- **Relief**: every biome multiplying the whole spectrum moves slopes a lot (example: mountains 4-5x the plain's on
  26 and 210 km squares); a few octaves' multipliers barely do (0-45%). `climate_report.sh`'s relief table measures it.


## Biome mixer

`adapter/godot/project/debug/biome_mixer/`, `tools/biome_mixer.sh`: a world's biomes edited as the engine reads them,
the engine's view of them beside it. It is an editor over the engine, never a copy of it: it keeps no model of its own,
computes no terrain and has no node kind the engine hasn't.
- **The world is the world file** (`world.gd`): `res://climate/world.json`, or the file `--open=PATH` names (a planet's
  too: `worlds/tiny_biomes.world.json` is a blank 2 km planet, five placeholder biomes on the template ground over three
  fields of 819.2 + 409.6 m), as it is (every biome's `graph`, place, surface, clouds, instances, features, every other
  key), with the mixer's keys beside it (`fields`, its climate's; `path`, the file it came from; `current`, `mode`,
  `view`, `mixer_version`, `sources`: the biome files and the biomes' list file it read). Biome files are read inline
  (the default moved last), a fields file too, each field's `mean` / `poles` kept on it. A biome's `at` is kept as
  `[centre, span]`. The world's engine section sets the octaves (`EngineInfo.use_world`); a planet's views start on the
  whole planet. The World mode's fields panel (`fields_panel.gd`) edits the fields' bands (each shown with the octave
  the engine snaps it to) and their mean and poles (poles on a sphere only), adds and removes fields, and sets the
  shape's radius, the seed and the sea (on or off, its level; debounced, a recompile). The world autosaves (user://),
  and *Export* (World menu) writes it over its own file once `ClimateReportNode.check`
  (`ClimateReport::check_world`: the whole file, at its engine and over its planet, which refuses a ground under its
  centre the flat world takes) accepts it: the file is the world minus the mixer's keys (a file it names, biome, biome
  list, climate or fields, kept while its content is unchanged, else inline: the mixer never writes a file another
  world may name). *New planet* gives the world a random seed; *Freeze* writes `<name>.frozen.world.json` next to it,
  everything inline. The CPU maps (`ClimateReport::preview`) are sampled over the world's planet too. Numbers keep 7 significant digits; a value fits on one line up to 120
  characters.
- **The graph is the biome's `graph`** (`graph.gd`), node for node: a link is an input key naming another node's id
  (input port i is the kind's input i, a mix's port i its row i's `in`, one port more a new row), each node one output.
  Linking, unlinking, a row's x, a node's x or Delete (what read it reads its `in`, else nothing) edit the nodes in place;
  the mixer checks no rule of its own: any output links to any input, and the engine says what it refuses (a loop, a
  warp not coarser than its note), naming the node, in the status line. `pos` is the mixer's (the parser allows and ignores it); a graph read
  without positions is laid out in columns by its distance from the output, spaced by its built nodes' sizes.
- **The kinds are the engine's** (`engine_info.gd`, `ClimateReportNode.graph_kinds`: `Climate::graph_kinds`, the
  catalogue the parser itself checks each node's keys against): every kind it reads (note, warp, shape, multifractal,
  mix, constant, field, add, subtract, multiply, min, max, smin, smax, curve, up, output), its inputs and its parameters with
  their type and default. `nodes.gd` builds each node from it: a port per input, a row per parameter the node has (a
  slider and number box, a seed, a power of two, a field, a shape, coefficients, a curve), a `-` taking an
  optional one out and `+ parameter` adding one. A note's warp is its input port (deleting the warp node leaves the
  note unwarped); a warp's ports are `in` (read three times, seeds + 0, 1, 2) or `x`, `y`, `z` (a node each); a
  multifractal's `parent` takes any node. `kinds.gd` only holds titles, tooltips and sliders' ranges. A shape
  node plots the engine's phi (`ClimateReportNode.shape`: `Climate::parse_shape`, `shape_phi`) or its refusal; the
  Height curve's editor draws the engine's curve (`ClimateReportNode.curve`). The output edits the biome's `surface`
  colours. *New graph* is the engine's template biome (`ClimateReportNode.template_biome`: `worlds/ground.biomes.json`'s) or an output alone; a biome
  without its own debug colour gets the parser's name hash (`ClimateReportNode.debug_color`).
- **Two modes**: *Biome*, the current biome's graph (the picker in the bar) and the view of it alone; *World*, the
  climate chart and the view of the whole world, its biomes' debug colours blended by their weights over the relief.
  The chart (`chart.gd`, `world_panel.gd`): two fields as axes, each biome a box (its span outlined, its whole core,
  `ClimateReportNode.kernel_core` of it, filled, the full axis on a field it doesn't name), over a cloud of the view's
  points (the engine's fields there): drag a box to move a biome, its edge to change its span; the list beside it edits
  each field's centre and span and whether the biome names it. A double-click opens a biome's graph.
- **The view is the engine's** (`engine_view.gd`): every edit sends the world's text (`World.preview_file`: the game's
  engine, the fields inline; the Biome mode's biome alone as the default) to `ClimateReportNode.preview`
  (`ClimateReport::preview`): the report's own sampler on the CPU, each column traced to its ground as the map's
  lattice Λ_m sees it (`SDFCascade::ground_along(start, m)`: to within 2^m voxels, then the last two distances' linear
  crossing; the report and the game's sites trace at LOD 0): every field's value at every point (the chart's cloud),
  the altitudes (the ground under the camera's target) and the largest (`relief_m`: the curve editor's input span), or
  the engine's refusal. No image: the view is the 3D terrain (the 2D maps, and their sharp 192-point redraw, are gone).
  A mix row's M and S (mute, solo) are the view's only: the text sent has those rows at volume 0, the file keeps their
  x. 48 points a side (30-60 ms), on a thread: the window never draws a map (at 48 on the main thread a drag ran at 17
  fps, the map 30 ms of every frame). A drag's edits draw as often as the thread keeps up (the newest world each time),
  and the status, the chart's cloud and the 3D view follow each map shown (`EngineView.drawn`). The map's loop runs at
  `ov::Priority::LOW` (`preview(..., background)`: Godot's low-priority group, on the pool's low-priority share of the
  threads), behind the terrain's: at HIGH the main thread's `GpuTable::build_lists` waited behind the map's rows, up
  to 40 ms a frame. The view (`terrain_view.gd`) is the game's terrain: a `VoxelTerrain` (GPU
  Direct, the game's sky, fog and sun) loading the same text (`set_world_json`, no file). An edit restarts its world in
  place (`VoxelTerrain.restart_world`: a new `ov::World` on the same node, device and page host) 50 ms after it (not
  pushed back by the next edits: maps every ~40 ms kept it from firing), or once the last restart's chunks in view are
  loaded (at most 1 s later): a slider drag shows each load. A world the terrain refuses (`restart_world` false, or a
  started one whose passes failed to compile) leaves no terrain: `VoxelTerrain.get_world_error` says why, in the view's
  line and in red in the mixer's status (`EngineView.terrain_error`), never another ground drawn instead. The climate's
  numbers being shader params (`Cascade::ShaderParams`, `docs/cascade.md`), an edit of a value builds no pipeline: the
  device parks the pipelines the old world frees, by text, and its buffers and textures by shape, and the new world
  takes those of the same text or shape (`RDDevice::park`); the page pool goes from the old world to the new
  (`World::reuse`, `docs/pipelines.md`), and so does the GPU discoverer's local device on its thread
  (`docs/discovery-lod.md`); the generic text holds any graph, so a new node compiles nothing either (a specialised
  world compiles once per new structure), and the device keeps
  every text's SPIR-V. While a restart loads, the view holds the last terrain's frame (a copy of it shown over the viewport, which renders
  on below, until the chunks in view are loaded, at most 1 s: a SubViewport stopped with `UPDATE_DISABLED` showed
  black, noise and stripes): no blank ground between two loads. The orbit's pivot is a player, a 1.7 m capsule (a
  scale): a left drag walks it, its feet put back on the ground by the CPU's SDF (`get_sdf_at` along `get_up_at`, the
  game player's spawn rule) after each move and each restart, the camera kept 0.3 m above the ground; *Noclip* flies it
  where the camera looks, through the ground. Right or middle drag orbits, the wheel zooms down to 2.5 m. *First person* puts the camera at
  the player's eyes (1.6 m, the capsule hidden): right drag looks round, left drag moves 5 cm a pixel. WASD (the view
  focused: a click) moves along the ground, or where the camera looks in noclip, Space and Shift up and down; 6 m/s in
  first person (a game character's run), the orbit's distance a second otherwise. A crosshair at the view's centre
  reads the distance from the camera to the terrain under it and the time to run it at 6 m/s (sphere tracing on
  `get_sdf_at`, steps of 0.7 of it, a crossing bisected; 10 times a second, up to 60 km). *Run min* (5 by default, 0:
  none) draws a faint sphere round the player where a run of that many minutes reaches at 6 m/s (2 min 720 m, 5 min
  1.8 km): its silhouette brighter, and its line on the
  terrain (the scene's depth within 15 m of the sphere's), the reach to design a biome's features for. It loads round the orbit's pivot
  (`set_lod_centre`), from the LOD its distance needs there (`lod_view_min`). It loads LODs 0-13 only
  (`lod_view_max`: 12.8 km chunks, ~54 km round the pivot; a reload 116 ms p50 at 13, 100 at 10). Its pixel error is the game's (4: at 12 each LOD seam showed as a ledge, for a restart only
  ~10% shorter), its world's clouds, instances and features are off (the clouds' shadows hid the relief over kilometres;
  the instance stream took half the workers' time of a drag and 10-33 ms of some frames' `GpuTable::build_lists`), its
  discoverer runs on the GPU (its device kept across restarts: a scan ~1.5 ms, 46 on the CPU behind the map's rows; the
  first world pays the device and its pipelines, ~2 s, and an edit in those seconds waits for them) and its site walk
  on the CPU (its device is still made per world). Measured at 25 LODs, before the LOD cap (`--shot`: the first load, then two restarts on the same world, an edit's path; RTX
  3070 laptop): World mode (~400 km) 0.89 s, then 0.18 and 0.18 s; the mountain alone 0.76 s, then 0.22 and 0.18 s
  (were 1.2-1.3 s: a new node each, its start 1.05 s in one frame, 0.4 s the mesher's pipelines, 0.4 s the cascade's,
  0.2 s zeroing its buffers on the CPU). Now the restart is ~12 ms of one frame (`restart_world` p50 over a 20 s drag;
  was 46 ms: 11 freeing ~150 pages, 4 making 64, 9 the palette texture's 8 MB of zeros from the CPU, ~20 allocating the
  mesher's buffers and textures). A 20 s slider drag (no instances): the CPU discoverer, 49 restarts, the chunks in view
  loaded 129 ms (p50) after an edit, 345 ms p95; the GPU one kept, 114 restarts, 100 ms, 371 ms p95 (the scan 46 ->
  1.5 ms; 43 fps, the restarts' frames, 50 with the CPU one). Before the LOD cap, ~3,000 chunks over 25 LODs load in ~170 ms at 16-26 ms a frame; the view after a restart is the first
  load's, pixel for pixel. Orbit, pan and zoom move the camera. *Open in game* writes the world
  (`user://biome_mixer_game.world.json`) and runs the main scene on it (`--world=`) in a second Godot.
- **`--check`** (`checks.gd`): every engine kind has the mixer's UI; world.json read as the mixer's world and written
  back is the same file (positions aside); every world.json biome and a graph of every engine kind drawn by the engine
  and every node's GraphNode built (the launcher fails on any script error); edits on the mountain (a row out, a shape
  out, a loop linked then refused by the engine, a muted row heard at 0) still drawn; the every-kind graph includes a
  warp on `x`, `y`, `z` ports and a multifractal of a ridged parent; the whole world drawn (every biome named, one chart sample
  per point); the export written, read back the same, written again the same text, and a span of 0 refused (nothing
  written).
