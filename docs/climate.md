# Climate

The climate in three parts: the **model** (what the ground is at a point, as math), the **implementation** (how the
cascade computes it, CPU and GPU, and what is there only for speed), and the **measurements**. Then the biome mixer,
the tool that designs a world's biomes.

## Model

A climate (`src/climate.hpp`, a table: format above `Climate::parse` in `src/climate_parse.cpp`; three files give it:
`engine.json` the voxel and LODs, `climate.json` the fields that place biomes, `world.json` the biomes, merged by
`WorldFile` into the table `parse` reads, `docs/terrain.md`) shapes **one ground**. Each biome is a **graph**: the
notes it mixes (`n_k`, each with its own octave k, seed, amplitude and stretch), each read through a warp, a shape and
a multifractal, mixed at a volume, then a height curve. A biome lists exactly what it uses: no engine default notes.
The **biome compiler** (`src/biome_compiler.{hpp,cpp}`) turns every biome's graph into that one ground: a note is kept
once whatever biome mixes it (identity: octave, seed, stretch), so a biome more costs only the notes it adds. Without
biomes the ground is the **template biome** (`Climate::template_biome`, the mixer's start for a new biome too): seed
0xEA27, every octave from k = 3 (0.4 m at 5 cm) to the first at the 3.3 km cap (k = 19 at 5 cm: 26.2 km), amplitude
1/8 of the wavelength. Notes and fields reach lod_count + 4 at most (`Ground::k_max`, the cascade's top); the fields
placing the biomes (temperature, humidity, wind: 105-419 km) are macro inputs until the planned macro engine computes
them.

1. **Fields** (`Field`): a few slow scalars, each a sum of plain perlin octaves of its own seed, plus linear terms
   (`coef * the ground` or `coef * an earlier field`, in list order): temperature, humidity, wind, the warp fields
   wx / wy / wz. A term on the ground makes a field follow the height (colder higher up).
2. **Weights** (`Climate::weights`): a biome is a place in the fields' space, `"at"`: a centre and a span on each field
   it names (`[centre, span]`, or `{min, max}`: their middle and half their gap). Its kernel `k_b` is the product over
   those fields of `smoothstep(clamp((1 - u^2) * 2, 0, 1))`, `u = (f - centre) / span`: on each field whole within
   0.71 of its span, gone at its span, each field's span its own (naming one more field doesn't shrink the others).
   The last biome, the default, has no place: `k_d = (1 - sum k)^2` where the kernels sum under 1. The weights are
   `w_b = k_b / S`, `S = sum k + k_d >= 3/4`: overlapping biomes blend by themselves, no borders to match and no
   priority. `1 / S` is `det_rsqrt(S)^2` and one Newton step (exact 1 at S = 1: a biome alone at its centre is whole),
   each weight capped at 1. It replaced edges (min / max with a fade) under priority layering, `w_b = rest * r_b`:
   world.json's edges became places with open sides closed at 0.85 and spans x1.1, its biome shares over 1678 km
   within a point of the edges' (desert 5.7% both, mountain 18.2% for 19.4%).
3. **The ground** at a point: each note k of the compiled ground is inserted as
   `A_k * phi_b(n_k(p + warp))` blended over the biomes,
   - `A_k = sum_b w_b * x_{b,k} * amp_k` (the biome's mix row volume times the note's amplitude: `"steepness"` times
     its wavelength, or `"amp_m"` times `"persistence"` per finer octave, under `"max_amp_m"`; 0 for a biome that
     doesn't mix it),
   - `phi_b(n) = c0 + c1 n + c2 |n| + c3 n^2` (the biome's shape: plain, billow, ridged, square), inserted without its
     mean,
   - times `clamp(gain (1 - c), 0, 1)` with a multifractal (c: the coarser notes' weighted signal, Musgrave's chain),
   - read at the point moved by `warp_m * (wx, wy, wz)` with a warp.
   A note every biome plays plain at one amplitude is a constant band (no modulation, the same bits: `A = a` exactly).
   A note's `"stretch_y"` reads it squeezed along the base's up.
4. **The biome's program** (`Biome::program`, `Climate::SampleProgram`): what its graph does past its notes' mix, at
   each sample, on the blended ground H (the cascade's: every biome's notes) and the fields: constants, fields, add,
   subtract, multiply, min, max, smooth min / max (Quilez's polynomial, over `k_m`), curves (`R(h)` through the biome's
   points `[h, R(h)]`: libnoise's cubic of `C = R - h`, constant past the ends). The ground is
   `h = H + sum_b w_b * D_b`, `D_b = f_b(H) - H`; a curve of H alone is `D = C(H)` itself (its bits from before
   programs). Inside a biome's core `H` is its own mix; in a blend each program reads the blend, as a curve did.
5. **The pick**: the biome with the largest weight (ties by a hash of the absolute voxel) gives the sample its material
   (`biome_material`: the procedural ground in the biome's colours).

The **LOD rule**: note k is inserted on the lattice Λ_{k-3} (8 samples per wavelength) and reads the climate of the
level above, before that level's insertions: so note k follows the weights of the fields' octaves coarser than k only,
and a LOD-L chunk's samples read the climate of Λ_{L+1}. Every value is a pure function of (point, LOD), the same bits
from any chunk or point query.

What the model doesn't have, on purpose: per-landform noise fields, regions, generic height terms on fields, terraces,
cell noise (the landform fitting's machinery, removed in the climate declutter: one way to shape the ground, the
biomes playing its notes; the region guards, a GPU skip of a region's biomes where its edges were off, went with the
regions), and anything macro (continents, the climate axes as smooth per-region values: the planned
macro layer in double on the CPU, `BaseSurface`'s region machinery).

## Implementation

- **Parse, compile and validate** (`climate_parse.cpp`, `biome_compiler.cpp`, `climate_validate.cpp`): metres to
  octaves (`from_m` .. `to_m` ranges keep the octaves the LOD count has, so one table covers 15 and 25 LODs; a single
  wavelength outside is refused), each biome's graph read back from its output (a curve, a mix, each row a note through
  a warp, a shape and a multifractal, in that order), at `SDFCascade::climate_ground()`. The compiler keeps a note per
  (octave, seed, stretch), turns each row into its biome's `Amp` on each octave (volume times the note's amplitude,
  under a planet's `note_cap`), folds the notes every biome plays alike into constant bands and lists the others in
  every biome (0 where it doesn't mix them). Refused: more than `MAX_FIELDS` (24) fields or `MAX_BIOMES` (16) biomes,
  more than `Cascade::MAX_LEVEL_NOTES` (8) notes on one octave or two multifractal ones (the state holds one octave's
  signal; an octave with a modulated note modulates all of its notes), a note mixed twice by one biome, a negative
  amplitude, a
  shaped amplitude past the cap (`Ground::amp_cap`: a planet's relief, the exact range's top on the flat world), a
  curve moving the ground past it, a curve whose
  offset is steeper than `MAX_CURVE_SLOPE` (4: the skirts and the gradient scale with it), a curve whose points all
  lie past the heights the ground reaches (a constant offset: a wrong unit or sign), an `"offset"`
  without a multifractal (bands are inserted without their mean, so only the state reads it), a field's multiplier
  without a ground one, a carry past `Cascade::MAX_CARRY`.
- **In the cascade** (`Climate::combined_spec`): the fields are `Spec::fields`, groups carried down with the
  ground (`Cascade::field_slot`; on the GPU after the ground's 4 channels, `Spec::gpu_base`). A note some biome names
  is modulated (`Cascade::Modulation`, `Climate::Modulator` in `climate_modulator.cpp`): at each point of its level
  the climate (`Climate::state`: field terms, then weights) gives a `Cascade::BandMod` (three amplitudes on n, |n|, n^2;
  the multifractal state's new channels; the warp), and the band inserts value and slopes as dual numbers
  (`Cascade::modulated_channels`; GLSL `cascade_shaped_glsl.cpp`). `|n|` is `Cascade::smooth_abs`,
  `sqrt(n^2 + 0.01) - 0.1`: with the bare kink the Hermite subdivision rang along the lattice axes (ridged bands'
  gradient channels 21% off the values' finite differences against 2.5% plain). The multifractal state is a band-less
  field group (`Modulation::state_group`) the ground's band overwrites. The warp moves the read point by int64 +
  fraction (`WARP_LIMIT`), its gradient by the chain rule through the fields' gradients; in the engine a warped note
  costs no extra noise read (the warp fields' own octaves are inserted on their coarse lattices and interpolated
  down). Slope damping and the swiss turbulence (warps read from the ground's slope) were built and dropped: their
  own gradient needs the ground's second derivatives, which the cascade doesn't carry.
- **One climate per point, none per sample** (the carry): a run's last level, its samples, evaluates no climate. The
  level above derives it from the climate its own bands read (`Cascade::Modulation::derive`; on the GPU at the end of
  `cascade_mods`) into the carry groups (`Cascade::carry_slot`): the last level's `BandMod` parts and the samples'
  terms (`Climate::SampleCarry`: each curved biome's weight, the pick, copied from the floor parent instead of
  interpolated). H carries them down with the ground. So a LOD-L chunk's samples read the last octave's climate from
  Λ_{L+1}: the model's rule above, not an approximation of another one.
- **The programs** (`Climate::ground_value`, GLSL `climate_ground` and one `cl_sample<b>` per biome, through
  `SDFCascade::ground_channels` on every path: chunk samples, point items, `ground_along`, the report):
  `H + sum_b w_b * D_b` with the carried weights, each program's registers in order with the climate's flushed dual
  ops (`dadd`, `dmul`, ...; GLSL `cl_*`), op for op. A field a program reads is carried to the samples like the weights
  (`SampleCarry::field`, derived on the level above). The compiler bounds each term (`SampleProgram::bound`, `slope`,
  `field_slope`: interval arithmetic on the value and on its deviation from H, so a curve of H keeps its own bound);
  validate holds the bound under the cap and the slope under `MAX_CURVE_SLOPE`, and the skirts take the step
  (`Climate::ground_step`: the ground's step times 1 + the largest slope, plus each weight's change times its term's
  bound, plus each field's step times its slope; unit test: the L | L+1 step stays within the skirt on
  `example.json`, `shaped_test.json`, `world.json` and `programs_test.json`). The programs are the climate's own SSA,
  not the archetype IR: a curve through the archetype IR wouldn't be the curves' bits (it doesn't flush under 2^-64,
  and its dual product orders a slope otherwise); one program format for both is later work. Not yet: a carve (caves,
  `docs/caves.md`: the discoverer's carve band first), and a single note read at the samples (its own carry).
- **GPU only, for speed** (the CPU mirrors them today; the plain-oracle stage moves them GPU-side): which levels carry
  the fields and the carry (`Cascade::FieldRange`: `carried`, `inserted`, `carries`, `batch`, `batch_carry`), how many
  carry groups a last level loads (`Modulation::carry_used`), the three generated views (`CascadeGLSL::View`: ground
  alone, ground + fields + carry for the levels that derive, ground + carry for a run's last level) on planar scratch,
  and skipping the empty top (`Spec::run_top`). None changes a value: the CPU's all-fields run gives the same ground
  bits (`test_climate.cpp`).
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
- **Worlds** (`adapter/godot/project/climate/`): `world.json` with `engine.json` and `climate.json` (eight biomes on temperature x humidity x wind,
  105-419 km octaves, needs 25 LODs; each biome's notes once fitted to a real Copernicus heightmap, see git history of
  `tools/landforms/`; dunes and mountain carry their height curve), `example.json` (continent and terrain-type fields
  choosing mountains, hills, badlands and plains, two curves), `shaped_test.json` (every operation, CascadeSpike `4s`),
  `plain.json` (amplitudes only; pins every slot, the SDF, the skirts and both level shaders' text: a table without
  shapes generates the old text bit for bit), `fieldless_test.json` (one biome, no field: CascadeSpike `4d`). The test tables are world files with an inline engine and climate.
- **Checks**: `tools/unit_test.sh "Climate*"`; `tools/climate_report.sh` (pure CPU: biome maps, the pick drift, where
  biomes meet with a `--spawn`, relief per biome, gradient error, field maps) before any GPU run;
  `cascade_test.sh --sets=4,4c,4f,4p,4s,4d,4w` (G1 on the carried channels, G3, points, G10 the
  pick); `discovery_audit.sh -- --world=... --discovery_compare=1`; `origin_parity.sh --world=...`.

## Measurements

Historic numbers, each on the tables of its time (the landform world.json had 16 fields and 8 region-gated height
terms; the declutter's own before / after is in its PR):

- **The carry** (CascadeSpike, 128-chunk batches, 25 LODs): the samples' level was 11.7 of the 16.2 ms of a world.json
  batch at LOD 0 with the weights evaluated three times per sample (the modulation 4.0 ms, the height terms 2.6,
  carrying 72 floats 3.7). With the carry: world.json -42 to -55% at every LOD; deriving from the first evaluation (one
  climate per point) took world.json another -39 to -46% (the level above the samples 5.7 -> 2.1 ms at LOD 0), the
  example -15 to -36%, point items +1 to +4%. Carrying the climate from every level to the next instead made the
  levels above the samples 2-5x dearer than what it saved.
- **Cost of a climate** (before the carry, the example table, 128-chunk batches): its passes 4.2-4.5x the bare cascade
  at LODs 0-8, 5.6-6.5x at 12-24; fields without multipliers 1.00x; the bigger `Ch` alone made one shader 1.4x
  (registers, not work). Forest5cm: steady +2%, tree placement +190 ms (the CPU traces carry the fields), a new
  table's first run pays its shaders' compile.
- **Relief**: every biome multiplying the whole spectrum moves slopes a lot (example: mountains 4-5x the plain's on
  26 and 210 km squares); a few octaves' multipliers barely do (0-45%). `climate_report.sh`'s relief table measures it.

## Biome mixer

`adapter/godot/project/debug/biome_mixer/`, `tools/biome_mixer.sh`: a world's biomes built from the ground's octaves in
node graphs (GraphEdit), the engine's view of them beside it; a design tool over the engine, never a copy of it (the
world file is the engine's input: the World menu's export writes the world back into it). It holds a **world**
(`world.gd`: world.json's fields, climate warp and every biome, its place, its graph, and the keys it doesn't edit,
`instances` and `features` among them, carried in `keep` from import to export) and has two modes:
- **Biome**: the current biome's notes in the graph (the picker in the bar), the view of that biome alone.
- **World**: the climate chart and the view of the whole world, its biomes' debug colours blended by their weights
  over the relief. The chart (`chart.gd`, `world_panel.gd`): two fields as axes, each biome a box (its span outlined,
  its whole core filled, the full axis on a field it doesn't name), over a cloud of the view's points (the engine's
  fields there): drag a box to move a biome, its edge to change its span; the list beside it edits each field's centre
  and span and whether the biome names it. A double-click opens a biome's notes. The world autosaves, and saves to
  `climate/worlds/*.world.json` (the World menu).

- **The view is the engine's** (`engine_view.gd`): every edit exports the graph as a world file's text
  (`ExportTable.preview_world`: the game's engine, the fields inline, the biomes as the export writes them; the Biome
  mode's biome alone as the default) and `ClimateReportNode.preview` (static, `ClimateReport::preview`) draws it: the
  report's own sampler (`sample`) on the CPU, each column traced to its ground, the shaded relief and the weights' blend
  (`shaded_rgb`, `blend_rgb`, the same functions as the report's PNGs) plus every field's value at every point (the
  chart's cloud) and the altitudes (the curve editor's input span). 48 points a side at once, 192 once the edits pause
  (`SETTLE` 0.35 s): world.json's biome alone ~140 us a point, the 8-biome world ~50 us at 16 (the check), so the small
  map ~0.3 s and the sharp one a few seconds. The 3D view is the game: *Open in game* writes the world
  (`user://biome_mixer_game.world.json`, the game world's zone, instances, species and features kept) and runs the main
  scene on it (`--world=`) in a second Godot. What the mixer's own copy did and the engine doesn't is gone with it: its
  multifractal was `clamp(gain * chain)` (the engine's `clamp(gain (1 - c))`), its `stretch_y` uncapped, its World mode
  read the fields at full resolution (the engine from coarser octaves), and Note warp, Carve, Add, Multiply and
  Constant only ever existed in the preview (the export refuses or drops them, below).
- **Nodes** (`kinds.gd`): *Note* (the ground's octave k on the template's seed, the engine's
  `ClimateReportNode.template_notes`: the mixer's biomes play the template's notes, so a note has no seed of its own. The
  notes are fixed: a model always has one per k, K_MIN .. `ground_top` (`Kinds.fix_notes` on every load: a missing one
  comes back as a silent row at volume 0, an extra or out-of-range one goes), none can be added, deleted or moved to
  another k, and a Mix row's x leaves its note silent. Its `warp` switch is a biome's `warp_m` on that octave in the
  engine, `stretch_y` the note's `stretch_y`), *Shape* (`c0 + c1 n + c2 |n| + c3 n^2`; presets, scale and offset as
  `climate_parse.cpp`; its node draws phi), *Multifractal* (the engine's state, wired to the next finer note), *Mix*
  (rows of volume, mute, solo, centre), *Height curve* (a remap R(h) through points: a biome's curve node in the engine;
  its editor draws the engine's cubic, `ClimateReportNode.curve`: `make_curve`, `eval_curve`), *Output* (flat and steep
  colours), and the kinds only the old preview drew (Note warp, Add, Multiply, Constant, Carve), kept so old files load.
  A row's x takes it out with the nodes only it used but its note (`Kinds.cut_row`); a node's x or Delete removes it and
  the Mix row it played, never a note (`Kinds.cut_nodes`).
- **Export** (`export_table.gd`, World menu: *Export this world to res://climate/world.json*): `ImportTable` read
  backwards. Each biome's height must be its Mix, through a Height curve or not, into the Output, each row a Note
  through a Shape then a Multifractal (each optional); anything else (Add, Multiply, a Constant, a note in two rows)
  stops the export, as does a table the engine refuses: its text is parsed by `ClimateReportNode.check` (static,
  `ClimateReport::check`: the cascade of the world's voxel and LOD count configured on it) before it is written over
  world.json. Each biome is its `graph`: a run of equal notes one range note node (the template's seed, steepness
  and cap, its `stretch_y`) through its warp, shape and multifractal nodes, a mix row at its volume (0 for a silent
  one), a curve node; the places `at` (the last biome the default, without one), the Output's colours over the table's
  surface (its other keys, roughness, kept), clouds, debug colour and the biome's other keys as imported (`instances`,
  `features`: the mixer edits neither yet); the climate warp sets the three fields of its seeds (or adds `wx`, `wy`, `wz`
  when a note is warped and the world has none). Left out with a line in the log: Carve, a Note warp (exported reading
  the climate warp), an uncentred row, a multifractal linked to another than the next coarser one. Mute and solo
  export as heard (a silent row at volume 0). An identity Shape (`"shape": "plain"`) goes. A note node is always a
  range (`from_m` .. `to_m`, one octave or more): the engine trims a range to a LOD count's octaves but refuses a single
  `wavelength_m` past them. Numbers keep 7 significant digits; a value fits on one line up to 120 characters.
- **The climate's warp** (wx, wy, wz): a model's `climate_warp`, world.json's three warp fields by default
  (`Kinds.CLIMATE_WARP`), shared by every biome.
- **Import** (`import_table.gd`, `graph_rows`): a biome of `world.json` as a graph (every note, its graph's rows as
  volumes, 0 for an octave it doesn't mix, shapes, multifractal chain, warped octaves as warp switches with their
  `warp_m`, the warp fields as the climate warp, its notes' `stretch_y`, its curve, colours; a note off the template's
  seed or steepness is warned of: the mixer plays the template's notes); *Plain ground* is every note at volume 1.
  Biomes save to `climate/biomes/*.biome.json`; `World.from_table` imports every biome of a table, with its place.
- **`--check`** (`checks.gd`): every template and world.json biome exported and drawn by the engine (no refusal), the
  same table graph after a file round trip, the fixed notes (K_MIN .. ground_top: 3 .. 19 at 5 cm), the cuts on the
  mountain and plain ground, an old file's Climate warp node and Terrace loading as the same graph, the every-kind graph
  refused, the whole world drawn (every biome named, one chart sample per point); and world.json exported back: every
  biome's notes as the engine reads them per octave, places, curves, surfaces, clouds, instances and features equal the
  table's, the export of that export the same file, the engine refusing a span of 0 (nothing written).
