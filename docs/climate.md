# Climate

The climate in three parts: the **model** (what the ground is at a point, as math), the **implementation** (how the
cascade computes it, CPU and GPU, and what is there only for speed), and the **measurements**. Then the biome mixer,
the tool that designs a world's biomes.

## Model

A climate (`src/climate.hpp`, a table: format above `Climate::parse` in `src/climate_parse.cpp`; three files give it:
`engine.json` the voxel and LODs, `climate.json` the fields that place biomes, `world.json` the biomes, merged by
`WorldFile` into the table `parse` reads, `docs/terrain.md`) shapes **one ground**. Each biome is a **graph**: the
notes it mixes (`n_k`, each one octave k with its own seed, amplitude and stretch), each read displaced by its warp,
through a shape and a multifractal, mixed at a volume, then a height curve. A biome lists exactly what it uses: no engine default notes.
The **biome compiler** (`src/biome_compiler.{hpp,cpp}`) turns every biome's graph into that one ground: a note is kept
once whatever biome mixes it (identity: octave, seed, stretch), so a biome more costs only the notes it adds. Without
biomes the ground is the **template biome** (`Climate::template_biome`, the mixer's start for a new biome too): seed
0xEA27, a note per octave from k = 3 (0.4 m at 5 cm) to the first at the 3.3 km cap (k = 19 at 5 cm: 26.2 km), each
1/8 of its wavelength, mixed as "ground". Notes and fields reach lod_count + 4 at most (`Ground::k_max`, the cascade's top); the fields
placing the biomes (temperature, humidity, wind: 6.5-26 km, under the top LOD like the ground: a world to walk, not
continents) are macro inputs until the planned macro engine computes
them.

1. **Fields** (`Field`): a few slow scalars, each a sum of plain perlin octaves of its own seed, plus linear terms
   (`coef * the ground` or `coef * an earlier field`, in list order): temperature, humidity, wind. A term on the
   ground makes a field follow the height (colder higher up). The climate is what places biomes: a warp's noise is the
   biome graph's own (3.).
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
   - `A_k = sum_b w_b * x_{b,k} * amp_k` (the biome's mix row volume times the note's `"amp_m"`; 0 for a biome that
     doesn't mix it),
   - `phi_b(n) = c0 + c1 n + c2 |n| + c3 n^2` (the biome's shape: plain, billow, ridged, square), inserted without its
     mean,
   - times `clamp(gain (1 - c), 0, 1)` with a multifractal (c: the coarser notes' weighted signal, Musgrave's chain),
   - read at the point moved by `amount_m * (x, y, z)` with a warp: the note's `"warp"` input is a warp node whose
     input is notes (a note or a mix of them, every one coarser than the note), read with their seed, seed + 1 and
     seed + 2 for x, y and z (libnoise's turbulence).
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
  octaves (a note is one octave, its wavelength's nearest power of two of voxels: finer than 8 voxels is refused, past
  the LOD count's top it plays nothing, so one table covers 15 and 25 LODs), each biome's graph read back from its output by the *relaxer* (the parser's
  `expr`), at `SDFCascade::climate_ground()`. A graph is free: any node may read any other, and the relaxer decides
  where each one plays. A node is either notes the cascade inserts (rows: a note's per-octave chain at a volume) or a
  register of the biome's sample program:
  - a note's chain plays per octave while it reads note -> shape -> multifractal, each at most once; the note's warp
    is its input, not a node of the chain (it moves where the note is read);
  - a warp's notes are coarser than every note it moves, the *planner's rule*: a note of octave k reads the fields as
    they stand before level k's insertions, its coarser octaves only. Each note of a warp's input (rows, every one
    plain) is three fields (`Field::warp`: x, y, z, its octave's noise at amplitude 1 and seed + 0, 1, 2, not
    modulated by the biomes), one per octave and seed whatever warp reads it, and a *term* of the warped note
    (`WarpTerm`: those fields at `amount_m` times the row's volume and amplitude, `Amp::warps`). The fields hold no
    value, so a value edit never changes the table's structure (a generic table also keeps its warp fields' count,
    `Caps::warps`, so deleting a warp compiles nothing); world.json's warps read 4 octaves, 12 fields. The modulator
    plays a band's terms as records of their own (`BiomeBand`: no amplitude, the warp), summed per axis like the
    biomes';
  - a mix is linear: its rows' notes at their volumes (a mix of mixes is one mix; a mix of one row at volume 1 is that
    row, so a shape over it still plays per octave); a row past the notes (a register) is summed at the samples
    (`CONST x`, `MUL`, `ADD`);
  - any other node reading notes (a second shape, a shape over a mix of several notes, a curve, min / max, ...)
    reads their sum at the samples: the notes are *lifted* to a register. The lift with the coarsest octave (then the most
    octaves, then the first) is the ground (`GROUND`: the cascade's H; chosen by structure, so no value edit moves it); every other one is a *sub-mix*, a channel field (`Field::channel`: its bands are those
    notes, each biome playing them like the ground's, `Amp::group`, modulated per biome, shapes and warps per octave
    too) the program reads with `FIELD`. The compiler puts each biome's sub-mixes on the first channel field none of
    whose octaves another note takes (a field group inserts one band per level), so sub-mixes of different biomes share
    fields;
  - a shape past the notes is `SampleOp::SHAPE`: phi of the sum (metres as LOD0 voxels), its constant term counting,
    `|a|` unsmoothed;
  - `up` is `SampleOp::UP`, the sample's height above the base (`up_value`: `BaseSurface::at` times `2^lod`, exact;
    GLSL `climate_ground(ch, p, lod)` reads `base_at` where a program reads it). A cave is a smooth min on the mix's
    output, `smin(mix, up - C)` (`docs/caves.md`);
  - refused: a multifractal past a chain (it weighs a note by the coarser multifractal notes), a warp read as a value
    (it is a note's input), a warp of a node past the notes, of shaped, multifractal, warped or stretched notes, or of
    a note not coarser than one it moves, a sub-mix with two notes on one octave or a multifractal.

  Existing graphs (the old row form) relax to what they were: the same notes, programs and bits. The compiler keeps a
  note per (octave, seed, stretch, group), turns each row into its biome's `Amp` on each octave (volume times the note's
  amplitude, under a planet's `note_cap`), folds the notes every biome plays alike into constant bands and lists the
  others in every biome (0 where it doesn't mix them). Refused: more than `MAX_FIELDS` (24) fields or `MAX_BIOMES` (16) biomes,
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
- **Generic or specialised** (`Ground::specialize`, set by `SDFCascade::specialize`, `World::Settings::
  specialize_shaders`, the `VoxelTerrain` property `specialize_shaders`; the game specialises, the mixer's 3D view
  doesn't): one compiler and one modulator, and a term's text is the same code either way. *Specialised*, the table's
  structure is its values': a note silent everywhere is dropped, a note every biome plays alike is a constant band,
  and a term whose coefficient is 0 (an amplitude on |n| or n^2, a multifractal gain, a state coefficient, a warp) is
  not in the text. *Generic* (`Table::generic`), the text is an interpreter of the compiled table: it holds every
  feature (shapes, the multifractal state, warps, on the ground and on the sub-mixes), loops over data where the
  specialised text unrolls (the modulator's levels, entries, biomes and parts; the field terms and the biome axes; the
  sample programs, `cl_run` over their instruction records; the extra notes of a level, from its ground entry in the
  band table; the per-LOD switches as param tables) and tests each value where `biome_mod` does (`if (k != 0.0)`,
  `gain > 0.0 ? ...`). Its numbers are in the shader params: a fixed part the text indexes with literals, then a heap
  (`ShaderParams::put`, read with `cl_heap`) of the variable-length records, which the fixed part points into, so a
  record growing moves no literal. Its sizes are capacities (`Climate::Caps`: notes per level, carry groups, sub-mix
  fields, warp fields; each a power of two (the warp fields: their count) at least what the table needs and at least the previous start's, `World::Kept`), and
  the param block is at its largest. So any edit that fits the capacities keeps the text, a node deleted, a shape's
  type changed, a curve removed included: a world's restart compiles nothing; one past them compiles once with larger
  ones. The specialised text is the same per-term code with the compile's knowledge folded (literals, dead terms
  out), so the bits are the same: the extra carry parts hold 0 (`Cascade::flush`), the extra state group is never
  read, a constant level takes the constant inserts, a silent band inserts nothing (the unit test "a table's text
  follows its structure, the generic one's through every edit, with the specialised bits": 7 tables, 7 edits each;
  on the GPU `cascade_test.sh --generic=1`). The template ground (no
  climate) is specialised either way: nothing edits it. Cost (`cascade_test.sh` timing, µs per chunk, 2026-11-03):
  generic is +7..9% on 4w, +12..21% on 4s, +9..13% on 4c over specialised, the same on the bare ground; a mixer edit
  restarts in ~15 ms (a warp to 0 froze 3.4 s specialised).
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
  `field_slope`: each register's upper and lower affine forms in `z = up - H` (unbounded) and H (`bound_program`), so
  a curve of H keeps its own bound and a cave's tent bounds its carve by its floor; a term without a bound is refused);
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
  6.5-26 km octaves, 15 LODs: the top, LOD 14, is the ground's largest note; each biome's notes once fitted to a real Copernicus heightmap, see git history of
  `tools/landforms/`; dunes and mountain carry their height curve), `example.json` (continent and terrain-type fields
  choosing mountains, hills, badlands and plains, two curves), `shaped_test.json` (every operation, CascadeSpike `4s`),
  `plain.json` (amplitudes only; pins every slot, the SDF, the skirts and both level shaders' text: a table without
  shapes keeps the bits it had before them; its text is the shaped one since the plain amplitudes' own was folded in), `fieldless_test.json` (one biome, no field: CascadeSpike `4d`). The test tables are world files with an inline engine and climate.
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

`adapter/godot/project/debug/biome_mixer/`, `tools/biome_mixer.sh`: a world's biomes edited as the engine reads them,
the engine's view of them beside it. It is an editor over the engine, never a copy of it: it keeps no model of its own,
computes no terrain and has no node kind the engine hasn't.
- **The world is the world file** (`world.gd`): `res://climate/world.json` as it is (every biome's `graph`, place,
  surface, clouds, instances, features, every other key), with the mixer's keys beside it (`fields`, its climate's,
  read only; `current`, `mode`, `view`, `mixer_version`). A biome's `at` is kept as `[centre, span]`. The world
  autosaves (user://), saves to `climate/worlds/*.world.json`, and *Export* (World menu) writes it over world.json once
  `ClimateReportNode.check` (`ClimateReport::check`) accepts its biomes over its fields: the file is the world minus the
  mixer's keys (its climate file untouched). Numbers keep 7 significant digits; a value fits on one line up to 120
  characters.
- **The graph is the biome's `graph`** (`graph.gd`), node for node: a link is an input key naming another node's id
  (input port i is the kind's input i, a mix's port i its row i's `in`, one port more a new row), each node one output.
  Linking, unlinking, a row's x, a node's x or Delete (what read it reads its `in`, else nothing) edit the nodes in place;
  a link that would close a loop is refused. `pos` is the mixer's (the parser allows and ignores it); a graph read
  without positions is laid out in columns by its distance from the output, spaced by its built nodes' sizes.
- **The kinds are the engine's** (`engine_info.gd`, `ClimateReportNode.graph_kinds`: `Climate::graph_kinds`, the
  catalogue the parser itself checks each node's keys against): every kind it reads (note, warp, shape, multifractal,
  mix, constant, field, add, subtract, multiply, min, max, smin, smax, curve, up, output), its inputs and its parameters with
  their type and default. `nodes.gd` builds each node from it: a port per input, a row per parameter the node has (a
  slider and number box, a seed, a power of two, a field, a shape, coefficients, a curve), a `-` taking an
  optional one out and `+ parameter` adding one. A note's warp is its input port (deleting the warp node leaves the
  note unwarped). `kinds.gd` only holds titles, tooltips and sliders' ranges. A shape
  node plots the engine's phi (`ClimateReportNode.shape`: `Climate::parse_shape`, `shape_phi`) or its refusal; the
  Height curve's editor draws the engine's curve (`ClimateReportNode.curve`). The output edits the biome's `surface`
  colours. *New graph* is the engine's template biome (`ClimateReportNode.template_biome`) or an output alone; a biome
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
  crossing; the report and the game's sites trace at LOD 0), the shaded relief
  and the weights' blend plus every field's value at every point (the chart's cloud) and the largest altitude
  (`relief_m`: the curve editor's input span), or the engine's refusal. A mix row's M and S (mute, solo) are the view's
  only: the text sent has those rows at volume 0, the file keeps their x. 48 points a side (30-60 ms), then 192 once the edits pause
  (`SETTLE` 0.35 s; ~1 s on 16 cores: ~0.5 ms of CPU a point, each point walking the cascade on its own, 6 reads), both
  on a thread: the window never draws a map (at 48 on the main thread a drag ran at 17 fps, the map 30 ms of every
  frame). A drag's edits draw as often as the thread keeps up (the newest world each time), and the status, the chart's
  cloud and the 3D view follow each map shown (`EngineView.drawn`). With 3D shown the map's loop runs at
  `ov::Priority::LOW` (`preview(..., background)`: Godot's low-priority group, on the pool's low-priority share of the
  threads), behind the terrain's: at HIGH the main thread's `GpuTable::build_lists` waited behind the map's rows, up
  to 40 ms a frame. *3D* (on by default, `terrain_view.gd`) is the game's terrain: a `VoxelTerrain` (GPU
  Direct, the game's sky, fog and sun) loading the same text (`set_world_json`, no file). An edit restarts its world in
  place (`VoxelTerrain.restart_world`: a new `ov::World` on the same node, device and page host) 50 ms after it (not
  pushed back by the next edits: maps every ~40 ms kept it from firing), or once the last restart's chunks in view are
  loaded (at most 1 s later): a slider drag shows each load. The climate's
  numbers being shader params (`Cascade::ShaderParams`, `docs/cascade.md`), an edit of a value builds no pipeline: the
  device parks the pipelines the old world frees, by text, and its buffers and textures by shape, and the new world
  takes those of the same text or shape (`RDDevice::park`); the page pool goes from the old world to the new
  (`World::reuse`, `docs/pipelines.md`), and so does the GPU discoverer's local device on its thread
  (`docs/discovery-lod.md`); a new structure (a node, a coefficient reaching 0) compiles once, and the device keeps
  every text's SPIR-V. While a restart loads, the view holds the last terrain's frame (a copy of it shown over the viewport, which renders
  on below, until the chunks in view are loaded, at most 1 s: a SubViewport stopped with `UPDATE_DISABLED` showed
  black, noise and stripes): no blank ground between two loads. It loads round the orbit's pivot
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
  out, a loop refused, a muted row heard at 0) still drawn; the whole world drawn (every biome named, one chart sample
  per point); the export written, read back the same, written again the same text, and a span of 0 refused (nothing
  written).
