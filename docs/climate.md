# Climate

The climate in three parts: the **model** (what the ground is at a point, as math), the **implementation** (how the
cascade computes it, CPU and GPU, and what is there only for speed), and the **measurements**. Then the biome mixer,
the tool that designs a biome.

## Model

A climate (`src/climate.hpp`, a JSON table: format above `Climate::parse` in `src/climate_parse.cpp`;
`VoxelTerrain.climate_path` / `--climate=PATH`, off by default) shapes **one ground**: the cascade's octaves, the
notes, `n_k` for k = 3 (0.4 m at 5 cm) up to the ground's top octave. A biome doesn't add noise: it plays those notes.
The ground stops at the micro engine's top (`SDFCascade::micro_top_k`, k = 19 at 5 cm: 26.2 km, the first note at the
3.3 km cap), so the climate is micro: a biome plays notes up to 26.2 km, and a range past it is clipped there. The
fields placing the biomes (temperature, humidity, wind: 105-419 km) are macro and may reach lod_count + 4
(`Ground::field_k_max`), above the ground's top, until the planned macro engine computes them.

1. **Fields** (`Field`): a few slow scalars, each a sum of plain perlin octaves of its own seed, plus linear terms
   (`coef * the ground` or `coef * an earlier field`, in list order): temperature, humidity, wind, the warp fields
   wx / wy / wz. A term on the ground makes a field follow the height (colder higher up).
2. **Weights** (`Climate::weights`): biome b's indicator `r_b` is the product of its edges' smoothsteps on the fields
   (`where`: min / max with a fade). Priority layering, `w_b = rest * r_b`, `rest -= w_b`, the last biome taking the
   rest, is a partition of unity without a division.
3. **The ground** at a point: each note k some biome names is inserted as
   `A_k * phi_b(n_k(p + warp))` blended over the biomes,
   - `A_k = sum_b w_b * x_{b,k} * amp_k` (the biome's multiplier times the note's own amplitude, lambda / 8 capped at
     3.3 km),
   - `phi_b(n) = c0 + c1 n + c2 |n| + c3 n^2` (the biome's shape: plain, billow, ridged, square), inserted without its
     mean,
   - times `clamp(gain (1 - c), 0, 1)` with a multifractal (c: the coarser notes' weighted signal, Musgrave's chain),
   - read at the point moved by `warp_m * (wx, wy, wz)` with a warp.
   Notes no biome names keep their amplitude, plain. `ground_stretch` reads chosen notes squeezed along the base's up.
4. **The biome's curve** (`Biome::curve`): at each sample, `h + sum_b w_b * C_b(h)`, `C_b = R_b - h` through the
   biome's points `[h, R(h)]` (libnoise's cubic, constant past the ends): the biome's height remap, its "LUT".
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

- **Parse and validate** (`climate_parse.cpp`, `climate_validate.cpp`): metres to octaves (`from_m` .. `to_m` ranges
  keep the octaves the LOD count has, so one table covers 15 and 25 LODs; a single wavelength outside is refused),
  against `SDFCascade::climate_ground()`. Refused: more than `MAX_FIELDS` (24) fields or `MAX_BIOMES` (16) biomes, a
  multiplier past the ground's amplitude cap (`max_ground_mult`), a curve moving the ground past it, a curve whose
  offset is steeper than `MAX_CURVE_SLOPE` (4: the skirts and the gradient scale with it), a curve whose points all
  lie past the heights the ground reaches (a constant offset: a wrong unit or sign), an `"offset"`
  without a multifractal (bands are inserted without their mean, so only the state reads it), a field's multiplier
  without a ground one, multipliers on a non-LINEAR4 ground, a carry past `Cascade::MAX_CARRY`.
- **In the cascade** (`Climate::combined_spec`): the fields are `Spec::fields`, LINEAR4 groups carried down with the
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
  own gradient needs the ground's second derivatives, which LINEAR4 doesn't carry.
- **One climate per point, none per sample** (the carry): a run's last level, its samples, evaluates no climate. The
  level above derives it from the climate its own bands read (`Cascade::Modulation::derive`; on the GPU at the end of
  `cascade_mods`) into the carry groups (`Cascade::carry_slot`): the last level's `BandMod` parts and the samples'
  terms (`Climate::SampleCarry`: each curved biome's weight, the pick, copied from the floor parent instead of
  interpolated). H carries them down with the ground. So a LOD-L chunk's samples read the last octave's climate from
  Λ_{L+1}: the model's rule above, not an approximation of another one.
- **The curve** (`Climate::ground_value`, GLSL `climate_ground`, through `SDFCascade::ground_channels` on every path:
  chunk samples, point items, `ground_along`, the report): `h + sum_b w_b * C_b(h)` with the carried weights. The
  skirts take its step (`Climate::ground_step`: the ground's step times 1 + the curves' largest slope, plus each
  weight's change times its curve's bound; unit test: the L | L+1 step stays within the skirt on `example.json`,
  `shaped_test.json` and `world.json`).
- **GPU only, for speed** (the CPU mirrors them today; the plain-oracle stage moves them GPU-side): which levels carry
  the fields and the carry (`Cascade::FieldRange`: `carried`, `inserted`, `carries`, `batch`, `batch_carry`), how many
  carry groups a last level loads (`Modulation::carry_used`), the three generated views (`CascadeGLSL::View`: ground
  alone, ground + fields + carry for the levels that derive, ground + carry for a run's last level) on planar scratch,
  and skipping the empty top (`Spec::run_top`). None changes a value: the CPU's all-fields run gives the same ground
  bits (`test_climate.cpp`).
- **Stretched octaves** (`"ground_stretch"`; `Band::stretch`, at most k − 3): through the base
  (`BaseSurface::stretched` / `stretch_gradient`), the noise is read at p − (1 − 2^-j) d n and its gradient's up
  component scaled by 2^−j. Flat: datum + (y − datum) / 2^j exactly (GLSL `base_stretched`); sphere: CPU and GLSL with
  FrameSpike's `stretched_sphere`, but its GPU path needs the regions of the band's insertion LOD, which `BaseRegions`
  doesn't upload yet, so `SDFCascade` refuses such a climate on a sphere. Why: the ground is 3D, and moving up a column
  meets other noise (needles, folds where dh/dy passes 1); on the Mont Blanc match a stretch of 4 from 100 m up took the
  100 m octave's amplitude from 8.7 m to 2.3 m (real: 2.6 m) and the slopes over 75° from 0.4% to none.
- **Clouds** (a biome's `"clouds": {"cover": 0..1, "type": 0..1}`, `Climate::BiomeClouds`): the volumetric clouds' cover
  and type over the biome, blended by the weights on a coarse map round the camera (`CloudClimate`, `docs/look.md`).
  They touch no cascade pass or shader.
- **Tables** (`adapter/godot/project/climate/`): `world.json` (eight biomes on temperature x humidity x wind,
  105-419 km octaves, needs 25 LODs; each biome's notes once fitted to a real Copernicus heightmap, see git history of
  `tools/landforms/`; dunes and mountain carry their height curve), `example.json` (continent and terrain-type fields
  choosing mountains, hills, badlands and plains, two curves), `shaped_test.json` (every operation, CascadeSpike `4s`),
  `plain.json` (amplitudes only; pins every slot, the SDF, the skirts and both level shaders' text: a table without
  shapes generates the old text bit for bit), `fieldless_test.json` (one biome, no field: CascadeSpike `4d`).
- **Checks**: `tools/unit_test.sh "Climate*"`; `tools/climate_report.sh` (pure CPU: biome maps, the pick drift, where
  biomes meet with a `--spawn`, relief per biome, gradient error, field maps) before any GPU run;
  `cascade_test.sh --sets=4,4c,4f,4p,4s,4d` and `--sets=4w --lods=25` (G1 on the carried channels, G3, points, G10 the
  pick); `discovery_audit.sh -- --voxel=0.05 --climate=... --discovery_compare=1`; `origin_parity.sh --voxel=0.05 --
  ... --climate=...`.

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

`adapter/godot/project/debug/biome_mixer/`, `tools/biome_mixer.sh`: one biome built from the ground's octaves in a node
graph (GraphEdit), raymarched live beside it; a design tool, not the engine's path (the table is the engine's input;
turning a graph into a table's biome is the next step).

- **Nodes** (`kinds.gd`): *Note* (the ground's octave k, `make_octave(SEED, ground_top - k, k)` ported in `octaves.gd`,
  pinned against `test_exact_noise.cpp`: every biome plays the same notes, so a note has no seed of its own; an
  optional warp input moves where it is read, `warp_m` metres per unit; `stretch_y` reads it at y / stretch from the
  datum, as `ground_stretch`), *Shape* (`c0 + c1 n + c2 |n| + c3 n^2` with `|n|` smoothed as `Cascade::smooth_abs`;
  presets, scale and offset as `climate_parse.cpp`), *Multifractal* (`w = clamp(gain * chain in)`, the signal times w,
  chain out `w * phi` to the next finer note: the engine's state, wired), *Climate warp* (the climate's three warp
  fields, a model's `climate_warp`, world.json's wx, wy, wz by default (`Kinds.CLIMATE_WARP`): no per-biome settings,
  one table slot however many nodes; the preview reads its 12 octaves per pixel, the engine interpolates them from
  their coarse lattices), *Note warp* (preview only, the cheap warp: x, y, z are Notes read earlier in the table,
  `F * N` of their rows, so notes the biome plays cost no read; a note warped by a finer one is warned of; the engine
  would need warp notes in a table's band, the chain rule from their gradients), *Mix* (rows of volume x the ground's
  amplitude lambda / 8 capped at 3.3 km, mute, solo, centre = the shape's mean out as the engine inserts bands), *Add*
  / *Multiply* / *Constant*, *Height curve* (a remap R(h) through points: a biome's `"curve"` in the engine, the same
  points, libnoise's cubic of R - h, constant past the ends; its bound and slope, and its editor's in axis, span its
  input's bound (`curve_in`): the mountain's ±7.2 km, its points ±3.8 km), *Carve* (preview only: air where a field
  passes a threshold inside a depth band), *Output* (height, holes, flat and steep colours with the engine's slope
  blend).
- **Shader** (`compile.gd`, `shader_lib.gd`): the graph's structure becomes code, every parameter a uniform (a slider
  redraws without a recompile). Every note and warp octave is a row of one uniform table read in one loop, and the
  march, bisection, normal and shadow share one `density()` call: inlined per note and per call, the driver took
  31.8 s to compile the mountain, 0.3 s since. Notes fade out under two pixels (a Shape fades around its mean, so a
  faded note moves no height); the march starts under the ground's bound and steps by the density over a slope
  estimate (independent notes in quadrature; overshoots are bisected), twice the estimate's safe step: the estimate is
  a worst case, and the imported biomes show no holes at twice it. The bisection ends on its outside end: the inside
  one's depth grows with the step, and a shadow ray started from it underground drew rings of acne on smooth slopes.
  The preview draws only on a change (low-processor mode, `UPDATE_ONCE`), in two layers (`preview.gd`): 1/3 of the
  resolution on every change, then the chosen one over it in strips fitted to ~25 ms a frame, which any edit cancels.
  Measured on the RTX 3070 laptop, world.json's mountain at 810 x 751 with shadows: the whole view 200 ms a frame at
  step 1, 150 ms at step 2; in strips, no frame over 38 ms and the view sharp in ~0.3 s.
- **Budget**: over the view, what the biome reads per point (its notes, the climate warp's octaves per axis, the notes
  moved by a Note warp for free, against plain ground's), its relief bound, the row that moves the most and how long
  the view took to sharpen; each Mix row shows its swing (volume x lambda / 8 x the shaped note's bound) and its share
  of the Mix's. A row's x takes it out with the nodes only it used (`Kinds.cut_row`); a node's x or Delete removes it
  and the Mix row it played (`Kinds.cut_nodes`); `--check` checks both on the mountain and plain ground.
- **Import** (`import_table.gd`): a biome of `world.json` as a graph (its ground multipliers, shapes, multifractal
  chain, warp fields, `ground_stretch`, its curve, colours). Biomes save to `climate/biomes/*.biome.json`.
