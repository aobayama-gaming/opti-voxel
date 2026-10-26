# Climate

Climate fields, biomes, shaped bands, stretched octaves and regions. Landform tables: `docs/landforms.md`.

- **Climate** (`src/climate.{hpp,cpp}`, `climate_modulator.cpp`, `climate_parse.cpp` / `climate_validate.cpp`,
  `climate_glsl.cpp`; `VoxelTerrain.climate_path` / `--climate=PATH`, bench and player; off by
  default): a climate is data, a JSON table (format above `parse()`, example `adapter/godot/project/climate/example.json`, tuned for
  5 cm: at 0.5 m its x2 mountains exceed the cap and the table is refused) checked against
  `SDFCascade::climate_ground()`: up to `MAX_FIELDS` (24) fields, each a LINEAR4 cascade group (`Field::bands`) of any octaves in
  the ground's range (seeds `make_octave(field seed, k)`, keyed by k; a `from_m` .. `to_m` range of bands or multipliers
  keeps the octaves the LOD count has, each at the same amplitude, so one table covers the whole spectrum at 15 and 25
  LODs; a single wavelength outside is refused), plus linear terms `coef * (the ground or an earlier
  field)` applied after the cascade, in list order (the lapse is a ground term). Biomes (<= `MAX_BIOMES`, 16): regions
  over any fields, an indicator = product of edge smoothsteps; priority layering (w_b = rest * r_b, the last biome takes
  the rest) is a partition of unity for any regions without a division; dual numbers, every product flushed under 2^-64.
  `pick`: argmax, ties by `hash_cell` of the absolute voxel (the same from every LOD). **One climate per point, none per sample:**
  the climate (`Climate::state`: field terms, then weights; GLSL `glsl_weights`) is evaluated once per point of each
  level above a run's last, from its channels before the level's insertions (the LOD rule: octave k on Λ_m reads the
  climate of H(V_{m+1}), low-passed at k). A run's last level, its samples (the most points of a run), evaluates none:
  the level above derives it from the climate its own bands read, the same evaluation (`Cascade::Modulation::derive`,
  `Modulator`; on the GPU one function, `cascade_mods` with the derive at its end), into the
  carry groups (`Cascade::carry_slot`, LINEAR4 like a field): the packed `BandMod` parts of the last level's octave
  (only the members some biome makes non-zero) and the samples' terms (`Climate::SampleCarry`: the height terms' sum,
  the terraced biomes' weights, the pick, copied from the floor parent instead of interpolated). H carries them down
  with the ground, and the samples interpolate. So a LOD-L chunk's samples are a pure function of (point, L), the same
  from every chunk and point query of that LOD, but they read the last octave's climate from Λ_{L+1} before its
  insertions (H(V_{L+2})): not the bits a finer LOD's level L holds there (no LOD shows those; the whole octave is the step a skirt covers). The top level
  reads a derive on an empty level above it (`Spec::run_top`). Measured why (CascadeSpike, world.json, LOD 0): the
  samples' level was 11.7 of the 16.2 ms of a 128-chunk batch with the weights evaluated three times per sample (the
  modulation 4.0 ms, the height terms 2.6, carrying 72 floats 3.7). Carrying the climate from every level to the next
  instead (every level reading a derive) made the levels above the samples 2-5x dearer than the evaluation it saved
  (they moved the fields and the carry; example.json +47%, point items +100-300%): only the last level reads one. No per-biome mean height: one lattice can't carry it without the coarser LODs
  missing it (a step no skirt hides); it needs its own design.
  **In the cascade** (`Climate::combined_spec`): the fields are `Spec::fields`, LINEAR4 groups carried with the ground
  (`Cascade::field_slot`: after the ground's 8 slots on the CPU, after its 4 channels on the GPU: `Spec::gpu_base`, `Spec::slot_of_gpu`).
  A band some biome multiplies is modulated (`Cascade::Modulation`, `Climate::Modulator`): at each point of its level,
  from the channels before that level's insertions (H(V_{m+1}): no band reads one of its own level), A = sum_b w_b amp_b
  and the band inserts A n, slopes n.s (A 2^-3) + n.w dA. So ground octave k follows the biome map of the fields'
  octaves coarser than k only: a region shapes the ground up to about its own size (a plain patch inside a mountain
  region keeps the region's big octaves), and flattening the continental octaves takes continental regions. The
  example's fields have km and continental octaves (to the top) and every biome multiplies the whole spectrum (raising
  stops at the 3.3 km cap): at 25 LODs, mountain slopes 4-5x the plain's on 26 and 210 km squares (a plain alone:
  slope 0.018, from 0.19 bare); the old example's few-octave multipliers moved slopes 0-45%, mountains +2%.
  `climate_report.sh`'s relief table (height std, mean slope per biome, `ground_height.png`) measures it. `fields_at` reads V_m of the combined spec, so it is a pure
  function of (lattice point, m). `SDFCascade` skirts take the largest amplitude any biome gives an octave
  (`Climate::max_amp`). On the GPU the climate code is generated from the table (`Modulator::glsl`, op for op) and
  `CascadePasses` compiles three views (`CascadeGLSL::View`, the scratch planar: vec4 v of every slot together, so
  the ground's loads stay contiguous): the ground alone, the ground + fields + carry (the levels that derive) and the
  ground + carry (a run's last level). A chunk of LOD L carries its fields at level m only where
  `Cascade::FieldRange::carried` (m >= lowest(L): one above the finest level reading a derive, the modulated ground
  levels or, with `sample`, the samples) and the carry where `carries` (one level lower), and inserts the fields'
  own octaves only above lowest(L) (`FieldRange::inserted`: that level's bands and derive read the fields before its
  insertions, and no level below carries them; GPU `g_field_bands`, which also gates the multifractal state's write,
  the field bands' BandMods and the fields' stores there); the CPU's ground queries follow it too, `fields_at` carries
  and inserts them everywhere. Measured (CascadeSpike, 128-chunk batches, 25 LODs, 3 interleaved runs per side vs the
  twice-evaluated derive): world.json -39 to -46% at every LOD (the level above the samples 5.7 -> 2.1 ms at LOD 0), the
  example (4c) -15 to -36%, point items +1 to +4%. LOD 12, the outlier, was where world.json's 100 m - 1.6 km field
  octaves are inserted on a run's two largest climate levels (6-8 of them on Λ_13 / Λ_14): 12.1 -> 6.5 ms, now 1.2x
  LOD 0 (1.4x before). Next: the climate per point of the levels above (j = 1 is still 1.2x the samples' level). A dispatch takes the fields view where `FieldRange::batch` says,
  else the samples' view where `batch_carry` says (`record_level`'s `p_lods`, the batch's LOD range). A region's core is
  exact (`Climate::core_regions`, `docs/landforms.md`). With the carry (CascadeSpike, 128-chunk batches vs dev): world.json
  -42 to -55% at every LOD, the coarse example (4c) -27 to -33% at LODs 16-24 but +7 to +30% at LODs 0-8, where the
  derive level (j = 1) evaluated the climate twice (before and after its insertions): it now derives from the first
  evaluation (one climate per point; the samples miss the field octave inserted on Λ_{L+1}, so the step between two LODs
  is the field octave k = L + 5, `Climate::ground_step`). Cost before the carry
  (CascadeSpike, 128-chunk batches, example climate, before the biome materials, `docs/materials.md`): fields without multipliers 1.00x; with them 1.09x at LODs 0-4,
  1.00x at 20-24, 2.2-3.5x at LODs 8-16 (their own level is modulated: the weights run on every sample, +9-19 us per
  chunk, once per mesh). The bigger `Ch` (20 floats) is what made a single shader 1.4x everywhere (registers, not work).
  Forest5cm fly: steady +2%, tree placement +190 ms (the CPU traces carry the fields), the first run of a new table
  pays its shaders' compile (cold pipeline cache). A biome that multiplies the ground with no field still takes the
  modulated variant (`CascadeGLSL::field_range` keys on the modulation, not on the fields; set `4d`,
  `fieldless_test.json`, a unit test): the GPU drew the bare ground under a flat desert before. Refused: a field's
  multiplier without a ground one (it would never reach the mesh), two bands of one group on one level, multipliers on
  a non-LINEAR4 ground (checked once, in `validate`). Checks: `cascade_test.sh --sets=4,4f,4c` (G1 on the carried
  channels, G3 at the unmodulated levels, points), `discovery_audit.sh -- --voxel=0.05 --climate=... --discovery_compare=1`,
  `origin_parity.sh --voxel=0.05 -- ... --climate=...`, a validated / audited bench load.
- **Shaped bands** (`src/climate_modulator.cpp`, `src/climate_shape.cpp`, GLSL `src/cascade_shaped_glsl.cpp`; the table keys above
  `Climate::parse`, `src/climate_parse.cpp`): libnoise's
  complex planet's operations as cascade data, every one optional per biome and band range, CPU and generated GLSL op
  for op. (1) A band's **shape** `c0 + cn n + cabs |n| + csq n^2` (`Climate::Shape`: plain, billow, ridged, square,
  scale / offset, or coeffs) is inserted without its mean (`Cascade::NOISE_ABS_MEAN` / `NOISE_SQ_MEAN`, measured: a
  coarser LOD dropping the band makes no per-biome step); `|n|` is `Cascade::smooth_abs`, `sqrt(n^2 + 0.01) - 0.1`
  (with the bare kink the Hermite subdivision rang along the lattice axes: ridged bands' gradient channels 21% off the
  values' finite differences against 2.5% plain, and a multifractal's state streaked). The `Modulation` returns a
  `Cascade::BandMod` per group: three amplitudes, the state, the warp, a feature mask. (2) **Multifractal** (`"multifractal":
  gain`): the octave is weighted by `clamp(gain (1 - c))`, c a band-less field group (`Modulation::state_group`, the
  table's fields + 1, so at most 23 fields) that the ground's band overwrites with the complement of its weighted
  uncentred signal (Musgrave's ridged multifractal: gain 2 + ridged; hybrid: 1 + plain + offset). (3) **Warp**
  (`"warp_m"` + three `"warp_fields"`): the band's noise is read at the point displaced by the fields' values (int64 +
  fraction, `WARP_LIMIT`), its gradient by the chain rule (the fields' gradients). Slope damping (Quilez, de Carpentier)
  and the swiss turbulence's warp by the ground's slope were built and dropped: their own gradient needs the ground's
  second derivatives, which LINEAR4 doesn't carry, and without it the finer levels' values streak (climate report:
  6-13% gradient error, visible lines). (4) **Height terms** (`"heights"`: `height_m * curve(field)`, libnoise's Curve)
  and per-biome **terraces** (libnoise's Terrace, blended by the weight) are added to the ground at each sample,
  after the cascade (`Climate::ground_value` / GLSL `climate_ground`, through `SDFCascade::ground_channels` on every
  path: chunk samples, point items, `ground_y`, the report): the terms' sum and the terraced weights derived on the
  level above the sample (`SampleCarry`), so the fields are carried down to it, point items too
  (`FieldRange::sample`), and the skirts take the terms' step (`Climate::ground_step`: the height fields'
  terms, twice that with terraces for their slope, plus the terrace blend's change with the weights, each terraced biome's
  weight step times a quarter of its step; unit test: the L | L+1 step stays within the skirt on the example and
  `shaped_test.json`). Refused: an `"offset"` without a multifractal (plain, billow and square bands are inserted without
  their mean, so only the state reads it), a curve the field's values never reach, a height term over the ground's
  amplitude cap. That is what
  lifts land per region (the per-biome mean height the plain cascade couldn't do). (5) **Cell noise** (`"noise":
  "cell"`): Worley F1 over 3 x 3 x 3 jittered cells (`ExactNoise::cell3` / `glsl_cell3`, FrameSpike's `cell3_k*`),
  centred by `CELL_MEAN`, read through height terms; creased (cones, cell borders), it measures 7.2% gradient error on
  25-100 m dunes, no streaks. A table without these keys is the old path bit for bit (unit
  test: `adapter/godot/project/climate/plain.json`, the example before them, pins every slot, the SDF, the skirts and both level
  shaders' text). `adapter/godot/project/climate/example.json` is now a world composed like libnoise's (continent curve, a twisted
  terrain type choosing ridged-multifractal mountains lifted inland, billow hills, terraced badlands, low plains,
  ridged river channels, three warp fields); `shaped_test.json` exercises every operation (CascadeSpike set `4s`; `4p`
  the plain table). The look target, a numpy port of libnoise's graph, and the gallery of the techniques are outside
  the repo (`~/Drive/perso/opti-voxel/terrain-research-2026-10-21/`). Checks: CascadeSpike 4c / 4p / 4s G1, G3, points
  and G10 (0 differing), FrameSpike, discovery audit (11 poses, compare) and origin parity on the example climate, a
  validated forest5cm load. Cost (128-chunk batches): the example's passes 4.2-4.5x the bare cascade at LODs 0-8,
  5.6-6.5x at 12-24 (the plain example 2.0-3.9x): 7 fields + the state carried, and level 0 computes the weights three
  times (modulation, height terms, pick: one shared evaluation is the obvious next step). Forest5cm vs the plain example
  (2 runs, GPU throttled 73% both sides): SDF pass per batch +17%, load +31%, steady render GPU +12% (+8% triangles:
  rougher ground), fly p50 out -1%, back -10%. `climate_report.sh` gained `ground_shaded.png` / `ground_normals.png`
  (hillshade from the heights and from the gradient channels), the per-biome gradient error, sea colours and
  `--center=U,V`.
- **Landforms** (`adapter/godot/project/climate/landforms/*.json`, `docs/landforms.md`, `tools/landforms/`): mountain, flat desert,
  dunes, mangrove, island, cliff, valley and caldera tables, each matched against a real Copernicus GLO-30 heightmap.
  - `fetch_dem.py` downloads the public AWS tiles; the sites are in `sites.json`.
  - `climate_report.sh --lattice=M` samples the table on the same grid and writes `heights.f32` / `fields.f32` /
    `dhdy.f32` / `picks.u8` plus `heights.json`.
  - `compare.py` and `terrain_stats.py` give octave amplitudes, slope percentiles, hypsometry, curvature skew,
    anisotropy and the share of folded columns, plus a sheet.
  - `match.py` fits one multiplier per octave, `--hypso` a height remap, `--texture` a biome without its features.
  - `world.py` composes them into `adapter/godot/project/climate/world.json` (25 LODs): each landform behind a region of temperature x
    humidity x wind, shared fields merged, the continental octaves (>= the axes' finest, 105 km) off in every table.
    Each region's core equals its own table, bit for bit (`test_climate_regions.cpp`); `climate_report.sh` lists a
    core `--spawn` per region.
  - The findings (which tool for which feature) are in the doc.
- **Stretched octaves** (`"ground_stretch"`, a band's `"stretch_y"`; `Band::stretch`, at most k − 3; the band table's
  `info.w`): through the base (`BaseSurface::stretched` / `stretch_gradient`), the noise is read at p − (1 − 2^-j) d n
  and its gradient's up component scaled by 2^−j. Flat: datum + (y − datum) / 2^j exactly (GLSL `base_stretched`),
  centred on the datum; sphere: the base sample's float offset (`base_stretched_at`, CPU and GLSL with FrameSpike's
  `stretched_sphere`; its GPU path needs the planet PR's regions). A table without stretch generates the same shader
  text (the `plain.json` pin).
  - Why: the ground is 3D, so moving up a column the surface meets other noise: amplitude a heightmap doesn't have,
    needles, and folds where dh/dy passes 1. On the Mont Blanc match, unstretched: 8.7 m at the 100 m octave against
    2.6 m real, 0.4% of slopes over 75°, spectrum score 0.40 (folds 0.05%). A stretch of 4 from 100 m up: 2.3 m, none,
    0.13.
- **Regions** (`"regions"`, `Climate::Region`, `Climate::edges_value` / `glsl_edges_value`, evaluated by the ground only: `ground_value` / `climate_ground`, over `Table::gated`):
  a named product of edges, for several landforms in one table picked by macro climate fields. A biome's `"region"` puts
  its edges first; a height term's multiplies the term (a curve required: `ground_step` adds the gate's step).
  `region_test.json` (CascadeSpike set `4r`, 11 field groups) equals `valley.json` bit for bit in its cold core. The
  field group cap is 24 (the GPU's arrays are sized per spec); every group is carried at every point. Details:
  `docs/landforms.md`, "Using a landform in a world".
