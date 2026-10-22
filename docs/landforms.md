# Landforms from real heightmaps

Eight landform tables in `project/climate/landforms/` were each matched against a real heightmap. They are built from
the cascade's own tools: Perlin octaves, shaped bands, multifractal weights, field warps, height terms, cell noise,
terraces and stretched octaves. This page explains:

- how the matching works;
- what each landform takes, with the real and generated statistics side by side;
- which tool fits which feature: what worked and what didn't.

The results, as comparison sheets and in-engine screenshots, are outside the repo in
`~/Drive/perso/opti-voxel/landforms-2026-10-22/`.

## The loop

1. **Real data.** `uv run tools/landforms/fetch_dem.py [site ...]` downloads Copernicus GLO-30 DSM tiles. The source is
   the public AWS bucket `copernicus-dem-30m` and needs no account. The script crops each site of
   `tools/landforms/sites.json` to a square of local metres, resampled bilinearly. The spacing is a power of two of
   5 cm voxels, so the engine can sample the same grid: 25.6 m or 51.2 m, 512 x 512.
2. **Generated data.** `tools/climate_report.sh --lattice=M` samples a climate table on lattice Λ_M, each column at
   its ground surface, using the CPU cascade at a few µs per point.
   - It writes `heights.f32`: metres above the datum.
   - It also writes `fields.f32` (the field values), `dhdy.f32` (the ground's vertical slope) and `picks.u8`.
3. **Comparison.** `uv run tools/landforms/compare.py --real=SITE --gen=RUN` (statistics in `terrain_stats.py`)
   prints the table below and draws a sheet with:
   - hillshades on the same colour scale, and oblique views;
   - a cross-section;
   - the octave spectrum, slope histogram, hypsometric curve and curvature histogram.
4. **Fitting.** `uv run tools/landforms/match.py SITE TABLE [--fit=N] [--hypso] [--texture=BIOME] [--center=X,Z]`
   runs steps 2 and 3, then fits the table. It rewrites the table in place, one line per band or multiplier.
   - Each round multiplies every single-wavelength ground multiplier by (real / generated amplitude at that
     octave)^0.8. The fitter keeps under the 3.3 km amplitude cap (`max_x`).
   - Octaves leak into their neighbours, through the shapes' harmonics and the multifractal weights, so 2-3 rounds
     are needed.
   - `--hypso` refits a height remap curve so the height distribution matches the real one (below).
   - `--texture` fits a biome's octaves on a copy of the table without its features.
   - `tools/landforms/survey.py` prints every site's statistics in one table.

### What the statistics measure

| statistic | what it tells | how a table moves it |
|---|---|---|
| octave amplitudes | the standard deviation of the height in one-octave annuli of the windowed spectrum: what each cascade octave must carry | one multiplier per octave (`"wavelength_m"` entries), fitted |
| spectral slope | d log amplitude / d log wavelength. 1 is self-similar (the bare ground everywhere); real terrain is 1.2-1.6 at fine scales and 0-0.9 at coarse ones | the fitted multipliers |
| slope percentiles, flat / steep / cliff share | walkability: under 2° flat, over 35° steep, over 55° cliff | amplitudes; shapes (ridged makes steep flanks); height-term curves (cliffs) |
| hypsometric integral | the mean of the normalised height: low for plateaus with incised valleys and for islands, high for domes | a height remap (`--hypso`), height terms |
| curvature skew | the skew of the 2-cell Laplacian. Negative: sharp crests, round valleys (ridged). Positive: sharp valleys, round tops (billow, incised channels) | the band shapes |
| anisotropy | the ratio of the principal axes of the spectral energy | not reachable yet (below) |
| fold share (generated only) | dh/dy > 1 of the 3D ground: the surface folds into overhangs | `ground_stretch` |

**Scores** (0 means identical):
- **spectrum:** the RMS of log2(generated / real) over the octaves from 4 cells to a quarter of the map;
- **slopes:** the mean gap between the two slope distributions' percentiles, in degrees;
- **hypsometry:** the mean gap between the two hypsometric curves;
- **curvature:** the gap between the curvature skews.

## The landforms

Each table's header comment says how it is built. The real site comes first, then the result. "Bare" is the ground
with no table: every octave at wavelength / 8.

| landform | real site | spectrum | slopes ° | hypsometry | curvature | bare spectrum / slopes |
|---|---|---|---|---|---|---|
| mountain | Mont Blanc massif (51.2 m) | 0.12 | 3.3 | 0.08 | 0.13 | 1.26 / 13.4 |
| flat desert | Selima Sand Sheet (25.6 m) | 0.18 | 0.02 | 0.03 | 0.10 | 5.48 / 17.0 |
| dunes | Namib Sand Sea, star dunes (25.6 m) | 0.21 | 2.0 | 0.05 | 0.69 | 1.08 / 7.8 |
| mangrove | Sundarbans (25.6 m) | 0.46 | 0.47 | 0.41 | 0.25 | 4.26 / 16.6 |
| island | Moorea (51.2 m) | 0.25 | 1.8 | 0.27 | 0.64 | 0.51 / 13.4 |
| cliff | Drakensberg Amphitheatre (25.6 m) | 1.02 | 2.6 | 0.18 | 0.42 | 0.69 / 3.6 |
| valley | Yosemite Valley (25.6 m) | 0.33 | 1.8 | 0.10 | 0.33 | 1.45 / 4.6 |
| caldera | Crater Lake (25.6 m) | 0.60 | 3.0 | 0.14 | 0.39 | 0.83 / 8.3 |

**Where the bare ground scores better, the statistics can't see the feature.**
- **Cliff:** the bare ground's spectrum score (0.69) beats the cliff table's (1.02). The Drakensberg's dissected lowland
  is close to self-similar, and the 700 m step the table adds puts energy into every octave.
- **Valley and caldera:** the bare ground wins on hypsometry (valley 0.08 against 0.10, caldera 0.08 against 0.14).
- **The reason:** the bare ground has no plateau, no trough and no crater at all. The scores measure texture and
  distributions, not whether the feature is there.
- **How to judge a feature:** read the sheets' cross-sections and the engine shots for it.

- **Mountain.** The fitted spectrum ends within 0.66-0.96 of the real one at every octave.
  - Median slope 25° against 30°, 2% cliffs against 3%, curvature skew −0.46 against −0.58. Relief is 3.9 km against
    4.0 km.
  - Built from Musgrave's ridged multifractal from 3.2 km down to 200 m, then plain 6.5 and 13 km octaves, each warped
    by a quarter of its wavelength.
  - The 13 km octave sits at the ground's cap. That is why the coarsest amplitude falls short (0.66).
  - In the engine: massifs with aretes and smooth valley floors. The Alps' long glacial troughs are missing (below).
- **Flat desert.** Every octave at 1-5% of the bare ground's. 100% of slopes under 2°, as on the real sand sheet.
  - The bare ground is 30x too rough at every scale: it is the worst match of all eight sites.
  - A 1.6-6.5 km field lays darker gravel plains (serir) among the sand.
- **Dunes.** The real spectrum peaks at 1.6 km and falls on both sides: a dune field has a wavelength. The fitted
  multipliers give the peak.
  - Ridged multifractal octaves up to 1.6 km make sharp crests, warped by a third of their wavelength so the arms bend.
  - The height remap flattens the corridors between dunes. Flat share is 0.10 against 0.34: the remap's slope bound
    keeps some texture there.
- **Mangrove.** About 5 m of relief, 89% flat, cut by tidal channels.
  - Each channel size is **one** ridged octave of its own field (6.5 km channels, 1.6 km creeks), read through a height
    term with a smoothstep bank and a flat bed.
  - The banks are set from the field's quantiles: the field is 3D noise, so a slice through the ground reads a narrower
    range than ±1.
  - The bed and bank biomes are water-coloured (the engine draws no water yet).
  - Blobby relief from the canopy and mud is under a metre (billow octaves at 0.3-6% of the bare ground). It reads as
    flat from more than a few metres up. The real blobs are the trees: they need instances, not ground.
- **Island.** The volcano is a height term on a 26 km cell noise field (Worley F1: one cone per cell), with a concave
  profile and a shallow shelf.
  - A 1.6-13 km noise added to it breaks the coast. Ridged multifractal octaves erode the cone, but only above 300 m
    (below).
  - The hypsometry stays at 0.30 against 0.03: the real island is narrow, with 80% of the map at sea level.
  - Moorea's drowned valleys (bays) are not reproduced.
- **Cliff.** A 700 m step across a narrow range of a 3-13 km field (scarp) gives the escarpment's trace. The face is
  sheer: 3.5% cliffs, against 0.6% on the real site, where the amphitheatre is a smaller part of the map.
  - Finer bands of the same field, stretched the same way, wiggle the face line into buttresses and gullies running
    straight down the wall.
  - The lowland is ridged multifractal (median slope 21.5° against 19.8°).
  - The spectrum score stays at 1.0 because the step puts energy into every octave. It is a feature, not texture.
- **Valley.** One ridged octave of a 13 km field (a clean meandering axis) through a U-shaped curve gives a flat floor
  900 m under the upland, with steep walls.
  - The upland is billow, so the domes come out round-topped.
  - The 3.2 and 6.5 km upland octaves are fixed at 0.4 and 0.2. Fitted, they took the trough's energy, made the
    uplands twice too high and doubled the slopes (32° median against 16.5°).
- **Caldera.** A cell-noise height term with a flat lake floor, a 600 m inner wall and a concave outer flank. A
  1.6-6.5 km noise added to the cell field makes the rim irregular. The flanks are warped ridged octaves.
  - The cell field is stretched 8 times in y. Without that, the 3D cell's slice at the ground cuts the crater's profile
    differently at every height.

## Findings: which tool fits which feature

### Each feature needs its own tool

- **One multiplier per octave** fits any spectrum: a rolling peak (dunes), self-similar relief (mountains) or nearly
  nothing (desert).
  - The real spectra are far from the bare ground's slope of 1. Desert and mangrove are 30x weaker at every scale;
    dunes have a peak.
  - A multiplier on a `from_m`..`to_m` range can't fit a spectrum. Give each octave its own `"wavelength_m"` entry and
    let `match.py` fit it.
- **Shapes** set the curvature skew: ridged gives crests (negative skew), billow gives domes and creases (positive skew).
  - Matching the sign of the real skew matters more than matching its size.
  - Ridged multifractal (gain 2) keeps fine detail on the crests and leaves valleys smooth.
  - The statistics don't demand it. On the unstretched mountain, dropping it moved the spectrum score from 0.76 to 0.75
    and *improved* the curvature score, from 0.75 to 0.31. It stays in for the look: smooth floors under sharp aretes.
    A measure of where the roughness sits (crests vs valleys) would settle it.
  - It also leaks energy into the octaves below it, so fit it after setting the shapes.
- **Warps** bend straight noise ridges into ranges and dune arms. A quarter of each octave's wavelength is enough. On the
  unstretched mountain, removing them raised the curvature score from 0.75 to 1.87.
- **Height terms** carry every sharp feature: a cliff (smoothstep), a trough (U curve on a ridged field), a crater rim,
  a cone (cell noise), a channel (narrow smoothstep).
  - Octaves can't make these: a feature narrower than its octave's wavelength / 8 aliases away.
  - A curve makes a feature only along an iso-line of its field.
- **One octave per linear feature.** A single ridged octave's zero crossings are clean meandering lines: channels,
  valley axes. Summing octaves of a ridged field makes a web of rings (mangrove, first try).
- **Cell noise** gives one radial feature per cell: cones, calderas. A coast or rim noise added to the field breaks the
  circles.
- **Remap curves:** libnoise's cubic curve treats its points as evenly spaced. On uneven points it overshoots between
  them into terraces and bumps. `tools/landforms/curves.py` builds even smoothstep curves; `--hypso` resamples its rank
  mapping on an even grid.

### The ground is 3D: stretch it for steep landforms

The ground is 3D noise, so dh/dy is as large as dh/dx. Wherever a landform is steep, it folds.

- **Before stretching:** 6% of the mountain's columns had dh/dy > 1. Those are needles and overhangs, and 18 m of
  spurious amplitude at 100 m, against 2.6 m real.
- **The fix, `ground_stretch` / `"stretch_y"`.** The noise is read at y / 2^j.
  - Octave k's wavelength along y becomes 2^(k+j), at most 2^(2k−3), so its insertion lattice divides exactly.
  - Its y gradient is scaled back by 2^−j.
  - It is bit-exact on the GPU (CascadeSpike 4s).
- **Result:** with a stretch of 4 from 100 m up, the mountain folds on 0% of its columns, and its spectrum score drops
  from 0.50 to 0.17.
- **Where to use it:**
  - **Every steep landform** (mountain, cliff, valley, island, caldera) uses a stretch of 4.
  - **Feature fields** (scarp, valley axis, crater) use 8, so the feature keeps its shape at every height.
  - **Flat landforms** don't need it: their slopes never approach 1.

### Fitting

- Fit the **octaves first**, then the **remap** (`--hypso --octaves=0`), then the octaves again gently (`--gain=0.5`).
  Run together, the two fits fight.
- A **sharp feature** (cliff, trough, coast) puts energy into every octave. A whole-map fit then starves the texture:
  the cliff's lowland came out 3x too smooth, with a 3° median slope against 20°.
  - Fit the texture alone: `--texture=BIOME` uses a copy without height terms and that biome everywhere, then copies
    the multipliers back (`--copy` puts them into the other biomes, times a factor).
  - Or **pin** the octaves the feature dominates: the valley's 3.2 and 6.5 km.
- **3D slice spread:** every threshold on a field (biome edges, curve points) must come from that field's measured
  quantiles at the ground. The ground cuts the 3D noise in a slice, so a single octave reads about ±0.25, not ±1.
  `heights.json` and `fields.f32` give them.
- **Cell noise spread:** a cell field stretched in y crosses fewer cell centres, so expect fewer cones or craters than
  cells.

### Limits found, and what would close them

- **Anisotropy.** Real ranges and dunes run in one direction: anisotropy 1.4-2.4 against 1.0-1.1 generated. Linear
  dunes and parallel ranges need an anisotropic octave, stretched in x or z like y now is.
- **Drowned and glacial valleys.** Moorea's bays and the Alps' troughs are erosion, i.e. flow. The research note's
  erosion filter (Johansen 2026, point-evaluated) is the reference to port.
- **Tiny relief.** Mangrove blobs and dune ripples under a metre are invisible from more than a few metres up. They
  belong to instances (trees, roots) and to the page shader.
- **Island erosion near the shore.** Valleys deeper than the cone below them ring the island with a moat. The island
  biome fades out 300 m up, so the lower slopes are smooth.
- **Water.** The engine draws no water. Sea and lake beds are coloured as water.

## Using a landform in a world

Each table is a whole climate. A world puts several landforms in one table, each behind a **region** of macro climate
fields (temperature, humidity, wind):

```json
"regions": [{"name": "cold", "where": [{"field": "temperature", "max": 0.0, "fade": 0.05}]}],
"heights": [{"field": "valley", "height_m": 2100, "curve": [...], "region": "cold"}],
"biomes":  [{"name": "floor", "region": "cold", "where": [{"field": "valley", "min": 0.19, "fade": 0.02}], ...}]
```

- A biome's `region` puts the region's edges before its own.
- A height term's `region` multiplies the term by the region. So the trough, the cliff or the coast exists only in its
  region, and the term needs a curve (the skirts bound it).
- Keep each landform's feature fields as they are: its edges and curve points are tuned to their quantiles.
- In a region's core the landform is its standalone table bit for bit: `project/climate/region_test.json` (the valley
  behind "cold", the cliff's and the mangrove's terms behind "warm") equals `valley.json` there.
- The core starts some way inside the region's edge. The modulation reads the biome weights at the cascade's coarser
  lattices, which sit kilometres apart for a 13 km axis. Leave room for that when laying regions out.
- Across a border the two halves of a landform fade at slightly different places: the height terms read the region
  at the sample, the octave multipliers at those coarser lattices. One evaluation point for both would need the region
  carried down the cascade like a field.

What is left before the eight landforms make one world:

1. **The field budget.** The eight tables need 8 feature fields, 3 warp triples, 2 `elev` fields and the multifractal
   state, plus the climate axes. A table carries at most 24 field groups. Next step: refit the landforms onto shared
   fields: one warp triple, one `elev`, the desert's patches on a warp field. That leaves about 16 groups.
2. **The ground's stretch** is one setting for the whole table. Use 4 from 100 m for every landform, and re-score the
   flat ones.
3. **The coarse octaves.** The climate axes can't gate the ground octaves at or above their own wavelength. Every biome
   takes the same multiplier there.
4. **The world table.** Temperature and humidity at 210 + 419 km, and wind at 105 + 210 km (25 LODs). Each landform
   takes a box of that space:

   |  | humid | dry |
   |---|---|---|
   | cold | mountain | mountain |
   | cool | valley | caldera |
   | warm | island | cliff |
   | hot | mangrove | dunes (windy), desert (calm) |

   A report then gives each region's area, its neighbours, and a spawn point in its core.
5. **Cost.** Every field group is carried at every point. 16 groups cost roughly 8 times the bare cascade at near LODs.
   The fix is to skip the groups whose region is 0 over a whole chunk. That is a per-chunk reduced SDF, which needs its
   own design.
