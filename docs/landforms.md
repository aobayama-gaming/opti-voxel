# Landforms from real heightmaps

Eight landform tables in `adapter/godot/project/climate/landforms/` were each matched against a real heightmap. They are built from
the cascade's own tools: Perlin octaves, shaped bands, multifractal weights, field warps, height terms, cell noise,
terraces and stretched octaves. This page explains:

- how the matching works;
- what each landform takes, with the real and generated statistics side by side;
- which tool fits which feature: what worked and what didn't.

The tools write their heightmaps and comparison sheets under `bench/results/` (gitignored): `fetch_dem.py` to
`bench/results/landforms/dem/` unless `--out=DIR`, `match.py` to `bench/results/<site>-match/` unless `--out=DIR`.

## The loop

1. **Real data.** `uv run tools/landforms/fetch_dem.py [site ...]` downloads Copernicus GLO-30 DSM tiles. The source is
   the public AWS bucket `copernicus-dem-30m` and needs no account. The script crops each site of
   `tools/landforms/sites.json` to a square of local metres, resampled bilinearly. The spacing is a power of two of
   5 cm voxels, so the engine can sample the same grid: 25.6 m or 51.2 m, 512 x 512.
2. **Generated data.** `tools/climate_report.sh --lattice=M` samples a climate table on lattice Λ_M, each column at
   its ground surface, using the CPU cascade at a few µs per point.
   - It writes `heights.f32`: metres above the base (the flat world's datum).
   - It also writes `fields.f32` (the field values), `dhdy.f32` (the ground's vertical slope) and `picks.u8`.
3. **Comparison.** `uv run tools/landforms/compare.py --real=SITE --gen=RUN` (statistics in `terrain_stats.py`)
   prints the table below and draws a sheet with:
   - hillshades on the same colour scale, and oblique views;
   - a cross-section;
   - the octave spectrum, slope histogram, hypsometric curve and curvature histogram.
4. **Fitting.** `uv run tools/landforms/match.py SITE TABLE [--fit=N] [--hypso] [--texture=BIOME] [--center=U,V]`
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
| fold share (generated only) | dh/dy > 1 of the 3D ground: the surface folds into overhangs (dh/dy < −1 only steepens) | `ground_stretch` |

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
| mountain | Mont Blanc massif (51.2 m) | 0.06 | 2.7 | 0.05 | 0.13 | 1.26 / 13.4 |
| flat desert | Selima Sand Sheet (25.6 m) | 0.23 | 0.02 | 0.03 | 0.10 | 5.48 / 17.0 |
| dunes | Namib Sand Sea, star dunes (25.6 m) | 0.10 | 1.8 | 0.01 | 0.53 | 1.08 / 7.8 |
| mangrove | Sundarbans (25.6 m) | 0.46 | 0.47 | 0.41 | 0.25 | 4.26 / 16.6 |
| island | Moorea (51.2 m), `--center=36000,34100` | 0.24 | 1.8 | 0.21 | 0.32 | 0.51 / 13.4 |
| cliff | Drakensberg Amphitheatre (25.6 m) | 1.15 | 2.6 | 0.09 | 0.04 | 0.69 / 3.6 |
| valley | Yosemite Valley (25.6 m) | 0.35 | 2.0 | 0.08 | 0.24 | 1.45 / 4.6 |
| caldera | Crater Lake (25.6 m), `--center=-11059,-22323` | 0.66 | 3.4 | 0.15 | 0.36 | 0.83 / 8.3 |

Scored with `match.py SITE TABLE --fit=0`, centred on the slice's (0, 0) unless the table names a `--center`: a cell
noise feature (the island's cone, the crater) sits wherever its cell put it, so those two are scored on it. The tables
were last changed for the world (below): shared warp and `elev` fields, one stretch, the continental octaves off, then
mountain's and dunes' remaps and octaves refit.

**Where the bare ground scores better, the statistics can't see the feature.**
- **Cliff:** the bare ground's spectrum score (0.69) beats the cliff table's (1.15). The Drakensberg's dissected lowland
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
  them into terraces and bumps. Write curves on evenly spaced points; `--hypso` resamples its rank mapping on an
  even grid. It also centres the curve on 0: the real site's mean altitude is a constant lift that no statistic
  sees, and carried by the curve it put Mont Blanc's remap over the ground's 3.3 km cap (refused).

### The ground is 3D: stretch it for steep landforms

The ground is 3D noise, so dh/dy is as large as dh/dx. Moving up a column, the surface meets other noise: the 3D
slice puts amplitude into the fine octaves that a heightmap doesn't have, and where dh/dy passes 1 the column folds
(the distance falls as 1 − dh/dy along the up; a negative dh/dy only makes it steeper).

- **Before stretching** (mountain.json without its `ground_stretch`): 8.7 m at the 100 m octave against 2.6 m real
  (13.4 m against 7.5 at 200 m), 0.4% of slopes over 75°, spectrum score 0.40. Folds are rare: 0.05% of the columns.
  - An earlier count gave 6% folded: it counted |dh/dy| > 1, and the columns with dh/dy < −1 don't fold.
- **The fix, `ground_stretch` / `"stretch_y"`.** The noise is read at y / 2^j.
  - Octave k's wavelength along the base's up (y on the flat world) becomes 2^(k+j), at most 2^(2k−3), so its insertion lattice divides exactly.
  - Its y gradient is scaled back by 2^−j.
  - It is bit-exact on the GPU (CascadeSpike 4s).
- **Result:** with a stretch of 4 from 100 m up (the table fitted with it), 2.3 m at 100 m, no slope over 75°, no
  fold, spectrum score 0.13.
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
  dunes and parallel ranges need an anisotropic octave, stretched along a tangent direction like the up one now is.
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
  region. The term needs a curve: the skirts bound its step with the region's.
- Keep each landform's feature fields as they are: its edges and curve points are tuned to their quantiles.
- In a region's core the landform is its standalone table, bit for bit. `adapter/godot/project/climate/region_test.json` (the valley
  behind "cold", the cliff's and the mangrove's terms behind "warm") equals `valley.json` there: 4,852 points at LODs
  0, 3 and 7, every SDF value and pick the same (`test_climate_regions.cpp`).
- The core (`Climate::core_regions`) is exact: the region is exactly 1 at every point whose climate the sample reads.
  Each octave above the sample's reads its own lattice's, the sample's octave and terms the lattice above (`docs/climate.md`),
  so those are the point's level boxes from the one above the sample up to the highest octave the biomes treat
  differently. They sit kilometres apart for a 13 km axis, so the core starts some way inside the region's edge. Leave
  room for that when laying regions out.
- Across a border the two halves of a landform fade at slightly different places: the height terms read the region
  derived on the level above the sample, the octave multipliers at their own coarser lattices.
- A table carries at most 24 field groups (`Cascade::MAX_FIELD_GROUPS`). Only the CPU's slot arrays are sized by it; the
  GPU code is generated per table.
- Every field group is carried down to the level above the samples, so a world pays for all its landforms on those
  levels. Skipping the groups whose region is 0 over a whole chunk is a per-chunk reduced SDF, with its own design.

### The landform world

`adapter/godot/project/climate/world.json` holds the eight landforms, at 25 LODs and 5 cm. It is generated:
`uv run tools/landforms/world.py` composes it from the landform tables, which stay the one source (refit a table, run it
again).

- **Regions** are boxes of three macro axes: temperature and humidity (419 + 210 km octaves) and wind (210 + 105 km).
  The boundaries are the axes' quartiles and medians over the ground (`world.py --axes=RUN` on a report 3,355 km
  across), so each landform gets its planned share:

  |  | humid | dry |
  |---|---|---|
  | cold | mountain | mountain |
  | cool | valley | caldera |
  | warm | island | cliff |
  | hot | mangrove | dunes (windy), desert (calm) |

- Two neighbours share a boundary: one fades out where the other fades in, over 0.04 of the axis. Mountain's biome is the
  default (no edges): it takes what the priority layering leaves in the transitions.
- **Shared fields.** A field defined the same way in several tables is one field of the world; one defined differently
  is renamed `<landform>_<name>`. To fit the budget the tables share one warp triple (caldera's seeds) and one `elev`
  (mountain's scale, dunes' remap curve rescaled): 16 fields, 17 groups with the multifractal state, 15 biomes.
- **One stretch:** 4 from 100 m, on every table (desert, dunes and mangrove had none: dunes' octaves were refit under it).
- **The continental octaves are off.** A ground octave reads the climate of the octaves coarser than itself, so octaves
  at or above the axes' finest wavelength (104,857.6 m) are inserted before the axes exist: no region can gate them.
  Every landform turns them off (multiplier 0; they were at 0-2% already), and `world.py` refuses a table that doesn't.
  `elev` reads the ground's own height, so mountain's and dunes' remaps were refit after that.
- **The cores are the tables.** In a region's core (`Climate::core_regions`), every SDF value and pick of the world
  equals the landform's own table: `test_climate_regions.cpp`, 92 to 2,476 points per landform over 800 km at LODs 0
  and 6. Closer to a border, the climate the octaves read from the coarser lattices falls in the fade.
- **Visiting.** `climate_report.sh --climate=res://climate/world.json --lods=25 --lattice=15 --size=512` lists each
  region's core share and its core point nearest the centre, with the player's `--spawn`.

## On a planet base

The ground reads its altitude and up from the base surface (`src/base_surface.hpp`), so the tables carry over to a
sphere unchanged:

- **Reads the base:** the ground's distance (`d_base`, `n`), the stretched octaves (squeezed along `n`, not y), the
  climate report (a tangent-plane grid traced down `n`) and the player's `--spawn` (tangent coordinates).
- **Still assumes y:** `SDFTable::ground_y` (column traces for the instance scenes' centres), the forest's and the rock
  scatter's xz zones, and the biome features' xz cells (lakes, islands). They move to the base with the planet PR.
