# Weather

The weather is a state that changes in time over the whole planet: wind, pressure, temperature, vapour, cloud water,
rain. It is not the climate. The climate (`docs/climate.md`) is a fixed function of the position that places the biomes
and shapes the ground; the weather is simulated, forced by the climate, and read by what is drawn (clouds, leaves, later
the sea and rain). It never feeds the SDF: the terrain stays a function of the absolute position.

Status: the grid (`src/weather_grid.*`) and the simulation (`src/weather_sim.*`) are in. Its worker and the clouds
reading it are the next steps (the plan at the end). Planets only: the weather is simulated over a sphere, never over a
flat shape or a torus (decided with the user, 2026-11-24: the engine is for planets).

## Grid
`WeatherGrid` (`src/weather_grid.*`, engine-free, double): the planet's directions as an equiangular cube sphere, 6 faces
of N x N cells.
- **A face** is a cube axis (`MAJOR`) with two tangent axes (`U`, `V`, `U x V = MAJOR`, the table `WeatherGrid::FACE`).
  A direction belongs to the face of its largest |component| (x before y before z on a tie), at the angles
  `a = atan(d.U / d.MAJOR)`, `b = atan(d.V / d.MAJOR)` in [-pi/4, pi/4]; its cell coordinates are `x = (a / (pi/2) + 1/2) N`
  and `y` likewise. Equal angles keep the cells' areas within a factor ~1.4 (unit test; a gnomonic grid's: ~5).
- **Size:** N in [4, 1024]. On the 2 km planets N = 64 gives ~49 m cells (24.6k cells), N = 128 ~25 m (98k).
- **Layout:** each face padded with a halo of 1 cell; x fastest, then y, then the face (`index`). A field has `comps`
  floats per padded cell, interleaved. The same layout is a 3D texture (N + 2 wide and high, one layer per face).
- **The halo** (`fill_halo`): each halo cell along an edge is a bilinear read of the neighbouring face's interior at the
  halo cell's centre extended past the edge (a table of 4 sources and weights, made once). A cube corner's halo cell,
  which no face holds (three faces meet there), is linear in its own face's chart: the two edge halo cells beside it
  minus the interior corner cell, filled after the edges. A smooth field read through the halo (`sample`, bilinear
  between the 4 centres round a point) is second order everywhere, edges and corners included (unit test: worst error
  under 0.01 at N = 32 for a field of unit-scale gradients, divided by ~4 at N = 64; reading a corner from a third face's
  nearest cells was first order, 0.015 at N = 32).
- **Positions:** the grid is directions only. A cell's place is its direction at an altitude over a SPHERE
  (`cell_voxel`: the centre plus the direction times radius + altitude, rounded to a LOD0 voxel), so the climate can be
  sampled under it with the integer frame (`SDFCascade::climate_at`).
- **GLSL:** `glsl_lookup` gives `vec3 weather_uvw(vec3 dir, float n)`: the padded texel coordinates over [0, 1] and the
  face's index, its face table generated from `WeatherGrid::FACE` (one source). It is float and not bit-exact with
  `sample` (atan in float32, hardware filtering): the weather is a look, not a terrain input, so the float-mirror rule
  (`docs/gotchas.md`) does not apply. The direction a shader passes is the base's up at the point (`base_up`), never an
  absolute position.

## Simulation
`WeatherSim` (`src/weather_sim.*`, engine-free, CPU): one layer of rotating shallow water over the whole planet on the
grid, carrying vapour and cloud water. Every interior cell is written from the previous state only, so `parallel_for`
over the rows (LOW priority) gives the same bits on any thread count; the halos are refilled between passes.
- **State** (`Field`, per padded cell): the wind, a 3D vector tangent to the sphere in the planet's axes (no per-face
  frames at the seams); `ETA`, the layer's thickness anomaly, its pressure `c^2 ETA` (`WAVE_MPS` = 20 m/s);
  `TEMP`, the air's temperature anomaly; `VAPOUR` (saturation 1) and `CLOUD`, cloud water. **Diagnostics** (`Diag`):
  the cover (`CLOUD / FULL_CLOUD`), the type (stratus 0 to cumulus 1, rising with the lift and the heat) and the rain.
  The cloud cover and type are only the simulation's: biomes don't set them.
- **A substep** (`step(dt)` cuts dt into substeps of at most half a cell over the wave speed, the gravity waves' CFL):
  1. The air from upstream (semi-Lagrangian: the state sampled where the wind brings it from), the Coriolis as an exact
     rotation of the wind about the up by `-f dt` (`f = 2 Omega n_y`), the wind relaxed towards its target, the drag,
     the temperature relaxed to the biome's heat, the vapour raised towards its moisture.
  2. The pressure: the wind pushed by `-grad(c^2 ETA - BUOYANCY TEMP)` (warm air makes a low), central differences
     through each cell's dual vectors (made once from its neighbours' positions, so the cube's distortion is in them).
  3. The column: `ETA` follows the divergence; converging air rises (the lift), lowering the saturation (`1 +
     SATURATION_HEAT TEMP - SATURATION_LIFT lift`): vapour past it condenses, cloud under it evaporates, cloud past
     `RAIN_CLOUD` rains out; a little diffusion (`DIFFUSION`, a stabiliser).
- **Coriolis:** a game value from the storm radius: `Omega = c / (storm_m sqrt 2)`, so the Rossby radius at 45 degrees
  is the planet file's `storm_m` (a quarter of the radius by default). The real Omega on a 2 km planet makes no storm;
  on it the default gives a 222 s rotation of the weather (not the sun's day).
- **The prevailing wind** (`prevailing`): east (the pole is +y) times `-sin(6 |latitude|)`: trade easterlies peaking at
  15 degrees, westerlies at 45, polar easterlies at 75, calm at the equator and the poles.
- **Storms:** each substep a cell starts one with its biome's chance (`storms`: what a whole planet of that biome
  starts in an hour), drawn from a hash of the seed, the tick, the substep and the cell; a storm is a warm, moist low of
  the storm radius (`inject_storm`) which the Coriolis spins up: counterclockwise in the north, clockwise in the south.
- **Cost** (a standalone run of the core, one thread, this laptop, 2026-11-24): a 0.1 s tick at N = 64 (24.6k cells,
  one substep) takes ~4 ms; at 10 ticks a second that is 4% of one core.

## Forcing
Each cell's biome weather (`Climate::BiomeWeather`, a biome's `"weather"` block, `docs/climate.md`): the cell's biome
weights at its centre on the ground blend their biomes' values. `wind` scales the prevailing wind there, `drag` the
ground's drag (forest high, sea and ice low: wind builds over open water), `heat` the temperature the air relaxes to
(hot ground: a thermal low the air flows into), `moisture` the vapour the ground gives (a biome over 1 makes cloud by
itself; under, cloud only where air converges), `gust` the look's gusts, `storms` the storms it starts. The forcing is
sampled once when the weather starts (the climate never changes); the weather never changes the ground.

## Planet file
`"weather"`: on by default over a sphere (`"on": false` turns it off), refused over a flat shape or a torus. `grid`
(cells a face side, 64), `tick_hz` (10), `time_scale` (game seconds a real second simulates, 1), `prevailing_mps` (8),
`storm_m` (0: a quarter of the radius), `spin_up_s` (game seconds simulated before the first frame, 60). At
`time_scale` 1 and the default prevailing wind a 2 km planet's weather crosses its half circumference in ~10 minutes.

## Plan
Agreed with the user (2026-11-24). Stacked PRs, `feature/weather-*`:
1. **Grid** (#178): the cube sphere, its halo and sampling, the GLSL lookup.
2. **Simulation:** `WeatherSim`, a biome's `"weather"` block and the planet file's.
3. **Worker:** `Weather`, an `ov::WorkerLoop` on a fixed tick (~10 Hz) catching up with the clock, two snapshots handed to
   the frame thread; `World::weather_at`, stats, trace, a HUD line, and `tools/weather_report.sh` (headless maps of
   cover, wind and rain over time, the tick's cost). Parallelised (`parallel_for` over rows) only if the tick bottlenecks.
4. **Clouds:** the clouds read cover and type from the weather texture (two states blended), their noise drifts with the
   wind at the camera, the leaves' sway and the 2D cloud layer read the local wind; `CloudClimate`, a biome's
   `"clouds"` block and the clouds over a flat shape or a torus are removed (one path: the simulation's).
5. **Mixer:** a biome's weather edited in the mixer, a time-lapse weather map in World mode.

Later: curl-noise gusts, the FFT sea reading the wind (U10, fetch, swell), rain and ground wetness, saving the weather,
the GPU if the CPU tick is too slow.
