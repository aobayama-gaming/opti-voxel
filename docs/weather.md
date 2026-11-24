# Weather

The weather is a state that changes in time over the whole planet: wind, pressure, temperature, vapour, cloud water,
rain. It is not the climate. The climate (`docs/climate.md`) is a fixed function of the position that places the biomes
and shapes the ground; the weather is simulated, forced by the climate, and read by what is drawn (clouds, leaves, later
the sea and rain). It never feeds the SDF: the terrain stays a function of the absolute position.

Status: the grid (`src/weather_grid.*`), the simulation (`src/weather_sim.*`), its worker in `World`
(`src/weather.*`) with a report (`tools/weather_report.sh`), and the clouds drawn from it (`docs/look.md` "Weather") are
in, and the mixer edits a biome's weather and plays a world's as a time-lapse. Later steps at the end. Planets only: the weather is simulated over a sphere, never over a
flat shape (decided with the user, 2026-11-24: the engine is for planets); a torus gets its grid next.

## Grid
`WeatherGrid` (`src/weather_grid.*`, engine-free, double) is the interface the simulation, the worker and the clouds
see: a planet's surface as faces of width x height cells, positions in metres from the base's centre. A base's shape is
one implementation (`WeatherGrid::for_base`): `CubeSphereGrid` for a sphere; a torus's grid (one face wrapping both
ways) is the next, nothing else changes for it.
- **What a grid gives:** its faces and their cells, the padded layout (`index`), each cell's surface point and up
  (`surface`, also at the halo's coordinates), the surface point under a position (`locate`), its halo and bilinear
  sampling, and its GLSL lookup. The simulation builds its metric (the gradients' dual vectors) from the surface
  points, so it holds no shape of its own.
- **Size:** the planet file's `cell_m`, the cells' width (0: 64 a cube face's side). On the 2 km planets 45 m gives 70
  a side (29.4k cells), the default 64 ~49 m (24.6k).
- **Layout:** each face padded with a halo of 1 cell; x fastest, then y, then the face (`index`). A field has `comps`
  floats per padded cell, interleaved. The same layout is the renderer's atlas: a 2D texture of `stride_x` texels by
  `faces * stride_y` rows, each face a block of rows.
- **The halo** (`fill_halo`, built by the base class from `surface` and `locate`): each halo cell along an edge is a
  bilinear read of the interior at its surface point (the neighbouring face's across a seam, the face's own across a
  wrap), a table of 4 sources and weights made once. A corner's halo cell, which no face may hold (three cube faces
  meet there), is linear in its own face's chart: the two edge halo cells beside it minus the interior corner cell,
  filled after the edges. A smooth field read through the halo (`sample`, bilinear between the 4 centres round a
  point) is second order everywhere, edges and corners included (unit test: worst error under 0.01 at N = 32 for a
  field of unit-scale gradients, divided by ~4 at N = 64; reading a corner from a third face's nearest cells was first
  order, 0.015 at N = 32).
- **The cube sphere:** a face is a cube axis (`MAJOR`) with two tangent axes (`U`, `V`, `U x V = MAJOR`, the table
  `CubeSphereGrid::FACE`). A direction belongs to the face of its largest |component| (x before y before z on a tie),
  at the angles `a = atan(d.U / d.MAJOR)`, `b = atan(d.V / d.MAJOR)` in [-pi/4, pi/4]; its cell coordinates are
  `x = (a / (pi/2) + 1/2) N` and `y` likewise. Equal angles keep the cells' areas within a factor ~1.4 (unit test; a
  gnomonic grid's: ~5). A position's cell is its direction's.
- **The climate under a cell:** its surface point rounded to a LOD0 voxel (`Weather::build`), sampled with the integer
  frame (`SDFCascade::climate_at`).
- **GLSL:** `glsl_lookup` gives `vec2 weather_uv(vec3 p)`: the atlas coordinates of the column under a render-space
  point, from the base's render functions (the cube's: `base_up(p)`, its face table generated from
  `CubeSphereGrid::FACE`, one source; its sizes substituted). A face's coordinates stay within its padded block, so
  bilinear filtering never mixes two faces. It is float and not bit-exact with `sample` (atan in float32, hardware
  filtering): the weather is a look, not a terrain input, so the float-mirror rule (`docs/gotchas.md`) does not apply.

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
`"weather"`: on by default over a sphere (`"on": false` turns it off), refused over a flat shape or a torus (until its
grid lands). `cell_m` (the cells' width in metres; 0 or absent: 64 cells a cube face's side; refused where the side
would leave [4, 1024]), `tick_hz` (10), `time_scale` (game seconds a real second simulates, 1), `prevailing_mps` (8),
`storm_m` (0: a quarter of the radius), `spin_up_s` (game seconds simulated before the first frame, 60). At
`time_scale` 1 and the default prevailing wind a 2 km planet's weather crosses its half circumference in ~10 minutes.

## Worker
`Weather` (`src/weather.*`, engine-free): the simulation on a worker of its own (`ov::executor().start_worker`,
"weather"), owned by `World` when the world file's weather is on. It needs no GPU: it runs headless too.
- **Start:** the worker's first step builds the forcing (each cell's `forcing_at` at its ground point, the climate read at
  the level whose cells are about the grid's, `parallel_for` at LOW priority) and runs the spin-up. On `ocean_test`: 98 ms.
- **The clock:** `World::frame` calls `update(now)`: the real time since the last frame (0.25 s at most) times `tick_hz`,
  from the first state on. The worker runs one tick (`time_scale / tick_hz` game seconds) per step while ticks are due;
  more than `MAX_BEHIND` (20) behind, the excess is dropped (the weather slows, it never runs away).
- **States:** each tick publishes `LOOK` floats per padded cell (wind, cover, type, rain, temperature, vapour); `update`
  collects the latest and keeps the previous one. The frame shows the time one tick behind the clock: `blend()` is where
  it stands between the two (the clouds' texture blend, next step).
- **Queries:** `World::weather_at(render position)` (the base's up there, sampled on the latest state), Godot's
  `VoxelTerrain.get_weather_at` (a Dictionary), the player's HUD line (wind, cover, rain); `debug_stats()["weather"]`
  (ticks, game seconds, storms, dropped and behind ticks, the tick's and the forcing's times, Omega, the substep).
- **Cost:** in Godot (its pool runs the rows), a tick at N = 64 takes 1.0 ms on average, 3.6 ms at most, on the worker.

## Report
`tools/weather_report.sh` (`WeatherReport`, CPU only, headless): the world's weather on a clock of its own, each tick
waited for, mapped at game minutes after the spin-up: `cover_<min>.png` (cover grey, rain blue) and `wind_<min>.png` (the
way the wind blows as a hue: east red, north yellow-green, west cyan, south violet; its speed as the brightness),
equirectangular with the +y pole at the top, and `report.md`. The test biomes of `ocean_test.planet.json` carry a
weather block (the ocean moist, smooth and stormy; the land drier, warmer, rougher): clouds over the ocean, sheared into
bands by the prevailing winds, one storm in ten minutes, a few percent of the planet raining.

## Mixer
The biome mixer (`debug/biome_mixer/`) edits a biome's weather and shows a world's.
- **A biome's push:** its output node carries one slider per key of the `"weather"` block (`_weather_row` in
  `nodes.gd`). The keys, ranges and defaults are the engine's (`Climate::WEATHER_KEYS`, which the parser reads too,
  through `ClimateReportNode.weather_keys`); `set_weather` writes the biome's block.
- **The time-lapse** (the toolbar's "Weather", `weather_lapse.gd`): the engine runs the mixer's world on the CPU on a
  thread (`ClimateReportNode.weather_frames`, `WeatherReport::frames`: the report's maps as images, from the world's text)
  and maps the whole planet each game minute for 20 minutes; a slider steps through them, the cover and rain or the
  wind. On `ocean_test`, about 12 s of CPU (12,000 ticks at ~1 ms). A world without weather says why.
- **Checked:** the mixer's headless check sets a weather key through the mixer and finds it in the file the engine
  accepts; unit tests run `WeatherReport::frames` from a world's text (sizes, the minute's time, no weather refused)
  and parse every key at its bounds and refuse it past them.

## Plan
Agreed with the user (2026-11-24). Stacked PRs, `feature/weather-*`:
1. **Grid** (#178): the cube sphere, its halo and sampling, the GLSL lookup.
2. **Simulation** (#179): `WeatherSim`, a biome's `"weather"` block and the planet file's.
3. **Worker** (#180): `Weather` in `World`, its queries and stats, the report.
4. **Clouds** (#181): the clouds read cover and type from the weather texture (two states blended), their noises drift with
   the wind at the camera, the leaves' sway reads it too; `CloudClimate`, a biome's `"clouds"` block, the flat 2D cloud
   layer and the clouds over a flat shape or a torus are removed (one path: the simulation's).
5. **Mixer** (#182): a biome's weather edited in the mixer, a time-lapse of the world's weather.
6. **Grid interface:** `WeatherGrid` an interface over a base, the cube sphere one implementation, positions in
   metres; the planet file's `grid` becomes `cell_m`. Then the layered simulation and the 3D clouds read from it, then
   the torus's grid.

Later: curl-noise gusts, the FFT sea reading the wind (U10, fetch, swell), rain and ground wetness, saving the weather,
the GPU if the CPU tick is too slow.
