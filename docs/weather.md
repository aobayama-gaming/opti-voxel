# Weather

The weather is a state that changes in time over the whole planet: wind, pressure, temperature, vapour, cloud water,
rain. It is not the climate. The climate (`docs/climate.md`) is a fixed function of the position that places the biomes
and shapes the ground; the weather is simulated, forced by the climate, and read by what is drawn (clouds, leaves, later
the sea and rain). It never feeds the SDF: the terrain stays a function of the absolute position.

Status: the grid is in (`src/weather_grid.*`). The simulation, its worker and the clouds reading it are the next steps
(the plan at the end).

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

## Plan
Agreed with the user (2026-11-24). Stacked PRs, `feature/weather-*`:
1. **Grid** (this): the cube sphere, its halo and sampling, the GLSL lookup.
2. **Simulation** (`weather_sim`): one-layer rotating shallow water on the grid. Wind as a 3D tangent vector, a thickness
   anomaly as the pressure, temperature, vapour, cloud water, rain. Semi-Lagrangian advection; forward-backward gravity
   waves substepped by CFL; Coriolis `2 Omega n_y` with a game `Omega` (on a 2 km planet the real one makes no storm);
   fixed dt, every cell written only by itself, so any thread count gives the same bits. Forced by the climate once at
   start: each biome's `"weather"` block (`wind`, `drag`, `heat`, `moisture`, `gust`, `storms`) blended by its weight like
   `CloudClimate::of_weights`, and a prevailing wind by latitude (`BaseSurface::pole_form`). The cloud cover is calibrated
   so still air over a biome gives its authored `clouds.cover`. Storms: warm-core lows seeded from `hash(seed, tick,
   cell)` at a biome's `storms` rate. A planet file's `"weather"` block: the grid, the tick, the time scale, the day.
3. **Worker:** `Weather`, an `ov::WorkerLoop` on a fixed tick (~10 Hz) catching up with the clock, two snapshots handed to
   the frame thread; `World::weather_at`, stats, trace, a HUD line, and `tools/weather_report.sh` (headless maps of
   cover, wind and rain over time, the tick's cost). Parallelised (`parallel_for` over rows) only if the tick bottlenecks.
4. **Clouds:** the clouds read cover and type from the weather texture (two states blended), their noise drifts with the
   wind at the camera, the leaves' sway and the 2D cloud layer read the local wind; `CloudClimate::Map` and its stream
   are removed (one path).
5. **Mixer:** a biome's weather edited in the mixer, a time-lapse weather map in World mode.

Later: curl-noise gusts, the FFT sea reading the wind (U10, fetch, swell), rain and ground wetness, saving the weather,
the GPU if the CPU tick is too slow.
