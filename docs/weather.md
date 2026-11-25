# Weather

The weather is a state that changes in time over the whole planet: wind, pressure, temperature, vapour, cloud water,
rain. It is not the climate. The climate (`docs/climate.md`) is a fixed function of the position that places the biomes
and shapes the ground; the weather is simulated, forced by the climate, and read by what is drawn (clouds, leaves, later
the sea and rain). It never feeds the SDF: the terrain stays a function of the absolute position.

Status: the grid (`src/weather_grid.*`), the simulation (the fluid `src/weather_fluid.*`, its moist physics
`src/weather_moist.*`, its forcing `src/weather_forcing.*`), its worker in `World`
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
The first simulation, `WeatherSim` (2D, then hydrostatic levels with a parcel convection, #179 and the levels PR), is
gone: at 45 m cells, as tall as they are wide, a hydrostatic model cannot hold a thermal nor the air sinking round it,
so it made one flat deck (seen in game, 2026-11-25). The weather is now the fluid below, its moist physics and its
forcing, stepped by the worker.

## Fluid
The simulation, approved with the user (2026-11-25): the non-hydrostatic moist Boussinesq equations of the cloud-resolving models
(SAM, Khairoutdinov & Randall 2003; Harris et al. 2003; Hadrich et al. 2020 "Stormscapes"), real constants, the
forcing physical and 2D (relief ignored: the SDF as the ground boundary comes later). Built in stages, each with its
tests.
- **`WeatherFluid`** (`src/weather_fluid.*`, engine-free): the dynamics only, `du/dt + u . grad u = -grad p + b up -
  f up x u`, `div u = 0`, over a `WeatherGrid`'s columns in `levels` levels of `top / levels` (shallow atmosphere).
  A padded cell's record holds per level the wind (3D, planet axes: tangent part and w) and the physics' scalars.
  A step: `advect` (semi-Lagrangian, the departure point in the face's own cells through the cell's dual vectors, no
  `locate`, bilinear in the cells and linear in the levels; `max_dt` keeps the wind within half a cell or level; the
  wind brought from upstream turned into the cell's frame, `transport`: the rotation carrying the departure's up onto
  the cell's. Without it a wind keeps its upstream direction: a 5 m/s rotation drifted 6.5 m/s off in 10 minutes at
  the faces' edges),
  the caller's physics on the state, `force` (the buoyancy along the up, the Coriolis turn about it, a sponge damping
  w in the top fifth), `project` (the pressure from the fluxes through each cell's 6 faces, ground and top rigid,
  red-black Gauss-Seidel over-relaxed 1.6, 20 sweeps warm-started; the wind at the centres loses dt times its
  central gradient: an approximate projection, collocated, no checkerboard). Rows run in `parallel_for`, the same
  bits on any thread count.
- **Tests** (`test_weather_fluid.cpp`): stratified air at rest stays exactly at rest; three projections take a
  converging updraft's divergence down 68x; a 2 K warm bubble of 150 m (the dry thermal of Robert 1993 / Bryan &
  Fritsch 2002 at the game's scale) rises 86 m in 120 s at 3.2 m/s, the thermal's `sqrt(g' R)` (3.1 m/s), every
  level's net vertical flux within 5% of its total (the air round it sinks as much: the return flow spreads wide,
  ~0.25 m/s over a ring twice its radius); the same bits run to run. A 300-sweep solve changes nothing: what
  divergence is left (~5% of `w / dz`) is the collocated projection's, not the solver's.
- **Cost** (unit test, this laptop, 2026-11-25): a dry step at `ocean_test`'s size (6 x 70^2 columns x 15 levels,
  441k cells) takes ~34 ms on the worker pool. At 5 Hz that is ~17% of the tick's time.
- **`WeatherMoist`** (`src/weather_moist.*`, engine-free): the moist physics between `advect` and `force`, per column,
  the fluid's scalars potential temperature, vapour, cloud water, rain. Kessler's warm rain ported step for step from
  the DCMIP2016 reference (`kessler.f90`, Klemp & Wilhelmson 1978): the saturation adjustment (Tetens, with its
  Clausius-Clapeyron denominator, latent heat into theta), autoconversion past 1 g/kg, accretion, rain falling at its
  terminal speed (sub-stepped under its CFL) and evaporating. The bulk surface fluxes into the lowest level (drag,
  sensible heat, evaporation times the ground's wetness, dew at full rate; `C = (0.4 / ln(z1 / z0))^2` at its centre,
  times `Params::exchange` = 15, the approved twist; at least 1 m/s of gustiness). Radiation: clear air cools 2 K/day,
  a cloud's longwave flux is DYCOMS-II RF01's (`F0 e^-Q(z,top) + F1 e^-Q(0,z)`, Stevens et al. 2005). Subsidence
  `w = -D z` on the scalars (`D` = 1.5e-5 /s: it holds a ~300 m inversion). The buoyancy `g ((theta - theta_ref) /
  theta_ref + 0.61 (q_v - q_v,ref) - q_c - q_r)` against the world's sounding (`WeatherMoist::sounding`: its
  hydrostatic Exner function and density from 1000 mb at the ground). A column's ground is 2D (`Ground`: temperature,
  wetness, roughness; relief ignored).
- **Moist tests** (`test_weather_moist.cpp`): Tetens' saturation at 0, 20, 30 C; one adjustment keeps the water and
  the liquid potential temperature; a column's rain balances its water (0.431 kg/m^2 lost, 0.431 rained); a cloud's
  top cools 4.7 K/h, clear air not at all. **Trade-wind cumulus** (BOMEX, Siebesma et al. 2003, fitted under 600 m: a
  mixed layer at 298 K and 18 g/kg, an inversion of 4 K at 300-340 m, a sea 1.5 K warmer, 0.1 K of noise): after 20
  minutes cumulus covering 2 to 5% of each level from 180 to 300 m (BOMEX's 5-10% near the base), no cloud at the
  ground nor above the inversion: broken, clear air between them. **Sea fog** (98% humidity over ground 6 K colder:
  its dew point above the ground): after 40 minutes the lowest level is cloud everywhere, 180 m clear. With the ground
  only 4 K colder no fog forms in 30 minutes: cooled and dried at the same rate the lowest level moves on the mixing
  line to the ground's saturated state, which only crosses saturation near its end; the 40 m lowest cell averages the
  surface layer where real fog starts.
- **Subgrid turbulence** (in `WeatherMoist`): Smagorinsky-Lilly in the vertical, `K = (0.18 D)^2 sqrt(max(S^2 -
  N^2 / Pr, 0))` per interface (D the cell's `(width^2 dz)^(1/3)`, Pr 1/3), implicit on theta, vapour, cloud and the
  horizontal wind: mixing where the wind shears or the air is unstable, none in the inversion. The cells resolve the
  thermals, not the surface layer: without it the lowest level took the ground's own state, a sea's saturated air, and
  fogged over a sea warmer than the air (5% of the lowest level, the cloudiest).
- **The twist** (`Params::exchange`, 15): the surface fluxes' coefficients only. Sped up as well, radiation and
  subsidence (one time compression of the slow processes, tried) outran the boundary layer's turbulence, which runs at
  its own speed: the humid layer cooled to fog, the fog case's top overturned it into a deck at 180 m.

## Forcing
`WeatherForcing` (`src/weather_forcing.*`, engine-free): what drives the air from outside it.
- **The ground** per column, 2D (relief ignored): the blend of its biomes' `"weather"` (`Climate::BiomeWeather`,
  `docs/climate.md`; their weights at the column's ground point, `Weather::forcing_at`, sampled once at start):
  `ground_k` the ground's or the sea's mean temperature, `diurnal_k` its day (`ground_k + diurnal_k (max(up . sun, 0)
  - 1 / pi)`: the day's mean is `ground_k`; the sun is the game's, `World::set_sun`), `wetness` (1 a sea, 0 a desert),
  `z0_m` its roughness, `storms`. The weather never changes the ground.
- **The large scale:** the pressure gradient holding the prevailing wind in geostrophic balance, `f up x u_g` on every
  level (`u_g`: east times `-sin(6 |latitude|)`, trades at 15 degrees, westerlies at 45). `f = 2 Omega up_y`, `Omega`
  the weather's day (`day_s` 300 s: the twist, storms spin at the game's scale). The jets are unstable: they roll up
  into mid-latitude eddies, which the cube's 4-fold symmetry seeds at wavenumber 4 (without the prevailing wind there
  are none: the grid makes no wind by itself).
- **Storm seeds:** each step a column starts one with the chance its `storms` gives (a planet of storms 1 starts one an
  hour; a hash of the seed, the step and the column): +2 K and +2 g/kg in a bubble of `storm_m` over the lower half of
  the levels. What it grows into is the physics'.
- **The sounding** (the air's start and the buoyancy's reference): a trade-wind boundary layer, BOMEX's shape scaled to
  `inversion_m`: a mixed layer 1.5 K under the grounds' mean up to 0.53 of the inversion, its vapour saturating a
  ground parcel exactly there (the cloud base at the mixed layer's top, as in BOMEX; at 88% humidity the base sat 80 m
  over the thermals' reach and nothing formed in 20 minutes), a cloud layer (+4 K/km, drying), a 4 K inversion over
  40 m, dry stable air above. 0.1 K of noise on the two lowest levels.
- **Tests** (`test_weather_forcing.cpp`): the sounding saturates at the base and not under it; the ground's noon, night
  and day's mean; the trades from the east, the westerlies from the west; storms at their rate, the same from the same
  seed. **A trade-wind sea** (all of it at once on 6 x 32^2 columns: a sea 1.5 K over its air, the rotation, the
  prevailing wind), sampled each minute from 15 to 40: cumulus with their base at 140-180 m, nothing at the ground nor
  above 340 m, 0.8% of the columns cloudy on average and 5.6% at most. BOMEX's ~10%: the twisted rotation (f up to
  0.04 /s, a thermal's Rossby number ~0.5) stiffens the thermals (a 50-minute day gave 5.6% on average).

## Planet file
`"weather"`: on by default over a sphere (`"on": false` turns it off), refused over a flat shape or a torus (until its
grid lands). `cell_m` (the cells' width in metres; 0 or absent: 64 cells a cube face's side; refused where the side
would leave [4, 1024]), `top_m` (the levels' top over the ground, 600; the levels about `cell_m` apart, at least 2),
`tick_hz` (5), `time_scale` (game seconds a real second simulates, 1), `prevailing_mps` (5), `storm_m` (0: 1/8 of the
radius), `spin_up_s` (game seconds simulated before the first frame, 600), `day_s` (the weather's day, 300),
`inversion_m` (the start sounding's inversion, 300, under `top_m`), `exchange` (the surface fluxes' twist, 15).

## Worker
`Weather` (`src/weather.*`, engine-free): the fluid, its moist physics and its forcing on a worker of its own (`ov::executor().start_worker`,
"weather"), owned by `World` when the world file's weather is on. It needs no GPU: it runs headless too.
- **Start:** the worker's first step builds the forcing (each cell's `forcing_at` at its ground point, the climate read at
  the level whose cells are about the grid's, `parallel_for` at LOW priority), the sounding and the start, and runs the
  spin-up.
- **The clock:** `World::frame` calls `update(now)`: the real time since the last frame (0.25 s at most) times `tick_hz`,
  from the first state on. The worker runs one tick (`time_scale / tick_hz` game seconds, in steps of at most the
fluid's `max_dt` and 5 s: advect, forcing, moist physics, force, project) per step while ticks are due;
  more than `MAX_BEHIND` (20) behind, the excess is dropped (the weather slows, it never runs away).
- **States:** each tick publishes `LOOK` floats per padded cell (the lowest level's wind, the cover: the column's
  largest cloud, the type: its cloudy depth over a third of the levels, the rain in mm/h, the lowest level's
  temperature in C and vapour in g/kg, the levels' mean wind), then each level's cloud water over `FULL_CLOUD_QC`
  (0.3 g/kg: the clouds' density 1); `update`
  collects the latest and keeps the previous one. The frame shows the time one tick behind the clock: `blend()` is where
  it stands between the two (the clouds' texture blend, next step).
- **Queries:** `World::weather_at(render position)` (the base's up there, sampled on the latest state), Godot's
  `VoxelTerrain.get_weather_at` (a Dictionary), the player's HUD line (wind, cover, rain); `debug_stats()["weather"]`
  (ticks, game seconds, storms, dropped and behind ticks, the tick's and the forcing's times, Omega, the substep, the
  forcing's summary per hemisphere).
- **Cost** (`tools/weather_report.sh`, `ocean_test`: 6 x 70^2 columns of 13 levels, this laptop, 2026-11-25): a 0.2 s
  tick takes 53 ms on average, 112 ms at most, on the worker (5 Hz: ~26% of the worker's time); the forcing and 1500 s
  of spin-up 42 s at load (TODO: a warm start or a saved state; the spin-up waits for the boundary layer's first
  adjustment, ~20 game minutes).

## Report
`tools/weather_report.sh` (`WeatherReport`, CPU only, headless): the world's weather on a clock of its own, each tick
waited for, mapped at game minutes after the spin-up: `cover_<min>.png` (cover grey, rain blue) and `wind_<min>.png` (the
way the wind blows as a hue: east red, north yellow-green, west cyan, south violet; its speed as the brightness),
equirectangular with the +y pole at the top, `section_<min>.png` (a cross-section along the equator: longitude across,
the levels up from the ground, the cloud water white at full cover) and `report.md` (per map: storms, mean cover, top
wind, raining share (rain over 0.1 mm/h), each level's cloudy share of the planet, and the forcing's summary). The test biomes of `ocean_test.planet.json` carry a
weather block (the ocean a sea at 300.5 K, smooth and stormy; the land at 299 K with a 3 K day, 30% wet, z0 0.1 m;
with an 8 K day the fixed sun's night side fogged whole, its land 1 K under the trade air's dew point). On it
(2026-11-25): no fog, a sparse cumulus field at 180 m along the eddies' edges, 4% of that level, mean cover 0.05.

## Mixer
The biome mixer (`debug/biome_mixer/`) edits a biome's weather and shows a world's.
- **A biome's ground:** its output node carries one slider per key of the `"weather"` block (`_weather_row` in
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
   metres; the planet file's `grid` becomes `cell_m`.
7. **Levels:** the simulation in levels up to `top_m` (hydrostatic pressure, vertical wind, parcel convection under an
   inversion, latent heat, drying aloft, saturation from the actual temperature), the report's cross-section.
8. **3D clouds:** the cloud layer from the ground to `top_m`, each level's cloud a slice of the texture (`docs/look.md`).
9. **Fluid** (the rebuild, "Fluid" above): the 3D dynamics and its tests.
10. **Moist physics:** `WeatherMoist` and its tests (cumulus, fog).
11. **Forcing and wiring:** `WeatherForcing`, the biome block's physical keys, `Weather` on the fluid at 5 Hz,
    `WeatherSim` deleted; the advection's parallel transport, the subgrid turbulence. Then the typhoon test on an
    f-plane, the clouds' blend carried by the wind and their cost, then the torus's grid.

Later: curl-noise gusts, the FFT sea reading the wind (U10, fetch, swell), rain and ground wetness, saving the weather,
the GPU if the CPU tick is too slow.
