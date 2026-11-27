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
  f up x u`, `div u = 0`, over a `WeatherGrid`'s columns in `levels` levels (`WeatherLevels`, below; shallow
  atmosphere).
  A padded cell's record holds per level the wind (3D, planet axes: tangent part and w) and the physics' scalars.
  A step: `advect` (finite volume: every field, the wind's 3 components included, changes only by its fluxes through
  its volume's 6 faces, "Turbulence" below; `max_dt` keeps the sum of a volume's outflow Courant numbers within 0.5),
  the caller's physics on the state, `force` (the buoyancy along the up, the Coriolis turn about it, a sponge damping
  w in the top fifth), `project` (the pressure from the fluxes through each cell's 6 faces, ground and top rigid,
  red-black Gauss-Seidel over-relaxed 1.6, 20 sweeps warm-started). The vertical wind lives on the levels' interfaces
  (staggered, SAM's C grid in the vertical: `vertical_w`; a centre's is the mean of its two): the buoyancy and the
  sponge act there, the divergence's top and bottom fluxes are its and it loses the difference of the two pressures
  round it, so the vertical projection is exact; the tangent wind at the centres loses the central gradient (approximate
  sideways). Collocated in the vertical before (2026-11-25), a w alternating level by level was invisible to the
  projection's face means and held 70-90% of w's rms on `ocean_test` (the debugger's checkerboard line: its (-1)^k
  part 0.18-0.21 m/s of 0.23-0.27): it layered the vapour and the cloud by level. Staggered: 0.019 m/s, under the
  noise level (0.045). Rows run in `parallel_for`, the same
  bits on any thread count.
- **Tests** (`test_weather_fluid.cpp`): stratified air at rest stays exactly at rest; three projections take a
  converging updraft's divergence down 68x; a 2 K warm bubble of 150 m (the dry thermal of Robert 1993 / Bryan &
  Fritsch 2002 at the game's scale) rises 86 m in 120 s at 3.2 m/s, the thermal's `sqrt(g' R)` (3.1 m/s), every
  level's net vertical flux within 5% of its total (the air round it sinks as much: the return flow spreads wide,
  ~0.25 m/s over a ring twice its radius); the same bits run to run.
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
  mixed layer at 298 K and 18 g/kg, an inversion of 4 K at 300-340 m, a sea 1.5 K warmer, 0.1 K of noise): averaged each minute from 15 to 25 minutes
  (one snapshot is one cloud's life), cumulus covering 0.5 to 1.6% of each level from 180 to 300 m (BOMEX's 5-10% near
  the base: the cells resolve few of them, the shallow convection makes the rest on the planet), no cloud at the
  ground nor above the inversion at any sample: broken, clear air between them. **Sea fog** (98% humidity over ground 6 K colder:
  its dew point above the ground): after 40 minutes the lowest level is cloud everywhere, 180 m clear. With the ground
  only 4 K colder no fog forms in 30 minutes: cooled and dried at the same rate the lowest level moves on the mixing
  line to the ground's saturated state, which only crosses saturation near its end; the 40 m lowest cell averages the
  surface layer where real fog starts.
- **Subgrid turbulence:** none of its own; the advection's limiter is the subgrid model ("Turbulence" below). The
  vertical Smagorinsky it had (2026-11-25) changed the convective boundary layer's w by under 1%: the then
  semi-Lagrangian interpolation's smoothing was doing the subgrid work.
- **The twist** (`Params::exchange`, 15): the surface fluxes' coefficients only. Sped up as well, radiation and
  subsidence (one time compression of the slow processes, tried) outran the boundary layer's turbulence, which runs at
  its own speed: the humid layer cooled to fog, the fog case's top overturned it into a deck at 180 m.

## Levels
`WeatherLevels` (`weather_fluid.hpp`): the one definition of the levels' heights, which the fluid, the physics, the
debugger and the cloud shader all ask. `count` levels from the ground to `top_m`, each `stretch` times as thick as
the one under it (the planet file's `stretch`, 1 by default: all alike): fine near the ground, where the thermals and
the cloud base need them, coarse aloft, where a deep atmosphere is smooth (every cloud-resolving model stretches its
levels: SAM, WRF). Geometric, so interface `m` is at `top (r^m - 1) / (r^n - 1)` and a height's level has a closed
form, `log(1 + z (r^n - 1) / top) / log r`: its GLSL twin (`glsl_fraction`) is the cloud shader's texture coordinate,
one constant, no table. `for_top(top, cell, stretch)` picks the count whose lowest level is about the cell's width.
Every volume has its own bottom, centre and top (the levels', and the interfaces' for the vertical wind: a level's
centre to the next's): the flux advection reconstructs a face's value at the face's height, the side faces' areas are
the level's thickness, the interfaces' side speed is the two levels' weighted by their halves, the projection's
vertical coefficient is the centres' distance, an interface's w and buoyancy are linear between the centres round it,
the surface layer is the lowest level's thickness, the longwave and the convection's fluxes divide by each level's.
Kessler's rain already fell between the levels' own heights. Checked (`test_weather_fluid.cpp`): the heights add up
to the top, each `stretch` times the one under it, the index inverts them; a blob stirred through levels stretched by
1.25 keeps its total to 1e-5 and its bounds.

## Turbulence
An implicit large-eddy simulation: the advection is the one subgrid model, for the wind and every scalar, sideways and
up alike (Margolin, Smolarkiewicz & Sorbjan 1999 ran the convective boundary layer this way, with MPDATA).
- **One flux-form scheme** (`WeatherFluid::advect`, 2026-11-25) for every field: the wind's 3 components (planet
  axes), the scalars, the interfaces' w (their volumes span a level's centre to the next's). A field changes only by
  its fluxes through its volume's 6 faces; a face's value is the upwind volume's, linear along its gradient, the
  gradient limited so no face value leaves the range of the volume and its neighbours (Barth & Jespersen 1989: no new
  extreme, water never negative); two stages (Heun, SSP-RK2). Each face is computed once, by its lower-index cell,
  from the true cell across it (across a seam the neighbour face's, not the halo's interpolation), so what one cell
  loses the other gains to the bit. The pressure is solved over the same faces and corrects the same face speeds (a
  MAC projection: Bell, Colella & Glaz 1989), so the fluxes are divergence-free to the solver's residual.
- **The faces' geometry** is the true cells': a side's normal perpendicular to its edge in the tangent plane, its
  distance the true neighbour's centre's offset along it, and the centre's pressure gradient the least-squares fit
  over the true neighbours' offsets (exact for a linear pressure), the same differences the faces lose. Measured
  (2026-11-26): the game's weather (the `fix` world) blew up from a cube corner, the strongest wind 17 -> 32 -> 74 m/s
  over game minutes 3-5, then 1.4 g/kg condensed everywhere and warmed the planet 3.5 K. The faces' geometry had been
  the halo's points' (the face's own grid past its edge, tens of metres off the cell across the seam near a corner)
  and the centres lost a gradient read from the halo; with rotation (a 5-minute day) the seams' error became wind the
  projection can't take out. A random wind over stable air with nothing forcing it (`test_weather_fluid.cpp`): the
  mean energy x95 between minutes 3 and 20 before, now it decays (6.5 -> 0.024 m^2/s^2 in 16 minutes), a corner
  regrows it x2.9 once, then it holds over 40 minutes. The `fix` world over 20 game minutes: the strongest wind
  10-14 m/s inside the faces, the mean 3.85 m/s, the cover 0.21-0.26; the air warms ~0.2 K and dries ~0.2 g/kg an hour.
  Open: the corner's regrowth (the two-point pressure flux assumes the centres' line perpendicular to their edge, far
  from it at a cube corner).
- **Why not MacCormack** (Selle et al. 2008; the scheme before, same day): a semi-Lagrangian step corrected by its
  traced-back error. It passed the dry benchmark (0.65 w*) but isn't conservative: on BOMEX it created +0.25 g/kg of
  water every 5 minutes at the clouds' sharp edges, rained it and evaporated it into a 5 K/h cooling, updrafts at
  16-31 m/s. The flux form, same case and hour: water by advection within 0.007 g/kg per 5 minutes (the budget weighs
  by density, the fluxes by volume), the surface theta steady, w at most 5-8 m/s. A test keeps a blob's total within
  1e-5 over 100 steps round the sphere and its range within its start's.
- **The benchmark** (`test_weather_cbl.cpp`): the dry convective boundary layer (Deardorff 1974, Schmidt & Schumann
  1989, Moeng & Sullivan 1994): a 400 m mixed layer under a 4 K inversion heated at 0.06 K m/s, ~49 m cells and 50 m
  levels (the game's), no rotation, no moisture, a 0.1 K random start; the profiles over minutes 20-30. The large-eddy
  simulations' w rms peaks at ~0.63 w* near 0.35 h, the heat flux falls linearly to ~-0.2 Q at h. Measured
  (2026-11-25):

  | advection | closure | peak w rms / w* (at z/h) | w rms at 0.81 h | entrainment flux / Q |
  |---|---|---|---|---|
  | semi-Lagrangian | none | 0.48 (0.31) | 0.20 | -0.08 |
  | semi-Lagrangian | vertical Smagorinsky | 0.47 (0.31) | 0.20 | -0.08 |
  | semi-Lagrangian | 3D Smagorinsky-Lilly | 0.41 (0.31) | 0.06 | -0.07 |
  | MacCormack | none | 0.65 (0.31) | 0.37 | -0.14 |
  | MacCormack | 3D Smagorinsky-Lilly | 0.50 (0.31) | 0.12 | -0.11 |
  | **flux form (limited linear, SSP-RK2)** | **none** | **0.625 (0.44)** | **0.34** | **-0.26** |

  With ~8 cells a boundary layer an explicit closure on top of the scheme's own dissipation over-damps (counted
  twice). The test keeps the profile in the references' range.
- **Removed** (2026-11-25): the vorticity confinement (Fedkiw, Stam & Jensen 2001) and the random kicks (a stochastic
  backscatter, Shutts 2005) propped up the plain semi-Lagrangian step's damped convection; now the
  thermals come from the buoyancy alone (the benchmark's start is a 0.1 K noise once, as large-eddy simulations seed
  theirs), and the confinement on top of it grew a vortex past its start. The planet file's `confinement` and
  `kick_mps` went with them.

## Partial cloud
The air inside a 45 m cell is not uniform: a cell a little under saturation holds some cloud where it is moister.
`weather_cloud_fraction.hpp` and `WeatherMoist::condense` (approved 2026-11-25):
- **The scheme:** Sommeria & Deardorff (1977) / Mellor (1977) with Cuijpers & Bechtold's (1995) fits to large-eddy
  simulations, as Bechtold & Siebesma (1998, JAS 55, appendices A-B) give them. `s = a (q_w - q_sl)` the mean
  saturation excess at the liquid water temperature (`a = 1 / (1 + L/c_p dq_s/dT)`, Kessler's own denominator),
  `Q1 = s / sigma_s`; the cloudy share `N = max(0, min(1, 0.5 + 0.36 atan(1.55 Q1)))`, the mean cloud water
  `q_l / sigma_s = exp(1.2 Q1 - 1)` under saturation. Over it their quadratic grows as `Q1^2` and never meets the
  uniform cell's `q_l = s`: `Q1 + exp(-1 - 1.2 Q1)` instead (continuous at 0, within 1% of theirs over [1, 2]). The
  fit's tail is the skewed one trade cumulus need (Bechtold et al. 1995: a Gaussian suits stratocumulus only).
- **In the physics:** Kessler's saturation adjustment goes to the partial cloud's mean (latent heat, buoyancy, the
  cloud's longwave and the ground's shade all from it); autoconversion from the cloudy part's water `q_c / N`. No
  spread: Kessler's all-or-nothing, unchanged.
- **The spread** `sigma_s` per cell: s's variance budget in balance (Stull 1988: production `-2 w's' ds/dz`,
  dissipation `sigma^2 / tau`): `sigma^2 = 2 tau |w's' ds/dz|`, the flux the shallow convection's updraft `M (s_u - s)`
  (the subgrid turbulence's `-K ds/dz` went with the closure: no updraft, no spread) (convection sets the humidity's variance, as Lenderink & Siebesma 2000
  couple a mass flux to a statistical cloud scheme); `tau = z_i / w*`, the boundary layer's turnover (z_i the updraft's
  top, w* the surface buoyancy flux's; at least u* and 0.1 m/s). The updraft's own cloud is no longer added on top:
  the partial cloud is the cloud.
- **The render** (`docs/look.md` "Density"): the simulation's cloud water as it is (each level's mean, published per
  cell, trilinear between cells and levels, the two states blended), times `0.5 + u`: a texture of mean 1, so where the
  cloud is and how much is the simulation's alone. `u` is `WeatherCloud::uniform_noise`: a tileable gradient fbm (128^3,
  4 lattice cells and 3 octaves, a period of 4 weather cells: features of a cell, a half and a quarter), its ranks
  turned into values, so it is exactly uniform and as smooth as the noise. Until 2026-11-26 the noise was thresholded
  against each cell's Q1 (`max(Q1 + spread(u), 0)`, a cell without a spread drawn with 0.2 g/kg): the noise then
  decided where the cloud was, and the clouds looked fluffy and flickered; with features of half a cell to an eighth,
  like cotton. The physics never reads the noise (`test_weather_cloud_fraction.cpp`: its histogram, its wrap).
- **Measured** before the closure and the confinement went (trade-wind sea, 6 x 32^2, minutes 15-30): cover 49%
  with no confinement, 17% at 0.02 (all-or-nothing: 86%, 1.8%); sigma_s 0.1-0.45 g/kg, largest in the cloud layer.
  To measure again on the benchmark below.

## Convection
`WeatherConvection` (`src/weather_convection.*`, engine-free): the shallow convection the cells cannot resolve (3-4
levels in the mixed layer, a thermal ~2 cells wide: without it the ground's heat only diffused, and a warm afternoon
made no cumulus). The mass-flux half of EDMF (Siebesma, Soares & Teixeira 2007, JAS 64, equation numbers theirs), its
eddy diffusivity the resolved flow's (no closure of its own since 2026-11-25); one updraft per column, each step before the moist physics
(whose saturation adjustment condenses what it brings).
- **Start** (17): the lowest level's mean plus `ALPHA w'phi'_s / sigma_w(z_1)` (ALPHA 1) of the ground's last fluxes
  (`WeatherMoist::surface`), `w_u = sigma_w / sqrt(0.15)` (section 3's `mu`); only while the ground heats the air (a
  buoyancy flux over 0: by day over land, day and night over a sea warmer than its air).
- **Rise:** `d phi_u / dz = -eps (phi_u - phi)` (10), `eps = 0.4 (1 / z + 1 / (z* - z))` (16, both distances at least
  a level, `z*` the column's last top), `1/2 (1 - 2 mu) dw_u^2 / dz = -0.5 eps w_u^2 + B` (15), 8 steps a level; it
  carries `theta_l` and `q_t`, saturation-adjusted on the way (the cloudy extension, Soares et al. 2004): its base where
  it saturates, its top `z*` where `w_u` reaches 0.
- **Transport:** `M = 0.3 sigma_w` (21), `sigma_w` Holtslag and Moeng's (7) of `w*` over `z*` and `u*`; the column's
  `theta_l` and `q_t` change by `-d/dz (M (phi_u - phi))` (22; explicit, `M dt / dz` under 0.1; the column's totals
  kept).
- **Its cloud:** the updraft's water over its area `M / w_u` (0.3 at most) per level, the columns' convective cloud: the
  published cloud and the ground's shade add it to the grid's (a cumulus narrower than a cell). It does not rain.
- **Tests** (`test_weather_convection.cpp`): the plume's saturation adjustment keeps `theta_l` and `q_t`; `sigma_w`
  peaks at 0.6 w* near a third of `z*`; over the trade-wind sounding with BOMEX's fluxes the updraft saturates under
  the inversion and stops at it, none over a ground colder than its air; a step keeps the column's totals, carries
  the surface's vapour up, makes a convective cloud.

## Forcing
`WeatherForcing` (`src/weather_forcing.*`, engine-free): what drives the air from outside it.
- **The ground** per column, 2D (relief ignored): the blend of its biomes' `"weather"` (`Climate::BiomeWeather`,
  `docs/climate.md`; their weights at the column's ground point, `Weather::forcing_at`, sampled once at start):
  `ground_k` the ground's or the sea's mean temperature, `thermal_inertia` how its day swings, `wetness` (1 a sea, 0 a
  desert), `z0_m` its roughness, `storms`. The weather never changes the ground.
- **The day:** the planet's spin is the weather's day (`day_s`, one Omega for the sun and the Coriolis): the sun turns
  round the pole once a day, west (`WeatherForcing::turn_sun`, `Weather::day_angle`: 0 at the spin-up's end, so the
  first frame shows the scene's sun; `World::shown_sun` the turned one for the atmosphere and the clouds, and VoxelSky
  turns its light by `VoxelTerrain::get_day_angle`, the authored sun kept as the noon one). The main scene's sun is 28
  degrees north: a northern summer (the far north never sets).
- **The ground's temperature:** a baseline the sun sets directly, `ground_k + sun_k (L - 1/4)`: `L = cos(latitude) / pi`
  the column's year's mean of `max(up . sun, 0)` (taken as the equinox's day), 1/4 its mean over the sphere, so the
  planet's mean stays the climate's. A warm equator and cold poles, sea and land alike (a long mean does not depend on
  the heat a ground holds); `sun_k` 40: the equator +2.7 K, the poles -10 K. The planet starts on it: the ground at its
  baseline, each column's boundary layer (the sounding under the inversion) moved by its baseline's offset from the
  planet's mean, its vapour keeping its relative humidity, the free atmosphere over the inversion the planet's
  (`level_equilibrium`, linear across the inversion). The tropics' weak temperature gradient (Sobel & Bretherton
  2000): gravity waves spread a warm patch aloft in minutes, so the air aloft is alike over warm and cold grounds and
  the warmest grounds are the warmest against their sky: their thermals cross the inversion (deep convection, where
  typhoons start), a cold ground's air is capped (a polar inversion). Moving the whole column instead (before
  2026-11-27) capped every ground alike, and a deep atmosphere turned its equator-pole contrast into the pressure of
  the whole depth: with `top_m` 3000 `sun_k` 80 drove 45 m/s at the top and ran away, 16 held; with the free
  atmosphere shared, `sun_k` 40 holds (15 game minutes: cover 0.20-0.27, the strongest wind 6-8 m/s, ~215 ms a tick).
  The day swings it round that: a force-restore slab per column (Deardorff 1978),
  `dT/dt = 2 sqrt(pi) Q / (I sqrt(P)) - 2 pi / P (T - baseline)`: `P` the day, `I` the biomes' `thermal_inertia`
  `sqrt(k rho c)`, `Q` the sunlight absorbed over the baseline's (`1000 W/m^2 (1 - 0.15) ((1 - A) max(up . sun, 0) - L)`,
  `A` the albedo of the cloud over it: its water path, droplets of 10 um, a two-stream `(1 - g) tau / (2 + (1 - g)
  tau)`). The slab warms into the afternoon (its warmest an eighth of a day after noon). With the 300 s day a moist soil
  (1500) swings 2-8 K, dry sand (300) five times as much, a sea (1e5, its mixed layer) under 0.1 K; a cloud's shade
  cools what is under it (`test_weather_forcing.cpp`). Measured on the `fix` world (2026-11-26, 20 game minutes): with
  every latitude held at `ground_k` (before) the strongest wind was 2.4-3.0 m/s, the cloud 1-2 cells of cumulus
  everywhere; with the baseline (`sun_k` 40) 6.8-7.6 m/s in eddies over the whole planet, cloud sheets and bands, the
  cover 0.31 -> 0.17 over the 20 minutes (the circulation mixes the air's gradient, only the grounds restore it).
- **The air's relaxation** (`relax_days`, 2 weather days): each level's theta and vapour relax to the column's
  equilibrium, the start's state (`level_equilibrium`). Theta: the air's radiation, Held & Suarez 1994's Newtonian relaxation (they take 40 days aloft; a game's
  weather has minutes). Vapour: the exchange with the air over the 600 m lid (the boundary layer's moisture carried
  up, dry air sinking in: BOMEX's large-scale drying). Measured on the `fix` world over 90 game minutes (2026-11-26):
  theta alone held the cover ~0.3 for 30 minutes, then the seas' vapour, which only rain took out, filled the layer
  (0.79 and fog over a quarter of the ground at 90); with both, the cover 0.20 at the start, then 0.10-0.14 from
  minute 15 to 90, the strongest wind 6.5-7.2 m/s, no fog nor rain. `sun_k` and `relax_days` set the level.
- **No large-scale wind is imposed:** the air starts at rest, every wind is the sun's heating, the convection's and the
  storms', turned by the Coriolis (`f = 2 Omega up_y`, `Omega` the weather's day, `day_s` 300 s: the twist, storms spin
  at the game's scale). Removed 2026-11-26: Earth's three cells (`-sin(6 |latitude|)` east, 5 m/s) put in at the start
  and held by their balancing pressure gradient. Measured on the `fix` world (after the corner fix) over 20 game
  minutes: the wind stayed in steady bands round the axis (a 5-minute day balances them almost exactly), and the
  polar easterlies' jet at 75 degrees, its surface fluxes the strongest, kept a full-cover ring round each pole.
- **Storm seeds:** each step a column starts one with the chance its `storms` gives (a planet of storms 1 starts one an
  hour; a hash of the seed, the step and the column): +2 K and +2 g/kg in a bubble of `storm_m` over the lower half of
  the levels. What it grows into is the physics'.
- **The sounding** (the air's start and the buoyancy's reference): a trade-wind boundary layer, BOMEX's shape scaled to
  `inversion_m`: a mixed layer 1.5 K under the grounds' mean up to 0.53 of the inversion, its vapour saturating a
  ground parcel exactly there (the cloud base at the mixed layer's top, as in BOMEX; at 88% humidity the base sat 80 m
  over the thermals' reach and nothing formed in 20 minutes), a cloud layer (+4 K/km, drying), a 4 K inversion over
  40 m; above it, air at 45% of the mixed layer's vapour on the mixed layer's moist adiabat (`WeatherMoist::moist_adiabat`),
  kept as much warmer than it as at the inversion's top (convective quasi-equilibrium, Emanuel 1994): a saturated
  thermal stays 1.7-2.7 K colder than the air up to any top, only a ground past its baseline breaks through. The old
  3 K/km let a parcel through from 1.25 km up (+5 K at 3 km): with `top_m` 3000 the whole sky filled in 2 game
  minutes, updrafts at 50 m/s. 0.1 K of noise on the two lowest levels.
- **Tests** (`test_weather_forcing.cpp`): the sounding saturates at the base and not under it; the ground's noon, night
  and day's mean; storms at their rate, the same from the same seed. **A trade-wind sea** (all of it at once on
  6 x 32^2 columns: a sea 1.5 K over its air, the rotation, the air at rest), sampled each minute from 15 to 40: 24% of
  the columns cloudy on average, 36% at most (2026-11-26, exchange 40; with the prevailing wind imposed and
  exchange 15 it was 0.8% and 5.6%).

## Planet file
`"weather"`: on by default over a sphere (`"on": false` turns it off), refused over a flat shape or a torus (until its
grid lands). `cell_m` (the cells' width in metres; 0 or absent: 64 cells a cube face's side; refused where the side
would leave [4, 1024]), `top_m` (the levels' top over the ground, 600), `stretch` (each level's thickness over the one under it, the
lowest about `cell_m`, 1: all alike; in [1, 2]; "Levels"),
`tick_hz` (2: a tick costs ~250 ms at 70 cells a face side, 5 a second fell behind), `time_scale` (game seconds a
real second simulates, 1), `storm_m` (0: 1/8 of the
radius), `spin_up_s` (game seconds simulated before the first frame, 600), `day_s` (the weather's day, 300), `sun_k` (the sun's warming of the grounds' baseline, 40), `relax_days` (the air's relaxation, 2),
`inversion_m` (the start sounding's inversion, 300, under `top_m`), `exchange` (the surface fluxes' twist, 40: measured on the `fix` world without an imposed wind, cumulus over 10-18% of the planet against 8-11% at 15 and ~1% at 1, the air warming 0.2 K in 20 game minutes),
`convection` (the shallow convection's updraft, true) and `cloud_spread` (the partial cloud, true; false: each cell all
or nothing): the switches the debugger isolates a process with.

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
wind, raining share (rain over 0.1 mm/h), the day side's and the night side's mean cover, air at the ground and ground
temperature, each level's cloudy share of the planet, and the forcing's summary). The report's sun is an equinox one
(no host sets it). The test biomes of `ocean_test.planet.json` carry a
weather block (the ocean a sea at 300.5 K, smooth and stormy; the land at 299 K, a thermal inertia of 1200, 30% wet,
z0 0.1 m). On it (2026-11-25): no fog, a sparse cumulus field at 180 m along the eddies' edges, 4% of that level, mean
cover 0.05. **The day** (the sun turning): the ground's day side 0.6-0.8 K over its night side planet-wide (~2 K on the
land), the air at the ground 0.1 K at most, the cover the same by day and night (0.04-0.06 both). A day of 1800 s
(and its slower spin) gives the ground 1.6 K, the air still ~0.1 K, and no cloud at all: the clouds come from the
eddies' convergence lines, not from the ground's thermals. The mixed layer is 3-4 levels deep and a thermal ~2
cells wide: the surface's heat mixes through the subgrid turbulence instead of rising as resolved plumes, so a warm
afternoon made no cumulus. **With the shallow convection** ("Convection"): updrafts over 83-87% of the columns (the
sea always, the land by day), their tops at 265-300 m (the inversion); the cover from 0.38 after the spin-up down to
0.16-0.18 from 15 to 30 minutes on, the day side's 0.18-0.22 and the night side's 0.13-0.15; the cumulus at ~200 m
over 26-29% of the planet, wrapped in bands round the eddies; no fog, no rain. A tick 55 ms on average (44 without),
the forcing and the spin-up 48 s at load.

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

## Debugger
`res://debug/weather_debug/weather_debug.tscn`: a world's weather alone (no terrain, no mesher, no clouds), every field
the simulation holds, cell by cell. One path: the engine's own `Weather`, built with `p_probing` so that each published
state also carries a `WeatherDebug::Snapshot` (`src/weather_debug.*`, engine-free: the fields captured on the worker
after the tick, field-major) and run by a `WeatherDebug::Session` (the world's SDF for the forcing, the weather, a clock
of its own; the report runs through one too). `WeatherDebugNode` (adapter) only maps them onto Godot's types.
- **Typhoon benchmark:** `--vortex=<lat,lon,vmax,radius>` seeds a weak cyclone at the start (`Weather::seed_vortex`, any
  thread, applied at the next tick: `WeatherForcing::seed_vortex`, `v = vmax x e^((1 - x^2) / 2)`, cyclonic for its
  hemisphere, fading to none at the top; Rotunno & Emanuel 1987, Nolan et al. 2007); with `--trace` each line then
  prints `WEATHER_STORM`: `WeatherDebug::strongest_storm`, the deepest low on the lowest level (its pressure under the
  planet's mean, m^2/s^2), the strongest wind within 4 radii of it and that wind's distance (the radius of maximum
  wind), where it is. The yardstick is Emanuel's potential intensity, `V^2 = C_k / C_d (T_s - T_o) / T_o (k*_s - k)`:
  the drag and the exchange share one coefficient (`C_k / C_d = 1`).
- **Fields:** per cell and level the wind (east, north, w, speed), theta and its anomaly over the sounding, the
  temperature, qv, qs, the relative humidity, qc, qr, the cloudy share, Q1, sigma_s, the updraft's water, the buoyancy
  and the pressure; per column the ground (its temperature now and its biomes' mean, the ground over the lowest air, its
  wetness, z0, thermal inertia, storm odds), the surface's sensible and latent fluxes (W/m^2), u*, the rain, the updraft's
  top, the cover (the largest share) and the cloud water path.
- **The view:** the planet is the grid's cells (`planet_mesh`: a quad per cell, its edges drawn), coloured by the field at
  the level (Turbo; a signed field blue-white-red round 0; the range the field's own unless locked), the wind at that
  level as lines. A click picks a cell: its column as a table (every level, the shown one yellow; the column's values
  under it) and the section through it (its face's row, the levels up).
- **Clock:** paused at the spin-up's end; space runs it (speed x0.1 to x20, the worker the limit: Weather::MAX_BEHIND),
  N one tick, M ten. The spin-up is the world file's unless the field or `--spin_up=` says (0: watch it from the start).
- **Run:** `godot --path adapter/godot/project res://debug/weather_debug/weather_debug.tscn -- [--world=PATH]
  [--spin_up=S] [--field=NAME] [--level=K] [--pick=FACE,I,J] [--shot=PNG --ticks=N]` (`--shot`: N ticks, the window
  saved, the picked column printed, quit; headless it prints the column without the picture).
- **Checked:** `test_weather_debug.cpp`: a probing weather's snapshot is its published state (the cloud water and the
  ground level's temperature), slices and sections read what `at()` reads, no snapshot unless probing.

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
12. **The day:** the sun turning with the planet's spin (the scene's light too), the ground a force-restore slab
    (thermal inertia per biome), clouds shading it; the report's day and night sides.
13. **Shallow convection:** EDMF's updraft per column (`WeatherConvection`), its cloud published and shading the
    ground.
14. **Turbulence:** the kicks and the vorticity confinement, then (2026-11-25) both removed for an implicit
    large-eddy simulation, checked on the dry convective boundary layer; the vertical staggering of w; the advection
    one conservative flux-form scheme for every field (MacCormack created water on BOMEX).
15. **Partial cloud:** the statistical cloud scheme in the physics; the render draws the simulation's water, the
    noise only its texture.

Later: curl-noise gusts, the FFT sea reading the wind (U10, fetch, swell), rain and ground wetness, saving the weather,
the GPU if the CPU tick is too slow.
