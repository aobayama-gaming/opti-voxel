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
  the caller's physics on the state, `force` (the vorticity confinement, "Turbulence" below; the buoyancy along the up, the Coriolis turn about it, a sponge damping
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
  mixed layer at 298 K and 18 g/kg, an inversion of 4 K at 300-340 m, a sea 1.5 K warmer, 0.1 K of noise): averaged each minute from 15 to 25 minutes
  (one snapshot is one cloud's life), cumulus covering 0.5 to 1.6% of each level from 180 to 300 m (BOMEX's 5-10% near
  the base: the cells resolve few of them, the shallow convection makes the rest on the planet), no cloud at the
  ground nor above the inversion at any sample: broken, clear air between them. **Sea fog** (98% humidity over ground 6 K colder:
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

## Turbulence
The 45 m cells resolve the thermals but the semi-Lagrangian's interpolation smears every eddy under ~4-8 cells, and the
cube's 4-fold symmetry seeds what is left: clumps, no swirls. Two additions, one tried and dropped (2026-11-25):
- **Vorticity confinement** (`WeatherFluid::confine`, at the start of `force`; Fedkiw, Stam & Jensen 2001, Harris
  2003's clouds): the curl `w` of the wind from central differences (the neighbours' winds as they are: the sphere's
  turn over a cell is 2% of a radian; one-sided at the ground and the top), `N = grad |w| / |grad |w||`, the force
  `epsilon D (N x w)`, `D` the cell's `(width^2 dz)^(1/3)`. It feeds back the swirls the grid's dissipation takes out.
  Only the curl's part about the up (the swirls seen from above). `epsilon` is the planet file's `confinement`, 0.02
  (with the partial cloud, below: the eddies at 0.52 w*, under a convective layer's ~0.6, the cover BOMEX's "<20%"). 0.05 balances the dissipation at 2 cells (a 2-cell vortex of 4 m/s keeps 0.26 of its peak over 5
  minutes without it, ~1.0 at 0.05, 2.4 at 0.15, 6 at 0.3: the smoke solvers' 0.3 runs away on a grid this coarse,
  `test_weather_turbulence.cpp`), but on the planet it cleared every cloud, even at 0.02 (table below; the whole
  curl did too, sharpening the shear over the ground). Times the subgrid closure's stability factor
  `sqrt(max(1 - Ri / Pr, 0))`: unlimited it stirred the stable air over the inversion (rms w 0.03-0.1 -> 0.5-0.76 m/s,
  K 0 -> 0.4-1 m^2/s, the inversion's jump 2.35 -> 0.57 K). Why it clears the clouds, measured on a trade-wind sea
  (6 x 32^2 columns, minutes 15-30, the game's step order, 2026-11-25):

  | epsilon | peak sigma_w / w* | cloudy columns | RH at 180 m | updraft cloud water |
  |---|---|---|---|---|
  | 0 | 0.40 | 86% | 0.996 | 3.8e-6 |
  | 0.005 | 0.41 | 76% | 0.994 | 3.2e-6 |
  | 0.01 | 0.41 | 54% | 0.984 | 1.2e-6 |
  | 0.02 | 0.51 | 1.8% | 0.940 | 8e-10 |
  | 0.05 | 0.8-0.9 | 0% | 0.86 | 0 |

  More resolved stirring entrains the dry, warm air over the inversion into the cloud layer (the layer warms 0.5-1 K,
  the air over the inversion gains 1-3 g/kg; without the shallow convection the same: not a double count). The clouds
  die long before the eddies are unphysical (0.51 w* at 0.02; convective boundary layers peak ~0.6 w*): a cell is
  cloudy only when its mean is saturated, so the cloud deck here lives at RH 0.99, kept that wet by the grid's
  under-mixing, and a 5% drier mean clears it. Real cumulus form at a mean RH of ~0.8-0.9 because the air inside a
  cell varies; the model had no such variance: the partial cloud below.
- **Kicks** (`WeatherForcing::kicks`, in `apply`): small swirls of random sign the grid's symmetry would never start,
  a stochastic backscatter (Shutts 2005, the motion under the grid fed back). Each step a column starts one with a
  chance that puts a new one over any point every `KICK_EVERY_S` (60 s) on average (a hash of the seed, the step and
  the column: the same every run): a Gaussian vortex, speed `kick_mps r exp((1 - r^2) / 2)` with `r` in units of 2
  cells, through every level, cut at its cube face's edge (the projection takes what that leaves out). The start is
  kicked all over once. `kick_mps` (planet file, 1): a minute of kicks on still air peaks at 5 m/s where they overlap.
- **MacCormack advection, dropped:** Selle et al. 2008's second-order correction with their limiter (a trace back, the
  first corrected by half the round trip's error, reverted where it leaves the interpolated range). On every field
  the trade cumulus became a deck (0.83 of a level) and a fog rose to 180 m; on the wind alone the resolved cumulus
  nearly stopped (0.07% of a level against 1.6%) and a bubble's air was 5% unbalanced. The moist physics are tuned to
  the semi-Lagrangian's mixing; a sharper scheme needs the horizontal subgrid mixing and the physics retuned with it
  (a horizontal Smagorinsky in the same try did not bring them back).
- **Measured** (`ocean_test`, 25-minute spin-up, `weather_report.sh`, 0 / 5 minutes after it, 2026-11-25):

  | confinement | kick_mps | cover | air (C) | updrafts' top | max wind (m/s) |
  |---|---|---|---|---|---|
  | 0 | 0 | 0.38 / 0.28 | 25.2 | 300 / 283 m | 9.0 / 8.0 |
  | 0 | 1 (default) | 0.22 / 0.19 | 25.2 | 302 / 282 m | 9.0 / 8.9 |
  | 0.02 | 0 | 0.04 / 0.01 | 25.3 | 272 / 252 m | 10.4 / 11.1 |
  | 0.05 | 0 | 0.00 / 0.00 | 25.4-25.5 | 252 / 247 m | 12.6 / 15.0 |
  | 0.05 | 1 | 0.00 / 0.00 | 25.8-26.0 | 205 / 196 m | 13.2 / 11.1 |

  The kicks cost a third of the cover at the start and less after; the shapes they make are judged in the viewer.

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
  dissipation `sigma^2 / tau`): `sigma^2 = 2 tau |w's' ds/dz|`, the flux the subgrid turbulence's `-K ds/dz` plus the
  shallow convection's updraft `M (s_u - s)` (convection sets the humidity's variance, as Lenderink & Siebesma 2000
  couple a mass flux to a statistical cloud scheme); `tau = z_i / w*`, the boundary layer's turnover (z_i the updraft's
  top, w* the surface buoyancy flux's; at least u* and 0.1 m/s). The updraft's own cloud is no longer added on top:
  the partial cloud is the cloud.
- **The render** (`docs/look.md` "Density"): the spread's distribution is fixed by the fit (`N = P(s > 0)` makes its
  CDF `0.5 + 0.36 atan(1.55 x)` on `|x| <= tan(0.5 / 0.36) / 1.55`), so a uniform noise `u` through
  `WeatherCloud::spread(u) = tan((u - 0.5) / 0.36) / 1.55` saturates exactly the share N of a cell's points. The weather
  publishes each level's mean cloud water and its Q1 (`WeatherMoist::q1`, within +-8, a uniform cell's +-8 by its
  sign); a point's cloud is `max(Q1 + spread(u), 0) / mean_excess(Q1)` of the cell's mean water, `mean_excess` the
  closed form of `E[max(Q1 + n, 0)]` over the distribution: the cell's points average to the simulation's water, the
  cloudy ones are its share. `u` is `WeatherCloud::uniform_noise`: a tileable gradient fbm (128^3, 8 lattice cells and
  3 octaves, a period of 4 weather cells: features of half a cell to an eighth), its ranks turned into values, so it is
  exactly uniform (each tenth holds a tenth of the texels to within one) and as smooth as the noise. The physics never
  reads it (`test_weather_cloud_fraction.cpp`: the closed form against the sum, the noise's histogram, its wrap).
- **Measured** (trade-wind sea, 6 x 32^2, the game's step order, minutes 15-30): cover (each column's most clouded
  level) 49% with no confinement, 37% at 0.01, 17% at 0.02, 5% at 0.05 (all-or-nothing: 86%, 54%, 1.8%, 0%): more
  stirring thins the clouds instead of clearing them. sigma_s 0.1-0.45 g/kg, largest in the cloud layer.
- **Open:** the levels alternate (180 and 260 m cloudy, 140, 220 and 300 m much less; the vapour alternates too, 19.6 /
  17.7 / 19.2 g/kg): a vertical odd-even mode, older than this, to find.

## Convection
`WeatherConvection` (`src/weather_convection.*`, engine-free): the shallow convection the cells cannot resolve (3-4
levels in the mixed layer, a thermal ~2 cells wide: without it the ground's heat only diffused, and a warm afternoon
made no cumulus). The mass-flux half of EDMF (Siebesma, Soares & Teixeira 2007, JAS 64, equation numbers theirs), its
eddy diffusivity the moist physics' Smagorinsky turbulence; one updraft per column, each step before the moist physics
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
- **The ground's temperature:** a force-restore slab per column (Deardorff 1978), `dT/dt = 2 sqrt(pi) Q / (I sqrt(P)) -
  2 pi / P (T - ground_k)`: `P` the day, `I` the biomes' `thermal_inertia` `sqrt(k rho c)`, `Q` the sunlight absorbed
  over the day's mean (`1000 W/m^2 (1 - 0.15) ((1 - A) max(up . sun, 0) - mean_light)`, `mean_light` Berger's daily
  insolation for the column's latitude and the sun's declination, `A` the albedo of the cloud over it: its water path,
  droplets of 10 um, a two-stream `(1 - g) tau / (2 + (1 - g) tau)`). Clear, the day's mean is `ground_k`; the slab
  warms into the afternoon (its warmest an eighth of a day after noon). With the 300 s day a moist soil (1500) swings
  2-8 K, dry sand (300) five times as much, a sea (1e5, its mixed layer) under 0.1 K; a cloud's shade cools what is
  under it (`test_weather_forcing.cpp`).
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
`inversion_m` (the start sounding's inversion, 300, under `top_m`), `exchange` (the surface fluxes' twist, 15),
`confinement` (the vorticity confinement's epsilon, 0.02, in [0, 1]), `kick_mps` (the random swirls' speed, 1).

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
14. **Turbulence:** the kicks and the vorticity confinement (the curl about the up, where the air can be turbulent);
    MacCormack tried and dropped.
15. **Partial cloud:** the statistical cloud scheme in the physics, the render's noise through its spread.

Later: curl-noise gusts, the FFT sea reading the wind (U10, fetch, swell), rain and ground wetness, saving the weather,
the GPU if the CPU tick is too slow.
