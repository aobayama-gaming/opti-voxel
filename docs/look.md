# Look (sky, fog, clouds, surface detail)

The shared environment and the page shader's surface detail.

- **Look (sky, fog, clouds, surface detail).** The scenes (`main_scene*.tscn`, `debug/bench.tscn`) share one environment:
  a sun at 28 degrees (warm, `light_angular_distance` 0.53 so the sky draws its disk), exponential fog with aerial
  perspective (`fog_aerial_perspective` 0.8: the fog takes the sky's colour in that direction, `fog_density` 0.00025: a
  quarter at 1 km), SSAO / SSIL, the grading LUT. `VoxelSky` (`src/voxel_sky.*`, a sibling of the terrain and the
  WorldEnvironment) installs the sky shader at ready: the atmosphere's sky, or without one Godot's PhysicalSkyMaterial
  (MIT), with the volumetric clouds over either (below: the weather's, planets only; the flat 2D cloud layer is gone).
  The page shader splices the same GLSL (`VoxelSky::glsl_clouds`: the base's `base_up` / `base_altitude` /
  `base_level_span` (core, `BaseSurface::glsl_render`) and `wind_sway`); its `light()` (Lambert, no specular: every
  material has roughness 1) takes the sun through the clouds' shadow map. `VoxelSky` hands the terrain `wind_sway_m`
  through `VoxelTerrain.set_page_params(Dictionary)` (kept until the pool exists). **Surface detail**, all procedural, world-space
  value noise (the same at every LOD): `ground_detail` on the procedural ground and `PATTERN_GROUND` (a tint over ~20 m,
  patches of ~2 m, a grain of 6 cm that fades past 2 pixels per feature), `PATTERN_LEAVES` on the leaves (clumps of
  0.5 m from dark to yellowish green, 8 cm leaves, darker undersides). **Wind** (`VoxelSky.wind_sway_m`, 3 cm; 0 =
  still): the vertex stage moves leaf corners by `wind_sway` (a gust wave riding the global `weather_wind`, the weather's
  wind at the camera that `VoxelTerrain` sets each frame from `World::camera_wind` (0 without weather), plus a finer wobble, a
  pure function of the position and TIME: shared corners agree, the mesh stays watertight) up to LOD 2, where each
  tree corner pays a palette and a material fetch, and the leaf pattern's noise drifts (a shimmer). Bark stays still:
  the pages carry no per-tree pivot to bend a trunk around. Cost unmeasured. **Shading frame** (`src/shade_frame_glsl.hpp`):
  every pattern of the position (the value noises, the bark's height, the lava and wind waves, the contour lines, the
  clouds' noises) is read at the render position plus the render origin's voxel modulo a period of the pattern
  (`shade_point`: a power of two of voxels holding at least 64 of its largest features, `shade_period_vox`) with its
  frequencies snapped to whole cycles per period (`shade_cells`, `shade_wave`; `value_noise` wraps its lattice): two
  render origins read points a whole number of periods apart, so a rebase changes nothing on screen and the floats stay
  small anywhere. The contour lines are halfway between whole multiples of 100 m of altitude above the
  base. The hosts give the frame its origin through `shade_origin_voxel()` /
  `shade_voxel_m()` (the page shaders from their decode uniforms, the sky shader from `VoxelSky::set_render_frame`,
  which VoxelTerrain calls at start and on every rebase).
  **The base** (`docs/terrain.md`): `World::base_render` gives both shaders the base in render space at start and on
  every rebase (`BaseSurface::Render`: its kind and, for a planet, its centre, radius and K = |c|^2 - R^2 from double,
  so the altitude (|p|^2 - 2 p.c + K) / (|p - c| + R) keeps ~1e-4 m in float32 at Earth's radius, where |p - c| - R
  keeps half a metre). No render code knows the kind: the page shader reads the up per fragment (`base_up`: the slope
  colours, the materials) and colours and contours by altitude (sea level 0), the sky and the clouds go through
  `base_up` and `base_level_span`;
  bark and lava read the up's tangent frame (`up_frame`, x / y / z on the flat world). The shading frame snaps per
  axis, not along a tilted frame: on a planet away from the poles bark and lava shift when the render origin moves
  (rare: `rebase_distance_m`); cloud shadows read three planes there (three fetches). The sky is drawn in the
  camera's local frame (its up on +y), so the horizon is the ground's and the global sun sets as the camera goes
  round.
  `tools/shot_diff.py` compares two shots of one pose (a fly far out and back with `--still=1`). The sea and its shore:
  below ("Sea"). Not done yet, in order: triplanar image textures per material, grass, time of day.

## Atmosphere

A volumetric atmosphere (`src/atmosphere.*`, `ov::Atmosphere`, owned by `World`; the world file's `look.atmosphere`, on by default) made
only from the base's shape: Hillaire 2020 ("A Scalable and Production Ready Sky and Atmosphere Rendering Technique",
UE's SkyAtmosphere; reference code `sebh/UnrealEngineSkyAtmosphere`, MIT), its sphere assumptions replaced by the base's
render contract (`BaseSurface::glsl_render`, `docs/terrain.md`), so another base kind gets an atmosphere that follows it.
- **Medium:** Rayleigh, Mie (Cornette-Shanks, g 0.8, with absorption) and an ozone tent, Hillaire's Earth values, all
  a function of `base_altitude` alone (clamped at 0 below the base). Every ray's extent is `base_level_span` (the top,
  the ground), at most `atmo_max_path_m` long (a ray along the flat world's plane never leaves it). **Scale**
  (`AtmosphereMedium::for_base`): under Earth's radius every height times R / R_earth and every coefficient over it (the
  same optical depths: a small planet keeps a blue sky and a limb), the top at least `RELIEF_TOPS` (4) times the
  ground's relief (`SDFCascade::ground_top_m`: the program's output bound over the base, its highest possible point,
  0 for a ground that never rises above its base) so the terrain stays inside the air: Earth and the flat world keep
  Earth's 100 km, a 2 km planet under `tiny.world.json`'s 819 m gets 3.3 km. `look.atmosphere_scale` sets the factor.
  A planet file's `"atmosphere": {"density", "tint"}` (`WorldFile::atmosphere_density|tint`) multiplies every
  coefficient (scattering and absorption: thinner or thicker air, the same colour) by the density and Rayleigh's by the
  tint too; 1 and white keep the bits.
- **Passes** (compute on the main device every frame, `device->post` like the mesher; a `atmosphere` GPU timestamp):
  the multiple-scattering table ψ_ms(sun height, altitude), 32 x 32 (Hillaire's: second order with an isotropic phase
  over 64 directions, 1 / (1 - f_ms)), built once on a sphere of the base's curvature radius (Earth's on the flat
  world): the only local approximation, Hillaire's own. Then the **camera volume**: a camera-centred grid, 128
  azimuths x 64 elevations round `base_up` at the camera (the elevation squared from the horizon, so the texels are
  finest there), 32 slices, quadratic out to `range_m` (the camera's horizon plus 128 km), the last slice the whole
  ray (the sky, the base's Lambert bounce at its end). One invocation marches a column; each sample's sun
  transmittance is an 8-step march through `base_altitude` (no (r, mu) transmittance table). A texel holds the
  in-scattered light for a sun of illuminance 1 (RGB) and the mean transmittance (A), like UE's aerial perspective:
  an RGBA16F atlas 128 x 2048. Independent of the camera's rotation: turning needs no reprojection.
- **Hosts** (`Atmosphere::glsl_lookup`): `atmo_at(camera, dir, dist)` (the slices around dist, linear between them) and
  `atmo_sky`. The sky shader draws the sky slice, or from above the atmosphere marches per pixel (24 steps:
  `atmo_integrate`, the volume's texels too coarse for the limb), plus the sun's disk through the air; the cloud layer
  is lit by the sun's transmittance at the cloud and hazed by the air in front of it; the sea pass (below) reflects
  `atmo_sky_radiance` and hazes its surface with `atmo_at`. The page shader writes Godot's
  `FOG` = (S / (1 - T), 1 - T), i.e. colour T + S, and `light()` takes the sun through an 8-step march from the
  fragment (red at dusk, none past the base's terminator). The sun's light is the `DirectionalLight3D`'s colour times
  its energy times `VoxelSky.atmosphere_gain` (pi: Godot's lights carry it, so the sky over the ground keeps
  Hillaire's physical ratio). Per frame three global shader uniforms (`atmo_range_m` from VoxelTerrain, `atmo_sun_light`
  and `atmo_sun_dir` from VoxelSky, which also hands the terrain the sun's direction; without a VoxelSky, as in the
  mixer's view, VoxelTerrain sets `atmo_sun_dir` from its DirectionalLight3D sibling); the textures reach the materials through
  `texture_rd_create` (the pages) and the sky material's RID. With the atmosphere on, the environment's fog stays
  enabled at density 0 (Godot drops a shader's `FOG` without it): the atmosphere is the haze.
- **Checked:** `AtmosphereCheck` (in `frame_test.sh`'s report, `atmosphere_check`): the passes against a CPU reference
  in double, sample for sample, for a flat base, Earth at its pole and 100 km up and a 2 km planet: worst relative
  error under 0.1% (half floats), and the running object's volume. Cost (forest5cm, steady, RTX 3070 Laptop, 2 runs
  each): 17.65 -> 18.09 ms, +0.44 ms.
- **Limits:** the multiple scattering is a sphere's (a strongly curved, non-spherical base would want it per region);
  the volume is rebuilt every frame from the camera, so a fast flight shows its one-frame lag far away; terrain past
  `range_m` takes the range's air; no light shafts through terrain, no eclipses, no spectral rendering. The Dawn viewer
  has no atmosphere yet. Volumetric clouds below.

## Volumetric clouds

Volumetric clouds (`src/clouds.*`, `ov::Clouds`, owned by `World` with the atmosphere and the weather; `look.clouds`, on
by default). They are the weather's (`docs/weather.md`): a world without it (a flat world, a torus, `"weather": {"on":
false}`) has no clouds. `World` starts the atmosphere and the clouds only for a host that binds their textures (`Hosts::atmosphere_textures`, `Hosts::cloud_textures`): the Dawn viewer runs neither yet.
They are made from the base's shape like the atmosphere: a layer
between two altitudes above the base (`base_altitude`; a ray's part in it from `base_level_span`), every noise read at
the 3D point in the shading frame (no 2D weather map, which a curved base would tear).
- **Density:** Schneider's (Horizon Zero Dawn's Nubis, SIGGRAPH 2015 / 2017; ported from clayjohn's
  godot-volumetric-cloud-demo, MIT): a 128^3 Perlin-Worley shape noise (R; G B A Worley fbms) and a 32^3 Worley detail
  noise, 3D RGBA16F textures made once by a compute pass, read through the device's linear sampler (hardware
  trilinear). The cover and the cloud type come from the weather (below), shaped by the shape noise read at a weather
  scale (its R past a threshold set by the cover, measured on the noise so the cover is about the fraction of sky
  covered); the type
  blends Schneider's stratus / stratocumulus / cumulus height gradients; the detail erodes the base (billows below,
  wisps above). Periods (Earth): shape ~12 km, detail ~1.5 km, weather ~48 km, each a power of two of voxels (the
  shading frame). On their own the whole field repeated every weather period (on the flat world at 5 cm: the same clouds
  every 26.2 km, the same shapes every 6.55 km, `--cam_pos` shifted top-down shots: 4% of cloud pixels differed a
  weather period over). So every read is **warped** (`cloud_warp`): a smooth offset of two sine waves per axis, whole
  cycles per a ~800 km period (`warp_period_vox`; at most 2^23 voxels, at least 4 weather periods), |k| = sqrt(5) cycles
  and 0.015 periods each (`WARP_AMPLITUDE`: 6 km at 5 cm), carried by the wind with the noises (the clouds drift whole):
  points a shape period apart read noises ~1.5 to 2 km apart (its features ~1.6 km), a weather period apart 5 to 9 km
  (~6.5 km), the clouds stretched by 2 x 2 pi sqrt(5) x 0.015 = 0.42 at most (~0.2 typical). The field now repeats every
  warp period (419 km at 5 cm; 34% of cloud pixels differ a weather period over, 42% a shape period over, ~42% for
  unrelated fields at that cover), for +0.17 ms of cloud passes (1280x720, the video preset; measured before the passes
  skipped the weather outside the layer). A warp read from the shape noise itself sheared the clouds into streaks (its
  fine octaves: gradients ~1.5). The wind's offset wraps at the warp's period.
- **Weather** (`Weather`, `docs/weather.md`): the simulation's cover and type over the whole planet. When a tick
  lands (`World::frame`: `Weather::update`), `Clouds::set_weather` uploads its last two states as one 3D RGBA16F
  texture of the cube-sphere grid, (N + 2) x (N + 2) x 6 (the halo included; RG the previous state's cover and type, BA
  the latest's: 0.3 MB at N = 64), and keeps the largest cover of the two (`cloud_cover_max`: the passes skip where it
  is 0, as before the first state lands). `cloud_weather` reads it along the base's up at the point (`weather_uvw`,
  `WeatherGrid::glsl_lookup`; the face's texels at their centre, so the linear sampler never blends two faces; the halo
  makes the bilinear read cross a seam), blends the two states by `Weather::blend` (where the frame stands between
  them: the cover moves smoothly between ticks) and moves the type round the weather's (the noise's offset from 0.5
  moves it towards 0 or 1 by up to 0.4 of the way, `TYPE_SPREAD`, so it stays in [0, 1]). The noises drift with the
  weather's wind at the camera times its time scale (`World::camera_wind`: the pattern moves with the cells' cover).
- **Layer** (`CloudLayer::for_base`): Earth's 1.5 to 4 km, extinction 0.04 /m at density 1 (Hillaire 2016's cumulus
  range is 0.05 to 0.12), albedo 0.99; under Earth's radius scaled like the atmosphere (lengths times s, extinction over
  s, s at least `RELIEF_BOTTOMS` reliefs over Earth's bottom), then lifted, its thickness kept, until its bottom is
  `RELIEF_BOTTOMS` (1.25) reliefs up: the march knows nothing of the terrain, so no cloud may reach into it. The
  relief is the program's top (`SDFCascade::ground_top_m`, a bound: the template ground's sum of amplitudes, ~6.6 km at
  5 cm and 0.5 m alike, puts the layer at ~8.2 to ~10.7 km; the old fixed 3.3 km relief had it at 4.1 to 6.6 km); a 2 km planet under
  `tiny.world.json`'s 819 m at ~1 to ~2.7 km.
- **Light:** Hillaire's (Frostbite, "Physically Based Sky, Atmosphere and Cloud Rendering", SIGGRAPH 2016): the
  energy-conserving step, a dual-lobe Henyey-Greenstein phase (0.8 forward, -0.3 back, 0.7 / 0.3), the sun through the
  air (`atmo_transmittance` at the ray's two ends, linear between) and through the cloud (6 steps from 1% of the
  thickness, growing 1.8 times: Frostbite's progressive shadow samples), Wrenninge's multiple-scattering octaves (3:
  octave o scatters 0.75^o of the light through 0.12^o of the extinction, its phase 0.5^o as eccentric) and the sky's
  light from the atmosphere's multiple-scattering table (its isotropic sky radiance at the altitude and the sun's height
  there), times a bottom-to-top gradient (Frostbite's). The octaves are Frostbite's "exaggerated" multiple scattering
  (a > b is not energy-conserving): with a <= b a cloud's inside, hundreds of optical depths from the sun, stays black.
- **Passes** (compute on the main device after the atmosphere's; a `clouds` GPU timestamp): the **grid**, camera-centred
  like the atmosphere's volume, 1536 azimuths x 768 elevations round `base_up` at the camera (linear elevation, 0.23
  degrees). One invocation per 2 x 2 block (every lane of a warp marches) marches one of its texels in turn (16 to 64 steps over the ray's part in the layer, out to the
  camera's horizon plus the top's, jittered by interleaved gradient noise), blended half with the last frame's; the
  others take the last frame's, reprojected by the cloud's mean distance (the camera moved; a jump over a thickness
  resets). A texel holds the clouds' light for a sun of illuminance 1 with the air in front of them (RGB: the
  atmosphere's volume at their mean distance) and their transmittance (A), and the front and mean distances in km
  times the opacity (0 where there is no cloud; divided back by the filtered opacity, so filtering and blending weigh
  them by it). The **shadow map**: 512^2 texels along the sun round
  the camera (64 km on Earth, or 4 camera altitudes; its origin snapped to its texels), each column marched (16 steps
  of the base shape) from where its line leaves the base towards the sun, its transmittance through the octaves.
- **Hosts** (`Clouds::glsl_lookup`): `cloud_at(camera, dir, dist)` (the grid at dir; a cloud starting past dist left
  out, one dist enters partly) and `cloud_sun_shadow(p)` (one fetch). The sky shader puts the whole ray's clouds over
  the sky; the sea pass (below) puts them in its reflection and in front of its surface and shades its surface with the
  shadow map; the page shader puts the clouds in front of a fragment over its `FOG` (combined: T = T_air T_clouds,
  S = S_air T_clouds + S_clouds) and its `light()` takes the sun through the shadow map. The grid's and the shadow map's
  textures reach the materials like the atmosphere's (`VoxelTerrain::_rt_share_textures`); the shadow map's frame is
  four global uniforms per frame (`cloud_shadow_origin_m`, `_u`, `_v`, `_extent_m`).
- **Checked:** unit tests (the layer's scale and lift, the periods, the substitution, the block, the shadow map's
  frame; the passes read the weather texture, every placeholder substituted); before the weather: origin parity
  with the clouds in view (34 rebases, 0.02% of pixels differ); `GPUDeviceCheck` covers the 3D textures and the sampler; screenshots from the forest floor, Earth's pole,
  inside the flat world's layer, Earth from 100 km and the 2 km planet from the ground and from 4 km out.
- **Limits:** the weather's cells (~49 m on the 2 km planet at N = 64) low-pass the cover; the noises drift with
  the wind at the camera, not each cell's; the layer's altitudes are the same everywhere (a cumulonimbus is a cumulus,
  not a taller layer); no clouds on a flat world or a torus (no weather there). The field still repeats every warp period, at most 2^23 voxels (`ShadeFrame::MAX_SHIFT`: 419 km at
  5 cm; a tile index from the origin's high bits could hash each tile instead); the light march reuses its start
  point's warp (as it does the weather); the grid's 0.23 degrees show as jagged edges on a small planet seen whole; the reach is a horizon on the
  base's curvature sphere, so a torus seen from space has clouds on its near side only; a cloud in front of terrain is
  composited by its front and mean distances, so a hill reaching into one would be cut by texels (the layer is lifted
  over the world's largest relief instead, so no cloud wraps a peak); the wind is a render-space vector, so on a planet
  the noise drifts through the shell away from the pole; the grid's history is copied every frame (a ping-pong would
  save the copies), the noises have no mipmaps (distant rays sample them minified) and the shadow map is marched every
  frame; no shadow past the shadow map (faded over its last tenth); the clouds take the scene's sun
  colour, warm here, through the air (no white balance); no cloud shadows on the atmosphere (light shafts), no
  lightning, no precipitation. The Dawn viewer has no clouds yet.

## Sea

A world file's `"sea"` (`WorldFile::Sea`: `level_m` above the base, `water` {`deep`, `shallow`, `clarity_m`}, `shore`
{`below_m`, `above_m`, `sand`}; `docs/terrain.md`) drawn by a full-screen pass over the scene, and its shore on the
ground. The look is core (`src/sea.*`, `Sea`): its GLSL (`glsl_water`, `glsl_shore`) and its uniforms (`Sea::Render`,
`World::sea_render`, set again on every rebase like the base's); a host splices the text and draws the pass.
- **The surface:** the level along each pixel's ray in closed form, `base_level_span_k` (`BaseSurface::glsl_render`),
  its `k` from double on the CPU (`BaseSurface::level_k`: a sphere's |c|^2 - (R + h)^2, so the level keeps its bits at
  Earth's radius where `base_level_span`'s h (2 R + h) term would not; the flat base's level plane). The water along the
  ray: from the camera, or the surface, to the scene's depth or the far side of the level. A torus has no closed form:
  the world file refuses a sea on one (TODO: a root solver; `base_traced_span` is too costly per pixel).
- **The water:** Beer-Lambert per channel over the ray's length in it, its extinction `-ln(shallow) / clarity_m` (a
  column of `clarity_m` metres passes `shallow`), the light it scatters `deep` times what it takes, lit by the sun (its
  transmittance through the air, the clouds' shadow map) and the sky above. The surface: Fresnel (Schlick, F0 0.02)
  times the sky in the reflected direction (`atmo_sky_radiance` and `cloud_at` with the atmosphere, else `sea_sky`,
  VoxelSky's fog colour), a GGX glint of the sun (`atmo_sun_dir`, `atmo_sun_light`), 5 Gerstner waves in the normals only
  (24 m down to 2.5 m, read through the shading frame: they stay put on a rebase), each faded into the glint's roughness
  where it is under 4 pixels (Bruneton's filtering in spirit: the waves under a pixel become roughness). Foam where the
  water is under 0.5 m deep, a drifting value noise. The air and its clouds in front of the surface (`atmo_at`,
  `cloud_at`) over the water's light; the scene behind already carries its own.
- **Under water** (the camera below the level): the water from the camera to the scene, or to the surface over it,
  absorbs and scatters as a fog, lit through the water above the camera. No surface seen from below (no total internal
  reflection, no Snell window) yet.
- **The shore** (the page shader, `sea_shore` on `v_height`): the sand colour over [level - `below_m`, level +
  `above_m`] with soft edges, darker up to 0.6 m above the waterline (wet; its roughness lowered too, which shows once
  `light()` takes a specular lobe). Neutral without a sea (`sea_on` 0). No tree or rock stands under level + `above_m`
  (the world file's site floor, `SDFTable::set_site_floor_m`).
- **Godot host** (`GodotPageHost::set_sea`, made when the world has a sea): one quad over the whole screen (its vertex
  stage writes `POSITION = vec4(VERTEX.xy, 1, 1)`; ignored by culling), drawn with the transparent objects (unshaded,
  no depth test or write, no fog). It reads the scene's depth (`hint_depth_texture`, Godot's reverse z: 0 is the far
  plane, the sky) and colour (`hint_screen_texture`) and replaces the colour where its ray meets water (`ALPHA` 1), so the
  transmittance is per channel. Its material takes every page uniform (`set_material_param`: the decode's origin and
  voxel, the base, the atmosphere and clouds, the sea), so the mixer's 3D view shows it too.
  `VoxelTerrain.get_sea_height_at(pos)` (`World::sea_height_at`: the base's altitude in double minus the level, NAN
  without a sea): the player's HUD says "under water".
- **Limits:** Dawn draws no sea yet (flat only there, no depth binding, a display-encoded target); waves are in the
  normals only (no displacement: the silhouette against the sky stays a smooth level; near-field swell is the next
  step); the reflection reads the sky only (no terrain, no screen-space reflection); no refraction offset, no caustics;
  the water's light is one scattering colour (`deep`), not a phase function. Cost unmeasured.
