# Look (sky, fog, clouds, surface detail)

The shared environment and the page shader's surface detail.

- **Look (sky, fog, clouds, surface detail).** The scenes (`main_scene*.tscn`, `debug/bench.tscn`) share one environment:
  a sun at 28 degrees (warm, `light_angular_distance` 0.53 so the sky draws its disk), exponential fog with aerial
  perspective (`fog_aerial_perspective` 0.8: the fog takes the sky's colour in that direction, `fog_density` 0.00025: a
  quarter at 1 km), SSAO / SSIL, the grading LUT. `VoxelSky` (`src/voxel_sky.*`, a sibling of the terrain and the
  WorldEnvironment) installs the sky shader at ready: Godot's PhysicalSkyMaterial (MIT) with a **cloud layer** over it, a
  layer `cloud_height_m` above the base (a plane on the flat world, a shell on a planet; read on three planes blended by
  the up so it has no seam, a plane with no weight skipped) reading `cloud_texture` (`adapter/godot/project/clouds.png`, `tools/make_clouds.py`: tileable spectral
  noise, rank-normalised so `cloud_cover` is the fraction covered) once per `cloud_scale_m`, carried by `cloud_wind`. The
  page shader splices the same GLSL (`VoxelSky::glsl_clouds`: uniforms, the base's `base_up` / `base_altitude` /
  `base_level_span` (core, `BaseSurface::glsl_render`), `cloud_hit` / `cloud_texel` / `cloud_density`) and
  its `light()` (Lambert, no specular: every material has roughness 1) cuts the sun by `cloud_shadow` where the
  fragment's ray to the sun crosses a cloud, so a shadow lies under its cloud. `VoxelSky` hands the terrain the
  uniforms through `VoxelTerrain.set_page_params(Dictionary)` (kept until the pool exists); a scene without a VoxelSky
  keeps the defaults (no texture = black = no clouds, `cloud_shadow` 0). **Surface detail**, all procedural, world-space
  value noise (the same at every LOD): `ground_detail` on the procedural ground and `PATTERN_GROUND` (a tint over ~20 m,
  patches of ~2 m, a grain of 6 cm that fades past 2 pixels per feature), `PATTERN_LEAVES` on the leaves (clumps of
  0.5 m from dark to yellowish green, 8 cm leaves, darker undersides). **Wind** (`VoxelSky.wind_sway_m`, 3 cm; 0 =
  still): the vertex stage moves leaf corners by `wind_sway` (a gust wave riding `cloud_wind` plus a finer wobble, a
  pure function of the position and TIME: shared corners agree, the mesh stays watertight) up to LOD 2, where each
  tree corner pays a palette and a material fetch, and the leaf pattern's noise drifts (a shimmer). Bark stays still:
  the pages carry no per-tree pivot to bend a trunk around. Cost unmeasured. **Shading frame** (`src/shade_frame_glsl.hpp`):
  every pattern of the position (the value noises, the bark's height, the lava and wind waves, the contour lines, the
  cloud tiles) is read at the render position plus the render origin's voxel modulo a period of the pattern
  (`shade_point`: a power of two of voxels holding at least 64 of its largest features, `shade_period_vox`) with its
  frequencies snapped to whole cycles per period (`shade_cells`, `shade_wave`; `value_noise` wraps its lattice): two
  render origins read points a whole number of periods apart, so a rebase changes nothing on screen and the floats stay
  small anywhere. The contour lines are halfway between whole multiples of 100 m (500 m on the continents) of altitude above the
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
  `tools/shot_diff.py` compares two shots of one pose (a fly far out and back with `--still=1`). Not done yet, in order: water (a sea plane with a
  depth shader), triplanar image textures per material, grass, height fog (the ground spans kilometres on the
  cascade terrains: a fixed fog height is wrong there), time of day.
