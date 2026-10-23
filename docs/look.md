# Look (sky, fog, clouds, surface detail)

The shared environment and the page shader's surface detail.

- **Look (sky, fog, clouds, surface detail).** The scenes (`main_scene*.tscn`, `debug/bench.tscn`) share one environment:
  a sun at 28 degrees (warm, `light_angular_distance` 0.53 so the sky draws its disk), exponential fog with aerial
  perspective (`fog_aerial_perspective` 0.8: the fog takes the sky's colour in that direction, `fog_density` 0.00025: a
  quarter at 1 km), SSAO / SSIL, the grading LUT. `VoxelSky` (`src/voxel_sky.*`, a sibling of the terrain and the
  WorldEnvironment) installs the sky shader at ready: Godot's PhysicalSkyMaterial (MIT) with a **cloud layer** over it, a
  plane at `cloud_height_m` reading `cloud_texture` (`project/clouds.png`, `tools/make_clouds.py`: tileable spectral
  noise, rank-normalised so `cloud_cover` is the fraction covered) once per `cloud_scale_m`, carried by `cloud_wind`. The
  page shader splices the same GLSL (`VoxelSky::glsl_clouds`: uniforms, `cloud_hit` / `cloud_uv` / `cloud_density`) and
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
  the pages carry no per-tree pivot to bend a trunk around. Cost unmeasured. Not done yet, in order: water (a sea plane with a
  depth shader), triplanar image textures per material, grass, height fog (the ground spans kilometres on the
  cascade terrains: a fixed fog height is wrong there), time of day.
