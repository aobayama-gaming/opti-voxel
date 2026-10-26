# Terrain: presets, the table fold, the base surface

One terrain (the cascade), one instance path (its table). Read before touching `sdf_cascade`, `sdf_table`, `base_surface` or adding a terrain.

- **One terrain, one instance path.** The ground is always the cascade (`SDFCascade`, `src/sdf_cascade.hpp`: 3D
  height field, octaves k = 3 .. lod_numbers + 4, steepness 1/8, amplitudes capped at 3.3 km, datum at 300 m,
  absolute position only) and instances always go through its table (`SDFTable`, below). Don't add a second way to
  make either: a new terrain is a cascade preset (parameters), a new feature a library type. `VoxelTerrain.terrain_mode`
  picks a preset: 1 the default, 2 continents, 4 flat (`SDFCascade::Preset::FLAT`: no band, the ground at `FLAT_GROUND_M` = 250 m, off
  the page shader's 100 m contours) under the shapes gallery (`SDFTable::Scene::SHAPES`, `--terrain=shapes`,
  `res://shapes_scene.tscn` with a 60 km far plane: hard-CSG cubes holed through every face, each twice the size and
  distance of the last, i.e. one per LOD, plus a die, stairs and a tunnelled cube, for dual contouring's creases; the
  composites are library types: `swiss_cube`, `die`, `stairs`, `tunnel_cube`; nothing in that scene is banded, its thin frames would tear). 0 was `SDFDummy` (it evaluated every
  octave at every LOD: detail finer than the cell, aliased far away) and 3 `SDFArchetype` (a plain fBm ground, and
  instances compiled into its program), both removed. `--terrain=cascade|continents|shapes` in `bench.gd` and the
  main scene (`godot --path adapter/godot/project -- --lods=25`). Outside the shapes gallery the player spawns 200 m above the
  ground (`--altitude=M`) and flies at its altitude per second (read every frame, the reading smoothed over 0.2 s in log space: no speed steps), and the bench camera starts 50 m above it unless
  `--cam_pos` or `--above_ground=M` (M metres above the terrain under the camera, negative below) says otherwise: the
  cascade's ground is kilometres from the datum.
  **The table** (`SDFTable`, `src/sdf_table.*`, `SDFCascade`'s base: the library, records, snapshot, GPU table,
  runtime edits and the scenes placed on the ground, i.e. the gallery, `--instances` and `--trees`, which only need
  `ground_y`) is folded into each lattice sample
  (`SDFCascade::sample_sdf`, GLSL `cascade_sample_sdf` at the level-0 store of the cascade passes): the register
  is the cascade's unit gradient and LOD-L distance times 2^L (exact), then `Archetype::fold_onto` and
  `finish_table`, **only where an instance passes its box test**, else the bare cascade bit for bit. That keeps a
  sample a pure function of the point: two chunks with different lists agree at a shared sample (unit test, 671
  pairs with different lists). `CascadePasses` compiles a second level-shader variant with the fold
  (`get_sample_fold_glsl` and the table buffers at 7 / 8, `CascadeGLSL::TABLE_*_BINDING`) when the mesher passes its
  table buffers; `record_level(..., p_fold)` uses it for level 0 of a batch with any instance listed, dev's bare
  shader otherwise (same bits over empty lists). The discoverer's corners stay bare on both back-ends
  (`SDFBase::evaluate_cull`, the GPU point items): a node near an instance is never culled (`cull_exempt`). With
  instances placed at configure, the cascade skirt is the bare one plus the instances' erosion step (both steps meet
  at a seam next to a smooth-unioned rock); without, the bare cascade's skirts, so a runtime edit on a cascade
  configured empty gets no extra depth. `SDFCascade::ground_y` sphere-traces the bare cascade.
- Base surface (`src/base_surface.{hpp,cpp}`, design note `planet-base-curvature-2026-10-22.md`): the ground's
  reference, not a constraint (the terrain stays 3D). `BaseSurface::at(voxel, lod)` = (up n, altitude) in voxels of
  the LOD, read by `CascadeTerrain::height_sdf` (t = n - grad h, distance (altitude - h) / |t|, GLSL
  `GLSL_HEIGHT_SDF`). `FLAT` (the world today) is the saturated int64 datum difference with up (-0, 1, -0), so
  the default terrain keeps its bits (pinned in `test_sdf_cascade.cpp`): the only code that knows about y.
  `SPHERE` (not wired into a world yet: regions per chunk, discoverer, camera altitude come with the planet PR):
  below `switch_lod` (12) the curvature form d(c) + n.δ + 1/2 δ^T H δ around the centre of the point's region
  (the cell of 2^(L+6) voxels holding it, from the point and the LOD only), d(c), n, H from the CPU in double
  (`make_region` takes any base's exact value, gradient and Hessian: the unit test runs a torus); from it the
  radius form |p - centre| - R in float32 (`det_rsqrt`), the centre snapped to 2^31 voxels so (p - c) >> L is exact.
  Measured at Earth radius / 5 cm (`test_base_surface.cpp`): curvature 7e-6 cells at LODs 0-11, < 1% to LOD 16;
  radius < 0.4% from LOD 12; the forms' gap at the switch 0.35%. FrameSpike's `base_*` probes check the GLSL
  bit for bit. The page shader's slope colours read `base_up` (a uniform, +y).
