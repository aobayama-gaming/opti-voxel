# Terrain: presets, the table fold, the base surface

One terrain (the cascade), one instance path (its table). Read before touching `sdf_cascade`, `sdf_table`, `base_surface` or adding a terrain.

- **One terrain, one instance path.** The ground is always the cascade (`SDFCascade`, `src/sdf_cascade.hpp`: 3D
  height field, octaves k = 3 up to the micro engine's top (`SDFCascade::micro_top_k`: the first octave whose
  amplitude reaches the 3.3 km cap, k = 19 at 5 cm, at most lod_numbers + 4; the same octaves and bits at any LOD count
  past it, the relief above being the planned macro engine's; the continents preset keeps lod_numbers + 4),
  steepness 1/8, amplitudes capped at 3.3 km, datum at 300 m,
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
  the base and `ground_under`) is folded into each lattice sample
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
  configured empty gets no extra depth. `SDFCascade::ground_along` sphere-traces the bare cascade along the base's up
  (`SDFTable::ground_under`, `ground_at` from the base's tangent plane: where the scenes and biome features stand).
- Base surface (`src/base_surface.{hpp,cpp}`, design note `planet-base-curvature-2026-10-22.md`): the ground's
  reference, not a constraint (the terrain stays 3D). `BaseSurface::at(voxel, lod)` = (up n, altitude) in voxels of
  the LOD, read by `CascadeTerrain::height_sdf` (t = n - grad h, distance (altitude - h) / |t|, GLSL
  `GLSL_HEIGHT_SDF`). `FLAT` (the world today) is the saturated int64 datum difference with up (-0, 1, -0), so
  the default terrain keeps its bits (pinned in `test_sdf_cascade.cpp`): the only code that knows about y.
  `SPHERE` (`World::Settings::planet_radius_m`, `--planet_radius_km=R` in bench and player: centred on the voxel
  (0, 0, 0), its +y pole at y = R, standing for the flat datum in the default world origin request ((0, 0, 0); any other
  is absolute) and in the hosts' scenes (`World::scene_shift`); `SDFCascade::planet_base` takes the switch LOD
  `SWITCH_MARGIN` (2) over the radius form's float bound: 12 at Earth / 5 cm, 16 for Jupiter, 1 for a 2 km planet, 0
  under ~0.5 km; a planet's climate (its biomes' multiplied octaves, their curves) is capped at `PLANET_RELIEF`
  (1/32) of its radius, so a small planet stays a ball, and its octaves start at an eighth of that (`PLANET_HEADROOM`:
  a biome may multiply them by up to 8, the shipped climates go to 4.5)):
  below `switch_lod` the curvature form d(c) + n.δ + 1/2 δ^T H δ around the centre of the point's region
  (the cell of 2^(L+6) voxels holding it, from the point and the LOD only), d(c), n, H from the CPU in double
  (`make_region` takes any base's exact value, gradient and Hessian: the unit test runs a torus); from it the
  radius form |p - centre| - R in float32 (`det_rsqrt`), the centre snapped to 2^31 voxels so (p - c) >> L is exact.
  On the GPU (`src/base_regions.{hpp,cpp}`) the cascade passes upload per batch the regions its items touch,
  deduplicated, and 8 region indices per item (`CascadePasses::upload_items`, with the batch's ChunkInput bytes,
  both meshers, the point items, CascadeSpike), at bindings 10 / 11 of the level shader; the sphere's `base_at` picks
  the slot from the parity of the point's region cell on each axis: a chunk's 32-point box meets at most two cells per
  axis, whose parities differ. A point item fills its own cell's slot; the radius form needs none. CascadeSpike's set
  `4g` (LINEAR4 on an Earth-sized sphere) checks the passes and the point items bit for bit. Everything placed,
  shaded or flown follows the base's up: the scenes and biome features (`docs/trees.md`), the page shader and the sky
  (`docs/look.md`), the bench and the player (`VoxelTerrain.get_up_at`). Not yet on a sphere: the stretched octaves (a
  climate with them is refused: their base sample is at the band's insertion LOD), the Dawn viewer (flat only), the
  leaves' wind sway (world xz).
  Measured at Earth radius / 5 cm (`test_base_surface.cpp`): curvature 7e-6 cells at LODs 0-11, < 1% to LOD 16;
  radius < 0.4% from LOD 12; the forms' gap at the switch 0.35%. FrameSpike's `base_*` probes check the GLSL
  bit for bit. The render side reads the base only through `BaseSurface::glsl_render` (render-space
  `base_altitude`, `base_up`, `base_level_span`, from `BaseSurface::render`'s uniforms in double, `World::base_render`):
  the pages' heights and slopes, the sky's frame, the cloud layer and the atmosphere (`docs/look.md`). A new base kind
  adds its case to those three functions, to `render` and to `curvature_radius_m`, and every render user follows it.
  `TORUS` (`World::Settings::planet_ring_m`, `--planet_ring_km=K` with `--planet_radius_km=R`: a ring of K round the y
  axis through the voxel (0, 0, 0), a tube of R, its start over the tube's top at +x) is such a kind: one `shape()` in
  double gives every curved base's altitude, gradient and analytic Hessian (the regions, `up_dir`, `altitude`), its
  radius form is |(|v.xz| - ring, v.y)| - tube with both lengths through `det_rsqrt`, the switch LOD from the outer
  radius, the relief and the atmosphere's scale from the tube. Its `base_level_span` is sphere-traced
  (`base_traced_span`, any base whose altitude is a distance) and gives a ray's first stretch below the level only:
  across the hole the far tube's air and clouds are missing in the sky; each crossing is a relaxed trace (1.5x steps, up
  to 128) refined by 10 halvings, so a torus's atmosphere and clouds cost more than a sphere's (unmeasured). The render
  altitude is float32 from the centre (no K form), about a metre at a planet-sized ring. The ring is at least 1.5 tubes
  (`MIN_RING_TUBES`), and FrameSpike's `base_torus` probes check its radius form bit for bit. Measured (`test_base_surface.cpp`, a 2^26 ring
  and 2^24 tube): both forms under 1% of a cell to LOD 21, up within 1e-9.
