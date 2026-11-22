# Terrain: presets, the table fold, the base surface

One terrain (the cascade), one instance path (its table), one world file. Read before touching `sdf_cascade`, `sdf_table`, `base_surface` or adding a terrain.

- **A run loads a world.** Every terrain setting is data: `World::Settings::world_path` (`VoxelTerrain.world_path`, `--world=PATH`
  on bench, player, Dawn and the tools; the web's `?world=`) names a world file (`WorldFile`, format in `src/world_file.hpp`). Its
  `engine` (a path or inline: `voxel_m`, `lods`, the master variables) and `climate` (a path or inline: the fields, none for no
  fields) sit beside `planet`, `look`, `instances` (they grow over the whole ground, streamed), `species` (a path or the list),
  `features` and `feature_region` (the square they stand in until they stream: `SDFTable::set_feature_region`; a world
  whose lists name features needs one), `biomes` (a path or the list), `instance_band` (false: every instance whole at every LOD) and `edits` (a path or inline: the edits file
  below); every key is optional to the file (the cascade refuses no `biomes`), and an empty `world_path` is `WorldFile::DEFAULT_PATH`
  (`worlds/default.world.json`). A variant is another file: `adapter/godot/project/worlds/` holds the presets' and the
  tools' (`default`, `cm`, `lods25`, `biomes`, `forest5cm`, `earth`, `earth25`, `tiny`, `tiny_biomes`, `shapes`), `climate/world.json`
  the game's (`--preset=video`). `--preset=` names keep working. **The world is the file:** the ground is the biomes'
  graphs, compiled into one `Terrain::LevelProgram` (`docs/climate.md`, `docs/cascade-v2.md`), and a world names its
  ground: one without biomes is refused (`SDFCascade::attach_climate`: "a world names its ground"). The bare worlds name
  the template ground, `"biomes": "res://worlds/ground.biomes.json"` (`WorldFile::GROUND_BIOMES`: one biome, no fields;
  one file at every voxel size, its notes finer than 8 voxels playing nothing); the worlds that grow instances name
  today's species, `"species": "res://worlds/species.json"` (three trees, a boulder, a pillar; no species without the
  key). Nothing stands in for a refused part: a world whose planet, climate (or no biomes) or species is refused has no
  ground at all, never the flat datum, a template or a flat ground instead. The world file refuses a
  planet `BaseSurface::planet` refuses; `SDFCascade::get_world_error` says why the rest was (configure then builds
  no program), `World::start` doesn't start and keeps the reason (`World::get_error`, `VoxelTerrain.get_world_error`:
  the player's HUD and the biome mixer show it in red), and `ClimateReport::check_world` and `CascadeCheck` (before
  any set runs) fail on it. An instance list naming a species the world doesn't have is refused (`SDFTable::species_error`),
  and a fold text that doesn't compile fails the mesher's passes (`World::get_error` from the next frame), never the bare
  cascade drawn without the instances.
  Engine tuning and debug flags stay flags.
- **One terrain, one instance path.** The ground is always the cascade (`SDFCascade`, `src/sdf_cascade.hpp`: 3D
  height field, the program the compiler makes of the biomes; the template ground `ground.biomes.json` is octaves
  0.4 m (k = 3 at 5 cm) up to 26.2 km (k = 19 at 5 cm, 16 at 0.5 m where the three finest play nothing), the notes past
  lods + 4 playing nothing, octave k seeded make_octave(seed, k) so the same octaves and bits at any LOD count past it, steepness 1/8;
  a 2 km planet's own world (`tiny.world.json`) stops them at 3.3 km wavelengths (819 m); datum at 300 m,
  absolute position only) and instances always go through its table (`SDFTable`, below). Don't add a second way to
  make either: a new terrain is a world file, a new feature a library type. The shapes gallery is one
  (`worlds/shapes.world.json`, `res://shapes_scene.tscn` with a 60 km far plane, `--terrain=shapes` in the bench and the
  player for `--world=` of it): a biome whose graph is a constant node (value 0: the ground is the datum, 300 m, altitude
  0, off the page shader's contour lines at 50 m + k 100 m), `"instance_band": false` (nothing banded: the thin frames
  would tear) and its edits file `shapes.edits.json` placing hard-CSG cubes holed through every face, each twice the size
  and distance of the last, i.e. one per LOD, plus a die, stairs and a tunnelled cube, for hard creases (the
  composites are library types, always in the library: `swiss_cube`, `die`, `stairs`, `tunnel_cube`). The flat preset
  (`terrain_mode` 4, `SDFCascade::Preset::FLAT`, `SDFTable::Scene::SHAPES`) it replaces is gone; before it, 0 was
  `SDFDummy` (it evaluated every octave at every LOD: detail finer than the cell, aliased far away) and 3 `SDFArchetype`
  (a plain fBm ground, and instances compiled into its program). The player spawns `spawn_altitude` (200 m, the shapes
  scene's 8 m) above the ground (`--altitude=M`) and flies at its altitude per second (read every frame, the reading smoothed over 0.2 s in log space: no speed steps), and the bench camera starts 50 m above it unless
  `--cam_pos` or `--above_ground=M` (M metres above the terrain under the camera, negative below) says otherwise: the
  cascade's ground is kilometres from the datum.
  **The table** (`SDFTable`, `src/sdf_table.*`, `SDFCascade`'s base: the library, records, snapshot, GPU table,
  runtime edits (replayed from the world's `edits` at configure) and the instances' sources placed on the ground, which
  only need the base and `ground_under`) is folded into each lattice sample
  (`SDFCascade::sample_sdf`, GLSL `cascade_sample_sdf` at the level-0 store of the cascade passes): the register
  is the cascade's unit gradient and LOD-L distance times 2^L (exact), then `Archetype::fold_onto` and
  `finish_table`, **only where an instance passes its box test**, else the bare cascade bit for bit. That keeps a
  sample a pure function of the point: two chunks with different lists agree at a shared sample (unit test, 671
  pairs with different lists). `CascadePasses` compiles a second level-shader variant with the fold
  (`get_sample_fold_glsl` and the table buffers at 7 / 8, `CascadeGLSL::TABLE_*_BINDING`) when the mesher passes its
  table buffers; `record_level(..., p_fold)` uses it for level 0 of a batch with any instance listed, dev's bare
  shader otherwise (same bits over empty lists). The discoverer's corners stay bare on both back-ends
  (`SDFBase::evaluate_cull`, the GPU point items): a node near an instance is never culled (`cull_exempt`). With
  instances placed at configure (the edits file's too), the cascade skirt is the bare one plus the instances' erosion
  step (both steps meet at a seam next to a smooth-unioned rock), or, with `"instance_band": false`, no erosion step but,
  when an edit is hard CSG (`SDFTable::has_hard_edits`: k 0), `skirt_depth` cells (a hard crease differs by a cell
  between LODs: the gallery's skirts on its constant ground, whose own step is 0);
  without, the bare cascade's skirts, so a runtime edit on a cascade configured empty gets no extra depth. `SDFCascade::ground_along` sphere-traces the bare cascade along the base's up
  (`SDFTable::ground_under`, `ground_at` from the base's tangent plane: where the scenes and biome features stand).
- **The edits file** (`<name>.edits.json`, format in `src/world_file.hpp`, `WorldFile::parse_edits` / `edits_text`): the
  runtime edits a world starts with, the world file's `"edits"`. `{"version": 1, "edits": [...]}`, one entry per live edit
  in id order, each a library type by name (`DefLibrary::find`: the names are unique, the crown is `crown`), an operator
  by name (`Archetype::COMBINE_NAMES`), the anchor as three int64 strings (JSON numbers are doubles), the rotation's
  rows (absent: identity), up to 8 params and k in voxels; the floats are written as their shortest text and read back
  bit for bit. It holds the set, not a log: a move rewrites its entry, a removal drops it (the tube's per-frame moves
  never pile up). `SDFTable::build_table` replays it after `gpu.set_scene`, so the edits get ids past the scene's and fold
  after the streamed instances (`compose_list`: a subtraction carves the streamed trees), each the record
  `add_instance` makes (`SDFTable::edit_record`), the table's capacity counting them; an entry naming no type or refused
  as an edit refuses the world, its index in `World::get_error`. `World::save_edits(path)` (`VoxelTerrain.save_edits`,
  the player's F5: `user://<world>.edits.json`) writes the live edits (`SDFTable::live_edits`; records the scene placed
  are never edits); nothing loads one but a world file's key. `VoxelTerrain.restart_world` keeps the live edits
  (`World::set_start_edits`), the file's only after a refused world. Dawn reads edits files but writes none (no file
  writer). Not yet: removing a streamed tree (it needs an `InstanceStream` invalidation).
- Base surface (`src/base_surface.{hpp,cpp}`, design note `planet-base-curvature-2026-10-22.md`): the ground's
  reference, not a constraint (the terrain stays 3D). `BaseSurface::at(voxel, lod)` = (up n, altitude) in voxels of
  the LOD, read by `CascadeTerrain::height_sdf` (t = n - grad h, distance (altitude - h) / |t|, GLSL
  `GLSL_HEIGHT_SDF`). `FLAT` (the world today) is the saturated int64 datum difference with up (-0, 1, -0), so
  the default terrain keeps its bits (pinned in `test_sdf_cascade.cpp`): the only code that knows about y.
  `SPHERE` (the world file's `planet.radius_m`: centred on the voxel
  (0, 0, 0), its +y pole at y = R, standing for the flat datum in the default world origin request ((0, 0, 0); any other
  is absolute) and in the hosts' scenes (`World::scene_shift`); `BaseSurface::planet` takes the switch LOD
  `BaseSurface::SWITCH_MARGIN` (2) over the radius form's float bound: 12 at Earth / 5 cm, 16 for Jupiter, 1 for a 2 km planet, 0
  under ~0.5 km; a note plays at its written amplitude on any base, and the compiler holds the program's output bound
  to the base's honest limits only (`Climate::Ground`): its reach within `BaseSurface::ALT_LIMIT` either side (a carve,
  reading `up`, its top only), and on a curved base its low side above the centre (minus the radius, the tube's on a
  torus): a ground that would cross the centre is refused, the template's 3.3 km octaves on a 2 km planet included, so
  a small planet's world brings its own ground, as `tiny.world.json` does):
  below `switch_lod` the curvature form d(c) + n.δ + 1/2 δ^T H δ around the centre of the point's region
  (the cell of 2^(L+6) voxels holding it, from the point and the LOD only), d(c), n, H from the CPU in double
  (`make_region` takes any base's exact value, gradient and Hessian: the unit test runs a torus); from it the
  radius form |p - centre| - R in float32 (`det_rsqrt`), the centre snapped to 2^31 voxels so (p - c) >> L is exact.
  On the GPU (`src/base_regions.{hpp,cpp}`) the cascade passes upload per batch the regions its items touch,
  deduplicated, and 8 region indices per item (`CascadePasses::upload_items`, with the batch's ChunkInput bytes,
  the mesher, the point items, CascadeSpike), at bindings 10 / 11 of the level shader; the sphere's `base_at` picks
  the slot from the parity of the point's region cell on each axis: a chunk's 32-point box meets at most two cells per
  axis, whose parities differ. A point item fills its own cell's slot; the radius form needs none. CascadeSpike's set
  `4g` (an Earth-sized sphere) checks the passes and the point items bit for bit. Everything placed,
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
  `TORUS` (`planet.ring_m` with `planet.radius_m` R: a ring of that radius round the y
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
