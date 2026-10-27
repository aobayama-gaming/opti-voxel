# Integer frame

No float ever holds an absolute coordinate. Read before touching `world_coord`, `gpu_frame`, `chunk_math`, `exact_noise` or anything that stores a position.

- CPU: chunk ids and voxel coordinates are int64 (`ChunkId` / `VoxelPos` = `Vector3i64`,
  `src/world_coord.hpp`). Chunk ids are chunk centres in chunk-id units (half a LOD0 chunk = 14
  voxels); LOD0 ids are odd, a LOD-L id has its lowest set bit at L.
- The SDF takes a `WorldPos` (LOD0 voxel + float fraction in [0, 1)) and the chunk's LOD:
  `SDFBase::evaluate(const WorldPos &, int lod)`, GLSL `cascade_sample_sdf(c0, WorldPos p, int lod, chunk)` (the
  meshers pass `ChunkFrame::lod`, the discoverer and validator `ChunkMath::get_lod`). The same point
  reached from two chunks gives a bit-identical `WorldPos`.
- GPU (`src/gpu_frame.hpp`): low 32 bits with wrapping arithmetic (exact for differences under 2^31),
  plus high words where absolute identity matters. `ChunkParam` UBO is 48 B (origin low/high
  words), per-chunk input is `ChunkInput` (32 B). Page headers carry the id's low words.
- Metres only come from `ChunkMath::chunk_render_center / grid_to_render / grid_to_render_lo32 /
  chunk_at_render`: render (Godot) space, relative to the **render origin** (`ChunkParam::origin_chunk`).
- World origin: `VoxelTerrain.world_origin_x/y/z` (LOD0 voxels, |v| <= 2^60), snapped down to the
  root chunk span (2^lod_numbers ids = 458,752 voxels with the defaults): where the render origin starts.
- **Render-origin rebasing** (`World::recenter` / `rebase`, `World::Settings::rebase_distance_m`,
  `VoxelTerrain.rebase_distance_m`, default 1024 m, 0 = a fixed origin; `bench.gd --rebase_m=M`): once the camera
  is farther than that from the render origin on any axis, the origin moves under it
  (`ChunkMath::recentered_origin`: the lower corner of the camera's LOD0 chunk; any even id is an origin,
  `is_render_origin`) and the host subtracts `ChunkMath::origin_shift_m` from every render position it holds, in the
  same frame: VoxelTerrain moves its Node3D siblings and emits `origin_shifted(shift)` for the positions scripts hold
  (bench.gd's paths); the Dawn viewer moves its camera and carries FSR's last projection over
  (`DawnUpscaler::shift_history`; Godot's TAA / FSR see a one-frame camera jump). Nothing is re-meshed: page rows are
  chunk-local under absolute ids, so the direct mesher re-sends the page decode (`origin_chunk_lo`) and recomputes
  every page's culling box (in-flight batches' pages too); the legacy meshes move (`MeshHost::shift_meshes`, under a
  lock the legacy worker holds while it creates meshes). Only render-space code reads the origin: the discoverer takes
  the camera's absolute chunk (`set_camera(ChunkId)`), digs and instances are absolute voxels. A rebase waits while the
  validator, the page auditor or the discovery audit run (they hold render positions). Render floats then stay within
  the rebase distance plus the view: the camera at 1,000 km jitters no more than at spawn. Scripts read the render
  origin with `get_render_origin_x/y/z` (int64 voxels) and `get_render_offset_m()` (metres from the world origin,
  float: player.gd's marks add it to `cam_pos`); `get_debug_stats()["world_origin"]` has `render_voxel`,
  `render_chunk` and `rebases`. The shading that reads positions goes through the shading frame
  (`shade_frame_glsl.hpp`, `docs/look.md`), so the ground looks the same after a rebase. The `ChunkParam` UBO's
  origin words are the origin at upload: no GPU pass reads them.
- LOD range: `VoxelTerrain.lod_numbers` (default 15, at most 31: the GPU reads a LOD from the id's low
  32 bits). `--lods=N` in `bench.gd` and in the main scene (`godot --path adapter/godot/project -- --lods=25`).
  At every LOD count the camera's clip planes follow
  its altitude (`Presets.clip_planes`, every frame in the player and the bench; |SDF| at the camera,
  `VoxelTerrain.get_sdf_at`): near = altitude / 1000 (at least the scene's), far = min(near * 2^21,
  `VoxelTerrain.get_load_radius()`), so the ground stays in view from 13 km up. Past far/near ~2^24 Godot's float
  projection loses its far plane and the light culler ERRORs every frame. The validator doesn't compare
  the normals of sliver triangles (all edges within 2 * tol): around a lone near-zero corner a whole
  closed sheet fits inside the tolerance and the vertex pairing is arbitrary (frequent at far LODs).
- Exact noise (`src/exact_noise.hpp`, CPU + `GLSL_EXACT_NOISE` in `exact_noise_glsl.hpp`): `hash_cell` = PCG3D of the cell's low
  words salted by a PCG3D of its high words and the seed (cells 2^32 apart don't collide);
  `perlin3` / `perlin2` = one octave of gradient noise with wavelength exactly 2^k LOD0 voxels
  (`cell = (voxel + offset) >> k`, the in-cell position as 24-bit fixed point, quintic fade, analytic
  derivatives); `make_fbm` / `fbm` / `glsl_fbm` = octave sets with per-octave seed and integer offset
  (a quarter to three quarters of a cell, so the nested lattices' zeros never line up). At LOD L the
  fBm gradient is per voxel of that LOD (2^L LOD0 voxels): octave factor `amp * 2^(L-k)`
  (`gradient_factor`), kept in [2^-16, 2^40] at every LOD the octave is evaluated at
  (`gradient_range_error`, checked when an archetype program is validated). Values are unscaled.
- Every terrain is a function of the absolute position: the same world point gives the same value from
  any origin. Heights and depths are in LOD0 voxels, gradients and distances in voxels of the chunk's
  LOD, and they return metres. Parameters are metres rounded to 2^k voxels in `configure()`, so another
  voxel size is another world. Flat world: the datum (`BaseSurface::DATUM_M`) is absolute y ~ 300 m, so far
  origins only show terrain with y = 0.
- Morton keys are relative to the render origin: batch/page locality only within ±2^20 ids of it (it follows the
  camera when rebasing is on).
- `GLSL_EXACT_NOISE` is spliced from fragments it shares with
  `GLSL_PERLIN3_CROSS` (FNV unit test on its text). `OPTI_VOXEL_SHADER_DUMP=<dir>` writes every composed
  shader to `<dir>/<name>_<hash>.glsl` to compare two builds.
