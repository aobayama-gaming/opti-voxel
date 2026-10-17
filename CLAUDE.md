# opti-voxel

Godot 4.6 GDExtension (godot-cpp, 4.4 API) for an LOD voxel terrain meshed with surface nets.
Engine binary: `~/.local/bin/godot`. Build: `scons platform=linux target=template_debug -j$(nproc)`
(installs into `project/bin/linux/`).

## Workflow (branches, worktrees, PRs)
- `main` only moves through merged PRs from `dev`. `dev` is the integration branch.
  Never push directly to either.
- One branch per feature, `feature/<name>`, started from `dev`, in its own worktree next to the
  main checkout. Open a PR into `dev`; `dev` goes into `main` through a PR as well.
  ```
  git worktree add ../opti-voxel-worktrees/<name> -b feature/<name> dev
  cd ../opti-voxel-worktrees/<name>
  git submodule update --init                               # godot-cpp starts empty
  scons platform=linux target=template_debug -j$(nproc)     # full build, a few minutes
  ~/.local/bin/godot --headless --path project --import     # once: .godot/ is gitignored
  ```
  Without the import, the extension isn't registered and `VoxelTerrain` doesn't exist. The import
  ends with a segfault in the extension (headless editor exit, pre-existing). It's harmless as
  long as `project/.godot/extension_list.cfg` exists afterwards.
- Perf PRs carry a measured before/after table and name the new bottlenecks.
  `tools/bench_ab.sh <dev checkout> <worktree> -- --mode=direct` builds both trees, interleaves
  the runs and writes the table to paste into the PR.
  Never build while a bench runs (thermals, see Gotchas).
- Commits are GPG-signed, and the user types the passphrase.
- Keep mesher and render work core-level. Order and bounds come from integer chunk ids. Sub-SDFs
  per chunk and biome-local SDF origins are upcoming features with their own design: don't
  anticipate them.
- When done: `git worktree remove ../opti-voxel-worktrees/<name>` after the PR is merged.

## Pipelines (`VoxelTerrain.pipeline_mode`)
- `1` GPU Legacy (`GPUMesher`): compute on a local RenderingDevice, full readback, CPU
  triangulation, one mesh + instance per chunk. Kept as the correctness oracle and perf baseline.
- `2` GPU Direct (`GPUDirectMesher`, default in `main_scene`): compute on the main RenderingDevice
  (render thread, `call_on_render_thread`), quads written straight into `QuadPagePool` pages, which
  are MultiMesh instances of 64-quad groups that pull their corners from a GPU-written RGBA16UI
  texture: one row per group, a header with the owning chunk id, then 8 B per corner (chunk-local
  14-bit fixed point + 22-bit octahedral normal). Layout and decoders: `src/page_format.hpp`. Only a ~2 KB per-batch directory comes back to the CPU (asynchronously) for bookkeeping.
  Shared GLSL: `src/gpu_glsl.hpp`, direct passes: `src/gpu_direct_glsl.hpp`.
  Chunks start on a 64-quad group and a texture row is one group, so a chunk is whole rows.
  `GPUDirectMesher::compact()` re-packs pages under half full GPU to GPU and rewrites the moved
  `ChunkOutput`s in `loaded_chunk`; spare free pages are freed (`QuadPagePool::trim`).

## Integer frame (no float ever holds an absolute coordinate)
- CPU: chunk ids and voxel coordinates are int64 (`ChunkId` / `VoxelPos` = `Vector3i64`,
  `src/world_coord.hpp`). Chunk ids are chunk centres in chunk-id units (half a LOD0 chunk = 14
  voxels); LOD0 ids are odd, a LOD-L id has its lowest set bit at L.
- The SDF takes a `WorldPos` (LOD0 voxel + float fraction in [0, 1)): `SDFBase::evaluate(const
  WorldPos &)`, GLSL `vec4 terrain_sdf(WorldPos p)`. The same point reached from two chunks gives a
  bit-identical `WorldPos`.
- GPU (`src/gpu_frame.hpp`): low 32 bits with wrapping arithmetic (exact for differences under 2^31),
  plus high words where absolute identity matters. `ChunkParam` UBO is 48 B (origin low/high
  words), per-chunk input is `ChunkInput` (32 B). Page headers carry the id's low words.
- Metres only come from `ChunkMath::chunk_render_center / grid_to_render / grid_to_render_lo32 /
  chunk_at_render`: render (Godot) space, relative to the world origin.
- World origin: `VoxelTerrain.world_origin_x/y/z` (LOD0 voxels, |v| <= 2^60), snapped down to the
  root chunk span (2^lod_numbers ids = 458,752 voxels with the defaults). No runtime rebasing yet.
- The current SDF (`SDFDummy`) is written in metres and goes through `SDFMetreBridge`: evaluated at
  the origin-relative position, so the terrain follows the origin. `WorldCoord::LEGACY_NUDGE_M`
  (0.123456 m) is its lattice offset (page shader: `lattice_offset_m`). Both go away with the exact
  noise.
- Morton keys are relative to the origin: batch/page locality only within ±2^20 ids of it.

## Verify changes (do this, don't eyeball)
- `tools/bench.sh [--no-build] --mode=direct|legacy --scenario=load|fly [--shot=loaded:x.png]
  [--view=0..7] [--validate=64] [--audit=1]`: builds, runs `project/debug/bench.tscn` with a fixed
  camera, writes `bench/results/<run>/stats.json` + `godot.log` + screenshots, prints a summary.
  Exits non-zero on build failure, crashes, `ERROR` lines, leaked RIDs, and (exit 4) a requested
  validation or audit that failed. `bench/results/LAST` holds the latest run dir.
  `--origin=X,Y,Z` sets the world origin (voxels).
- `--validate=N` (direct): re-meshes N loaded chunks with the legacy mesher and compares
  triangle by triangle. `ok` requires identical triangle counts and 0 *unexplained* anomalies;
  a handful of winding flips / normal outliers at corners where |SDF| ~ 0 are expected (two
  differently compiled SDF evaluations disagree on the sign of ~0).
- `--audit=1` (direct): reads every page back and classifies all drawn slots
  (live / dead_zero / stale / partial / broken). Anything but live/dead_zero is a bug.
- `--view=N` debug views (direct): 1 normals, 2 LOD, 3 page, 4 wireframe, 5 chunk, 6 group, 7 quad index.
- `res://debug/spike.tscn` (`QuadPoolSpike`): isolated test of the GPU -> page path (write, indirect
  count, kills) with CPU verification; run it after touching `QuadPagePool`.
- `tools/bench_compare.sh --runs=3 "a:--mode=legacy" "b:--mode=direct"`: interleaved runs,
  median table with spread and Δ%, written to `bench/results/<date>-compare.md`.
- `tools/frame_test.sh [--no-build]` (`FrameSpike`): GPU vs CPU check of the integer frame (WorldPos,
  anchors, pairs reaching the same voxel, continuity across low-word wraps) at far origins. Run it
  after touching `world_coord.hpp`, `gpu_frame.hpp`, `chunk_math` or the SDF plumbing. A new noise
  adds an entry (CPU fn + GLSL fn) to the probe table in `src/frame_spike.cpp`.
- `tools/origin_parity.sh [--voxel=M] -- --mode=direct --validate=64 --audit=1`: the same run at
  origins 0, 0 again (noise floor), ~1e15 m and a low-word wrap; counts must match exactly and
  screenshots stay at the noise floor. Report: `bench/results/<date>-parity.md`. Only meaningful
  while the SDF follows the origin (`SDFMetreBridge`).

## Gotchas (all hit for real)
- Laptop GPU: in the `quiet` platform profile the GPU is power-capped to ~300 MHz under load and
  every GPU number is meaningless. `bench.sh` records clocks/throttling in `stats.json["telemetry"]`;
  check `gpu_clock_mhz_median` before trusting a result.
- Locked screen (`loginctl show-session <id> -p LockedHint` = yes): KWin stops presenting the
  window, GPU Direct readbacks never arrive and the bench times out with 0 chunks loaded at a
  210 MHz GPU clock. Benches need an unlocked session.
- GPU Direct pages draw indirectly, so Godot's "primitives drawn" counts each visible page at full
  capacity (draw calls × 65,536), not real triangles. Compare culling with the bench's
  `draw_calls_view` / `draw_calls_shadow` (camera pass vs shadow pass), never with primitives.
  The bench light's shadow pass (1,000 m) also draws pages behind the camera.
- `thread_model=2` (separate render thread): Godot 4.6.3 aborts in `RenderingDevice::finalize` on
  every exit, even for an empty scene. `bench.sh` accepts that one abort if the bench finished.
- Reading render GPU time (`viewport_get_measured_render_time_*`) syncs the render thread: the
  bench only does it in the steady phase. Godot's official build only captures viewport begin/end
  timestamps, so there is no per-pass GPU breakdown.
- MultiMesh keeps ~190 B of CPU RAM per instance (interpolation arrays): never use one instance
  per quad. Pages use 64-quad groups for that reason.
- Never call `multimesh_set_buffer` / `multimesh_instance_*` / `multimesh_get_buffer` on a page: it
  creates a CPU cache that overwrites GPU-written data. `multimesh_get_*_rd_rid` from the main thread
  blocks: only call it on the render thread.
- Many MB of `texture_get_data_async` in one frame crashes 4.6.3 (download staging overflow):
  throttle readbacks (validator/auditor do 2 pages per frame).
- `RenderingServer::shader_set_code` does not run the shader preprocessor: no `#define`.
- `texture_rd_create` only wraps RD formats that have an `Image` equivalent: RG32UI / RGBA32UI fail
  ("Unsupported image format"), 16-bit integer formats (RGBA16UI) work.
- Compaction moves chunks between pages: anything that holds page ranges across frames (validator,
  auditor) must run with compaction paused (`VoxelTerrain::_process` skips it while they run), and
  `to_destroy_chunk` holds stale copies until `_destroy_chunk`, so compact after it.
- `Array` is shared by reference (Packed*Array are copy-on-write): don't clear an Array after
  binding it into a deferred callable.
- `PackedByteArray::encode_s32/u32`, `PackedInt32Array` element assignment and `Vector3i(...)`
  truncate int64 silently: ids reach the GPU only through `GPUFrame::encode_chunk_input` /
  `WorldCoord::lo32_i`.
- Godot's JSON parser reads numbers back as doubles: reports carry int64 ids as strings
  (`WorldCoord::to_string`).
- CPU mirrors of GPU float math (`ChunkMath::sample_pos`, `legacy_rel_voxels`, `legacy_metres`)
  must stay op for op identical in float32: separate statements, `Float3` (not `Vector3`, which is
  double in precision=double builds), and `precise` on the GPU side. `FrameSpike` checks bit-exactness.
- `FrameSpike` uses a local RenderingDevice: it needs a window (not `--headless`) and an unlocked session.
- Main-RD resources must be created/freed on the render thread; objects owning them delete
  themselves there (`destroy()` -> `_rt_destroy()`), since callables bound to a freed object are
  skipped.
