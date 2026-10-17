# opti-voxel

Godot 4.6 GDExtension (godot-cpp, 4.4 API) for an LOD voxel terrain meshed with surface nets.
Engine binary: `~/.local/bin/godot`. Build: `scons platform=linux target=template_debug -j$(nproc)`
(installs into `project/bin/linux/`). `src/` builds with `-Wall -Wextra -Werror`: `werror=no` only to
iterate past a warning, never to commit one.

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
  tools/setup_hooks.sh                                      # once per clone: format check on commit
  ```
  Without the import, the extension isn't registered and `VoxelTerrain` doesn't exist. The import
  ends with a segfault in the extension (headless editor exit, pre-existing). It's harmless as
  long as `project/.godot/extension_list.cfg` exists afterwards.
- When a feature is done and its checks pass, commit, push and open the PR into `dev` without waiting
  to be asked. Don't merge it: the user reviews and merges.
- Perf PRs carry a measured before/after table and name the new bottlenecks.
  `tools/bench_ab.sh <dev checkout> <worktree> -- --mode=direct` builds both trees, interleaves
  the runs (2 per side by default, both through the worktree's `bench.sh`) and writes the table to paste
  into the PR. Comparing a tree against itself still shows load-time deltas of ~5%: that's the noise.
  Never build while a bench runs (thermals, see Gotchas).
- One GPU, several agents: other sessions work in other worktrees and run their own builds, benches
  and check.sh on the same laptop. Perf numbers are only valid when nothing else runs. Before a perf
  run (`bench_ab.sh`, `bench_compare.sh`, `--cool=1`), check `ps -eo pid,etime,args | grep -E
  "godot|scons"` for processes outside your worktree; if any, wait or ask the user. Correctness checks
  are fine in parallel, perf A/Bs never are (an A/B swung from +175% to -29% on the same metric
  because another session's tests overlapped).
- Commits are GPG-signed, and the user types the passphrase.
- Keep mesher and render work core-level. Order and bounds come from integer chunk ids. Sub-SDFs
  per chunk and biome-local SDF origins are upcoming features with their own design: don't
  anticipate them.
- When done: `git worktree remove ../opti-voxel-worktrees/<name>` after the PR is merged.

## Code rules (keep it from turning into spaghetti)
Most of this code is written with AI agents. These rules are what keeps it reviewable; they apply
to every change, and a reviewer may reject a PR on any of them.
- One source of truth for every constant and layout. Page layout lives in `page_format.hpp`, the
  integer frame in `world_coord.hpp` / `gpu_frame.hpp`. Extend those; never retype a literal (64,
  14 bits, byte offsets, binding numbers) on the C++, GLSL or shader side. GLSL gets C++ constants
  by substitution (`GPUFrame::glsl_functions()` shows how; no `#define` in shaders that go through
  `shader_set_code`).
- Two copies that must stay in sync are one function. Before pasting a block, extract it (the two
  `get_max_lod` copies had already drifted). Exception: the legacy mesher's GLSL stays separate
  from GPU Direct's on purpose, it is the oracle.
- No new `if (direct_mesher) ... else gpu_mesher` branch in `VoxelTerrain`: a mesher interface is
  planned; until then, put new mesher behaviour behind the meshers' common methods.
- No commented-out code (git has it) and no debug prints left behind: use `VLOG`. A TODO says why
  and what would close it.
- Functions over ~80 lines get split, files over ~800 lines need a reason. New debug tooling goes
  in its own class (like `MeshValidator`, `PageAuditor`), not into `VoxelTerrain`.
- Logic in a pure module (`world_coord`, `chunk_math`, `page_format`, `lod_structures`,
  `chunk_memory`, `cascade`, anything without a GPU) ships with doctest cases in `src/tests/`. A bug fix ships
  with the test that would have caught it.
- `src/` is clang-formatted (`tools/format.sh`; a Claude Code hook in `.claude/settings.json`
  formats each file as it is edited, the pre-commit hook and CI refuse unformatted code) and builds
  warning-free.
- Before opening a PR: run `/simplify`, then `/code-review` on the branch diff, and fix what holds.
  Keep a PR reviewable: split anything over ~1,000 hand-written lines.

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
- The SDF takes a `WorldPos` (LOD0 voxel + float fraction in [0, 1)) and the chunk's LOD:
  `SDFBase::evaluate(const WorldPos &, int lod)`, GLSL `vec4 terrain_sdf(WorldPos p, int lod)` (the
  meshers pass `ChunkFrame::lod`, the discoverer and validator `ChunkMath::get_lod`). The same point
  reached from two chunks gives a bit-identical `WorldPos`.
- GPU (`src/gpu_frame.hpp`): low 32 bits with wrapping arithmetic (exact for differences under 2^31),
  plus high words where absolute identity matters. `ChunkParam` UBO is 48 B (origin low/high
  words), per-chunk input is `ChunkInput` (32 B). Page headers carry the id's low words.
- Metres only come from `ChunkMath::chunk_render_center / grid_to_render / grid_to_render_lo32 /
  chunk_at_render`: render (Godot) space, relative to the world origin.
- World origin: `VoxelTerrain.world_origin_x/y/z` (LOD0 voxels, |v| <= 2^60), snapped down to the
  root chunk span (2^lod_numbers ids = 458,752 voxels with the defaults). No runtime rebasing yet.
- Exact noise (`src/exact_noise.hpp`, CPU + `GLSL_EXACT_NOISE`): `hash_cell` = PCG3D of the cell's low
  words salted by a PCG3D of its high words and the seed (cells 2^32 apart don't collide);
  `perlin3` / `perlin2` = one octave of gradient noise with wavelength exactly 2^k LOD0 voxels
  (`cell = (voxel + offset) >> k`, the in-cell position as 24-bit fixed point, quintic fade, analytic
  derivatives); `make_fbm` / `fbm` / `glsl_fbm` = octave sets with per-octave seed and integer offset
  (a quarter to three quarters of a cell, so the nested lattices' zeros never line up). At LOD L the
  fBm gradient is per voxel of that LOD (2^L LOD0 voxels): octave factor `amp * 2^(L-k)`
  (`gradient_factor`), kept in [2^-16, 2^40] at every LOD the octave is evaluated at
  (`gradient_range_error`, checked in `SDFDummy::configure`). Values are unscaled.
- The terrain (`SDFDummy`) is a function of the absolute position: the same world point gives the same
  value from any origin, and `SDFDummy::sdf()` is bit-identical to GLSL `terrain_sdf` (gradient
  included). Heights and depths are in LOD0 voxels, gradients and distances in voxels of the chunk's
  LOD, and it returns metres; every LOD factor is an exact power of two, so the metres and the
  normalised gradient are bit-identical at every LOD (unit-tested). Parameters are metres rounded to
  2^k voxels in `configure()`, so another voxel size is another world. Flat world: the datum is
  absolute y ~ 300 m, so far origins only show terrain with y = 0. Debug shapes (test rig, zero pillar) are placed
  relative to the world origin (exact `sat_diff` to it). The 3D caves are two thirds of the SDF's
  GPU cost: they are only evaluated where the band lets them reach the smooth max (bit-identical to
  evaluating them everywhere, `cave_skip`, unit-tested).
- Morton keys are relative to the origin: batch/page locality only within ±2^20 ids of it.
- Hermite cascade (`src/cascade.hpp`, CPU reference only, not used by the meshers yet; the GPU pass must
  match it bit for bit): each noise band (octave k) is inserted once on the lattice Λ_{k-3} (multiples of
  2^(k-3) voxels: 8 samples per wavelength), then carried down by tricubic Hermite subdivision with 8
  channels (value, 3 slopes, 3 cross slopes, d3/dxdydz from `perlin3_cross`, in units of the lattice
  spacing). A LOD-L chunk stops at Λ_L, so `V_m(p)` is a pure function of the point: every LOD gets the
  same bits where they share points. Fixed order x, y, z; only + - * by 0.5, 0.125, 0.75 (no exact sums:
  determinism comes from the fixed expressions); after each level |c| < 2^-64 is flushed to 0 (without it a
  band carried far down makes denormals). `Cascade::point` = the same run on a one-point box. 4
  samples per wavelength (worst case 2.2% value error) and the 4-channel variant (`LINEAR4`, twice the slope
  error) were measured and rejected: `tools/cascade_report.sh`. Test terrain: `cascade_terrain.hpp`.

## Verify changes (do this, don't eyeball)
- `tools/unit_test.sh [--no-build] [filter]`: the C++ unit tests (`src/tests/`, doctest) inside
  headless Godot, a few seconds, no GPU. `filter` is a doctest wildcard on test names
  (`'ChunkMath*'`). Tests are compiled into non-release builds only (`OPTI_VOXEL_TESTS`).
- `tools/format.sh [--check] [files]`: clang-format 23.1.3 through `uvx` on `src/`.
- `tools/cascade_report.sh [--no-build] [--scale=N]` (`CascadeReport`, headless, CPU only, ~40 s at
  `--scale=8`): fidelity of a cascaded octave vs analytic perlin3 (4 / 8 samples per wavelength, 8 / 4
  channels), points and CPU ms per chunk per LOD on an Earth, PNG slices (analytic | cascaded | error).
- CI (`.github/workflows/ci.yml`) runs format, the `-Werror` build and the unit tests on every PR.
  Everything that needs a GPU stays local.
- `tools/check.sh [--no-build] [--spike] [--discovery] [--keep-going]`: the correctness gate, ~2 min.
  One build, the unit tests, then a direct load run with `--validate=64 --audit=1 --discovery_audit=1`,
  `origin_parity.sh`, `frame_test.sh`, the QuadPagePool spike (with `--spike`, or automatically when
  `src/quad_page_pool*` differs from `dev`) and `discovery_audit.sh` (with `--discovery`, or
  automatically when `src/chunk_discoverer*`, `src/sdf*` or `src/chunk_math*` differ from `dev`, ~2 min)
  and `zero_test.sh`.
  Prints PASS/FAIL per step. Perf numbers come from `bench_ab.sh` instead.
- `tools/bench.sh [--no-build] --mode=direct|legacy --scenario=load|fly [--shot=loaded:x.png]
  [--view=0..7] [--validate=64] [--audit=1] [--cool=1]`: builds, runs `project/debug/bench.tscn` with a fixed
  camera, writes `bench/results/<run>/stats.json` + `godot.log` + screenshots, prints a summary.
  Exits non-zero on build failure, crashes, `ERROR` lines, leaked RIDs, and (exit 4) a requested
  validation or audit that failed. `bench/results/LAST` holds the latest run dir.
  `--origin=X,Y,Z` sets the world origin (voxels). `--cool=1` waits for the GPU to cool first (perf
  runs only; `bench_ab.sh` / `bench_compare.sh` pass it): until 65 C, a plateau, or 30 s at most.
  `BENCH_ROOT=<checkout>` benches another tree with this harness.
- `--validate=N` (direct): re-meshes N loaded chunks with the legacy mesher and compares
  triangle by triangle. `ok` requires identical triangle counts and 0 *unexplained* anomalies.
  The terrain SDF is bit-exact, so both meshers see the same signs: winding flips at |SDF| ~ 0 only
  come from non-exact debug shapes (test rig). Vertex positions differ by up to ~0.02 cells between
  the two meshers, so a quad smaller than the 0.02-cell tolerance can't be matched one to one; a
  leftover sliver near a same-size triangle of the other side counts as `ambiguous_slivers`, not an anomaly.
- `--audit=1` (direct): reads every page back and classifies all drawn slots
  (live / dead_zero / stale / partial / broken). Anything but live/dead_zero is a bug. Also writes a
  per-chunk mesh digest (quad count + order-independent hash of the chunk-local corners) to
  `<run>/chunks.json`.
- `--discovery_audit=1` (direct): rebuilds the current scan without culling (`--discovery_cull=0`) and
  counts the quads of every chunk the cull dropped on the GPU (passes sdf/edges/vertices, count only).
  A dropped chunk with quads is a miss, i.e. a hole; `ok` needs 0 misses and a max LOD jump of 1
  between face neighbours. `bench.sh` also fails a load-complete run whose loaded set != scanned set
  (`chunks.loaded_not_scanned` / `scanned_not_loaded`).
  `tools/discovery_audit.sh [--only=a,b] [-- bench args]` runs it over 9 poses/settings (default,
  lod_error 4, underground, east, high, deep, far origin, 5 cm voxels, fly) and writes
  `bench/results/<date>-discovery.md`. Run it after touching the discoverer or the SDF.
- Discoverer: the cull verdict of a node only depends on its id and the (fixed) SDF, so it is cached
  across scans (`discovery_cache`, default on); `--discovery_verify=1` recomputes every hit and
  ERRORs on a mismatch. `--cam_pos=X,Y,Z --pitch=DEG --yaw=DEG` place the bench camera;
  `--camera_voxel=X,Y,Z` makes the scene (camera, lights, cam_pos) relative to that absolute voxel
  instead of the world origin.
- `tools/zero_test.sh`: the debug zero pillar (`--zero_pillar=1|2`, `SDFDummy`) has its flat top
  exactly on a lattice row (SDF exactly 0.0 at LODs 0-6); run 2 moves it up 1 cm. Same triangle count
  and the same picture are required: a sample of exactly 0.0 must neither lose its quad nor flip it.
- Hole hunting: run the main scene with `godot --path project -- --marks=DIR`; H saves the camera
  pose, a screenshot, the chunk counts and a discovery audit to `DIR/marks.jsonl` (closing the window
  saves a last pose). Replay a mark with `bench.sh --cam_pos=... --pitch=... --yaw=... --lod_error=4`.
- `--view=N` debug views (direct): 1 normals, 2 LOD, 3 page, 4 wireframe, 5 chunk, 6 group, 7 quad index.
- `res://debug/spike.tscn` (`QuadPoolSpike`): isolated test of the GPU -> page path (write, indirect
  count, kills) with CPU verification; run it after touching `QuadPagePool`.
- `tools/bench_compare.sh [--runs=2] "a:--mode=legacy" "b:--mode=direct"`: interleaved runs,
  median table with spread and Δ%, written to `bench/results/<date>-compare.md`.
- `tools/frame_test.sh [--no-build] [--voxel=M]` (`FrameSpike`): GPU vs CPU check of the integer
  frame and the exact noise at 5 far origins: WorldPos and anchors, then every probe of the probe
  table bit for bit (hash, octave decomposition at k 0..62, perlin3/2, the terrain's fBms and
  `terrain_sdf` at LODs 0, 1, 5, 14, 24), pairs reaching the same voxel, absolute points identical
  at every origin, and continuity along lines across voxel and noise-cell low-word wraps. Run it after touching
  `world_coord.hpp`, `gpu_frame.hpp`, `chunk_math`, `exact_noise.hpp` or the SDF. A new noise or SDF
  term adds an entry (CPU fn + GLSL fn) to `make_probes()` in `src/frame_spike.cpp`.
- `tools/origin_parity.sh [--voxel=M] -- --mode=direct --validate=64`: the same world point seen
  from different origins (`--camera_voxel`): runs zero, zero2 (determinism), then pairs one root span
  apart near 0, at ~1e15 m and at 2^48 (voxel and noise-cell low-word wraps). Each pair must load the
  same non-empty chunks with identical mesh digests (`chunks.json`); pair screenshots are informational
  (a camera 229 km from its origin jitters at the cm level). Report: `bench/results/<date>-parity.md`.

## Gotchas (all hit for real)
- Laptop GPU: in the `quiet` platform profile the GPU is power-capped to ~300 MHz under load and
  every GPU number is meaningless. `bench.sh` records clocks/throttling in `stats.json["telemetry"]`;
  check `gpu_clock_mhz_median` before trusting a result.
- Right after a run the GPU sits at 70-87 C and doesn't get back to 65 C within minutes. The old
  cooldown always waited out its full 120 s, about 95% of the time a test run took. Correctness runs don't cool
  down; perf runs settle at most 30 s and record `start_temp_c` / `cooldown_s` in the telemetry.
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
- An interrupted scons (session moved, killed shell) can leave 0-byte `.os` files that scons later
  treats as up to date: the `.so` links but fails to load (`undefined symbol: _ZTV15GPUDirectMesher`)
  and the bench loops on script errors. `find src -name "*.os" -size 0 -delete`, then rebuild.
- SDF samples of exactly 0.0 happen (float heights hit the 0.5 m grid ~1 in 16k vertical crossings).
  Every pass uses the same side rule, solid = `!(w > 0)`, and a crossing's `t` is clamped above 0
  because `t > 0` is what marks a crossing downstream. Mixing `< 0` and `> 0`, or letting t reach 0,
  drops or flips quads: small diamond/hexagon holes, in both pipelines (the validator can't see it).
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
- CPU mirrors of GPU float math (`ChunkMath::sample_pos`, `exact_noise.hpp`, `SDFDummy::sdf`) must
  stay op for op identical in float32: same expression order, `Float3` (not `Vector3`, which is
  double in precision=double builds), `precise` on every GPU float, and `-ffp-contract=off` (in
  `SConstruct`; aarch64 or `-march=native` would fuse FMAs otherwise). Only correctly rounded ops:
  + - *, compares, min/max/clamp, floor, int <-> float below 2^24. No GLSL `mix` / `dot` / `length` /
  `normalize` / `sqrt` / division (implementation-defined precision): write dots and lerps out, use
  `det_rsqrt` (integer seed + Newton, mul/add only). Float constants go to the GLSL as bits
  (`ExactNoise::glsl_float`). No denormals: the GPU may flush them, the CPU keeps them (24-bit
  fixed-point cell positions, fades flushed below 2^-24, amplitudes and per-LOD gradient factors in
  [2^-16, 2^40]).
  `FrameSpike` checks bit-exactness.
- A distance estimate c / |grad c| of a noise field badly underestimates far from its zero set; under a
  `max()` it can win and make deep rock read |SDF| < one cell, and skirts then build walls kilometres
  underground (the cave field is capped by the distance to its depth band for that reason).
- GPU shader cost: a `const` array indexed by a per-thread value (a gradient table indexed by a hash)
  serializes on NVIDIA; it cost 15% of the SDF pass, selects on the hash bits are free. Integer
  multiplies (PCG3D) run at half the float rate; `precise` costs nothing measurable. To find where a
  shader's time goes, swap variants of the stitched GLSL and compare `gpu_pass_sdf` per batch.
- `FrameSpike` uses a local RenderingDevice: it needs a window (not `--headless`) and an unlocked session.
- Main-RD resources must be created/freed on the render thread; objects owning them delete
  themselves there (`destroy()` -> `_rt_destroy()`), after the render-thread work already queued
  for them (FIFO).
- Async readback callbacks (`texture_get_data_async`, `buffer_get_data_async`) are NOT skipped when
  their object is freed: they arrive frames later and run on freed memory (segfault in
  `MeshValidator::_rt_on_page` when quitting mid-validation). An object with readbacks in flight
  counts them and deletes itself when the last one lands (`MeshValidator` / `PageAuditor::destroy`).
  `GPUDirectMesher::_rt_destroy` does not do this yet.
