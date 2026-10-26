# Verifying changes

Every check tool in detail. The short list of what to run when is in `CLAUDE.md`.

- `tools/unit_test.sh [--no-build] [filter]`: the C++ unit tests (`src/tests/`, doctest) inside
  headless Godot, a few seconds, no GPU. `filter` is a doctest wildcard on test names
  (`'ChunkMath*'`). Tests are compiled into non-release builds only (`OPTI_VOXEL_TESTS`).
- `tools/format.sh [--check] [files]`: clang-format 23.1.3 through `uvx` on `src/`.
- `tools/cascade_report.sh [--no-build] [--scale=N]` (`CascadeReport`, headless, CPU only, ~40 s at
  `--scale=8`): fidelity of a cascaded octave vs analytic perlin3 (4 / 8 samples per wavelength, 8 / 4
  channels), points and CPU ms per chunk per LOD on an Earth, PNG slices (analytic | cascaded | error).
- `tools/climate_report.sh [--no-build] [--climate=PATH] [--voxel=M] [--lods=N] [--extent_km=KM] [--size=N] [--surface=0|1] [--lattice=M]`
  (`ClimateReport`, headless, CPU only, seconds): a climate table sampled on one lattice over the base's tangent plane, each point traced down to its ground
  surface along the base's up (`--surface=0`: the plane itself, kilometres off the ground at 25 LODs): per-biome share and mean
  weight, the pick drift (the share of points whose pick at the coarsest modulated level differs: what the ground's
  coarsest modulated octave reads vs the material), where each pair of biomes meets nearest the centre with the player's `--spawn=U,V` (metres along the base's tangent plane: x, z on the flat world), per-field range,
  the ground under each biome (mean altitude above the base, height std, mean slope), CPU us per point, `biomes_blend.png` /
  `biomes_pick.png` / `ground_height.png` / `field_*.png`, and `heights.f32` + `heights.json` for `tools/landforms`.
  Tune a table from it before any GPU run. To match a real heightmap: `uv run tools/landforms/match.py SITE TABLE --fit=3`
  (`docs/landforms.md`).
- `tools/cascade_test.sh [--no-build] [--lods=25] [--check=0|1] [--points=0|1] [--table=0|1] [--timing=0|1] [--runs=10] [--sets=8,4,4c]`
  (`CascadeSpike`, `res://debug/cascade_spike.tscn`, local RenderingDevice: needs a window, ~3 min): the
  GPU cascade passes vs `Cascade::run`, every channel of every level (read back level by level) and the
  `sdf_sample` of 64+ chunks per LOD 0-24 near 0, +-2^60 and across 2^32 boundaries, in mixed-LOD batches,
  both channel sets and `4c` (LINEAR4 + the example climate at 5 cm; `4f` its fields without multipliers; `4r` the two-landform region table) (G1); the LOD relation on the GPU outputs (G3, at the levels no climate modulates: a modulated level reads the carry its parent level derives from fields the parent chunk's samples don't carry); the discoverer's point items vs
  `Cascade::point` + `SDFCascade::sdf`, 5,120 corners per LOD (`--points`, D1); the meshers' samples of a cascade
  with 600 scattered instances and the chunks' real lists vs `SDFCascade::sample_sdf` + skirt (`--table`, G9); the
  example climate's per-sample biome material vs `SDFCascade::ground_material`, every LOD (`--biomes`, G10); GPU timestamps of the passes for
  single-LOD batches of 128 at LOD 0, 4, ... 24 (G6, G7). Report: `<run>/cascade.md`. Run it after touching
  `cascade*`, `exact_noise.hpp`, `gpu_frame.hpp`, `sdf_cascade.hpp` or `cascade_points_gpu*` / `local_points_gpu*`.
- CI (`.github/workflows/ci.yml`) runs format, the `-Werror` build and the unit tests on every PR.
  Everything that needs a GPU stays local.
- `tools/check.sh [--no-build] [--spike] [--discovery] [--keep-going]`: the correctness gate, ~2 min.
  One build, the unit tests, then a direct load run with `--validate=64 --audit=1 --discovery_audit=1`,
  `origin_parity.sh`, `frame_test.sh`, the QuadPagePool spike (with `--spike`, or automatically when
  `src/quad_page_pool*` differs from `dev`) and `discovery_audit.sh` (with `--discovery`, or
  automatically when `src/chunk_discoverer*`, `src/sdf*` or `src/chunk_math*` differ from `dev`, ~2 min).
  Prints PASS/FAIL per step. Perf numbers come from `bench_ab.sh` instead.
- `tools/bench.sh [--no-build] --mode=direct|legacy --scenario=load|fly [--shot=loaded:x.png]
  [--view=0..7] [--validate=64] [--audit=1] [--cool=1]`: builds, runs `adapter/godot/project/debug/bench.tscn` with a fixed
  camera, writes `bench/results/<run>/stats.json` + `godot.log` + screenshots, prints a summary.
  Exits non-zero on build failure, crashes, `ERROR` lines, leaked RIDs, and (exit 4) a requested
  validation or audit that failed. `bench/results/LAST` holds the latest run dir.
  A load is complete once the received scan is the camera's current one and the mesher is idle
  (`VoxelTerrain::is_load_complete`).
  `--origin=X,Y,Z` sets the world origin (voxels). `--cool=1` waits for the GPU to cool first (perf
  runs only; `bench_ab.sh` / `bench_compare.sh` pass it): until 65 C, a plateau, or 30 s at most.
  `BENCH_ROOT=<checkout>` benches another tree with this harness. `BENCH_RESOLUTION=1920x1080` sets the window
  (default 1280x720; the LOD follows the pixels, so 1080p draws ~25% more quads).
  `--profile=1`: a per-frame profile, `<run>/profile.json` and `profile.md` (`tools/profile_report.py <run>`):
  per phase the wall frame time, each `VoxelTerrain::_process` section, the GPU and render-thread CPU per
  renderer pass and mesher step, and what the frames over 16.7 ms spend against the others (see `docs/gotchas.md` for
  how the per-pass timestamps are obtained). It also tabulates, from `stats.json`, the stream per phase
  (`table.stream`: keys computed / evicted per kind and LOD in `computed_by_key`, worker busy, admit / poll / land /
  evict ms, `park_frames`, the gate's `stale_parked`, `destroy.handoff_frames`: frames a stalled parent stays drawn,
  {count, p50, p95, max}, cumulative) and `mesher.gpu_by_list_words` (GPU µs per batch by its list words, log2 bins).
  `_process` sections: `stream_poll`, and `gate_admit` / `build_lists` / `record_bytes` inside `mesher_dispatch`.
- `--trace=1` (`--trace_max=N` events, default 2M): the engine trace (`src/trace.hpp`), `<run>/trace.json`, a Chrome
  trace for ui.perfetto.dev: one track per thread (main, render, discoverer, field_gpu, the pool's workers) with
  `OV_TRACE(system, stage)` spans, a GPU track (every captured frame's intervals: the mesher's `vx_*` always, every
  renderer pass with `--profile=1`, placed as early as their recording allows) and flows for the hand-offs (`scan`
  discoverer -> main, `batch` main -> render -> GPU -> readback -> main, `stream_job` start -> worker, `field` worker ->
  FieldGPU, `kill` / `compact` / `hidden` main -> render). Each span records its parent (the span open on its thread
  when it started) and each flow point the span it is in: the call tree is exact whatever the clock's resolution.
  `tools/engine_map.py <run>` (run by bench.sh) draws `engine_map.html` (a lane per thread, each stage's
  ms per frame against the 16.7 ms budget, the hand-offs as arrows with their median latency, per phase) and
  `engine_map.md` (hot spots by self time, hand-off latencies). `tools/engine_structure.py <run>` (run by bench.sh)
  draws how the engine is built, without timings: `engine_structure.html`, per thread the stages nested in the stage
  that runs them in execution order, how often each runs, the hand-offs as arrows, and per stage its code (found by
  scanning `src/` for the trace points: file:line, function, the comments there and on its declaration);
  `engine_structure.md`, the same tree as text to diff between runs, plus the trace points no run reached. A new system gets a span where its work runs and a
  flow where it hands work to another thread; `_process` sections are spans through `Section`. Off, a span costs one
  atomic load.
- `tools/cpu_profile.sh [--no-build] [--sampler=perf|eu-stack|auto] -- <bench.sh args>`: builds with
  `debug_symbols=yes` (+ frame pointers; the next plain build recompiles everything), samples the bench's Godot
  (perf at 499 Hz on-CPU, else eu-stack every 0.5 s, wall clock, which stalls the frames) and writes `<run>/cpu.md`
  (`tools/cpu_report.py`): top self symbols of the main thread and of the other threads, top inclusive ones.
  perf at 499 Hz slows a 1M load by ~40%: shares from the sampled run, times from an unsampled one. The 1M profile of
  2026-10-23 and its ranked causes: `docs/forest-1m-profile.md`.
- `bench.gd --scenario=approach`: loads 1,500 m above and 2 km behind the start (`--approach_from`, `--approach_out`,
  along -X), flies straight to it in `--approach_seconds=20` without waiting, then settle / steady;
  `--approach_shots=DIR` saves a shot per second in the run's `--view` (run it twice for shaded | LOD).
- `--validate=N` (direct): re-meshes N loaded chunks with the legacy mesher and compares
  triangle by triangle. `ok` requires identical triangle counts and 0 *unexplained* anomalies.
  The terrain SDF is bit-exact, so both meshers see the same signs (no winding flips at |SDF| ~ 0).
  Vertex positions differ by up to ~0.02 cells between
  the two meshers, so a quad smaller than the 0.02-cell tolerance can't be matched one to one; a
  leftover sliver near a same-size triangle of the other side counts as `ambiguous_slivers`, not an anomaly.
  The legacy triangulation takes each point's cell from the index the GPU wrote, never from its quantised
  position (a vertex on its cell's upper face, a crossing at t = 1.0, would floor into the next cell).
- `--audit=1` (direct): reads every page back and classifies all drawn slots
  (live / dead_zero / stale / partial / broken). Anything but live/dead_zero is a bug. Also writes a
  per-chunk mesh digest (quad count + order-independent hash of the chunk-local corners) to
  `<run>/chunks.json`.
- `--discovery_audit=1` (direct): rebuilds the current scan without culling (`--discovery_cull=0`) and
  counts the quads of every chunk the cull dropped on the GPU (passes sdf/edges/vertices, count only).
  A dropped chunk with quads is a miss, i.e. a hole; `ok` needs 0 misses and a max LOD jump of 1
  between face neighbours. `bench.sh` also fails a load-complete run whose loaded set != scanned set
  (`chunks.loaded_not_scanned` / `scanned_not_loaded`).
  `tools/discovery_audit.sh [--only=a,b] [-- bench args]` runs it over 11 poses/settings (default,
  lod_error 4, underground, east, high, deep, far origin, 5 cm voxels, fly, and with `--lods=25`
  `lods25_10km` looking down from 10 km up, `lods25_100km` at the horizon from 100 km up) and writes
  `bench/results/<date>-discovery.md`. Run it after touching the discoverer or the SDF.
- Hole hunting: run the main scene with `godot --path adapter/godot/project -- --marks=DIR`; H saves the camera
  pose, a screenshot, the chunk counts and a discovery audit to `DIR/marks.jsonl` (closing the window
  saves a last pose). Replay a mark with `bench.sh --cam_pos=... --pitch=... --yaw=... --lod_error=4`.
- `--view=N` debug views (direct): 1 normals, 2 LOD, 3 page, 4 wireframe, 5 chunk, 6 group, 7 quad index,
  8 faceted (clay, per-triangle normals: crisp creases; the shapes scene starts in it), 9 material (palette ids),
  10 / 11 the bark seam probe (raw bytes, `bench.gd --probe`), 12 biome (each material's debug colour: a climate's
  biomes; B toggles it in the player), 13 coverage (hidden quads magenta, see Coverage). The player's `--spawn=U,V`
  (metres along the base's tangent plane, `VoxelTerrain.get_tangent_position`) starts it above that point.
- `res://debug/spike.tscn` (`QuadPoolSpike`): isolated test of the GPU -> page path (write, indirect
  count, kills) with CPU verification; run it after touching `QuadPagePool`.
- `tools/bench_compare.sh [--runs=2] "a:--mode=legacy" "b:--mode=direct"`: interleaved runs,
  median table with spread and Δ%, written to `bench/results/<date>-compare.md`.
- `tools/bark_seams.sh [--no-build] [--only=a,b]` (needs a window, ~2.5 min): the 5 cm forest at six close-up poses
  (marks where seams were seen), `bench.gd --probe` (page shader views 10 / 11: the bark's grain, chunk, LOD, depth,
  normal azimuth as raw bytes) and `tools/bark_seams.py`: pairs of neighbouring bark pixels, inside a chunk / across a
  same-LOD border / across a LOD change, where the grain steps 4x more than next to it; fails on lines of such jumps
  (> 100 pairs) or > 5% of same-LOD border pairs jumping. Writes `report.md` and `<pose>_seams.png` / `_shaded.png`.
  Run it after touching the bark, `value_noise` or the page shader's normals. The float hash failed it (270 pairs on
  lines on one pose); the current bark: 0.02% of pairs jump inside chunks, 3% across borders.
- `tools/frame_test.sh [--no-build] [--voxel=M]` (`FrameSpike`): GPU vs CPU check of the integer
  frame and the exact noise at 5 far origins: WorldPos and anchors, then every probe of the probe
  table bit for bit over a fixed grid plus 1,024 hash-random points per origin (hash, octave
  decomposition at k 0..62, perlin3/2, a 2D and a 3D fBm, the archetype IR's `arch_*`
  sets, at LODs 0, 1, 5, 14, 24), pairs reaching the same voxel, absolute points identical
  at every origin, and continuity along lines across voxel and noise-cell low-word wraps. Run it after touching
  `world_coord.hpp`, `gpu_frame.hpp`, `chunk_math`, `exact_noise.hpp` or the SDF. A new noise or SDF
  term adds an entry (CPU fn + GLSL fn) to `make_probes()` in `src/frame_spike.cpp`.
- `tools/origin_parity.sh [--voxel=M] -- --mode=direct --validate=64`: the same world point seen
  from different origins (`--camera_voxel`): runs zero, zero2 (determinism), then pairs one root span
  apart near 0, at ~1e15 m and at 2^48 (voxel and noise-cell low-word wraps). Each pair must load the
  same non-empty chunks with identical mesh digests (`chunks.json`); pair screenshots are informational
  (a camera 229 km from its origin jitters at the cm level). Report: `bench/results/<date>-parity.md`.
