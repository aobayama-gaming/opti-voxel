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
- Dual contouring (`VoxelTerrain.dual_contouring`, `--dual_contouring=1` in `bench.gd` and the main scene;
  off by default): both meshers' vertex pass places its vertex at a regularised QEF of the cell's Hermite
  data instead of the mass point (`qef_vertex` in `gpu_glsl.hpp`, rewritten into both passes by
  `dual_contouring_pass`). The Hermite normal at a crossing is the lerp of the edge's two corner gradients
  (no SDF evaluation, the same code on every terrain); tau = 0.15 * trace(A) / 3; the move is capped at
  0.5 cells from the mass point, then clamped to the cell; where the gradient at the placed vertex
  vanishes, its normal is the mean of the Hermite normals. The Hermite data must agree with the
  crossings: on an SDF with detail finer than the cell vertices jitter (the archetype IR's band rule drops such noise).

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
- LOD range: `VoxelTerrain.lod_numbers` (default 15, at most 31: the GPU reads a LOD from the id's low
  32 bits). `--lods=N` in `bench.gd` and in the main scene (`godot --path project -- --lods=25`).
  Above 15 only: the camera's clip planes follow
  its altitude (|SDF| at the camera, `VoxelTerrain.get_sdf_at`): near = altitude / 1000 (at least the
  scene's), far = min(near * 2^21, `VoxelTerrain.get_load_radius()`). Past far/near ~2^24 Godot's float
  projection loses its far plane and the light culler ERRORs every frame. The validator doesn't compare
  the normals of sliver triangles (all edges within 2 * tol): around a lone near-zero corner a whole
  closed sheet fits inside the tolerance and the vertex pairing is arbitrary (frequent at far LODs).
- Exact noise (`src/exact_noise.hpp`, CPU + `GLSL_EXACT_NOISE`): `hash_cell` = PCG3D of the cell's low
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
  voxel size is another world. Flat world: the datum (`SDFBase::DATUM_M`) is absolute y ~ 300 m, so far
  origins only show terrain with y = 0.
- Morton keys are relative to the origin: batch/page locality only within ±2^20 ids of it.
- Hermite cascade (`src/cascade.hpp`: CPU reference, the oracle; `src/cascade_glsl.{hpp,cpp}`: the GPU
  passes, bit-identical to it): each noise band (octave k) is inserted once on the lattice Λ_{k-3}
  (multiples of 2^(k-3) voxels: 8 samples per wavelength), then carried down by tricubic Hermite
  subdivision. Channels in units of the lattice spacing: `HERMITE8` (value, 3 slopes, 3 cross slopes,
  d3/dxdydz from `perlin3_cross`) or `LINEAR4` (value + gradient, transverse slopes linear). A LOD-L chunk
  stops at Λ_L, so `V_m(p)` is a pure function of the point: every LOD gets the same bits where they share
  points. Fixed order x, y, z; only + - * by 0.5, 0.125, 0.75; after each level |c| < 2^-64 is flushed to 0
  (without it a band carried far down makes denormals). `Cascade::point` = the same run on a one-point box.
  4 samples per wavelength (worst case 2.2% value error) was rejected (`tools/cascade_report.sh`); the
  meshed set is `LINEAR4` (G7 of the design: 8 channels cost 1.30x the GPU time of 4, threshold 1.15).
- GPU cascade: one dispatch per level for the whole batch (level L + j of every chunk, j from T minus the
  batch's smallest LOD down to 0), boxes in closed form from the chunk's anchor on (lo, hi) words
  (`GLSL_FRAME_INT64`, `Cascade::level_box`), a child point replays the CPU's x, y, z passes on its 2x2x2
  parents (code generated from `Cascade::PAIRS`), ping-pong scratch A (odd j, 17^3 points per chunk) / B
  (even j >= 2, 10^3), j = 0 straight to `sdf_sample`. Bands come from a table indexed by level
  (`CascadeGLSL::band_table`). `CascadePasses` owns the passes per RenderingDevice; the direct mesher runs
  them in place of its sdf pass (and in the discovery audit), the legacy one in sub-batches of 128.
- Terrain: `VoxelTerrain.terrain_mode` 1 = `SDFCascade` (default, `src/sdf_cascade.hpp`: 3D height field,
  octaves k = 3 .. lod_numbers + 4, steepness 1/8, amplitudes capped at 3.3 km, datum at 300 m, absolute
  position only), 2 = its continents preset, 3 = `SDFArchetype` (the archetype IR's demo, below), 4 = its shapes
  gallery (`SDFArchetype::shapes`, `--terrain=shapes`, `res://shapes_scene.tscn` with a 60 km far plane: hard-CSG
  cubes with sphere holes, each twice the size and distance of the last, i.e. one per LOD, plus a die, stairs and a
  tunnelled cube on a flat ground at 250 m, off the page shader's 100 m contours, for dual contouring's creases;
  compiled into the table's ground program, with no table instances: its composites nest, which the table doesn't). 0 was
  `SDFDummy`, removed: it evaluated every octave at every LOD (detail finer than the cell, aliased far
  away), the cascade never inserts a band finer than 8 cells of the chunk's LOD. `--terrain=cascade|continents|archetypes|shapes`
  in `bench.gd` and the main scene (`godot --path project -- --lods=25`). On the cascade terrains the
  player spawns 200 m above the ground and flies at half its altitude per second, and the bench camera
  starts 50 m above it unless `--cam_pos` or `--above_ground=M` (M metres above the terrain under the
  camera, negative below) says otherwise: the cascade's ground is kilometres from the datum.
- Archetype IR (`src/archetype_ir.*`, builders and fuzzer in `archetype_build.cpp`): an SDF as a linear SSA
  program of dual numbers (value + forward-mode gradient), the one source of its GLSL (`archetype_glsl.*`,
  straight-line `precise` code, one statement per instruction) and of its CPU evaluation (`Archetype::run`,
  an interpreter: archetypes are runtime data), bit-identical. On top of the exactness rules (Gotchas): no
  division (1 / k is a constant); RSQRT only of `max(x, c > 0)` (`Program::validate`), because `det_rsqrt`
  reads the bits and the GPU doesn't keep the sign of a zero (NVIDIA folds x + 0.0 to x): nothing else lets
  a zero's sign reach a non-zero value, and results are compared through `zero_canonical` (-0 read as +0).
  A point is in the exact domain when every register component is 0 or within [2^-63, 2^63], so that a
  product of two is normal (`exact_domain`). Blends keep their end values exact at any magnitude
  (`smooth_min` = min - k/4 h^2, BLEND = parent (1 - w) + child w): a far instance reads 2^30 voxels
  (saturated INPUT), and the lerp b + (a - b) h rounded the ground to its ulp, terraces of 64 voxels at far
  origins. Instances
  (`archetype_instance.*`): a `Def` at an int64 anchor (INPUT = exact `sat_diff`) with a float rotation
  applied op for op, children combined by union / smooth union / subtract / smooth subtract / intersect /
  blend; `compile` flattens the tree into one Program, with a world box and a Lipschitz bound per node
  (sampled on 100k points per scene in the unit tests). Adding an op: `Op`, `op_arity`, `op_cost`, `is_flat`,
  `validate`, the interpreter's `eval` and the generator's `statement` (same formula, same order), a case
  in the float32 reference test (`test_archetype_ir.cpp`), a branch in `random_program`, then
  `tools/frame_test.sh` and a new text pin (the FNV test). FrameSpike's `arch_*` probes
  (`archetype_probes.cpp`): every shape builder, each operator, the test scene, the demo (`arch_demo`),
  `arch_frames` (one scene at the same offset from every origin, compared across origins) and 200 fuzzed
  programs, output plus a hash of every register; evaluations outside the exact domain are counted apart
  (`inexact_*`, gated like the rest). `SDFArchetype` (`src/sdf_archetype.hpp`, `--terrain=archetypes`): the
  compiled scene as `terrain_sdf` and `evaluate()`: an fBm ground (`height_fbm`, first-order distance), four
  noisy rocks and three trunk capsules smooth-unioned in, a cave mouth smooth-subtracted, absolute anchors
  in front of the main_scene spawn; far from them it is the bare ground, bit for bit (unit-tested).
- Archetype table (step 3, `src/archetype_table.*`, `instance_table.*`, `archetype_gpu_table.*`): the
  instances are no longer compiled into one program, they live in a GPU table and each chunk folds only
  the instances near it. A library type (`TableDef`) is a Program written once in the instance's frame:
  its sizes are `T_PARAM` (one sphere type serves every radius), its anchor, rotation and noise lattice
  come from the instance's record through `T_INPUT` / `T_PERLIN3|2` (`i64_sub` of the Def offset minus the
  record anchor, seed salted by the record). `generate_library_glsl` emits `vec4 arch_<k>(WorldPos p, int
  lod, DefRecord rec)`, the CPU interpreter runs the same program with a `DefRecord` (unit test: identical
  to the compiled path per type / rotation / anchor). `generate_table_sdf` emits the ground (a compiled
  Program), the library, the fold and the epilogue; the fold reads `sdf_records` / `sdf_list` and the
  `ChunkInput` global `g_sdf_chunk` (the chunk's list offset in `id.w`, count in `anchor_hi.w`).
  `SDFBase::get_gpu_table()` hands the mesher the `DefRecord` bytes (`record_bytes`, re-uploaded when the
  snapshot version changes) and the per-batch lists (`build_lists`); each mesher owns the buffers on its
  own RenderingDevice (direct bindings 8/9, legacy 6/7). **Locality rule:** an instance's influence box
  is its world box grown by `M(lod) = ceil(k + margin_c * 2^lod) + 2` voxels, `margin_c = 2 L +
  skirt_depth`; outside it the value is >= M, so folding it in or out only changes F where `|F| > 2 L
  2^lod` (the mesh and normals are unchanged). The test is integer (`sat_diff` vs an int32 box), so the
  CPU and the GPU, and two chunks of the same LOD, take the same branch: seams stay exact.
  **Band rule (step 4):** a noise op of octave k (PERLIN*, T_PERLIN*) is +0 at LODs above k - 3
  (`ExactNoise::band_kept`, `BAND_SHIFT` = the cascade's insertion lattice: 8 samples per wavelength), on
  both back-ends. Each record has a `max_lod` (`set_band`): at LOD L its distance is
  raised by `ERODE_CELLS * (2^L - 1)` voxels (1 cell per LOD: it shrinks uniformly, GPU and CPU alike), and
  it is listed only below the first LOD whose erosion reaches its depth (half its box's smallest side times
  L) plus k, where the eroded shape no longer reaches zero, so the drop changes no sign. The band steps
  between LOD L and L+1 (an eroded instance, a dropped ground octave) are hidden by skirts: the archetype
  terrains ask for `SDFBase::min_skirt_depth` (ground octave step + erosion, 3.34 cells on the demo) and
  VoxelTerrain raises `skirt_depth` to it before `configure` (unit test: the L | L+1 field step within the
  skirt at LODs 0-6 on a 4,000-rock scatter). `archetype_band` / `--instance_band=0` keep every instance
  whole (the reference). A rock scatter at a fixed density gives the same list lengths whatever N, empty
  from LOD 5 on.
  **The index** (`instance_table`): an instance is stored once, at the finest LOD whose chunk span holds its
  grown box, in the node holding its centre (every touched node only when wider than a top-LOD node);
  subtree lists stop above each instance's `max_lod`. A chunk's list gathers the subtrees of the 125 chunks
  within 2 and the 27 nodes around each coarser ancestor, and keeps the instances listed at its LOD whose
  influence box meets its sample box (`chunk_sample_box`, `influence_meets`), in id order; the CPU
  `evaluate` does the same for the point (`point_candidates(voxel, lod)`). Unit-tested against the brute
  force over six decades of sizes at LODs 0-13. Two layers: a base built once (flat arrays, an
  open-addressing `NodeRangeMap`, Morton-ordered levels; 1M rocks in ~1.8 s) and an overlay over the records
  edited since, rebuilt by every edit (~15 us at 1M); edited base entries are masked; past
  `InstanceSnapshot::REBASE` (2,048) edited records the base is rebuilt (unit test: the overlay lists what a
  whole new index lists), on the main thread for now: ~1.8 s at 1M every 2,048 edited records (moving it to a
  worker and publishing when done is the follow-up). Records live in `RecordStore` pages of 4,096 shared by the snapshots (copy on
  write per page); the GPU re-uploads the runs of pages changed since its version (`record_bytes` spans,
  `SDFGpuTable::upload_records`). `build_lists` runs on the worker pool; the list buffer holds batch size *
  `max_chunk_list()` (4,096) words, a batch over it is cut (ERR, `table.list_truncated`; the GPU then
  disagrees with the CPU there). `get_debug_stats()["table"]`: list length avg / max per LOD,
  list build ms. `SDFBase::cull_exempt` (`near_instance`, from the index: per-LOD occupancy + coarser
  nodes' boxes) feeds the discoverer's no-cull rule, band or not: a node near an instance's box is never
  culled (asked only when the corners would cull). FrameSpike's table probe checks the library + fold (with
  erosion) GPU == CPU. **To add a
  library type:** a `Builder` in `archetype_table.cpp` using `t_local` / `t_param` / `t_perlin*` and a
  bounds lambda of the params, then `DefLibrary::add` (a new *op* also needs the "Adding an op" list).
  **Runtime edits:** `VoxelTerrain.add_instance(type, x, y, z, rotation, params, op, k)` /
  `remove_instance(id)` / `move_instance(id, x, y, z, rotation)` (anchors are int64 LOD0 voxels, never
  through `Vector3i`); the op must be 0-3, k >= 0, params finite and normal, the rotation orthonormal,
  else the edit is refused (ERR, -1 / false). Each edit makes a new snapshot (the base kept, the overlay
  rebuilt) and bumps its version. **The snapshot is read from other threads** (the discoverer's `evaluate` / `cull_exempt`):
  it is published with `std::atomic_store` and every reader takes one `current()` copy (a plain
  assignment freed it under the reader, unit-tested). The meshers build a batch's lists and, when the
  version changed, its records on the main thread from the same snapshot and hand both to the GPU with
  the batch. The edit region (old box of a remove/move plus new box
  of an add/move) drives two things: the loaded chunks whose box grown by `M(lod)` meets it, at the LODs
  the changed record is listed at, are re-queued
  in the mesher's `remesh_queue` (dispatched before new chunks, finest LOD first, `remesh_chunks` /
  `remesh_pops`), and the discoverer's `invalidate` drops the cull verdicts near it and forces a rescan
  even if the camera chunk did not change. A chunk carries a `required_version`: a result computed from
  an older table version is dropped and re-queued (`remesh_stale_dropped`), and its quads, already in the
  page, are killed (else the old surface stays drawn); an edit also requires the new version from the
  chunks in flight for their first mesh (not loaded yet, so not listed). `remesh_queue` holds a chunk once. `add_new_loaded` drops a
  pending `to_destroy_chunk` entry so a re-meshed chunk is not freed twice. The GPU buffers are sized for
  `instance_capacity()` instances up front (default 16384 or the scatter + 4096, removed ones count); an add
  past it is refused. The bench's
  `--scenario=edit --edits=N` scripts add / move / remove edits, then settles; validate / audit /
  discovery_audit should stay `ok`. `--instances=N --instance_area=M` scatter N rocks and pillars in a
  square of side M metres around the demo objects (set before the node enters the tree). **Experimental,
  default 0:** with N > 0 a rare (~1e-7) non-finite sample turns into a broken quad (the auditor's
  `broken` / `nan_vertices`), CPU and GPU alike, so the mesh otherwise validates; the scatter is off
  until that edge is understood (a follow-up, likely in the smooth-op / epilogue numerics), not the
  table wiring.
- **The cascade only exists on lattice points.** `SDFCascade::evaluate(p, lod)` takes `frac == 0` and a voxel
  multiple of 2^lod only (ERR and 0 otherwise); the discoverer's corners and the validator's cell corners are
  such points, `get_sdf_at` snaps to the LOD0 voxel. There is no off-lattice evaluation and no GLSL
  `terrain_sdf` (a NaN stub): vertex normals are the trilinear blend of the cell's 8 sample gradients
  (`CascadeGLSL::GLSL_VERTEX_GRADIENT`), in both meshers. A new cascade terrain keeps this rule.
- Cascade code is only added to a mesher shader in cascade mode, and `GLSL_EXACT_NOISE` is spliced from fragments it shares with
  `GLSL_PERLIN3_CROSS` (FNV unit test on its text). `OPTI_VOXEL_SHADER_DUMP=<dir>` writes every composed
  shader to `<dir>/<name>_<hash>.glsl` to compare two builds.

## Verify changes (do this, don't eyeball)
- `tools/unit_test.sh [--no-build] [filter]`: the C++ unit tests (`src/tests/`, doctest) inside
  headless Godot, a few seconds, no GPU. `filter` is a doctest wildcard on test names
  (`'ChunkMath*'`). Tests are compiled into non-release builds only (`OPTI_VOXEL_TESTS`).
- `tools/format.sh [--check] [files]`: clang-format 23.1.3 through `uvx` on `src/`.
- `tools/cascade_report.sh [--no-build] [--scale=N]` (`CascadeReport`, headless, CPU only, ~40 s at
  `--scale=8`): fidelity of a cascaded octave vs analytic perlin3 (4 / 8 samples per wavelength, 8 / 4
  channels), points and CPU ms per chunk per LOD on an Earth, PNG slices (analytic | cascaded | error).
- `tools/cascade_test.sh [--no-build] [--lods=25] [--check=0|1] [--points=0|1] [--timing=0|1] [--runs=10] [--sets=8,4]`
  (`CascadeSpike`, `res://debug/cascade_spike.tscn`, local RenderingDevice: needs a window, ~3 min): the
  GPU cascade passes vs `Cascade::run`, every channel of every level (read back level by level) and the
  `sdf_sample` of 64+ chunks per LOD 0-24 near 0, +-2^60 and across 2^32 boundaries, in mixed-LOD batches,
  both channel sets (G1); the LOD relation on the GPU outputs (G3); the discoverer's point items vs
  `Cascade::point` + `SDFCascade::sdf`, 5,120 corners per LOD (`--points`, D1); GPU timestamps of the passes for
  single-LOD batches of 128 at LOD 0, 4, ... 24 (G6, G7). Report: `<run>/cascade.md`. Run it after touching
  `cascade*`, `exact_noise.hpp`, `gpu_frame.hpp`, `sdf_cascade.hpp` or `discoverer_gpu*`.
- CI (`.github/workflows/ci.yml`) runs format, the `-Werror` build and the unit tests on every PR.
  Everything that needs a GPU stays local.
- `tools/check.sh [--no-build] [--spike] [--discovery] [--keep-going]`: the correctness gate, ~2 min.
  One build, the unit tests, then a direct load run with `--validate=64 --audit=1 --discovery_audit=1`,
  `origin_parity.sh`, `frame_test.sh`, the QuadPagePool spike (with `--spike`, or automatically when
  `src/quad_page_pool*` differs from `dev`) and `discovery_audit.sh` (with `--discovery`, or
  automatically when `src/chunk_discoverer*`, `src/sdf*` or `src/chunk_math*` differ from `dev`, ~2 min).
  Prints PASS/FAIL per step. Perf numbers come from `bench_ab.sh` instead.
- `tools/bench.sh [--no-build] --mode=direct|legacy --scenario=load|fly [--shot=loaded:x.png]
  [--view=0..7] [--validate=64] [--audit=1] [--cool=1]`: builds, runs `project/debug/bench.tscn` with a fixed
  camera, writes `bench/results/<run>/stats.json` + `godot.log` + screenshots, prints a summary.
  Exits non-zero on build failure, crashes, `ERROR` lines, leaked RIDs, and (exit 4) a requested
  validation or audit that failed. `bench/results/LAST` holds the latest run dir.
  A load is complete once the received scan is the camera's current one and the mesher is idle
  (`VoxelTerrain::is_load_complete`).
  `--origin=X,Y,Z` sets the world origin (voxels). `--cool=1` waits for the GPU to cool first (perf
  runs only; `bench_ab.sh` / `bench_compare.sh` pass it): until 65 C, a plateau, or 30 s at most.
  `BENCH_ROOT=<checkout>` benches another tree with this harness.
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
- Discoverer: the cull verdict of a node only depends on its id and the (fixed) SDF, so it is cached
  across scans (`discovery_cache`, default on); `--discovery_verify=1` recomputes every hit and
  ERRORs on a mismatch. The walk is breadth first, one wave per LOD: each wave's uncached nodes give
  their 8 corners, deduplicated on the voxel (~6 nodes share a corner), evaluated in one batch
  (`SDFBase::evaluate_batch`; `SDFCascade` spreads it over the `WorkerThreadPool` with the
  allocation-free `Cascade::point_into`). `VoxelTerrain.discoverer_mode` / `--discoverer=cpu|gpu` (bench
  and player): -1 auto (default: GPU for a cascade terrain, CPU otherwise), 0 CPU, 1 GPU. The GPU path,
  `DiscovererGPU` (`src/discoverer_gpu.*`), runs the cascade passes on point items
  (`CascadePasses::create_points`) on a **local RenderingDevice owned by the discoverer thread**: created
  at the top of `discovery_loop` and freed there before the thread exits; no other thread may touch it,
  and it does nothing on the main RD or the render thread. No local device (headless) or no cascade: one
  warning, CPU. Both paths give the same bits; `--discovery_compare=1` runs a fresh CPU scan for every
  scan and ERRORs on any difference (its time counts in the discoverer's latency stat): run `discovery_audit.sh -- --terrain=cascade --discovery_compare=1`
  (and `--lods=25`) after touching the discoverer, the cascade passes or `DiscovererGPU`.
  `stats.json["discoverer"]`: mode, scratch MB, `gpu_init_ms` (device + shader, before the first scan),
  compare counts, first and last scan (scan / cull / walk
  ms, corners, unique corners, dedup ratio, per wave GPU ms, batches, readback bytes). `--cam_pos=X,Y,Z --pitch=DEG --yaw=DEG` place the bench camera;
  `--camera_voxel=X,Y,Z` makes the scene (camera, lights, cam_pos) relative to that absolute voxel
  instead of the world origin.
- Destroy check (`VoxelState`, `src/chunk_memory.*`) is incremental: the discoverer publishes each scan
  with its delta (`ScanDelta`: entered / left, merged across the scans the main thread skipped), and
  VoxelState keeps per event the dirty chunks' nearest loaded ancestors, the stall count per loaded chunk
  (split rule) and `leaving` (loaded, not scanned); `update_destroy` only looks at `leaving`. It relies
  on the scan being an octree cut. `compute_destroy_full` is the oracle: `--destroy_verify=1` checks every
  delta and every destroy check against it and ERRORs on a difference (unit test: the randomized run in
  `test_chunk_memory.cpp`). Run a fly with it after touching `chunk_memory`, the discoverer's publish or
  the meshers' result path. `destroy.rescans` counts the rare pass over the dirty set (a stale result).
- Hole hunting: run the main scene with `godot --path project -- --marks=DIR`; H saves the camera
  pose, a screenshot, the chunk counts and a discovery audit to `DIR/marks.jsonl` (closing the window
  saves a last pose). Replay a mark with `bench.sh --cam_pos=... --pitch=... --yaw=... --lod_error=4`.
- `--view=N` debug views (direct): 1 normals, 2 LOD, 3 page, 4 wireframe, 5 chunk, 6 group, 7 quad index,
  8 faceted (clay, per-triangle normals: crisp creases; the shapes scene starts in it).
- `res://debug/spike.tscn` (`QuadPoolSpike`): isolated test of the GPU -> page path (write, indirect
  count, kills) with CPU verification; run it after touching `QuadPagePool`.
- `tools/bench_compare.sh [--runs=2] "a:--mode=legacy" "b:--mode=direct"`: interleaved runs,
  median table with spread and Δ%, written to `bench/results/<date>-compare.md`.
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
- `--render-thread safe` (rendering on the main thread): every bench run reports ~581 leaked
  StorageBuffer RIDs at exit, so `bench.sh` fails (exit 3). Bench runs need the separate thread.
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
- CPU mirrors of GPU float math (`ChunkMath::sample_pos`, `exact_noise.hpp`, `Archetype::run`, `Cascade::run`) must
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
- A local RenderingDevice's timestamps (`get_captured_timestamp_*`) are readable right after `sync()` and
  only until its next frame begins: `buffer_get_data` flushes one, so read them first (`DiscovererGPU`).
- Main-RD resources must be created/freed on the render thread; objects owning them delete
  themselves there (`destroy()` -> `_rt_destroy()`), after the render-thread work already queued
  for them (FIFO).
- Async readback callbacks (`texture_get_data_async`, `buffer_get_data_async`) are NOT skipped when
  their object is freed: they arrive frames later and run on freed memory (segfault in
  `MeshValidator::_rt_on_page` when quitting mid-validation). An object with readbacks in flight
  counts them and deletes itself when the last one lands (`MeshValidator` / `PageAuditor::destroy`).
  `GPUDirectMesher::_rt_destroy` does not do this yet.
