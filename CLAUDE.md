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
  opens the main scene headless: the terrain finds no RenderingDevice, prints one ERROR and stays
  empty (it used to segfault in `QuadPagePool::_rt_create_page`). A fresh import may abort at exit (Godot 4.6.3,
  dev too): CI runs it a second time, which must exit 0 with `extension_list.cfg` written.
- When a feature is done and its checks pass, commit, push and open the PR into `dev` without waiting
  to be asked. Don't merge it: the user reviews and merges.
- Perf PRs carry a measured before/after table and name the new bottlenecks.
  `tools/bench_ab.sh <dev checkout> <worktree> -- --mode=direct` builds both trees, interleaves
  the runs (2 per side by default, both through the worktree's `bench.sh`) and writes the table to paste
  into the PR. Comparing a tree against itself still shows load-time deltas of ~5%: that's the noise.
  **Perf target:** `--preset=forest5cm` (`project/debug/presets.gd`: 4,000 trees, 5 cm voxels, lod_error 2,
  the camera in the forest) with `--scenario=fly`, aiming at 60 fps while moving at error 2 (stretch: 1).
  Perf PRs report it next to the default bench; the player takes the same `--preset` (main_scene_5cm).
  Baseline on the cascade (PR #48, 1280x720, budget 4, 2 runs): steady frame p50 13.7 ms (render GPU 14.0), fly out
  p50 11.7 ms, back 13.5 ms, 30.4 M live triangles, 865 MB of pages; open: the fly drops half its chunks on the way
  out and re-meshes for 6.4 s after the return (623 ms on the old fBm ground). The PR #34 numbers below were measured on
  the removed fBm ground (15% more triangles): fly frame p50 21-27 ms, steady 12 ms of render GPU; the mesher's tree
  folds (~7 ms of GPU per batch, 4x a treeless batch) and the 31 M triangles drawn are the two costs.
  Where the frame goes (`bench.sh --profile=1`, 1920x1080, error 2 | back 8, budget 4): GPU-bound in every phase
  (wall = GPU total, render thread 1.5-4 ms, terrain `_process` 3-5 ms while flying). Steady in the forest: 17.3 ms,
  of which the three geometry passes are 16.4 (opaque 6.6, shadows 4.9, depth pre-pass 4.9) and SSAO / SSIL /
  post 0.9; they follow the pages drawn (25-34 us of geometry per camera-pass page in every phase; live quads count off-screen chunks too). The fly's misses are the old chunks kept
  drawn until their replacements land (fly_back: 2x the live quads, geometry 21 ms, mesher still 4).
  Turning: geometry flat, the frames over budget are the mesher's batches. The pre-pass can't simply go:
  without it SSAO / SSIL lose their normals and shadowed surfaces turn black.
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
  13-bit fixed point, 1/256 cell + a 9-bit palette index + 16-bit octahedral normal, at most ~1 degree off).
  Layout, decoders and the GLSL of both (constants substituted, no literal elsewhere): `src/page_format.hpp`. Only a ~2 KB per-batch directory comes back to the CPU (asynchronously) for bookkeeping.
  Shared GLSL: `src/gpu_glsl.hpp`, direct passes: `src/gpu_direct_glsl.hpp`.
  Chunks start on a 64-quad group and a texture row is one group, so a chunk is whole rows.
  GPU budget (`VoxelTerrain.direct_gpu_budget_ms`, `--mesher_budget=MS` in bench and player; 0 = off, 4 in
  `main_scene_5cm`): a frame dispatches mesh chunks until their estimated GPU time reaches the budget, cutting
  a batch short. The estimate is `GPUCostModel` (`src/gpu_cost_model.hpp`): µs per chunk per LOD, corrected
  from each batch's timestamps (measured / estimated ~1.05 on the forest; a LOD-3 forest chunk costs 200-300 µs,
  a treeless one ~15). Forest fly at error 2: frame p50 27.8 -> 11.9 ms, at the price of a slower first load
  (3.0 -> 5.0 s) and merges lagging while moving (coarse parents queue behind the fine chunks: 2x the chunks
  loaded mid-fly). Ordering the queue by displayed error instead (the loaded ancestor's 2^lod / distance, merges
  at their own LOD) was measured worse: same GPU work, 4-6x the detail left pending at the end of a fly, +20%
  draw calls (`load_queue` keeps in-cone, finest LOD first).
  `GPUDirectMesher::compact()` re-packs pages under half full GPU to GPU and rewrites the moved
  `ChunkOutput`s in `loaded_chunk`; spare free pages are freed (`QuadPagePool::trim`).
- Dual contouring (`VoxelTerrain.dual_contouring`, `--dual_contouring=1` in `bench.gd` and the main scene;
  off by default): both meshers' vertex pass places its vertex at a regularised QEF of the cell's Hermite
  data instead of the mass point (`qef_vertex` in `gpu_glsl.hpp`, rewritten into both passes by
  `dual_contouring_pass`). The Hermite normal at a crossing is the lerp of the edge's two corner gradients
  (no SDF evaluation, the same code on every terrain); tau = 0.15 * trace(A) / 3; the move is capped at
  0.5 cells from the mass point, then clamped to the cell; where the gradient at the placed vertex
  vanishes, its normal is the mean of the Hermite normals (`vertex_normal`, below). The Hermite data must agree with the
  crossings: on an SDF with detail finer than the cell vertices jitter (the archetype IR's band rule drops such noise).

## Integer frame (no float ever holds an absolute coordinate)
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
  chunk_at_render`: render (Godot) space, relative to the world origin.
- World origin: `VoxelTerrain.world_origin_x/y/z` (LOD0 voxels, |v| <= 2^60), snapped down to the
  root chunk span (2^lod_numbers ids = 458,752 voxels with the defaults). No runtime rebasing yet.
- LOD range: `VoxelTerrain.lod_numbers` (default 15, at most 31: the GPU reads a LOD from the id's low
  32 bits). `--lods=N` in `bench.gd` and in the main scene (`godot --path project -- --lods=25`).
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
  main scene (`godot --path project -- --lods=25`). Outside the shapes gallery the player spawns 200 m above the
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
- Archetype IR (`src/archetype_ir.*`, builders and fuzzer in `archetype_build.cpp`): an SDF as a linear SSA
  program of dual numbers (value + forward-mode gradient), the one source of its GLSL (`archetype_glsl.*`,
  straight-line `precise` code, one statement per instruction) and of its CPU evaluation (`Archetype::run`,
  an interpreter: archetypes are runtime data), bit-identical. On top of the exactness rules (Gotchas): no
  division (1 / k is a constant); RSQRT only of `max(x, c > 0)` (`Program::validate`), because `det_rsqrt`
  reads the bits and the GPU doesn't keep the sign of a zero (NVIDIA folds x + 0.0 to x): nothing else lets
  a zero's sign reach a non-zero value, and results are compared through `zero_canonical` (-0 read as +0).
  A point is in the exact domain when every register component is 0 or within [2^-63, 2^63], so that a
  product of two is normal (`exact_domain`). Blends keep their end values exact at any magnitude
  (`smooth_min` = min - k/4 h^2): a far instance reads 2^30 voxels (saturated INPUT), and the lerp
  b + (a - b) h rounded the ground to its ulp, terraces of 64 voxels at far origins. Adding an op: `Op`,
  `op_arity`, `op_cost`, `is_flat`, `validate`, the interpreter's `eval` and the generator's `statement` (same
  formula, same order), a case in the float32 reference test (`test_archetype_ir.cpp`), a branch in
  `random_program`, then `tools/frame_test.sh` and a new text pin (the FNV test). FrameSpike's `arch_*` probes
  (`archetype_probes.cpp`): every shape builder, `arch_frames` (every shape at the same offset from every origin,
  compared across origins) and 200 fuzzed programs, output plus a hash of every register; evaluations outside the
  exact domain are counted apart (`inexact_*`, gated like the rest). The table probe covers the operators and
  rotations.
- Archetype table (`src/archetype_table.*`, `instance_table.*`, `archetype_gpu_table.*`): the instances live
  in a GPU table and each chunk folds only the instances near it. A library type (`TableDef`) is a Program written
  once in the instance's frame:
  its sizes are `T_PARAM` (one sphere type serves every radius), its anchor, rotation and noise lattice
  come from the instance's record through `T_INPUT` / `T_PERLIN3|2` (`i64_sub` of the Def offset minus the
  record anchor, seed salted by the record). `generate_library_glsl` emits `vec4 arch_<k>(WorldPos p, int
  lod, DefRecord rec)`, the CPU interpreter runs the same program with a `DefRecord` (unit test: the same bits for
  the same local point at any anchor, per type / rotation / LOD). `generate_fold_glsl` emits the library, `arch_fold(r,
  p, lod, touched)` (the GPU `fold_onto`) and `arch_finish` (`finish_table`). The fold reads `sdf_records` / `sdf_list` and the `ChunkInput` global
  `g_sdf_chunk` (the chunk's list offset in `id.w`, count in `anchor_hi.w`), all declared by
  `Archetype::glsl_table_buffers` (the one source, also FrameSpike's).
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
  raised by `erode * max(2^L - erode_from, 0)` voxels (`band_erosion`: `erode` cells per LOD past its onset, it
  shrinks uniformly, GPU and CPU alike; `erode_from` = 2^onset rides in `anchor_hi.w`; the timing is below, with
  `--instance_resolve`), and
  it is listed only below the first LOD whose erosion reaches its depth (`band_depth`: half its box's smallest
  side times L, plus k; the smaller of the local box grown by `rotation_margin` and the world box, so a turned
  record drops at the same LOD), where the eroded shape no longer reaches zero, so the drop changes no sign. The band steps
  between LOD L and L+1 (an eroded instance, a dropped octave) are hidden by skirts: the table asks for
  `SDFBase::min_skirt_depth` (the erosion rate), VoxelTerrain raises `skirt_depth` to it before `configure`, and the
  cascade adds it to its own octave step. Unit test: at the samples a LOD L and a LOD L+1 chunk share near the surface,
  LODs 0-6, a 4,000-rock scatter and a 5 cm forest, the field step is within the outer chunk's skirt, except over a
  record dropped at L+1: no skirt bounds that step, but what is left of the record at L is at most `erode` cells thick
  and the fine rim's skirt lifts it to air (checked). `archetype_band` / `--instance_band=0` keep every instance whole
  (the reference). A rock scatter at a fixed density gives the same list lengths whatever N, empty from LOD 6 on.
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
  whole new index lists) in one `WorkerThreadPool` task (`BaseRebuild`, over the records as they were: copy on
  write keeps them) while the edits go on through the overlay and log their ids; `VoxelTerrain::_process`
  (`SDFTable::poll_rebuild`) swaps in the finished base plus an overlay of the ids changed since, at the same
  version (nothing re-uploaded or re-meshed). Past 2x REBASE edited ids while it runs, an edit waits for it. A
  configure or the terrain's destruction still waits for a running build (no abort yet). Records live in `RecordStore` pages of 4,096 shared by the snapshots (copy on
  write per page); the GPU re-uploads the runs of pages changed since its version (`record_bytes` spans,
  `SDFGpuTable::upload_records`). `build_lists` runs on the worker pool; the list buffer holds batch size *
  `max_chunk_list()` (4,096) words whatever the batch's chunk count; a batch keeps the leading chunks whose lists fit
  it whole and the meshers queue the rest again (`table.list_split`, `list_requeued`: a LOD 6 forest chunk lists ~9k
  records, and the budget-cut batches of 2-3 chunks used to truncate them, empty chunk squares in the canopy); only a
  single list over the whole buffer is cut (ERR, `table.list_truncated`; the GPU then disagrees with the CPU there).
  `get_debug_stats()["table"]`: list length avg / max per LOD,
  list build ms. `SDFBase::cull_exempt` (`near_instance`, from the index: per-LOD occupancy + coarser
  nodes' boxes) feeds the discoverer's no-cull rule, band or not: a node near an instance's box is never
  culled (asked only when the corners would cull; a streamed source's node also while its nearest corner is within
  `ground_reach(lod)` past the cull band, in its zone). FrameSpike's table probe checks the library + fold (with
  erosion) GPU == CPU. **To add a
  library type:** a `Builder` in `archetype_table.cpp` using `t_local` / `t_param` / `t_perlin*` and a
  bounds lambda of the params, then `DefLibrary::add` (a new *op* also needs the "Adding an op" list).
  **Runtime edits:** `VoxelTerrain.add_instance(type, x, y, z, rotation, params, op, k)` /
  `remove_instance(id)` / `move_instance(id, x, y, z, rotation)` (anchors are int64 LOD0 voxels, never
  through `Vector3i`); the op must be 0-3, k >= 0, params finite and normal, the rotation orthonormal,
  the anchor within +-2^60 voxels and the box (any rotation) plus 2 k at most
  `SDFTable::max_instance_extent()` (4 top-LOD chunk spans, at most 2^29: `Archetype::record_fit_error`),
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
  `instance_capacity()` records up front: the table's `table_capacity()` (default 16384 or the scene + 4096, removed ones
  count; an add past it is refused), then the streamed slots. The bench's
  `--scenario=edit --edits=N` scripts add / move / remove edits, then settles; validate / audit /
  discovery_audit should stay `ok`. `--instances=N --instance_area=M` scatter about N rocks and pillars
  in a square of side M metres around (-50, -150) m (`SDFTable::SCENE_X / Z`, set before the node enters the tree),
  default 0: one per cell site (`SCATTER_SITE_SALT`), a rock or a pillar by the site's variant, turned to `up_at`.
  **Sources (`src/cell_scatter.*`):** the scatter and the forest are two `CellScatter`s {salt, zone, side, ground, up,
  accept, generate}: a site's records are a pure function of its cell, so two halves of a zone give the whole
  (unit-tested). `SDFTable::get_sources()`, streamed (below; stats under `get_debug_stats()["table"]["scatter" | "trees"]`:
  the profile, and `["stream"]`), or with `--stream=0` (`VoxelTerrain.archetype_stream`, bench and player) placed whole in
  the table at configure by `add_source` (count, records per site, moments per material; the reference, and the tests that
  edit their records by id). Biome features and the shapes gallery stay table records (whole-build blobs). `SourceProfile`
  reads `SAMPLES` (256) hashed sites at the origin: reach, `lod_end`, k / margin max (times `REACH_SAFETY` 1.25), joins,
  records per site and the mass lost per LOD (what a far proxy stands for); unit-tested to bound every record of the
  forest and the scatter (`covers`).
  **Keys (`src/scatter_source.*`):** `ScatterSource` cuts a source into keys `{source, PARTS | BLOBS | SITES, lod,
  tile}`, each a pure function of itself: `keys(box, lod)` (conservative, unit-tested against every tile's records),
  `inputs(key)`, `content(key, inputs)`, `ground_reach(lod)`. SITES holds a site walk and no records (`SourceContent::sites`,
  never listed, kept resident: the walk is ~80% of a part tile's CPU): `SITES 0` on the LOD-0 part-tile grid (8 source
  cells), `SITES L` on a far layer's blob grid (the proxies' walk, `L > 0`). A PARTS tile of LOD L (side doubled until it
  holds 2 chunks) grows the sites of the `SITES 0` tiles inside it (n^3, in the old record pyramid's z y x order, kept as
  the unit tests' oracle: bit-identical tiles and lists): every record at LOD 0, those listed at L coarser. A LOD with the
  LOD-0 side (1-3 at 5 cm) has no PARTS key: `keys()` emits `part_key_lod(L)`'s and the chunk's `meeting` filters at L.
  A BLOBS tile (8 layer cells, at least a part tile and 2 chunks) builds the layers on the items within
  `MARGIN_CELLS` (`PRUNE_CELLS` + 1) cells and judges its own cells (`MomentLayerParams::judged`): on part layers the
  LOD-0 tiles' records (the whole build's blobs bit for bit, unit test), above them proxies (unsnapped `surface_sites`,
  2 per cell side, the profile's lost moments: mass within 10%). Part layers are those under `Setup::part_cell_sides`
  source cells: 3 by default (the whole build's blobs), 1 for the streamed sources (`SDFTable::source_setup`): proxies
  from the source cell on (the 2 switched layers: mass within 20%, peak within 40%, unit test). Eviction for input records
  passes over placed keys and SITES (`count == 0`); the entries cap evicts them; `computed_by_key` names `sites L`.
  **Stream (`src/instance_stream.*`):** `InstanceStream` holds the keys asked for in one table (empty ones
  too) with an intrusive LRU: `admit(chunk, owner, tag)` answers READY (every key placed), PARKED (its keys pinned;
  `take_ready(owner)` hands it back, in park order, once they are) or LATER (a key finds no slot: stop popping). Keys are computed by drain
  tasks on Godot's pool at low priority (`StreamExecutor`, run by `poll()` in the tests; `low_priority_thread_ratio` 0.5), a
  key's inputs handed to it as shared pointers; the task lands each key under the stream's lock (its dependents start at once
  on the same task: landing on the main thread, 2 ms a frame, held forest5cm's 135k keys ~900 frames) and places the listed
  ones in slots past the table's capacity (`chunk_list` reads under a shared lock), logged per placement (`uploads_since(seq)`; a reader past the log gets every placed key). Eviction:
  oldest first, never pinned or asked for in the last `GRACE_FRAMES` (16). `point_records`: resident keys, else computed and
  dropped. Unit tests: park / return order, pins and grace, slot reuse with the log, slots full -> LATER with no deadlock.
  **Wiring:** `SDFTable` owns the stream over its valid sources (`add_stream`; `~SDFCascade` stops it first: its tasks read
  `ground_field`). The meshers gate every chunk they pop through `SDFBase::admit / take_ready / parked_count`, called by
  `ChunkGate` (`src/chunk_gate.hpp`, both meshers): one owner per mesher and queue kind (chunk, remesh, discovery audit),
  the scan generation as the tag (a first mesh parked before the last `send_chunk` is dropped: that scan queued again the
  ones it wants), a parked chunk keeps its mesher from idle. A chunk is meshed only once all its keys are placed, so its
  content never depends on what streamed when. `GpuTable` lists in `compose_list` order, the same for the CPU's points
  (`SDFTable::point_list`): the scene's records (ids below `scene_end`), the streamed ones (slots past `table_capacity()`),
  then the edits', so an edit carves streamed records too; the records go up from an `UploadCursor` {snapshot version,
  placement seq} (`record_bytes`: the snapshot's changed pages and the keys placed since). `VoxelTerrain::_process` calls
  `SDFBase::poll()` (the stream, then `poll_rebuild`) before the meshers. Unit tests: a streamed chunk's list folds like
  the CPU's point (`evaluate`) bit for bit, with the uploaded spans; the LOD L | L+1 field step stays in the skirt on the
  streamed scatter and forest. Presets `forest5cm_1m` (1M trees on 5 x 5 km) and `forest5cm_41m` (41M on 32 x 32 km): the
  stream computes what the view asks for, so their configure is forest5cm's.
  **Trees (step 5, `src/tree_archetype.*`):** a tree is a contiguous run of plain records, all smooth union
  (`SDFTable::get_trees()`: `TreeSpan` first id + count), no change to the IR, the fold or the meshers.
  Library types 3 `branch` (a tapered capsule {height, radius, slope}, slope = (r1 - r0) / height from the
  CPU, L = 1 + slope^2 / 2), 4 `crown` (`make_rock_type(crown_k(voxel), CROWN_SEED)`, bumps of 0.8-3.2 m whose
  amplitude adds `CROWN_BUMP` = 1.5 to L: lumpy foliage up close, smooth once the band rule drops the octaves) and
  5 `cone` (`make_cone_type`, the same bumps on a cone {radius, height, cos, sin, amp}, its rim a smooth max: a
  conifer's tier). Three species: a broadleaf, a slender one and a conifer (`cone_tiers` = 4 stacked cones,
  no limbs, ~12 records). `grow_tree` is one recursion for topology and bends: a stem's axis split
  at midpoints moved by a hash of their path (joints on whole voxels: a segment's end is the next one's
  anchor), children at hashed forks with pipe-model radii (r0^2 >= tip^2 + sum of the children's), stems
  grown thickest first so the per-tree budget cuts the thinnest; then a hull of 1-3 crown spheres on the
  upper trunk (the last on its tip) and a clump at each leaf stem's tip, smooth-unioned at half their radius
  into one mass; segments and clumps the hull always covers are skipped.
  Nothing is LOD-specific: the band rule is the LOD rule (thin parts erode away first, a tapered branch
  from its tip, the hull last; nothing appears with the LOD). Sites are 3D cells (planets come; `src/cell_sites.*`, any
  source, salted: `TREE_SITE_SALT` for the forest, the species is the site's variant): `cell_site` reads the coarse field
  (`SDFBase::ground_field(voxel, lod)`, `evaluate_cull`'s field with its gradient; at `site_lod` a lattice sample extended
  along its gradient) at the cell's centre, projects the cell's hashed voxel on that plane, keeps it if it lands in the
  cell's middle part (`SITE_INNER` 0.15 off each side across the normal: neighbours stay 0.3 cells apart with no
  neighbour pass), then snaps it onto the LOD-0 ground (`SNAP_STEPS` Newton steps on whole voxels, one owner per surface
  point); `surface_sites` tries only the cells an absolute octree walk keeps, one coarse reading per node centre, while
  its plane crosses the node (`SITE_SLACK` cells + `SITE_BEND` nodes): ~9 evaluations per site, 1 per empty box;
  unit-tested equal to the brute force on hills, a 60 degree slope and the 5 cm cascade. A tree grows along
  `SDFBase::up_at(base)` (+y on the flat world, the one source of "up" for placement; `UP_Y` keeps the records from
  before, pinned by a hash). Still column-based (flat debt): the scene's centre (`ground_y`) and the lake keep-out
  (`in_lake(x, z)`, the sources' `accept`); the zone is the `--tree_area` square over 4 half sides of height up and down. Hashing 3D cells moved
  every tree, the coarse-plane projection (feature/cell-sites) again: forest5cm is another forest than before. `--trees=N --tree_area=M` (default 0, 60 m) in bench and player; meant for 5 cm voxels
  (at 0.5 m only the trunk and hull remain). `get_debug_stats()["table"]["trees"]`: count, records per site.
  Past its parts' drops the forest is drawn by its moment blobs (below). **Band timing (moments,
  `src/instance_moments.*`):** every record has moments (`record_moments`: volume, centroid, 3x3 central second
  moment; analytic per library type via `TableDef::moments`, else its program on a 24^3 grid), an instance their sum
  per material (`span_moments`: a tree's bark and leaves). Its half thickness is the equal-covariance ellipsoid's
  smallest semi-axis. `set_band` drops it at the later of the first LOD whose cells no longer fit `resolve` times in
  that half thickness (`VoxelTerrain.archetype_resolve` / `--instance_resolve=F`, default 1) and the first LOD erosion
  at `rate` from LOD 0 can make it vanish by; its onset is the latest LOD from which that erosion still reaches its
  depth at the drop: whole while resolved, shrinking only in the LODs before its drop, `erode` cells per LOD at most
  (unit-tested). At 5 cm (30 trees, by volume): drawn whole 98 / 93 / 61% at LODs 1 / 3 / 5 (0% when every part
  eroded from LOD 1), drawn at all 97% at LOD 5 and 48% at LOD 6 (65% and 0%). The rate is
  `VoxelTerrain.archetype_erode` / `--instance_erode=F` (cells per LOD, default `ERODE_CELLS` = 1, bench and player;
  the record carries it as `erode`, the skirt follows it): a higher one moves onsets later.
  **Moment blobs (the far field, `src/moment_layers.*`):** what the parts a LOD no longer draws still weigh. For each
  LOD L, every part hands the share of its mass the LOD lost (its equal-covariance ellipsoid shrunk by its erosion,
  `MomentItem::lost_share`; all of it from its drop on: parts drawn + blobs = its mass at every LOD) to cells of
  2^(L + 3) voxels (8 cells of L) on an absolute grid, per material, with tent weights over the 2 x 2 x 2 cells around
  each centroid (a uniform cover reads a near-constant occupancy, a thin canopy stays thin); a material joins once a
  quarter of its mass is lost (lone twigs would come back fattened). Taking only the parts gone whole left the forest
  thinned out at LODs 4-6 (canopy columns 34 -> 26 of 48 at LOD 5; the unit test now holds 9/10 of LOD 2's). Each cell
  is a blob, `a max(1 - q, 0)^2` in its principal frame (the record's rotation), its spread the cell's covariance plus
  the LOD's box filter, its peak the mass-kept one or, higher, the cell's **cover**: each part's area tensor (its
  silhouette along n is n^T P n, pi r^2 for a sphere, `prepare_moment_item`), summed and read along the cell's thinnest
  axis, 3 A / (pi s_1 s_2), at most 1 (a canopy reads its cover, ~0.44, at every LOD 3-9 where keeping the mass faded
  0.36 -> 0.08; edges and far LODs included, where cells look round and the old sheet gain vanished the forest; a lone
  clump fades once the filter outgrows it), at most 2^(L - the LOD it lost half its mass at) times the mass-kept peak
  (`MomentJoin::cover_from`: seams step no more, the rock scatter's 3 | 4 the worst at 0.90 of its skirt); blobs
  fainter than `MOMENT_MATERIAL_ISO` with their neighbours are pruned. They are table records (`moment_blob` type,
  `Combine::ACCUM`, listed at their LOD only: `TableRecord::min_lod`), built by `SDFTable::add_moment_blobs` after the
  scene; the fold sums them into an occupancy phi and its epilogue (`moment_surface` / GLSL `arch_moments`, bit for
  bit, FrameSpike's table probe) draws `min(r - h phi, (MOMENT_ISO - phi) h / max(|grad phi|, MOMENT_ISO / 64))`: the
  ground lifted under thin mass, a surface where phi reaches `MOMENT_ISO` (0.25: an opaque canopy), the strongest
  blob's material from half of it. Each layer is listed at one LOD, so a L | L+1 seam steps: the samples a blob touches
  take `SDFTable::MOMENT_SKIRT_CELLS` (2.5) more skirt (`cascade_sample_sdf`'s `skirt_extra`, `SDFCascade::with_skirt` /
  `skirt_depth_at`; 0.93 of it on the rock scatter, the worst), and the locality margins take it whole when the scene
  places instances; a global skirt that deep cost +20% triangles in the forest (taller walls on every chunk). The
  discoverer's no-cull rule skips a blob below its LOD (`near_layer`: `min_lod`): its box would keep every finer node
  under a coarse canopy uncullable (+18% loaded chunks).
  A dense 5 cm forest keeps a canopy over most columns at LODs 6-8 (unit test); from 1,100 m the forest5cm square
  is a canopy instead of nothing (`bench.sh --preset=forest5cm --cam_pos=-50,320,750 --above_ground=1100 --pitch=-55
  --yaw=0 [--moments=0]`). The hand-off over every altitude: `tools/forest_lod_sheet.sh` (bench `--scenario=rise
  --altitudes=... --rise_shots=DIR`, 20 m to 13 km over the forest, shaded | LOD view, `sheet.png` + lists per step)
  and the unit test of the canopy columns at LODs 2-9. `VoxelTerrain.archetype_moments` / `--moments=0` (bench, player) leaves them out;
  `get_debug_stats()["table"]["moments"]`: items, blobs per LOD, build ms (`["trees"]["build_ms"]`: the forest). The
  build runs on the worker pool (`parallel_for.hpp`: each layer's members side by side, then each cell's sums and pruning;
  the forest's ground traces and trees per site, admitted in site order), the same bits: forest5cm_x10 (40,000 trees,
  2.2 M blobs) configures in 15 s, 93 s serial (blobs 89 -> 11 s, the pruning's map lookups were 2/3 of it; forest
  18 -> 2 s); the main thread still waits for it (async configure: next). **Edits** keep the blobs exact:
  `Archetype::MomentLayers` holds each cell's members (by record id), their partial sums per block of 256 ids and the
  pruning verdict; an edit moves its record between the cells its old and new tents touch, re-sums one block per cell
  and re-judges the blobs whose box meets a changed one (bounds are summed in cell order), so the table holds the same
  bits as a build of the edited records with the same joins (unit test: 6,829 blobs over 40 edits). The changed blobs go
  through the edit's snapshot like any record (`SDFTable::update_moment_blobs`); `edit_affects_chunk` re-meshes each at
  its LOD only (`EditRegion`). ~8 ms per edit on a 120-tree table in the debug build (0.17 without the blobs); the
  forest5cm edit scenario re-meshes 6% more chunks; validate / audit / discovery_audit stay ok. The joins and the first
  layer are frozen at the build: a material the build didn't have, or a part gone below its first layer, joins at the
  next `configure`. A blob that vanishes keeps its record id for its key (removed, reused when it comes back: edits
  don't eat the instance capacity). Coarse cells hold nearly every part: their member lists cost memory (~4 B per part
  per cell its tent touches) and an O(parts) insert / erase per edit (TODO: per-block member lists would bound it).
  **Biome features (`src/biome_features.*`):** lava lakes and floating islands, plain records like the trees (no
  change to the IR, the fold or the meshers). Placement is by biome cells: each kind tiles the plane with cells of
  640 m rounded to 2^k voxels, and a cell keeps one site in its middle half with the kind's chance, from
  `hash_cell` of the int64 cell (a site depends only on its cell: the same from any square or origin, unit-tested
  near 0, 2^32 and +-2^60). A lake is two records: a `basin` (a cylinder plus the rock's bumps, 16 m lattice)
  smooth-subtracted from the ground, then a hard-unioned `cylinder` slab whose top face is the flat lava, 4% of the
  radius below the lowest of 72 ground samples on the slab's rim. `Biome::place` tries `LAKE_ATTEMPTS` spots of the
  cell and gives up where the rim's ground spans over half the radius (`lakes_too_steep`). An island is one `island`
  record (a spherical cap on a hanging cone, smooth-unioned, plus bumps), its box bottom 25-70 m above the highest
  ground under it. Both follow the band rule like any record; the slab erodes away a few LODs before the basin (it
  can't be thicker without poking out of the shore), so a far lake is a shallow pit (0.15 r) whose scorched basin
  material shows. `--biomes=KM` (bench and player, `VoxelTerrain.biome_radius` in metres) places the cells within KM km of
  (SCENE_X, SCENE_Z) once, at configure, on the worker pool, before the scatter and the forest, which keep 1.5 radii
  out of the lakes (`SDFTable::in_lake`); streaming cells needs the index rebuilt off the main thread.
  `get_debug_stats()["table"]["biomes"]`: counts, `build_ms`, and up to `SITES_LISTED_MAX` sites (`kind`, x / y / z
  in metres, y the lava or the island's rim, radius). `--preset=biomes` looks at a lava lake with two islands behind it.
- **Materials (per-chunk palettes).** A record carries a global material (`TableRecord::material`, ids from
  `Material::Library`, `src/material_library.*`: built-ins ground, bark, leaves, lava, scorched rock, island rock;
  id 0 `GROUND` is the page shader's procedural slope / height colours; ids < 2^24). By default it is its type's
  (`DefLibrary::add(def, material)`, `Material::FROM_TYPE` resolved by `SDFTable::finish_record`: branch bark, crown
  and cone leaves, basin scorched rock, island island rock; the generic shapes (sphere, capsule, rock, cylinder, the
  shapes gallery) are `Material::INHERIT`, the lava slab sets lava), so runtime `add_instance` edits get theirs too.
  An `INHERIT` cutter colours nothing: the fold skips its win, so the sample keeps what it had (the dig tool's
  capsule carves a tree and the hole shows bark). On a union it resolves to ground (`Archetype::record_material`, in
  `SDFTable::finish_record`): an inheriting union would take whatever record won the sample before it in list order. It takes the list word's reserved
  palette index `ChunkPalette::INHERIT_INDEX` (511) and no palette entry. A corner stores a 9-bit index into its
  chunk's **palette** (`src/chunk_palette.*`: entry 0 ground, then the listed records' materials in list order, at
  most `ChunkPalette::MAX_ENTRIES` = 511; past that the rest draws as ground, `palette_truncated`), so ids are unbounded and a chunk only carries
  what it uses. `build_lists` builds the palettes on the worker pool and packs each entry's index above the record id
  in the list word (`SDFGpuTable::list_word`; `LIST_ID_BITS` = 32 - 9: **a table holds at most 2^23 = 8M instances**,
  removed ones included; a separate per-entry index would lift it). The fold keeps the palette index of
  the record that wins (`record_wins`: the operator's own comparison, so distances are untouched; a subtraction's
  carved face takes the cutter's material unless it inherits; CPU `fold_onto` == GPU `arch_fold`, FrameSpike's table probe), the cascade's
  fold variant's level 0 writes it per sample for chunks with a list (`sdf_material`, cascade binding 9), and the vertex
  takes the material of its cell's solid corner closest to the surface (`cascade_vertex_material`, ties in x + 2y +
  4z order; both meshers), except in a chunk on the ground block (direct) or without a list (legacy): material 0, no
  reads.
  `PalettePool` (`src/palette_pool.*`, a buddy allocator of 16-512-entry blocks in one RGBA16UI texture, 2 ids per
  texel, `GPUDirectMesher::Settings::palette_capacity` = 2M entries, 8 MB) holds the loaded chunks' palettes: placed
  at dispatch, its texels written by the emit pass (a texel per thread, `PalettePool::glsl_batch`), the block offset
  in each row header (texel 1 `.zw`, so compaction moves it with the rows), freed with the chunk (or at the end of
  the batch's results for the chunks that loaded nothing); a `{GROUND}` palette, or one that finds no room (`full`),
  uses block 0 and draws as ground. The page shader reads header -> palette -> id -> `material_tex`
  (`Material::Library::make_image`, owned by the mesher) for corners of index > 0, and replaces the ground colours with
  the material's albedo, roughness and emission (lava: `PATTERN_LAVA`'s crust; `material_shading`). **Blend:** corner A (the
  provoking vertex) reads the quad's other corners: a = its material, b = the first other one, plus a mask of the corners
  (B, C, D) that are not a, packed above b's id (flat `ivec2 v_materials`); the fragment interpolates the mask over the
  base quad's uv (no new interpolant) and shades the higher id of the pair where the weight towards it, jittered by a
  value noise of about a cell of the chunk's LOD, passes 1/2 (towards the higher id, so two quads holding the pair in
  either order agree on their edge). A third material inside one quad draws as b. **Cost:** a per-pixel choice of
  material costs ~1.3 ms of opaque pass and ~0.4 of pre-pass in the forest at 1080p (+6% render GPU), whatever its
  form (select, branch, two shadings, mix) and even when no quad holds two materials: it is the shader's compiled form,
  not executed work (feature/material-blend's bisection; 2 smooth vec4 of per-vertex parameters cost more, +2.2 ms).
  `--view=9` shows the ids, blended (V reaches it in the player). Legacy
  carries the id in `ARRAY_CUSTOM0`; the validator (half its sample from chunks with materials,
  `MeshValidator::pick_targets`) requires `material_mismatch == 0` on paired vertices, the auditor
  `bad_material == 0` (index past the palette) and the headers' blocks. Stats: `stats.json` `palettes` (direct) and
  `table.list_len_per_lod.*.palette_avg / palette_max`. **Bark** (`PATTERN_BARK`, `bark_grain` / `bark_plate` in the page
  shader): a 3D noise with no surface frame: a smooth grain stretched along world up (plates, V grooves ~11 cm apart), a
  sparse cell field of knots (2 x 2 x 2 nearest cells) bending it; colour only (a bump and a roughness following the
  plates cost ~1.3 ms of depth pre-pass in the forest at 1080p, which writes normals and roughness for SSAO / SSIL, for
  relief barely visible); it fades to the plates' mean past ~4 pixels per groove (metres per pixel from the projection).
  It is read at `bark_point`: the normal's horizontal direction at 0.45 m (~25 grooves around any trunk) and the height,
  so moving the surface along its normal (erosion, skirt push, the mesher's error) reads the same bark; read at the
  position, it jumped at LOD changes (70% of the pairs across one vs 27%, `tools/bark_seams.sh`). Trees share their bark at
  one height, thick trunks get wide plates, and its axis is world up (a near-horizontal branch gets grooves around it).
  The page shader's lattice noise hashes its cells with integers (`cell_hash`, `GLSL_PCG3D`): a float hash of
  coordinates in the thousands made every lattice plane a seam (the compiler reassociates `(i + 1) * k`, the hash
  amplifies the last bit), the bark's horizontal lines. **Grading:** the scenes' Environment has adjustments on and a 32^3 LUT
  (`res://grade_lut.png`, a Texture3D of 32 slices, written by `tools/make_lut.py`: lifted cool shadows, warm mids,
  +15% saturation on greens and browns); `--grade=0` (bench, player) turns it off. The scenes' old brightness /
  contrast / saturation values never applied (adjustments were off) and are gone. The ground's material per biome:
  **Biome materials** below. Next: materials inside one record (an IR change), image textures (triplanar).
- **Climate** (`src/climate.{hpp,cpp}`, `climate_modulator.cpp`, `climate_parse.cpp` / `climate_validate.cpp`,
  `climate_glsl.cpp`; `VoxelTerrain.climate_path` / `--climate=PATH`, bench and player; off by
  default): a climate is data, a JSON table (format above `parse()`, example `project/climate/example.json`, tuned for
  5 cm: at 0.5 m its x2 mountains exceed the cap and the table is refused) checked against
  `SDFCascade::climate_ground()`: up to `MAX_FIELDS` (24) fields, each a LINEAR4 cascade group (`Field::bands`) of any octaves in
  the ground's range (seeds `make_octave(field seed, k)`, keyed by k; a `from_m` .. `to_m` range of bands or multipliers
  keeps the octaves the LOD count has, each at the same amplitude, so one table covers the whole spectrum at 15 and 25
  LODs; a single wavelength outside is refused), plus linear terms `coef * (the ground or an earlier
  field)` applied after the cascade, in list order (the lapse is a ground term). Biomes (<= `MAX_BIOMES`, 16): regions
  over any fields, an indicator = product of edge smoothsteps; priority layering (w_b = rest * r_b, the last biome takes
  the rest) is a partition of unity for any regions without a division; dual numbers, every product flushed under 2^-64.
  `pick`: argmax, ties by `hash_cell` of the absolute voxel (the same from every LOD). **One climate:** every reader
  (the modulation, the pick, the terraces, the report) goes through `Climate::state` (field terms, then weights; GLSL `glsl_weights`),
  and what reaches it is the LOD rule: octave k reads the climate of its insertion level before that level's
  insertions, the pick and the report read level 0 (the report's pick drift shows the gap). No per-biome mean height: one
  lattice can't carry it without the coarser LODs missing it (a step no skirt hides); it needs its own design.
  **In the cascade** (`Climate::combined_spec`): the fields are `Spec::fields`, LINEAR4 groups carried with the ground
  (`Cascade::field_slot`: after the ground's 8 slots on the CPU, after its 4 channels on the GPU: `Spec::gpu_base`, `Spec::slot_of_gpu`).
  A band some biome multiplies is modulated (`Cascade::Modulation`, `Climate::Modulator`): at each point of its level,
  from the channels before that level's insertions (H(V_{m+1}): no band reads one of its own level), A = sum_b w_b amp_b
  and the band inserts A n, slopes n.s (A 2^-3) + n.w dA. So ground octave k follows the biome map of the fields'
  octaves coarser than k only: a region shapes the ground up to about its own size (a plain patch inside a mountain
  region keeps the region's big octaves), and flattening the continental octaves takes continental regions. The
  example's fields have km and continental octaves (to the top) and every biome multiplies the whole spectrum (raising
  stops at the 3.3 km cap): at 25 LODs, mountain slopes 4-5x the plain's on 26 and 210 km squares (a plain alone:
  slope 0.018, from 0.19 bare); the old example's few-octave multipliers moved slopes 0-45%, mountains +2%.
  `climate_report.sh`'s relief table (height std, mean slope per biome, `ground_height.png`) measures it. `fields_at` reads V_m of the combined spec, so it is a pure
  function of (lattice point, m). `SDFCascade` skirts take the largest amplitude any biome gives an octave
  (`Climate::max_amp`). On the GPU the climate code is generated from the table (`Modulator::glsl`, op for op) and
  `CascadePasses` compiles a ground-only variant and a fields variant (`level_shader(..., p_fields)`, the scratch planar:
  vec4 v of every slot together, so the ground's loads stay contiguous): a chunk of LOD L carries its fields at level m
  only where `Cascade::FieldRange::carried` (m >= the finest modulated ground level, L <= the coarsest; the CPU's ground
  queries follow it too, `fields_at` carries them everywhere), and a dispatch takes the fields variant where
  `FieldRange::batch` says (`record_level`'s `p_lods`, the batch's LOD range). Cost
  (CascadeSpike, 128-chunk batches, example climate, before the biome materials below): fields without multipliers 1.00x; with them 1.09x at LODs 0-4,
  1.00x at 20-24, 2.2-3.5x at LODs 8-16 (their own level is modulated: the weights run on every sample, +9-19 us per
  chunk, once per mesh). The bigger `Ch` (20 floats) is what made a single shader 1.4x everywhere (registers, not work).
  Forest5cm fly: steady +2%, tree placement +190 ms (the CPU traces carry the fields), the first run of a new table
  pays its shaders' compile (cold pipeline cache). A biome that multiplies the ground with no field still takes the
  modulated variant (`CascadeGLSL::field_range` keys on the modulation, not on the fields; set `4d`,
  `fieldless_test.json`, a unit test): the GPU drew the bare ground under a flat desert before. Refused: a field's
  multiplier without a ground one (it would never reach the mesh), two bands of one group on one level, multipliers on
  a non-LINEAR4 ground (checked once, in `validate`). Checks: `cascade_test.sh --sets=4,4f,4c` (G1 on the carried
  channels, G3 except at the ceiling, points), `discovery_audit.sh -- --voxel=0.05 --climate=... --discovery_compare=1`,
  `origin_parity.sh --voxel=0.05 -- ... --climate=...`, a validated / audited bench load.
- **Biome materials** (with any climate that has fields, `SDFCascade::has_ground_materials`): each biome is a ground material `biome:<name>` (`Climate::biome_material`:
  `PATTERN_GROUND`, the page shader's slope blend and contours in the biome's `"surface"` colours, sRGB `#rrggbb`
  `flat` / `steep` + `roughness` in the table, the procedural ground's without; its `debug_color` as the material's
  debug colour). Every chunk palette starts with the **base palette** (`SDFGpuTable::base_palette`: GROUND, then biome
  b at 1 + b; `ChunkPalette::build(base, ...)`), held by the palette pool's ground block (seeded at `rt_create`). The
  cascade's level 0 writes a material for **every** chunk: the fold's winner, else at a solid sample (air keeps 0: a
  vertex reads its cell's solid corners) `cascade_ground_material` = 1 + `Climate::pick` of the sample's fields
  (`Climate::pick_glsl`, `climate_glsl.cpp`, generated from the table op for op, ties by
  `hash_cell`; CPU `SDFCascade::ground_material`), so the chunk passes carry the fields at every level of every LOD
  (`CascadeGLSL::field_range`: point items keep the spec's FieldRange, the ground bits are the same). Both meshers read
  every chunk's materials when the base is longer than one (`ChunkPalette::glsl_base`); on the ground block a record's
  index past the base reads as ground. Unit-tested (box run == point query per sample); CascadeSpike G10
  (`--biomes`): GPU == CPU on every sample of every LOD (58.7 M, all 7 example biomes seen). Cost (128-chunk
  batches, the example's 3 fields, every ground octave modulated): the climate's passes 2.0x the bare cascade at LODs
  0-8, 3.3-3.9x at 12-24 (the weights run at every level; with multipliers on a few octaves it was 1.5-1.6x at 0-4
  and 20-24: 12 more channels carried everywhere); its fields alone 1.4-1.5x. Forest5cm, retuned vs the few-octave
  example (2 runs, GPU at 1.5 GHz): load +2%, SDF pass per batch +10-13%, steady render GPU +0%, fly p50 +11%
  (the climate vs none, earlier and throttled: steady render GPU +6%, SDF pass +19%). `--view=12` (B in the player, V cycles to it) paints each biome its
  debug colour. A field no edge, term or multiplier reads is carried anyway: leave it out of the table.
- **Shaped bands** (`src/climate_modulator.cpp`, `src/climate_shape.cpp`, GLSL `src/cascade_shaped_glsl.cpp`; the table keys above
  `Climate::parse`, `src/climate_parse.cpp`): libnoise's
  complex planet's operations as cascade data, every one optional per biome and band range, CPU and generated GLSL op
  for op. (1) A band's **shape** `c0 + cn n + cabs |n| + csq n^2` (`Climate::Shape`: plain, billow, ridged, square,
  scale / offset, or coeffs) is inserted without its mean (`Cascade::NOISE_ABS_MEAN` / `NOISE_SQ_MEAN`, measured: a
  coarser LOD dropping the band makes no per-biome step); `|n|` is `Cascade::smooth_abs`, `sqrt(n^2 + 0.01) - 0.1`
  (with the bare kink the Hermite subdivision rang along the lattice axes: ridged bands' gradient channels 21% off the
  values' finite differences against 2.5% plain, and a multifractal's state streaked). The `Modulation` returns a
  `Cascade::BandMod` per group: three amplitudes, the state, the warp, a feature mask. (2) **Multifractal** (`"multifractal":
  gain`): the octave is weighted by `clamp(gain (1 - c))`, c a band-less field group (`Modulation::state_group`, the
  table's fields + 1, so at most 23 fields) that the ground's band overwrites with the complement of its weighted
  uncentred signal (Musgrave's ridged multifractal: gain 2 + ridged; hybrid: 1 + plain + offset). (3) **Warp**
  (`"warp_m"` + three `"warp_fields"`): the band's noise is read at the point displaced by the fields' values (int64 +
  fraction, `WARP_LIMIT`), its gradient by the chain rule (the fields' gradients). Slope damping (Quilez, de Carpentier)
  and the swiss turbulence's warp by the ground's slope were built and dropped: their own gradient needs the ground's
  second derivatives, which LINEAR4 doesn't carry, and without it the finer levels' values streak (climate report:
  6-13% gradient error, visible lines). (4) **Height terms** (`"heights"`: `height_m * curve(field)`, libnoise's Curve)
  and per-biome **terraces** (libnoise's Terrace, blended by the weight) are added to the ground at each sample,
  after the cascade (`Climate::ground_value` / GLSL `climate_ground`, through `SDFCascade::ground_channels` on every
  path: chunk samples, point items, `ground_y`, the report); the fields are then carried at every level, point items
  too (`CascadeGLSL::field_range`), and the skirts take the terms' step (`Climate::ground_step`: the height fields'
  terms, twice that with terraces for their slope, plus the terrace blend's change with the weights, each terraced biome's
  weight step times a quarter of its step; unit test: the L | L+1 step stays within the skirt on the example and
  `shaped_test.json`). Refused: an `"offset"` without a multifractal (plain, billow and square bands are inserted without
  their mean, so only the state reads it), a curve the field's values never reach, a height term over the ground's
  amplitude cap. That is what
  lifts land per region (the per-biome mean height the plain cascade couldn't do). (5) **Cell noise** (`"noise":
  "cell"`): Worley F1 over 3 x 3 x 3 jittered cells (`ExactNoise::cell3` / `glsl_cell3`, FrameSpike's `cell3_k*`),
  centred by `CELL_MEAN`, read through height terms; creased (cones, cell borders), it measures 7.2% gradient error on
  25-100 m dunes, no streaks. A table without these keys is the old path bit for bit (unit
  test: `project/climate/plain.json`, the example before them, pins every slot, the SDF, the skirts and both level
  shaders' text). `project/climate/example.json` is now a world composed like libnoise's (continent curve, a twisted
  terrain type choosing ridged-multifractal mountains lifted inland, billow hills, terraced badlands, low plains,
  ridged river channels, three warp fields); `shaped_test.json` exercises every operation (CascadeSpike set `4s`; `4p`
  the plain table). The look target, a numpy port of libnoise's graph, and the gallery of the techniques are outside
  the repo (`~/Drive/perso/opti-voxel/terrain-research-2026-10-21/`). Checks: CascadeSpike 4c / 4p / 4s G1, G3, points
  and G10 (0 differing), FrameSpike, discovery audit (11 poses, compare) and origin parity on the example climate, a
  validated forest5cm load. Cost (128-chunk batches): the example's passes 4.2-4.5x the bare cascade at LODs 0-8,
  5.6-6.5x at 12-24 (the plain example 2.0-3.9x): 7 fields + the state carried, and level 0 computes the weights three
  times (modulation, height terms, pick: one shared evaluation is the obvious next step). Forest5cm vs the plain example
  (2 runs, GPU throttled 73% both sides): SDF pass per batch +17%, load +31%, steady render GPU +12% (+8% triangles:
  rougher ground), fly p50 out -1%, back -10%. `climate_report.sh` gained `ground_shaded.png` / `ground_normals.png`
  (hillshade from the heights and from the gradient channels), the per-biome gradient error, sea colours and
  `--center=U,V`.
- **Landforms** (`project/climate/landforms/*.json`, `docs/landforms.md`, `tools/landforms/`): mountain, flat desert,
  dunes, mangrove, island, cliff, valley and caldera tables, each matched against a real Copernicus GLO-30 heightmap.
  - `fetch_dem.py` downloads the public AWS tiles; the sites are in `sites.json`.
  - `climate_report.sh --lattice=M` samples the table on the same grid and writes `heights.f32` / `fields.f32` /
    `dhdy.f32` / `picks.u8` plus `heights.json`.
  - `compare.py` and `terrain_stats.py` give octave amplitudes, slope percentiles, hypsometry, curvature skew,
    anisotropy and the share of folded columns, plus a sheet.
  - `match.py` fits one multiplier per octave, `--hypso` a height remap, `--texture` a biome without its features.
  - `world.py` composes them into `project/climate/world.json` (25 LODs): each landform behind a region of temperature x
    humidity x wind, shared fields merged, the continental octaves (>= the axes' finest, 105 km) off in every table.
    Each region's core equals its own table, bit for bit (`test_climate_regions.cpp`); `climate_report.sh` lists a
    core `--spawn` per region.
  - The findings (which tool for which feature) are in the doc.
- **Stretched octaves** (`"ground_stretch"`, a band's `"stretch_y"`; `Band::stretch`, at most k − 3; the band table's
  `info.w`): through the base (`BaseSurface::stretched` / `stretch_gradient`), the noise is read at p − (1 − 2^-j) d n
  and its gradient's up component scaled by 2^−j. Flat: datum + (y − datum) / 2^j exactly (GLSL `base_stretched`),
  centred on the datum; sphere: the base sample's float offset (`base_stretched_at`, CPU and GLSL with FrameSpike's
  `stretched_sphere`; its GPU path needs the planet PR's regions). A table without stretch generates the same shader
  text (the `plain.json` pin).
  - Why: the ground is 3D, so moving up a column the surface meets other noise: amplitude a heightmap doesn't have,
    needles, and folds where dh/dy passes 1. On the Mont Blanc match, unstretched: 8.7 m at the 100 m octave against
    2.6 m real, 0.4% of slopes over 75°, spectrum score 0.40 (folds 0.05%). A stretch of 4 from 100 m up: 2.3 m, none,
    0.13.
- **Regions** (`"regions"`, `Climate::Region`, `Climate::edges_value` / `glsl_edges_value`, evaluated by the ground only: `ground_value` / `climate_ground`, over `Table::gated`):
  a named product of edges, for several landforms in one table picked by macro climate fields. A biome's `"region"` puts
  its edges first; a height term's multiplies the term (a curve required: `ground_step` adds the gate's step).
  `region_test.json` (CascadeSpike set `4r`, 11 field groups) equals `valley.json` bit for bit in its cold core. The
  field group cap is 24 (the GPU's arrays are sized per spec); every group is carried at every point. Details:
  `docs/landforms.md`, "Using a landform in a world".
- **The cascade only exists on lattice points.** `SDFCascade::evaluate(p, lod)` takes `frac == 0` and a voxel
  multiple of 2^lod only (ERR and 0 otherwise); the discoverer's corners and the validator's cell corners are
  such points, `get_sdf_at` snaps to the LOD0 voxel. There is no off-lattice evaluation and no GLSL
  `terrain_sdf`: vertex normals are the trilinear blend of the cell's 8 sample gradients
  (`CascadeGLSL::GLSL_VERTEX_GRADIENT`), in both meshers. A new cascade terrain keeps this rule.
- `GLSL_EXACT_NOISE` is spliced from fragments it shares with
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
  both channel sets and `4c` (LINEAR4 + the example climate at 5 cm; `4f` its fields without multipliers; `4r` the two-landform region table) (G1); the LOD relation on the GPU outputs (G3); the discoverer's point items vs
  `Cascade::point` + `SDFCascade::sdf`, 5,120 corners per LOD (`--points`, D1); the meshers' samples of a cascade
  with 600 scattered instances and the chunks' real lists vs `SDFCascade::sample_sdf` + skirt (`--table`, G9); the
  example climate's per-sample biome material vs `SDFCascade::ground_material`, every LOD (`--biomes`, G10); GPU timestamps of the passes for
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
  `BENCH_ROOT=<checkout>` benches another tree with this harness. `BENCH_RESOLUTION=1920x1080` sets the window
  (default 1280x720; the LOD follows the pixels, so 1080p draws ~25% more quads).
  `--profile=1`: a per-frame profile, `<run>/profile.json` and `profile.md` (`tools/profile_report.py <run>`):
  per phase the wall frame time, each `VoxelTerrain::_process` section, the GPU and render-thread CPU per
  renderer pass and mesher step, and what the frames over 16.7 ms spend against the others (see Gotchas for
  how the per-pass timestamps are obtained). It also tabulates, from `stats.json`, the stream per phase
  (`table.stream`: keys computed / evicted per kind and LOD in `computed_by_key`, worker busy, admit / poll / land /
  evict ms, `park_frames`, the gate's `stale_parked`, `destroy.handoff_frames`: frames a stalled parent stays drawn,
  {count, p50, p95, max}, cumulative) and `mesher.gpu_by_list_words` (GPU µs per batch by its list words, log2 bins).
  `_process` sections: `stream_poll`, and `gate_admit` / `build_lists` / `record_bytes` inside `mesher_dispatch`.
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
- Discoverer: the cull verdict of a node only depends on its id and the (fixed) SDF, so it is cached
  across scans (`discovery_cache`, default on); `--discovery_verify=1` recomputes every hit and
  ERRORs on a mismatch. The walk is breadth first, one wave per LOD: each wave's uncached nodes give
  their 8 corners, deduplicated on the voxel (~6 nodes share a corner), evaluated in one batch
  (`SDFBase::evaluate_batch`; `SDFCascade` spreads it over the `WorkerThreadPool` with the
  allocation-free `Cascade::point_into`). `VoxelTerrain.discoverer_mode` / `--discoverer=cpu|gpu` (bench
  and player): -1 auto (default, the same as 1), 0 CPU, 1 GPU. The GPU path,
  `DiscovererGPU` (`src/discoverer_gpu.*`), runs the cascade passes on point items
  (`CascadePasses::create_points`) on a **local RenderingDevice owned by the discoverer thread**: created
  at the top of `discovery_loop` and freed there before the thread exits; no other thread may touch it,
  and it does nothing on the main RD or the render thread. No local device (headless): one
  warning, CPU. Both paths give the same bits; `--discovery_compare=1` runs a fresh CPU scan for every
  scan and ERRORs on any difference (its time counts in the discoverer's latency stat): run `discovery_audit.sh -- --terrain=cascade --discovery_compare=1`
  (and `--lods=25`) after touching the discoverer, the cascade passes or `DiscovererGPU`.
  `stats.json["discoverer"]`: mode, scratch MB, `gpu_init_ms` (device + shader, before the first scan),
  compare counts, first and last scan (scan / cull / walk
  ms, corners, unique corners, dedup ratio, per wave GPU ms, batches, readback bytes). `--cam_pos=X,Y,Z --pitch=DEG --yaw=DEG` place the bench camera;
  `--camera_voxel=X,Y,Z` makes the scene (camera, lights, cam_pos) relative to that absolute voxel
  instead of the world origin.
- View-dependent LOD (`VoxelTerrain.lod_pixel_error_back` / `lod_view_margin_deg`, `--lod_error_back=F
  --lod_view_margin=DEG` in bench.gd and the main scene; off by default): a scan carries a `ScanView` (camera
  chunk, forward, the view cone = the frustum's diagonal half-FOV + margin, `back_scale` = back / front error - 1).
  The split test compares `ChunkMath::lod_distance`: the distance plus back_scale times the bounding sphere's
  gap to the cone (1-Lipschitz). In the cone it is the front rule bit for bit; far behind, the back error's.
  Face neighbours stay within one LOD up to a ratio of 4 (unit-tested at 720p; 8 gave jumps of 2, so
  `setup_lod` warns past 4). View memory (`lod_view_memory_m`, `--lod_view_memory=M`; default -1 = where LOD 2
  starts, 0 = off): the view carries up to 64 `ScanView::memories` (camera chunk, view axis, fade = distance from
  the camera past M), sorted by fade, and the gap is `ChunkMath::lod_cone_gap` = min over them of gap to their
  cone + fade: still 1-Lipschitz, still a pure function of the view (audit / compare re-scan it). `remember_view`
  adds the camera chunk's view unless one there is within `RESCAN_TURN_DEG` (8 degrees), replacing the most faded
  (least recently used among equals) when full; memory 0 keeps one view, replaced on every new chunk or turn. The
  remembered axis lags the camera by up to 8 degrees, so `lod_view_margin_deg` is at least that; besides camera chunk crossings, a rescan only follows an added memory. Turning in place: about 40 memories in the
  first turn, then no rescan and no re-mesh. Walking on: the views fade with distance (no timer). `bench.gd
  --scenario=turn --turns=N` and `--scenario=turn_walk --walk=M`; `discoverer.delta_entered_total /
  delta_left_total` in each phase's stats count the churn.
  Both meshers' queues are indexed by `ChunkMath::load_queue`: in the cone first, then by LOD, Morton order
  inside (batches stay compact for page culling). `--scenario=turn` (bench.gd) turns the camera 360 degrees after the load.
  Cost with memory 0 (measured in PR 34): a turn re-meshes the cone's edges, and that churn fragments pages like a fly does (on dev, a fly out and
  back already leaves +64% draw calls): after one full turn at error 1 | 4, the steady state is +26% render GPU
  vs error 1 without the split. A locality-aware compaction would close it.
- Destroy check (`VoxelState`, `src/chunk_memory.*`) is incremental: the discoverer publishes each scan
  with its delta (`ScanDelta`: entered / left, merged across the scans the main thread skipped), and
  VoxelState keeps per event the dirty chunks' nearest loaded ancestors, the stall count per loaded chunk
  (split rule) and `leaving` (loaded, not scanned); `update_destroy` only looks at `leaving`. It relies
  on the scan being an octree cut. `compute_destroy_full` is the oracle: `--destroy_verify=1` checks every
  delta and every destroy check against it and ERRORs on a difference (unit test: the randomized run in
  `test_chunk_memory.cpp`). Run a fly with it after touching `chunk_memory`, the discoverer's publish or
  the meshers' result path. `destroy.rescans` counts the rare pass over the dirty set (a stale result).
- Coverage (the split rule per sub-cube): a stalled chunk stays loaded until its last dirty descendant lands, but
  it stops drawing where the chunks replacing it already have. `VoxelState::stalled` (a `Stall` per stalled chunk)
  counts the dirty chunks anchored to it, in total (the split rule) and in each of its 4x4x4 sub-cubes
  (`subcube.hpp`, `ChunkMath::subcube_bits`; `--destroy_verify` and the randomized test check it against the full
  recompute); `hidden_mask` = the sub-cubes at 0. `push_coverage` (after compaction) hands the changed masks to the
  chunks' `ChunkReleaser` owner; the direct mesher writes them, at its next `process_queue`, into header texel 2 of
  the chunk's rows and flags them in texel 1's palette word (`PageFormat::COVERAGE_FLAG`: unflagged rows skip the
  fetch; `QuadPagePool::rt_set_hidden`: a staging texture + `texture_copy`, since `texture_update` writes whole
  layers), so kills, fresh pages and the emit pass clear it and compaction carries it. The page shader collapses a quad
  whose 4 corners all lie in hidden sub-cubes (overlap at the cut, never a gap). View 13 draws the hidden quads
  magenta and the stalled chunks' still drawn ones orange. `destroy.coverage_*` in the stats.
- Hole hunting: run the main scene with `godot --path project -- --marks=DIR`; H saves the camera
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
  bench only does it in the steady phase. Without a debugger Godot only captures the viewport's begin/end
  timestamps; with one attached its "visual" profiler exists, and turning it on
  (`EngineDebugger.profiler_enable("visual", true)`) captures a GPU + CPU timestamp per renderer pass.
  `bench.sh --profile=1` attaches `tools/debug_sink.py` (a listener that discards the stream) for that.
  The renderer's timestamp names start a section, the direct mesher's (`vx<batch>_p1`) end one
  (`gpu_interval_labels`). With a debugger attached a script error pauses the game.
  Godot's `Performance.TIME_PROCESS` is not this frame's process time (125 ms on 6 ms frames): use the
  terrain's own section timers.
- Killing a chunk's quads (`release_chunk`) zeroes its rows, and the page still draws them as degenerate
  quads until compaction moves it (pages under half full): a "removed" quad keeps its vertex and setup cost.
  The occlusion measurement (`feature/occlusion`, `--occlusion_hide`) removed quads this way, so it understated
  the gain.
- MultiMesh keeps ~190 B of CPU RAM per instance (interpolation arrays): never use one instance
  per quad. Pages use 64-quad groups for that reason.
- Never call `multimesh_set_buffer` / `multimesh_instance_*` / `multimesh_get_buffer` on a page: it
  creates a CPU cache that overwrites GPU-written data. `multimesh_get_*_rd_rid` from the main thread
  blocks: only call it on the render thread.
- `texture_update` of one texture several times in a frame, each followed by a `texture_copy` out of it: the copies
  read the last contents (measured: the coverage masks of a multi-round flush went to the wrong rows, the hidden quads
  mostly never showed). One staging texture per upload within a frame (`QuadPagePool::rt_set_hidden`).
- Many MB of `texture_get_data_async` in one frame crashes 4.6.3 (download staging overflow):
  throttle readbacks (validator/auditor do 2 pages per frame).
- An interrupted scons (session moved, killed shell) can leave 0-byte `.os` files that scons later
  treats as up to date: the `.so` links but fails to load (`undefined symbol: _ZTV15GPUDirectMesher`)
  and the bench loops on script errors. `find src -name "*.os" -size 0 -delete`, then rebuild.
- The SDF gradient at a vertex can be exactly 0: on a medial axis, where a cell symmetric about it puts
  its vertex (a capsule thinner than the cell on a lattice line, the scatter's pillars at LODs 1-4), and the
  nearby corner and Hermite gradients cancel by the same symmetry. `normalize()` of it is NaN and the
  vertex was dropped (broken quads, `nan_vertices`); both vertex passes take their normal from
  `vertex_normal` (`gpu_glsl.hpp`: the gradient, else the fallback, else straight up).
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
  counts them and deletes itself when the last one lands (`MeshValidator` / `PageAuditor::destroy`,
  `GPUDirectMesher::_rt_destroy`, which frees its RIDs at once and only defers the delete).
