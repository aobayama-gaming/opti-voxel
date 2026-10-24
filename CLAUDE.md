# opti-voxel

Godot 4.6 GDExtension (godot-cpp, 4.4 API) for an LOD voxel terrain meshed with surface nets.
Engine binary: `~/.local/bin/godot`. Build: `scons platform=linux target=template_debug -j$(nproc)`
(installs into `project/bin/linux/`). `src/` builds with `-Wall -Wextra -Werror`: `werror=no` only to
iterate past a warning, never to commit one.

This file holds the rules. The design of each subsystem lives in `docs/` (index below): read the doc of
what you touch before changing it, and update it in the same PR.

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
  **Perf target:** `--preset=forest5cm` with `--scenario=fly`, 60 fps while moving at error 2 (stretch: 1),
  reported next to the default bench. Baselines and where the frame goes: `docs/perf.md`.
  Never build while a bench runs (thermals).
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
- The core is engine-free (the liboptivoxel plan, `docs/liboptivoxel.md`). `tools/check_core_purity.sh`
  (CI) compiles each of its `core_files` alone with no godot include path, and grep-checks the
  `pending_files` (own lines converted, godot still reached through a named header). A core file uses
  only core types (`ov::ivec2|ivec3` / `Float3` / `ov::str` + `ov::format` / `ov::bytes` /
  `ov::HashMap|HashSet` / the `ov` error sink and `OV_ERR_*`) and `std::` (no godot `MIN` / `ERR_*`);
  conversions to engine types happen at engine call sites (`godot_conv.hpp`), never inside the core.
  The GLSL text pipeline (exact_noise's `glsl_*`, `gpu_frame`'s `glsl_functions`, the splice files)
  still speaks godot String on purpose: it converts as one unit.
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
- A PR that changes a subsystem's behaviour updates its doc in `docs/`. This file only gets a rule
  every change must know; design notes, measurements and flag lists go in the docs.

## Invariants (details in the docs)
- No float ever holds an absolute coordinate. Chunk ids and voxels are int64 (`world_coord.hpp`), the
  GPU gets low words plus high words (`gpu_frame.hpp`), metres only come from `ChunkMath`'s render
  helpers. Ids reach the GPU only through `GPUFrame::encode_chunk_input` / `WorldCoord::lo32_i`
  (`encode_s32`, `Vector3i` and `PackedInt32Array` truncate silently). `docs/integer-frame.md`.
- Every terrain is a function of the absolute position: the same point gives the same bits from any
  origin, any chunk, any LOD where they share samples. The cascade only exists on lattice points.
- CPU mirrors of GPU float math are op for op identical in float32 (no `mix` / `dot` / `normalize` /
  `sqrt` / division in GLSL, no denormals, `precise` everywhere, constants as bits): `docs/gotchas.md`.
- One terrain, one instance path: a new terrain is a cascade preset, a new feature a library type of
  the table. Don't add a second way to make either (`docs/terrain.md`).
- Main-RD resources are created and freed on the render thread; async readbacks outlive their object
  (count them, delete on the last). `docs/gotchas.md`.

## Docs (read the one you touch)
| Doc | Covers | Source |
|---|---|---|
| `docs/pipelines.md` | GPU Legacy (oracle) and GPU Direct meshers, page layout, GPU budget and cost model, compaction, dual contouring | `GPU_*mesher*`, `quad_page_pool*`, `page_format.hpp`, `gpu_cost_model.hpp`, `gpu_glsl.hpp` |
| `docs/integer-frame.md` | int64 ids, `WorldPos`, GPU words, origin, LOD range and clip planes, exact noise, Morton keys | `world_coord.hpp`, `gpu_frame.hpp`, `chunk_math*`, `exact_noise*` |
| `docs/terrain.md` | presets and `--terrain`, the table folded into the cascade samples, the base surface (flat / sphere) | `sdf_cascade*`, `sdf_table*`, `base_surface*` |
| `docs/cascade.md` | Hermite cascade, CPU and GPU passes, lattice-only evaluation | `cascade*` |
| `docs/archetypes.md` | archetype IR (adding an op), library types, locality and band rules, the index, runtime edits | `archetype_*`, `instance_table*` |
| `docs/instance-stream.md` | scatter sources, keys, the stream and its wiring | `cell_scatter*`, `scatter_source*`, `instance_stream*` |
| `docs/trees.md` | trees, moment blobs (far field), biome features (lava lakes, floating islands) | `tree_archetype*`, `moment_layers*`, `biome_features*` |
| `docs/materials.md` | per-chunk palettes, biome ground materials | `material_library*`, page shader |
| `docs/look.md` | sky, fog, clouds, surface detail, wind | scenes, page shader |
| `docs/climate.md` | climate fields and biomes, shaped bands, stretched octaves, regions | `climate*`, `cascade_shaped_glsl*` |
| `docs/landforms.md` | landform tables and matching real heightmaps | `project/climate/landforms/`, `tools/landforms/` |
| `docs/discovery-lod.md` | discoverer (CPU / GPU, cache), view-dependent LOD and view memory, destroy check, coverage | `chunk_discoverer*`, `local_points_gpu*`, `chunk_memory*`, `subcube.hpp` |
| `docs/testing.md` | every check tool and bench flag in detail, debug views, hole hunting | `tools/`, `project/debug/` |
| `docs/perf.md` | perf target, baselines, where the frame goes | |
| `docs/gotchas.md` | engine, GPU, build and float traps already hit | |
| `docs/forest-1m-profile.md` | the 1M-tree load profile and its ranked causes | |

## Verify changes (do this, don't eyeball)
Details and flags: `docs/testing.md`.
- `tools/unit_test.sh [--no-build] [filter]`: doctest unit tests in headless Godot, seconds, no GPU.
- `tools/format.sh [--check] [files]`: clang-format on `src/`.
- `tools/check.sh [--no-build] [--spike] [--discovery] [--keep-going]`: the correctness gate (~2 min),
  run before every PR: build, unit tests, a direct load with `--validate=64 --audit=1
  --discovery_audit=1`, origin parity, frame test, and the spike / discovery audit when their sources changed.
- CI runs format, the `-Werror` build and the unit tests; everything that needs a GPU stays local.
- `tools/bench.sh` runs the bench scene (`--scenario=load|fly|turn|edit|approach`, `--preset=`, `--profile=1`);
  `tools/bench_ab.sh` / `bench_compare.sh` give perf tables. Correctness runs never use `--cool=1`.
- `bench.sh --trace=1` maps the engine: `<run>/engine_structure.html` (`tools/engine_structure.py`: how it is built,
  which stage runs inside which on each thread, the hand-offs, each stage's code), `<run>/engine_map.html`
  (`tools/engine_map.py`: which thread runs what, ms per frame, hand-offs between threads) and `trace.json` (the timeline, ui.perfetto.dev). New threaded or GPU
  work gets an `OV_TRACE` span and its hand-offs a `Trace::flow` (`src/trace.hpp`).

| After touching | Run |
|---|---|
| `cascade*`, `exact_noise*`, `gpu_frame.hpp`, `sdf_cascade*`, `cascade_points_gpu*`, `local_points_gpu*` | `tools/cascade_test.sh` (needs a window) |
| `cell_sites*`, `field_gpu*`, `scatter_source*` walks | a 1M load with `--walk_verify=1` (`--preset=forest5cm_1m`), forest5cm `--walk_gpu=0` vs `1` chunk digests |
| `world_coord.hpp`, `gpu_frame.hpp`, `chunk_math`, `exact_noise*`, the SDF, a new noise or SDF term | `tools/frame_test.sh` (+ an entry in `make_probes()`) |
| the discoverer, the SDF, the cascade passes | `tools/discovery_audit.sh -- --terrain=cascade --discovery_compare=1` (and `--lods=25`) |
| `chunk_memory`, the discoverer's publish, the meshers' result path | a fly with `--destroy_verify=1` |
| `QuadPagePool` | `res://debug/spike.tscn` (`check.sh --spike`) |
| the bark, `value_noise`, the page shader's normals | `tools/bark_seams.sh` |
| a climate table | `tools/climate_report.sh` before any GPU run |
| the archetype IR | the list in `docs/archetypes.md` ("Adding an op") |

## Gotchas index (`docs/gotchas.md`)
Laptop GPU power cap and thermals; locked screen stops readbacks; "primitives drawn" counts full pages;
render-thread abort at exit and leaked RIDs with `--render-thread safe`; profiler timestamps need a
debugger; killed quads still draw; MultiMesh CPU caches; `texture_update` + `texture_copy` in one frame;
readback staging overflow; 0-byte `.os` files after an interrupted scons; zero gradients (NaN normals);
SDF samples of exactly 0 (side rule); no `#define` in `shader_set_code`; `texture_rd_create` formats;
compaction vs page ranges; shared `Array`s; int64 truncation; JSON doubles; float32 bit-exactness;
distance estimates under `max()`; GPU shader costs; local RenderingDevice windows and timestamps;
render-thread ownership; readbacks after free.
