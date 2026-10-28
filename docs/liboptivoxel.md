# liboptivoxel: the engine-agnostic core

The plan: the Godot game stays Godot (inventory, animals, gameplay — where the engine is best);
opti-voxel itself becomes **an engine-agnostic library plus thin per-engine adapters** — the physics
engine model (Jolt: one C++ core, Godot / Unity / Rust via generated bindings). The public surface is
a small C API (`ov_world_create / ov_world_update(camera, budget) / ov_world_next_mesh_event`),
the deep integration lives behind one thin GPU vtable, and `page_format.hpp` is the interchange
format between engines. Rust consumers come from generated bindings (`cxx` / `bindgen`), not from
us writing Rust.

```
                 liboptivoxel (pure C++, zero engine includes; src/ov/ + the purity list)
                 ├─ core: world_coord, chunk_math, page_format, cascade, archetype IR / records / index,
                 │        chunk memory, the shader text; host services: ov::Executor, log, clock
                 ├─ mesher: GLSL -> SPIR-V (portable); Tint -> WGSL on web
                 ├─ GPU backends:  Godot-RD (today)  |  Dawn/WebGPU  |  wgpu interop (Bevy, zero-copy)
                 └─ C API (versioned from day one; the inner C++ API may churn behind it)
                        │
     ┌──────────────────┼─────────────────────┐
 Godot adapter      Bevy plugin           Web demo
 (GDExtension,      (cxx/bindgen —        (wasm, own
  what we have)      we write no Rust)     surface)
```

Two integration depths, both real:
- **Shallow** (part of the public API): the lib computes on its own device, hands back buffers. The
  just-works path.
- **Deep** (per-engine adapter, where the perf is): the host *lends* its device and the lib's compute
  writes straight into host-resident resources (today: Godot's MultiMesh pages through `PageHost`, the work posted to
  the render thread by `Device::post`;
  Bevy: a wgpu buffer; web: an emdawnwebgpu surface). The vtable is small because the design already
  is GPU-resident with a ~2 KB readback: `acquire / create_pages / schedule / describe_draws`.

## Done — PR #101 (`feature/core-std-types`)

The vocabulary sweep. `tools/check_core_purity.sh` (CI) has two lists:
- `core_files` compile **on their own with no godot include path** (`-fsyntax-only`, no godot-cpp
  `-I`): src/ov/, world_coord, subcube, latency_hist, chunk_math, page_format, gpu_cost_model. The list
  grows, never shrinks.
- `pending_files` have their own lines converted (grep-checked) but still reach godot through one
  header: lod_structures / chunk_memory through `chunk_output.hpp` (RIDs, `debug_log`),
  instance_table through `archetype_table.hpp` (GLSL, materials), archetype_ir through
  `exact_noise.hpp` (the GLSL text helpers). cascade is not swept yet (godot `ERR_*`, `String`,
  `Vector2i` in its own lines; the GLSL step).

Core vocabulary: `ov::ivec2|ivec3` / `Float3` / `ov::str` + `ov::format` / `ov::bytes` (+ `put_*` /
`get_*` at a slot pointer) / `ov::HashMap|HashSet` / the ov error sink. Conversions to engine types
are explicit at engine call sites (`godot_conv.hpp`: `godot_str`, `godot_v3`, `ov_f3`, `godot_bytes`,
`godot_dict`).

## Done — steps 1, 1b, 3 (`feature/de-godot-core`)

- **Shader text in `ov::str`.** Every GLSL generator (exact_noise, gpu_frame, gpu_glsl, cascade, base_surface,
  cascade_shaped_glsl, archetype_glsl, the climate's GLSL, the table's fold and library, the SDF's GLSL virtuals)
  returns `ov::str`; `String::replace` chains are `ov::replace_all` over (key, text) pairs in the same order
  (`ov/str.hpp`, godot's semantics). The meshers and FrameSpike convert once, at `set_stage_source`. cascade_glsl
  splits at the device: the text stays, `CascadePasses` moves to `cascade_passes`; the level shader reads the
  terrain through `CascadeSource` (the part of `SDFBase` it needs). The FNV text pins did not move (after a first
  commit making four GLSL comments ASCII: `String(const char *)` reads Latin-1, so their UTF-8 was double-encoded).
- **The chains cut.** `ChunkOutput` holds no engine resource (the legacy mesher keeps its mesh and instance under the
  output's handle, as a `ChunkReleaser`); the table's records are `archetype_record.hpp` and the index places a record
  by its finished box, without the library.
- **One executor** (`ov/executor.hpp`): tasks (submit / done / wait, never on the caller), `parallel_for`, and
  `WorkerLoop` workers (enter, a step per wake, exit, on one thread). `GodotExecutor` (WorkerThreadPool + threads) is
  installed by register_types; the default is inline (`pump()` once per frame), the threads=no web build's.
- **Host services**: `ov/log.hpp` (`OV_LOG`, the level `DebugLog::level` aliases) and `ov/clock.hpp` (`now_usec`,
  Godot's ticks installed).

`pending_files` is empty: every file of `src/` (132) compiles with no godot include path.

## Next steps, in order

1. **The GPU seam** (done): `ov::gpu::Device` (`gpu_device.hpp`: buffers, compute pipelines from GLSL,
   2D and 3D textures, separate sampled textures and a linear sampler, bind sets, dispatch with push constants,
   timestamps, readbacks, `post`), Godot's `RDDevice` and Dawn's
   `DawnDevice`; the drawing behind `PageHost` / `MeshHost`. Then (done) the frame loop into `ov::World`
   (`VoxelTerrain` is a property map over its settings) and the spikes into core checks any device runs
   (`FrameCheck`, `CascadeCheck`, `QuadPoolCheck`; `ov_dawn --frame-check | --cascade-check`).
2. **The headless Dawn/WebGPU backend + the C ABI + a web demo**: the second consumer and the
   showcase. Readback is async on WebGPU (first call returns empty; poll across frames) — the
   directory readback is already async-shaped, the legacy full readback goes. No COOP/COEP headers
   on GitHub Pages: thread + dlink support needs the coi-serviceworker trick.
3. **The Bevy adapter**: bindings + the wgpu-device interop (Bevy's device handles into the vtable,
   zero-copy); no Rust written by us.

Per-PR rules: every PR keeps the gates green (unit, parity, frame when world_coord/gpu_frame/
chunk_math/exact_noise/SDF move, discovery when the discoverer/SDF move); perf-relevant PRs carry
the `bench_ab.sh` before/after table on `--preset=forest5cm --scenario=fly`; the purity list only
grows. The spike steps open a real window (no headless RD) — warn before running.

## Gotchas hit on the way (recorded so they stay hit only once)

- An encoder called with the batch index over a one-element staging buffer writes `(count-1) * size`
  bytes past it — heap corruption that shows up as `free(): invalid size` / garbage RIDs / SIGSEGV at
  pseudo-random points, with unit tests green. The fixed pattern: encoders write one slot at a pointer
  (`encode_chunk_input(uint8_t *)`, `encode_record(uint8_t *)`), the caller computes the slot; no staging.
- std containers iterate in another order than godot's: no reader may depend on iteration order — readers sort by id, counters are order-independent. Doctest
  assertion *counts* can shift with iteration order without any result changing.
- Godot's `vformat` is the printf-family `%` operator (`String % Array`): `%s` renders any Variant;
  the shim mirrors that, including `%s` of a float (shortest round-trip text).
- `PackedByteArray::encode_*` is little-endian explicit (marshalls); on the little-endian hosts the
  project runs on, `ov::bytes`' host-order memcpy is byte-identical.
- The laptop GPU is power-capped under load: shader compile times swing ~20x between runs
  (7.1 s vs 0.34 s for the same fuzz set). A "hang" at 3% CPU waiting on the driver is thermal
  state, not a deadlock; a frame-test timeout needs a fresh look at clocks before a code suspect.
