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
                 ├─ core: world_coord, chunk_math, page_format, subcube, latency_hist, gpu_cost_model
                 │        (pending: lod_structures, chunk_memory, instance_table, archetype_ir, cascade)
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
  writes straight into host-resident resources (today: MultiMesh pages via `call_on_render_thread`;
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

## Next steps, in order

1. **GLSL text pipeline to std::string** (one coherent commit): `exact_noise`'s `glsl_*`,
   `gpu_frame`'s `glsl_functions/glsl_int64`, `archetype_glsl.cpp`, `cascade_glsl.cpp`, and the
   splice sites in `frame_spike` / the meshers. Godot's `vformat` is printf-family, so call sites
   map 1:1 to `ov::format`; `String::replace` -> a small str helper. These files then join the
   purity list. The FNV text pins (test_exact_noise, test_archetype_ir) must not move. With
   exact_noise split, archetype_ir moves to `core_files` and cascade can be swept.
1b. **Cut the other two chains**: `ChunkOutput` holds engine RIDs inside the core's chunk map: split
   it into a core record (page, quads, palette, an opaque handle) and the engine-side owner, with a core
   log sink for `debug_log`; move the index's types (`DefRecord`, boxes, bands) out of
   `archetype_table.hpp` into a small core header.
1c. **The containers**: `ov::HashMap|HashSet` sit on node-based `std::unordered_*` where godot's were
   open addressing; `LODHashSet` is rebuilt every scan. If `bench_ab.sh` shows it, back them with a
   flat table.
2. **The GPU seam**: `sdf_base.hpp`'s `RenderingDevice*`/`RID` interface becomes the backend vtable
   (~15 ops: storage/uniform buffer create, texture create/copy, compute pipeline + list begin /
   bind / push / dispatch, buffer_update/get_data, timestamps, free). Godot-RD stays the first
   backend, benched with `tools/bench_ab.sh` to prove the indirection is free.
3. **The headless Dawn/WebGPU backend + the C ABI + a web demo**: the second consumer and the
   showcase. Readback is async on WebGPU (first call returns empty; poll across frames) — the
   directory readback is already async-shaped, the legacy full readback goes. No COOP/COEP headers
   on GitHub Pages: thread + dlink support needs the coi-serviceworker trick.
4. **The Bevy adapter**: bindings + the wgpu-device interop (Bevy's device handles into the vtable,
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
