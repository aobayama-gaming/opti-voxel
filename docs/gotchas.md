# Gotchas (all hit for real)

Engine, GPU and float traps that already cost a debugging session each.

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
- A RenderingDevice push block is set in whole 16 bytes (SPIR-V reflection rounds its size up): `RDDevice::dispatch`
  pads a smaller one with zeros, else the dispatch is dropped ("requires (16) bytes of push constant data").
- Dawn (Tint) does not keep the exact noise bit for bit: `ov_dawn --check` and `--frame-check` miss a few low bits
  of the fbm and perlin probes (open); everything else matches.
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
  only until its next frame begins: `buffer_get_data` flushes one, so read them first (`CascadePointsGPU`). On a local device
  `buffer_get_data` after `sync()` is a second round trip (~4 ms): queue `buffer_get_data_async` before `submit()`, its
  callback runs in the `sync()` (`CascadePointsGPU::_run_batch`, a `CallableCustom`; it falls back to the blocking read if not).
- Main-RD resources must be created/freed on the render thread; objects owning them delete
  themselves there (`destroy()` -> `_rt_destroy()`), after the render-thread work already queued
  for them (FIFO).
- Async readback callbacks (`texture_get_data_async`, `buffer_get_data_async`) are NOT skipped when
  their object is freed: they arrive frames later and run on freed memory (segfault in
  `MeshValidator::_rt_on_page` when quitting mid-validation). An object with readbacks in flight
  counts them and deletes itself when the last one lands (`MeshValidator` / `PageAuditor::destroy`,
  `GPUDirectMesher::_rt_destroy`, which frees its RIDs at once and only defers the delete).
