# Meshing pipelines (`VoxelTerrain.pipeline_mode`)

The two meshers, the GPU budget and dual contouring.

Both are core code on `ov::gpu::Device` (`gpu_device.hpp`); what draws their output is the adapter's (`PageHost`,
`MeshHost`: Godot's in `adapter/godot/src/godot_page_host.*` and `godot_mesh_host.*`).

- `1` GPU Legacy (`GPUMesher`): compute on a local device, full readback, CPU
  triangulation, one mesh + instance per chunk (`MeshHost`). Kept as the correctness oracle and perf baseline.
- `2` GPU Direct (`GPUDirectMesher`, default in `main_scene`): compute on the main device (Godot's main
  RenderingDevice: its work posted to the render thread, `Device::post`), quads written straight into `QuadPagePool`
  pages, which the host draws (Godot: MultiMesh instances of 64-quad groups, the page shader in
  `godot_page_host.cpp`) pulling their corners from a GPU-written RGBA16UI
  texture: one row per group, a header with the owning chunk id, then 8 B per corner (chunk-local
  13-bit fixed point, 1/256 cell + a 9-bit palette index + 16-bit octahedral normal, at most ~1 degree off).
  Layout, decoders and the GLSL of both (constants substituted, no literal elsewhere): `src/page_format.hpp`. Only a ~2 KB per-batch directory comes back to the CPU (asynchronously) for bookkeeping.
  Shared GLSL: `src/gpu_glsl.hpp`, direct passes: `src/gpu_direct_glsl.hpp`.
  Chunks start on a 64-quad group and a texture row is one group, so a chunk is whole rows.
  GPU budget (`VoxelTerrain.direct_gpu_budget_ms`, `--mesher_budget=MS` in bench and player; 0 = off, 4 in
  `main_scene_5cm`): a frame dispatches mesh chunks until their estimated GPU time reaches the budget, cutting
  a batch short. The estimate is `GPUCostModel` (`src/gpu_cost_model.hpp`): µs per chunk per LOD plus µs per list
  word (seeded 1.0, the 1M fit), corrected from each batch's timestamps (measured / estimated ~1.05 on the forest; a
  LOD-3 forest chunk costs 200-300 µs, a treeless one ~15). Before its list a chunk's words are its keys' records
  (`SDFBase::admit`'s bound, through `ChunkGate::next`) times `list_share[lod]` (its LOD's streamed list words over their
  bounds, `build_lists`' `r_streamed`) plus `words_per_chunk[lod]` (the scene's and the edits' words); a LOD's first list
  replaces both seeds (before it, the nearest LOD's, the whole bound when none); a batch where a LOD takes its first
  measure moves only that LOD (clamped to `MAX_SCALE` when the batch has words); stats `cost_us_per_word`,
  `cost_list_share`, `cost_words_per_chunk`, `list_bound_total`. Forest fly at error 2: frame p50 27.8 -> 11.9 ms, at the price of a slower first load
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
  vanishes, its normal is the mean of the Hermite normals (`vertex_normal`, `docs/gotchas.md`). The Hermite data must agree with the
  crossings: on an SDF with detail finer than the cell vertices jitter (the archetype IR's band rule drops such noise).
