# Archetype IR and instance table

The SDF program IR, the GPU table, the locality and band rules, the index and runtime edits.

- Archetype IR (`src/archetype_ir.*`, builders and fuzzer in `archetype_build.cpp`): an SDF as a linear SSA
  program of dual numbers (value + forward-mode gradient), the one source of its GLSL (`archetype_glsl.*`,
  straight-line `precise` code, one statement per instruction) and of its CPU evaluation (`Archetype::run`,
  an interpreter: archetypes are runtime data), bit-identical. On top of the exactness rules (`docs/gotchas.md`): no
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
  **Culled per cube:** in the level pass (`generate_fold_glsl(..., group_size)`, the cascade's fold variant) a
  workgroup is a 4x4x4 cube of level-0 samples (`CascadeGLSL::FOLD_CUBE`; for a chunk size that is
  not a multiple of it, the fold runs per sample, as before). The workgroup tests the list 64 words at a time against the cube's box (`arch_meets_box`:
  `sat_diff` of the cube's first and last samples vs the influence box), the words that pass go to a shared bit
  mask, and every sample walks the mask lowest bit first: the per-sample loop body (`arch_in_box`, the eval, the
  combine) runs on the survivors in list order, the same text as FrameSpike's per-sample loop. `sat_diff` keeps the
  order, so a record a sample would fold always passes its cube's test (unit test over random records, saturated
  samples included, CPU mirror `in_influence_cube`): the bits don't change. The template's early returns become an
  `sample_live` flag in the fold variant, so every invocation reaches the barriers.
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
