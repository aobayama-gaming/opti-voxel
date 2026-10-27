# Discoverer, view-dependent LOD, destroy and coverage

Which chunks are scanned, loaded and dropped. Run `tools/discovery_audit.sh` after touching the discoverer or the SDF.

- Discoverer: the cull verdict of a node only depends on its id and the (fixed) SDF, so it is cached
  across scans (`discovery_cache`, default on); `--discovery_verify=1` recomputes every hit and
  ERRORs on a mismatch. The walk is breadth first, one wave per LOD: each wave's uncached nodes give
  their 8 corners, deduplicated on the voxel (~6 nodes share a corner), evaluated in one batch
  (`SDFBase::evaluate_batch`; `SDFCascade` spreads it over the `WorkerThreadPool` with the
  allocation-free `Cascade::point_into`). `VoxelTerrain.discoverer_mode` / `--discoverer=cpu|gpu` (bench
  and player): -1 auto (default, the same as 1), 0 CPU, 1 GPU. The GPU path,
  `LocalPointsGPU` (`src/local_points_gpu.*`, shared with the site walks' `FieldGPU`), runs the cascade passes on point
  items (`CascadePasses::create_points`, `CascadePointsGPU::cull_values`: the unit register read back, metres on the CPU
  with `SDFCascade::to_metres`) on a **local device (`ov::gpu::create_local_device`) owned by the discoverer thread**: created
  at the top of `discovery_loop` and freed there before the thread exits; no other thread may touch it,
  and it does nothing on the main RD or the render thread. No local device (headless): one
  warning, CPU. Both paths give the same bits; `--discovery_compare=1` runs a fresh CPU scan for every
  scan and ERRORs on any difference (its time counts in the discoverer's latency stat): run `discovery_audit.sh -- --terrain=cascade --discovery_compare=1`
  (and `--lods=25`) after touching the discoverer, the cascade passes or `CascadePointsGPU`.
  `stats.json["discoverer"]`: mode, scratch MB, `gpu_init_ms` (device + shader, before the first scan),
  compare counts, first and last scan (scan / cull / walk
  ms, corners, unique corners, dedup ratio, per wave GPU ms, batches, readback bytes). `--cam_pos=X,Y,Z --pitch=DEG --yaw=DEG` place the bench camera;
  `--camera_voxel=X,Y,Z` makes the scene (camera, lights, cam_pos) relative to that absolute voxel
  instead of the world origin.
- Error per LOD (`VoxelTerrain.lod_pixel_errors`, `World::Settings::lod_pixel_errors`, `--lod_errors=4,4,2,1,1` in bench.gd
  and the player; empty by default): entry L replaces `lod_pixel_error` in LOD L's distance (0 or past the list:
  `lod_pixel_error`). LOD L's band runs from LOD L-1's distance to its own, so a finer error than the LOD below widens it
  (more of its chunks) and a coarser one narrows it: at 2x coarser it vanishes (`setup_lod` warns). On a ground, a band
  holds about (1/e_L^2 - 1/(4 e_{L-1}^2)) / (3/16) times the chunks of the uniform error-2 band. The back ratio
  (`lod_pixel_error_back / lod_pixel_error`) applies on top. Check a new profile with `--discovery_audit=1`.
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
