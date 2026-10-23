# Instance sources, keys and stream

`CellScatter` sources, `ScatterSource` keys, `InstanceStream` and how `SDFTable` wires them.

- **Sources (`src/cell_scatter.*`):** the scatter and the forest are two `CellScatter`s {salt, zone, side, ground, up,
  accept, generate}: a site's records are a pure function of its cell, so two halves of a zone give the whole
  (unit-tested). `SDFTable::get_sources()`, streamed (below; stats under `get_debug_stats()["table"]["scatter" | "trees"]`:
  the profile, and `["stream"]`), or with `--stream=0` (`VoxelTerrain.archetype_stream`, bench and player) placed whole in
  the table at configure by `add_source` (count, records per site, moments per material; the reference, and the tests that
  edit their records by id). Biome features and the shapes gallery stay table records (whole-build blobs). `SourceProfile`
  reads `SAMPLES` (256) hashed sites per class at the origin (`CellScatter::classes`: a site's class is its variant
  modulo it, the forest's species; class c sampled with hashed variants of class c, one class bit for bit as before): reach, `lod_end`, k / margin max (times
  `REACH_SAFETY` 1.25), joins and records per site over every sample, the mass lost per LOD per class (`lost_of(class,
  lod)`: what a far proxy of a site of that class stands for; `blob_reach` takes the widest class's); unit-tested to bound
  every record of the forest and the scatter (`covers`). A source with a `profile_key` (the forest's: voxel size, species
  count, type ids) takes its profile from a process-wide registry keyed by it plus the salt, finish, layer params and LOD
  count: every forest of a voxel size shares one, whatever its zone or density (stats `profile_classes`,
  `profile_shared`).
  **Keys (`src/scatter_source.*`):** `ScatterSource` cuts a source into keys `{source, PARTS | BLOBS | SITES | PROXIES,
  lod, tile}`, each a pure function of itself: `keys(box, lod)` (conservative, unit-tested against every tile's records),
  `inputs(key)`, `content(key, inputs)`, `ground_reach(lod)`. SITES and PROXIES hold a walk and no records
  (`SourceContent::sites`, never listed, kept resident: the walk is ~80% of a part tile's CPU): `SITES` (lod 0) on the
  LOD-0 part-tile grid (8 source cells), `PROXIES L` on far layer L's blob grid (the proxies' walk of its own tile in
  `proxy_zone`, from `proxy_level`: an unclipped region's octree level; a far blob tile reads the tiles its `proxy_region`
  meets and sorts their sites in it by cell: the region's walk from that level by construction, `surface_sites`'
  `p_min_level`, unit-tested on the cascade too; the old blobs bit for bit where the zone doesn't clip the region). A
  PARTS tile of LOD L (side doubled until it holds 2 chunks) grows the sites of its box: at LOD 0 those of its `SITES`
  tile (its one input: the near layers read it), coarser the site tiles of its cube that can hold a site, merged in cell
  order. Every walk of a source starts at its zone's level (`CellScatter::sites`: `surface_sites`' `p_min_level` =
  `walk_level(zone, side)`), so a box keeps exactly the sites the zone's walk keeps in it (a box's own level is coarser for
  a wider box and may prune a cell a finer start keeps); a site tile is a node of that walk (`site_level`: the LOD-0 tile
  side over the cell side), so the walk stopped there (`CellScatter::site_tiles`, `surface_nodes`: a few coarse readings,
  in the key's expansion job) names the tiles a coarse tile asks for, and a tile it drops walks to no site. The 1M
  forest's LOD 6-7 tiles asked 4,096 site tiles each as jobs before (758k jobs, 98 s of workers); one walk of the coarse
  box per tile was measured slower still (145 s: every LOD re-walked and re-snapped its sites, serially in one job). Every record at LOD 0, those listed at L coarser, grown from the forest's
  `hull_only_lod` hull-only (Trees, below); the record pyramid (LOD-0 tiles' groups merged in cell order, each filtered at
  L) is the unit tests' oracle: bit-identical tiles and lists. A LOD with the
  LOD-0 side (1-3 at 5 cm) has no PARTS key: `keys()` emits `part_key_lod(L)`'s and the chunk's `meeting` filters at L.
  A BLOBS tile (8 layer cells, at least a part tile and 2 chunks) builds the layers on the items within
  `MARGIN_CELLS` (`PRUNE_CELLS` + 1) cells and judges its own cells (`MomentLayerParams::judged`): on part layers the
  LOD-0 tiles' records (the whole build's blobs bit for bit, unit test), above them proxies (unsnapped `surface_sites`,
  2 per cell side, the profile's lost moments: mass within 10%). Part layers are those under `Setup::part_cell_sides`
  source cells: 3 by default (the whole build's blobs), `VoxelTerrain.archetype_stream_part_sides` /
  `--stream_part_sides=N` for the streamed sources (default 1: proxies from the source cell on; the 2 switched layers:
  mass within 20%, peak within 40%, 45% on a layer losing bark only (layer 4 at 41%: peaks far under the iso), unit test;
  3 = the exact layers). The stream keeps three LRU chains: the evictable
  keys with records (slots, the entries cap `ENTRIES_MAX`), the walks (`WALKS_MAX` 2^20, `Settings::walks`: ~450 B per
  walk + 56 per site, the 1M zone's ~500k stay resident; `walks_ready`) and those holding input records (not placed,
  count > 0: the inputs cap pops its oldest in O(1), the walks and placed keys never relinked); `computed_by_key` names
  `sites 0` and `proxies L`. An air site tile's walk is one evaluation (unit test: on a LOD-7 footprint on the cascade,
  89% of the tiles, 3% of the evaluations; the ground tiles ~330 each), so no emptiness test runs before it.
  **Stream (`src/instance_stream.*`):** `InstanceStream` holds the keys asked for in one table (empty ones
  too) with an intrusive LRU: `admit(chunk, owner, tag, rank)` answers READY (every key placed), PARKED (its keys pinned;
  `take_ready(owner)` hands it back, in park order, once they are) or LATER (a key finds no slot: stop popping). `admit` creates the chunk's own keys only (`admit_keys_created_max`): a key's
  first job, on a drain task, lists its inputs (`keys_expanded`) and `expand()` asks for them at its rank under the lock, the
  key running again once they land (`expand_ms_total`; a key `set_tag` dropped while its job listed them is erased then;
  `HeldExecutor::steps` in the tests stops a drain after N jobs). Keys are computed by drain
  tasks on Godot's pool at low priority (`StreamExecutor`, run by `poll()` in the tests; `low_priority_thread_ratio` 0.5), a
  key's inputs handed to it as shared pointers; the task lands each key under the stream's lock (its dependents start at once
  on the same task: landing on the main thread, 2 ms a frame, held forest5cm's 135k keys ~900 frames) and places the listed
  ones in slots past the table's capacity (`chunk_list` reads under a shared lock), logged per placement (`uploads_since(seq)`; a reader past the log gets every placed key). Eviction:
  oldest first, never pinned or asked for in the last `GRACE_FRAMES` (16). `point_records`: resident keys, else computed and
  dropped. Jobs run by (rank, order asked) (`jobs`: a map under `job_mutex`, a drain task pops `begin()`): `ChunkGate`
  passes re-meshes 0, a first mesh 1 + its `load_queue` (the mesher's pop order), audits `AUDIT_RANK`; a key's rank is
  the best of its requesters', inherited by its inputs not landed, a queued job asked better is re-keyed (`jobs_reranked`).
  `set_tag(owner, tag, rank, kept)` (`ChunkGate::next_scan(scan, load_queue)`, the CHUNK owner only): the chunks parked
  under an older scan that the new one holds are retagged in place at their new rank (a better one re-keys their queued
  jobs; `ChunkGate::keeps`: `send_chunk` doesn't queue them again), the others are forgotten, unpinned
  (`stale_handed_back`), and the keys then needed by no pin or dependent and not asked for this frame are erased if their
  job hasn't started (`keys_dropped`, of which `jobs_dropped` were queued; a dropped waiting key leaves its inputs'
  `dependents`).
  `PENDING_MAX` 2^14 (2^12 under the FIFO) caps the keys chunks and the camera ask for (`keys_pending_asked`), not the
  inputs expansions ask for (`keys_pending_inputs`); `admit_later_pending` counts the LATER answers at the cap. An
  expansion's new inputs' jobs are made before `job_mutex` and merged under one take. `StreamExecutor::landing()` (between
  a job's run and its landing) lets the tests forget a chunk while its job runs (the orphan path: dropped when it lands, even
  if asked for in the frame it was forgotten). `stop()` (the destructor's first step): a job computing lands nothing. A far blob tile
  computed alone (`point_records`) walks its region once from `proxy_level` instead of its proxy tiles. `ChunkGate::next(chunk, kind, pop, requeue)` is the meshers' pop / admit /
  requeue loop; `begin_frame(cap)` closes a frame at a LATER (`admit_later`) or at `cap` PARKED / LATER answers
  (`admit_capped_frames`; READY ones ask for no key; direct: `max_admits_per_frame`, 0 = 4 x batch_size). Unit tests: park / return order, pins and grace,
  slot reuse with the log, slots full -> LATER with no deadlock, rank order, inherited and improved ranks, `set_tag` (kept, forgotten, dropped jobs), the gate's cap.
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

## Site walks in lockstep, on the GPU

The walks (`cell_sites`: an octree walk reading the ground at node centres at `site_lod`, then the LOD-0 snap) are
`surface_sites_many`: many boxes advance a level per round and each round asks the field once (`FieldBatchFn`), a
reading two walks share read once; the snap's `SNAP_STEPS + 1` rounds follow. A lone walk is the same code with one box.
`ScatterSource::walk_tiles` computes many walk keys at once; a drain task takes its walk job and the queue's next walk
jobs of the same source (`InstanceStream::set_walk_batch`, 1 by default: on the CPU field one task per tile keeps the
workers busy).

`FieldGPU` (`VoxelTerrain.walk_gpu`, `--walk_gpu=0|1`, default on): a thread owning a `LocalPointsGPU` (local device +
`CascadePointsGPU`: any point of any LOD's lattice per item, `SDFCascade::ground_unit` bit for bit, `ground_field` composed on
the CPU). `SDFTable::start_field_gpu` after configure (the point shader needs the configured terrain), `poll()` attaches it
once ready (the sources' `ground_batch`, the stream's walk batch `--walk_batch`, 64) and detaches it if a batch fails.
Only the lockstep walks read it; a lone walk (`sites`, `site_tiles`, the proxy region fallback) reads `ground` (a round trip
per round of a few readings costs more). A submission takes every request queued, its GPU time capped at
`SUBMIT_GPU_MS` from the µs per point its timestamps measure (the renderer shares the device); the readback rides in the
submission (`buffer_get_data_async`). No device, a failure, or stopped: the CPU's field, the same bits. `--walk_verify=1`
reads every GPU batch again on the CPU (ERR and the CPU's bits on a difference). Stats: `table.walk_gpu`.
Measured (1M on `world.json`, 960x540, error 2 | back 4, mesher budget 4): load 139 -> 48 s, desert 27 -> 13 s; the GPU
pays ~1.4 µs per reading on the 17-group climate when idle, ~5-15 in the game (the renderer shares the device).

