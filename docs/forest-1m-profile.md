# The 1M forest: where the time goes (profile of 2026-10-23)

Merged code (`origin/dev` eb91399, PR #94) plus the instrumentation of `feature/forest-1m-profile` (no behaviour
change). Laptop, NVIDIA screen, 1280x720, `--mode=direct`, `--cool=1`, nothing else on the GPU. The run directories are
local (`bench/results/` is gitignored); the numbers below are copied from their `profile.md`, `cpu.md` and `stats.json`.

| run | command | dir |
|---|---|---|
| 1M fly, perf attached | `tools/cpu_profile.sh --sampler=perf -- --preset=forest5cm_1m --scenario=fly --profile=1` | `20261023-012834-direct` |
| 1M approach, perf attached | same with `--scenario=approach --approach_shots=approach` | `20261023-013158-direct` |
| forest5cm fly (control) | `tools/bench.sh --preset=forest5cm --scenario=fly --profile=1` | `20261023-013555-direct` |
| 1M fly, the viewer's settings | `--preset=forest5cm_1m --climate=res://climate/landforms/desert.json --lod_error=2 --lod_error_back=4 --scenario=fly --profile=1` | `20261023-013654-direct` |

`perf record -F 499 -g` on the 20 threads slows the 1M load by ~40% (106 s against 73 s measured on 2026-10-22 without
it, 63 s here with the desert climate): the sampled runs give shares and ratios, the unsampled ones the times.

## Headline

| | forest5cm (4,000 trees) | 1M, desert, back 4 (no sampler) | 1M, perf attached |
|---|---|---|---|
| load to complete | 3.4 s | 63.3 s | 105.7 s |
| fly out p50 / p99 (ms) | 17.2 / 76 | 62.7 / 78 | 37.2 / 92 |
| fly back p50 / p99 | 17.7 / 82 | 48.3 / 122 | 18.5 / 71 |
| steady p50 (GPU-bound) | 15.8 | 17.9 | 20.0 |
| stream keys computed / evicted | 94.7 k / 0 | 3.29 M / 3.10 M | 6.51 M / 6.45 M |
| worker CPU (8 drain tasks) | 3.3 s | 207 s | 387 s |
| stream lock held (admit + poll + land) | 1.9 s | 37.8 s of 63 s | 71 s of 106 s |
| RSS / VRAM (MB) | 1,317 / 1,600 | 3,356 / 1,574 | 3,294 / 1,931 |

Five things, in the order of their measured share:

1. **The LOD-0 part tiles are recomputed 120x** (76% of the worker CPU), and four fifths of a tile is the cascade ground
   evaluation of its site walk, not the tree growth.
2. **The stream's exclusive lock is held two thirds of the load**: `land()` on the workers (40 s), `admit()` (19 s)
   and `poll()` (12 s) on the main thread serialise 8 workers that are 39% busy.
3. **The main thread's frames over budget are the stream's**: +6.6 ms of `gate_admit` and +3.2 ms of `stream_poll`
   per over-budget frame; an admit creates up to 16,970 keys (recursing into inputs under the lock).
4. **Children land in queue order, not distance order**: hand-off p95 of 33-61 frames in a fly and 4,607 frames after
   an approach; the FIFO holds up to 7,654 jobs, 5.6-13 k parked chunks go stale.
5. **The GPU fold costs 5.8 ms + 1.0 µs per list word per batch**; 16% of the batches (over 8 k words) take 61% of
   the mesher's GPU time, with the budget off and a cost model that doesn't see the words.

## 1. Worker CPU (perf, 1M fly: 271,673 samples, 88% on the pool threads)

Inclusive shares of all samples (`cpu.md`, "All threads: top inclusive"):

| symbol | inclusive % | what |
|---|---|---|
| `InstanceStream::drain` / `Job::run` | 80.9 / 77.5 | the drain tasks |
| `ScatterSource::part_tile` | 58.3 | the LOD-0 tiles (the LOD 1-7 tiles only concatenate) |
| `surface_sites` (from `part_tile`, `far_items`, `blob_tile`) | 61.0 | the site walk |
| `SDFCascade::ground_field` = `Cascade::point_into` | 60.7 | the ground evaluated point by point for the walk |
| of which `surface_cells` / `centre_field` | 48.3 | the octree walk of the cells the surface may cross |
| of which `site_of` | 12.5 | the site per surface cell |
| `ScatterSource::blob_tile` | 18.8 | the moment layers (blobs 5-7: 9-20 ms each) |
| `ScatterSource::far_items` | 13.3 | the far layers' proxies (`surface_sites` again) |
| `grow_tree` and `finish_table_record` (TableRecord construct, `listed_at`, `world_bounds`) | < 5 | the trees themselves |

Self time: `ExactNoise::pcg3d` 17.8%, `perlin3_cell` + `perlin3_value_gradient` + `grad3` + `fade` 17%,
`Cascade::band_noise` / `hermite_mid` / `subdivide` 6%, `Moments::principal` 2.8%. The forest's cost at 1M is the
terrain's noise, evaluated per cell of every LOD-0 tile walk, each tile walked ~120 times.

Per key kind (`computed_by_key`, cumulative at the end of the fly run):

| key | computed | unique (computed - evicted) | recompute ratio (load phase) | ms each | worker s | share |
|---|---|---|---|---|---|---|
| parts 0 | 1,592,696 | 11,839 | 120 | 0.173 | 275 | 71% |
| parts 1-3 | 4,682,573 | ~39 k | 114-119 | 0.001-0.002 | 8 | 2% |
| parts 4-7 | 222,906 | ~2.5 k | 8-89 | 0.007-0.7 | 4 | 1% |
| blobs 2-13 | 10,488 | ~3.7 k | 1.8-7 (inf for 2-6: all evicted) | 2-22 | 100 | 26% |

The control (forest5cm) computes 32 k LOD-0 tiles once (0.029 ms each: most are empty), evicts nothing and spends 78% of
its 3.3 s on the blobs. Why 1M differs: the records cap (`INPUT_RECORDS` = 2^21, `slots_peak` 2,093,191 of 2,097,152 in
every 1M run) holds ~51 k sites' records, 5% of the forest, while every LOD 4-7 part tile and every part-layer blob
tile within the load radius is full and is built from the LOD-0 tiles under it (the record pyramid, up to
(2^(L-3))^3 inputs). An evicted LOD-0 tile is walked again for the next consumer: 1.59 M computations of 11.8 k tiles.
`entries` (57 k) never reaches its own cap (2^18): the records drive the eviction, not the key count.

## 2. The stream's lock

`stats.json["table"]["stream"]`, 1M fly, load phase (105.7 s wall):

| | ms | where |
|---|---|---|
| `land()` incl. `place()` (under the exclusive lock) | 40,115 | the 8 drain tasks, after each job |
| `admit()` (exclusive) | 19,197 (5,634 waiting for the lock) | main thread, from the gate |
| `poll()` (exclusive) | 11,919 (3,493 evicting) | main thread, once per frame; max 107 ms |
| compute (no lock) | 328,304 / 8 workers | the jobs |

71 s of a 106 s load is spent holding the one lock. 6.08 M lands at 6.6 µs: a land is a hash lookup, the dependents'
hand-over, `place()` (first-fit `allocate` over `free_runs`, `evict_oldest` loops when full) and the LRU link. 4.7 M of
those lands are the parts 1-3 keys, whose compute is 1-2 µs: at 1M, a key costs more to land than to compute. The
workers are 39% busy over the load (52% in the approach, 65% in the fly back): the rest is the lock and an empty queue
behind a FIFO head waiting on inputs. In the desert run the same three add to 37.8 s of 63 s.

## 3. Main thread

Perf, 1M fly, main thread (10.3% of the samples): `[unknown] Godot` 16.6% self, `SourceKey` hash table lookups
(`_Hashtable::_M_find_before_node`, `_Hashtable_base::_M_equals`, `_Mod_range_hashing`) 26%, `malloc` / `free`
(`_int_malloc` 10.1, `unlink_chunk` 4.5, `_int_free*` 4.5, `tcache` 2.3) 21%. Inclusive: `InstanceStream::request`
32.9%, `evict_oldest` 11.0%, `find_ancestor_in` / `HashSet<Vector3i64>` (destroy check) 3.6%.

`_process` sections, 1M fly, load phase, ms per frame:

| section | mean | p95 | p99 | max | over-budget frames vs the others |
|---|---|---|---|---|---|
| terrain total | 7.22 | 39.2 | 64.4 | 108 | +11.9 |
| mesher_dispatch | 4.91 | 35.8 | 45.3 | 75.6 | +8.7 |
| of which gate_admit | 3.68 | 34.6 | 38.8 | 62.0 | +6.6 |
| stream_poll | 2.28 | 12.8 | 36.6 | 106.7 | +3.2 |
| record_bytes | 0.28 | 0.26 | 8.35 | 39.6 | |
| build_lists | 0.08 | 0.52 | 1.65 | 6.7 | |

48% of the load frames are over 16.7 ms; against the others they carry +11.9 ms of terrain and +7.7 ms of GPU (mesher
+3.6, geometry +4.1). Admits: 114 k READY, 36.8 k PARKED, 5.6 k LATER; 42 keys created per admit on average, 16,970 at
most (one admit recursing through a coarse tile's pyramid); `jobs_queued_max` 7,654. The desert fly back has a terrain
p50 of 15.0 ms (109 k keys admitted in 6 s). The control's admits create 1 key each.

## 4. GPU mesher vs list words

`mesher.gpu_by_list_words`, 1M fly, 1,092 timed batches, sdf pass 11.1 s in all:

| list words per batch | batches | sdf pass µs avg |
|---|---|---|
| 0 | 310 | 877 |
| 512-1,023 | 111 | 6,909 |
| 2,048-4,095 | 103 | 9,753 |
| 8,192-16,383 | 75 | 17,567 |
| 32,768-65,535 | 31 | 54,300 |
| 131,072-262,143 | 5 | 199,591 |
| 262,144-524,287 | 1 | 294,734 |

Weighted fit over the 567 batches with 512+ words: **sdf µs = 5,842 + 1.004 x words**. The 170 batches over 8 k words
(16%) take 6.8 s (61%) of the sdf time; the biggest list is 292,678 words (429 ms in the desert run). `cost_us_per_chunk`
reads 1.1-1.4 ms for LODs 5-7 against 74 µs at LOD 0, `cost_model_ratio_avg` 1.23 (desert 1.26, approach 1.58, control
1.05): the model is a per-LOD average of lists that vary 100x. The bench runs with `gpu_budget_ms` 0.

Beyond this plan: the desert fly out is GPU render-bound at 62 ms p50 (66 k chunks loaded, 73.6 M triangles drawn, the
old chunks kept until their replacements land); `lod_error_back=4` and the fly's double drawing, not the stream.

## 5. Hand-off (the blob)

`destroy.handoff_frames` = frames a loaded parent stays drawn with a stalled sub-cube (per sub-cube hiding of #80 is
in place: the children's keys are what come late). `park_frames` = stream frames from a chunk's park to its hand-back.

| run / phase | hand-off p50 / p95 / max (count) | park p50 / p95 / max | stale parked | FIFO max |
|---|---|---|---|---|
| control fly out | 2 / 5 / 78 (8,206) | 0 / 2 / 6 | 337 | 6,046 |
| 1M fly out | 2 / 33 / 135 (7,595) | 7 / 24 / 80 | 5,301 | 7,654 |
| 1M fly back | 2 / 61 / 387 (8,620) | 7 / 24 / 80 | 305 | |
| 1M desert fly back | 2 / 75 / 192 (6,753) | 7 / 37 / 103 | 5,821 (13,314 in all) | 7,601 |
| 1M approach (20 s) | 3 / 83 / 1,814 (2,612) | 17 / 63 / 955 | 6,998 | 7,656 |
| 1M approach, settle after | 14 / 4,607 / 6,530 (4,047) | 7 / 59 / 955 | | |

The approach: 19 s to load the start 1,500 m up (12 k chunks), 20 s of flight ending with 30.6 k loaded and 26.2 k
pending, then **109 s of settle** to complete the forest at the spawn, parents drawn over their children for a p95 of
4,607 frames (over a minute at 19 ms). The contact sheet (`20261023-013158-direct/approach/approach_<s>s.png`): at
10 s the forest is the far layers' smooth green blanket; at 16 s one fine square patch has landed inside it, the rest
of the blanket is untouched; at 20 s the camera sits on the ground inside a crown blob with the near field not loaded.
The patches come in the order the FIFO produces them (a coarse tile's pyramid at a time), not by distance, and a parked
chunk's 8 siblings wait for the last shared key.

## Ranked causes and the fix PRs

| # | cause | measured share | PR |
|---|---|---|---|
| 1 | LOD-0 tile thrash through the record pyramid; 80% of a tile is the ground walk | 71-76% of worker CPU, ratio 120-157 | PR 3: `SITES` keys (sites + ground, resident), `PARTS L` grown from them with the LOD cut, proxies for part layers at or above the source cell. The site walk is the cost to cache, before the growth. |
| 2 | one exclusive lock: land on the workers, admit and poll on the main thread | 71 s of 106 s; workers 39% busy; 4.7 M parts 1-3 lands at 1-2 µs of compute | PR 3 (no parts 1-3 keys: one tile, lists per LOD) and PR 4 (land in batches outside the lock, `allocate` by size, bounded eviction) |
| 3 | admit recursion and poll eviction on the main thread | +6.6 / +3.2 ms per over-budget frame, max 62 / 107 ms, 16,970 keys per admit | PR 2 (cap per frame, no-create admit on LATER) and PR 4 |
| 4 | FIFO, no camera priority, stale keys kept | hand-off p95 4,607 frames after an approach; 5-13 k stale parked | PR 2: priority heap from `load_queue` rank, inputs inherit it, stale jobs dropped |
| 5 | GPU batches with long lists, cost model blind to words, budget off | 16% of batches = 61% of sdf GPU; 5.8 ms + 1.0 µs/word | PR 4: words in `GPUCostModel`, `--mesher_budget=4` in the forest presets |

Targets to judge them on (same runs, `bench_ab.sh` against this branch): load < 40 s and worker CPU < 60 s at 1M,
recompute ratio < 1.5 per key kind, `_process` avg < 3 ms and max < 40 ms, hand-off p95 < 60 frames in the approach's
settle, fly out p50 back to ~17 ms, the control unchanged.

## Tooling notes

- `tools/cpu_profile.sh`: `perf_event_paranoid` 2 is enough (an owned process, user space); the per-thread report
  sorts by `comm,pid,sym` (`tid` is not a sort key); `cpu_report.py` streams `perf script` (1 GB of text per 1M run,
  no longer kept). `GODOT` may be a wrapper script (`--screen 2`): the bench's process is found by cmdline, skipping
  `timeout`.
- The telemetry's `throttled` flag reads 81-97% on every run here, control included, at a median clock of
  1.4-1.65 GHz (`performance` profile): a power cap, not the `quiet` profile's 300 MHz; the runs compare with each other.
- `--scenario=approach` ran in view 0 only; the LOD view (`--view=2`) is the same run again.
