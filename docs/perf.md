# Perf target and baseline

The numbers perf PRs are measured against (`tools/bench_ab.sh`, `bench.sh --profile=1`).

- **Perf target:** `--preset=forest5cm` (`project/debug/presets.gd`: 4,000 trees, 5 cm voxels, lod_error 2,
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
  Never build while a bench runs (thermals, see `docs/gotchas.md`).
- **Fold culled per 4x4x4 cube** (`docs/archetypes.md`, 1280x720, `bench_ab.sh`, 2 runs per side, the same bits):

  | forest5cm | before | after | |
  |---|---|---|---|
  | fly out, frame p50 / p99 (ms) | 37.0 / 71.0 | 19.9 / 49.5 | -46% / -30% |
  | fly back, frame p50 / p99 (ms) | 24.4 / 108 | 22.2 / 57.2 | -9% / -47% |
  | settle after the return (ms) | 1,680 | 847 | -50% |
  | load: time / frame p50 / p95 (ms) | 3,057 / 30.7 / 62.0 | 2,667 / 25.0 / 51.7 | -13% / -18% / -17% |
  | mesher GPU per batch, fly (ms; SDF pass) | 7.11 (6.20) | 5.50 (4.59) | -23% (-26%) |
  | steady frame p50 / render GPU p50, fly (ms) | 20.4 / 20.4 | 21.0 / 21.0 | +3% |

  SDF pass per batch by list words (fly, us): 128-255 2,680 -> 2,390; 512-1023 6,040 -> 5,090; 2-4k 9,810 ->
  6,350; 4-8k 11,730 -> 6,230; 8-16k 16,660 -> 8,400; 16-32k 23,800 -> 10,480; below 128 words unchanged (the slice
  barriers cost what the cull saves). GpuCostModel `us_per_word` 1.24-1.41 -> 0.59-0.63 (clamp 1/64-64), list share
  per LOD unchanged. The default bench (no instances) is unchanged within noise (SDF pass 1.04-1.09 ms both).
  **Now the bottleneck:** the render GPU (steady 17.5-21 ms in the forest: the frame is the render's), and in the
  mesher a long list still costs ~6 ms of SDF pass, now in the survivors' evaluations rather than the box tests.
