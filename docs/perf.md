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
