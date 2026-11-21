# Audit: integer frame and exact noise

2026-11-21, on `dev` at `14a460c` (read at `baa6b83`; the audited files did not change in between, except the
legacy mesher's removal). Scope: `world_coord.hpp`, `gpu_frame.hpp`, `chunk_math.*`, `exact_noise.hpp`,
`exact_noise_glsl.hpp`, their tests, and the cascade shader's point arithmetic (`cascade_glsl.cpp`).
Design: `docs/integer-frame.md`.

## Verdict

No correctness bug in the production path. One latent overflow in the GPU check path, and coverage gaps: the
GPU side of the frame is only verified by windowed checks, and the top LODs the settings accept are tested nowhere.

## What was checked

| Check | How | Result |
|---|---|---|
| CPU and GLSL are op for op identical (hash, `octave_axis`, fades, corner gradients, derivatives, fBm factor) | read side by side | identical |
| `octave_axis` in 32-bit words vs int64, every k in 0..62 | CPU emulation of the GLSL, 12.6M cases around 0, ±2^31, ±2^32, 2^45, ±2^60 | 0 mismatches |
| `i64_asr`, `i64_asr_ceil`, `i64_shl`, `i64_add`, `i64_sub`, `sat_words` vs int64 | same emulation, 4M cases, shifts 0..63 | 0 mismatches |
| `encode_chunk_input` / `decode_item` round trip | every LOD 0..30, ids up to 2^60 | exact |
| `world_pos_from_grid` vs `ChunkMath::sample_pos` | emulation with the GLSL's int32 conversion, every LOD | equal up to LOD 26, out of int32 from LOD 27 (gap 1) |
| Denormals in `perlin2` / `perlin3` outputs and `gradient * 2^-16` | 6M samples within 64 voxels of lattice corners, k 12..23 | none found |
| Unit tests `ExactNoise*`, `ChunkMath*`, `WorldCoord*` | `tools/unit_test.sh --no-build` | pass (39 cases) |
| int32 narrowing of ids, floats built from absolute coordinates | grep sweep of `src/` and the adapters | none: every hit subtracts an origin or an anchor first |

Not run: `tools/frame_test.sh` and `tools/cascade_test.sh` (they open a window). Nothing here was verified on a real GPU.

## Gaps

| # | Severity | Gap | Where |
|---|---|---|---|
| 1 | latent bug, check path only | `world_pos_from_grid` converts `floor((grid - skirt) * 2^lod)` with `ivec3()`: undefined once it reaches 2^31, i.e. from LOD 27 with a 32-wide chunk (84 to 99% of samples at LODs 27..30). `MAX_LOD_NUMBERS` allows LOD 30. Only `frame_check.cpp` calls it: the terrain builds its points with `i64_add` + `i64_shl`. | `gpu_frame.hpp:148` |
| 2 | coverage | LODs 25..30 are accepted by `world_file.cpp` and `World::start` but tested nowhere: `PROBE_LODS` and the unit tests stop at 24. | `frame_probe.hpp:31`, `test_exact_noise.cpp:21` |
| 3 | coverage | The GLSL word math has no CI coverage: only the windowed frame and cascade checks run it. | `tools/frame_test.sh` |
| 4 | drift | `world_pos_from_grid` and `sample_pos` are no longer on any production path (the legacy mesher and the validator are gone): the frame check's `worldpos_mismatch` and `pairs` guard a function only the check uses. The doc still gives `cascade_sample_sdf(c0, p, lod, chunk)`, an old signature. | `frame_check.cpp`, `docs/integer-frame.md:9` |
| 5 | low | The origin bound (`MAX_ABS_ORIGIN_VOXEL`) is only enforced by the Godot adapter's setter. `World::start` takes any `world_origin_request`: another host could start where int64 voxel arithmetic overflows. | `voxel_terrain.cpp:354`, `world.cpp` |
| 6 | low, future | Octave constants go through libm: `std::pow(persistence, n)` for amplitudes, `log2` in `lattice_k`. CPU and GPU agree on one machine (the bits are spliced into the GLSL), but two machines (native, web) may differ by one ulp, so by a world. Matters once worlds are shared. | `climate_parse.cpp:393`, `exact_noise.hpp:122` |
| 7 | theoretical | The denormal rule is proven for the products inside an octave, not for the last one: after a cancellation a gradient can be as small as 2^-119, times the minimum factor 2^-16. Needs 23 bits of coincidence next to a lattice corner; none found. | `exact_noise.hpp:32` |
| 8 | coverage | `perlin2` has no bit digest (`perlin3` has one): a refactor can change its bits unseen by the unit tests. | `test_exact_noise.cpp:222` |

Checked and fine, for the record: negative coordinates everywhere (floor shifts, parents, `chunk_at_voxel`), the
carry between low and high words, the salt sharing of a cell's corners across a 2^32 boundary, octave offsets up to
k = 62 added to origins up to 2^60 (no int64 overflow), `sat_diff`, `grid_to_render_lo32` across a low-word wrap.

## Plan

One PR into `dev`, `feature/frame-audit-fixes`, about 300 lines, no GPU needed for its tests.

1. **Word-mirror doctest** (gaps 3, 8): `src/tests/test_frame_words.cpp`, the CPU emulation of the GLSL words used
   for this audit, against the int64 code: `octave_axis`, the `i64_*` helpers, `sat_words`, the `ChunkInput` round
   trip. Plus a `perlin2` digest. Runs in CI.
2. **`world_pos_from_grid`** (gaps 1, 4): split the integer and fractional parts so the offset goes through
   `i64_add` (one add per 2^31 at most) or, simpler, delete it and `sample_pos` and let the frame check build its
   points like the cascade (`level_point`); keep fractions only if a caller still needs them. Decide on reading
   `frame_check.cpp`: deletion if nothing but the check reads fractions.
3. **Top LODs** (gap 2): unit-test cases at LODs 25..30 (fBm gradient scale, `chunk_min_voxel`, `decode_item`), and
   `PROBE_LODS` extended to 30 if the frame check stays in budget; otherwise lower `MAX_LOD_NUMBERS` to what is
   tested and say so in the doc.
4. **Origin bound in the core** (gap 5): `World::start` refuses a `world_origin_request` past
   `MAX_ABS_ORIGIN_VOXEL`; the adapter's setter keeps its early error.
5. **Doc** (gaps 4, 7): the current `cascade_sample_sdf` signature, the denormal rule's actual bound.
6. Verify: `tools/unit_test.sh`, `tools/check.sh`, then `tools/frame_test.sh` and `tools/cascade_test.sh` (windowed:
   ask first).

Left out on purpose: gap 6 (libm) waits for a feature that shares worlds between machines; the fix is integer or
table-driven constants (`ldexp` for power-of-two persistence, a correctly rounded `pow` otherwise).

## Template for the next audits

Same sections: scope and commit, verdict, what was checked (how, result, what was not run), gaps (severity, where),
plan. One file per subsystem in `docs/audits/`, named after its doc in `docs/`.
