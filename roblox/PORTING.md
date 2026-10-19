# Porting status

The Luau port mirrors the engine at `ORACLE_REV` = `29c2f8b` (`tools/cpp_oracle/build.sh`, PR #1 into
`main`, 2026-10-16): `CPUMesher`, `chunk_math`, simplex `noise`, `SDFDummy`, the CPU `ChunkDiscoverer`.

Dev PRs touching `src/` since then (`git log --oneline --first-parent 29c2f8b..origin/dev -- src/`).
Status guessed from commit subjects, not from reading the diffs; correct a row when you look closer.

| PR | Feature | Status |
|---|---|---|
| #3 | GPU Direct: Morton-ordered pages | n/a (GPU only) |
| #4 | GPU Direct: page compaction | n/a (GPU only) |
| #5 | GPU Direct: 8-byte corners | n/a (GPU only) |
| #6 | Remove the CPU mesher pipeline | n/a (the port is that pipeline; the oracle stays at `ORACLE_REV`) |
| #7 | Integer frame: int64 chunk ids, `WorldPos`, world origin | pending (Luau numbers are doubles: design needed) |
| #8 | Far LOD: split threshold off-by-one, camera FOV, skirt | pending |
| #10 | Discovery: cull verdict cache, miss audit | pending (cache); audit n/a |
| #11 | Mesher: holes where an SDF sample is exactly 0.0 (one side rule) | pending (check `Mesher.luau`) |
| #12 | Hygiene: format, `-Werror`, unit tests | n/a (C++ tooling) |
| #13 | Exact noise: integer-hash gradient noise | pending |
| #14 | LOD-aware SDF, gradients per voxel of the LOD | pending (after #13) |
| #15 | Hermite cascade, CPU reference | pending (after #13, #14) |
| #16 | LOD range up to 31, altitude clip planes | pending (needs #7) |
| #17 | Cascade GPU passes, `SDFCascade` terrain | pending (`SDFCascade` CPU side); passes n/a (GPU only) |
| #18 | Skirt depth per LOD, continents preset | pending |
| #19 | GPU discoverer | n/a (GPU only); the wave walk on the CPU is pending |
| #20 | Mesher pacing | n/a (GPU only) |
| #21 | Incremental destroy check (`VoxelState`, scan deltas) | pending |
| #22, #23 | Archetype IR, instances, demo terrain | pending (CPU interpreter); GLSL n/a |
| #24 | Archetype table, runtime instance edits | pending (CPU index + evaluate); GPU buffers n/a |
| #25 | Dual contouring; `SDFDummy` dropped | pending (QEF variant); the port still runs `SDFDummy` |
| #27, #28 | Shapes gallery, faceted debug view | pending (gallery); view n/a |
| #30 | Vertex normal: no NaN where the gradient vanishes | pending |
| #31 | Archetype band rule, million-instance table | pending |
| #32 | Trees as archetype-table instances | pending (after #31) |
| #33 | Edit validation, no-RD fallback, readback lifetime | pending (edit validation); rest n/a |
| #36 | View-dependent LOD, front-first load order | pending |
| #38 | GPU-time budget for mesher batches | n/a (GPU only) |
