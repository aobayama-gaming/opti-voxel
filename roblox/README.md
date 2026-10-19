# opti-voxel on Roblox

Port of the opti-voxel SDF terrain to Roblox, in Luau. It lives in this folder on `dev` and lags behind
the engine: features land in `src/` first and get ported here progressively (`PORTING.md`; working
rules in `CLAUDE.md`).

The GPU pipelines can't be ported (Roblox has no compute shaders). What runs here is the CPU side:
the same SDF, the same surface-nets/QEF mesher, the same LOD chunk scan.

## Rendering: what Roblox allows (measured in Studio 0.743)

- **EditableMesh: at most 8 alive per client**, whatever their size (0 or 19k triangles each).
  `AssetService:CreateEditableMesh()` returns nil past that ("reaching memory budget limits").
- A MeshPart made from an EditableMesh stays **linked** to it: destroy the EditableMesh and the part
  turns invisible. `ApplyMesh` copies and `Clone()` are linked too (`probe/`).
- Each EditableMesh holds ≤ 20k triangles / 60k vertices, so custom geometry is capped at
  **~160k triangles per client**. One mesh per chunk (`default.project.json`) shows ~16 chunks.
- **Smooth Terrain** has none of these limits: 4-stud voxels (fixed), ±128k studs per axis,
  automatic collision, built-in LOD for distant terrain, replication of server edits.

So the current direction: **our SDF stays the source of truth, Smooth Terrain renders it**
(`terrain.project.json`). A visual-only far field (clipmap rings in the 8 EditableMeshes, or thin
far terrain) is the next step being evaluated.

Benchmark, same 336×336-stud region with caves, same 10 sphere edits (`bench.project.json`):

| | pack: our mesher → 8 shared EditableMeshes | Smooth Terrain: SDF → WriteVoxels |
|---|---|---|
| Build | 10.5 s (8.6 s Luau meshing, 1 thread) | 0.42 s (0.39 s sampling) |
| Coverage | budget full at 159k tris, 45/252 chunks dropped | complete |
| One edit (r = 15 studs) | 226 ms + 350 ms collision re-cook | 0.9 ms (engine re-meshes async) |
| 2048×2048 studs | impossible | 12.3M voxels in 18.5 s, ~56 FPS |

## Layout

```
src/shared/OptiVoxel/   pure Luau, runs in Roblox and in the Luau CLI
  Noise.luau            src/noise.cpp (simplex, value only)
  Sdf.luau              SDFDummy::evaluate (mountain + caves) + sphere edits (carve / fill)
  ChunkMath.luau        src/chunk_math.cpp
  Mesher.luau           CPUMesher (surface nets + QEF + skirts)
  Discoverer.luau       ChunkDiscoverer (LOD scan)
  Config.luau           Roblox-side knobs
  TerrainGen.luau       Roblox only: SDF -> Smooth Terrain voxels (materials by depth / slope)
  ChunkBuilder.luau     Roblox only: Mesher output -> EditableMesh -> MeshPart
src/terrain_server/     Smooth Terrain playground: generation + server-side edits
src/terrain_client/     dig / fill tool + overlay
src/bench/              pack vs Smooth Terrain benchmark
src/client/, worker/    chunk-per-EditableMesh client (Actors), limited by the 8-mesh cap
src/server/             spawn platform for the chunk client
probe/                  EditableMesh budget / linking probe
tools/                  validation against the real C++ code + Luau benchmark
```

## Places

Install [Rojo](https://rojo.space/docs/v7/getting-started/installation/) (`rokit install` here picks
the version pinned in `rokit.toml`), then build and open in Studio:

| Project | Build | What |
|---|---|---|
| `terrain.project.json` | `rojo build terrain.project.json -o terrain.rbxl` | Smooth Terrain playground: LMB dig, F fill, R/T radius, H overlay |
| `bench.project.json` | `rojo build bench.project.json -o bench.rbxl` | benchmark; set `workspace` attributes `BenchMode` = `pack` / `terrain`, `BenchHalfStuds` before Play; result printed as `[Bench] {json}` |
| `probe.project.json` | `rojo build probe.project.json -o probe.rbxl` | EditableMesh budget and linking probe |
| `default.project.json` | `rojo build -o opti-voxel.rbxl` | chunk-per-EditableMesh client (historical) |

## Gotchas (all hit for real)

- `Sdf.height(0, z)` is degenerate exactly at x = 0 (257.8 vs ~300 around it): the simplex noise
  misbehaves on that lattice point. Never place anything with exact integer samples; spawn code uses
  the max height over a footprint at non-integer offsets.
- Terrain collision is built asynchronously, on each client too: keep a spawn platform solid until
  generation is done, or players fall through.
- Before the character exists the camera sits at the origin, deep underground: don't scan from it.
- Studio throttles an unfocused window to ~15 FPS: frame times are only meaningful with Studio
  focused.
- Studio writes Output to `~/Library/Logs/Roblox/*_last.log`, but long lines get truncated there.
- The Roblox Studio MCP (`StudioMCP`) can Play/Stop, run Luau in the Edit/Client/Server datamodels,
  read the console and take screenshots: verify in Studio directly.

## What was verified here (Linux, no Roblox)

`tools/validate.sh` compiles the **unchanged** `src/CPU_mesher.cpp`, `src/chunk_math.cpp`,
`src/noise.cpp` and `src/sdf_dummy.hpp` of the engine commit the port mirrors (`ORACLE_REV` in
`tools/cpp_oracle/build.sh`; dev has since dropped the CPU pipeline) against a small stand-in for the godot-cpp types
(`tools/cpp_oracle/`) and compares them with the Luau port:

- SDF, 20 000 points: no sign flips; max difference 5e-7 vs the C++ built with doubles,
  3e-4 vs the real float32 build.
- 15 chunks (surface, caves, empty, negative coordinates, LOD 0..5): **identical triangles,
  winding included**, vs the C++ built with doubles. Vs the float32 build: same vertex and triangle
  counts. Positions drift at coarse LODs (up to ~1 unit at LOD 4): the C++ edge normals use a 1e-4
  central difference at coordinates ~300, close to float32 resolution, and the QEF amplifies it.

```sh
tools/validate.sh path/to/luau          # Luau CLI: github.com/luau-lang/luau/releases
luau --codegen tools/luau/bench.luau    # mesher benchmark
```

Benchmark (one core of a 2.1 GHz Xeon, `--codegen` = native code like `--!native` in Roblox):

| | per chunk with geometry (median / p90) |
|---|---|
| Luau, native codegen | 61 ms / 95 ms |
| Luau, interpreter | 298 ms / 441 ms |
| C++ `CPUMesher` (float, -O2), same chunks | 117 ms / 144 ms |

The Luau port is faster than the C++ CPU mesher only because it skips work the C++ does for
nothing (the noise gradient, and cave noise far above the surface); it is not a language result.
Empty chunks still cost ~16 ms (full grid sampling) when the discoverer lets them through.

Static check of the Roblox-specific code: `luau-lsp analyze` against the Roblox API definitions
passes with no errors (strict mode). Everything that needs a live Roblox client (EditableMesh
memory budget, CreateMeshPartAsync cost, winding convention, Actor scheduling) is what the Studio
run above is for.
