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
- **Any mesh or part is at most 2048 studs per axis** (`CreateMeshPartAsync`: "Mesh extents size too
  large"; `Part.Size` clamps).
- **Smooth Terrain** has none of the mesh limits: 4-stud voxels (fixed), ±128k studs per axis,
  automatic collision, built-in LOD, replication of server edits. It is drawn only **~3k studs around
  the camera** (~750 voxels whatever the scale) at maximum graphics quality: voxels written farther
  out cost memory and are never seen. The draw distance shrinks with the client's graphics quality,
  and "Automatic" quality drops when frames are slow and stays down: after a heavy test the view
  looked a few hundred studs deep until Studio was set back to Level 21 (Studio Settings → Rendering
  → Quality Level). Players pick their own quality, so their view will often be shorter than Studio's.
- **One Terrain per place**: `Instance.new("Terrain")` and `Terrain:Clone()` are refused, and
  ViewportFrames and WorldModels don't render terrain. `CopyRegion` gives an offline `TerrainRegion`
  (data, not drawn). Terrain written on a client stays on that client (no replication).

So the direction: **our SDF stays the source of truth, Smooth Terrain renders it**
(`terrain.project.json`), with a thin height-field shell out to ±3072 studs, streamed around the
player and moved back to the box centre by a treadmill jump (below). A far field past the draw
distance was tried and dropped: EditableMesh rings don't fit (2048-stud and 8-mesh limits), and rings
of WedgePart triangles (29k parts out to 32k studs) looked poor next to the terrain and took ~700 ms
to rebuild a ring.

Scale: `STUDS_PER_UNIT` 8 makes a voxel 0.5 m (the engine's) with characters scaled to 1.8 m, so the
draw distance is ~375 m and the terrain world ±16 km. 5 cm voxels were tried: ~37 m of view, and
`workspace.Gravity` caps at 1000 (scaled gravity needs ~5,200), so jumps turn floaty.

## Infinite world: streaming window + treadmill (`PLAN-infinite-world.md`)

Positions are logical (the world: the SDF and edits) = physical (Roblox space) + origin. The server
(`terrain_server/Window.luau`) keeps the 64-stud slabs around the player written: full detail out to
±1,024 studs (written in 32-stud pieces), a height-field shell out to ±3,072, cleared beyond, one slab
of hysteresis, 8 ms of server time per frame. Past 64k studs from the box centre (or on J), it writes
the window again around the centre with the next origin, then the client moves the character and
camera there in one frame. Measured in Studio (2026-10-19, solo, Studio unfocused):

| | |
|---|---|
| First window | 25.5 s (near square ~12 s): 1,089 near + 8,320 shell slabs, ~1.4 GB of terrain voxels |
| Costs | near piece (32×32) 1.6 ms p50 (90% sampling); shell slab 0.6 ms; clear 0.05 ms; ~310 ms of server time per row of slabs crossed |
| Walking (44 studs/s), 30 s | ≤ 490 jobs queued, nearest unwritten near slab ≥ 991 studs away, server ≤ 16 ms/frame |
| 3× walking, diagonal | near work lags (nearest unwritten slab 668 studs away) but stays far; the budget holds up to ~100 studs/s |
| Memory after 3.9 km | 1,395 → 1,415 MB (flat; the extra is the hysteresis row) |
| Treadmill jump | copy written in 25.6 s, jump, old window cleared in 3.4 s; worst client frame after it 184 ms (first visit) |
| Edits | converted with the origin of the version they were sent in; a dug hole survives its slab being cleared and rewritten |

`WriteVoxels` itself stalls now and then (2 of 1,600 near pieces took 100–250 ms, sampling 2 ms):
Roblox-side. In Studio, the server and client share a process, so it shows as a client hitch there.

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
src/terrain_server/     Smooth Terrain playground: streaming window (Window.luau), treadmill, edits
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
| `terrain.project.json` | `rojo build terrain.project.json -o terrain.rbxl` | Smooth Terrain playground, streamed: LMB dig, F fill, R/T radius, J treadmill jump now, B walk ×3, H overlay |
| `bench.project.json` | `rojo build bench.project.json -o bench.rbxl` | benchmark; set `workspace` attributes `BenchMode` = `pack` / `terrain`, `BenchHalfStuds` before Play; result printed as `[Bench] {json}` |
| `probe.project.json` | `rojo build probe.project.json -o probe.rbxl` | EditableMesh budget and linking probe |
| `default.project.json` | `rojo build -o opti-voxel.rbxl` | chunk-per-EditableMesh client (historical) |

## Gotchas (all hit for real)

- `Sdf.height(0, z)` is degenerate exactly at x = 0 (257.8 vs ~300 around it): the simplex noise
  misbehaves on that lattice point. Never place anything with exact integer samples; spawn code uses
  the max height over a footprint at non-integer offsets.
- Terrain collision is built asynchronously, on each client too: keep a spawn platform solid until
  generation is done, or players fall through.
- `Terrain:FillRegion` refuses big regions ("Region is too large"; 1,300×4,000×1,300 studs failed):
  clear slab by slab.
- A part the client anchors no longer replicates its position: to move a character for a test, pivot
  it every frame and zero its velocity instead.
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

- Edit focus (`tools/luau/edit_focus.luau`): `Sdf.focusEdits` keeps only the edits near a box; on
  40 random edit sets, the Smooth Terrain occupancies of ~660k voxels are identical with and without it.
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
