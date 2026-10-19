# opti-voxel on Roblox (proof of concept)

A Luau port of the **CPU pipeline** (`pipeline_mode = 0`) running in a Roblox client:
same SDF, same surface-nets/QEF mesher, same LOD chunk scan, with meshes built through
`EditableMesh` → `MeshPart`. The GPU pipelines can't be ported: Roblox has no compute shaders.

```
src/shared/OptiVoxel/   pure Luau, runs in Roblox and in the Luau CLI
  Noise.luau            src/noise.cpp (simplex, value only)
  Sdf.luau              SDFDummy::evaluate (mountain + caves)
  ChunkMath.luau        src/chunk_math.cpp
  Mesher.luau           CPUMesher (surface nets + QEF + skirts)
  Discoverer.luau       ChunkDiscoverer (LOD scan)
  Config.luau           Roblox-side knobs
  ChunkBuilder.luau     Roblox only: Mesher output -> EditableMesh -> MeshPart
src/worker/             Actor template: meshes in parallel, builds parts, owns them
src/client/             LocalScript: scan, queue, dispatch to actors, unload, stats overlay
tools/                  validation against the real C++ code + benchmark
```

## Try it on the Mac (no Studio editing needed)

1. Install [Rojo](https://rojo.space/docs/v7/getting-started/installation/): with
   [Rokit](https://github.com/rojo-rbx/rokit), `rokit install` in this folder picks the version
   pinned in `rokit.toml`.
2. Build a place file from this folder and open it:
   ```sh
   cd roblox
   rojo build -o opti-voxel.rbxl
   open opti-voxel.rbxl        # opens in Roblox Studio
   ```
3. Press **Play** (F5). You spawn on the mountain (frozen until the ground under you is meshed).
   The overlay in the top-left shows what matters:
   - `chunks loaded/desired`, `queue`: streaming progress
   - `mesh median`: Luau meshing time per chunk, inside an Actor
   - `build median`: EditableMesh + CreateMeshPartAsync time per chunk (serial phase)
   - `failures` + the last error: EditableMesh memory budget or API errors

Live-sync instead of rebuilding: `rojo serve` + the Rojo Studio plugin ("Connect").

Changing settings means editing `src/shared/OptiVoxel/Config.luau` and rebuilding:

| Symptom | Knob |
|---|---|
| Terrain visible only from below | `FLIP_WINDING = false` |
| `failures` grow (memory budget) | lower `LOD_LEVELS` / raise `PIXEL_ERROR`, or try `DESTROY_EDITABLE_AFTER_PART = true` |
| Chunks disappear with `DESTROY_EDITABLE_AFTER_PART = true` | the MeshPart still references the EditableMesh: set it back to `false` |
| Slow loading | `WORKERS` (one Actor each), `VERTEX_COLORS = false` (fewer API calls) |

## What was verified here (Linux, no Roblox)

`tools/validate.sh` compiles the **unchanged** `src/CPU_mesher.cpp`, `src/chunk_math.cpp`,
`src/noise.cpp` and `src/sdf_dummy.hpp` against a small stand-in for the godot-cpp types
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
