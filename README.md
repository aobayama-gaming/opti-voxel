# opti-voxel : LOD voxel terrain meshed on the GPU, for Godot

Why this project ? I wanted a voxel terrain with **5 cm voxels** on a world much bigger than
**10,000 km**, that you can still fly over smoothly. Most voxel engines pick one of the two. This
one is a Godot 4.6 GDExtension (C++, godot-cpp) that builds a chunk octree around the camera and
meshes every chunk with surface nets, directly on the GPU.

**Disclaimer :** this is a work in progress. Only Linux is built and tested, and the terrain
itself (`SDFDummy`) is a placeholder noise until the exact, integer-based noise lands.

## How it works

Before diving into the code, the big picture:

- **Chunk octree.** Every chunk has an integer id (`ChunkId`, int64). The LOD of a chunk is the
  lowest set bit of its id, so parents, children and neighbours are bit tricks, never floats
  (`src/chunk_math.*`, `src/world_coord.hpp`).
- **Discovery.** A worker thread (`ChunkDiscoverer`) walks the octree down from the largest
  chunks around the camera, splits a chunk while its screen-space error is too big, and drops the
  ones the SDF says are empty.
- **Meshing.** Two pipelines, picked by `VoxelTerrain.pipeline_mode`:
  - `2` GPU Direct (default): compute passes write the quads straight into pooled pages that the
    GPU draws indirectly. Only a ~2 KB directory per batch comes back to the CPU.
  - `1` GPU Legacy: same passes on a local RenderingDevice, full readback, triangulation on the
    CPU. Slow, but simple: it is kept as the correctness oracle of GPU Direct.
- **Integer frame.** No float ever holds an absolute coordinate: the GPU gets the low 32 bits of
  the ids, render positions are relative to the world origin.

`CLAUDE.md` has the details (page layout, threading rules, and every gotcha already hit).

## Build

```sh
git clone --recursive <this repo>      # godot-cpp is a submodule
scons platform=linux target=template_debug -j$(nproc)   # first build compiles godot-cpp too, a few minutes
godot --headless --path project --import                # once: registers the extension
tools/setup_hooks.sh                                    # once per clone: format check before each commit
```

*The import ends with a crash in the extension when the headless editor exits. It's harmless as
long as `project/.godot/extension_list.cfg` exists afterwards.*

The build treats warnings as errors (`-Wall -Wextra -Werror` on `src/`). To iterate past one, add
`werror=no`, but fix it before committing.

Then open `project/` in Godot 4.6 and run the main scene.

## Checks

| Command | What | Needs |
|---|---|---|
| `tools/unit_test.sh [filter]` | C++ unit tests (`src/tests/`, doctest) inside headless Godot | nothing, ~5 s |
| `tools/check.sh` | The correctness gate: build, unit tests, a validated + audited load run, origin parity, integer frame, zero SDF test | GPU, unlocked session, ~2 min |
| `tools/format.sh [--check]` | clang-format (pinned, through `uvx`) on `src/` | `uv` |
| `tools/bench_ab.sh <a> <b>` | Before/after perf table of two checkouts | GPU |

The unit tests only cover the pure parts so far (integer frame, chunk math, page format, LOD
lookups, load/unload rules). The GPU side is checked end to end by `check.sh`: the legacy mesher
re-meshes chunks to compare triangle by triangle, and an auditor reads every page back.

## Contributing

One branch per feature from `dev`, one PR into `dev`. The rules for code (and for the AI agents
that help writing it) are in `CLAUDE.md`, section "Code rules".
