# roblox/: Luau port

Read `README.md` first (what runs here, layout, places, measurements). These are the working rules for
this folder; where they differ from the root `CLAUDE.md`, they win inside `roblox/`.

## Branches
- The port lives only in `roblox/` on `dev`. There is no long-lived `roblox` branch any more.
- Work on a short `roblox-<feature>` branch off `dev`, open a PR into `dev`, the user merges.
- A roblox branch never edits `src/` (C++); a C++ feature branch never edits `roblox/`.
- Commits from the user's Mac are unsigned (`git -c commit.gpgsign=false commit`): it has no signing
  key. This overrides the root signing rule for this folder only.

## Checks
- The root `/simplify` + `/code-review` checklist is for `src/`. Here, run
  `tools/validate.sh <path to luau>` (the Luau port against the real C++ CPU references) instead.
- Anything Roblox-only (Smooth Terrain, EditableMesh, Actors) is verified in Studio, through the
  Studio MCP: Play, read the console, take screenshots.

## What the port mirrors
- The Luau code mirrors the C++ CPU references at `ORACLE_REV` in `tools/cpp_oracle/build.sh`.
- `git log --oneline $ORACLE_REV..origin/dev -- src/` lists what dev did since; `PORTING.md` tracks
  which of it is ported, pending, or GPU only. Port something, update the row; catch up with a newer
  engine, bump `ORACLE_REV` with it.

## Studio gotchas
- Smooth Terrain is only drawn ~3,000 studs around the camera: terrain written farther out costs
  memory and generation time and is never seen.
- `Terrain:FillRegion` refuses big regions ("Region is too large"): clear slab by slab.
- A part anchored by the client stops replicating its position (to fake a walk, pivot every frame).
- Studio builds terrain render meshes and runs at full fps only while its window is focused (unfocused:
  ~15 fps, meshes not built). Check the fps before trusting a screenshot or a timing.
