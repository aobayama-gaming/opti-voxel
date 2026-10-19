# Plan: infinite world on Smooth Terrain (streaming window + treadmill)

Status 2026-10-19: findings and the solo swap test are done (branch `roblox-treadmill`). Next: part 1.

## What we know (measured in Studio)

- **Scale:** `STUDS_PER_UNIT = 8`, so a Smooth Terrain voxel (4 studs, fixed) is 0.5 m. Characters are
  scaled to 1.8 m. Gravity and jumps are kept Roblox-style (×2.67) on purpose.
- **Terrain box:** `Terrain.MaxExtents` is ±32,000 voxels = **±128k studs** per axis (256k across,
  32 km at 0.5 m). There is only one Terrain: `Instance.new` and `Clone()` are refused, and
  ViewportFrames don't draw it.
- **Draw distance:** about 3,000 studs (~375 m) around the camera, at graphics quality Level 21.
  - It shrinks with the client's quality, and "Automatic" drops after heavy load. Check
    `settings().Rendering.QualityLevel` before blaming the code for a short view.
  - Atmosphere density 0.3 hides the edge.
- **Far meshes don't work:** a mesh or part is at most 2,048 studs per axis, and the client gets 8
  EditableMeshes. WedgePart rings were tried and rejected. Smooth Terrain alone is the target.
- **Cost today:**
  - near square ±1,024 studs (full depth, caves): ~19 s, ~14 M voxels;
  - crust out to ±3,072 studs: ~5 s;
  - terrain memory: ~1.1–1.2 GB for one window, 1.4 GB on the server with two windows;
  - sampling is ~90% of generation time, `WriteVoxels` ~10%.
- **Swap test** (`ee1a953`, J in the playground): the same window written 40,960 studs away, with the
  character and camera moved by that offset in one frame.
  - The voxels are identical in both copies (0 of 2.4 M differ).
  - The user sees a near-immediate coarse→fine LOD pop and accepts it once per ~10 km.
  - First jump into a fresh area: one 195 ms frame. Jumps into an already visited area: none.
  - Roblox's collision mesh differs by up to ~1 voxel between identical copies on steep rock.

## Coordinates (the rule everything depends on)

- **Logical** position: where something is in the world. **Physical** position: where it is in Roblox
  space. `physical = logical - origin`.
- The **origin** is a Vector3 in studs, snapped to whole slabs (64 studs), so voxels and slabs line up
  after a change. Luau numbers are exact integers up to 2^53, so logical coordinates need no
  special type.
- The SDF and its edits only see logical coordinates. `TerrainGen` already takes a `shift`
  (physical = logical + shift); `shift = -origin`.
- Only the server owns the origin. It publishes it as the workspace attribute `Origin`, and clients
  convert anything they send (edits) or display (position).
- Solo first: one origin for the whole server. Multiplayer is part 3.

## Part 1: streaming window (continuous, invisible)

**Goal:** the player walks anywhere inside the box, never sees an edge, and memory stays flat.

1. **Slab grid in logical space.** A slab is 64×64 studs (as now), keyed by its logical index
   `(floor(x / 64), floor(z / 64))`. States: absent, queued, written.
2. **Desired set**, recomputed when the player crosses a slab boundary:
   - near: full detail (`writeBox`, caves, edits) within ±`TERRAIN_HALF_STUDS` (1,024) of the player;
   - crust: `writeFarTile` from there to ±`TERRAIN_FAR_HALF_STUDS` (3,072);
   - a slab that passes from crust to near is rewritten at full detail;
   - hysteresis: clear only past half + 1 slab, so walking back and forth on a boundary doesn't churn.
3. **Work queue:** missing slabs nearest first, near before crust, run within a per-frame time budget
   (start at 8 ms of server time; make it a Config knob).
4. **Clearing** slabs that leave the set: `Terrain:FillRegion(region, 4, Air)` over the slab's column
   (check its cost against `WriteVoxels` of air).
5. **Edits:** they already live in the SDF, so a slab regenerated later still shows them.
   - `Sdf` scans its edit list linearly. Before many edits, bucket them by slab so `writeBox` only
     tests the edits near its box.
   - The edit handler converts physical → logical with the origin (the "nearest copy" hack from the
     swap test goes away).
6. **Spawn:** generate the slabs under the spawn first, then raycast to the ground (as now).
7. **Overlay:** slabs written / queued / cleared, server ms per frame, terrain MB, player logical
   position, distance to the window edge.

**Measure:**
- Does generation keep up while walking (16×2.78 ≈ 44 studs/s) and running? Estimate: one row of 32
  near slabs is ~0.6 s of server time, crossed every ~1.5 s.
- Is memory flat after a long walk?
- Are there holes at slab or near/crust boundaries?
- Try a 2–3× faster test speed to find the limit. Faster travel (vehicles, flying) would need a
  smaller near radius or Parallel Luau sampling (sampling is the cost).

## Part 2: treadmill jump (about every 8 km)

**Trigger:** the player's physical position passes `TREADMILL_STUDS` (start at 64k = 8 km; up to
~120k is possible) from the box centre on x or z. Vertical isn't needed yet: heights are ~1.6–3.4k
studs.

1. **Destination:** `newOrigin = origin + D`, with D the player's physical offset snapped to the slab
   grid, so the player lands near the centre.
2. **Pre-generate** the player's current desired set at the new origin: the same slab code, a second
   window with its own queue at a lower budget. It's out of view (more than 3k studs away), so time
   doesn't matter: ~25 s today.
   - Edits made meanwhile go to both windows (the swap test already does this).
   - Keep streaming the live window, and add slabs to the copy as the set moves.
3. **Jump** once the copy is complete:
   - The server sets `Origin`, then tells the client.
   - The client moves the character and camera by −D in one frame (as J does now).
   - Physics objects keep their velocity.
4. **Clean up:** clear the old window after the jump, behind the player and out of view.
5. **Edge cases:**
   - jump while airborne or in water;
   - an edit in flight during the jump: the server converts it with the origin of the moment it was
     sent; tag requests with an origin version;
   - a respawn during pre-generation;
   - Studio unfocused: render meshes don't build.
6. **Measure:** the stutter frame (195 ms on a fresh area), the pop, memory while two windows exist,
   and time from trigger to jump.

## Part 3: objects and multiplayer (later)

- **Object registry:** everything persistent is stored by logical position, and only objects in the
  window physically exist (streamed like slabs). On a jump, move them all with
  `workspace:BulkMoveTo`.
- **Multiplayer, shared origin:**
  - players near each other share the window union and jump together;
  - players far apart need separate regions of Roblox space ("slots") or terrain written on each
    client.
- **The client-terrain alternative:** one origin per player, so no slots.
  - Other players are drawn as local stand-ins: a clone whose parts copy the real ones each frame,
    offset by the origin difference, ~15–20 parts per player.
  - No character-to-character collision.
  - Server NPCs get simulated by the nearest client (network ownership).
  - Decide after part 2.

## Working notes

- Studio testing goes through the Studio MCP. Inject sources from
  `python3 -m http.server 34877 --bind 127.0.0.1` at the repo root with `HttpService:GetAsync`, set
  `HttpEnabled` back afterwards, then run `workspace.Terrain:Clear()` before Play.
- Only trust screenshots and frame times while Studio is focused: at ~15 fps unfocused it stops
  building terrain meshes.
- Rebuild the place with `rojo build terrain.project.json -o opti-voxel-terrain.rbxl`.
- Branches: PR #39 (far crust) and #40 (0.5 m scale) are open. `roblox-treadmill` sits on #40 and is
  pushed, with no PR yet.
