# Caves: a 3D noise carve density

The surface stays the climate's height field (`docs/climate.md`); caves are carved out of it by a second density made
of 3D noise, read only in depth bands below the surface. Islands, arches and cave mouths placed one by one stay
instances (`docs/trees.md`, biome features); this is the field-wide part.

## Built: caves as a graph

A cave is a node chain of a biome's graph, no carve kind: a smooth min on the mix's output (the mix builds the ground's
height), `h' = smin(h, up - C)`, which is the model's `smax(up - h, C)` below exactly. `up` is the sample's height
above the base (`SampleOp::UP`); the carve density `C` is a sub-mix of cave notes (the relaxer carries it to the
samples as a channel field, its octaves inserted by the cascade like any band) under the band's tent, all of it
ordinary nodes (`climate/caves_test.json`):

```
depth = mix - up
tent  = min(depth - top, floor - depth)
C     = min(noise + threshold, tent)
out   = smin(mix, up - C, k)
```

- **Bounds.** `bound_program` keeps each register's affine forms in `z = up - H` and `H`, so the tent bounds the term:
  `-floor - k/4 <= h' - H <= 0` (unit test: exactly `floor + k/4`). A graph reading `up` without a tent has no bound
  and is refused. The skirts and validate take that bound like any program's.
- **Where it moves h.** Below the band's top the term is nonzero wherever `C > -depth`, also deep in the rock: `h'`
  moves but `up - h' = C < 0` stays solid. Air opens only where `C > 0`, inside the band (unit test: columns through
  the band, air only between top and floor within the dip; GPU set `4v`, bit for bit).
- **Paths.** CPU `SDFCascade::ground_channels(slots, voxel, lod)` and GLSL `climate_ground(ch, p, lod)` read the base
  only where a program reads `up` (generic: always); the discoverer's corners (CPU and GPU point items) and the samples
  go through them, so the cull sees the caves.

Still the plan below: per-layer strengths per biome as a world table (the graph has them per biome today), the
per-chunk skip of the noise outside the band, the placement on the bare ground with the cave-mouth reject, the rock
material and no sky light underground, the skirt fix on ceilings, culling hidden caves, the `L`-scaled noise (a graph's
`C` is whatever its notes give: its gradient reaches the SDF's normalisation like the ground's), and a slice view in
the mixer.

## Model

The terrain's value today is `g(p) = up(p) - h(p)`, the height above the datum minus the ground's height (positive in
air). Caves add air where a carve density `C(p)` is positive:

```
d(p) = smax_r(g(p), C(p))                       air where either is positive, blended over r metres
C(p) = min(N(p), T(depth))                      the noise, only inside the band
N(p) = (sum_k a_k n_k(q(p)) - t) / L            the cave noise above its threshold t, scaled to metres
T(z) = min(z - top, floor - z)                  the band's tent: > 0 inside, the distance to it outside
depth = h(p) - up(p) = -g(p)
```

- `n_k` are the ground's own exact 3D octaves (`make_octave`, another seed): the same lattices, so the cascade carries
  them like any band, and every point gives the same bits from any origin, chunk or LOD.
- `q(p)` squashes the read along the base's up by `s` (a note's `stretch_y` code, `BaseSurface::stretched`): `s = 2..4`
  turns round blobs into flat, horizontal galleries.
- `L` is the noise's slope bound, `sum_k a_k * slope(perlin3) / lambda_k`, so `N` is 1-Lipschitz: a lower bound on the
  distance to the cave walls, never more. That keeps the discoverer's |d| > 2 chunk cull exact. Never `c / |grad c|`:
  it overestimates near saddles (unsafe for the cull) and underestimates badly far away. Under a `max()` that turns deep
  rock into "near the surface" and the skirts build walls kilometres down (`docs/gotchas.md`: the SDFDummy terrain had
  caves until 5cc9c76 and hit exactly this. Its fix, the distance to the band outside it, is `T` here).
- `smax_r` instead of `max`: a hard max creases the walls, and opposed gradients there cancel to a zero normal
  (`vertex_normal`'s fallback).

**Layers, the depth biomes.** A world has a few cave layers, each a band (`top_m`, `floor_m`) with its own notes,
threshold and squash: shallow tunnels at 5–60 m, big caverns at 200–600 m, and so on. The surface biome scales each
layer: a biome's `caves: {"<layer>": x}` multiplies that layer's amplitudes (0: none under it), blended by the
climate's weights like the ground's multipliers. So a karst biome is riddled, a desert has none. A layer is a biome
along depth: its own notes, faded in and out by its band as a surface biome is by its place.

Table form (draft):

```json
"caves": [
  {"name": "tunnels", "top_m": 4, "floor_m": 60, "threshold": 0.35, "squash": 3, "blend_m": 2,
   "amp": [{"from_m": 3.2, "to_m": 51.2, "x": 1, "shape": "ridged"}]},
  {"name": "caverns", "top_m": 200, "floor_m": 600, "threshold": 0.2, "squash": 2, "blend_m": 8,
   "amp": [{"from_m": 51.2, "to_m": 819.2, "x": 1}]}
],
"biomes": [{"name": "karst", "caves": {"tunnels": 1.5, "caverns": 1}, ...}]
```

Ridged notes give tunnels: air where `|n|` is small, along the noise's zero sheets. Where two sheets cross gives
worm-like tubes, which is the reason to allow two notes sets per layer later.

## Where it plugs in (from the code survey)

- **One place on both sides.** The cascade interpolates `h` and can carry the carve noise's sum the same way (a sum of
  bands in LINEAR4). The max, the threshold, the band and `smax` are not linear: they run at the samples only, like
  the biome curves, between `ground_channels` and `height_sdf` (`sdf_cascade.hpp:272-292`;
  GPU: `level_shader`'s `/*GROUND_C0*/`, `cascade_glsl.cpp:648-678`, and the point items' fills `:178-188`). The CPU
  and the GLSL stay op for op (`det_rsqrt`, `precise`, constants as bits).
- **The structural change.** The samples' level is compiled `without_fields()` and the point items `ground_only()`
  (`cascade.hpp:407-414`), so the carve can't be another climate field: it is a new channel group carried down to the
  samples in every view. That is about 4 more channels everywhere (8 measured 1.30x the GPU time of 4: G7,
  `docs/cascade.md`). The cost is skipped per chunk where the band can't reach it: a chunk whose corners are all above
  `top - r`, or all below `floor + r`, never reads the noise (the SDFDummy `cave_skip`, which saved two thirds of its SDF
  cost).
- **Per-layer strength** rides in `SampleCarry` (`climate.hpp:267`, `MAX_CARRY` 24) like the curve weights.
- **Discoverer** (`chunk_discoverer.cpp:51-67`): `evaluate_cull` returns `d` with the carve, so a cave chunk deep
  underground is no longer "solid" and is found. The GPU path (`CascadePointsGPU::cull_values`) must match it bit for
  bit (`--discovery_compare`).
- **Skirts** (`sdf_cascade.hpp:170-186`, `Climate::ground_step`): the carve's step between LOD L and L+1 is its dropped
  octaves over `L`, plus the tent's slope times `h`'s step, plus `BLEND_SLOPE` times the layer weights where a
  biome scales it (the curves' form). Bounded only because `N` is scaled by a constant.
- **What assumes a height field:**
  - Placement: `ground_along` (`sdf_cascade.hpp:380`) traces up from the datum and stops at the first air, possibly
    a cave. It and the cell-site walks (`ground_field`, `FieldGPU`) keep reading the bare ground `g`, plus a reject
    test where `C > 0` at a site (no trees over a cave mouth).
  - Shading: the slope blend gives cave floors the surface's flat colour, and past the 1 km shadow range caves are
    lit by the sun and cloud shadow (`page_shading_glsl.hpp:148`, `godot_page_host.cpp:147, 295-304`). Caves need a
    rock material by depth and no sky light below the surface.
  - The skirt height fix `push / max(dot(n, up), 0.2)` is wrong on ceilings.

## Stages (one PR each, measured)

1. **One cave layer in the engine, the world's, no biomes.** `C` on the CPU and GPU at the samples, the channel group,
   the per-chunk skip, the discoverer, the skirts.
   - Checks: a FrameSpike probe (`make_probes`), a CascadeSpike set (`cascade_check_sets.hpp`: G1 / G3 / D1 points and
     G10 bit-exact), unit tests (pinned bits unchanged with no `caves`, the L | L+1 step within the skirt, the cull),
     `discovery_audit.sh --discovery_compare=1` (its "underground" and "deep" poses: 0 misses), origin parity.
   - Measured, before / after on the video preset fly (`bench_ab.sh`): the chunk count with caves, the GPU cost of the
     channels, the skip's gain. This number decides stage 4.
2. **Layers and biomes.** The table's `caves` list and each biome's strengths, the climate report's per-layer map, the
   a carve node in the engine's graph format (one per layer, so the mixer shows it: its kinds are the engine's), and a **slice view** in the mixer: a
   vertical cut through the ground, so a layer is designed as easily as the surface.
3. **Living with caves.** Placement on the bare ground with the cave-mouth reject, the rock material by depth, no sun
   or clouds underground, the skirt fix on ceilings.
4. **Culling hidden caves** (if stage 1's chunk count needs it). Nothing occludes underground today: every cave chunk in
   the LOD ranges is meshed. The plan is a flood fill through a coarse air graph (32–64 m cells, "maybe air" from `C`'s
   bound per cell), started from the camera's cell, keeping what the camera's air can reach (Minecraft's cave culling,
   conservative). It is a pure function of the view, so it never changes the terrain's bits.
5. **Shaping.** Tilted strata (the squash axis from a direction field), depth layers `A cos(2 pi (depth + w(p)) / T)`
   for stacked galleries, two-sheet crossings for tubes, a karst network skeleton (paths as tube instances), water at a
   water table (`docs/look.md`).

## Open decisions (defaults proposed)

- **Layers per world, strengths per biome** (above) rather than caves per biome: one set of notes per layer keeps the
  channel count fixed however many biomes there are.
- **`smax` radius** per layer in metres (default 2 m tunnels, 8 m caverns); it also bounds the creases' normals.
- **The planet:** stretched (squashed) octaves are refused on a sphere today (`BaseSurface::stretched`); stage 1 ships
  `squash: 1` on a sphere and a flat base's squash, the sphere's waiting for that.
