# Cascade v2: one graph, coarse to fine

Status: **implemented** (stages A to F, below): `SDFCascade` runs `LevelProgram` (`level_program.*`), compiled by
`graph_compile.*`, its GLSL forms in `level_glsl.*`; the old machinery is deleted; the biome mixer edits v2 graphs.
Stage G (guards, fold tuning, the bench table) is open. This note says what the terrain evaluation became, what it deleted
and in which order; it is the contract the v2 PRs were reviewed against. `docs/cascade.md` and `docs/climate.md`
describe the code as it is. v2 deliberately broke the bits of every existing world once: the old terrain is not an
oracle, and nothing below exists to reproduce it. (The *Why* and *What goes* sections name v1's code, now gone.)

## Why

The ground is computed today by five mechanisms stacked on the Hermite cascade, each with its own record layout, CPU
mirror, GLSL and parser rules:

| Mechanism | Where | What it does |
|---|---|---|
| constant bands | `Cascade::Spec::bands`, `band_table` | a note every biome plays plain at one amplitude |
| modulated bands | `BandMod`, `Modulation`, `climate_modulator.cpp` (RECORD = 11, `plan()`) | per-biome amplitudes on n, abs(n), n², blended by the weights |
| warp fields and terms | `Field::warp`, `WarpTerm`, `cascade_warped` | a note read at a displaced point |
| multifractal parent fields | `parent_field`, `Multifractal` | a note weighed by a coarser note's height |
| sub-mix channels and the sample program | `~submix` fields, `SampleProgram`, `run_program` / `program_glsl` / `interpreter_glsl` | nonlinear ops, run at the samples on the blended H |

They meet in the relaxer (`climate_parse.cpp` `expr`, `node_value`, `per_octave`), the biome compiler's `gather` /
`fold` / `channels`, `FieldRange` and the carry groups, `MAX_LEVEL_NOTES` extra notes and the generic caps padding
(`~warp%d`). Most refusals the parser makes come from that implementation, not from the math: one note per octave in a
sub-mix, a multifractal weighing one note through a note-only parent, a warp whose input is plain notes, an octave
modulated for all its notes or none. Each new node kind so far added a sixth special case.

The general form behind all five is one rule.

## The rule

- **A node's level** `ℓ(n)` is the level of the finest note it depends on. A note of octave k is at level `k − 3`
  (inserted on Λ_{k−3}: 8 samples per wavelength, `BAND_SHIFT`, as today). `constant` and `up` have no level: they are
  exact at every lattice point (a constant, and an affine function of the position, which Hermite subdivision carries
  exactly).
- **A node is computed on its level's lattice, from its inputs' values there, and Hermite-subdivided below.** Each
  node has exactly one definition, a function of the absolute lattice point, so every LOD and every origin gets the
  same value where they share points. A nonlinear node (min, a curve, a shape) is *defined* this way, not
  approximated: a crease made on Λ_m is rounded at the spacing of Λ_m, which is the band limit the LOD needs anyway.
- **A run of LOD L stops at Λ_L** (as today). A node with `ℓ < L` is computed on Λ_L with the notes finer than L
  absent. The step this makes between LOD L and L + 1 is what the skirts cover (below).
- **The gate.** A node that changes *where a note is read* must have its control strictly coarser than that note:
  the control has to exist before the note is inserted. Only the **warp** is a gate. Its input is any node with
  `ℓ(input) > ℓ(note)`. A note's `stretch_y` is a fixed parameter of the read, not a node.
- **Everything else is ungated**: add, subtract, multiply, mix, min, max, smin, smax, curve, shape, constant, field,
  up. Each is computed at the finest level of its inputs.

Consequences that the parser stops refusing:
- a shape on anything (a mix, a sub-mix, a curve's output);
- several notes on one octave anywhere (no per-level note table);
- a warp of any coarser node, for example a ridged one;
- a multifractal parent that is any node.

### Sugar (expanded by the compiler, no engine code of their own)

- **multifractal** `{in, parent, low_m, high_m}` = `multiply(in, ramp(parent))`, where `ramp(h) = clamp((h − low_m) /
  (high_m − low_m), 0, 1)` is a two-point curve. It scales the note without moving it, so it is ungated and its parent
  can be any node. A coarser parent is Musgrave's chain; a parent finer than `in` puts the product at the parent's level.
- **biomes.** A field (temperature, humidity, wind) is a graph of notes (its octaves, its terms on earlier fields).
  A biome's weight `w_b` is a node: its `at` kernels and the default biome's `(1 − Σk)²`, normalised (`docs/climate.md`,
  Weights). The ground is `Σ_b w_b · out_b`, where `out_b` is biome b's graph output. The weights' level is their
  fields' (coarse: 6.5–26 km notes), so the climate is computed a few times per chunk and subdivided, like today's carry.
- **the pick** (the sample's biome: material, species, features) is a discrete node: the largest weight, ties by
  `hash_cell`. It is carried by copy (the floor parent's value, today's `copied_carry`), never Hermite-subdivided.

### Format changes (world files)

- **warp:**
  - `in` is any coarser node, read three times with its notes' seeds + 0, + 1, + 2 for x, y and z (libnoise's
    turbulence, as today);
  - optional `x`, `y`, `z` ports instead give each axis its own node.
- **multifractal:** `parent` is any node.
- **field terms:** the `"source": "ground"` term goes. Only the test tables `plain.json` and `shaped_test.json` use it,
  `world.json` doesn't. The ground depends on the weights, which depend on the fields, so the term was a cycle broken
  by a level-above read. A field's terms read earlier fields only.
  The graph is acyclic and the parser says where it isn't.
- **unchanged:** everything else (notes, mixes and rows, shapes, curves, programs' kinds, `at`, biome keys, engine.json,
  climate.json fields).

## Programs

The compiler turns a world into one **`LevelProgram`**: for each level m, from the top down, a list of instructions
over registers.

- **Inputs:** the channels carried from Λ_{m+1} (subdivided) and the noise reads of the notes inserted at m (perlin3
  at the note's seed and octave, stretched, at the warped point).
- **Values:** dual numbers, value plus gradient in the lattice's units, so every node has its slope. The existing ops
  are reused: `dadd`, `dmul`, `dscale`, `smooth_abs` (`cascade.hpp`), smooth min (`climate_shape.cpp`) and the curves'
  segments.
- **Outputs:** the channels a finer level reads. A channel lives from the level that computes it to the last level
  that reads it (liveness); the output node is a channel to level 0, where `height_sdf` and the skirts turn it into the
  sample (`cascade_glsl.cpp`'s level-0 output, unchanged).

### Compiler passes

1. **Parse:** `KINDS` and `node_keys` as today (`climate_parse.cpp`), the acyclic check, the gate check, the sugar
   expanded.
2. **Identity:** a note is one node whatever reads it (octave, seed, stretch, warp node). A biome more costs the nodes
   it adds, as the biome compiler's `gather` does today.
3. **Levels:** `ℓ` of every node, bottom up; each instruction placed at its node's level.
4. **Fold:**
   - constants;
   - dead nodes (unreachable from the output, a weight of an absent biome, a note at amplitude 0);
   - **linear crossings**: Hermite subdivision is linear, so a linear combination of channels crossing a level is
     carried as one channel. A plain fBm is one channel from top to bottom, as the ground band is today.
   - **blend factoring**: a node every biome plays alike comes out of `Σ w_b ·` (today's constant bands, as an algebra
     rule).

   Value-dependent folds (zeros, equal amplitudes) run only in the specialised program; the generic program keeps the
   graph's structure so an edit never changes it.
5. **Bounds:** see below.

### Backends: one op table, three forms

Each op is written once, in a table that holds its CPU form and its GLSL form side by side. The same float discipline
as today applies (`precise`, no fma, `-ffp-contract=off`, no `mix` / `dot` / `normalize` / `sqrt` / division in a
mirrored path, constants as bits, the flush under 2^-64), so **the CPU and the GPU give the same bits for v2**.

Placement relies on that:
- trees and rocks are placed on the CPU's or FieldGPU's ground over a GPU mesh, with a per-batch CPU fallback
  (`docs/instance-stream.md`);
- the player's feet and the camera use `World::sdf_at`;
- the discoverer's CPU and GPU scans are compared.

Writing each op once is what makes keeping it cheap.

| Backend | Used by | Form |
|---|---|---|
| CPU evaluator | `SDFCascade` CPU paths (`evaluate*`, `ground_field`, `ground_along`, `channels_at`), BiomePick, CloudClimate, ClimateReport, the tests | the program interpreted per box (and per point: `point_slots` on a one-point box, as today) |
| specialised GLSL | the game | per level, straight-line code, the values in the `ShaderParams` block (binding 12) |
| generic GLSL | the biome mixer | a fixed interpreter over the program's records in `ShaderParams`: its text depends on the op set and the caps only, so no edit recompiles |

The GPU keeps the current machinery around the program:
- one dispatch per level for the batch;
- closed-form boxes;
- scratch A/B, planar;
- `hermite_functions` / `storage_functions`;
- the level template and the fold variant for instances.

`Ch` becomes the level's live channels.

### Bounds and skirts

Each node gets a value bound and a Lipschitz bound in each input, generalising `bound_program`'s affine Form / Range
(`biome_compiler.cpp`):
- add and mix: 1 per input, times the row volume;
- multiply: the other operand's bound;
- curve: its largest segment slope;
- min, max and smin: 1;
- shape: its coefficients' bound (`shape_bound`);
- warp: the warped note's slope bound times the warp's change.

The **step** of LOD L is the output's largest change when the notes inserted at L are dropped, propagated through these
bounds. That includes a weight's change times its biome's output bound, which `Climate::ground_step` does by hand today.
`skirt_depth_table[L]` is that step plus the instances' erosion (`min_skirt_depth`), as today.

Validation keeps the limits that protect the skirts and the exact range:
- the output bound under `amp_cap`;
- a curve's slope under `MAX_CURVE_SLOPE`;
- the noise's exact range.

It drops the ones that only protected the record layouts.

### Caps

| Cap | Today | v2 |
|---|---|---|
| live channels per level | `MAX_FIELD_GROUPS` 24, `MAX_CARRY` 24, `SLOTS` 196 | one `MAX_CHANNELS`, sized by the GPU's registers (start at today's 48 groups) |
| notes per octave | `MAX_LEVEL_NOTES` 8 | none (instructions) |
| instructions | `MAX_SAMPLE_INSTRS` 64 at the samples | one cap per level |
| biomes | `MAX_BIOMES` 16 | kept if the weights' channels need it |
| shader params | 16 KB | `MAX_PARAM_BYTES` 32 KB: the generic records of `example.json` take 17.7 KB (registers: `MAX_REGISTERS` 96) |

## What stays

- **Noise and frame:** `exact_noise` (perlin3, octave seeds, `make_octave`), the int64 frame, `GPUFrame` words.
- **The kernel:** Hermite subdivision (`hermite_mid` / `hermite_copy`, x then y then z, the flush), lattices and boxes
  (`level_box`, `item_box`, `run_top`).
- **The ground:** `height_sdf`, the base surface and its regions, the skirt formula at level 0.
- **Instances:** the instance table and its fold variant.
- **`CascadePasses`' interface** (`create`, `level_count`, `record_level`, `upload_items`, the bindings). GPU Direct, the
  page pool and the page shader don't change.
- **`SDFCascade`'s public API** (`evaluate`, `evaluate_view`, `evaluate_cull`, `ground_unit`, `ground_field`,
  `ground_along`, `skirt_depth_m`, `ground_material*`). Its callers (discoverer, FieldGPU, scatter, features, World)
  don't change.
- **The JSON format,** apart from the changes above.

## What goes (gone)

- **Modulation and bands:**
  - `climate_modulator.cpp`, `cascade_shaped_glsl.*`;
  - `BandMod`, `Cascade::Modulation`, `MOD_*`, `note_mod` / `group_mod`;
  - the extra notes and the band table's per-level ranges.
- **Carry rules:** `FieldRange`, carry groups and `copied_carry` as special slots (they become channels), `SampleCarry`,
  the three GPU Views (ground / fields / samples), `Spec::stretched()` forcing.
- **The relaxer and old compiler:**
  - the relaxer, `GraphRow`, `per_octave`, `rank` and the GROUND lift;
  - `biome_compiler.cpp`'s `gather` / `fold` / `channels` (its bounds move to the compiler);
  - the generic caps padding (`~warp%d`, `~submix%d`, `~parent…` fields) and `World::Kept.caps`.
- **The split interpreters:** `climate_shape.cpp`'s `run_program` / `program_glsl` / `interpreter_glsl` /
  `program_records` (one op table instead).
- **Climate glue:** `Climate::state`, `apply_terms`, `derive`, `ground_value` / `climate_ground` as separate paths; the
  ground term of fields.
- **Refusal rules** that only protected the layouts (listed in *The rule*).
- **Engine settings that are data:** the template biome built in C++ (`template_graph`) became world files
  (`worlds/template_5cm.world.json`, `template_50cm.world.json`: a file per voxel size, since its octaves are counted in
  voxels; `SDFCascade::template_path` picks it, a missing or refused one falls back to the flat table with an ERR). Its
  cap on a small planet is data too: the template's 3.3 km octaves are over a 2 km planet's relief and refused there,
  so `tiny.world.json` carries its own ground, the template's octaves stopped under the relief. `SDFCascade::Preset::
  FLAT` and `terrain_mode` stay: the flat ground is already a constant table (`flat_table`), and the preset also picks
  the shapes gallery's ground height, skirts and spawn in World and `player.gd`; folding those into world files is
  left for later.
- **Pins of the old terrain:**
  - `plain.json`'s hashes;
  - the generic == specialised pins on the old tables;
  - "neutral modulation leaves the ground bit for bit";
  - the FrameCheck probes of deleted functions.
  (The default-terrain hash of `test_sdf_cascade.cpp` kept its v1 value: the template is a plain fBm, one linear
  channel, so v2 gives it v1's bits, and its world files are the old C++ graph node for node.)
- **The legacy mesher and its oracle** (this PR, below).

## Checks after v2

| Check | Becomes |
|---|---|
| CascadeCheck G1 (GPU == CPU every level) | the same on `LevelProgram`: every channel, every level, bits |
| G3 (LOD relation) | per chunk: the GPU's level L + 1 channels subdivided on the CPU, then `last_code(L)`, against the GPU's output and pick |
| G9 (fold samples), G10 (materials), D1 (points) | unchanged in meaning, on v2 |
| `--generic=1` | every check on the generic text (GPU generic == CPU generic program, bits); generic vs specialised differ by blend factoring, so a doctest bounds them on the CPU (a few float steps of the output's bound) |
| unit tests | level rule, gate refusals, linear fold (an fBm is one channel), blend factoring, box == point, the L / L+1 step within the skirt on every test table, a ridged multifractal parent |
| new pins | one hash per test table (`test_level_program.cpp`, taken at stage E): they guard v2 against itself, not against the old terrain |
| `--walk_verify`, `--discovery_compare`, origin parity, `--destroy_verify`, audits | unchanged |

## What looks different

Every world changes bits. Visible differences are expected where:
- a node was nonlinear on coarse inputs. Today it runs at the samples, sharp; in v2 it is computed at its level, rounded
  at that lattice's spacing. Curves of the full ground stay at level 0, as today.
- a biome blend crosses a program. Today each program reads the blended H; in v2 each biome's graph reads its own notes
  and the outputs are blended.
- blend factoring makes `Σ w_b` multiply a shared note, while today a constant band is inserted outside the blend.
  The difference is at most the weights' normalisation error.

## Cost model

- **Linear graphs:** a plain fBm and today's constant bands cost what they cost now: one channel, one noise read per
  note.
- **Nonlinear coarse nodes:** they are cheap. Coarse lattices have about 8x fewer points per level, and the result is
  one channel where today a sample program carried each input as a sub-mix field.
- **The real cost is live channels:** value plus gradient per point per level, in scratch and in registers. The
  compiler's liveness and linear folds keep the count down; `MAX_CHANNELS` bounds it.
- **Biome blends:** a point in a blend runs each present biome's own nodes. A per-chunk guard keeps a chunk to its
  biomes: a biome whose weight's bound is 0 over the chunk is skipped. This is new; the old region guards went with the
  regions in the climate declutter.
- **Generic mode:** it pays the interpreter, as today; specialised mode emits straight code.

Measured, not assumed: PR G carries a `tools/bench_ab.sh` table against dev (the default bench, and `--preset=forest5cm
--scenario=fly` at error 2) and names the new bottlenecks.

## Stages

| PR | Content |
|---|---|
| **A + B** (done) | the legacy mesher and `--validate` deleted (GPU Direct is the only pipeline); this note |
| **C** (done) | `LevelProgram`, the op table's CPU form, the compiler; doctests above |
| **D** (done) | the GLSL forms (specialised, generic); `SDFCascade` on v2; CascadeCheck on v2 |
| **E** (done) | deletion of *What goes*; tests ported, test tables re-checked and pinned, the template biome as world files (the flat preset kept, above) |
| **F** (done) | biome mixer: node kinds from the engine's catalogue (it already built its nodes from `graph_kinds`; its own loop refusal deleted, the engine's error names the node; the tooltips' v1 rules rewritten), the warp's `x` / `y` / `z` ports in `graph_kinds` (all three or `in`, each gated), any multifractal parent, worlds migrated on read (a field's ground term dropped) |
| G | the constant relaxation per LOD (`fold_lods`, `docs/cascade.md`: a node of absent notes is a constant, the last level skips what nothing reads); guards; the bench table; `docs/perf.md` |

Each PR updates the docs it makes wrong: `cascade.md`, `climate.md`, `terrain.md`, `testing.md`, `gotchas.md`,
`integer-frame.md`, `instance-stream.md`, and the CLAUDE.md invariants (the "CPU mirrors" rule points at the op table
since stage E).

## Open questions

1. **Sharp nodes:** should a node be able to ask to be computed at the samples (`"at": "samples"`), for a sharp
   crease from coarse inputs? Not planned until a world needs it.
2. **Blend semantics:** do biome programs read their own notes (proposed) or the blended ground (today)? The proposal
   is simpler and has no cycle. A world that wants "the blend" mixes explicitly.
3. **`MAX_CHANNELS`:** start at 48 (today's slot budget) and lower it after G's measurements?
4. **Weights' channels:** carry every biome's weight (today's carry) or only the biomes a chunk's guard keeps?
