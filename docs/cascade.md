# Hermite cascade

The kernel (`src/cascade.hpp`), the programs it runs (`src/level_program.*`, compiled by `src/graph_compile.*`: cascade
v2, `docs/cascade-v2.md`) and the GPU passes (`src/cascade_glsl.*`, `src/level_glsl.*`), bit-identical. Run
`tools/cascade_test.sh` after touching them.

- **The kernel.** A node of a world's `Terrain::LevelProgram` is computed on its level's lattice Λ_m (multiples of 2^m
  voxels; a note of octave k on Λ_{k-3}: 8 samples per wavelength) and carried down by tricubic Hermite subdivision.
  Channels in units of the lattice spacing: value + gradient, the slope along the axis by the Hermite rule, the
  transverse slopes linear. A LOD-L chunk stops at Λ_L, so `V_m(p)` is a pure function of the point: every LOD gets the
  same bits where they share points. Fixed order x, y, z; only + - * by 0.5, 0.125, 0.75; after each level |c| < 2^-64
  is flushed to 0 (without it a node carried far down makes denormals). 4 samples per wavelength (worst case 2.2% value
  error) was rejected. An 8-channel set (cross slopes from a `perlin3_cross`) cost 1.30x the GPU time of 4 against a
  1.15 threshold; it and its report tool are in git history.
- **CPU runs.** `Terrain::run(prog, box, lod)` runs a program from its top level down to Λ_lod over a box (level m: its
  inputs, the level above's live channels, subdivided and flushed, then level m's code); the last level runs
  `last_code(lod)`, the code of every level from lod down to 0, so the output (and the pick) exist there.
  `Terrain::point` is the same run on a one-point box, allocation-free once warm and limited to the nodes its targets
  read: a box and a point give the same bits. The op table (`op_def`: each op's CPU and GLSL form side by side) is the
  only definition of every op.
- **Folding (the constant relaxation, `Terrain::fold_lods`).** A run of LOD L reads every note finer than L as 0, so
  the last level used to run every finer level's ops on zeros: LOD 24 cost 6x LOD 0 on the example table. Now a node
  of absent notes and constants only is a constant from LOD `const_lod` up (computed once by its own op, `folded`), a
  product by such a 0 is 0, and the last level of LOD L skips what nothing it computes reads (`live_lod`) and stops at
  the finest level it still computes a node on (`last_stop`; a constant output or pick read as such). The CPU run,
  the point query and both GPU texts (a guard on the run's LOD per statement, per generic record) read the same
  fields. Exact: every op's result is flushed (no -0), so x + 0 and x * 0 give what the fold gives; a doctest compares
  folded and unfolded runs at every LOD of every test table, and CascadeCheck's G3 replays the last level unfolded
  against the GPU's folded output. Measured (`cascade_test.sh` G6, the example table, 128-chunk batches): the LOD 24
  batch 238 -> 99 ms (its last level 233 -> 99 ms), LOD 12 182 -> 108 ms, LOD 0 unchanged (51 ms: no finer level);
  the bare set's coarse LODs 0.88 -> 0.25 ms. What the last level of a coarse LOD still computes is real: the biome
  blend and its weights on the samples, the sums of its coarse notes (a doctest prints the counts per LOD).
- **Biome pruning (`Terrain::guard_biomes`).** Biome b's share of the ground is its products `w_b * g_b`; the nodes
  only those products (or other such nodes) read are b's own, and each gets `guard = b` on the levels finer than
  `w_b`'s. On every level above a run's last, the weight's statement sets bit b of its item's mask (`Values::mask`;
  the GPU's `biome_mask`, binding 13, a uint per item zeroed by `upload_items`, `atomicOr`) when `w_b` is nonzero
  anywhere on the item's box; a guarded node whose bit is off is 0 over the box. On the last level the test is per
  point (`w_b != 0` there). Exact: a weight that is 0 at every point of its level's box is 0 at every finer point
  (Hermite of zeros), so the product is +0 either way; the CPU run applies the same mask, and the point query ignores
  guards (it computes everything, the same bits). Statement order: dead, off, folded, computed. A doctest runs
  chunk-sized boxes over every test table pruned and not (same output and pick bits) and counts the runs a biome was
  off. The generic text always binds the mask (`uses_biome_mask`); a specialised program only with guards.
- **GPU cascade.** One dispatch per level for the whole batch (level L + j of every chunk, j from T minus the batch's
  smallest LOD down to 0), boxes in closed form from the chunk's anchor on (lo, hi) words (`GLSL_FRAME_INT64`,
  `Cascade::level_box`), a child point replays the CPU's x, y, z passes on its 2x2x2 parents (code generated from the
  same axis rule), ping-pong scratch A (odd j, 17^3 points per chunk) / B (even j >= 2, 10^3), j = 0 straight to
  `sdf_sample`. The level's own work is `level_run` (`level_glsl.hpp`): the channels it receives loaded into
  registers, its code, the channels it carries stored back; the pick's channel is copied from the floor parent, never
  interpolated. The notes reach the GPU as a table (`Terrain::band_table`, binding 5).
- **Specialised and generic.** The world's numbers are data (`Cascade::ShaderParams`): every value the generated code
  reads (the biomes' centres and spans, the amplitudes, shape coefficients, warp amounts, the curves' segments, the
  constants and smooth k, and the skirt depth per LOD) is a slot of a per-variant uniform block of vec4s at
  `CascadeGLSL::PARAM_BINDING` (the value's bits, constant indices, as cheap as literals; not a storage buffer, which
  WebGPU caps per stage; at most `MAX_PARAM_BYTES`), never a literal. Specialised (`SDFCascade::specialize`, the
  game's), the text is each level's code straight, with the program's value-dependent folds, except its notes: a
  level's notes are one loop over their range of the band table (`Terrain::note_order`) at the level's start, so the
  text holds one `perlin3` whatever the note count (one inlined call per note took the NVIDIA compiler minutes and
  9 GB on the example table's 80 notes; a doctest pins one `tp_note` call per text). Generic (the biome
  mixer's, `Terrain::Mode::GENERIC`), it is a fixed interpreter over the program's records: its text depends on the op
  set and the caps only (`MAX_CHANNELS`, `MAX_REGISTERS`), so an edit of a value compiles nothing, and the same
  generic program gives the same bits on the CPU. The two modes differ by blend factoring (a few float steps of the
  output's bound, a doctest bounds them). The Godot device keeps the SPIR-V of every text by text (`RDDevice`).
- **The fine variant.** One specialised text holds every level's code behind a switch on the level, so the GPU gives
  each thread the registers of the heaviest level (`world.json`: 168 per thread, a quarter of the warps). The levels
  `0 .. LevelProgram::fine_cut()` compute nothing (each note sits on a coarser level): `CascadePasses` compiles a
  second text of those levels' cases alone (`level_shader`'s `p_upto`: 51 registers on `world.json`, 96 with the
  instance fold) and a dispatch runs it when the batch's coarsest chunk's level, its largest LOD + j, sits there. The
  same statements in the same order: the same bits. The generic text stays one (an edit compiles nothing). Measured
  (`cascade_test.sh` G6 on `world.json`, 128-chunk batches): the last level at LOD 0 1.52 -> 0.42 ms (v1: 0.75), a
  batch at LOD 0 2.31 -> 0.66 ms; LOD 8 and 12, whose last level runs code, are unchanged (the next lever: a text per
  level range). Registers per text: `VK_KHR_pipeline_executable_properties` on the NVIDIA driver.
  Engine constants (the kernel's fade, the noise's means) stay literals. `CascadePasses` owns the passes per device
  (`ov::gpu::Device`); the mesher runs them in place of its sdf pass (and in the discovery audit).
- **The cascade only exists on lattice points.** `SDFCascade::evaluate(p, lod)` takes `frac == 0` and a voxel
  multiple of 2^lod only (ERR and 0 otherwise); the discoverer's corners and the mesher's cell corners are
  such points, `get_sdf_at` snaps to the LOD0 voxel. There is no off-lattice evaluation and no GLSL
  `terrain_sdf`: vertex normals are the trilinear blend of the cell's 8 sample gradients
  (`CascadeGLSL::GLSL_VERTEX_GRADIENT`). A new cascade terrain keeps this rule.
- **Tests.** `test_cascade.cpp`: the boxes, the Hermite copy below the notes bit for bit, slopes against finite
  differences, the fidelity of one octave against perlin3, no denormal and FTZ/DAZ changing no bit;
  `test_level_program.cpp`: a level's purity across runs, box == point on every test table, the LOD step within the
  skirt, the weights op for op, one bit pin per test table; `test_sdf_cascade.cpp`: `evaluate` = `point` +
  `height_sdf`, the default terrain's pin.
