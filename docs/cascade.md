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
