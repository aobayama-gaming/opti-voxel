# Hermite cascade

CPU reference (`src/cascade.hpp`) and GPU passes (`src/cascade_glsl.*`), bit-identical. Run `tools/cascade_test.sh` after touching them.

- Hermite cascade (`src/cascade.hpp`: CPU reference, the oracle; `src/cascade_glsl.{hpp,cpp}`: the GPU
  passes, bit-identical to it): each noise band (octave k) is inserted once on the lattice Λ_{k-3}
  (multiples of 2^(k-3) voxels: 8 samples per wavelength), then carried down by tricubic Hermite
  subdivision. Channels in units of the lattice spacing: `HERMITE8` (value, 3 slopes, 3 cross slopes,
  d3/dxdydz from `perlin3_cross`) or `LINEAR4` (value + gradient, transverse slopes linear). A LOD-L chunk
  stops at Λ_L, so `V_m(p)` is a pure function of the point: every LOD gets the same bits where they share
  points (a climate's last level reads its octave's amplitude from the level above: Carry groups, below). Fixed order x, y, z; only + - * by 0.5, 0.125, 0.75; after each level |c| < 2^-64 is flushed to 0
  (without it a band carried far down makes denormals). `Cascade::point` = the same run on a one-point box.
  4 samples per wavelength (worst case 2.2% value error) was rejected (`tools/cascade_report.sh`); the
  meshed set is `LINEAR4` (G7 of the design: 8 channels cost 1.30x the GPU time of 4, threshold 1.15).
- GPU cascade: one dispatch per level for the whole batch (level L + j of every chunk, j from T minus the
  batch's smallest LOD down to 0), boxes in closed form from the chunk's anchor on (lo, hi) words
  (`GLSL_FRAME_INT64`, `Cascade::level_box`), a child point replays the CPU's x, y, z passes on its 2x2x2
  parents (code generated from `Cascade::PAIRS`), ping-pong scratch A (odd j, 17^3 points per chunk) / B
  (even j >= 2, 10^3), j = 0 straight to `sdf_sample`. Bands come from a table indexed by level
  (`CascadeGLSL::band_table`). `CascadePasses` owns the passes per device (`ov::gpu::Device`); the direct mesher runs
  them in place of its sdf pass (and in the discovery audit), the legacy one in sub-batches of 128.
- **Carry groups** (`Cascade::carry_slot`, `Modulation::derive`): LINEAR4 slots after the field groups that a run's
  level above its last writes after its insertions and flush, and H carries to the last level like a field (the
  `copied_carry` group takes its floor parent's channels instead of a blend: a value that can't be interpolated, the
  climate's pick). The climate uses them so that a run's last level, its samples, computes no climate at all
  (`docs/climate.md`). `FieldRange` says where the fields (`carried`) and the carry (`carries`: Λ_{L+1} and Λ_L) are,
  CPU and GPU alike; the GPU compiles a view per kind of level (`CascadeGLSL::View`: the ground alone, ground + fields,
  which derives at j = 1, ground + carry at j = 0) on the same planar scratch, the carry loaded and stored only up to
  the groups the last level reads without the fields (`Modulation::carry_used`: the samples' terms and the ground's
  parts; the GPU's carry, `Spec::gpu_carry_groups`; the CPU derives the fields' too, for `fields_at`), the fields
  stored on every level that carries them. The rules reach the GLSL as per-LOD tables generated from `FieldRange`
  (no formula retyped). A run whose last level reads the carry starts at least one level above it, empty above the
  highest band (`Spec::run_top`, `checked_top`, the GPU's `top`).
- **The cascade only exists on lattice points.** `SDFCascade::evaluate(p, lod)` takes `frac == 0` and a voxel
  multiple of 2^lod only (ERR and 0 otherwise); the discoverer's corners and the validator's cell corners are
  such points, `get_sdf_at` snaps to the LOD0 voxel. There is no off-lattice evaluation and no GLSL
  `terrain_sdf`: vertex normals are the trilinear blend of the cell's 8 sample gradients
  (`CascadeGLSL::GLSL_VERTEX_GRADIENT`), in both meshers. A new cascade terrain keeps this rule.
