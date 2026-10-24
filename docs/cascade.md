# Hermite cascade

CPU reference (`src/cascade.hpp`) and GPU passes (`src/cascade_glsl.*`), bit-identical. Run `tools/cascade_test.sh` after touching them.

- Hermite cascade (`src/cascade.hpp`: CPU reference, the oracle; `src/cascade_glsl.{hpp,cpp}`: the GPU
  passes, bit-identical to it): each noise band (octave k) is inserted once on the lattice Λ_{k-3}
  (multiples of 2^(k-3) voxels: 8 samples per wavelength), then carried down by tricubic Hermite
  subdivision. Channels in units of the lattice spacing: `HERMITE8` (value, 3 slopes, 3 cross slopes,
  d3/dxdydz from `perlin3_cross`) or `LINEAR4` (value + gradient, transverse slopes linear). A LOD-L chunk
  stops at Λ_L, so `V_m(p)` is a pure function of the point: every LOD gets the same bits where they share
  points. Fixed order x, y, z; only + - * by 0.5, 0.125, 0.75; after each level |c| < 2^-64 is flushed to 0
  (without it a band carried far down makes denormals). `Cascade::point` = the same run on a one-point box.
  4 samples per wavelength (worst case 2.2% value error) was rejected (`tools/cascade_report.sh`); the
  meshed set is `LINEAR4` (G7 of the design: 8 channels cost 1.30x the GPU time of 4, threshold 1.15).
- GPU cascade: one dispatch per level for the whole batch (level L + j of every chunk, j from T minus the
  batch's smallest LOD down to 0), boxes in closed form from the chunk's anchor on (lo, hi) words
  (`GLSL_FRAME_INT64`, `Cascade::level_box`), a child point replays the CPU's x, y, z passes on its 2x2x2
  parents (code generated from `Cascade::PAIRS`), ping-pong scratch A (odd j, 17^3 points per chunk) / B
  (even j >= 2, 10^3), j = 0 straight to `sdf_sample`. Bands come from a table indexed by level
  (`CascadeGLSL::band_table`). `CascadePasses` owns the passes per RenderingDevice; the direct mesher runs
  them in place of its sdf pass (and in the discovery audit), the legacy one in sub-batches of 128.
- **Fields stop at their node level; below, one interpolation** (the fold: `Spec::node_level`, `Cascade::node_interp`,
  `src/cascade_fold_glsl.cpp`). A field group with bands exists as lattice values down to its node level ν, its finest
  insertion level (at most `NODE_MAX` = 24, so the dyadic offset stays exact in float32). Below it a point of Λ_m reads
  it from the 8 nodes of its Λ_ν cell: H's per-axis rule at t = (p mod 2^d) 2^-d, d = ν - m (cubic Hermite and its
  derivative on the axis pair, transverse slopes lerped, every product flushed), then the slopes times 2^-d. It happens
  once per point, before the level's insertions, for every reader: the modulation, the ground terms, the pick,
  `fields_at`. The ground, the multifractal state and band-less fields are carried at every level. Per group, not per
  table, so a field shared by two tables keeps the same bits in both (world.json's region cores). At Λ_ν's points and
  one level below it gives the carried cascade's bits; deeper it differs by the cross derivatives LINEAR4 drops at
  every level (world.json: about 0.1% of a smooth field's amplitude in value, 3% in gradient, `test_climate_fold.cpp`;
  up to 13% at rare points of its ridged fields). On a 210 km `climate_report.sh` slice of world.json: 19 picks of 262,144
  change, the ground moves 7 cm on average (99.9th percentile 2.8 m, all in the region borders' bands), the hillshade
  looks the same.
  - GPU: a group's nodes are read in place, in the scratch plane its node level wrote. A level only writes the planes
    it carries (`g_carry_<ν>` per node level, per item), and a fold spec gives each item a fixed scratch region
    (`cascade_stride`, `CascadePasses::item_stride`). With today's c n_j^3 slots, a mixed-LOD batch's coarser chunk
    would overwrite a finer one's node plane.
  - Specs without a fold (no climate, a fieldless table) generate the shader text of before, byte for byte (checked
    against dev for every variant).
- **The cascade only exists on lattice points.** `SDFCascade::evaluate(p, lod)` takes `frac == 0` and a voxel
  multiple of 2^lod only (ERR and 0 otherwise); the discoverer's corners and the validator's cell corners are
  such points, `get_sdf_at` snaps to the LOD0 voxel. There is no off-lattice evaluation and no GLSL
  `terrain_sdf`: vertex normals are the trilinear blend of the cell's 8 sample gradients
  (`CascadeGLSL::GLSL_VERTEX_GRADIENT`), in both meshers. A new cascade terrain keeps this rule.
