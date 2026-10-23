# Materials

Per-chunk palettes and per-biome ground materials.

- **Materials (per-chunk palettes).** A record carries a global material (`TableRecord::material`, ids from
  `Material::Library`, `src/material_library.*`: built-ins ground, bark, leaves, lava, scorched rock, island rock;
  id 0 `GROUND` is the page shader's procedural slope / height colours; ids < 2^24). By default it is its type's
  (`DefLibrary::add(def, material)`, `Material::FROM_TYPE` resolved by `SDFTable::finish_record`: branch bark, crown
  and cone leaves, basin scorched rock, island island rock; the generic shapes (sphere, capsule, rock, cylinder, the
  shapes gallery) are `Material::INHERIT`, the lava slab sets lava), so runtime `add_instance` edits get theirs too.
  An `INHERIT` cutter colours nothing: the fold skips its win, so the sample keeps what it had (the dig tool's
  capsule carves a tree and the hole shows bark). On a union it resolves to ground (`Archetype::record_material`, in
  `SDFTable::finish_record`): an inheriting union would take whatever record won the sample before it in list order. It takes the list word's reserved
  palette index `ChunkPalette::INHERIT_INDEX` (511) and no palette entry. A corner stores a 9-bit index into its
  chunk's **palette** (`src/chunk_palette.*`: entry 0 ground, then the listed records' materials in list order, at
  most `ChunkPalette::MAX_ENTRIES` = 511; past that the rest draws as ground, `palette_truncated`), so ids are unbounded and a chunk only carries
  what it uses. `build_lists` builds the palettes on the worker pool and packs each entry's index above the record id
  in the list word (`SDFGpuTable::list_word`; `LIST_ID_BITS` = 32 - 9: **a table holds at most 2^23 = 8M instances**,
  removed ones included; a separate per-entry index would lift it). The fold keeps the palette index of
  the record that wins (`record_wins`: the operator's own comparison, so distances are untouched; a subtraction's
  carved face takes the cutter's material unless it inherits; CPU `fold_onto` == GPU `arch_fold`, FrameSpike's table probe), the cascade's
  fold variant's level 0 writes it per sample for chunks with a list (`sdf_material`, cascade binding 9), and the vertex
  takes the material of its cell's solid corner closest to the surface (`cascade_vertex_material`, ties in x + 2y +
  4z order; both meshers), except in a chunk on the ground block (direct) or without a list (legacy): material 0, no
  reads.
  `PalettePool` (`src/palette_pool.*`, a buddy allocator of 16-512-entry blocks in one RGBA16UI texture, 2 ids per
  texel, `GPUDirectMesher::Settings::palette_capacity` = 2M entries, 8 MB) holds the loaded chunks' palettes: placed
  at dispatch, its texels written by the emit pass (a texel per thread, `PalettePool::glsl_batch`), the block offset
  in each row header (texel 1 `.zw`, so compaction moves it with the rows), freed with the chunk (or at the end of
  the batch's results for the chunks that loaded nothing); a `{GROUND}` palette, or one that finds no room (`full`),
  uses block 0 and draws as ground. The page shader reads header -> palette -> id -> `material_tex`
  (`Material::Library::make_image`, owned by the mesher) for corners of index > 0, and replaces the ground colours with
  the material's albedo, roughness and emission (lava: `PATTERN_LAVA`'s crust, leaves: `PATTERN_LEAVES`' clumps;
  `material_shading`). **Blend:** corner A (the
  provoking vertex) reads the quad's other corners: a = its material, b = the first other one, plus a mask of the corners
  (B, C, D) that are not a, packed above b's id (flat `ivec2 v_materials`); the fragment interpolates the mask over the
  base quad's uv (no new interpolant) and shades the higher id of the pair where the weight towards it, jittered by a
  value noise of about a cell of the chunk's LOD, passes 1/2 (towards the higher id, so two quads holding the pair in
  either order agree on their edge). A third material inside one quad draws as b. **Cost:** a per-pixel choice of
  material costs ~1.3 ms of opaque pass and ~0.4 of pre-pass in the forest at 1080p (+6% render GPU), whatever its
  form (select, branch, two shadings, mix) and even when no quad holds two materials: it is the shader's compiled form,
  not executed work (feature/material-blend's bisection; 2 smooth vec4 of per-vertex parameters cost more, +2.2 ms).
  `--view=9` shows the ids, blended (V reaches it in the player). Legacy
  carries the id in `ARRAY_CUSTOM0`; the validator (half its sample from chunks with materials,
  `MeshValidator::pick_targets`) requires `material_mismatch == 0` on paired vertices, the auditor
  `bad_material == 0` (index past the palette) and the headers' blocks. Stats: `stats.json` `palettes` (direct) and
  `table.list_len_per_lod.*.palette_avg / palette_max`. **Bark** (`PATTERN_BARK`, `bark_grain` / `bark_plate` in the page
  shader): a 3D noise with no surface frame: a smooth grain stretched along world up (plates, V grooves ~11 cm apart), a
  sparse cell field of knots (2 x 2 x 2 nearest cells) bending it; colour only (a bump and a roughness following the
  plates cost ~1.3 ms of depth pre-pass in the forest at 1080p, which writes normals and roughness for SSAO / SSIL, for
  relief barely visible); it fades to the plates' mean past ~4 pixels per groove (metres per pixel from the projection).
  It is read at `bark_point`: the normal's horizontal direction at 0.45 m (~25 grooves around any trunk) and the height,
  so moving the surface along its normal (erosion, skirt push, the mesher's error) reads the same bark; read at the
  position, it jumped at LOD changes (70% of the pairs across one vs 27%, `tools/bark_seams.sh`). Trees share their bark at
  one height, thick trunks get wide plates, and its axis is world up (a near-horizontal branch gets grooves around it).
  The page shader's lattice noise hashes its cells with integers (`cell_hash`, `GLSL_PCG3D`): a float hash of
  coordinates in the thousands made every lattice plane a seam (the compiler reassociates `(i + 1) * k`, the hash
  amplifies the last bit), the bark's horizontal lines. **Grading:** the scenes' Environment has adjustments on and a 32^3 LUT
  (`res://grade_lut.png`, a Texture3D of 32 slices, written by `tools/make_lut.py`: lifted cool shadows, warm mids,
  +15% saturation on greens and browns); `--grade=0` (bench, player) turns it off. The scenes' old brightness /
  contrast / saturation values never applied (adjustments were off) and are gone. The ground's material per biome:
  **Biome materials** below. Next: materials inside one record (an IR change), image textures (triplanar).
- **Biome materials** (with any climate that has fields, `SDFCascade::has_ground_materials`): each biome is a ground material `biome:<name>` (`Climate::biome_material`:
  `PATTERN_GROUND`, the page shader's slope blend and contours in the biome's `"surface"` colours, sRGB `#rrggbb`
  `flat` / `steep` + `roughness` in the table, the procedural ground's without; its `debug_color` as the material's
  debug colour). Every chunk palette starts with the **base palette** (`SDFGpuTable::base_palette`: GROUND, then biome
  b at 1 + b; `ChunkPalette::build(base, ...)`), held by the palette pool's ground block (seeded at `rt_create`). The
  cascade's level 0 writes a material for **every** chunk: the fold's winner, else at a solid sample (air keeps 0: a
  vertex reads its cell's solid corners) `cascade_ground_material` = 1 + `Climate::pick` of the sample's fields
  (`Climate::pick_glsl`, `climate_glsl.cpp`, generated from the table op for op, ties by
  `hash_cell`; CPU `SDFCascade::ground_material`), so the chunk passes carry the fields at every level of every LOD
  (`CascadeGLSL::field_range`: point items keep the spec's FieldRange, the ground bits are the same). Both meshers read
  every chunk's materials when the base is longer than one (`ChunkPalette::glsl_base`); on the ground block a record's
  index past the base reads as ground. Unit-tested (box run == point query per sample); CascadeSpike G10
  (`--biomes`): GPU == CPU on every sample of every LOD (58.7 M, all 7 example biomes seen). Cost (128-chunk
  batches, the example's 3 fields, every ground octave modulated): the climate's passes 2.0x the bare cascade at LODs
  0-8, 3.3-3.9x at 12-24 (the weights run at every level; with multipliers on a few octaves it was 1.5-1.6x at 0-4
  and 20-24: 12 more channels carried everywhere); its fields alone 1.4-1.5x. Forest5cm, retuned vs the few-octave
  example (2 runs, GPU at 1.5 GHz): load +2%, SDF pass per batch +10-13%, steady render GPU +0%, fly p50 +11%
  (the climate vs none, earlier and throttled: steady render GPU +6%, SDF pass +19%). `--view=12` (B in the player, V cycles to it) paints each biome its
  debug colour. A field no edge, term or multiplier reads is carried anyway: leave it out of the table.
