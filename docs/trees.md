# Trees, moment blobs and biome features

Library-built instances placed by the stream: trees, the far-field moment blobs, lava lakes and floating islands.

- **Trees (step 5, `src/tree_archetype.*`):** a tree is a contiguous run of plain records, all smooth union
  (`SDFTable::get_trees()`: `TreeSpan` first id + count), no change to the IR, the fold or the meshers.
  Library types 3 `branch` (a tapered capsule {height, radius, slope}, slope = (r1 - r0) / height from the
  CPU, L = 1 + slope^2 / 2), 4 `crown` (`make_rock_type(crown_k(voxel), CROWN_SEED)`, bumps of 0.8-3.2 m whose
  amplitude adds `CROWN_BUMP` = 1.5 to L: lumpy foliage up close, smooth once the band rule drops the octaves) and
  5 `cone` (`make_cone_type`, the same bumps on a cone {radius, height, cos, sin, amp}, its rim a smooth max: a
  conifer's tier). Three species: a broadleaf, a slender one and a conifer (`cone_tiers` = 4 stacked cones,
  no limbs, ~12 records). `grow_tree` is one recursion for topology and bends: a stem's axis split
  at midpoints moved by a hash of their path (joints on whole voxels: a segment's end is the next one's
  anchor), children at hashed forks with pipe-model radii (r0^2 >= tip^2 + sum of the children's), stems
  grown thickest first so the per-tree budget cuts the thinnest; then a hull of 1-3 crown spheres on the
  upper trunk (the last on its tip) and a clump at each leaf stem's tip, smooth-unioned at half their radius
  into one mass; segments and clumps the hull always covers are skipped.
  Nothing is LOD-specific: the band rule is the LOD rule (thin parts erode away first, a tapered branch
  from its tip, the hull last; nothing appears with the LOD). **Hull-only growth:** `CellScatter::generate` takes the
  chunk LOD (-1: all) and returns the records listed there as a subset of the full growth; from
  `tree_hull_only_lod(species)` (the larger `max_lod` of the species' largest trunk segment and clump, `tree_bounds`
  padded by `TREE_BOUND_PAD`; `max_lod` grows with a part's size, unit-tested) only the hull or tiers are listed, so the
  forest grows the trunk's polyline and the hull alone (`grow_tree(..., hull_only)`), bit-identical to the full growth's
  listed records (unit test: 1,000 trees, every LOD from there; tight at 5 cm: 8 / 7 / 5 for the three species, a LOD
  late at 0.5 m). `set_forest_scatter` is the forest's generate, classes and profile key, for the table and the tests; a
  species whose `tree_bounds` don't fit the table never grows hull-only (a site's refusal never depends on the LOD). Sites are 3D cells (planets come; `src/cell_sites.*`, any
  source, salted: `TREE_SITE_SALT` for the forest, the species is the site's variant): `cell_site` reads the coarse field
  (`SDFBase::ground_field(voxel, lod)`, `evaluate_cull`'s field with its gradient; at `site_lod` a lattice sample extended
  along its gradient) at the cell's centre, projects the cell's hashed voxel on that plane, keeps it if it lands in the
  cell's middle part (`SITE_INNER` 0.15 off each side across the normal: neighbours stay 0.3 cells apart with no
  neighbour pass), then snaps it onto the LOD-0 ground (`SNAP_STEPS` Newton steps on whole voxels, one owner per surface
  point); `surface_sites` tries only the cells an absolute octree walk keeps, one coarse reading per node centre, while
  its plane crosses the node (`SITE_SLACK` cells + `SITE_BEND` nodes): ~9 evaluations per site, 1 per empty box;
  unit-tested equal to the brute force on hills, a 60 degree slope and the 5 cm cascade. A tree grows along
  `SDFBase::up_at(base)` (+y on the flat world, the one source of "up" for placement; `UP_Y` keeps the records from
  before, pinned by a hash). The scene's centre is the ground under (SCENE_X, SCENE_Z) of the base's tangent plane
  (`ground_at`), the lake keep-out the distance to a lake's axis (`LavaPool::near`, the sources' `accept`); the zone is the `--tree_area` square over 4 half sides of height up and down. Hashing 3D cells moved
  every tree, the coarse-plane projection (feature/cell-sites) again: forest5cm is another forest than before. `--trees=N --tree_area=M` (default 0, 60 m) in bench and player; meant for 5 cm voxels
  (at 0.5 m only the trunk and hull remain). `get_debug_stats()["table"]["trees"]`: count, records per site.
  Past its parts' drops the forest is drawn by its moment blobs (below). **Band timing (moments,
  `src/instance_moments.*`):** every record has moments (`record_moments`: volume, centroid, 3x3 central second
  moment; analytic per library type via `TableDef::moments`, else its program on a 24^3 grid), an instance their sum
  per material (`span_moments`: a tree's bark and leaves). Its half thickness is the equal-covariance ellipsoid's
  smallest semi-axis. `set_band` drops it at the later of the first LOD whose cells no longer fit `resolve` times in
  that half thickness (`VoxelTerrain.archetype_resolve` / `--instance_resolve=F`, default 1) and the first LOD erosion
  at `rate` from LOD 0 can make it vanish by; its onset is the latest LOD from which that erosion still reaches its
  depth at the drop: whole while resolved, shrinking only in the LODs before its drop, `erode` cells per LOD at most
  (unit-tested). At 5 cm (30 trees, by volume): drawn whole 98 / 93 / 61% at LODs 1 / 3 / 5 (0% when every part
  eroded from LOD 1), drawn at all 97% at LOD 5 and 48% at LOD 6 (65% and 0%). The rate is
  `VoxelTerrain.archetype_erode` / `--instance_erode=F` (cells per LOD, default `ERODE_CELLS` = 1, bench and player;
  the record carries it as `erode`, the skirt follows it): a higher one moves onsets later.
  **Moment blobs (the far field, `src/moment_layers.*`):** what the parts a LOD no longer draws still weigh. For each
  LOD L, every part hands the share of its mass the LOD lost (its equal-covariance ellipsoid shrunk by its erosion,
  `MomentItem::lost_share`; all of it from its drop on: parts drawn + blobs = its mass at every LOD) to cells of
  2^(L + 3) voxels (8 cells of L) on an absolute grid, per material, with tent weights over the 2 x 2 x 2 cells around
  each centroid (a uniform cover reads a near-constant occupancy, a thin canopy stays thin); a material joins once a
  quarter of its mass is lost (lone twigs would come back fattened). Taking only the parts gone whole left the forest
  thinned out at LODs 4-6 (canopy columns 34 -> 26 of 48 at LOD 5; the unit test now holds 9/10 of LOD 2's). Each cell
  is a blob, `a max(1 - q, 0)^2` in its principal frame (the record's rotation), its spread the cell's covariance plus
  the LOD's box filter, its peak the mass-kept one or, higher, the cell's **cover**: each part's area tensor (its
  silhouette along n is n^T P n, pi r^2 for a sphere, `prepare_moment_item`), summed and read along the cell's thinnest
  axis, 3 A / (pi s_1 s_2), at most 1 (a canopy reads its cover, ~0.44, at every LOD 3-9 where keeping the mass faded
  0.36 -> 0.08; edges and far LODs included, where cells look round and the old sheet gain vanished the forest; a lone
  clump fades once the filter outgrows it), at most 2^(L - the LOD it lost half its mass at) times the mass-kept peak
  (`MomentJoin::cover_from`: seams step no more, the rock scatter's 3 | 4 the worst at 0.90 of its skirt); blobs
  fainter than `MOMENT_MATERIAL_ISO` with their neighbours are pruned. They are table records (`moment_blob` type,
  `Combine::ACCUM`, listed at their LOD only: `TableRecord::min_lod`), built by `SDFTable::add_moment_blobs` after the
  scene; the fold sums them into an occupancy phi and its epilogue (`moment_surface` / GLSL `arch_moments`, bit for
  bit, FrameSpike's table probe) draws `min(r - h phi, (MOMENT_ISO - phi) h / max(|grad phi|, MOMENT_ISO / 64))`: the
  ground lifted under thin mass, a surface where phi reaches `MOMENT_ISO` (0.25: an opaque canopy), the strongest
  blob's material from half of it. Each layer is listed at one LOD, so a L | L+1 seam steps: the samples a blob touches
  take `SDFTable::MOMENT_SKIRT_CELLS` (2.5) more skirt (`cascade_sample_sdf`'s `skirt_extra`, `SDFCascade::with_skirt` /
  `skirt_depth_at`; 0.93 of it on the rock scatter, the worst), and the locality margins take it whole when the scene
  places instances; a global skirt that deep cost +20% triangles in the forest (taller walls on every chunk). The
  discoverer's no-cull rule skips a blob below its LOD (`near_layer`: `min_lod`): its box would keep every finer node
  under a coarse canopy uncullable (+18% loaded chunks).
  A dense 5 cm forest keeps a canopy over most columns at LODs 6-8 (unit test); from 1,100 m the forest5cm square
  is a canopy instead of nothing (`bench.sh --preset=forest5cm --cam_pos=-50,320,750 --above_ground=1100 --pitch=-55
  --yaw=0 [--moments=0]`). The hand-off over every altitude: `tools/forest_lod_sheet.sh` (bench `--scenario=rise
  --altitudes=... --rise_shots=DIR`, 20 m to 13 km over the forest, shaded | LOD view, `sheet.png` + lists per step)
  and the unit test of the canopy columns at LODs 2-9. `VoxelTerrain.archetype_moments` / `--moments=0` (bench, player) leaves them out;
  `get_debug_stats()["table"]["moments"]`: items, blobs per LOD, build ms (`["trees"]["build_ms"]`: the forest). The
  build runs on the worker pool (`parallel_for.hpp`: each layer's members side by side, then each cell's sums and pruning;
  the forest's ground traces and trees per site, admitted in site order), the same bits: forest5cm_x10 (40,000 trees,
  2.2 M blobs) configures in 15 s, 93 s serial (blobs 89 -> 11 s, the pruning's map lookups were 2/3 of it; forest
  18 -> 2 s); the main thread still waits for it (async configure: next). **Edits** keep the blobs exact:
  `Archetype::MomentLayers` holds each cell's members (by record id), their partial sums per block of 256 ids and the
  pruning verdict; an edit moves its record between the cells its old and new tents touch, re-sums one block per cell
  and re-judges the blobs whose box meets a changed one (bounds are summed in cell order), so the table holds the same
  bits as a build of the edited records with the same joins (unit test: 6,829 blobs over 40 edits). The changed blobs go
  through the edit's snapshot like any record (`SDFTable::update_moment_blobs`); `edit_affects_chunk` re-meshes each at
  its LOD only (`EditRegion`). ~8 ms per edit on a 120-tree table in the debug build (0.17 without the blobs); the
  forest5cm edit scenario re-meshes 6% more chunks; validate / audit / discovery_audit stay ok. The joins and the first
  layer are frozen at the build: a material the build didn't have, or a part gone below its first layer, joins at the
  next `configure`. A blob that vanishes keeps its record id for its key (removed, reused when it comes back: edits
  don't eat the instance capacity). Coarse cells hold nearly every part: their member lists cost memory (~4 B per part
  per cell its tent touches) and an O(parts) insert / erase per edit (TODO: per-block member lists would bound it).
  **Biome features (`src/biome_features.*`):** lava lakes and floating islands, plain records like the trees (no
  change to the IR, the fold or the meshers). Placement is by biome cells: each kind tiles space with cubic cells of
  640 m rounded to 2^k voxels, and a cell keeps one site with the kind's chance, from `hash_cell` of the int64 cell: a
  point of its middle half projected onto the base surface (`BaseSurface::on_base`), kept when the projection stays
  in the cell, so every point of the base has one cell (on the flat base the datum's layer: one per column, as the
  plane's cells were; a planet's sites cover it all round, unit-tested). A site depends only on its cell and the base:
  the same from any box or origin, unit-tested near 0, 2^32 and +-2^60. Hashing 3D cells moved every lake and island.
  On a sphere the density per area depends on the surface's slant to the cells (where it runs diagonal, more cells
  each keep a share: ~1.2x at 45 degrees), so features thin and thicken in bands of the grid; the 640 m cells keep that
  invisible at Earth's scale. A lake's other spots stay in its cell, and its keep-out is bounded along its axis.
  A feature is built in its site's tangent frame (`up_rotation` of the base's up: heights along it, the records
  turned onto it; the identity on the flat base), unit-tested 90 degrees round a planet where y is level. A lake is two records: a `basin` (a cylinder plus the rock's bumps, 16 m lattice)
  smooth-subtracted from the ground, then a hard-unioned `cylinder` slab whose top face is the flat lava, 4% of the
  radius below the lowest of 72 ground samples on the slab's rim. `Biome::place` tries `LAKE_ATTEMPTS` spots of the
  cell and gives up where the rim's ground spans over half the radius (`lakes_too_steep`). An island is one `island`
  record (a spherical cap on a hanging cone, smooth-unioned, plus bumps), its box bottom 25-70 m above the highest
  ground under it. Both follow the band rule like any record; the slab erodes away a few LODs before the basin (it
  can't be thicker without poking out of the shore), so a far lake is a shallow pit (0.15 r) whose scorched basin
  material shows. `--biomes=KM` (bench and player, `VoxelTerrain.biome_radius` in metres) places the sites within KM km
  of the base's point under (SCENE_X, SCENE_Z) once, at configure, on the worker pool, before the scatter and the forest, which keep 1.5 radii
  out of the lakes (`SDFTable::in_lake`); streaming cells needs the index rebuilt off the main thread.
  `get_debug_stats()["table"]["biomes"]`: counts, `build_ms`, and up to `SITES_LISTED_MAX` sites (`kind`, x / y / z
  in metres of the lava's centre or the island's rim plane's, radius). `--preset=biomes` looks at a lava lake with two islands behind it.
