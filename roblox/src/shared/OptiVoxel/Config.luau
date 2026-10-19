--!strict
-- Knobs for the Roblox client. Everything else (chunk size, voxel size, skirt) is in ChunkMath and
-- matches the Godot defaults.

return {
	-- Studs per Godot world unit (metre). Roblox's default avatar is ~5 studs tall, i.e. ~3 studs per
	-- metre. Smooth Terrain voxels are 4 studs whatever we do, so this picks the voxel size in metres:
	-- 8 studs per metre = 0.5 m voxels (the engine's), with characters scaled by PLAYER_SCALE.
	STUDS_PER_UNIT = 8,

	-- LOD scan (ChunkDiscoverer::setup_lod). Godot uses 15 levels at 4 px; each level doubles the
	-- view distance. 6 levels at 8 px is ~1.3k chunks and ~2.3k studs of view distance.
	LOD_LEVELS = 6,
	PIXEL_ERROR = 8,
	VIEWPORT_HEIGHT = 1080,
	FOV_DEGREES = 70,

	-- Parallel Luau actors meshing chunks.
	WORKERS = 4,

	-- Godot treats clockwise triangles as front faces, Roblox counter-clockwise: flip by default.
	-- If the terrain is only visible from below, set this to false.
	FLIP_WINDING = true,

	-- EditableMesh hard caps are 60k vertices / 20k triangles: bigger chunks are split.
	MAX_TRIANGLES_PER_MESH = 19000,

	-- Only LOD 0 chunks get collision (the player walks on them); the rest are visual only.
	COLLISION_MAX_LOD = 0,

	-- Destroy the EditableMesh once its MeshPart exists. Measured in Studio: the MeshPart stays linked
	-- to it and turns invisible, so this must stay false (the client allows only 8 live EditableMeshes,
	-- which is why this chunk-per-mesh client shows ~16 chunks; see README).
	DESTROY_EDITABLE_AFTER_PART = false,

	-- Smooth Terrain playground (terrain.project.json): characters are scaled to PLAYER_HEIGHT_M
	-- (measured body, head to feet, without accessories); walk speed and jump follow their scale,
	-- gravity, camera zoom and the brush follow STUDS_PER_UNIT / 3.
	PLAYER_HEIGHT_M = 1.8,
	-- Generated square, half side in studs.
	TERRAIN_HALF_STUDS = 1024,
	-- Far field around it, out to this half side: the height field only (no caves, no edits), written
	-- as a thin shell, sampled every 8 / 16 studs by distance (TerrainGen.writeFarTile). 0 turns it off.
	-- Roblox draws Smooth Terrain only ~3k studs around the camera (measured), so a wider shell costs
	-- memory for nothing: +103 MB at 3072, +330 MB at 8192 (the near square alone is 457 MB).
	TERRAIN_FAR_HALF_STUDS = 3072,

	-- Vertex colors by slope and height (false: plain part color, fewer API calls per chunk).
	VERTEX_COLORS = true,
}
