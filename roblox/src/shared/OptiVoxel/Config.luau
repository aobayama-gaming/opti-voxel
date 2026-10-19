--!strict
-- Knobs for the Roblox client. Everything else (chunk size, voxel size, skirt) is in ChunkMath and
-- matches the Godot defaults.

return {
	-- Studs per Godot world unit. A Roblox character is ~5 studs tall, so ~3 studs per meter.
	STUDS_PER_UNIT = 3,

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

	-- Smooth Terrain playground (terrain.project.json): generated square, half side in studs.
	TERRAIN_HALF_STUDS = 1024,

	-- Vertex colors by slope and height (false: plain part color, fewer API calls per chunk).
	VERTEX_COLORS = true,
}
