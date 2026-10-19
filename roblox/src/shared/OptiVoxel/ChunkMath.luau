--!strict
--!native
-- Port of src/chunk_math.cpp. Chunk coordinates encode the LOD in their lowest set bit:
-- LOD 0 chunks have odd coordinates, a LOD n chunk spans 2^(n+1) coordinate units.

local ChunkMath = {}

-- ChunkParam defaults used by VoxelTerrain (src/voxel_terrain.hpp).
ChunkMath.CHUNK_SIZE = 32 -- cells per axis, skirt included
ChunkMath.VOXEL_SIZE = 0.5
ChunkMath.SKIRT_SIZE = 2
ChunkMath.USE_SKIRT = true

ChunkMath.CHUNK_VOXEL_DIM = ChunkMath.CHUNK_SIZE - 2 * ChunkMath.SKIRT_SIZE -- 28
ChunkMath.CHUNK_WORLD_DIM = ChunkMath.CHUNK_VOXEL_DIM * ChunkMath.VOXEL_SIZE -- 14

local WORLD_NUDGE = 0.12356700003147125 -- 0.123567f
local CHUNK_WORLD_DIM = ChunkMath.CHUNK_WORLD_DIM

local function toSigned(v: number): number
	return if v >= 0x80000000 then v - 0x100000000 else v
end

function ChunkMath.getLod(cx: number): number
	if cx == 0 then
		return 0
	end
	return bit32.countrz(cx)
end

function ChunkMath.lodSize(cx: number): number
	return bit32.lshift(1, ChunkMath.getLod(cx))
end

-- get_parent_from_child for one coordinate: clear the lowest set bit, set the next one.
function ChunkMath.parentCoord(c: number): number
	if c == 0 then
		return 0
	end
	local lsb = bit32.countrz(c)
	local cleared = bit32.band(c, bit32.bnot(bit32.lshift(1, lsb)))
	return toSigned(bit32.bor(cleared, bit32.lshift(1, lsb + 1)))
end

function ChunkMath.parentUntil(cx: number, cy: number, cz: number, targetLod: number): (number, number, number)
	while ChunkMath.getLod(cx) < targetLod do
		cx = ChunkMath.parentCoord(cx)
		cy = ChunkMath.parentCoord(cy)
		cz = ChunkMath.parentCoord(cz)
	end
	return cx, cy, cz
end

function ChunkMath.chunkToWorld(cx: number, cy: number, cz: number): (number, number, number)
	local h = CHUNK_WORLD_DIM / 2
	return cx * h + WORLD_NUDGE, cy * h + WORLD_NUDGE, cz * h + WORLD_NUDGE
end

function ChunkMath.worldChunkSize(cx: number): number
	return ChunkMath.lodSize(cx) * CHUNK_WORLD_DIM
end

-- Nearest LOD 0 chunk (odd coordinates) containing a world position.
function ChunkMath.worldToChunk(x: number, y: number, z: number): (number, number, number)
	return math.floor(x / CHUNK_WORLD_DIM) * 2 + 1,
		math.floor(y / CHUNK_WORLD_DIM) * 2 + 1,
		math.floor(z / CHUNK_WORLD_DIM) * 2 + 1
end

return ChunkMath
