--!strict
--!native
-- Port of ChunkDiscoverer (src/chunk_discoverer.cpp): which chunks to load, at which LOD, for a
-- camera position. Distances are in chunk-coordinate units (one unit = CHUNK_WORLD_DIM / 2).

local ChunkMath = if script then require(script.Parent.ChunkMath) else require("./ChunkMath")
local Sdf = if script then require(script.Parent.Sdf) else require("./Sdf")

local evaluate = Sdf.evaluate
local getLod = ChunkMath.getLod

export type Chunk = { x: number, y: number, z: number, lod: number, key: string }

local Discoverer = {}
Discoverer.__index = Discoverer

export type Discoverer = typeof(setmetatable({} :: { lodDistances: { number }, visited: number }, Discoverer))

-- ChunkDiscoverer::setup_lod: a LOD switches when its geometric error (half a voxel) would exceed
-- target_pixel_error on screen.
function Discoverer.new(lodLevels: number, pixelError: number, viewportHeight: number, fovRad: number): Discoverer
	local focal = viewportHeight / (2 * math.tan(fovRad * 0.5))
	local distances = table.create(lodLevels + 1, 0)
	local previous = 0
	for i = 0, lodLevels do
		local geometricError = ChunkMath.VOXEL_SIZE * bit32.lshift(1, i) * 0.5
		local d = (geometricError * focal) / pixelError / (ChunkMath.CHUNK_WORLD_DIM * 0.5)
		if i == 0 and d < 1 then
			d = 1
		end
		if d <= previous then
			d = previous + 1
		end
		distances[i + 1] = d
		previous = d
	end
	return setmetatable({ lodDistances = distances, visited = 0 }, Discoverer)
end

-- _chunk_is_outside_sdf: no corner within 2 chunk sizes of the surface (offsets as in the C++).
local function outsideSdf(cx: number, cy: number, cz: number): boolean
	local size = ChunkMath.worldChunkSize(cx)
	local wx, wy, wz = ChunkMath.chunkToWorld(cx, cy, cz)
	local limit = size * 2
	for corner = 0, 7 do
		local ox = if bit32.band(corner, 1) ~= 0 then 0 else size
		local oy = if bit32.band(corner, 2) ~= 0 then 0 else size
		local oz = if bit32.band(corner, 4) ~= 0 then 0 else size
		if math.abs(evaluate(wx + ox, wy + oy, wz + oz)) <= limit then
			return false
		end
	end
	return true
end

function Discoverer.scan(self: Discoverer, camX: number, camY: number, camZ: number): { [string]: Chunk }
	local distances = self.lodDistances
	local out: { [string]: Chunk } = {}
	self.visited = 0

	local function emit(x: number, y: number, z: number)
		local key = `{x},{y},{z}`
		out[key] = { x = x, y = y, z = z, lod = getLod(x), key = key }
	end

	local function recurse(x: number, y: number, z: number)
		self.visited += 1
		local lod = getLod(x)
		if outsideSdf(x, y, z) then
			return
		end
		if lod == 0 then
			emit(x, y, z)
			return
		end
		local dx, dy, dz = camX - x, camY - y, camZ - z
		local dist = math.sqrt(dx * dx + dy * dy + dz * dz)
		local radius = math.sqrt(3) * bit32.lshift(1, lod)
		if dist - radius < distances[lod] then -- lod_distances[current_lod - 1]
			local child = bit32.lshift(1, lod - 1)
			for i = 0, 7 do
				recurse(
					x + (if bit32.band(i, 1) ~= 0 then child else -child),
					y + (if bit32.band(i, 2) ~= 0 then child else -child),
					z + (if bit32.band(i, 4) ~= 0 then child else -child)
				)
			end
		else
			emit(x, y, z)
		end
	end

	local lodCount = #distances - 1
	local maxLod = lodCount - 1
	local maxLodRadius = bit32.lshift(2, maxLod)
	local maxLodNumber = math.min(math.ceil(distances[lodCount] / maxLodRadius) + 2, maxLod)
	local ax, ay, az = ChunkMath.parentUntil(camX, camY, camZ, maxLod)
	local span = maxLodNumber * maxLodRadius
	for x = ax - span, ax + span, maxLodRadius do
		for y = ay - span, ay + span, maxLodRadius do
			for z = az - span, az + span, maxLodRadius do
				recurse(x, y, z)
			end
		end
	end
	return out
end

return Discoverer
