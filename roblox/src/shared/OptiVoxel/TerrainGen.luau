--!strict
--!native
-- Roblox only: samples the SDF (with its edits) into Smooth Terrain. Roblox voxels are 4 studs and
-- store an occupancy in [0, 1]; occupancy = 0.5 - sdf / voxel puts the 0.5 isosurface on sdf = 0.

local Config = require(script.Parent.Config)
local Sdf = require(script.Parent.Sdf)

local TerrainGen = {}

TerrainGen.VOXEL = 4 -- studs, fixed by the Terrain API
local VOX = TerrainGen.VOXEL
local S = Config.STUDS_PER_UNIT
local UNITS_PER_VOX = VOX / S
local SNOW_ABOVE = 335 -- units, same threshold as ChunkBuilder's vertex colors

local Air, Grass, Ground, Rock, Snow, Slate =
	Enum.Material.Air, Enum.Material.Grass, Enum.Material.Ground, Enum.Material.Rock, Enum.Material.Snow, Enum.Material.Slate

local function align(v: number, up: boolean): number
	return if up then math.ceil(v / VOX) * VOX else math.floor(v / VOX) * VOX
end

-- Studs range [y0, y1) worth sampling over a column box: from `depth` units under the lowest surface
-- point to a bit above the highest.
function TerrainGen.yRange(x0: number, z0: number, x1: number, z1: number, depth: number): (number, number)
	local lo, hi = math.huge, -math.huge
	for x = x0, x1, VOX * 2 do
		for z = z0, z1, VOX * 2 do
			local h = Sdf.height(x / S, z / S)
			lo, hi = math.min(lo, h), math.max(hi, h)
		end
	end
	return align((lo - depth) * S, false), align((hi + 2) * S, true)
end

-- Samples the box [x0, x1) x [y0, y1) x [z0, z1) (studs, any values: aligned outward to the grid)
-- and writes it. Returns ms spent sampling, ms spent in WriteVoxels, voxel count.
function TerrainGen.writeBox(terrain: Terrain, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): (number, number, number)
	x0, y0, z0 = align(x0, false), align(y0, false), align(z0, false)
	x1, y1, z1 = align(x1, true), align(y1, true), align(z1, true)
	local nx, ny, nz = (x1 - x0) // VOX, (y1 - y0) // VOX, (z1 - z0) // VOX
	if nx <= 0 or ny <= 0 or nz <= 0 then
		return 0, 0, 0
	end
	local t0 = os.clock()
	local mats = table.create(nx)
	local occs = table.create(nx)
	for i = 1, nx do
		local mi, oi = table.create(ny), table.create(ny)
		mats[i], occs[i] = mi, oi
		for j = 1, ny do
			mi[j], oi[j] = table.create(nz, Air), table.create(nz, 0)
		end
		local wx = (x0 + (i - 0.5) * VOX) / S
		for k = 1, nz do
			local wz = (z0 + (k - 0.5) * VOX) / S
			local h = Sdf.height(wx, wz)
			-- Slope of the height field (rise per unit): steep faces are bare rock.
			local slope = math.max(math.abs(Sdf.height(wx + 1, wz) - h), math.abs(Sdf.height(wx, wz + 1) - h))
			for j = 1, ny do
				local wy = (y0 + (j - 0.5) * VOX) / S
				local d = Sdf.evaluateWithHeight(wx, wy, wz, h)
				local occ = 0.5 - d / UNITS_PER_VOX
				if occ > 0 then
					oi[j][k] = if occ > 1 then 1 else occ
					local below = h - wy -- depth under the original surface (edits ignore this)
					mi[j][k] = if below < -1 then Slate -- matter added above the original ground
						elseif below < 1.5 then (if wy > SNOW_ABOVE then Snow elseif slope > 0.9 then Rock else Grass)
						elseif below < 6 then (if slope > 0.9 then Rock else Ground)
						else Rock
				end
			end
		end
	end
	local sampleMs = (os.clock() - t0) * 1000
	local t1 = os.clock()
	terrain:WriteVoxels(Region3.new(Vector3.new(x0, y0, z0), Vector3.new(x1, y1, z1)), VOX, mats, occs)
	return sampleMs, (os.clock() - t1) * 1000, nx * ny * nz
end

return TerrainGen
