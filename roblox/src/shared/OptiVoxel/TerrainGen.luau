--!strict
--!native
-- Roblox only: samples the SDF (with its edits) into Smooth Terrain. Roblox voxels are 4 studs and
-- store an occupancy in [0, 1]; occupancy = 0.5 - sdf / voxel puts the 0.5 isosurface on sdf = 0.
-- `shift` (studs, x and z): the box is physical, the SDF is sampled at physical - shift, i.e. a copy of
-- the world written `shift` away (the treadmill's copies).

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
function TerrainGen.yRange(x0: number, z0: number, x1: number, z1: number, depth: number, shift: Vector2?): (number, number)
	local sx, sz = if shift then shift.X else 0, if shift then shift.Y else 0
	local lo, hi = math.huge, -math.huge
	for x = x0, x1, VOX * 2 do
		for z = z0, z1, VOX * 2 do
			local h = Sdf.height((x - sx) / S, (z - sz) / S)
			lo, hi = math.min(lo, h), math.max(hi, h)
		end
	end
	return align((lo - depth) * S, false), align((hi + 2) * S, true)
end

-- Samples the box [x0, x1) x [y0, y1) x [z0, z1) (studs, any values: aligned outward to the grid)
-- and writes it. Returns ms spent sampling, ms spent in WriteVoxels, voxel count.
function TerrainGen.writeBox(terrain: Terrain, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, shift: Vector2?): (number, number, number)
	local sx, sz = if shift then shift.X else 0, if shift then shift.Y else 0
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
		local wx = (x0 + (i - 0.5) * VOX - sx) / S
		for k = 1, nz do
			local wz = (z0 + (k - 0.5) * VOX - sz) / S
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

local SHELL = 2 * VOX -- studs of solid under the lowest neighbouring column top

local function surfaceMaterial(h: number, slope: number): Enum.Material
	return if h > SNOW_ABOVE * S then Snow elseif slope > 0.9 then Rock else Grass
end

-- Far field: the height field only (no caves, no edits), sampled every `step` studs (a multiple of
-- VOX) and interpolated bilinearly per voxel column, written as a shell SHELL thick under the lowest
-- neighbouring column top (hollow underneath: only seen from above). Box [x0, x1) x [z0, z1), aligned
-- on `step`. Returns ms spent sampling, ms spent in WriteVoxels, voxel count.
function TerrainGen.writeFarTile(terrain: Terrain, x0: number, z0: number, x1: number, z1: number, step: number, shift: Vector2?): (number, number, number)
	local sx, sz = if shift then shift.X else 0, if shift then shift.Y else 0
	local t0 = os.clock()
	-- Height samples (studs) from x0 - step to x1 + step: the border columns need their neighbours.
	local gn, gm = (x1 - x0) // step + 3, (z1 - z0) // step + 3
	local grid = table.create(gn * gm)
	for i = 0, gn - 1 do
		local x = (x0 + (i - 1) * step - sx) / S
		for k = 0, gm - 1 do
			grid[i * gm + k + 1] = Sdf.height(x, (z0 + (k - 1) * step - sz) / S) * S
		end
	end
	-- Column tops (studs), with a one-column border.
	local nx, nz = (x1 - x0) // VOX, (z1 - z0) // VOX
	local cm = nz + 2
	local tops = table.create((nx + 2) * cm)
	local lo, hi = math.huge, -math.huge
	for i = 0, nx + 1 do
		local fx = (i - 0.5) * VOX / step + 1
		local gi = math.floor(fx)
		local tx = fx - gi
		for k = 0, nz + 1 do
			local fz = (k - 0.5) * VOX / step + 1
			local gk = math.floor(fz)
			local tz = fz - gk
			local a = grid[gi * gm + gk + 1]
			local b = grid[(gi + 1) * gm + gk + 1]
			local c = grid[gi * gm + gk + 2]
			local d = grid[(gi + 1) * gm + gk + 2]
			local h = a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz
			tops[i * cm + k + 1] = h
			lo, hi = math.min(lo, h), math.max(hi, h)
		end
	end
	local y0, y1 = align(lo - SHELL, false), align(hi + VOX, true)
	local ny = (y1 - y0) // VOX
	local mats = table.create(nx)
	local occs = table.create(nx)
	local voxels = 0
	for i = 1, nx do
		local mi, oi = table.create(ny), table.create(ny)
		mats[i], occs[i] = mi, oi
		for j = 1, ny do
			mi[j], oi[j] = table.create(nz, Air), table.create(nz, 0)
		end
		for k = 1, nz do
			local c = i * cm + k + 1
			local h = tops[c]
			local hx, hz = tops[c + cm], tops[c + 1]
			local slope = math.max(math.abs(hx - h), math.abs(hz - h)) / UNITS_PER_VOX / S
			local bottom = math.min(h, hx, hz, tops[c - cm], tops[c - 1]) - SHELL
			local surface = surfaceMaterial(h, slope)
			local below = if slope > 0.9 then Rock else Ground
			for j = math.max(1, (bottom - y0) // VOX + 1), math.min(ny, (h - y0) // VOX + 2) do
				local wy = y0 + (j - 0.5) * VOX
				local occ = 0.5 - (wy - h) / VOX
				if occ > 0 then
					oi[j][k] = if occ > 1 then 1 else occ
					mi[j][k] = if h - wy < 1.5 * S then surface else below
					voxels += 1
				end
			end
		end
	end
	local sampleMs = (os.clock() - t0) * 1000
	local t1 = os.clock()
	terrain:WriteVoxels(Region3.new(Vector3.new(x0, y0, z0), Vector3.new(x1, y1, z1)), VOX, mats, occs)
	return sampleMs, (os.clock() - t1) * 1000, voxels
end

return TerrainGen
