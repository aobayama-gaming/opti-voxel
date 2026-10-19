--!strict
-- Sdf.focusEdits must not change a Smooth Terrain occupancy: random edit sets near the surface, random
-- boxes, every voxel centre of the box sampled with all edits and with the focused subset; occupancies
-- (0.5 - d / voxel, clamped to [0, 1]) must be identical. Prints "ok", or the first mismatch and
-- exits non-zero.
local Sdf = require("../../src/shared/OptiVoxel/Sdf")

local VOXEL = 0.5 -- units: a 4-stud Smooth Terrain voxel at 8 studs per unit

local seed = 777
local function rnd(): number
	seed = (seed * 1103515245 + 12345) % 2147483648
	return seed / 2147483648
end

local function occupancy(d: number): number
	return math.clamp(0.5 - d / VOXEL, 0, 1)
end

local checked, total = 0, 0
for trial = 1, 40 do
	Sdf.clearEdits()
	local cx, cz = (rnd() - 0.5) * 2000, (rnd() - 0.5) * 2000
	local n = 20 + math.floor(rnd() * 200)
	for _ = 1, n do
		local x, z = cx + (rnd() - 0.5) * 120, cz + (rnd() - 0.5) * 120
		local y = Sdf.height(x, z) + (rnd() - 0.6) * 20
		local r = 0.5 + rnd() * 12
		if rnd() < 0.6 then
			Sdf.addSubtractSphere(x, y, z, r)
		else
			Sdf.addUnionSphere(x, y, z, r)
		end
	end
	local x0, z0 = cx + (rnd() - 0.5) * 100, cz + (rnd() - 0.5) * 100
	local sx, sz = 2 + rnd() * 16, 2 + rnd() * 16
	local h = Sdf.height(x0, z0)
	local y0, y1 = h - 15, h + 10
	local reference = {}
	for x = x0 + VOXEL / 2, x0 + sx, VOXEL do
		for z = z0 + VOXEL / 2, z0 + sz, VOXEL do
			for y = y0 + VOXEL / 2, y1, VOXEL do
				table.insert(reference, occupancy(Sdf.evaluate(x, y, z)))
			end
		end
	end
	Sdf.focusEdits(x0, y0, z0, x0 + sx, y1, z0 + sz, VOXEL)
	local i = 0
	for x = x0 + VOXEL / 2, x0 + sx, VOXEL do
		for z = z0 + VOXEL / 2, z0 + sz, VOXEL do
			for y = y0 + VOXEL / 2, y1, VOXEL do
				i += 1
				local o = occupancy(Sdf.evaluate(x, y, z))
				if o ~= reference[i] then
					print(`trial {trial}: occupancy {o} focused vs {reference[i]} at {x}, {y}, {z}`)
					os.exit(1)
				end
			end
		end
	end
	Sdf.unfocusEdits()
	checked += i
	total += n
end
Sdf.clearEdits()
print(`ok: {checked} voxels over 40 edit sets ({total} edits)`)
