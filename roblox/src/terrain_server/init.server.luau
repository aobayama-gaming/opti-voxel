--!strict
--!native
-- Smooth Terrain playground, server side (terrain.project.json):
--   - generates the SDF terrain into Roblox Terrain, slab by slab, nearest the spawn first;
--   - applies dig / fill requests from clients as SDF edits, then resamples the edited box.
-- Terrain is edited here, on the server, so it replicates to every player.
-- Progress and timings go to workspace attributes (read by the client overlay).

local ReplicatedStorage = game:GetService("ReplicatedStorage")

local OptiVoxel = ReplicatedStorage:WaitForChild("OptiVoxel")
local Config = require(OptiVoxel:WaitForChild("Config"))
local Sdf = require(OptiVoxel:WaitForChild("Sdf"))
local TerrainGen = require(OptiVoxel:WaitForChild("TerrainGen"))

local S = Config.STUDS_PER_UNIT
local HALF = Config.TERRAIN_HALF_STUDS
local SLAB = 64 -- studs per side of one generation step
local DEPTH = 40 -- units of ground (and caves) under the lowest surface point of each slab
local terrain = workspace.Terrain

-- Spawn platform until the ground under it exists. Highest surface point under its footprint;
-- never Sdf.height(0, 0) itself: the simplex noise is degenerate exactly on that lattice point
-- (257.8 there, ~300 everywhere around it).
local surface = -math.huge
for dx = -8, 8, 2 do
	for dz = -8, 8, 2 do
		surface = math.max(surface, Sdf.height((dx + 0.37) / S, (dz + 0.37) / S) * S)
	end
end
local spawn = Instance.new("SpawnLocation")
spawn.Name = "OptiVoxelSpawn"
spawn.Anchored = true
spawn.Neutral = true
spawn.Duration = 0
spawn.Size = Vector3.new(16, 1, 16)
spawn.Transparency = 0.5
spawn.Position = Vector3.new(0, surface + 2, 0)
spawn.Parent = workspace

local edit = Instance.new("RemoteEvent")
edit.Name = "OptiVoxelEdit"
edit.Parent = ReplicatedStorage

-- Generation, nearest slab first.
local slabs: { Vector2 } = {}
for x = -HALF, HALF - 1, SLAB do
	for z = -HALF, HALF - 1, SLAB do
		table.insert(slabs, Vector2.new(x, z))
	end
end
table.sort(slabs, function(a, b)
	return (a + Vector2.new(SLAB / 2, SLAB / 2)).Magnitude < (b + Vector2.new(SLAB / 2, SLAB / 2)).Magnitude
end)
workspace:SetAttribute("GenTotal", #slabs)
workspace:SetAttribute("GenDone", 0)
workspace:SetAttribute("TerrainHalfStuds", HALF)

task.spawn(function()
	local sampleMs, writeMs, voxels = 0, 0, 0
	local t0 = os.clock()
	for i, s in slabs do
		local y0, y1 = TerrainGen.yRange(s.X, s.Y, s.X + SLAB, s.Y + SLAB, DEPTH)
		local a, b, n = TerrainGen.writeBox(terrain, s.X, y0, s.Y, s.X + SLAB, y1, s.Y + SLAB)
		sampleMs += a
		writeMs += b
		voxels += n
		workspace:SetAttribute("GenDone", i)
		task.wait()
	end
	-- Terrain collision is built asynchronously (on each client too): dropping the platform as soon
	-- as the ground under it was written let players fall through into the caves. Keep the spawn
	-- point, drop its collision once everything is written and has had time to settle.
	task.delay(3, function()
		spawn.CanCollide = false
		spawn.Transparency = 1
	end)
	workspace:SetAttribute("GenSeconds", os.clock() - t0)
	workspace:SetAttribute("GenSampleMs", sampleMs)
	workspace:SetAttribute("GenWriteMs", writeMs)
	workspace:SetAttribute("GenVoxels", voxels)
	print(
		`[OptiVoxel] terrain {HALF * 2}x{HALF * 2} studs: {voxels} voxels, sample {math.floor(sampleMs)} ms,`
			.. ` WriteVoxels {math.floor(writeMs)} ms, wall {string.format("%.1f", os.clock() - t0)} s`
	)
end)

-- Edits: SDF spheres, in studs on the wire, units in the SDF.
local lastEdit: { [Player]: number } = {}
edit.OnServerEvent:Connect(function(player: Player, kind: unknown, position: unknown, radius: unknown)
	if typeof(position) ~= "Vector3" or typeof(radius) ~= "number" or (kind ~= "dig" and kind ~= "fill") then
		return
	end
	local now = os.clock()
	if now - (lastEdit[player] or 0) < 0.05 then
		return
	end
	lastEdit[player] = now
	local r = math.clamp(radius, 2, 40)
	local p = position / S
	if kind == "dig" then
		Sdf.addSubtractSphere(p.X, p.Y, p.Z, r / S)
	else
		Sdf.addUnionSphere(p.X, p.Y, p.Z, r / S)
	end
	local m = r + TerrainGen.VOXEL * 2
	local a, b, n = TerrainGen.writeBox(terrain, position.X - m, position.Y - m, position.Z - m, position.X + m, position.Y + m, position.Z + m)
	workspace:SetAttribute("LastEditSampleMs", a)
	workspace:SetAttribute("LastEditWriteMs", b)
	workspace:SetAttribute("LastEditVoxels", n)
	workspace:SetAttribute("EditCount", Sdf.editCount())
end)

game:GetService("Players").PlayerRemoving:Connect(function(player)
	lastEdit[player] = nil
end)
