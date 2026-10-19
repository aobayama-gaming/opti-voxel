--!strict
-- Spawn platform on the terrain surface at the origin. The terrain is client-side only and can
-- take a while to mesh (or fail to): this keeps the player from falling into the void meanwhile.

local ReplicatedStorage = game:GetService("ReplicatedStorage")

local OptiVoxel = ReplicatedStorage:WaitForChild("OptiVoxel")
local Config = require(OptiVoxel:WaitForChild("Config"))
local Sdf = require(OptiVoxel:WaitForChild("Sdf"))

local S = Config.STUDS_PER_UNIT
-- Highest surface point under the platform; never Sdf.height(0, 0) itself: the simplex noise is
-- degenerate exactly on that lattice point (257.8 there, ~300 everywhere around it).
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
spawn.Position = Vector3.new(0, surface + 0.5, 0)
spawn.Parent = workspace
