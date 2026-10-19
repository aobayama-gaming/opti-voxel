--!strict
-- Runs inside one cloned Actor (see OptiVoxelClient): meshes chunks in parallel, builds the
-- MeshParts in the serial phase, and owns them until told to unload. Only plain stats go back
-- through the BindableEvent (EditableMesh is not an Instance and must stay here).

local ReplicatedStorage = game:GetService("ReplicatedStorage")

local OptiVoxel = ReplicatedStorage:WaitForChild("OptiVoxel")
local ChunkBuilder = require(OptiVoxel:WaitForChild("ChunkBuilder"))
local Mesher = require(OptiVoxel:WaitForChild("Mesher"))

local actor = script:GetActor()
assert(actor, "OptiVoxel worker must run inside an Actor")
local done = actor.Parent:WaitForChild("OptiVoxelDone") :: BindableEvent
local terrain = workspace:WaitForChild("OptiVoxelTerrain")

local mesher = Mesher.new()
local builtByKey: { [string]: ChunkBuilder.Built } = {}

actor:BindToMessageParallel("mesh", function(key: string, cx: number, cy: number, cz: number)
	local t0 = os.clock()
	local ok, result = pcall(mesher.mesh, mesher, cx, cy, cz)
	local meshMs = (os.clock() - t0) * 1000

	task.synchronize()
	if not ok then
		done:Fire("done", actor.Name, key, { meshMs = meshMs, buildMs = 0, triangles = 0, parts = 0, failures = 1, lastError = tostring(result) })
		return
	end
	local t1 = os.clock()
	local built = ChunkBuilder.build(result, cx, cy, cz, terrain)
	builtByKey[key] = built
	done:Fire("done", actor.Name, key, {
		meshMs = meshMs,
		buildMs = (os.clock() - t1) * 1000,
		triangles = result.triangleCount,
		parts = #built.parts,
		failures = built.failures,
		lastError = built.lastError,
	})
end)

actor:BindToMessage("unload", function(key: string)
	local built = builtByKey[key]
	if built then
		ChunkBuilder.destroy(built)
		builtByKey[key] = nil
	end
end)

done:Fire("ready", actor.Name)
