--!strict
-- Runs inside one cloned Actor (see OptiVoxelClient): meshes chunks in parallel, builds the
-- MeshParts in the serial phase, and owns them until told to unload. Only plain stats go back
-- through the BindableEvent (EditableMesh is not an Instance and must stay here).
-- Every "mesh" message gets exactly one "done" back, even on error, or the client would keep the
-- worker busy forever.

local ReplicatedStorage = game:GetService("ReplicatedStorage")

local actor = script:GetActor()
assert(actor, "OptiVoxel worker must run inside an Actor")
local done = actor.Parent:WaitForChild("OptiVoxelDone") :: BindableEvent
local terrain = workspace:WaitForChild("OptiVoxelTerrain")

local OptiVoxel = ReplicatedStorage:WaitForChild("OptiVoxel")
local ChunkBuilder = require(OptiVoxel:WaitForChild("ChunkBuilder"))
local Mesher = require(OptiVoxel:WaitForChild("Mesher"))

local mesher = Mesher.new()
local builtByKey: { [string]: ChunkBuilder.Built } = {}

local function report(key: string, info: { [string]: any })
	info.aliveMeshes = ChunkBuilder.aliveMeshes
	info.aliveTriangles = ChunkBuilder.aliveTriangles
	done:Fire("done", actor.Name, key, info)
end

actor:BindToMessageParallel("mesh", function(key: string, cx: number, cy: number, cz: number)
	local t0 = os.clock()
	local ok, result = xpcall(mesher.mesh, debug.traceback, mesher, cx, cy, cz)
	local meshMs = (os.clock() - t0) * 1000

	task.synchronize()
	if not ok then
		report(key, { meshMs = meshMs, buildMs = 0, triangles = 0, parts = 0, failures = 1, lastError = `mesh: {result}` })
		return
	end
	local t1 = os.clock()
	local okBuild, built = xpcall(ChunkBuilder.build, debug.traceback, result, cx, cy, cz, terrain)
	if not okBuild then
		report(key, { meshMs = meshMs, buildMs = 0, triangles = 0, parts = 0, failures = 1, lastError = `build: {built}` })
		return
	end
	if built.budgetExhausted then
		-- The client requeues the chunk: drop the pieces that did get built.
		ChunkBuilder.destroy(built)
	else
		builtByKey[key] = built
	end
	report(key, {
		meshMs = meshMs,
		buildMs = (os.clock() - t1) * 1000,
		triangles = result.triangleCount,
		parts = #built.parts,
		failures = built.failures,
		lastError = built.lastError,
		budgetExhausted = built.budgetExhausted,
	})
end)

actor:BindToMessage("unload", function(key: string)
	local built = builtByKey[key]
	if built then
		ChunkBuilder.destroy(built)
		builtByKey[key] = nil
	end
	actor:SetAttribute("AliveMeshes", ChunkBuilder.aliveMeshes)
	actor:SetAttribute("AliveTriangles", ChunkBuilder.aliveTriangles)
end)

-- The client may connect to `done` after this script starts: it also polls this attribute, and
-- asks again with "ping" until it hears back.
actor:BindToMessage("ping", function()
	done:Fire("ready", actor.Name)
end)
actor:SetAttribute("OptiVoxelReady", true)
done:Fire("ready", actor.Name)
