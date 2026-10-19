--!strict
-- OptiVoxel client: streams LOD chunks around the camera, like VoxelTerrain in CPU mode.
--   scan (Discoverer) -> queue by distance -> Actor workers (Mesher in parallel + ChunkBuilder)
-- Old chunks are only removed once the whole new set is loaded, so a rescan never opens holes.

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")

local OptiVoxel = ReplicatedStorage:WaitForChild("OptiVoxel")
local ChunkMath = require(OptiVoxel:WaitForChild("ChunkMath"))
local Config = require(OptiVoxel:WaitForChild("Config"))
local Discoverer = require(OptiVoxel:WaitForChild("Discoverer"))
local Sdf = require(OptiVoxel:WaitForChild("Sdf"))

local S = Config.STUDS_PER_UNIT
local player = Players.LocalPlayer
local playerScripts = script.Parent

local terrainFolder = Instance.new("Folder")
terrainFolder.Name = "OptiVoxelTerrain"
terrainFolder.Parent = workspace

local done = Instance.new("BindableEvent")
done.Name = "OptiVoxelDone"
done.Parent = playerScripts

type Worker = { actor: Actor, ready: boolean, busy: boolean }
local workers: { [string]: Worker } = {}
local template = ReplicatedStorage:WaitForChild("OptiVoxelWorker") :: Actor
for i = 1, Config.WORKERS do
	local actor = template:Clone()
	actor.Name = `OptiVoxelWorker{i}`
	workers[actor.Name] = { actor = actor, ready = false, busy = false }
	actor.Parent = playerScripts
end

local discoverer = Discoverer.new(Config.LOD_LEVELS, Config.PIXEL_ERROR, Config.VIEWPORT_HEIGHT, math.rad(Config.FOV_DEGREES))

local desired: { [string]: Discoverer.Chunk } = {}
local loaded: { [string]: string } = {} -- key -> owning worker name
local inflight: { [string]: boolean } = {}
local queue: { Discoverer.Chunk } = {}
local cameraChunk = ""
local camX, camY, camZ = 0, 0, 0

local stats = {
	scanMs = 0,
	meshMs = {} :: { number },
	buildMs = {} :: { number },
	triangles = 0,
	parts = 0,
	failures = 0,
	lastError = nil :: string?,
	loadStart = os.clock(),
	loadSeconds = 0,
}

local function median(list: { number }): number
	if #list == 0 then
		return 0
	end
	local copy = table.clone(list)
	table.sort(copy)
	return copy[math.ceil(#copy / 2)]
end

local function rescan()
	local t0 = os.clock()
	desired = discoverer:scan(camX, camY, camZ)
	stats.scanMs = (os.clock() - t0) * 1000
	queue = {}
	for key, chunk in desired do
		if not loaded[key] and not inflight[key] then
			table.insert(queue, chunk)
		end
	end
	-- Farthest first: the dispatcher pops from the end.
	table.sort(queue, function(a, b)
		local da = (a.x - camX) ^ 2 + (a.y - camY) ^ 2 + (a.z - camZ) ^ 2
		local db = (b.x - camX) ^ 2 + (b.y - camY) ^ 2 + (b.z - camZ) ^ 2
		return da > db
	end)
	stats.loadStart = os.clock()
end

local function unloadStale()
	for key, owner in loaded do
		if not desired[key] then
			workers[owner].actor:SendMessage("unload", key)
			loaded[key] = nil
		end
	end
end

local function dispatch()
	for _, worker in workers do
		if worker.ready and not worker.busy then
			local chunk: Discoverer.Chunk? = nil
			while #queue > 0 do
				local c = table.remove(queue) :: Discoverer.Chunk
				if desired[c.key] and not loaded[c.key] and not inflight[c.key] then
					chunk = c
					break
				end
			end
			if not chunk then
				return
			end
			worker.busy = true
			inflight[chunk.key] = true
			worker.actor:SendMessage("mesh", chunk.key, chunk.x, chunk.y, chunk.z)
		end
	end
end

done.Event:Connect(function(kind: string, workerName: string, key: string?, info: any)
	local worker = workers[workerName]
	if kind == "ready" then
		worker.ready = true
		return
	end
	worker.busy = false
	if not key then
		return
	end
	inflight[key] = nil
	loaded[key] = workerName
	table.insert(stats.meshMs, info.meshMs)
	table.insert(stats.buildMs, info.buildMs)
	if #stats.meshMs > 500 then
		table.remove(stats.meshMs, 1)
		table.remove(stats.buildMs, 1)
	end
	stats.triangles += info.triangles
	stats.parts += info.parts
	if info.failures > 0 then
		stats.failures += info.failures
		if info.lastError ~= stats.lastError then
			stats.lastError = info.lastError
			warn(`OptiVoxel chunk {key} failed: {info.lastError}`)
		end
	end
	if not desired[key] then
		worker.actor:SendMessage("unload", key)
		loaded[key] = nil
	end
	if #queue == 0 and next(inflight) == nil then
		stats.loadSeconds = os.clock() - stats.loadStart
		unloadStale()
	end
end)

-- Overlay with the numbers that matter for the feasibility test.
local gui = Instance.new("ScreenGui")
gui.Name = "OptiVoxelStats"
gui.ResetOnSpawn = false
local label = Instance.new("TextLabel")
label.Size = UDim2.fromOffset(460, 150)
label.Position = UDim2.fromOffset(8, 8)
label.BackgroundTransparency = 0.4
label.BackgroundColor3 = Color3.new(0, 0, 0)
label.TextColor3 = Color3.new(1, 1, 1)
label.Font = Enum.Font.Code
label.TextSize = 14
label.TextXAlignment = Enum.TextXAlignment.Left
label.TextYAlignment = Enum.TextYAlignment.Top
label.Parent = gui
gui.Parent = player:WaitForChild("PlayerGui")

-- Spawn on the surface, frozen until the ground under the player exists.
local function onCharacter(character: Model)
	local root = character:WaitForChild("HumanoidRootPart") :: BasePart
	root.Anchored = true
	local surface = Sdf.height(0, 0)
	character:PivotTo(CFrame.new(0, (surface + 2) * S, 0))
	local x, y, z = ChunkMath.worldToChunk(0, surface, 0)
	local groundKey = `{x},{y},{z}`
	local deadline = os.clock() + 60
	while not loaded[groundKey] and os.clock() < deadline do
		task.wait(0.1)
	end
	root.Anchored = false
end
player.CharacterAdded:Connect(onCharacter)
if player.Character then
	task.spawn(onCharacter, player.Character)
end

RunService.Heartbeat:Connect(function()
	local camera = workspace.CurrentCamera
	if camera then
		local p = camera.CFrame.Position / S
		local x, y, z = ChunkMath.worldToChunk(p.X, p.Y, p.Z)
		local key = `{x},{y},{z}`
		if key ~= cameraChunk then
			cameraChunk = key
			camX, camY, camZ = x, y, z
			rescan()
		end
	end
	dispatch()

	local busy, ready = 0, 0
	for _, w in workers do
		busy += if w.busy then 1 else 0
		ready += if w.ready then 1 else 0
	end
	local loadedCount = 0
	for _ in loaded do
		loadedCount += 1
	end
	local desiredCount = 0
	for _ in desired do
		desiredCount += 1
	end
	label.Text = table.concat({
		`OptiVoxel  camera chunk {cameraChunk}  scan {string.format("%.1f", stats.scanMs)} ms`,
		`chunks {loadedCount}/{desiredCount}  queue {#queue}  workers {busy}/{ready} busy`,
		`mesh median {string.format("%.1f", median(stats.meshMs))} ms  build median {string.format("%.1f", median(stats.buildMs))} ms`,
		`triangles built {stats.triangles}  parts {stats.parts}  last full load {string.format("%.1f", stats.loadSeconds)} s`,
		`failures {stats.failures}  {stats.lastError or ""}`,
	}, "\n")
end)
