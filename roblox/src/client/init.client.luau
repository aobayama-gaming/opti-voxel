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

local S = Config.STUDS_PER_UNIT
local player = Players.LocalPlayer
local playerScripts = script.Parent

local terrainFolder = Instance.new("Folder")
terrainFolder.Name = "OptiVoxelTerrain"
terrainFolder.Parent = workspace

local done = Instance.new("BindableEvent")
done.Name = "OptiVoxelDone"
done.Parent = playerScripts

type Worker = {
	actor: Actor,
	ready: boolean,
	busy: boolean,
	busySince: number,
	sent: number,
	completed: number,
	aliveMeshes: number,
	aliveTriangles: number,
}
local workers: { [string]: Worker } = {}
local template = ReplicatedStorage:WaitForChild("OptiVoxelWorker") :: Actor
for i = 1, Config.WORKERS do
	local actor = template:Clone()
	actor.Name = `OptiVoxelWorker{i}`
	workers[actor.Name] = { actor = actor, ready = false, busy = false, busySince = 0, sent = 0, completed = 0, aliveMeshes = 0, aliveTriangles = 0 }
	actor.Parent = playerScripts
end
local startTime = os.clock()
local lastPing = 0
local lastStatusPrint = 0
-- CreateEditableMesh returned nil: dispatching more chunks only produces failures. Wait for
-- unloads to free memory (or this time to pass) before trying again.
local budgetPausedUntil = 0
local budgetHits = 0

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
	if os.clock() < budgetPausedUntil then
		return
	end
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
			worker.busySince = os.clock()
			worker.sent += 1
			inflight[chunk.key] = true
			worker.actor:SendMessage("mesh", chunk.key, chunk.x, chunk.y, chunk.z)
		end
	end
end

done.Event:Connect(function(kind: string, workerName: string, key: string?, info: any)
	local worker = workers[workerName]
	if not worker then
		return
	end
	if kind == "ready" then
		worker.ready = true
		return
	end
	worker.busy = false
	worker.completed += 1
	if not key then
		return
	end
	inflight[key] = nil
	worker.aliveMeshes = info.aliveMeshes
	worker.aliveTriangles = info.aliveTriangles
	if info.budgetExhausted then
		-- Not loaded: requeue it (nearest first, so next to go) and free what is no longer wanted.
		budgetHits += 1
		budgetPausedUntil = os.clock() + 2
		if budgetHits == 1 then
			warn(`OptiVoxel: {info.lastError}`)
		end
		if desired[key] then
			table.insert(queue, desired[key])
		end
		unloadStale()
		return
	end
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

-- Overlay with the numbers that matter for the feasibility test. Bottom-left: the chat window
-- (CoreGui, always drawn on top) owns the top-left corner.
local gui = Instance.new("ScreenGui")
gui.Name = "OptiVoxelStats"
gui.ResetOnSpawn = false
gui.DisplayOrder = 100
local label = Instance.new("TextLabel")
label.AnchorPoint = Vector2.new(0, 1)
label.Size = UDim2.fromOffset(620, 190)
label.Position = UDim2.new(0, 8, 1, -8)
label.TextWrapped = true
label.BackgroundTransparency = 0.4
label.BackgroundColor3 = Color3.new(0, 0, 0)
label.TextColor3 = Color3.new(1, 1, 1)
label.Font = Enum.Font.Code
label.TextSize = 14
label.TextXAlignment = Enum.TextXAlignment.Left
label.TextYAlignment = Enum.TextYAlignment.Top
label.Parent = gui
gui.Parent = player:WaitForChild("PlayerGui")

-- The player spawns on the surface on the server's OptiVoxelSpawn platform (src/server), which
-- holds them whether or not the ground chunk meshed. Until the character exists the camera sits at
-- the origin, deep underground: don't scan from there.
RunService.Heartbeat:Connect(function()
	local camera = workspace.CurrentCamera
	local character = player.Character
	if camera and character and character:FindFirstChild("HumanoidRootPart") then
		local p = camera.CFrame.Position / S
		local x, y, z = ChunkMath.worldToChunk(p.X, p.Y, p.Z)
		local key = `{x},{y},{z}`
		if key ~= cameraChunk then
			cameraChunk = key
			camX, camY, camZ = x, y, z
			rescan()
		end
	end
	-- Worker handshake: the "ready" event can be missed if a worker starts before the connection
	-- above, so also read the attribute it sets and re-ping the silent ones.
	local now = os.clock()
	local pingDue = now - lastPing > 1
	for _, w in workers do
		if not w.ready then
			if w.actor:GetAttribute("OptiVoxelReady") then
				w.ready = true
			elseif pingDue then
				w.actor:SendMessage("ping")
			end
		end
	end
	if pingDue then
		lastPing = now
	end

	dispatch()

	local busy, ready, stuck, sent, completed = 0, 0, 0, 0, 0
	local aliveMeshes, aliveTriangles = 0, 0
	for _, w in workers do
		local m = w.actor:GetAttribute("AliveMeshes")
		if typeof(m) == "number" then
			-- Set by unloads, which don't report through `done`.
			w.aliveMeshes = m
			w.aliveTriangles = w.actor:GetAttribute("AliveTriangles") :: number
			w.actor:SetAttribute("AliveMeshes", nil)
		end
		aliveMeshes += w.aliveMeshes
		aliveTriangles += w.aliveTriangles
		busy += if w.busy then 1 else 0
		ready += if w.ready then 1 else 0
		stuck += if w.busy and now - w.busySince > 10 then 1 else 0
		sent += w.sent
		completed += w.completed
	end
	local workerNote = ""
	if ready < Config.WORKERS and now - startTime > 5 then
		workerNote = "  WORKERS NOT STARTED: see Output for script errors"
	elseif stuck > 0 then
		workerNote = `  {stuck} STUCK >10s`
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
		`chunks {loadedCount}/{desiredCount}  queue {#queue}  inflight {sent - completed}`,
		`workers ready {ready}/{Config.WORKERS}  busy {busy}  sent {sent}  done {completed}{workerNote}`,
		`mesh median {string.format("%.1f", median(stats.meshMs))} ms  build median {string.format("%.1f", median(stats.buildMs))} ms`,
		`triangles built {stats.triangles}  parts {stats.parts}  last full load {string.format("%.1f", stats.loadSeconds)} s`,
		`editable meshes alive {aliveMeshes} ({aliveTriangles} tris)  budget full x{budgetHits}{if now < budgetPausedUntil then "  PAUSED" else ""}`,
		`failures {stats.failures}  {stats.lastError or ""}`,
	}, "\n")

	-- Same numbers in Output (and Studio's log file) every 5 s, for runs nobody is watching.
	if now - lastStatusPrint > 5 then
		lastStatusPrint = now
		print(
			`[OptiVoxel] chunks {loadedCount}/{desiredCount} queue {#queue} workers ready {ready}/{Config.WORKERS}`
				.. ` busy {busy} sent {sent} done {completed} stuck {stuck} parts {stats.parts} failures {stats.failures}`
				.. ` alive {aliveMeshes} meshes/{aliveTriangles} tris budget-full x{budgetHits}`
		)
	end
end)
