--!strict
--!native
-- Smooth Terrain playground, server side (terrain.project.json):
--   - streams the SDF terrain into Roblox Terrain around the player (Window.luau): full detail near,
--     a thin shell farther out, cleared behind, within a time budget per frame;
--   - treadmill: past Config.TREADMILL_STUDS from the box centre, writes the window again around the
--     centre (out of view), then the client jumps there in one frame (J forces one);
--   - applies dig / fill requests from clients as SDF edits, then resamples the edited box.
-- Positions: logical (the world, what the SDF and edits see) = physical (Roblox space) + origin. Only
-- this script moves the origin; it publishes it as workspace attributes Origin / OriginVersion.
-- Solo for now: the window follows the first player with a character, and a jump moves every client.
-- Terrain is edited here, on the server, so it replicates to every player.
-- Progress and timings go to workspace attributes (read by the client overlay).

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local StarterPlayer = game:GetService("StarterPlayer")

local OptiVoxel = ReplicatedStorage:WaitForChild("OptiVoxel")
local Config = require(OptiVoxel:WaitForChild("Config"))
local Sdf = require(OptiVoxel:WaitForChild("Sdf"))
local TerrainGen = require(OptiVoxel:WaitForChild("TerrainGen"))
local Window = require(script:WaitForChild("Window"))

local S = Config.STUDS_PER_UNIT
local K = S / 3 -- player scale: Roblox's default avatar is sized for ~3 studs per metre
local SLAB = Config.STREAM_SLAB_STUDS
local BUDGET = Config.STREAM_BUDGET_MS / 1000
local COPY_BUDGET = Config.STREAM_COPY_BUDGET_MS / 1000
local TREADMILL = Config.TREADMILL_STUDS
-- A copy closer than this would overlap the live window (both reach the crust's half side + a slab).
local MIN_JUMP = 2 * (Config.TERRAIN_FAR_HALF_STUDS + 2 * SLAB)
local TEST_JUMP = 40960 -- forced jumps (J) near the box centre go this far on z
-- A tracked position moving more than this in a frame is a teleport (or a stale replica right after a
-- jump): only followed once it lasts TELEPORT_ACCEPT seconds, or on a respawn.
local TELEPORT_STUDS = 512
local TELEPORT_ACCEPT = 1
local terrain = workspace.Terrain

-- Origin, and the origin of every version (edits carry the version their position was taken in).
local origin = Vector3.zero
local originVersion = 0
local origins: { [number]: Vector3 } = { [0] = origin }
local function publishOrigin()
	workspace:SetAttribute("Origin", origin)
	workspace:SetAttribute("OriginVersion", originVersion)
end
publishOrigin()

-- Spawn platform until the ground under it exists. Highest surface point under its footprint;
-- never Sdf.height(0, 0) itself: the simplex noise is degenerate exactly on that lattice point
-- (257.8 there, ~300 everywhere around it).
local surface = -math.huge
for dx = -8, 8, 2 do
	for dz = -8, 8, 2 do
		surface = math.max(surface, Sdf.height((dx * K + 0.37) / S, (dz * K + 0.37) / S) * S)
	end
end
local spawnLogical = Vector3.new(0, surface + 2 * K, 0)
local spawn = Instance.new("SpawnLocation")
spawn.Name = "OptiVoxelSpawn"
spawn.Anchored = true
spawn.Neutral = true
spawn.Duration = 0
spawn.Size = Vector3.new(16 * K, 1, 16 * K)
spawn.Transparency = 0.5
spawn.Position = spawnLogical - origin
spawn.Parent = workspace

local live = Window.new(terrain, origin)
local copy: Window.Window? = nil -- the treadmill's next window, being written
local retiring: Window.Window? = nil -- the window left behind by the last jump, being cleared
live.setCenter(spawnLogical.X, spawnLogical.Z)

-- Bigger players: PLAYER_HEIGHT_M tall whatever the avatar, same timings (jump duration, fall time).
workspace.Gravity = 196.2 * K

-- Scaling is about the pivot: put the feet back on top of the spawn.
local function placeOnSpawn(character: Model)
	local size = character:GetExtentsSize()
	character:PivotTo(CFrame.new(spawn.Position + Vector3.new(0, spawn.Size.Y / 2 + size.Y / 2 + K, 0)))
end

-- Head top to feet, in studs: body parts only (accessories and the root box excluded).
local function bodyHeight(character: Model): number
	local lo, hi = math.huge, -math.huge
	for _, part in character:GetChildren() do
		if part:IsA("BasePart") and part.Name ~= "HumanoidRootPart" then
			lo = math.min(lo, part.Position.Y - part.Size.Y / 2)
			hi = math.max(hi, part.Position.Y + part.Size.Y / 2)
		end
	end
	return hi - lo
end

local recenter = true -- the next tracked position is followed whatever its jump (a spawn)

local function scaleCharacter(player: Player, character: Model)
	if not player:HasAppearanceLoaded() then
		player.CharacterAppearanceLoaded:Wait()
	end
	local humanoid = character:WaitForChild("Humanoid") :: Humanoid
	-- The rig is only laid out after a physics step: measured earlier, the parts overlap and the body
	-- reads short. Measure, scale, then correct once against the scaled body.
	while not character:IsDescendantOf(workspace) do
		character.AncestryChanged:Wait()
	end
	RunService.Heartbeat:Wait()
	RunService.Heartbeat:Wait()
	local target = Config.PLAYER_HEIGHT_M * S
	local scale = character:GetScale() * target / bodyHeight(character)
	character:ScaleTo(scale)
	RunService.Heartbeat:Wait()
	scale *= target / bodyHeight(character)
	character:ScaleTo(scale)
	humanoid.WalkSpeed = 16 * scale
	humanoid.JumpPower = 50 * scale
	humanoid.JumpHeight = 7.2 * scale
	placeOnSpawn(character)
	-- Respawned where the window isn't written yet (far from where the player died): hold it there.
	local root = character:FindFirstChild("HumanoidRootPart") :: BasePart?
	if root and not live.nearReady(spawnLogical.X, spawnLogical.Z, 1) then
		root.Anchored = true
		while character.Parent and not live.nearReady(spawnLogical.X, spawnLogical.Z, 1) do
			RunService.Heartbeat:Wait()
		end
		placeOnSpawn(character)
		root.Anchored = false
	end
end
-- Roblox copies these from StarterPlayer onto the player at every spawn: set them there.
StarterPlayer.CameraMaxZoomDistance = 128 * K
StarterPlayer.CameraMinZoomDistance = 0.5 * K

local function setupPlayer(player: Player)
	player.CameraMaxZoomDistance = StarterPlayer.CameraMaxZoomDistance
	player.CameraMinZoomDistance = StarterPlayer.CameraMinZoomDistance
	player.CharacterAdded:Connect(function(character)
		recenter = true
		scaleCharacter(player, character)
	end)
	if player.Character then
		task.spawn(scaleCharacter, player, player.Character)
	end
end
Players.PlayerAdded:Connect(setupPlayer)
for _, player in Players:GetPlayers() do
	setupPlayer(player)
end

local terrainOnly = RaycastParams.new()
terrainOnly.FilterType = Enum.RaycastFilterType.Include
terrainOnly.FilterDescendantsInstances = { terrain }

-- Puts the spawn on the ground under its column (the height field misses the cave and overhang
-- terms: it read ~20 m too high at the first spawn) and the players waiting on it.
local function settleSpawn(movePlayers: boolean)
	local hit = workspace:Raycast(spawn.Position + Vector3.new(0, 100 * K, 0), Vector3.new(0, -400 * K, 0), terrainOnly)
	if hit then
		spawn.Position = hit.Position + Vector3.new(0, K, 0)
		spawnLogical = spawn.Position + origin
	end
	if movePlayers then
		for _, player in Players:GetPlayers() do
			if player.Character then
				placeOnSpawn(player.Character)
			end
		end
	end
end

-- The tracked player's logical position, followed by the windows.
local center: Vector3 = spawnLogical
local teleportSince: number? = nil
local jumping: { version: number, origin: Vector3, since: number }? = nil

local function trackedRoot(): BasePart?
	for _, player in Players:GetPlayers() do
		local root = player.Character and player.Character:FindFirstChild("HumanoidRootPart")
		if root and root:IsA("BasePart") then
			return root
		end
	end
	return nil
end

local function updateCenter()
	local root = trackedRoot()
	if not root or jumping then
		return
	end
	local p = root.Position + origin
	if not recenter and Vector2.new(p.X - center.X, p.Z - center.Z).Magnitude > TELEPORT_STUDS then
		teleportSince = teleportSince or os.clock()
		if os.clock() - (teleportSince :: number) < TELEPORT_ACCEPT then
			return
		end
	end
	teleportSince, recenter = nil, false
	center = p
	live.setCenter(p.X, p.Z)
	if copy then
		copy.setCenter(p.X, p.Z)
	end
end

-- Treadmill. A jump moves the origin by D (slab multiple): the player lands near the box centre.
local treadmill = Instance.new("RemoteEvent")
treadmill.Name = "OptiVoxelTreadmill"
treadmill.Parent = ReplicatedStorage
local jumps = 0
local copyStarted = 0
local lastJumpSeconds = 0

local function startCopy(force: boolean)
	if copy or jumping or retiring then
		return
	end
	local p = center - origin
	local d = Vector3.new(math.round(p.X / SLAB) * SLAB, 0, math.round(p.Z / SLAB) * SLAB)
	if math.max(math.abs(d.X), math.abs(d.Z)) < MIN_JUMP then
		if not force then
			return
		end
		d += Vector3.new(0, 0, TEST_JUMP)
	end
	local w = Window.new(terrain, origin + d)
	w.setCenter(center.X, center.Z)
	copy = w
	copyStarted = os.clock()
	print(`[OptiVoxel] treadmill: writing the window at origin {origin + d}`)
end

local function finishJump(version: number)
	local j = jumping
	local next = copy
	if not j or j.version ~= version or not next then
		return
	end
	origin = j.origin
	live.retire()
	retiring, live, copy, jumping = live, next, nil, nil
	publishOrigin()
	jumps += 1
	lastJumpSeconds = os.clock() - copyStarted
	-- The spawn keeps its logical place while it stays inside the box; past the treadmill distance it
	-- becomes a checkpoint where the player landed.
	spawn.Position = spawnLogical - origin
	if math.max(math.abs(spawn.Position.X), math.abs(spawn.Position.Z)) > TREADMILL then
		spawn.Position = center - origin
		settleSpawn(false)
	end
	print(`[OptiVoxel] treadmill: jump {jumps} done, origin {origin}, {string.format("%.1f", lastJumpSeconds)} s after the trigger`)
end

-- Clients ignore a jump to a version they already have (and say "landed" again): resending is safe.
local JUMP_RESEND = 2

local function updateTreadmill()
	local j = jumping
	if j then
		if os.clock() - j.since > JUMP_RESEND then
			j.since = os.clock()
			treadmill:FireAllClients("jump", origin - j.origin, j.version, j.origin)
		end
		return
	end
	if copy then
		if copy.complete() then
			originVersion += 1
			jumping = { version = originVersion, origin = copy.origin, since = os.clock() }
			origins[originVersion] = copy.origin
			if #Players:GetPlayers() == 0 then
				finishJump(originVersion)
			else
				treadmill:FireAllClients("jump", origin - copy.origin, originVersion, copy.origin)
			end
		end
		return
	end
	local p = center - origin
	if math.max(math.abs(p.X), math.abs(p.Z)) > TREADMILL then
		startCopy(false)
	end
end

treadmill.OnServerEvent:Connect(function(_player: Player, kind: unknown, version: unknown)
	if kind == "force" then
		startCopy(true)
	elseif kind == "landed" and typeof(version) == "number" then
		finishJump(version)
	end
end)

-- Frame loop: follow the player, write within the budget, publish stats.
local loadStart = os.clock()
local spawnSettled, platformDropped, loaded = false, false, false
local frameMs, frameMax, frames, statsAt = 0, 0, 0, 0

local function publishStats(now: number)
	local st = live.stats
	workspace:SetAttribute("StreamNear", st.near)
	workspace:SetAttribute("StreamCrust", st.crust)
	workspace:SetAttribute("StreamQueued", st.queued)
	workspace:SetAttribute("StreamQueuedNear", st.queuedNear)
	workspace:SetAttribute("StreamCleared", st.cleared + (if retiring then retiring.stats.cleared else 0))
	workspace:SetAttribute("StreamSampleMs", st.sampleMs)
	workspace:SetAttribute("StreamWriteMs", st.writeMs)
	workspace:SetAttribute("StreamClearMs", st.clearMs)
	workspace:SetAttribute("StreamRebuildMs", st.rebuildMs)
	workspace:SetAttribute("StreamSlowestJob", `{st.slowestJob} {math.floor(st.slowestJobMs)} ms`)
	workspace:SetAttribute("StreamFrameMs", frameMs / math.max(frames, 1))
	workspace:SetAttribute("StreamFrameMaxMs", frameMax)
	workspace:SetAttribute("StreamEdge", live.edge())
	workspace:SetAttribute("CopyQueued", if copy then copy.stats.queued else 0)
	workspace:SetAttribute("CopySlabs", if copy then copy.stats.near + copy.stats.crust else 0)
	workspace:SetAttribute("RetireQueued", if retiring then retiring.stats.queued else 0)
	workspace:SetAttribute(
		"TreadmillState",
		if jumping then "jumping" elseif copy then "copy" elseif retiring then "clearing" else "idle"
	)
	workspace:SetAttribute("TreadmillJumps", jumps)
	workspace:SetAttribute("TreadmillLastSeconds", lastJumpSeconds)
	local p = center - origin
	workspace:SetAttribute("TreadmillLeft", TREADMILL - math.max(math.abs(p.X), math.abs(p.Z)))
	frameMs, frameMax, frames, statsAt = 0, 0, 0, now
end

RunService.Heartbeat:Connect(function()
	local t0 = os.clock()
	updateCenter()
	live.step(t0 + BUDGET)
	local w = copy
	if w then
		w.step(math.max(t0 + BUDGET, os.clock() + COPY_BUDGET))
	end
	local r = retiring
	if r then
		r.step(math.max(t0 + BUDGET, os.clock() + COPY_BUDGET))
		if r.complete() then
			retiring = nil
		end
	end
	updateTreadmill()

	if not spawnSettled and live.nearReady(spawnLogical.X, spawnLogical.Z, 1) then
		spawnSettled = true
		settleSpawn(true)
	end
	-- Terrain collision is built asynchronously (on each client too): dropping the platform as soon
	-- as the ground under it was written let players fall through into the caves. Drop it once the
	-- near square is written and has had time to settle.
	if not platformDropped and live.stats.queuedNear == 0 and live.complete() then
		platformDropped = true
		task.delay(3, function()
			spawn.CanCollide = false
			spawn.Transparency = 1
		end)
	end
	if not loaded and live.complete() then
		loaded = true
		workspace:SetAttribute("LoadSeconds", os.clock() - loadStart)
		local st = live.stats
		print(
			`[OptiVoxel] window loaded in {string.format("%.1f", os.clock() - loadStart)} s: {st.near} near + {st.crust} crust slabs,`
				.. ` sample {math.floor(st.sampleMs)} ms, WriteVoxels {math.floor(st.writeMs)} ms`
		)
	end

	local ms = (os.clock() - t0) * 1000
	frameMs, frameMax, frames = frameMs + ms, math.max(frameMax, ms), frames + 1
	if t0 - statsAt > 0.5 then
		publishStats(t0)
	end
end)

-- Edits: SDF spheres, in studs on the wire (physical, in the sender's origin version), units in the SDF.
local edit = Instance.new("RemoteEvent")
edit.Name = "OptiVoxelEdit"
edit.Parent = ReplicatedStorage
local lastEdit: { [Player]: number } = {}
edit.OnServerEvent:Connect(function(player: Player, kind: unknown, position: unknown, radius: unknown, version: unknown)
	if typeof(position) ~= "Vector3" or typeof(radius) ~= "number" or (kind ~= "dig" and kind ~= "fill") then
		return
	end
	local from = if typeof(version) == "number" then origins[version] else nil
	if not from then
		return
	end
	local now = os.clock()
	if now - (lastEdit[player] or 0) < 0.05 then
		return
	end
	lastEdit[player] = now
	local r = math.clamp(radius, 2 * K, 40 * K)
	local c = position + from
	local p = c / S
	if kind == "dig" then
		Sdf.addSubtractSphere(p.X, p.Y, p.Z, r / S)
	else
		Sdf.addUnionSphere(p.X, p.Y, p.Z, r / S)
	end
	local m = r + TerrainGen.VOXEL * 2
	local a, b, n = live.applyEdit(c.X - m, c.Y - m, c.Z - m, c.X + m, c.Y + m, c.Z + m)
	if copy then
		local sa, sb, sn = copy.applyEdit(c.X - m, c.Y - m, c.Z - m, c.X + m, c.Y + m, c.Z + m)
		a, b, n = a + sa, b + sb, n + sn
	end
	workspace:SetAttribute("LastEditSampleMs", a)
	workspace:SetAttribute("LastEditWriteMs", b)
	workspace:SetAttribute("LastEditVoxels", n)
	workspace:SetAttribute("EditCount", Sdf.editCount())
end)

Players.PlayerRemoving:Connect(function(player)
	lastEdit[player] = nil
end)
