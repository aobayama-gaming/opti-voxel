--!strict
-- Smooth Terrain playground, client side: dig / fill tool and overlay.
--   Left mouse (hold): dig      F (hold): fill      R / T: radius -/+      H: hide overlay
--   J: treadmill jump now (the server writes the window elsewhere, then we jump)      B: walk x3
-- Edits go to the server (OptiVoxelEdit), which updates the SDF and the Terrain for everyone.
-- Positions: logical = physical + origin. The origin changes on a treadmill jump: the server sends
-- the move, we apply it (character and camera in one frame) and answer "landed"; edits carry the
-- origin version their position belongs to.

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local Stats = game:GetService("Stats")
local UserInputService = game:GetService("UserInputService")

local player = Players.LocalPlayer
local edit = ReplicatedStorage:WaitForChild("OptiVoxelEdit") :: RemoteEvent
local treadmill = ReplicatedStorage:WaitForChild("OptiVoxelTreadmill") :: RemoteEvent
local camera = workspace.CurrentCamera

local Config = require(ReplicatedStorage:WaitForChild("OptiVoxel"):WaitForChild("Config"))
local K = Config.STUDS_PER_UNIT / 3 -- player scale, as on the server
local radius = 8 * K -- studs
local REPEAT = 0.08 -- seconds between edits while held

-- Brush preview.
local brush = Instance.new("Part")
brush.Name = "OptiVoxelBrush"
brush.Shape = Enum.PartType.Ball
brush.Anchored = true
brush.CanCollide = false
brush.CanQuery = false
brush.CanTouch = false
brush.CastShadow = false
brush.Material = Enum.Material.ForceField
brush.Color = Color3.fromRGB(255, 200, 60)
brush.Parent = workspace

local params = RaycastParams.new()
params.FilterType = Enum.RaycastFilterType.Include
params.FilterDescendantsInstances = { workspace.Terrain }

local function aim(): Vector3?
	local m = UserInputService:GetMouseLocation()
	local ray = camera:ViewportPointToRay(m.X, m.Y)
	local hit = workspace:Raycast(ray.Origin, ray.Direction * 2000 * K, params)
	return if hit then hit.Position else nil
end

-- Overlay, bottom-left (the chat owns the top-left corner).
local gui = Instance.new("ScreenGui")
gui.Name = "OptiVoxelTerrainStats"
gui.ResetOnSpawn = false
local label = Instance.new("TextLabel")
label.AnchorPoint = Vector2.new(0, 1)
label.Position = UDim2.new(0, 8, 1, -8)
label.Size = UDim2.fromOffset(640, 184)
label.BackgroundTransparency = 0.4
label.BackgroundColor3 = Color3.new(0, 0, 0)
label.TextColor3 = Color3.new(1, 1, 1)
label.Font = Enum.Font.Code
label.TextSize = 14
label.TextXAlignment = Enum.TextXAlignment.Left
label.TextYAlignment = Enum.TextYAlignment.Top
label.Parent = gui
gui.Parent = player:WaitForChild("PlayerGui")

-- Treadmill: the origin we are in, and the frame times after the last jump.
local origin = workspace:GetAttribute("Origin")
local originVersion = workspace:GetAttribute("OriginVersion")
if typeof(origin) ~= "Vector3" or typeof(originVersion) ~= "number" then
	origin, originVersion = Vector3.zero, 0
end
local jumps, watch, worstMs = 0, 0, 0
treadmill.OnClientEvent:Connect(function(kind: string, delta: Vector3, version: number, newOrigin: Vector3)
	if kind ~= "jump" then
		return
	end
	if version > originVersion then
		local character = player.Character
		if character then
			character:PivotTo(character:GetPivot() + delta)
		end
		camera.CFrame += delta
		camera.Focus += delta
		origin, originVersion = newOrigin, version
		jumps += 1
		watch, worstMs = 120, 0
	end
	treadmill:FireServer("landed", version)
end)

local fast = false
local function toggleFast()
	local humanoid = player.Character and player.Character:FindFirstChildOfClass("Humanoid")
	if humanoid then
		fast = not fast
		humanoid.WalkSpeed *= if fast then 3 else 1 / 3
	end
end
player.CharacterAdded:Connect(function()
	fast = false
end)

local digging, filling = false, false
UserInputService.InputBegan:Connect(function(input, processed)
	if processed then
		return
	end
	if input.UserInputType == Enum.UserInputType.MouseButton1 then
		digging = true
	elseif input.KeyCode == Enum.KeyCode.F then
		filling = true
	elseif input.KeyCode == Enum.KeyCode.R then
		radius = math.max(2 * K, radius - 2 * K)
	elseif input.KeyCode == Enum.KeyCode.T then
		radius = math.min(40 * K, radius + 2 * K)
	elseif input.KeyCode == Enum.KeyCode.H then
		label.Visible = not label.Visible
	elseif input.KeyCode == Enum.KeyCode.J then
		treadmill:FireServer("force")
	elseif input.KeyCode == Enum.KeyCode.B then
		toggleFast()
	end
end)
UserInputService.InputEnded:Connect(function(input)
	if input.UserInputType == Enum.UserInputType.MouseButton1 then
		digging = false
	elseif input.KeyCode == Enum.KeyCode.F then
		filling = false
	end
end)

local function num(name: string): number
	local v = workspace:GetAttribute(name)
	return if typeof(v) == "number" then v else 0
end

local function metres(studs: number): string
	return `{math.floor(studs / Config.STUDS_PER_UNIT)} m`
end

local lastSend = 0
RunService.RenderStepped:Connect(function(dt)
	local target = aim()
	brush.Size = Vector3.one * radius * 2
	if target then
		brush.Position = target
		brush.Transparency = 0.6
	else
		brush.Transparency = 1
	end
	if watch > 0 then
		watch -= 1
		worstMs = math.max(worstMs, dt * 1000)
	end
	local now = os.clock()
	if target and (digging or filling) and now - lastSend > REPEAT then
		lastSend = now
		edit:FireServer(if digging then "dig" else "fill", target, radius, originVersion)
	end

	local root = player.Character and player.Character:FindFirstChild("HumanoidRootPart") :: BasePart?
	local physical = if root then root.Position else Vector3.zero
	local pos = physical + origin
	local state = workspace:GetAttribute("TreadmillState")
	local load = if num("LoadSeconds") > 0 then `loaded in {string.format("%.1f", num("LoadSeconds"))} s` else "loading"
	local progress = if state == "copy"
		then `: {num("CopySlabs")} slabs, {num("CopyQueued")} queued`
		elseif state == "clearing" then `: {num("RetireQueued")} left`
		else ""
	label.Text = table.concat({
		`Smooth Terrain streaming   fps {math.floor(1 / math.max(dt, 1e-3))}   voxel {4 / Config.STUDS_PER_UNIT} m`,
		`window: {num("StreamNear")} near + {num("StreamCrust")} crust slabs, {num("StreamQueued")} queued ({num("StreamQueuedNear")} near), {num("StreamCleared")} cleared`,
		`server {string.format("%.1f", num("StreamFrameMs"))} ms/frame (max {math.floor(num("StreamFrameMaxMs"))})   sample {math.floor(num("StreamSampleMs") / 1000)} s  write {math.floor(num("StreamWriteMs") / 1000)} s  clear {math.floor(num("StreamClearMs"))} ms  rebuild max {math.floor(num("StreamRebuildMs"))} ms`,
		`{load}   unwritten near slab in {metres(num("StreamEdge"))}`,
		`terrain voxels {math.floor(Stats:GetMemoryUsageMbForTag(Enum.DeveloperMemoryTag.TerrainVoxels))} MB   total {math.floor(Stats:GetTotalMemoryUsageMb())} MB`,
		`treadmill {state}{progress}`
			.. `   next in {metres(num("TreadmillLeft"))}   jumps {jumps}   worst frame after jump {math.floor(worstMs)} ms{if watch > 0 then " (watching)" else ""}`,
		`last edit: sample {string.format("%.2f", num("LastEditSampleMs"))} ms  write {string.format("%.2f", num("LastEditWriteMs"))} ms  ({num("LastEditVoxels")} voxels, {num("EditCount")} edits)`,
		`position {metres(pos.X)}, {metres(pos.Y)}, {metres(pos.Z)}   in Roblox space {metres(physical.X)}, {metres(physical.Z)}   brush {string.format("%.1f", radius / Config.STUDS_PER_UNIT)} m{if fast then "   walk x3" else ""}`,
		"",
		"LMB dig   F fill   R/T radius   J treadmill jump   B walk x3   H hide",
	}, "\n")
end)
