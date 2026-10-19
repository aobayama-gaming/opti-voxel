--!strict
-- Smooth Terrain playground, client side: dig / fill tool and overlay.
--   Left mouse (hold): dig      F (hold): fill      R / T: radius -/+      H: hide overlay
-- Edits go to the server (OptiVoxelEdit), which updates the SDF and the Terrain for everyone.

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local Stats = game:GetService("Stats")
local UserInputService = game:GetService("UserInputService")

local player = Players.LocalPlayer
local edit = ReplicatedStorage:WaitForChild("OptiVoxelEdit") :: RemoteEvent
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
label.Size = UDim2.fromOffset(560, 164)
label.BackgroundTransparency = 0.4
label.BackgroundColor3 = Color3.new(0, 0, 0)
label.TextColor3 = Color3.new(1, 1, 1)
label.Font = Enum.Font.Code
label.TextSize = 14
label.TextXAlignment = Enum.TextXAlignment.Left
label.TextYAlignment = Enum.TextYAlignment.Top
label.Parent = gui
gui.Parent = player:WaitForChild("PlayerGui")

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
	local now = os.clock()
	if target and (digging or filling) and now - lastSend > REPEAT then
		lastSend = now
		edit:FireServer(if digging then "dig" else "fill", target, radius)
	end

	local root = player.Character and player.Character:FindFirstChild("HumanoidRootPart") :: BasePart?
	local pos = if root then root.Position else Vector3.zero
	local genDone, genTotal = num("GenDone"), num("GenTotal")
	local half = num("TerrainHalfStuds")
	label.Text = table.concat({
		`Smooth Terrain  {half * 2}x{half * 2} studs   fps {math.floor(1 / math.max(dt, 1e-3))}`,
		if genDone < genTotal
			then `generating slab {genDone}/{genTotal}`
			else `generated in {string.format("%.1f", num("GenSeconds"))} s: {num("GenVoxels")} voxels (sample {math.floor(num("GenSampleMs"))} ms, write {math.floor(num("GenWriteMs"))} ms)`,
		if num("FarTotal") == 0 then "no far field"
			elseif num("FarDone") < num("FarTotal") then `far field {num("FarHalfStuds") * 2} studs: tile {num("FarDone")}/{num("FarTotal")}`
			else `far field {num("FarHalfStuds") * 2} studs in {string.format("%.1f", num("FarSeconds"))} s: {num("FarVoxels")} voxels (sample {math.floor(num("FarSampleMs"))} ms, write {math.floor(num("FarWriteMs"))} ms)`,
		`terrain voxels {math.floor(Stats:GetMemoryUsageMbForTag(Enum.DeveloperMemoryTag.TerrainVoxels))} MB   total {math.floor(Stats:GetTotalMemoryUsageMb())} MB`,
		`last edit: sample {string.format("%.2f", num("LastEditSampleMs"))} ms  write {string.format("%.2f", num("LastEditWriteMs"))} ms  ({num("LastEditVoxels")} voxels, {num("EditCount")} edits)`,
		`brush radius {string.format("%.1f", radius / Config.STUDS_PER_UNIT)} m   position {math.floor(pos.X / Config.STUDS_PER_UNIT)}, {math.floor(pos.Y / Config.STUDS_PER_UNIT)}, {math.floor(pos.Z / Config.STUDS_PER_UNIT)} m   voxel {4 / Config.STUDS_PER_UNIT} m`,
		"",
		"LMB dig   F fill   R/T radius   H hide",
	}, "\n")
end)
