--!strict
-- EditableMesh budget probe (probe.project.json). Answers, in Output / Studio's log:
--   1. how many EditableMeshes a client can keep alive, and whether triangle count matters;
--   2. whether a MeshPart copied from an EditableMesh-backed part (ApplyMesh / Clone) stays
--      visible and frees the budget once the EditableMesh is destroyed.
-- Look in front of the spawn: RED = ApplyMesh copy, GREEN = Clone, BLUE = original part whose
-- EditableMesh was destroyed (expected invisible), WHITE = original part, EditableMesh kept.

local AssetService = game:GetService("AssetService")
local Players = game:GetService("Players")

local function log(msg: string)
	print(`[Probe] {msg}`)
end

local function fill(mesh: EditableMesh, triangles: number)
	-- A strip of quads, 2 triangles each, in a 20 x 20 stud square.
	local quads = math.max(1, triangles // 2)
	local side = math.ceil(math.sqrt(quads))
	local step = 20 / side
	local n = 0
	for i = 0, side - 1 do
		for j = 0, side - 1 do
			if n >= quads then
				return
			end
			n += 1
			local x, z = i * step - 10, j * step - 10
			local a = mesh:AddVertex(Vector3.new(x, 0, z))
			local b = mesh:AddVertex(Vector3.new(x + step, 0, z))
			local c = mesh:AddVertex(Vector3.new(x + step, 0, z + step))
			local d = mesh:AddVertex(Vector3.new(x, 0, z + step))
			mesh:AddTriangle(a, c, b)
			mesh:AddTriangle(a, d, c)
		end
	end
end

local function countAlive(triangles: number): number
	local meshes = {}
	for _ = 1, 200 do
		local ok, mesh = pcall(function()
			return AssetService:CreateEditableMesh()
		end)
		if not ok or not mesh then
			break
		end
		local okFill = pcall(fill, mesh, triangles)
		table.insert(meshes, mesh)
		if not okFill then
			break
		end
	end
	local n = #meshes
	for _, m in meshes do
		m:Destroy()
	end
	return n
end

local base = Vector3.new(0, 6, 0)

local function makePart(color: Color3, x: number): (MeshPart?, EditableMesh?)
	local mesh = AssetService:CreateEditableMesh()
	if not mesh then
		return nil, nil
	end
	fill(mesh, 2)
	local part = AssetService:CreateMeshPartAsync(Content.fromObject(mesh))
	part.Anchored = true
	part.Color = color
	part.Material = Enum.Material.Neon
	part.Size = Vector3.new(16, 1, 16)
	part.CFrame = CFrame.new(base + Vector3.new(x, 0, -30))
	part.Parent = workspace
	return part, mesh
end

local function sourceOf(part: MeshPart): string
	local ok, r = pcall(function()
		return tostring(part.MeshContent.SourceType)
	end)
	return if ok then r else `? ({r})`
end

local function run()
	task.wait(3)
	-- 1. Budget shape.
	for _, tris in { 0, 2, 2000, 19000 } do
		log(`alive EditableMeshes with {tris} triangles each: {countAlive(tris)}`)
	end

	-- 2. Decoupling.
	local white = makePart(Color3.new(1, 1, 1), -54)
	local blue, blueMesh = makePart(Color3.fromRGB(40, 80, 255), -18)
	local src, srcMesh = makePart(Color3.new(1, 0, 0), 200)
	if not (white and blue and blueMesh and src and srcMesh) then
		log("could not create the test parts (budget already spent?)")
		return
	end
	local red = Instance.new("MeshPart")
	local okApply, errApply = pcall(function()
		red:ApplyMesh(src)
	end)
	red.Anchored = true
	red.Color = Color3.new(1, 0, 0)
	red.Material = Enum.Material.Neon
	red.Size = Vector3.new(16, 1, 16)
	red.CFrame = CFrame.new(base + Vector3.new(18, 0, -30))
	red.Parent = workspace
	local green = src:Clone()
	green.Color = Color3.new(0, 1, 0)
	green.CFrame = CFrame.new(base + Vector3.new(54, 0, -30))
	green.Parent = workspace
	log(`ApplyMesh ok={okApply} {errApply or ""}`)
	log(`before destroy: red source={sourceOf(red)} green source={sourceOf(green)} src={sourceOf(src)}`)

	srcMesh:Destroy()
	src:Destroy()
	blueMesh:Destroy()
	task.wait(1)
	log(`after destroy: red source={sourceOf(red)} size={red.MeshSize} green source={sourceOf(green)} size={green.MeshSize}`)

	-- 3. Does copy-then-destroy free the budget? Build many copies, each from a fresh EditableMesh.
	local copies = 0
	for i = 1, 60 do
		local p, m = makePart(Color3.new(1, 1, 0), 1000 + i * 20)
		if not p or not m then
			break
		end
		local c = Instance.new("MeshPart")
		local ok = pcall(function()
			c:ApplyMesh(p)
		end)
		m:Destroy()
		p:Destroy()
		c:Destroy()
		if not ok then
			break
		end
		copies += 1
	end
	log(`ApplyMesh copies made with the EditableMesh destroyed each time: {copies} (60 = budget freed)`)
	log("done: tell Claude which of WHITE / BLUE / RED / GREEN you see in front of the spawn")
end

local character = Players.LocalPlayer.Character or Players.LocalPlayer.CharacterAdded:Wait()
local root = character:WaitForChild("HumanoidRootPart") :: BasePart
task.wait(1)
base = root.Position + Vector3.new(0, 4, 0)
run()
