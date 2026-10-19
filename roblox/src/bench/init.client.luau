--!strict
--!native
-- Smooth Terrain vs pack-of-chunks benchmark (bench.project.json). Same SDF, same region, same edits.
--   workspace attributes (set before Play): BenchMode = "pack" | "terrain", BenchHalfStuds (168),
--   BenchLod (1, pack only).
-- "pack": our mesher; chunks appended into at most 8 shared EditableMeshes (the client cap), one
--   MeshPart each, like QuadPagePool pages in Godot. Edits re-mesh the touched chunks in place.
-- "terrain": the SDF sampled at Roblox's 4-stud voxels into Terrain:WriteVoxels. Edits resample
--   the sphere's box. Terrain:FillBall is timed too, as the native reference.
-- Prints one "[Bench] {json}" line and sets workspace.BenchDone.

local AssetService = game:GetService("AssetService")
local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local Stats = game:GetService("Stats")

local OptiVoxel = ReplicatedStorage:WaitForChild("OptiVoxel")
local ChunkMath = require(OptiVoxel:WaitForChild("ChunkMath"))
local Config = require(OptiVoxel:WaitForChild("Config"))
local Mesher = require(OptiVoxel:WaitForChild("Mesher"))
local Sdf = require(OptiVoxel:WaitForChild("Sdf"))

local S = Config.STUDS_PER_UNIT
local MODE = (workspace:GetAttribute("BenchMode") or "pack") :: string
local HALF = (workspace:GetAttribute("BenchHalfStuds") or 168) :: number
local LOD = (workspace:GetAttribute("BenchLod") or 1) :: number
local EDITS = 10
local EDIT_RADIUS = 5 -- units
local DEPTH = 40 -- units below the lowest surface point (caves)

local result: { [string]: any } = { mode = MODE, halfStuds = HALF }
workspace:SetAttribute("BenchDone", false)

local function ms(t0: number): number
	return (os.clock() - t0) * 1000
end

local function median(list: { number }): number
	if #list == 0 then
		return 0
	end
	local c = table.clone(list)
	table.sort(c)
	return c[math.ceil(#c / 2)]
end

-- Region: x, z in [-HALF, HALF] studs, y from DEPTH below the lowest surface to above the highest.
local minH, maxH = math.huge, -math.huge
for x = -HALF, HALF, 8 do
	for z = -HALF, HALF, 8 do
		local h = Sdf.height(x / S, z / S)
		minH, maxH = math.min(minH, h), math.max(maxH, h)
	end
end
local Y0, Y1 = (minH - DEPTH) * S, (maxH + 4) * S
result.yRangeStuds = { Y0, Y1 }

-- Edit points: on the surface, spread around the spawn, all inside the region.
local editPoints: { Vector3 } = {}
for i = 1, EDITS do
	local a = i * 2.4
	local r = (HALF / S) * 0.5 * (i / EDITS)
	local x, z = math.cos(a) * r, math.sin(a) * r
	table.insert(editPoints, Vector3.new(x, Sdf.height(x, z), z)) -- units
end

local character = Players.LocalPlayer.Character or Players.LocalPlayer.CharacterAdded:Wait()
local root = character:WaitForChild("HumanoidRootPart") :: BasePart
root.Anchored = true

-------------------------------------------------------------------------------------------------
-- terrain
-------------------------------------------------------------------------------------------------
local VOX = 4
local terrain = workspace.Terrain

-- Samples [x0, x1) x [y0, y1) x [z0, z1) (studs, multiples of 4) and writes it. Returns ms spent
-- sampling and writing.
local function writeTerrainBox(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): (number, number)
	local nx, ny, nz = (x1 - x0) // VOX, (y1 - y0) // VOX, (z1 - z0) // VOX
	local t0 = os.clock()
	local mats = table.create(nx)
	local occs = table.create(nx)
	local unitsPerVox = VOX / S
	for i = 1, nx do
		local mi, oi = table.create(ny), table.create(ny)
		mats[i], occs[i] = mi, oi
		for j = 1, ny do
			mi[j], oi[j] = table.create(nz, Enum.Material.Air), table.create(nz, 0)
		end
		local wx = (x0 + (i - 0.5) * VOX) / S
		for k = 1, nz do
			local wz = (z0 + (k - 0.5) * VOX) / S
			local h = Sdf.height(wx, wz)
			for j = 1, ny do
				local wy = (y0 + (j - 0.5) * VOX) / S
				local d = Sdf.evaluateWithHeight(wx, wy, wz, h)
				local occ = 0.5 - d / unitsPerVox
				if occ > 0 then
					oi[j][k] = if occ > 1 then 1 else occ
					mi[j][k] = if wy > 335 then Enum.Material.Snow
						elseif h - wy < 2 then Enum.Material.Grass
						else Enum.Material.Rock
				end
			end
		end
	end
	local sampleMs = ms(t0)
	local t1 = os.clock()
	terrain:WriteVoxels(Region3.new(Vector3.new(x0, y0, z0), Vector3.new(x1, y1, z1)), VOX, mats, occs)
	return sampleMs, ms(t1)
end

local function runTerrain()
	terrain:Clear()
	local x0 = math.floor(-HALF / VOX) * VOX
	local x1 = math.ceil(HALF / VOX) * VOX
	local y0 = math.floor(Y0 / VOX) * VOX
	local y1 = math.ceil(Y1 / VOX) * VOX
	local SLAB = 64 -- studs per side per WriteVoxels call
	local sampleMs, writeMs, voxels = 0, 0, 0
	local wall = os.clock()
	for x = x0, x1 - 1, SLAB do
		for z = x0, x1 - 1, SLAB do
			local sx, sz = math.min(x + SLAB, x1), math.min(z + SLAB, x1)
			local a, b = writeTerrainBox(x, y0, z, sx, y1, sz)
			sampleMs += a
			writeMs += b
			voxels += ((sx - x) // VOX) * ((y1 - y0) // VOX) * ((sz - z) // VOX)
			task.wait()
		end
	end
	result.setup = { sampleMs = sampleMs, writeMs = writeMs, wallMs = ms(wall), voxels = voxels }

	return function(p: Vector3): { [string]: number }
		Sdf.addSubtractSphere(p.X, p.Y, p.Z, EDIT_RADIUS)
		local r = EDIT_RADIUS * S + VOX
		local c = p * S
		local bx0, by0, bz0 = math.floor((c.X - r) / VOX) * VOX, math.floor((c.Y - r) / VOX) * VOX, math.floor((c.Z - r) / VOX) * VOX
		local bx1, by1, bz1 = math.ceil((c.X + r) / VOX) * VOX, math.ceil((c.Y + r) / VOX) * VOX, math.ceil((c.Z + r) / VOX) * VOX
		local a, b = writeTerrainBox(bx0, by0, bz0, bx1, by1, bz1)
		-- Native reference, mirrored to the other side of the spawn so it hits untouched ground.
		local t = os.clock()
		terrain:FillBall(Vector3.new(-c.X, Sdf.height(-p.X, -p.Z) * S, -c.Z), EDIT_RADIUS * S, Enum.Material.Air)
		return { computeMs = a, applyMs = b, totalMs = a + b, fillBallMs = ms(t) }
	end
end

-------------------------------------------------------------------------------------------------
-- pack
-------------------------------------------------------------------------------------------------
local MAX_PAGES = 8
local PAGE_TRIS = 20000 - 2 -- two bound anchors
local PAGE_VERTS = 60000 - 6

type Page = { mesh: EditableMesh, part: MeshPart?, tris: number, verts: number }
type Chunk = {
	cx: number,
	cy: number,
	cz: number,
	minW: Vector3, -- units, skirt included
	maxW: Vector3,
	page: Page?,
	faces: { number },
	tris: number,
	verts: number,
}

local pages: { Page } = {}
local CENTER = Vector3.new(0, (Y0 + Y1) / 2, 0) -- studs: every page is placed here

local function newPage(): Page?
	if #pages >= MAX_PAGES then
		return nil
	end
	local mesh = AssetService:CreateEditableMesh()
	if not mesh then
		return nil
	end
	-- Two tiny triangles at the region corners: the page's bounds never change with edits, so the
	-- MeshPart's size/position stay valid while the geometry is rewritten.
	for _, corner in { Vector3.new(-HALF, Y0, -HALF), Vector3.new(HALF, Y1, HALF) } do
		local c = corner - CENTER
		local a = mesh:AddVertex(c)
		local b = mesh:AddVertex(c + Vector3.new(0.01, 0, 0))
		local d = mesh:AddVertex(c + Vector3.new(0, 0, 0.01))
		mesh:AddTriangle(a, b, d)
	end
	local page = { mesh = mesh, part = nil, tris = 0, verts = 0 }
	table.insert(pages, page)
	return page
end

local GRASS = Color3.fromRGB(86, 125, 70)
local ROCK = Color3.fromRGB(120, 112, 104)
local SNOW = Color3.fromRGB(235, 238, 242)

-- Appends a mesher result to a page with room. Returns false when all pages are full.
local function addChunk(chunk: Chunk, r: Mesher.MeshResult): boolean
	chunk.tris, chunk.verts, chunk.faces = r.triangleCount, r.vertexCount, {}
	if r.triangleCount == 0 then
		return true
	end
	local page: Page? = nil
	for _, p in pages do
		if p.tris + r.triangleCount <= PAGE_TRIS and p.verts + r.vertexCount <= PAGE_VERTS then
			page = p
			break
		end
	end
	page = page or newPage()
	if not page or page.tris + r.triangleCount > PAGE_TRIS then
		return false
	end
	local mesh = page.mesh
	local ox, oy, oz = ChunkMath.chunkToWorld(chunk.cx, chunk.cy, chunk.cz)
	local vIds, nIds, cIds = table.create(r.vertexCount), table.create(r.vertexCount), table.create(r.vertexCount)
	for i = 0, r.vertexCount - 1 do
		local x, y, z, nx, ny, nz = Mesher.readVertex(r, i)
		local wy = y + oy
		vIds[i + 1] = mesh:AddVertex(Vector3.new((x + ox) * S - CENTER.X, wy * S - CENTER.Y, (z + oz) * S - CENTER.Z))
		nIds[i + 1] = mesh:AddNormal(Vector3.new(nx, ny, nz))
		cIds[i + 1] = mesh:AddColor(if wy > 335 and ny > 0.6 then SNOW elseif ny > 0.75 then GRASS else ROCK, 1)
	end
	local idx = r.indices
	for t = 0, r.triangleCount - 1 do
		local o = t * 12
		-- Godot is clockwise-front, Roblox counter-clockwise: swap b and c.
		local a, c, b = buffer.readu32(idx, o) + 1, buffer.readu32(idx, o + 4) + 1, buffer.readu32(idx, o + 8) + 1
		local f = mesh:AddTriangle(vIds[a], vIds[b], vIds[c])
		mesh:SetFaceNormals(f, { nIds[a], nIds[b], nIds[c] })
		mesh:SetFaceColors(f, { cIds[a], cIds[b], cIds[c] })
		table.insert(chunk.faces, f)
	end
	page.tris += r.triangleCount
	page.verts += r.vertexCount
	chunk.page = page
	return true
end

local function removeChunk(chunk: Chunk)
	local page = chunk.page
	if not page then
		return
	end
	for _, f in chunk.faces do
		page.mesh:RemoveFace(f)
	end
	page.tris -= chunk.tris
	page.verts -= chunk.verts
	chunk.page, chunk.faces, chunk.tris, chunk.verts = nil, {}, 0, 0
end

local function runPack()
	local mesher = Mesher.new()
	local step = bit32.lshift(2, LOD)
	local off = bit32.lshift(1, LOD)
	local h = ChunkMath.CHUNK_WORLD_DIM / 2
	local size = ChunkMath.worldChunkSize(off)
	local skirt = ChunkMath.SKIRT_SIZE * ChunkMath.VOXEL_SIZE * bit32.lshift(1, LOD)
	local halfU = HALF / S

	-- Every chunk touching the region, nearest first (so a full budget keeps the middle).
	local chunks: { Chunk } = {}
	local lo = off - math.ceil(halfU / h / step + 1) * step
	local ylo, yhi = math.floor(Y0 / S / h), math.ceil(Y1 / S / h)
	for cx = lo, -lo, step do
		for cz = lo, -lo, step do
			for cy = ylo - (ylo - off) % step, yhi, step do
				local wx, wy, wz = ChunkMath.chunkToWorld(cx, cy, cz)
				local minW = Vector3.new(wx - size / 2 - skirt, wy - size / 2 - skirt, wz - size / 2 - skirt)
				local maxW = Vector3.new(wx + size / 2 + skirt, wy + size / 2 + skirt, wz + size / 2 + skirt)
				if maxW.X > -halfU and minW.X < halfU and maxW.Z > -halfU and minW.Z < halfU and maxW.Y > Y0 / S and minW.Y < Y1 / S then
					table.insert(chunks, { cx = cx, cy = cy, cz = cz, minW = minW, maxW = maxW, page = nil, faces = {}, tris = 0, verts = 0 })
				end
			end
		end
	end
	table.sort(chunks, function(a, b)
		return a.cx * a.cx + a.cz * a.cz < b.cx * b.cx + b.cz * b.cz
	end)

	local meshMs, apiMs, overflow, nonEmpty, tris = 0, 0, 0, 0, 0
	local wall = os.clock()
	local lastYield = os.clock()
	for _, chunk in chunks do
		local t0 = os.clock()
		local r = mesher:mesh(chunk.cx, chunk.cy, chunk.cz)
		meshMs += ms(t0)
		local t1 = os.clock()
		if not addChunk(chunk, r) then
			overflow += 1
		elseif r.triangleCount > 0 then
			nonEmpty += 1
			tris += r.triangleCount
		end
		apiMs += ms(t1)
		if os.clock() - lastYield > 0.1 then
			task.wait()
			lastYield = os.clock()
		end
	end
	local t2 = os.clock()
	for _, page in pages do
		local part = AssetService:CreateMeshPartAsync(Content.fromObject(page.mesh), { CollisionFidelity = Enum.CollisionFidelity.Default })
		part.Anchored = true
		part.Material = Enum.Material.SmoothPlastic
		part.Color = Color3.new(1, 1, 1)
		part.CFrame = CFrame.new(CENTER)
		part.Parent = workspace
		page.part = part
	end
	result.setup = {
		meshMs = meshMs,
		apiMs = apiMs,
		partsMs = ms(t2),
		wallMs = ms(wall),
		chunks = #chunks,
		nonEmpty = nonEmpty,
		overflowChunks = overflow,
		triangles = tris,
		pages = #pages,
	}

	return function(p: Vector3): { [string]: number }
		Sdf.addSubtractSphere(p.X, p.Y, p.Z, EDIT_RADIUS)
		local reach = EDIT_RADIUS + ChunkMath.VOXEL_SIZE * bit32.lshift(1, LOD)
		local tm, ta = 0, 0
		local touched: { [Page]: boolean } = {}
		local n = 0
		for _, chunk in chunks do
			local mn, mx = chunk.minW, chunk.maxW
			if p.X + reach > mn.X and p.X - reach < mx.X and p.Y + reach > mn.Y and p.Y - reach < mx.Y and p.Z + reach > mn.Z and p.Z - reach < mx.Z then
				n += 1
				local t0 = os.clock()
				local r = mesher:mesh(chunk.cx, chunk.cy, chunk.cz)
				tm += ms(t0)
				local t1 = os.clock()
				local old = chunk.page
				removeChunk(chunk)
				addChunk(chunk, r)
				ta += ms(t1)
				if old then
					touched[old] = true
				end
				if chunk.page then
					touched[chunk.page :: Page] = true
				end
			end
		end
		-- Visuals are live; collision is not. Refreshing it means cooking a new part per page.
		local t2 = os.clock()
		for page in touched do
			local cooked = AssetService:CreateMeshPartAsync(Content.fromObject(page.mesh), { CollisionFidelity = Enum.CollisionFidelity.Default })
			if page.part then
				page.part:ApplyMesh(cooked)
			end
			cooked:Destroy()
		end
		return { computeMs = tm, applyMs = ta, totalMs = tm + ta, collisionMs = ms(t2), chunks = n }
	end
end

-------------------------------------------------------------------------------------------------
-- run
-------------------------------------------------------------------------------------------------
local function memory(): { [string]: number }
	local out: { [string]: number } = { total = Stats:GetTotalMemoryUsageMb() }
	for _, tag in { "TerrainVoxels", "GraphicsTerrain", "GraphicsMeshParts", "Geometry", "PhysicsCollision", "GraphicsParts" } do
		local ok, v = pcall(function()
			return Stats:GetMemoryUsageMbForTag((Enum.DeveloperMemoryTag :: any)[tag])
		end)
		if ok then
			out[tag] = v
		end
	end
	return out
end

local function frames(seconds: number): { [string]: number }
	local dts: { number } = {}
	local conn = RunService.RenderStepped:Connect(function(dt)
		table.insert(dts, dt * 1000)
	end)
	task.wait(seconds)
	conn:Disconnect()
	table.sort(dts)
	return { median = median(dts), p95 = dts[math.max(1, math.ceil(#dts * 0.95))] or 0, frames = #dts }
end

task.wait(2)
local memBefore = memory()
local edit = if MODE == "terrain" then runTerrain() else runPack()
task.wait(1)
result.memoryBeforeMb = memBefore
result.memoryAfterMb = memory()
result.framesMs = frames(5)

local edits: { { [string]: number } } = {}
for _, p in editPoints do
	table.insert(edits, edit(p))
	task.wait()
end
local summary: { [string]: number } = {}
for key in edits[1] do
	local list = {}
	for _, e in edits do
		table.insert(list, e[key])
	end
	summary[key] = median(list)
end
result.editMedian = summary
result.editsMs = edits
result.framesAfterEditsMs = frames(3)

print("[Bench] " .. HttpService:JSONEncode(result))
workspace:SetAttribute("BenchDone", true)
