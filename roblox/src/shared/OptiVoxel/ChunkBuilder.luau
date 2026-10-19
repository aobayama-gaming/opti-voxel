--!strict
-- Roblox only: turns a Mesher result into MeshParts through EditableMesh. Must run in the serial
-- phase (AssetService calls). Chunks over Config.MAX_TRIANGLES_PER_MESH are split.

local AssetService = game:GetService("AssetService")

local ChunkMath = require(script.Parent.ChunkMath)
local Config = require(script.Parent.Config)
local Mesher = require(script.Parent.Mesher)

local readu32 = buffer.readu32
local readVertex = Mesher.readVertex

export type Built = {
	parts: { MeshPart },
	meshes: { EditableMesh },
	meshTriangles: number, -- triangles held by `meshes`
	budgetExhausted: boolean,
	failures: number,
	lastError: string?,
}

local GRASS = Color3.fromRGB(86, 125, 70)
local ROCK = Color3.fromRGB(120, 112, 104)
local SNOW = Color3.fromRGB(235, 238, 242)

local function colorFor(worldY: number, ny: number): Color3
	if worldY > 335 and ny > 0.6 then
		return SNOW
	end
	return if ny > 0.75 then GRASS else ROCK
end

local ChunkBuilder = {}

-- EditableMeshes alive in this Lua VM (one per worker Actor). The client has a strict memory budget
-- for them, and CreateEditableMesh returns nil once it is spent.
ChunkBuilder.aliveMeshes = 0
ChunkBuilder.aliveTriangles = 0

function ChunkBuilder.destroy(built: Built)
	for _, part in built.parts do
		part:Destroy()
	end
	for _, mesh in built.meshes do
		mesh:Destroy()
	end
	ChunkBuilder.aliveMeshes -= #built.meshes
	ChunkBuilder.aliveTriangles -= built.meshTriangles
	built.meshTriangles = 0
	table.clear(built.parts)
	table.clear(built.meshes)
end

function ChunkBuilder.build(result: Mesher.MeshResult, cx: number, cy: number, cz: number, parent: Instance): Built
	local built: Built = { parts = {}, meshes = {}, meshTriangles = 0, budgetExhausted = false, failures = 0, lastError = nil }
	local triCount = result.triangleCount
	if triCount == 0 then
		return built
	end

	local S = Config.STUDS_PER_UNIT
	local lod = ChunkMath.getLod(cx)
	local collide = lod <= Config.COLLISION_MAX_LOD
	local centerX, centerY, centerZ = ChunkMath.chunkToWorld(cx, cy, cz)
	local indices = result.indices
	local maxTris = Config.MAX_TRIANGLES_PER_MESH

	for first = 0, triCount - 1, maxTris do
		local last = math.min(first + maxTris, triCount) - 1

		-- Center each piece on its own bounds, so the result is right whichever origin Roblox
		-- gives the generated MeshPart.
		local minX, minY, minZ = math.huge, math.huge, math.huge
		local maxX, maxY, maxZ = -math.huge, -math.huge, -math.huge
		for t = first, last do
			for k = 0, 2 do
				local x, y, z = readVertex(result, readu32(indices, (t * 3 + k) * 4))
				minX, minY, minZ = math.min(minX, x), math.min(minY, y), math.min(minZ, z)
				maxX, maxY, maxZ = math.max(maxX, x), math.max(maxY, y), math.max(maxZ, z)
			end
		end
		local midX, midY, midZ = (minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5

		local ok, err = pcall(function()
			local mesh = AssetService:CreateEditableMesh()
			if not mesh then
				built.budgetExhausted = true
				error(`EditableMesh memory budget exhausted ({ChunkBuilder.aliveMeshes} meshes / {ChunkBuilder.aliveTriangles} triangles alive in this worker)`, 0)
			end
			table.insert(built.meshes, mesh)
			ChunkBuilder.aliveMeshes += 1
			ChunkBuilder.aliveTriangles += last - first + 1
			built.meshTriangles += last - first + 1

			local vertexIds: { [number]: number } = {}
			local normalIds: { [number]: number } = {}
			local colorIds: { [number]: number } = {}
			local function vertex(i: number): (number, number, number?)
				local cached = vertexIds[i]
				if cached then
					return cached, normalIds[i], colorIds[i]
				end
				local x, y, z, nx, ny, nz = readVertex(result, i)
				local id = mesh:AddVertex(Vector3.new((x - midX) * S, (y - midY) * S, (z - midZ) * S))
				local n = mesh:AddNormal(Vector3.new(nx, ny, nz))
				vertexIds[i] = id
				normalIds[i] = n
				if Config.VERTEX_COLORS then
					local c = mesh:AddColor(colorFor(y + centerY, ny), 1)
					colorIds[i] = c
					return id, n, c
				end
				return id, n, nil
			end

			for t = first, last do
				local o = t * 12
				local a, b, c = readu32(indices, o), readu32(indices, o + 4), readu32(indices, o + 8)
				if Config.FLIP_WINDING then
					b, c = c, b
				end
				local va, na, ca = vertex(a)
				local vb, nb, cb = vertex(b)
				local vc, nc, cc = vertex(c)
				local face = mesh:AddTriangle(va, vb, vc)
				mesh:SetFaceNormals(face, { na, nb, nc })
				if ca and cb and cc then
					mesh:SetFaceColors(face, { ca, cb, cc })
				end
			end

			local part = AssetService:CreateMeshPartAsync(Content.fromObject(mesh), {
				CollisionFidelity = if collide then Enum.CollisionFidelity.Default else Enum.CollisionFidelity.Box,
			})
			part.Name = `{cx},{cy},{cz}#{first // maxTris}`
			part.Anchored = true
			part.CanCollide = collide
			part.CanQuery = collide
			part.CanTouch = false
			part.Material = Enum.Material.SmoothPlastic
			part.Color = if Config.VERTEX_COLORS then Color3.new(1, 1, 1) else GRASS
			part.CFrame = CFrame.new((centerX + midX) * S, (centerY + midY) * S, (centerZ + midZ) * S)
			part.Parent = parent
			table.insert(built.parts, part)

			if Config.DESTROY_EDITABLE_AFTER_PART then
				mesh:Destroy()
				table.remove(built.meshes)
				ChunkBuilder.aliveMeshes -= 1
				ChunkBuilder.aliveTriangles -= last - first + 1
				built.meshTriangles -= last - first + 1
			end
		end)
		if not ok then
			built.failures += 1
			built.lastError = tostring(err)
		end
	end
	return built
end

return ChunkBuilder
