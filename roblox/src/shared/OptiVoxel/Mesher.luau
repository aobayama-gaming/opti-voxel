--!strict
--!native
-- Surface nets / dual contouring for one chunk: a Luau port of CPUMesher (src/CPU_mesher.cpp).
--
-- Same math as the C++ passes (sample grid with skirt offset, 5-step binary search on crossing
-- edges, QEF with Tikhonov regularization, same quad winding), written as straightforward loops
-- instead of the row-trimmed multi-pass layout. tools/compare.py checks it against the real C++.
--
-- Output positions are relative to ChunkMath.chunkToWorld(chunk), in Godot world units.

local ChunkMath = if script then require(script.Parent.ChunkMath) else require("./ChunkMath")
local Sdf = if script then require(script.Parent.Sdf) else require("./Sdf")

local evaluate = Sdf.evaluate
local evaluateWithHeight = Sdf.evaluateWithHeight
local height = Sdf.height
local sdfNormal = Sdf.normal

local readf64, writef64 = buffer.readf64, buffer.writef64
local readi32, writei32 = buffer.readi32, buffer.writei32
local readf32, writef32 = buffer.readf32, buffer.writef32

local N = ChunkMath.CHUNK_SIZE -- cells per axis
local S = N + 1 -- samples per axis
local S2 = S * S
local S3 = S2 * S
local SKIRT = ChunkMath.SKIRT_SIZE
local VOXEL_SIZE = ChunkMath.VOXEL_SIZE
local USE_SKIRT = ChunkMath.USE_SKIRT
local BINARY_SEARCH_STEP = 5
local EDGE_NORMAL_ALPHA = 9.99999974737875e-05 -- evaluate_normal default alpha (1e-4f)
local QEF_ALPHA = 0.05000000074505806 -- fourth_pass alpha (0.05f)
local SKIRT_HASH_SCALE = 0.029999999329447746 -- 0.03f
local MICRO_SLOPE = 0.05000000074505806 -- 0.05f

export type MeshResult = {
	vertexCount: number,
	triangleCount: number,
	positions: buffer, -- f32 xyz per vertex, relative to the chunk center
	normals: buffer, -- f32 xyz per vertex
	indices: buffer, -- u32 per corner, 3 per triangle
}

local Mesher = {}
Mesher.__index = Mesher

export type Mesher = typeof(setmetatable(
	{} :: {
		values: buffer,
		edgeSlot: buffer,
		edgeData: buffer,
		cellVertex: buffer,
		vertexData: buffer,
		edgeCapacity: number,
		vertexCapacity: number,
		samplesEvaluated: number,
	},
	Mesher
))

function Mesher.new(): Mesher
	local self = setmetatable({
		values = buffer.create(S3 * 8), -- f64 SDF value per sample
		edgeSlot = buffer.create(3 * S3 * 4), -- i32 per (axis, start sample): slot in edgeData or -1
		edgeData = buffer.create(0), -- per crossing edge: local pos xyz, normal xyz (f64)
		cellVertex = buffer.create(N * N * N * 4), -- i32 per cell: vertex index or -1
		vertexData = buffer.create(0), -- per vertex: local pos xyz, normal xyz (f64)
		edgeCapacity = 0,
		vertexCapacity = 0,
		samplesEvaluated = 0,
	}, Mesher)
	return self
end

-- FNV-1a + murmur finalizer on the chunk id, as in CPUMesher::skirt_offset (uint32 arithmetic).
local function mul32(a: number, b: number): number
	local lo = a * bit32.band(b, 0xFFFF)
	local hi = (a * bit32.rshift(b, 16)) % 0x10000 * 0x10000
	return (lo + hi) % 0x100000000
end

local function chunkHash(cx: number, cy: number, cz: number): number
	local seed = 2166136261
	seed = mul32(bit32.bxor(seed, cx), 16777619)
	seed = mul32(bit32.bxor(seed, cy), 16777619)
	seed = mul32(bit32.bxor(seed, cz), 16777619)
	seed = bit32.bxor(seed, bit32.rshift(seed, 16))
	seed = mul32(seed, 0x7feb352d)
	seed = bit32.bxor(seed, bit32.rshift(seed, 15))
	seed = mul32(seed, 0x846ca68b)
	seed = bit32.bxor(seed, bit32.rshift(seed, 16))
	local unit = bit32.band(seed, 0x00ffffff) / 16777215
	return unit * 2 - 1
end

-- Pushes the SDF outward near the chunk faces so neighbouring LODs overlap instead of cracking.
local function skirtAxis(coord: number, size: number): number
	local d = math.min(coord, N - coord)
	local offset = 0
	local skirtSize = SKIRT * 0.75
	local slopeLimit = SKIRT * 2
	if d < skirtSize then
		offset = 1 - d / skirtSize
	end
	if d < slopeLimit then
		offset += MICRO_SLOPE * (1 - d / slopeLimit)
	end
	return offset * size
end

-- Solves A x = b (3x3, partial pivoting). Returns nil when singular, like solve_linear_system_3x3.
local function solve3(
	a00: number, a01: number, a02: number,
	a10: number, a11: number, a12: number,
	a20: number, a21: number, a22: number,
	b0: number, b1: number, b2: number
): (number?, number, number)
	local eps = 1e-10
	local m = {
		{ a00, a01, a02, b0 },
		{ a10, a11, a12, b1 },
		{ a20, a21, a22, b2 },
	}
	for col = 1, 3 do
		local best, bestAbs = col, math.abs(m[col][col])
		for row = col + 1, 3 do
			local v = math.abs(m[row][col])
			if v > bestAbs then
				best, bestAbs = row, v
			end
		end
		if bestAbs < eps then
			return nil, 0, 0
		end
		if best ~= col then
			m[col], m[best] = m[best], m[col]
		end
		local pivotRow = m[col]
		for row = col + 1, 3 do
			local r = m[row]
			local factor = r[col] / pivotRow[col]
			if math.abs(factor) >= eps then
				for c = col, 4 do
					r[c] -= factor * pivotRow[c]
				end
			end
		end
	end
	local x2 = m[3][4] / m[3][3]
	if math.abs(m[3][3]) < eps then
		return nil, 0, 0
	end
	local x1 = (m[2][4] - m[2][3] * x2) / m[2][2]
	local x0 = (m[1][4] - m[1][2] * x1 - m[1][3] * x2) / m[1][1]
	return x0, x1, x2
end

local function ensureCapacity(buf: buffer, needed: number): buffer
	if buffer.len(buf) >= needed then
		return buf
	end
	local grown = buffer.create(math.max(needed, buffer.len(buf) * 2))
	buffer.copy(grown, 0, buf, 0, buffer.len(buf))
	return grown
end

function Mesher.mesh(self: Mesher, cx: number, cy: number, cz: number): MeshResult
	local lodSize = ChunkMath.lodSize(cx)
	local cell = VOXEL_SIZE * lodSize
	local centerX, centerY, centerZ = ChunkMath.chunkToWorld(cx, cy, cz)
	local half = ChunkMath.worldChunkSize(cx) * 0.5
	-- world(l) = minCorner + (l - SKIRT) * cell
	local originX = centerX - half - SKIRT * cell
	local originY = centerY - half - SKIRT * cell
	local originZ = centerZ - half - SKIRT * cell

	local skirtScale = lodSize * VOXEL_SIZE * 0.5
	local hashOffset = if USE_SKIRT then chunkHash(cx, cy, cz) * skirtScale * SKIRT_HASH_SCALE else 0

	local function skirt(lx: number, ly: number, lz: number, value: number): number
		if not USE_SKIRT then
			return value
		end
		local avg = (skirtAxis(lx, skirtScale) + skirtAxis(ly, skirtScale) + skirtAxis(lz, skirtScale)) / 3
		return value - avg + hashOffset
	end

	-- 1. Sample grid (the 2D height noise is shared by a whole column).
	local values = self.values
	local anyInside, anyOutside = false, false
	for z = 0, N do
		local wz = originZ + z * cell
		for x = 0, N do
			local wx = originX + x * cell
			local h = height(wx, wz)
			local idx = x + S2 * z
			for y = 0, N do
				local v = skirt(x, y, z, evaluateWithHeight(wx, originY + y * cell, wz, h))
				writef64(values, (idx + S * y) * 8, v)
				if v < 0 then
					anyInside = true
				else
					anyOutside = true
				end
			end
		end
	end
	self.samplesEvaluated = S3

	local empty: MeshResult = {
		vertexCount = 0,
		triangleCount = 0,
		positions = buffer.create(0),
		normals = buffer.create(0),
		indices = buffer.create(0),
	}
	if not (anyInside and anyOutside) then
		return empty
	end

	-- 2. Crossing edges: binary search for the surface, then the SDF normal there.
	local edgeSlot = self.edgeSlot
	buffer.fill(edgeSlot, 0, 0xFF) -- all -1
	local edgeCount = 0
	local edgeData = self.edgeData

	for axis = 0, 2 do
		local dx = if axis == 0 then 1 else 0
		local dy = if axis == 1 then 1 else 0
		local dz = if axis == 2 then 1 else 0
		local stride = if axis == 0 then 1 elseif axis == 1 then S else S2
		local axisBase = axis * S3
		for z = 0, N - dz do
			for y = 0, N - dy do
				local rowIdx = S * y + S2 * z
				for x = 0, N - dx do
					local i0 = rowIdx + x
					local startInside = readf64(values, i0 * 8) < 0
					local endInside = readf64(values, (i0 + stride) * 8) < 0
					if startInside ~= endInside then
						-- Same search as _find_edge_intersection, in local (sample) coordinates.
						local lx, ly, lz = x, y, z
						local rx, ry, rz = x + dx, y + dy, z + dz
						local mx, my, mz = (lx + rx) * 0.5, (ly + ry) * 0.5, (lz + rz) * 0.5
						for _ = 1, BINARY_SEARCH_STEP do
							local v = skirt(mx, my, mz, evaluate(originX + mx * cell, originY + my * cell, originZ + mz * cell))
							if (v < 0) == startInside then
								lx, ly, lz = mx, my, mz
							else
								rx, ry, rz = mx, my, mz
							end
							mx, my, mz = (lx + rx) * 0.5, (ly + ry) * 0.5, (lz + rz) * 0.5
						end
						local nx, ny, nz =
							sdfNormal(originX + mx * cell, originY + my * cell, originZ + mz * cell, EDGE_NORMAL_ALPHA)

						edgeData = ensureCapacity(edgeData, (edgeCount + 1) * 48)
						local o = edgeCount * 48
						writef64(edgeData, o, mx)
						writef64(edgeData, o + 8, my)
						writef64(edgeData, o + 16, mz)
						writef64(edgeData, o + 24, nx)
						writef64(edgeData, o + 32, ny)
						writef64(edgeData, o + 40, nz)
						writei32(edgeSlot, (axisBase + i0) * 4, edgeCount)
						edgeCount += 1
					end
				end
			end
		end
	end
	self.edgeData = edgeData

	-- 3. One vertex per cell touching a crossing edge, placed by the regularized QEF.
	local cellVertex = self.cellVertex
	buffer.fill(cellVertex, 0, 0xFF)
	local vertexData = self.vertexData
	local vertexCount = 0
	local qefAlpha = QEF_ALPHA * lodSize

	-- (axis, dx, dy, dz) of the 12 cell edges, relative to the cell's min sample.
	local CELL_EDGES = {
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
		1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1,
		2, 0, 0, 0, 2, 1, 0, 0, 2, 0, 1, 0, 2, 1, 1, 0,
	}

	for z = 0, N - 1 do
		for y = 0, N - 1 do
			for x = 0, N - 1 do
				local a00, a01, a02, a11, a12, a22 = 0, 0, 0, 0, 0, 0
				local b0, b1, b2 = 0, 0, 0
				local px, py, pz = 0, 0, 0
				local count = 0
				local midX, midY, midZ = x + 0.5, y + 0.5, z + 0.5
				for e = 1, 48, 4 do
					local axis = CELL_EDGES[e]
					local i0 = (x + CELL_EDGES[e + 1]) + S * (y + CELL_EDGES[e + 2]) + S2 * (z + CELL_EDGES[e + 3])
					local slot = readi32(edgeSlot, (axis * S3 + i0) * 4)
					if slot >= 0 then
						local o = slot * 48
						-- Position relative to the cell center, in cell units (as in fourth_pass).
						local qx = readf64(edgeData, o) - midX
						local qy = readf64(edgeData, o + 8) - midY
						local qz = readf64(edgeData, o + 16) - midZ
						local nx = readf64(edgeData, o + 24)
						local ny = readf64(edgeData, o + 32)
						local nz = readf64(edgeData, o + 40)
						a00 += nx * nx
						a01 += nx * ny
						a02 += nx * nz
						a11 += ny * ny
						a12 += ny * nz
						a22 += nz * nz
						local d = nx * qx + ny * qy + nz * qz
						b0 += nx * d
						b1 += ny * d
						b2 += nz * d
						px += qx
						py += qy
						pz += qz
						count += 1
					end
				end

				if count > 0 then
					px, py, pz = px / count, py / count, pz / count
					local sx, sy, sz = px, py, pz
					if qefAlpha < 1 then
						local rx, ry, rz = solve3(
							a00 + qefAlpha, a01, a02,
							a01, a11 + qefAlpha, a12,
							a02, a12, a22 + qefAlpha,
							b0 + qefAlpha * px, b1 + qefAlpha * py, b2 + qefAlpha * pz
						)
						if rx then
							sx, sy, sz = rx, ry, rz
						end
					end
					local wx = originX + (midX + sx) * cell
					local wy = originY + (midY + sy) * cell
					local wz = originZ + (midZ + sz) * cell
					local nx, ny, nz = sdfNormal(wx, wy, wz, cell)

					vertexData = ensureCapacity(vertexData, (vertexCount + 1) * 48)
					local o = vertexCount * 48
					writef64(vertexData, o, wx - centerX)
					writef64(vertexData, o + 8, wy - centerY)
					writef64(vertexData, o + 16, wz - centerZ)
					writef64(vertexData, o + 24, nx)
					writef64(vertexData, o + 32, ny)
					writef64(vertexData, o + 40, nz)
					writei32(cellVertex, (x + N * (y + N * z)) * 4, vertexCount)
					vertexCount += 1
				end
			end
		end
	end
	self.vertexData = vertexData

	-- 4. One quad per crossing edge whose 4 surrounding cells are all inside the chunk.
	local indices = buffer.create(edgeCount * 6 * 4)
	local triCount = 0

	local function cellAt(x: number, y: number, z: number): number
		return readi32(cellVertex, (x + N * (y + N * z)) * 4)
	end

	-- write_quad from fifth_pass: "flip" picks the (A, D, C) (A, C, B) winding.
	local function quad(flip: boolean, a: number, b: number, c: number, d: number)
		local o = triCount * 12
		if flip then
			b, d = d, b
		end
		buffer.writeu32(indices, o, a)
		buffer.writeu32(indices, o + 4, b)
		buffer.writeu32(indices, o + 8, c)
		buffer.writeu32(indices, o + 12, a)
		buffer.writeu32(indices, o + 16, c)
		buffer.writeu32(indices, o + 20, d)
		triCount += 2
	end

	for axis = 0, 2 do
		local axisBase = axis * S3
		for z = 0, N - 1 do
			for y = 0, N - 1 do
				for x = 0, N - 1 do
					local i0 = x + S * y + S2 * z
					if readi32(edgeSlot, (axisBase + i0) * 4) >= 0 then
						local startInside = readf64(values, i0 * 8) < 0
						if axis == 0 then
							if y >= 1 and z >= 1 then
								quad(
									startInside,
									cellAt(x, y - 1, z - 1),
									cellAt(x, y, z - 1),
									cellAt(x, y, z),
									cellAt(x, y - 1, z)
								)
							end
						elseif axis == 1 then
							if x >= 1 and z >= 1 then
								quad(
									not startInside,
									cellAt(x - 1, y, z - 1),
									cellAt(x, y, z - 1),
									cellAt(x, y, z),
									cellAt(x - 1, y, z)
								)
							end
						else
							if x >= 1 and y >= 1 then
								quad(
									startInside,
									cellAt(x - 1, y - 1, z),
									cellAt(x, y - 1, z),
									cellAt(x, y, z),
									cellAt(x - 1, y, z)
								)
							end
						end
					end
				end
			end
		end
	end

	-- Pack to f32 for the consumer (EditableMesh / dumps).
	local positions = buffer.create(vertexCount * 12)
	local normals = buffer.create(vertexCount * 12)
	for i = 0, vertexCount - 1 do
		local o = i * 48
		local p = i * 12
		writef32(positions, p, readf64(vertexData, o))
		writef32(positions, p + 4, readf64(vertexData, o + 8))
		writef32(positions, p + 8, readf64(vertexData, o + 16))
		writef32(normals, p, readf64(vertexData, o + 24))
		writef32(normals, p + 4, readf64(vertexData, o + 32))
		writef32(normals, p + 8, readf64(vertexData, o + 40))
	end

	return {
		vertexCount = vertexCount,
		triangleCount = triCount,
		positions = positions,
		normals = normals,
		indices = indices,
	}
end

-- Reads a vertex back (helpers for consumers that do not want to touch buffers).
function Mesher.readVertex(result: MeshResult, i: number): (number, number, number, number, number, number)
	local p = i * 12
	return readf32(result.positions, p),
		readf32(result.positions, p + 4),
		readf32(result.positions, p + 8),
		readf32(result.normals, p),
		readf32(result.normals, p + 4),
		readf32(result.normals, p + 8)
end

return Mesher
