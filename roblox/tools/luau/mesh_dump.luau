--!strict
-- Meshes one chunk with the Luau port and prints it in the C++ oracle's format.
--   luau --codegen mesh_dump.luau -a cx cy cz
local Mesher = require("../../src/shared/OptiVoxel/Mesher")

local args = { ... }
local cx, cy, cz = tonumber(args[1]) or 1, tonumber(args[2]) or 43, tonumber(args[3]) or 1

local result = Mesher.new():mesh(cx, cy, cz)
local out = table.create(result.vertexCount + result.triangleCount + 2)
table.insert(out, `V {result.vertexCount}`)
for i = 0, result.vertexCount - 1 do
	local x, y, z, nx, ny, nz = Mesher.readVertex(result, i)
	table.insert(out, string.format("%.9g %.9g %.9g %.9g %.9g %.9g", x, y, z, nx, ny, nz))
end
table.insert(out, `T {result.triangleCount}`)
for t = 0, result.triangleCount - 1 do
	local o = t * 12
	table.insert(
		out,
		`{buffer.readu32(result.indices, o)} {buffer.readu32(result.indices, o + 4)} {buffer.readu32(result.indices, o + 8)}`
	)
end
print(table.concat(out, "\n"))
