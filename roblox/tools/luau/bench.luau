--!strict
-- Times the Luau mesher on the chunks a player standing at the origin would load first:
-- an 8x8 patch of LOD 0 columns, each meshed at every height that intersects the surface.
--   luau --codegen bench.luau
local ChunkMath = require("../../src/shared/OptiVoxel/ChunkMath")
local Mesher = require("../../src/shared/OptiVoxel/Mesher")
local Sdf = require("../../src/shared/OptiVoxel/Sdf")

-- SDF cost alone.
local t0 = os.clock()
local acc = 0
for i = 1, 100000 do
	acc += Sdf.evaluate(i * 0.37, 290 + (i % 40), i * 0.11)
end
local sdfUs = (os.clock() - t0) / 100000 * 1e6

local mesher = Mesher.new()
local meshed, empty, verts, tris = 0, 0, 0, 0
local times = {}
local t1 = os.clock()
for cx = -7, 7, 2 do
	for cz = -7, 7, 2 do
		for cy = 33, 49, 2 do
			local ts = os.clock()
			local r = mesher:mesh(cx, cy, cz)
			local dt = os.clock() - ts
			if r.triangleCount > 0 then
				meshed += 1
				verts += r.vertexCount
				tris += r.triangleCount
				table.insert(times, dt)
			else
				empty += 1
			end
		end
	end
end
local total = os.clock() - t1
table.sort(times)
print(string.format("sdf evaluate: %.2f us/call (checksum %.3f)", sdfUs, acc))
print(string.format("chunks: %d with geometry, %d empty, %.0f ms total", meshed, empty, total * 1000))
print(string.format(
	"per non-empty chunk: median %.1f ms, p90 %.1f ms, max %.1f ms; avg %d verts / %d tris",
	times[math.ceil(#times / 2)] * 1000,
	times[math.ceil(#times * 0.9)] * 1000,
	times[#times] * 1000,
	verts // meshed,
	tris // meshed
))
print(string.format("chunk is %.0f x %.0f units of terrain at LOD 0", ChunkMath.CHUNK_WORLD_DIM, ChunkMath.CHUNK_WORLD_DIM))
