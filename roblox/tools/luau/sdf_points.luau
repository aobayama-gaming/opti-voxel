--!strict
-- Prints "x y z value" for a fixed pseudo-random point set around the terrain surface.
-- Used by tools/compare.py against the C++ oracle.
local Sdf = require("../../src/shared/OptiVoxel/Sdf")

local seed = 12345
local function rnd(): number
	seed = (seed * 1103515245 + 12345) % 2147483648
	return seed / 2147483648
end

for _ = 1, 20000 do
	local x = (rnd() - 0.5) * 4000
	local y = 150 + rnd() * 250
	local z = (rnd() - 0.5) * 4000
	print(string.format("%.17g %.17g %.17g %.17g", x, y, z, Sdf.evaluate(x, y, z)))
end
