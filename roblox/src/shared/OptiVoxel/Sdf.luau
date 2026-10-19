--!strict
--!native
-- Port of SDFDummy::evaluate (src/sdf_dummy.hpp): noise mountain carved by sparse cave blobs.
-- Units are the Godot world units (meters); the Roblox side scales to studs at the very end.

local Noise = if script then require(script.Parent.Noise) else require("./Noise")

local fractal = Noise.fractal
local max = math.max

local Sdf = {}

-- Max of |fractal(3 octaves)| is below 1.75 (snoise is in [-1, 1]); with that bound,
-- -cave_sdf <= 0.8 * 1.75 - 0.66 + depth / 400 < 0.75 whenever terrain_sdf > 0.75.
-- Above that the cave term can never win the max(): skip the 3D noise (exact, not an approximation).
local CAVE_SKIP_ABOVE = 1.0

-- Frequencies are the float32 values of the C++ literals (0.0035f, 0.008f), see Noise.luau.
local MOUNTAIN_FREQ = 0.0035000001080334187
local CAVE_FREQ = 0.00800000037997961

function Sdf.height(x: number, z: number): number
	return 300 + fractal(5, MOUNTAIN_FREQ, 1, 2, 0.5, x, 0, z) * 50
end

-- Same as evaluate() with the column height already known (the sample grid reuses it per column).
function Sdf.evaluateWithHeight(x: number, y: number, z: number, mountain_height: number): number
	local terrain_sdf = y - mountain_height
	if terrain_sdf > CAVE_SKIP_ABOVE then
		return terrain_sdf
	end

	local cave_blob = fractal(3, CAVE_FREQ, 1, 2, 0.5, x, y, z)
	local depth = mountain_height - y
	local blob_mask = (cave_blob + 0.3) * 0.5
	local depth_fade = depth / 40
	local cave_threshold = 0.5 + blob_mask * 0.4 + (1 - depth_fade) * 0.1
	local cave_sdf = cave_threshold - cave_blob

	return max(terrain_sdf, -cave_sdf)
end

local evaluateWithHeight = Sdf.evaluateWithHeight
local height = Sdf.height

function Sdf.evaluate(x: number, y: number, z: number): number
	return evaluateWithHeight(x, y, z, height(x, z))
end

local evaluate = Sdf.evaluate

-- SDFBase::evaluate_normal: central differences, normalized.
function Sdf.normal(x: number, y: number, z: number, alpha: number): (number, number, number)
	local gx = evaluate(x + alpha, y, z) - evaluate(x - alpha, y, z)
	local gy = evaluate(x, y + alpha, z) - evaluate(x, y - alpha, z)
	local gz = evaluate(x, y, z + alpha) - evaluate(x, y, z - alpha)
	local len = math.sqrt(gx * gx + gy * gy + gz * gz)
	if len == 0 then
		return 0, 0, 0 -- Vector3::normalized() of zero is zero
	end
	return gx / len, gy / len, gz / len
end

return Sdf
