--!strict
--!native
-- Port of src/noise.cpp (Ashima/stegu 3D simplex noise). Only the value (.w) is computed: the
-- CPU SDF never uses the analytic gradient, and skipping it roughly halves the cost.

local floor = math.floor
local abs = math.abs

local Noise = {}

-- The C++/GLSL constants are float32 literals. They are kept float32-rounded here: the noise takes
-- floor() of products with them, and e.g. 147 * n_^2 lands on 3 with the float32 n_ but on
-- 2.9999999 with the exact 1/7, which changes the gradient picked for that corner.
local INV_289 = 0.0034602077212184668 -- 1.0f / 289.0f
local C_X = 0.1666666716337204 -- 1.0f / 6.0f
local C_Y = 0.3333333432674408 -- 1.0f / 3.0f
local TAYLOR_A = 1.7928428649902344 -- 1.79284291400159f
local TAYLOR_B = 0.8537347316741943 -- 0.85373472095314f

local function mod289(x: number): number
	return x - floor(x * INV_289) * 289
end

local function permute(x: number): number
	return mod289((x * 34 + 10) * x)
end

-- ns = n_ * D.wyz - D.xzx with n_ = 0.142857142857f
local N_ = 0.1428571492433548
local NS_X = N_ * 2
local NS_Y = N_ * 0.5 - 1
local NS_Z = N_

-- One simplex corner: gradient from the permutation value, dotted with the corner offset.
local function corner(pv: number, ox: number, oy: number, oz: number): number
	local m = 0.5 - (ox * ox + oy * oy + oz * oz)
	if m <= 0 then
		return 0
	end
	local j = pv - 49 * floor(pv * NS_Z * NS_Z)
	local x_ = floor(j * NS_Z)
	local y_ = floor(j - 7 * x_)
	local gx = x_ * NS_X + NS_Y
	local gy = y_ * NS_X + NS_Y
	local h = 1 - abs(gx) - abs(gy)
	if h <= 0 then -- sh = -step(h, 0)
		gx -= floor(gx) * 2 + 1
		gy -= floor(gy) * 2 + 1
	end
	local norm = TAYLOR_A - TAYLOR_B * (gx * gx + gy * gy + h * h)
	local m2 = m * m
	return m2 * m2 * (gx * ox + gy * oy + h * oz) * norm
end

function Noise.snoise(vx: number, vy: number, vz: number): number
	-- First corner
	local s = (vx + vy + vz) * C_Y
	local ix = floor(vx + s)
	local iy = floor(vy + s)
	local iz = floor(vz + s)
	local t = (ix + iy + iz) * C_X
	local x0 = vx - ix + t
	local y0 = vy - iy + t
	local z0 = vz - iz + t

	-- Other corners: g = step(x0.yzx, x0), l = 1 - g, i1 = min(g, l.zxy), i2 = max(g, l.zxy)
	local gx = if x0 < y0 then 0 else 1
	local gy = if y0 < z0 then 0 else 1
	local gz = if z0 < x0 then 0 else 1
	local lx, ly, lz = 1 - gx, 1 - gy, 1 - gz
	local i1x, i1y, i1z = math.min(gx, lz), math.min(gy, lx), math.min(gz, ly)
	local i2x, i2y, i2z = math.max(gx, lz), math.max(gy, lx), math.max(gz, ly)

	local x1, y1, z1 = x0 - i1x + C_X, y0 - i1y + C_X, z0 - i1z + C_X
	local x2, y2, z2 = x0 - i2x + C_Y, y0 - i2y + C_Y, z0 - i2z + C_Y
	local x3, y3, z3 = x0 - 0.5, y0 - 0.5, z0 - 0.5

	-- Permutations
	local mx = mod289(ix)
	local my = mod289(iy)
	local mz = mod289(iz)
	local p0 = permute(permute(permute(mz) + my) + mx)
	local p1 = permute(permute(permute(mz + i1z) + my + i1y) + mx + i1x)
	local p2 = permute(permute(permute(mz + i2z) + my + i2y) + mx + i2x)
	local p3 = permute(permute(permute(mz + 1) + my + 1) + mx + 1)

	return 105
		* (
			corner(p0, x0, y0, z0)
			+ corner(p1, x1, y1, z1)
			+ corner(p2, x2, y2, z2)
			+ corner(p3, x3, y3, z3)
		)
end

local snoise = Noise.snoise

-- Fractal Brownian motion, value only (SimplexNoise::fractal(...).w).
function Noise.fractal(
	octaves: number,
	frequency: number,
	amplitude: number,
	lacunarity: number,
	persistence: number,
	x: number,
	y: number,
	z: number
): number
	local acc = 0
	x, y, z = x * frequency, y * frequency, z * frequency
	for _ = 1, octaves do
		acc += snoise(x, y, z) * amplitude
		x, y, z = x * lacunarity, y * lacunarity, z * lacunarity
		amplitude *= persistence
	end
	return acc
end

return Noise
