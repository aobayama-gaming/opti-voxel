--!strict
--!native
-- Port of SDFDummy::evaluate (src/sdf_dummy.hpp): noise mountain carved by sparse cave blobs.
-- Units are the Godot world units (meters); the Roblox side scales to studs at the very end.

local Noise = if script then require(script.Parent.Noise) else require("./Noise")

local fractal = Noise.fractal
local max = math.max

local Sdf = {}

-- Edits: spheres applied in order, 4 f64 each (x, y, z, radius). A positive radius carves
-- (max(sdf, radius - distance)), a negative one adds matter (min(sdf, distance - |radius|)).
-- Per Lua VM: an Actor that samples or meshes needs its own copy.
-- `edits` / `editCount` are what evaluate() applies: every edit, or the subset focusEdits() picked.
local all = buffer.create(0)
local allCount = 0
local scratch = buffer.create(0)
local edits, editCount = all, 0
local focused = false

local function addEdit(x: number, y: number, z: number, radius: number)
	assert(not focused, "Sdf: edit added while focused")
	if (allCount + 1) * 32 > buffer.len(all) then
		local grown = buffer.create(math.max(32, buffer.len(all) * 2))
		buffer.copy(grown, 0, all)
		all = grown
	end
	local o = allCount * 32
	buffer.writef64(all, o, x)
	buffer.writef64(all, o + 8, y)
	buffer.writef64(all, o + 16, z)
	buffer.writef64(all, o + 24, radius)
	allCount += 1
	edits, editCount = all, allCount
end

function Sdf.addSubtractSphere(x: number, y: number, z: number, radius: number)
	addEdit(x, y, z, radius)
end

function Sdf.addUnionSphere(x: number, y: number, z: number, radius: number)
	addEdit(x, y, z, -radius)
end

function Sdf.editCount(): number
	return allCount
end

function Sdf.clearEdits()
	assert(not focused, "Sdf: edits cleared while focused")
	allCount = 0
	edits, editCount = all, 0
end

-- Squared distance from (x, y, z) to the box [x0, x1] x [y0, y1] x [z0, z1].
local function boxDistance2(x: number, y: number, z: number, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): number
	local dx = math.max(x0 - x, 0, x - x1)
	local dy = math.max(y0 - y, 0, y - y1)
	local dz = math.max(z0 - z, 0, z - z1)
	return dx * dx + dy * dy + dz * dz
end

-- Until unfocusEdits(), evaluate() only applies the edits whose sphere grown by `margin` meets the
-- box. An edit leaves every value it can change past `margin` of its sphere on the same side of
-- +-margin (min / max with it keep a value <= -margin below it and one >= margin above it), so a
-- consumer that clamps values to [-margin, margin] (Smooth Terrain occupancy) sees no difference.
function Sdf.focusEdits(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, margin: number)
	assert(not focused, "Sdf: already focused")
	focused = true
	if buffer.len(scratch) < buffer.len(all) then
		scratch = buffer.create(buffer.len(all))
	end
	local readf64 = buffer.readf64
	local n = 0
	for i = 0, allCount - 1 do
		local o = i * 32
		local reach = math.abs(readf64(all, o + 24)) + margin
		if boxDistance2(readf64(all, o), readf64(all, o + 8), readf64(all, o + 16), x0, y0, z0, x1, y1, z1) <= reach * reach then
			buffer.copy(scratch, n * 32, all, o, 32)
			n += 1
		end
	end
	edits, editCount = scratch, n
end

function Sdf.unfocusEdits()
	focused = false
	edits, editCount = all, allCount
end

-- Vertical extent (lo, hi) of the edits whose sphere grown by `margin` overlaps the column
-- [x0, x1] x [z0, z1], or nil: a box sampled from the height field alone would miss what they carved
-- under it or built over it.
function Sdf.editYRange(x0: number, z0: number, x1: number, z1: number, margin: number): (number?, number?)
	local readf64 = buffer.readf64
	local lo: number?, hi: number? = nil, nil
	for i = 0, allCount - 1 do
		local o = i * 32
		local y, reach = readf64(all, o + 8), math.abs(readf64(all, o + 24)) + margin
		if boxDistance2(readf64(all, o), y, readf64(all, o + 16), x0, y, z0, x1, y, z1) <= reach * reach then
			lo, hi = math.min(lo or math.huge, y - reach), math.max(hi or -math.huge, y + reach)
		end
	end
	return lo, hi
end

local function applyEdits(x: number, y: number, z: number, d: number): number
	local readf64 = buffer.readf64
	for i = 0, editCount - 1 do
		local o = i * 32
		local dx, dy, dz = x - readf64(edits, o), y - readf64(edits, o + 8), z - readf64(edits, o + 16)
		local r = readf64(edits, o + 24)
		local dist = math.sqrt(dx * dx + dy * dy + dz * dz)
		if r > 0 then
			local carve = r - dist
			if carve > d then
				d = carve
			end
		else
			local fill = dist + r
			if fill < d then
				d = fill
			end
		end
	end
	return d
end

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
		return if editCount > 0 then applyEdits(x, y, z, terrain_sdf) else terrain_sdf
	end

	local cave_blob = fractal(3, CAVE_FREQ, 1, 2, 0.5, x, y, z)
	local depth = mountain_height - y
	local blob_mask = (cave_blob + 0.3) * 0.5
	local depth_fade = depth / 40
	local cave_threshold = 0.5 + blob_mask * 0.4 + (1 - depth_fade) * 0.1
	local cave_sdf = cave_threshold - cave_blob

	local d = max(terrain_sdf, -cave_sdf)
	return if editCount > 0 then applyEdits(x, y, z, d) else d
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
