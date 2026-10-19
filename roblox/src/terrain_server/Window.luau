--!strict
--!native
-- Terrain playground, server only: one streaming window, the slabs written around a center (logical
-- studs, x and z) at one origin. physical = logical - origin, the SDF is sampled at logical positions
-- (TerrainGen's shift is -origin). Origins only move on x and z.
--   near:  full detail (caves, edits) within NEAR slabs of the center's slab (Chebyshev distance),
--          written in pieces: a whole near slab is too many ms of sampling for one frame;
--   crust: the height field as a thin shell (TerrainGen.writeFarTile) out to CRUST slabs;
--   farther: cleared. Both rings keep one slab of hysteresis, so pacing on a boundary doesn't churn.
-- The work is a queue (near writes, then crust writes, then clears, each nearest first), rebuilt when
-- the center enters another slab and run by step() until a deadline.

local ReplicatedStorage = game:GetService("ReplicatedStorage")

local OptiVoxel = ReplicatedStorage:WaitForChild("OptiVoxel")
local Config = require(OptiVoxel:WaitForChild("Config"))
local TerrainGen = require(OptiVoxel:WaitForChild("TerrainGen"))

local S = Config.STUDS_PER_UNIT
local SLAB = Config.STREAM_SLAB_STUDS
local PIECE = Config.STREAM_NEAR_PIECE_STUDS
local PIECES_PER_SIDE = SLAB // PIECE
local PIECES = PIECES_PER_SIDE * PIECES_PER_SIDE
local NEAR = Config.TERRAIN_HALF_STUDS // SLAB
local CRUST = Config.TERRAIN_FAR_HALF_STUDS // SLAB
local CRUST_STEP = Config.STREAM_CRUST_STEP_STUDS
local DEPTH = 120 / S -- units of ground (and caves) under the lowest surface point of a piece: 30 voxels

-- Slab levels, and job kinds (a job brings its slab to that level; sorted by kind, highest first).
local NONE, CRUST_LEVEL, NEAR_LEVEL = 0, 1, 2

-- Slab keys: exact while |ix|, |iz| < 2^25 slabs (2^31 studs).
local KEY_OFFSET, KEY_SPAN = 2 ^ 25, 2 ^ 26
local function keyOf(ix: number, iz: number): number
	return (ix + KEY_OFFSET) * KEY_SPAN + iz + KEY_OFFSET
end

type Slab = {
	ix: number,
	iz: number,
	level: number,
	pieces: number, -- near pieces written towards NEAR_LEVEL (0 once there)
	y0: number, -- studs range ever written in its column, cleared as a whole
	y1: number,
}
type Job = { key: number, ix: number, iz: number, kind: number, dist: number }

export type Stats = {
	near: number, -- slabs at each level
	crust: number,
	queued: number, -- jobs left
	queuedNear: number,
	cleared: number, -- slabs cleared so far
	sampleMs: number, -- totals so far
	writeMs: number,
	clearMs: number,
	rebuildMs: number, -- the slowest queue rebuild so far
	slowestJobMs: number, -- the slowest job (one near piece, a crust slab or a clear) so far, and its kind
	slowestJob: string,
}

export type Window = {
	origin: Vector3,
	setCenter: (x: number, z: number) -> (),
	step: (deadline: number) -> (),
	complete: () -> boolean,
	nearReady: (x: number, z: number, radius: number) -> boolean,
	edge: () -> number,
	applyEdit: (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) -> (number, number, number),
	retire: () -> (),
	stats: Stats,
}

local Window = {}

function Window.new(terrain: Terrain, origin: Vector3): Window
	local shift = Vector2.new(-origin.X, -origin.Z)
	local slabs: { [number]: Slab } = {}
	local queue: { Job } = {}
	local head = 1
	local cx, cz = 0, 0 -- center, logical studs
	local cix: number?, ciz: number? = nil, nil -- its slab
	local retired = false
	local stats: Stats = { near = 0, crust = 0, queued = 0, queuedNear = 0, cleared = 0, sampleMs = 0, writeMs = 0, clearMs = 0, rebuildMs = 0, slowestJobMs = 0, slowestJob = "" }

	local function setLevel(slab: Slab, level: number)
		if slab.level == NEAR_LEVEL then
			stats.near -= 1
		elseif slab.level == CRUST_LEVEL then
			stats.crust -= 1
		end
		if level == NEAR_LEVEL then
			stats.near += 1
		elseif level == CRUST_LEVEL then
			stats.crust += 1
		end
		slab.level = level
	end

	local function grow(slab: Slab, y0: number, y1: number)
		slab.y0, slab.y1 = math.min(slab.y0, y0), math.max(slab.y1, y1)
	end

	-- Physical studs of a slab's corner.
	local function corner(ix: number, iz: number): (number, number)
		return ix * SLAB - origin.X, iz * SLAB - origin.Z
	end

	local function clear(slab: Slab)
		local x0, z0 = corner(slab.ix, slab.iz)
		stats.clearMs += TerrainGen.clearBox(terrain, x0, slab.y0, z0, x0 + SLAB, slab.y1, z0 + SLAB)
		slab.y0, slab.y1 = math.huge, -math.huge
		slab.pieces = 0
		setLevel(slab, NONE)
	end

	-- Partly written near slabs count as near (hysteresis, and they need a clear before a crust).
	local function nearish(slab: Slab): boolean
		return slab.level == NEAR_LEVEL or slab.pieces > 0
	end

	local function target(slab: Slab?, d: number): number
		local level = if slab == nil then NONE elseif nearish(slab) then NEAR_LEVEL else slab.level
		if d <= NEAR or (d == NEAR + 1 and level == NEAR_LEVEL) then
			return NEAR_LEVEL
		elseif d <= CRUST or (d == CRUST + 1 and level ~= NONE) then
			return CRUST_LEVEL
		end
		return NONE
	end

	-- The job that brings the slab to `level`, if it isn't there.
	local function needed(slab: Slab?, level: number): number?
		if level == NEAR_LEVEL then
			return if slab and slab.level == NEAR_LEVEL then nil else NEAR_LEVEL
		elseif level == CRUST_LEVEL then
			return if slab and slab.level == CRUST_LEVEL and slab.pieces == 0 then nil else CRUST_LEVEL
		end
		return if slab then NONE else nil
	end

	local function push(ix: number, iz: number, kind: number)
		local dx, dz = (ix + 0.5) * SLAB - cx, (iz + 0.5) * SLAB - cz
		table.insert(queue, { key = keyOf(ix, iz), ix = ix, iz = iz, kind = kind, dist = dx * dx + dz * dz })
	end

	local function sortQueue()
		table.sort(queue, function(a: Job, b: Job): boolean
			if a.kind ~= b.kind then
				return a.kind > b.kind
			end
			return a.dist < b.dist
		end)
		head = 1
		stats.queued = #queue
		stats.queuedNear = 0
		for _, job in queue do
			if job.kind == NEAR_LEVEL then
				stats.queuedNear += 1
			end
		end
	end

	local function rebuild(ix: number, iz: number)
		local t0 = os.clock()
		queue = {}
		for _, slab in slabs do
			local d = math.max(math.abs(slab.ix - ix), math.abs(slab.iz - iz))
			if d > CRUST then -- the square below covers the rest
				local kind = needed(slab, target(slab, d))
				if kind then
					push(slab.ix, slab.iz, kind)
				end
			end
		end
		for x = ix - CRUST, ix + CRUST do
			for z = iz - CRUST, iz + CRUST do
				local slab = slabs[keyOf(x, z)]
				local kind = needed(slab, target(slab, math.max(math.abs(x - ix), math.abs(z - iz))))
				if kind then
					push(x, z, kind)
				end
			end
		end
		sortQueue()
		stats.rebuildMs = math.max(stats.rebuildMs, (os.clock() - t0) * 1000)
	end

	local function slabFor(job: Job): Slab
		local slab = slabs[job.key]
		if not slab then
			slab = { ix = job.ix, iz = job.iz, level = NONE, pieces = 0, y0 = math.huge, y1 = -math.huge }
			slabs[job.key] = slab
		end
		return slab
	end

	-- One piece of a near slab; true once the slab is done.
	local function runNear(job: Job): boolean
		local slab = slabFor(job)
		if slab.level == NEAR_LEVEL then
			return true
		end
		local x0, z0 = corner(job.ix, job.iz)
		x0 += (slab.pieces % PIECES_PER_SIDE) * PIECE
		z0 += (slab.pieces // PIECES_PER_SIDE) * PIECE
		-- Written over the crust in place: the box holds the whole shell (it reaches DEPTH under the
		-- lowest surface point and above the highest), so no clear and no hole while upgrading.
		local y0, y1 = TerrainGen.yRange(x0, z0, x0 + PIECE, z0 + PIECE, DEPTH, shift)
		local a, b = TerrainGen.writeBox(terrain, x0, y0, z0, x0 + PIECE, y1, z0 + PIECE, shift)
		stats.sampleMs += a
		stats.writeMs += b
		grow(slab, y0, y1)
		slab.pieces += 1
		if slab.pieces < PIECES then
			return false
		end
		slab.pieces = 0
		setLevel(slab, NEAR_LEVEL)
		stats.queuedNear -= 1
		return true
	end

	local function runCrust(job: Job)
		local slab = slabFor(job)
		if nearish(slab) then
			clear(slab) -- in the same frame as the shell: the column is never seen empty
		end
		local x0, z0 = corner(job.ix, job.iz)
		local a, b, _, y0, y1 = TerrainGen.writeFarTile(terrain, x0, z0, x0 + SLAB, z0 + SLAB, CRUST_STEP, shift)
		stats.sampleMs += a
		stats.writeMs += b
		grow(slab, y0, y1)
		setLevel(slab, CRUST_LEVEL)
	end

	local function runClear(job: Job)
		local slab = slabs[job.key]
		if slab then
			clear(slab)
			slabs[job.key] = nil
			stats.cleared += 1
		end
	end

	local function setCenter(x: number, z: number)
		cx, cz = x, z
		local ix, iz = x // SLAB, z // SLAB
		if retired or (ix == cix and iz == ciz) then
			return
		end
		cix, ciz = ix, iz
		rebuild(ix, iz)
	end

	local function step(deadline: number)
		while head <= #queue and os.clock() < deadline do
			local job = queue[head]
			local t0 = os.clock()
			if job.kind == NEAR_LEVEL then
				if runNear(job) then
					head += 1
				end
			else
				if job.kind == CRUST_LEVEL then
					runCrust(job)
				else
					runClear(job)
				end
				head += 1
			end
			local ms = (os.clock() - t0) * 1000
			if ms > stats.slowestJobMs then
				stats.slowestJobMs, stats.slowestJob = ms, if job.kind == NEAR_LEVEL then "near" elseif job.kind == CRUST_LEVEL then "crust" else "clear"
			end
		end
		stats.queued = #queue - head + 1
	end

	local function complete(): boolean
		return cix ~= nil and head > #queue
	end

	-- Every slab within `radius` slabs of the one holding (x, z) (logical) is at full detail.
	local function nearReady(x: number, z: number, radius: number): boolean
		local ix, iz = x // SLAB, z // SLAB
		for i = ix - radius, ix + radius do
			for k = iz - radius, iz + radius do
				local slab = slabs[keyOf(i, k)]
				if not slab or slab.level ~= NEAR_LEVEL then
					return false
				end
			end
		end
		return true
	end

	-- Studs from the center to the nearest slab of the near square not at full detail yet (the edge of
	-- the square when it is all written).
	local function edge(): number
		if not cix or not ciz then
			return 0
		end
		local best = math.min(cx - (cix - NEAR) * SLAB, (cix + NEAR + 1) * SLAB - cx, cz - (ciz - NEAR) * SLAB, (ciz + NEAR + 1) * SLAB - cz)
		for ix = cix - NEAR, cix + NEAR do
			for iz = ciz - NEAR, ciz + NEAR do
				local slab = slabs[keyOf(ix, iz)]
				if not slab or slab.level ~= NEAR_LEVEL then
					local dx = math.max(ix * SLAB - cx, 0, cx - (ix + 1) * SLAB)
					local dz = math.max(iz * SLAB - cz, 0, cz - (iz + 1) * SLAB)
					best = math.min(best, math.max(dx, dz))
				end
			end
		end
		return best
	end

	-- Rewrites the logical box where it is at full detail (slabs being upgraded included: the pieces
	-- still to come sample the edit anyway). Crust and missing slabs pick the edit up when written.
	-- Returns ms sampling, ms writing, voxels.
	local function applyEdit(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): (number, number, number)
		local hit: { Slab } = {}
		local whole = true
		for ix = x0 // SLAB, x1 // SLAB do
			for iz = z0 // SLAB, z1 // SLAB do
				local slab = slabs[keyOf(ix, iz)]
				if slab and nearish(slab) then
					table.insert(hit, slab)
				else
					whole = false
				end
			end
		end
		local a, b, n = 0, 0, 0
		if whole then
			a, b, n = TerrainGen.writeBox(terrain, x0 - origin.X, y0, z0 - origin.Z, x1 - origin.X, y1, z1 - origin.Z, shift)
		else
			for _, slab in hit do
				local sx, sz = slab.ix * SLAB, slab.iz * SLAB
				local bx0, bz0 = math.max(x0, sx), math.max(z0, sz)
				local bx1, bz1 = math.min(x1, sx + SLAB), math.min(z1, sz + SLAB)
				local sa, sb, sn = TerrainGen.writeBox(terrain, bx0 - origin.X, y0, bz0 - origin.Z, bx1 - origin.X, y1, bz1 - origin.Z, shift)
				a, b, n = a + sa, b + sb, n + sn
			end
		end
		for _, slab in hit do
			grow(slab, y0, y1)
		end
		return a, b, n
	end

	-- Clears every slab (the window behind a treadmill jump). The center no longer matters.
	local function retire()
		retired = true
		queue = {}
		for _, slab in slabs do
			push(slab.ix, slab.iz, NONE)
		end
		sortQueue()
	end

	return {
		origin = origin,
		stats = stats,
		setCenter = setCenter,
		step = step,
		complete = complete,
		nearReady = nearReady,
		edge = edge,
		applyEdit = applyEdit,
		retire = retire,
	}
end

return Window
