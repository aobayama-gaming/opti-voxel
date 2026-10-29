extends RefCounted
## The engine's octaves in GDScript, so the mixer's notes are the ground's: make_octave (src/exact_noise.hpp), the
## ground's seeds (src/cascade_terrain.hpp make_spec: salt = k_top - k, k_top = lods + 4) and a field's (salt = k,
## src/climate_parse.cpp), and each note's lattice words for the preview shader. Pinned by the mixer's --check and
## test_exact_noise.cpp ("as the biome mixer pins them").

const GROUND_SEED := 0xEA27
const OCTAVE_SALT := 0x51ED270B
const SALT_Y := 0x9E3779B9
const SALT_Z := 0x85EBCA6B
const MASK := 0xFFFFFFFF
# The ground's amplitude: steepness 1/8 of the wavelength, capped at 3.3 km (SDFCascade's MAX_AMP_M, 2^16 voxels of
# 5 cm, the same metres at any voxel size).
const STEEPNESS := 0.125
const MAX_AMP_M := 3276.8
# The first octave LOD 0 inserts (8 voxels, 0.4 m at 5 cm).
const K_MIN := 3
# The ground's datum (BaseSurface::DATUM_M): the preview's y = 0. A biome file's defaults: world.json's 5 cm, 25 LODs.
const DATUM_M := 300.0
const DEFAULT_VOXEL_M := 0.05
const DEFAULT_LODS := 25

# make_octave's words per (seed, salt, k) and shader_words' per origin too: every edit rebuilds the octave table.
static var _octaves := {}
static var _words := {}


# The ground's top octave (SDFCascade: lods + 4); the notes span K_MIN .. k_top.
static func k_top(p_lods: int) -> int:
	return p_lods + 4


# a * b mod 2^32 without leaving int64 (a, b < 2^32).
static func mul32(p_a: int, p_b: int) -> int:
	var lo := p_a * (p_b & 0xFFFF)
	var hi := (p_a * (p_b >> 16)) & 0xFFFF
	return (lo + (hi << 16)) & MASK


static func pcg3d(p_v: Array) -> Array:
	var x: int = (mul32(p_v[0], 1664525) + 1013904223) & MASK
	var y: int = (mul32(p_v[1], 1664525) + 1013904223) & MASK
	var z: int = (mul32(p_v[2], 1664525) + 1013904223) & MASK
	x = (x + mul32(y, z)) & MASK
	y = (y + mul32(z, x)) & MASK
	z = (z + mul32(x, y)) & MASK
	x ^= x >> 16
	y ^= y >> 16
	z ^= z >> 16
	x = (x + mul32(y, z)) & MASK
	y = (y + mul32(z, x)) & MASK
	z = (z + mul32(x, y)) & MASK
	return [x, y, z]


# {seed, offset: [x, y, z]} of the octave of wavelength 2^k voxels hashed from (seed, salt), as make_octave.
static func make_octave(p_seed: int, p_salt: int, p_k: int) -> Dictionary:
	var key := [p_seed, p_salt, p_k]
	if not _octaves.has(key):
		_octaves[key] = _make_octave(p_seed, p_salt, p_k)
	return _octaves[key]


static func _make_octave(p_seed: int, p_salt: int, p_k: int) -> Dictionary:
	var h := pcg3d([p_seed & MASK, p_salt & MASK, OCTAVE_SALT])
	var h2 := pcg3d(h)
	var offset := [0, 0, 0]
	if p_k >= 2:
		var pairs := [[h[1], h2[0]], [h[2], h2[1]], [h2[2], h[0]]]
		for i in 3:
			var r: int = (pairs[i][0] << 32) | pairs[i][1]
			offset[i] = (1 << (p_k - 2)) + (r & ((1 << (p_k - 1)) - 1))
	return {"seed": h[0], "offset": offset}


# The ground's note k (variant 0) or another noise at the same wavelength (variant > 0: the ground's seed hashed).
static func ground_octave(p_k: int, p_lods: int, p_variant: int = 0) -> Dictionary:
	var seed: int = GROUND_SEED if p_variant == 0 else pcg3d([GROUND_SEED, p_variant, SALT_Y])[0]
	return make_octave(seed, k_top(p_lods) - p_k, p_k)


static func field_octave(p_seed: int, p_k: int) -> Dictionary:
	return make_octave(p_seed, p_k, p_k)


static func wavelength_m(p_k: int, p_voxel_m: float) -> float:
	return p_voxel_m * pow(2.0, p_k)


# The ground's amplitude of the octave of wavelength p_lam_m, metres (lambda / 8, capped at MAX_AMP_M whatever the voxel).
static func ground_amp_m(p_lam_m: float) -> float:
	return minf(STEEPNESS * p_lam_m, MAX_AMP_M)


static func to_s32(p_v: int) -> int:
	p_v &= MASK
	return p_v - 0x100000000 if p_v >= 0x80000000 else p_v


# The shader's words for an octave seen from a preview origin (absolute voxel, int64 per axis): base (the origin's
# cell, low words), frac (its position in the cell, 0 .. 1) and salt (cell_salt of the high words and the seed). The
# preview reads cell = base + floor(frac + p / 2^k), so no float ever holds an absolute coordinate; a preview crossing
# a 2^32-cell block keeps the origin's high words (a seam there, far beyond any preview at k >= 3).
static func shader_words(p_octave: Dictionary, p_k: int, p_origin: Array) -> Dictionary:
	var key := [p_octave.seed, p_octave.offset, p_k, p_origin]
	if not _words.has(key):
		_words[key] = _shader_words(p_octave, p_k, p_origin)
	return _words[key]


static func _shader_words(p_octave: Dictionary, p_k: int, p_origin: Array) -> Dictionary:
	var base := Vector3i()
	var frac := Vector3()
	var hi := [0, 0, 0]
	for i in 3:
		var v: int = p_origin[i] + p_octave.offset[i]
		var cell := v >> p_k
		frac[i] = float(v & ((1 << p_k) - 1)) / pow(2.0, p_k)
		base[i] = to_s32(cell)
		hi[i] = (cell >> 32) & MASK
	var s: int = p_octave.seed
	var salt := pcg3d([hi[0] ^ s, hi[1] ^ (s ^ SALT_Y), hi[2] ^ (s ^ SALT_Z)])
	return {"base": base, "frac": frac, "salt": Vector3i(to_s32(salt[0]), to_s32(salt[1]), to_s32(salt[2]))}


# The values test_exact_noise.cpp pins: [seed, salt, k, octave seed, offset].
const PINS := [
	[GROUND_SEED, 26, 3, 3435443487, [2, 2, 5]],
	[GROUND_SEED, 17, 12, 3158081444, [2901, 1439, 2980]],
	[GROUND_SEED, 12, 17, 2370181367, [88676, 52952, 39159]],
	[GROUND_SEED, 0, 29, 3616334499, [141560218, 304881832, 260891299]],
	[905, 14, 14, 1949008538, [10651, 9326, 4762]],
]


# The pins that differ (empty: the port is the engine's).
static func check_pins() -> Array:
	var bad := []
	for p in PINS:
		var o := make_octave(p[0], p[1], p[2])
		if o.seed != p[3] or o.offset != p[4]:
			bad.append({"pin": p, "got": o})
	return bad
