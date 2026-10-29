extends RefCounted
## The fixed parts of the mixer's shaders (Godot shading language); compile.gd writes the graph's uniforms and its
## density() between them. The noise is the engine's perlin3 (src/exact_noise.hpp: the same pcg3d hash, cube-edge
## gradients and quintic fade) in float math relative to the preview's origin: the same notes, not the same bits.

const Kinds := preload("res://debug/biome_mixer/kinds.gd")

# The octave table's rows (notes, and three per warp octave) and the warp fields the preview takes.
const MAX_OCTAVES := 64
const MAX_WARPS := 8

const NOISE := """
uvec3 pcg3d(uvec3 v) {
	v = v * 1664525u + 1013904223u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v ^= v >> uvec3(16u);
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	return v;
}

// Perlin's 12 cube edges for i in [0, 16) (ExactNoise::grad3).
vec3 grad3(uint i) {
	float sa = (i & 1u) != 0u ? -1.0 : 1.0;
	float sb = (i & 2u) != 0u ? -1.0 : 1.0;
	uint plane = i >> 2u;
	if (plane == 1u) {
		return vec3(sa, 0.0, sb);
	}
	if (plane == 2u) {
		return vec3(0.0, sa, sb);
	}
	return i < 14u ? vec3(sa, sb, 0.0) : vec3(0.0, sb, sa);
}

float corner(uvec3 c, uvec3 s, vec3 f) {
	return dot(grad3(pcg3d(c ^ s).x >> 28u), f);
}

// One octave at pv (voxels from the preview's origin): base, frac and salt from octaves.gd shader_words, inv = 2^-k.
float perlin(vec3 pv, ivec3 base, vec3 frac0, ivec3 salt, float inv) {
	vec3 x = frac0 + pv * inv;
	vec3 fl = floor(x);
	vec3 f = x - fl;
	uvec3 c = uvec3(base + ivec3(fl));
	uvec3 s = uvec3(salt);
	vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
	float va = corner(c, s, f);
	float vb = corner(c + uvec3(1u, 0u, 0u), s, f - vec3(1.0, 0.0, 0.0));
	float vc = corner(c + uvec3(0u, 1u, 0u), s, f - vec3(0.0, 1.0, 0.0));
	float vd = corner(c + uvec3(1u, 1u, 0u), s, f - vec3(1.0, 1.0, 0.0));
	float ve = corner(c + uvec3(0u, 0u, 1u), s, f - vec3(0.0, 0.0, 1.0));
	float vf = corner(c + uvec3(1u, 0u, 1u), s, f - vec3(1.0, 0.0, 1.0));
	float vg = corner(c + uvec3(0u, 1u, 1u), s, f - vec3(0.0, 1.0, 1.0));
	float vh = corner(c + uvec3(1u, 1u, 1u), s, f - vec3(1.0, 1.0, 1.0));
	return mix(mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y), mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y), u.z);
}

// The engine's smoothed |n| (Cascade::smooth_abs; E from kinds.gd SMOOTH_ABS_E).
float sabs(float n) {
	return sqrt(n * n + {E} * {E}) - {E};
}

// A band's shape c0 + c1 n + c2 |n| + c3 n^2 (Climate::Shape).
float shape(vec4 c, float n) {
	return c.x + c.y * n + c.z * sabs(n) + c.w * n * n;
}

// A note fades out as its wavelength nears two pixels (its footprint fp): no aliasing, and the fine notes cost
// nothing far away. Every note is centred, so dropping one moves no height.
float lod_fade(float lam, float fp) {
	return smoothstep(2.0, 4.0, lam / max(fp, 1e-6));
}

// libnoise's Terrace every step (Climate::Terrace): v0 + step t^2, inverted v1 - step (1 - t)^2.
float terrace(float h, float step, bool inv) {
	float u = h / step;
	float fl = floor(u);
	float a = u - fl;
	float v0 = fl * step;
	if (inv) {
		float b = 1.0 - a;
		return v0 + step - step * b * b;
	}
	return v0 + step * a * a;
}
"""

const HEAD := """shader_type canvas_item;
render_mode unshaded;

uniform vec3 u_cam_pos = vec3(0.0, 500.0, 1000.0);
uniform vec3 u_cam_fwd = vec3(0.0, 0.0, -1.0);
uniform vec3 u_cam_right = vec3(1.0, 0.0, 0.0);
uniform vec3 u_cam_up = vec3(0.0, 1.0, 0.0);
uniform float u_tan_half_fov = 0.5;
uniform int u_max_steps = 256;
// The march step: the density times u_step over its Lipschitz bound u_lip (compile.gd's slope bound).
uniform float u_step = 1.0;
uniform float u_lip = 1.0;
// The highest the ground reaches (metres, the exaggeration in), and the farthest ray.
uniform float u_hmax = 100.0;
uniform float u_far = 20000.0;
uniform float u_exag = 1.0;
uniform vec3 u_sun = vec3(0.45, 0.6, 0.3);
uniform bool u_shadows = true;
uniform bool u_water_on = false;
uniform float u_water = 0.0;
uniform float u_contour = 0.0;
uniform vec3 u_flat = vec3(0.24, 0.36, 0.16);
uniform vec3 u_steep = vec3(0.36, 0.32, 0.28);
uniform float u_voxel = 0.05;

// The octave table (compile.gd _table): every note and warp octave, read in one loop so the shader holds one perlin
// call (the driver compiled one per note, inlined at each density call, in 30 s). Rows below u_n_warp add to their
// warp W[u_slot] (axis u_axis, times u_amp); the others are the notes N[i - u_n_warp], read at pv + u_wm W[u_slot].
// Columns: OCTAVE_TABLE (cell base and salt words, frac, 2^-k, wavelength, warp amplitude, warp distance,
// 1 / stretch_y, slot, axis, the note warp's notes x, y, z: earlier rows, -1 for none). A note's fade goes to F, apart from its noise N (a Shape fades around its mean).
uniform int u_n_oct = 0;
uniform int u_n_warp = 0;
"""

const OCTAVE_TABLE := [
	["u_bx", "int"], ["u_by", "int"], ["u_bz", "int"], ["u_sx", "int"], ["u_sy", "int"], ["u_sz", "int"],
	["u_fr", "vec3"], ["u_inv", "float"], ["u_lam", "float"], ["u_amp", "float"], ["u_wm", "float"], ["u_st", "float"],
	["u_slot", "int"], ["u_axis", "int"], ["u_jx", "int"], ["u_jy", "int"], ["u_jz", "int"],
]

# W and N are sized to the graph's warps and notes (structural: a new note recompiles anyway), not to the table's
# maximum: dynamically indexed, they live in per-pixel memory.
const DENSITY_HEAD := """
float density(vec3 p, float fp) {
	vec3 pv = p / u_voxel;
	vec3 W[{warps}];
	for (int j = 0; j < {warps}; j++) {
		W[j] = vec3(0.0);
	}
	float N[{notes}];
	float F[{notes}];
	for (int i = 0; i < u_n_oct; i++) {
		bool warp_row = i < u_n_warp;
		int s = u_slot[i];
		vec3 q = vec3(pv.x, pv.y * u_st[i], pv.z);
		if (!warp_row && s >= 0) {
			q += u_wm[i] * W[s];
		}
		ivec3 j = ivec3(u_jx[i], u_jy[i], u_jz[i]);
		if (!warp_row && j != ivec3(-1)) {
			q += u_wm[i] * vec3(j.x >= 0 ? F[j.x] * N[j.x] : 0.0, j.y >= 0 ? F[j.y] * N[j.y] : 0.0, j.z >= 0 ? F[j.z] * N[j.z] : 0.0);
		}
		float f = lod_fade(u_lam[i], fp);
		float n = f > 0.0 ? perlin(q, ivec3(u_bx[i], u_by[i], u_bz[i]), u_fr[i], ivec3(u_sx[i], u_sy[i], u_sz[i]), u_inv[i]) : 0.0;
		if (warp_row) {
			int a = u_axis[i];
			W[s] += vec3(a == 0 ? n : 0.0, a == 1 ? n : 0.0, a == 2 ? n : 0.0) * (f * u_amp[i]);
		} else {
			N[i - u_n_warp] = n;
			F[i - u_n_warp] = f;
		}
	}
"""

const MAIN := """
vec3 sky(vec3 rd) {
	float t = clamp(rd.y, 0.0, 1.0);
	vec3 c = mix(vec3(0.62, 0.72, 0.82), vec3(0.18, 0.36, 0.66), sqrt(t));
	return c + vec3(1.0, 0.85, 0.6) * pow(max(dot(rd, normalize(u_sun)), 0.0), 64.0) * 0.6;
}

// The normal's tetrahedron taps (one per pass of the loop below).
vec3 tap(int k) {
	return k == 0 ? vec3(1.0, -1.0, -1.0) : (k == 1 ? vec3(-1.0, -1.0, 1.0) : (k == 2 ? vec3(-1.0, 1.0, -1.0) : vec3(1.0)));
}

// One loop, one density() call: the march (phase 0), its bisection once inside the ground (1), the normal's four taps
// (2), the soft shadow towards the sun (3), done (4). Each call site would inline the whole graph: the driver took
// half a minute to compile seven.
void fragment() {
	// The pixel in its viewport, not UV: the view draws in strips (preview.gd), each a rect of its own.
	vec2 uv = FRAGCOORD.xy * SCREEN_PIXEL_SIZE;
	vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	float aspect = SCREEN_PIXEL_SIZE.y / SCREEN_PIXEL_SIZE.x;
	vec3 rd = normalize(u_cam_fwd + ndc.x * u_tan_half_fov * aspect * u_cam_right + ndc.y * u_tan_half_fov * u_cam_up);
	vec3 ro = u_cam_pos;
	float pix = 2.0 * u_tan_half_fov * SCREEN_PIXEL_SIZE.y;
	vec3 sun = normalize(u_sun);
	float t = 0.0;
	bool hit = false;
	int phase = ro.y > u_hmax && rd.y >= 0.0 ? 4 : 0;
	if (phase == 0 && ro.y > u_hmax) {
		t = (ro.y - u_hmax) / -rd.y;
	}
	float prev_t = t;
	float last_d = 1e9;
	float a = 0.0;
	float b = 0.0;
	int k = 0;
	int steps = 0;
	vec3 hp = vec3(0.0);
	float hfp = 1.0;
	float e = 0.01;
	vec3 nsum = vec3(0.0);
	vec3 n = vec3(0.0, 1.0, 0.0);
	vec3 sp = vec3(0.0);
	float st = 0.5;
	float sh = 1.0;
	for (int i = 0; i < u_max_steps + 80 && phase < 4; i++) {
		vec3 q = vec3(0.0);
		float fq = 0.0;
		if (phase == 0) {
			q = ro + rd * t;
			fq = t * pix;
			if (q.y > u_hmax && rd.y >= 0.0) {
				phase = 4;
				continue;
			}
		} else if (phase == 1) {
			float m = 0.5 * (a + b);
			q = ro + rd * m;
			fq = m * pix;
		} else if (phase == 2) {
			q = hp + tap(k) * e;
			fq = hfp;
		} else {
			q = sp + sun * st;
			fq = hfp + st * 0.002;
			if (q.y > u_hmax) {
				phase = 4;
				continue;
			}
		}
		float d = density(q, fq);
		if (phase == 0) {
			last_d = d;
			steps++;
			if (d < 0.0) {
				a = prev_t;
				b = t;
				k = 0;
				phase = 1;
			} else if (d < 0.0005 * t) {
				hit = true;
			} else {
				prev_t = t;
				// At least a pixel's footprint: past it the march can't tell the ground anyway.
				t += max(d * u_step / u_lip, t * pix + 0.02);
				if (t > u_far) {
					phase = 4;
				} else if (steps >= u_max_steps) {
					// Out of steps near the ground (a grazing ray): shade it where it got to, not the sky through it.
					hit = last_d < 0.05 * t;
					phase = hit ? 0 : 4;
				}
			}
		} else if (phase == 1) {
			float m = 0.5 * (a + b);
			if (d < 0.0) {
				b = m;
			} else {
				a = m;
			}
			k++;
			if (k == 8) {
				// The end outside the ground: from the inside one (its depth grows with the step) the shadow ray would
				// start underground (rings of acne on smooth slopes).
				t = a;
				hit = true;
			}
		} else if (phase == 2) {
			nsum += tap(k) * d;
			k++;
			if (k == 4) {
				n = normalize(nsum);
				sp = hp + n * max(hfp, 0.05) * 2.0;
				k = 0;
				phase = u_shadows && dot(n, sun) > 0.0 ? 3 : 4;
			}
		} else {
			sh = min(sh, 8.0 * d / st);
			st += clamp(d * u_step / u_lip, 0.2, 400.0);
			k++;
			if (sh < 0.01 || k == 64) {
				phase = 4;
			}
		}
		if (hit && phase < 2) {
			hp = ro + rd * t;
			hfp = t * pix;
			e = max(hfp, 0.01);
			k = 0;
			phase = 2;
		}
	}
	sh = clamp(sh, 0.0, 1.0);
	vec3 col = sky(rd);
	float t_water = u_water_on && rd.y < 0.0 && ro.y > u_water ? (ro.y - u_water) / -rd.y : -1.0;
	if (hit) {
		vec3 p = hp;
		// The engine's slope blend of a ground's colours (page_shading_glsl.hpp ground_flat_weight).
		vec3 base = mix(u_steep, u_flat, smoothstep(0.8, 0.95, n.y));
		float diff = max(dot(n, sun), 0.0);
		vec3 lit = base * (vec3(1.0, 0.95, 0.85) * 1.5 * diff * sh + vec3(0.22, 0.27, 0.36) * (0.5 + 0.5 * n.y));
		if (u_contour > 0.0) {
			float h = p.y / u_contour;
			float l = abs(fract(h + 0.5) - 0.5) / max(fwidth(h), 1e-5);
			lit *= mix(0.6, 1.0, smoothstep(0.5, 1.5, l));
		}
		col = mix(lit, sky(rd), 1.0 - exp(-t / (u_far * 0.8)));
	}
	if (t_water > 0.0 && (!hit || t_water < t)) {
		vec3 deep = vec3(0.04, 0.16, 0.24);
		vec3 refl = sky(reflect(rd, vec3(0.0, 1.0, 0.0)));
		float fres = 0.04 + 0.96 * pow(1.0 - max(-rd.y, 0.0), 5.0);
		float depth = hit ? clamp((t - t_water) * 0.02, 0.0, 1.0) : 1.0;
		vec3 water = mix(mix(col, deep, depth), refl, fres);
		col = mix(water, sky(rd), 1.0 - exp(-t_water / (u_far * 0.8)));
	}
	col = 1.0 - exp(-col * 1.1);
	COLOR = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
"""

# A note's thumbnail: its noise alone, seen from above over u_span voxels, through a shape (u_c).
const THUMB := """shader_type canvas_item;
uniform ivec3 u_base;
uniform vec3 u_frac;
uniform ivec3 u_salt;
uniform float u_inv = 0.125;
uniform float u_span = 64.0;
%s
void fragment() {
	vec3 pv = vec3((UV.x - 0.5) * u_span, 0.0, (UV.y - 0.5) * u_span);
	float v = perlin(pv, u_base, u_frac, u_salt, u_inv);
	COLOR = vec4(vec3(clamp(v * 0.5 + 0.5, 0.0, 1.0)), 1.0);
}
"""


static func _noise() -> String:
	return NOISE.replace("{E}", String.num(Kinds.SMOOTH_ABS_E))


# The preview shader: the fixed head and the table's columns, the graph's uniforms and functions, density() (its octave
# loop, then the graph's body p_body, sized to its p_notes notes and p_warps warp fields), the march.
static func preview(p_uniforms: String, p_functions: String, p_body: String, p_notes: int, p_warps: int) -> String:
	var table := PackedStringArray()
	for c in OCTAVE_TABLE:
		table.append("uniform %s %s[%d];" % [c[1], c[0], MAX_OCTAVES])
	var density := DENSITY_HEAD.format({"warps": maxi(p_warps, 1), "notes": maxi(p_notes, 1)})
	return HEAD + "\n".join(table) + "\n" + p_uniforms + _noise() + p_functions + density + p_body + "}\n" + MAIN


static func thumb() -> String:
	return THUMB % _noise()
