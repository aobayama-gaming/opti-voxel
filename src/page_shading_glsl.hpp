#pragma once

#include "exact_noise_glsl.hpp"
#include "ov/str.hpp"
#include "shade_frame_glsl.hpp"

// The page shader's materials, shared by both adapters' page shaders (Godot's godot_page_host.cpp, Dawn's
// dawn_page_host.cpp): a corner's material through its chunk's palette, the quad's two materials and the one a fragment
// shows, and each material's shading (Material::Pattern: bark, leaves, lava, a biome's ground) with the ground's
// surface detail. The host declares, before this text, PageFormat::glsl_decode(), Material::glsl_constants() and three
// textures read with texelFetch: quad_tex (unsigned: the page), palette_tex (unsigned: the palettes, palette_pool.hpp)
// and material_tex (float: Material::Library::make_image), and the shading frame's two functions (shade_frame_glsl.hpp:
// every pattern of the position reads it through the frame, the same at every render origin). Plain GLSL that Godot's
// shading language also takes (no #define, no arrays).
namespace PageShading {

inline constexpr const char GLSL[] = R"glsl(
// A lattice cell's hash in [0, 1)^3 (c: the cell's integer coordinates, as floats from floor). Integer arithmetic
// (PCG3D), so every neighbour of a cell reads the same value. A float hash of the coordinate did not: the compiler may
// reassociate (i + 1) * k, and at coordinates in the thousands the hash turned that last bit into another value, a
// seam along every lattice plane (the bark's horizontal lines, tools/bark_seams.sh).
vec3 cell_hash(vec3 c, uint salt) {
    return vec3(pcg3d(uvec3(ivec3(c)) ^ uvec3(salt)) >> 8u) * (1.0 / 16777216.0);
}

// Trilinear value noise in [0, 1], one lattice cell per unit, repeating every n cells (whole numbers per axis): x and
// x + n read the same value.
float value_noise(vec3 x, vec3 n) {
    vec3 i = floor(x);
    vec3 f = x - i;
    f = f * f * (3.0 - 2.0 * f);
    vec3 a = shade_wrap(i, n);
    vec3 b = shade_wrap_next(a, n);
    float x00 = mix(cell_hash(a, 0u).x, cell_hash(vec3(b.x, a.y, a.z), 0u).x, f.x);
    float x10 = mix(cell_hash(vec3(a.x, b.y, a.z), 0u).x, cell_hash(vec3(b.x, b.y, a.z), 0u).x, f.x);
    float x01 = mix(cell_hash(vec3(a.x, a.y, b.z), 0u).x, cell_hash(vec3(b.x, a.y, b.z), 0u).x, f.x);
    float x11 = mix(cell_hash(vec3(a.x, b.y, b.z), 0u).x, cell_hash(b, 0u).x, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Value noise of cells of about cell_m metres at render position p, through the shading frame (shade_frame_glsl.hpp):
// the same at every render origin. offset: added in cells.
float shade_noise(vec3 p, float cell_m, vec3 offset) {
    float period_m;
    vec3 s = shade_frame(p, cell_m, period_m);
    vec3 n = shade_cells(vec3(1.0 / cell_m), period_m);
    return value_noise(s * (n / period_m) + offset, n);
}

// Bark (PATTERN_BARK) as a 3D noise, so it needs no surface frame: noise stretched along world up makes plates and V
// grooves that run up a trunk and wander slowly; a sparse cell field puts knots where a kept cell's point lies near
// the surface, and the grain bends around them. Its axis is world up: a near-horizontal branch gets grooves around it
// (the branch's axis is not known here). p: bark_point's, in metres. Returns the grain, a smooth noise in [0, 1]
// (bark_plate makes the plates and grooves of it); knot: the knot's core, ring: its growth rings, in [0, 1]. Its
// lattices repeat in the shading frame's period of its longest feature (BARK_FEATURE_M, bark_point). Colour
// only: a bump from it (and a roughness following it) cost ~1.3 ms of depth pre-pass in the forest at 1080p (the
// pre-pass writes normals and roughness for SSAO / SSIL), for relief barely visible past a few metres.
const float BARK_FEATURE_M = 4.0; // the bark's longest feature (the wander): its shading-frame period
float bark_grain(vec3 p, out float knot, out float ring) {
    float period_m = shade_period_m(shade_period_vox(BARK_FEATURE_M));
    vec3 knot_cells = shade_cells(1.0 / vec3(0.7, 1.6, 0.7), period_m);
    vec3 knot_cell = period_m / knot_cells;
    // The 2 x 2 x 2 cells nearest p: a knot point lies in the middle 70% of its cell, and a knot reaches ~0.4 m.
    vec3 g = floor(p / knot_cell - 0.5);
    vec3 g_key = shade_wrap(g, knot_cells);
    vec3 bend = vec3(0.0);
    float nearest = 1e9;
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                vec3 c = g + vec3(float(x), float(y), float(z));
                vec3 key = g_key + vec3(float(x), float(y), float(z));
                key -= knot_cells * step(knot_cells, key);
                if (cell_hash(key, 1u).x > 0.35) {
                    continue; // about a third of the cells hold a knot
                }
                vec3 dk = p - (c + 0.15 + 0.7 * cell_hash(key, 2u)) * knot_cell;
                vec3 dn = dk / vec3(0.08, 0.13, 0.08);
                float r2 = dot(dn, dn);
                nearest = min(nearest, r2);
                float fall = max(1.0 - r2 * (1.0 / 12.0), 0.0);
                bend.xz += dk.xz * (1.5 * fall * fall);
            }
        }
    }
    knot = exp(-nearest);
    ring = nearest < 40.0 ? exp(-0.15 * nearest) * (0.5 + 0.5 * cos(sqrt(nearest) * 7.0)) : 0.0;
    // Grooves ~11 cm apart, 6x longer than wide.
    vec3 grain_cells = shade_cells(vec3(9.0, 9.0 / 6.0, 9.0), period_m);
    vec3 wander_cells = shade_cells(vec3(1.2, 0.25, 1.2), period_m);
    vec3 wander = vec3(1.6, 0.0, 1.6) * value_noise(p * (wander_cells / period_m), wander_cells);
    return value_noise((p + bend) * (grain_cells / period_m) + wander, grain_cells);
}

// The plate height in [0, 1] of a grain: plates with flat tops, V grooves where the grain crosses its middle.
float bark_plate(float grain) {
    return min(abs(2.0 * grain - 1.0) * 1.8, 1.0);
}

// Where the bark noise is read: the normal's horizontal direction at BARK_RADIUS (~25 grooves around any trunk; its
// slope is left out, so a tapering or leaning trunk keeps its groove spacing) and the height. A surface moved along
// its normal reads the same bark: erosion (a LOD L surface lies 2^L - 1 cells inside the LOD 0 one), the skirt's push
// along chunk borders and the mesher's error move it by a good part of a groove, and read at the position the bark
// jumped across LOD changes (tools/bark_seams.sh). Two trees read the same bark only at the same height; a flat face
// reads one column of it. The height is read through the shading frame (period: BARK_FEATURE_M, the wander's 4 m).
// Horizontal and height are the base's up's (trees grow along it, up_frame): x, y and z on the flat world. On a planet
// the height's frame is a whole period off per axis, not along the up: the bark shifts when the render origin moves.
const float BARK_RADIUS = 0.45;
vec3 bark_point(vec3 p, vec3 n, vec3 up) {
    vec3 t1, t2;
    up_frame(up, t1, t2);
    vec2 h = vec2(dot(n, t1), dot(n, t2));
    vec2 u = h * (BARK_RADIUS * inversesqrt(max(dot(h, h), 1e-8)));
    float period_m;
    return vec3(u.x, dot(shade_frame(p, BARK_FEATURE_M, period_m), up), u.y);
}

// The ground's surface detail, a multiplier on its albedo (the procedural ground's and a biome's PATTERN_GROUND):
// a slow tint over ~20 m between a cooler and a warmer shade, patches of ~2 m, and a grain of ~6 cm that fades out
// past 2 pixels per feature. Without it a flat biome is one colour to the horizon. World-space noise: the same at
// every LOD. p: the fragment's position (metres), footprint: metres per pixel there.
vec3 ground_detail(vec3 p, float footprint) {
    float macro = shade_noise(p, 19.0, vec3(0.0));
    vec3 d = mix(vec3(0.86, 0.88, 0.94), vec3(1.10, 1.06, 0.98), macro);
    d *= 0.92 + 0.16 * shade_noise(p, 2.3, vec3(7.0));
    float grain_w = 1.0 - smoothstep(0.015, 0.04, footprint);
    if (grain_w > 0.0) {
        d *= 1.0 + grain_w * 0.18 * (shade_noise(p, 0.06, vec3(0.0)) - 0.5);
    }
    return d;
}

// A corner's material: its chunk's palette block (header h1), its index in it, its id (palette). Index 0 is always
// the ground: no fetch (most corners).
uint corner_material(uvec4 h1, uint index) {
    if (index == 0u) {
        return MATERIAL_GROUND;
    }
    uint entry = page_header_palette(h1) + index;
    return page_palette_id(texelFetch(palette_tex, page_palette_texel(entry), 0), entry);
}

// A material's first texel in material_tex (albedo, roughness; the next one: emission, pattern).
ivec2 material_texel(uint mat_id) {
    return ivec2(int(MATERIAL_TEXELS * (mat_id % MATERIALS_PER_ROW)), int(mat_id / MATERIALS_PER_ROW));
}

// The procedural ground's slope weight: 1 on gentle slopes (its flat albedo), 0 on steep ones. n: the normal, up: the
// base surface's.
float ground_flat_weight(vec3 n, vec3 up) {
    return smoothstep(0.8, 0.95, clamp(dot(n, up), 0.0, 1.0));
}

// A material's pattern (Material::Pattern, as a float).
float material_pattern(uint mat_id) {
    return mat_id == MATERIAL_GROUND ? 0.0 : texelFetch(material_tex, material_texel(mat_id) + ivec2(1, 0), 0).a;
}

// The quad's two materials, from its corners' palette indices (ia: A's; B, C, D's from quad_tex, the texels after A's
// at first + 1 .. 3 of the group's row): x = A's material (global id), y =
// the other one's | its corner mask << MATERIAL_ID_BITS (bit k, k = 1..3: corner k is not A's material). The other one
// is the first in B, C, D that is not A's (A's when the quad is uniform); a third material in one quad (three meeting
// inside one cell) draws as it. The page shaders pass it flat from the provoking corner, A in both triangles: only A
// computes it, the others' values are discarded. The fragment's weight is the mask interpolated over the quad's uv
// (fragment_material), so the blend adds no interpolant: triangles are smaller than a pixel here, every attribute the
// fragment reads costs setup per triangle.
ivec2 quad_materials(uvec4 h1, uint ia, int first, int row) {
    uint ib = ia;
    uint mask = 0u;
    for (int k = 1; k < 4; k++) {
        uint ik = page_decode_palette(texelFetch(quad_tex, ivec2(first + k, row), 0));
        ib = ib == ia ? ik : ib;
        mask |= ik != ia ? 1u << uint(k) : 0u;
    }
    uint mat_a = corner_material(h1, ia);
    uint mat_b = ib == ia ? mat_a : corner_material(h1, ib);
    return ivec2(int(mat_a), int(mat_b | mask << MATERIAL_ID_BITS));
}

// The material a fragment shows of its quad's two (quad_materials): the higher id where the weight towards it (the
// corner mask over the quad's uv) jittered by a noise of about a cell passes 1/2, so the border wanders instead of
// following the cell grid. Towards the higher id, not towards b: two quads sharing an edge may hold the pair in either
// order, and must agree on the border. p: the fragment's position (metres), cell_m: a cell of the quad's LOD (metres).
uint fragment_material(ivec2 materials, vec2 quad_uv, vec3 p, float cell_m) {
    int mat_id = materials.x;
    if (materials.y != materials.x) {
        int mat_b = materials.y & ((1 << int(MATERIAL_ID_BITS)) - 1);
        int mask = materials.y >> int(MATERIAL_ID_BITS);
        float to_b = mix(mix(0.0, float(mask >> 1 & 1), quad_uv.x), mix(float(mask >> 2 & 1), float(mask >> 3 & 1), quad_uv.x), quad_uv.y);
        float to_high = mat_b > mat_id ? to_b : 1.0 - to_b;
        float jitter = shade_noise(p, cell_m / 1.7, vec3(0.0)) - 0.5;
        mat_id = to_high + 0.6 * jitter > 0.5 ? max(mat_id, mat_b) : min(mat_id, mat_b);
    }
    return uint(mat_id);
}

// A material's shading: the ground's procedural albedo (computed once by the caller), or its parameters from
// material_tex and its pattern. p: the fragment's position (metres, render space; the patterns read it through the
// shading frame), n its normal (interpolated, not
// normalised), up: the base surface's up; footprint: metres per pixel there (patterns fade out below a few pixels per
// feature); flat_w and shade: the procedural ground's slope weight (1 on gentle slopes) and shading (detail, contours),
// which PATTERN_GROUND (a biome's ground) applies to its own colours; time: seconds (lava drifts); sway_m: the wind's
// sway amplitude (0: still air, the leaves don't flutter).
void material_shading(uint mat_id, vec3 ground, vec3 p, vec3 n, vec3 up, float footprint, float flat_w, vec3 shade, float time,
        float sway_m, out vec3 albedo, out float roughness, out vec3 emission) {
    albedo = ground;
    roughness = 1.0;
    emission = vec3(0.0);
    if (mat_id == MATERIAL_GROUND) {
        return;
    }
    ivec2 mt = material_texel(mat_id);
    vec4 m0 = texelFetch(material_tex, mt, 0);
    vec4 m1 = texelFetch(material_tex, mt + ivec2(1, 0), 0);
    albedo = m0.rgb;
    roughness = m0.a;
    emission = m1.rgb;
    if (m1.a == MATERIAL_PATTERN_GROUND) {
        albedo = mix(m1.rgb, m0.rgb, flat_w) * shade; // m1.rgb: its steep albedo (no emission)
        emission = vec3(0.0);
    } else if (m1.a == MATERIAL_PATTERN_LAVA) {
        // Lava: a warped crust, dark seams over a glow that drifts slowly. The crust is q = 0.15 xz warped by a and b
        // (xz: the up's tangent plane, up_frame); each wave of q is written as a wave of the position, snapped in the
        // shading frame (period: its 60 m wave).
        float period_m;
        vec3 t1, t2;
        up_frame(up, t1, t2);
        vec3 sp = shade_frame(p, 60.0, period_m);
        vec2 s = vec2(dot(sp, t1), dot(sp, t2));
        vec3 kx = shade_wave(0.15 * vec3(1.3, 2.1, 3.3), period_m); // q.x's waves: the warp b, the seams, the cross seams
        vec3 kz = shade_wave(0.15 * vec3(1.7, 1.9, 3.3), period_m); // q.y's: the warp a, the seams, the cross seams
        vec2 kp = shade_wave(0.15 * vec3(0.7, 0.9, 0.0), period_m).xy; // the pulse's
        float a = sin(s.y * kz.x + time * 0.15);
        float b = sin(s.x * kx.x - time * 0.11);
        float seam = abs(sin(s.x * kx.y + 2.1 * a) * sin(s.y * kz.y + 1.9 * b) + 0.5 * sin(s.x * kx.z - s.y * kz.z + 3.3 * (a - b)));
        float pulse = 0.75 + 0.25 * sin(s.x * kp.x + s.y * kp.y + 0.7 * a + 0.9 * b + time * 0.3);
        emission = mix(m1.rgb * 0.11, m1.rgb * pulse, smoothstep(0.03, 0.18, seam));
    } else if (m1.a == MATERIAL_PATTERN_BARK) {
        // Bark: dark grooves, lighter plates, knots darker with rings. Past ~4 pixels per groove (11 cm) the detail
        // fades to the plates' mean.
        float detail = 1.0 - smoothstep(0.012, 0.03, footprint);
        float plate = 0.65;
        float knot_weight = 0.0;
        vec3 knot_albedo = vec3(0.0);
        if (detail > 0.0) { // else the plates' mean, no bark_grain
            float knot, ring;
            plate = mix(plate, bark_plate(bark_grain(bark_point(p, n, up), knot, ring)), detail);
            knot_weight = max(knot, ring * 0.6) * detail;
            knot_albedo = m0.rgb * mix(0.35, 0.7, ring);
        }
        albedo = mix(mix(m0.rgb * 0.5, m0.rgb * 1.4, plate), knot_albedo, knot_weight);
    } else if (m1.a == MATERIAL_PATTERN_LEAVES) {
        // Leaves: clumps of ~0.5 m, each a shade of the material's green from dark to yellowish, with ~8 cm leaves
        // inside them that fade out past 2 pixels per leaf; the underside and the sides of a canopy are darker
        // than its top (the sky light that a real canopy gets more of from above than SSAO shows).
        float clump = shade_noise(p, 0.55, vec3(0.0));
        float leaf_w = 1.0 - smoothstep(0.02, 0.05, footprint);
        float leaf = 0.0;
        if (leaf_w > 0.0) {
            // The leaves flutter: their noise drifts with the wind's wobble (a shimmer over the swaying surface). One
            // frame for both (period: the wobble's 2.1 m wave).
            float period_m;
            vec3 s = shade_frame(p, 2.1, period_m);
            vec3 flutter = sway_m > 0.0 ? 0.4 * sin(time * vec3(2.9, 2.3, 3.4) + s.yzx * shade_wave(vec3(3.0), period_m)) : vec3(0.0);
            vec3 cells = shade_cells(vec3(1.0 / 0.08), period_m);
            leaf = value_noise(s * (cells / period_m) + 3.0 + flutter, cells) - 0.5;
        }
        vec3 shade_color = mix(m0.rgb * vec3(0.55, 0.6, 0.6), m0.rgb * vec3(1.45, 1.3, 0.8), clump);
        float up_w = 0.75 + 0.25 * clamp(dot(n, up) * inversesqrt(max(dot(n, n), 1e-8)), 0.0, 1.0);
        albedo = shade_color * (1.0 + leaf_w * 0.45 * leaf) * up_w;
    }
}
)glsl";

// The text to splice after the host's declarations: the lattice hash (pcg3d), the shading frame and the functions above.
inline ov::str glsl() {
    return ov::str(GLSL_PCG3D) + ShadeFrame::glsl() + GLSL;
}

} // namespace PageShading
