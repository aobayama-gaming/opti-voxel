#include "godot_page_host.hpp"

#include "exact_noise_glsl.hpp"
#include "godot_callable.hpp"
#include "godot_conv.hpp"
#include "page_format.hpp"
#include "quad_page_pool.hpp"
#include "voxel_sky.hpp"

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>

#include <cstring>

using namespace godot;

static constexpr const char PAGE_SHADER[] = R"gdshader(
shader_type spatial;
// Corners are chunk-local (page_format.hpp): world positions are rebuilt here from the group
// header's chunk id; the MultiMesh instance transforms are unused (left zero), so transform to
// view space ourselves.
render_mode skip_vertex_transform, cull_back;

// (RenderingServer::shader_set_code does not run the preprocessor: no #define here.)
/*GLSL_PAGE_DECODE*/
// The cloud layer's uniforms and functions (VoxelSky::glsl_clouds, the same text as the sky shader's): the sun's
// light in light() is cut where a cloud lies between the fragment and the sun.
/*GLSL_CLOUDS*/

uniform usampler2D quad_tex : filter_nearest, repeat_disable;
// The chunks' palettes (palette_pool.hpp) and the materials' parameters (Material::Library::make_image).
uniform usampler2D palette_tex : filter_nearest, repeat_disable;
uniform sampler2D material_tex : filter_nearest, repeat_disable;
// 0 shaded, 1 normals, 2 lod, 3 page, 4 shaded + quad wireframe, 5 chunk hash,
// 6 group (INSTANCE_ID), 7 quad index within the group (VERTEX_ID / 4), 8 faceted, 9 material, 10 / 11 bark seam
// probe (bench.gd --probe), 12 biome (the materials' debug colours), 13 coverage (the quads a stalled chunk hides are
// drawn magenta, those it still draws orange)
uniform int debug_view = 0;
uniform float voxel_size = 0.5;
// Sea level in metres above the world origin: below -1e29, no height colours (VoxelTerrain sets it for
// the continents preset).
uniform float sea_level_m = -1e30;
// The base surface's up (BaseSurface::up_dir), which the slope colours read: +y on the flat world. A sphere base will
// need it per vertex.
uniform vec3 base_up = vec3(0.0, 1.0, 0.0);
// World origin in chunk-id units, low 32 bits: positions are drawn relative to it.
uniform ivec3 origin_chunk_lo = ivec3(0);
uniform int chunk_size = 32;
uniform int skirt_size = 2;
uniform float pos_scale = 256.0; // PageFormat::pos_scale(chunk_size), set by the mesher
// Skirt depth per LOD (metres, SDFBase::skirt_depth_m): v_height adds the skirt push back, so the height
// colours see the unpushed surface and the dip along chunk borders keeps the colour around it.
uniform float skirt_depth_m[32];
instance uniform int page_id = 0;

varying vec3 v_world_normal;
varying vec2 v_quad_uv;
varying flat float v_lod;
varying flat vec3 v_chunk;
varying flat ivec2 v_ids; // INSTANCE_ID, quad index | coverage << QUAD_INDEX_BITS (0 drawn, 1 stalled, 2 hidden)
varying float v_height; // metres above the world origin (shaded view: contour lines)
// The quad's two materials, from the provoking corner A: x = A's (global id), y = the other one's | its corner mask
// << MATERIAL_ID_BITS (bit k, k = 1..3: corner k is not A's material). The fragment's weight is the mask interpolated
// over v_quad_uv, so the blend adds no interpolant: triangles are smaller than a pixel here, every attribute the
// fragment reads costs setup per triangle.
varying flat ivec2 v_materials;
varying vec3 v_world_pos; // the fragment's position (metres, render space), set by fragment() for light()

vec3 hash3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

// A lattice cell's hash in [0, 1)^3 (c: the cell's integer coordinates, as floats from floor). Integer arithmetic
// (PCG3D), so every neighbour of a cell reads the same value. A float hash of the coordinate did not: the compiler may
// reassociate (i + 1) * k, and at coordinates in the thousands the hash turned that last bit into another value, a
// seam along every lattice plane (the bark's horizontal lines, tools/bark_seams.sh).
vec3 cell_hash(vec3 c, uint salt) {
    return vec3(pcg3d(uvec3(ivec3(c)) ^ uvec3(salt)) >> 8u) * (1.0 / 16777216.0);
}

// Trilinear value noise in [0, 1], one lattice cell per unit.
float value_noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    float x00 = mix(cell_hash(i, 0u).x, cell_hash(i + vec3(1, 0, 0), 0u).x, f.x);
    float x10 = mix(cell_hash(i + vec3(0, 1, 0), 0u).x, cell_hash(i + vec3(1, 1, 0), 0u).x, f.x);
    float x01 = mix(cell_hash(i + vec3(0, 0, 1), 0u).x, cell_hash(i + vec3(1, 0, 1), 0u).x, f.x);
    float x11 = mix(cell_hash(i + vec3(0, 1, 1), 0u).x, cell_hash(i + vec3(1, 1, 1), 0u).x, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Bark (PATTERN_BARK) as a 3D noise, so it needs no surface frame: noise stretched along world up makes plates and V
// grooves that run up a trunk and wander slowly; a sparse cell field puts knots where a kept cell's point lies near
// the surface, and the grain bends around them. Its axis is world up: a near-horizontal branch gets grooves around it
// (the branch's axis is not known here). p: bark_point's, in metres. Returns the grain, a smooth noise in [0, 1]
// (bark_plate makes the plates and grooves of it); knot: the knot's core, ring: its growth rings, in [0, 1]. Colour
// only: a bump from it (and a roughness following it) cost ~1.3 ms of depth pre-pass in the forest at 1080p (the
// pre-pass writes normals and roughness for SSAO / SSIL), for relief barely visible past a few metres.
float bark_grain(vec3 p, out float knot, out float ring) {
    const vec3 KNOT_CELL = vec3(0.7, 1.6, 0.7);
    // The 2 x 2 x 2 cells nearest p: a knot point lies in the middle 70% of its cell, and a knot reaches ~0.4 m.
    vec3 g = floor(p / KNOT_CELL - 0.5);
    vec3 bend = vec3(0.0);
    float nearest = 1e9;
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                vec3 c = g + vec3(float(x), float(y), float(z));
                if (cell_hash(c, 1u).x > 0.35) {
                    continue; // about a third of the cells hold a knot
                }
                vec3 dk = p - (c + 0.15 + 0.7 * cell_hash(c, 2u)) * KNOT_CELL;
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
    const vec3 SCALE = vec3(9.0, 9.0 / 6.0, 9.0);
    vec3 wander = vec3(1.6, 0.0, 1.6) * value_noise(p * vec3(1.2, 0.25, 1.2));
    return value_noise((p + bend) * SCALE + wander);
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
// reads one column of it.
const float BARK_RADIUS = 0.45;
vec3 bark_point(vec3 p, vec3 n) {
    vec2 u = n.xz * (BARK_RADIUS * inversesqrt(max(dot(n.xz, n.xz), 1e-8)));
    return vec3(u.x, p.y, u.y);
}

// The ground's surface detail, a multiplier on its albedo (the procedural ground's and a biome's PATTERN_GROUND):
// a slow tint over ~20 m between a cooler and a warmer shade, patches of ~2 m, and a grain of ~6 cm that fades out
// past 2 pixels per feature. Without it a flat biome is one colour to the horizon. World-space noise: the same at
// every LOD. p: the fragment's position (metres), footprint: metres per pixel there.
vec3 ground_detail(vec3 p, float footprint) {
    float macro = value_noise(p * (1.0 / 19.0));
    vec3 d = mix(vec3(0.86, 0.88, 0.94), vec3(1.10, 1.06, 0.98), macro);
    d *= 0.92 + 0.16 * value_noise(p * (1.0 / 2.3) + 7.0);
    float grain_w = 1.0 - smoothstep(0.015, 0.04, footprint);
    if (grain_w > 0.0) {
        d *= 1.0 + grain_w * 0.18 * (value_noise(p * (1.0 / 0.06)) - 0.5);
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

// A material's shading: the ground's procedural albedo (computed once by the caller), or its parameters from
// material_tex and its pattern. p: the fragment's position (metres, render space), n its normal (interpolated, not
// normalised); footprint: metres per pixel there (patterns fade out below a few pixels per feature); flat_w and shade:
// the procedural ground's slope weight (1 on gentle slopes) and contour shading, which PATTERN_GROUND (a biome's
// ground) applies to its own colours.
void material_shading(uint mat_id, vec3 ground, vec3 p, vec3 n, float footprint, float flat_w, vec3 shade, out vec3 albedo,
        out float roughness, out vec3 emission) {
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
        // Lava: a warped crust, dark seams over a glow that drifts slowly.
        vec2 q = p.xz * 0.15;
        q += vec2(sin(q.y * 1.7 + TIME * 0.15), sin(q.x * 1.3 - TIME * 0.11));
        float seam = abs(sin(q.x * 2.1) * sin(q.y * 1.9) + 0.5 * sin((q.x - q.y) * 3.3));
        float pulse = 0.75 + 0.25 * sin(q.x * 0.7 + q.y * 0.9 + TIME * 0.3);
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
            plate = mix(plate, bark_plate(bark_grain(bark_point(p, n), knot, ring)), detail);
            knot_weight = max(knot, ring * 0.6) * detail;
            knot_albedo = m0.rgb * mix(0.35, 0.7, ring);
        }
        albedo = mix(mix(m0.rgb * 0.5, m0.rgb * 1.4, plate), knot_albedo, knot_weight);
    } else if (m1.a == MATERIAL_PATTERN_LEAVES) {
        // Leaves: clumps of ~0.5 m, each a shade of the material's green from dark to yellowish, with ~8 cm leaves
        // inside them that fade out past 2 pixels per leaf; the underside and the sides of a canopy are darker
        // than its top (the sky light that a real canopy gets more of from above than SSAO shows).
        float clump = value_noise(p * (1.0 / 0.55));
        float leaf_w = 1.0 - smoothstep(0.02, 0.05, footprint);
        // The leaves flutter: their noise drifts with the wind's wobble (a shimmer over the swaying surface).
        vec3 flutter = wind_sway_m > 0.0 ? 0.4 * sin(TIME * vec3(2.9, 2.3, 3.4) + p.yzx * 3.0) : vec3(0.0);
        float leaf = leaf_w > 0.0 ? value_noise(p * (1.0 / 0.08) + 3.0 + flutter) - 0.5 : 0.0;
        vec3 shade_color = mix(m0.rgb * vec3(0.55, 0.6, 0.6), m0.rgb * vec3(1.45, 1.3, 0.8), clump);
        float up_w = 0.75 + 0.25 * clamp(dot(n, base_up) * inversesqrt(max(dot(n, n), 1e-8)), 0.0, 1.0);
        albedo = shade_color * (1.0 + leaf_w * 0.45 * leaf) * up_w;
    }
}

void vertex() {
    int corner = VERTEX_ID % 4; // base quad corners A, B, C, D
    uvec4 h0 = texelFetch(quad_tex, ivec2(0, INSTANCE_ID), 0);
    uvec4 h1 = texelFetch(quad_tex, ivec2(1, INSTANCE_ID), 0);
    uvec4 c = texelFetch(quad_tex, ivec2(int(HEADER_TEXELS) + VERTEX_ID, INSTANCE_ID), 0);
    ivec3 chunk = page_header_chunk(h0, h1);
    int lod = page_chunk_lod(chunk);
    float voxel_dim = float(chunk_size - 2 * skirt_size);
    ivec3 rel = chunk - origin_chunk_lo;
    vec3 chunk_min = page_chunk_min(rel, lod, voxel_dim, voxel_size);
    vec3 local = page_decode_local(c, pos_scale);
    vec3 world = page_corner_world(chunk_min, lod, local, skirt_size, voxel_dim, voxel_size);
    // Wind: leaf corners sway (wind_sway, a pure function of the position: shared corners agree). Only up to LOD 2:
    // farther the amplitude is under a pixel, and the corner's material costs a palette and a material fetch.
    uint palette_index = page_decode_palette(c);
    if (palette_index != 0u && lod <= 2 && wind_sway_m > 0.0
            && texelFetch(material_tex, material_texel(corner_material(h1, palette_index)) + ivec2(1, 0), 0).a == MATERIAL_PATTERN_LEAVES) {
        world += wind_sway(world);
    }
    // Coverage (page_format.hpp, header texel 2): a quad whose 4 corners all lie in hidden sub-cubes collapses to one
    // point, so a cut overlaps the chunks replacing it instead of leaving a gap. Every corner of the quad decides alike.
    // Only the rows flagged in texel 1 read texel 2.
    int coverage = 0;
    if (page_header_has_coverage(h1)) {
        uvec4 h2 = texelFetch(quad_tex, ivec2(int(PAGE_COVERAGE_TEXEL), INSTANCE_ID), 0);
        int first = int(HEADER_TEXELS) + (VERTEX_ID & ~3);
        bool hidden = true;
        for (int k = 0; k < 4; k++) {
            vec3 lk = page_decode_local(texelFetch(quad_tex, ivec2(first + k, INSTANCE_ID), 0), pos_scale);
            hidden = hidden && page_subcube_hidden(h2, page_corner_subcube(lk, skirt_size, voxel_dim));
        }
        coverage = hidden ? 2 : 1;
        if (hidden && debug_view != 13) {
            world = chunk_min;
        }
    }
    vec3 n = c != uvec4(0u) ? page_decode_normal(c) : vec3(0.0, 1.0, 0.0);
    // The quad's materials: a = corner A's, b = the first other one in B, C, D (a when uniform), and which corners
    // are not a. Flat values come from the provoking corner, A in both triangles: only A reads the other corners (the
    // quad's texels are consecutive) and goes through the palette, the others' values are discarded. A third
    // material in one quad (three meeting inside one cell) draws as b.
    v_materials = ivec2(0);
    if (corner == 0) {
        uint ia = palette_index;
        uint ib = ia;
        uint mask = 0u;
        for (int k = 1; k < 4; k++) {
            uint ik = page_decode_palette(texelFetch(quad_tex, ivec2(int(HEADER_TEXELS) + VERTEX_ID + k, INSTANCE_ID), 0));
            ib = ib == ia ? ik : ib;
            mask |= ik != ia ? 1u << uint(k) : 0u;
        }
        uint mat_a = corner_material(h1, ia);
        uint mat_b = ib == ia ? mat_a : corner_material(h1, ib);
        v_materials = ivec2(int(mat_a), int(mat_b | mask << MATERIAL_ID_BITS));
    }

    v_world_normal = n;
    v_ids = ivec2(INSTANCE_ID, VERTEX_ID / 4 | coverage << int(QUAD_INDEX_BITS));
    v_quad_uv = vec2(float(corner & 1), float(corner >> 1));
    v_lod = float(lod);
    v_chunk = vec3(rel);
    // The cascade skirt pushed the SDF towards air by clamp(-inside) * depth past the chunk bounds, which lowers
    // the surface by push / n.y. The push is linear inside a cell, so on a planar surface the vertex lies
    // on the pushed plane and this gives back the exact height.
    float lo = float(skirt_size);
    vec3 in3 = min(local - lo, lo + voxel_dim - local);
    float push = clamp(-min(in3.x, min(in3.y, in3.z)), 0.0, 1.0) * skirt_depth_m[min(lod, 31)];
    v_height = world.y + push / max(n.y, 0.2);

    VERTEX = (VIEW_MATRIX * vec4(world, 1.0)).xyz;
    NORMAL = normalize(mat3(VIEW_MATRIX) * n);
}

void fragment() {
    // Shaded: grass on gentle slopes, rock on steep ones, faint contour lines every 100 m (mid tones, so
    // the tonemapper keeps the relief readable).
    float up = clamp(dot(v_world_normal, base_up), 0.0, 1.0);
    float flat_w = smoothstep(0.8, 0.95, up);
    vec3 albedo = mix(GROUND_STEEP, GROUND_FLAT, flat_w); // Material::GROUND_STEEP / GROUND_FLAT
    float contour_m = 100.0;
    if (sea_level_m > -1e29) {
        // Height colours (dark albedos: the scene's tonemapper brightens them): sea floor blue, darker with
        // depth, beach, lowland green, highland brown, snow; rock on steep land.
        float e = v_height - sea_level_m;
        vec3 steep = vec3(0.20, 0.18, 0.16);
        if (e < 0.0) {
            albedo = mix(vec3(0.005, 0.02, 0.10), vec3(0.04, 0.16, 0.35), smoothstep(-4000.0, 0.0, e));
        } else if (e < 30.0) {
            albedo = vec3(0.45, 0.40, 0.26);
        } else {
            vec3 land = mix(vec3(0.08, 0.20, 0.05), vec3(0.24, 0.18, 0.10), smoothstep(200.0, 1800.0, e));
            land = mix(land, vec3(0.80), smoothstep(2800.0, 3600.0, e));
            albedo = mix(steep, land, smoothstep(0.7, 0.9, up));
        }
        contour_m = 500.0;
    }
    float h = v_height / contour_m;
    float fw = max(fwidth(h), 1e-6);
    float line = abs(fract(h + 0.5) - 0.5) / fw;
    // Contours fade out where they would be denser than ~10 pixels apart.
    float contour = 1.0 - smoothstep(0.03, 0.1, fw);
    vec3 p = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
    v_world_pos = p;
    float footprint = -VERTEX.z * 2.0 / (PROJECTION_MATRIX[1][1] * VIEWPORT_SIZE.y); // metres per pixel
    vec3 shade = ground_detail(p, footprint) * mix(1.0 - 0.35 * contour, 1.0, smoothstep(0.0, 1.5, line));
    albedo *= shade;
    vec3 ground = albedo;
    // Two materials: one is shaded per fragment, the higher id where the weight towards it (the corner mask over the
    // quad), jittered by a noise of about a cell (of this LOD), passes 1/2, so the border wanders instead of following
    // the cell grid. Towards the higher id, not towards b: two quads sharing an edge may hold the pair in either
    // order, and must agree on the border.
    int mat_id = v_materials.x;
    if (v_materials.y != v_materials.x) {
        int mat_b = v_materials.y & ((1 << int(MATERIAL_ID_BITS)) - 1);
        int mask = v_materials.y >> int(MATERIAL_ID_BITS);
        float to_b = mix(mix(0.0, float(mask >> 1 & 1), v_quad_uv.x), mix(float(mask >> 2 & 1), float(mask >> 3 & 1), v_quad_uv.x),
                v_quad_uv.y);
        float to_high = mat_b > mat_id ? to_b : 1.0 - to_b;
        float jitter = value_noise(p / (voxel_size * exp2(v_lod)) * 1.7) - 0.5;
        mat_id = to_high + 0.6 * jitter > 0.5 ? max(mat_id, mat_b) : min(mat_id, mat_b);
    }
    float roughness;
    vec3 emission;
    material_shading(uint(mat_id), ground, p, v_world_normal, footprint, flat_w, shade, albedo, roughness, emission);
    if (debug_view == 1) {
        albedo = vec3(0.0);
        emission = v_world_normal * 0.5 + 0.5;
    } else if (debug_view == 2) {
        albedo = vec3(0.0);
        emission = hash3(vec3(v_lod * 7.0 + 1.0)) * 0.8 + 0.2;
    } else if (debug_view == 3) {
        albedo = vec3(0.0);
        emission = hash3(vec3(float(page_id) + 0.5)) * 0.8 + 0.2;
    } else if (debug_view == 5) {
        albedo = vec3(0.0);
        emission = hash3(v_chunk * 0.37 + v_lod * 13.0) * 0.8 + 0.2;
    } else if (debug_view == 6) {
        albedo = vec3(0.0);
        emission = hash3(vec3(float(v_ids.x) + 0.5)) * 0.8 + 0.2;
    } else if (debug_view == 7) {
        albedo = vec3(0.0);
        float quad = float(v_ids.y & int(QUAD_GROUP - 1u));
        emission = vec3(quad / 63.0, 0.2, 1.0 - quad / 63.0);
    } else if (debug_view == 8) {
        // Faceted: one clay colour, each triangle lit by its own plane (the vertex normals, shared across a
        // crease, smear the shading of sharp edges).
        albedo = vec3(0.5);
        emission = vec3(0.0);
        NORMAL = normalize(cross(dFdy(VERTEX), dFdx(VERTEX)));
    } else if (debug_view == 9) {
        albedo = vec3(0.0);
        emission = uint(mat_id) == MATERIAL_GROUND ? vec3(0.15) : hash3(vec3(float(mat_id) * 0.61 + 0.5)) * 0.8 + 0.2;
    } else if (debug_view == 12) {
        // Biomes: each material's debug colour (a biome's ground carries its biome's), lit and with the contours so the
        // relief stays readable; dark grey for the rest (instances, the ground without a climate).
        vec4 dbg = uint(mat_id) == MATERIAL_GROUND ? vec4(0.0) : texelFetch(material_tex, material_texel(uint(mat_id)) + ivec2(MATERIAL_TEXEL_DEBUG, 0), 0);
        albedo = (dbg.a > 0.0 ? dbg.rgb : vec3(0.06)) * shade;
        emission = albedo * 0.3;
        roughness = 1.0;
    } else if (debug_view == 13) {
        int coverage = v_ids.y >> int(QUAD_INDEX_BITS);
        albedo = coverage == 2 ? vec3(0.9, 0.1, 0.9) : (coverage == 1 ? vec3(1.0, 0.5, 0.0) : albedo);
    } else if (debug_view == 10 || debug_view == 11) {
        // Bark seam probe (bench.gd --probe, read by tools/bark_seams.py): bytes, written through the inverse of the
        // sRGB encoding (the bench sets a linear tonemapper, no grade, glow or fog; SPECULAR 0 below), so each
        // channel reads back as the byte. Bark fragments at full detail only, black elsewhere. 10: the grain (smooth, so
        // a jump in it is a seam; the plates' V grooves step like one where they are a few pixels wide), a hash of the
        // chunk, 8 * LOD + 4; 11: log2 of the depth in 12 bits (6 per channel, times 4), the normal's azimuth.
        albedo = vec3(0.0);
        emission = vec3(0.0);
        bool bark = uint(mat_id) != MATERIAL_GROUND
                && texelFetch(material_tex, material_texel(uint(mat_id)) + ivec2(1, 0), 0).a == MATERIAL_PATTERN_BARK;
        if (bark && footprint < 0.012) {
            vec3 bytes;
            if (debug_view == 10) {
                float knot, ring;
                float grain = bark_grain(bark_point(p, v_world_normal), knot, ring);
                bytes = vec3(round(grain * 255.0), 4.0 * floor(cell_hash(v_chunk, uint(v_lod)).x * 64.0),
                        8.0 * v_lod + 4.0);
            } else {
                float depth = floor(clamp((log2(-VERTEX.z) + 4.0) / 16.0, 0.0, 1.0) * 4095.0);
                float azimuth = atan(v_world_normal.z, v_world_normal.x) / 6.2831853 + 0.5;
                bytes = vec3(4.0 * floor(depth / 64.0), 4.0 * mod(depth, 64.0), round(azimuth * 255.0));
            }
            vec3 c = bytes / 255.0;
            emission = mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
        }
    }
    if (debug_view == 4) {
        float d = min(min(v_quad_uv.x, 1.0 - v_quad_uv.x), min(v_quad_uv.y, 1.0 - v_quad_uv.y));
        if (d < fwidth(d) * 1.2) {
            albedo = vec3(0.05);
        }
    }
    ALBEDO = albedo;
    EMISSION = emission;
    ROUGHNESS = roughness;
    if (debug_view == 10 || debug_view == 11) {
        SPECULAR = 0.0; // the probe's bytes are its emission only
    }
}

// Lambert (Godot's default diffuse: LIGHT_COLOR carries the PI; the albedo is applied after), with the sun cut by
// the cloud layer where the fragment's ray to the sun crosses a cloud: the shadow of the cloud the sky shader draws
// there. The terrain's materials have a roughness of 1, so the specular lobe is left out.
void light() {
    float sun = 1.0;
    if (LIGHT_IS_DIRECTIONAL && cloud_shadow > 0.0) {
        vec3 to_sun = mat3(INV_VIEW_MATRIX) * LIGHT;
        if (to_sun.y > 0.05) { // VERTEX is not readable in light(): the fragment's position comes through v_world_pos
            sun = 1.0 - cloud_shadow * cloud_density(texture(cloud_tex, cloud_uv(cloud_hit(v_world_pos, to_sun))).r);
        }
    }
    DIFFUSE_LIGHT += clamp(dot(NORMAL, LIGHT), 0.0, 1.0) * ATTENUATION * LIGHT_COLOR * sun;
}
)gdshader";

GodotPageHost::GodotPageHost(RID p_scenario, RDDevice *p_device) :
        scenario(p_scenario), device(p_device) {
    RenderingServer *rs = RenderingServer::get_singleton();
    // Shared base mesh: QUAD_GROUP quads, corners A B C D = vertices 4q+0..3, triangles A,B,D / A,D,C
    // (== the legacy emit_quad positive winding). Positions are placeholders: the shader ignores them.
    PackedVector3Array vertices;
    PackedInt32Array indices;
    for (uint32_t q = 0; q < QuadPagePool::QUAD_GROUP; q++) {
        vertices.push_back(Vector3(0, 0, q));
        vertices.push_back(Vector3(1, 0, q));
        vertices.push_back(Vector3(0, 1, q));
        vertices.push_back(Vector3(1, 1, q));
        for (int i : { 0, 1, 3, 0, 3, 2 }) {
            indices.push_back(q * 4 + i);
        }
    }
    Array arrays;
    arrays.resize(Mesh::ARRAY_MAX);
    arrays[Mesh::ARRAY_VERTEX] = vertices;
    arrays[Mesh::ARRAY_INDEX] = indices;
    base_mesh = rs->mesh_create();
    rs->mesh_add_surface_from_arrays(base_mesh, RenderingServer::PRIMITIVE_TRIANGLES, arrays);

    shader = rs->shader_create();
    rs->shader_set_code(shader,
            String(PAGE_SHADER)
                    .replace("/*GLSL_PAGE_DECODE*/", godot_str(PageFormat::glsl_decode() + Material::glsl_constants() + GLSL_PCG3D))
                    .replace("/*GLSL_CLOUDS*/", VoxelSky::glsl_clouds()));
}

void GodotPageHost::destroy() {
    RenderingServer *rs = RenderingServer::get_singleton();
    rs->free_rid(shader);
    rs->free_rid(base_mesh);
    device->post([this]() { delete this; });
}

void GodotPageHost::set_material_param(const StringName &p_name, const Variant &p_value) {
    material_params[p_name] = p_value;
    for (const Page &page : pages) {
        if (page.material.is_valid()) {
            RenderingServer::get_singleton()->material_set_param(page.material, p_name, p_value);
        }
    }
}

void GodotPageHost::create_page(int p_page, uint32_t p_rows) {
    RenderingServer *rs = RenderingServer::get_singleton();
    Page page;
    page.material = rs->material_create();
    rs->material_set_shader(page.material, shader);
    const Array keys = material_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        rs->material_set_param(page.material, keys[i], material_params[keys[i]]);
    }

    page.multimesh = rs->multimesh_create();
    // Order matters: the indirect command buffer is only created by set_mesh when the data was
    // allocated with use_indirect first.
    rs->multimesh_allocate_data(page.multimesh, int32_t(p_rows), RenderingServer::MULTIMESH_TRANSFORM_2D, false, false, true);
    rs->multimesh_set_mesh(page.multimesh, base_mesh);
    // Without CPU data the engine cannot compute bounds; set by the user of the page.
    rs->multimesh_set_custom_aabb(page.multimesh, AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)));

    page.instance = rs->instance_create2(page.multimesh, scenario);
    rs->instance_geometry_set_material_override(page.instance, page.material);
    rs->instance_set_visible(page.instance, false);
    rs->instance_geometry_set_shader_parameter(page.instance, "page_id", p_page);

    if (int(pages.size()) <= p_page) {
        pages.resize(size_t(p_page) + 1);
    }
    pages[p_page] = page;
    // The render thread's copy, before the core's work on the page (FIFO).
    device->post([this, p_page, page]() {
        if (int(rt_pages.size()) <= p_page) {
            rt_pages.resize(size_t(p_page) + 1);
        }
        rt_pages[p_page].multimesh = page.multimesh;
        rt_pages[p_page].material = page.material;
    });
}

void GodotPageHost::free_page(int p_page) {
    RenderingServer *rs = RenderingServer::get_singleton();
    Page &page = pages[p_page];
    rs->free_rid(page.instance);
    rs->free_rid(page.multimesh);
    rs->free_rid(page.material);
    page = Page();
}

void GodotPageHost::set_page_visible(int p_page, bool p_visible) {
    RenderingServer::get_singleton()->instance_set_visible(pages[p_page].instance, p_visible);
}

void GodotPageHost::set_page_bounds(int p_page, const RenderBox &p_bounds) {
    RenderingServer::get_singleton()->multimesh_set_custom_aabb(pages[p_page].multimesh, AABB(godot_v3(p_bounds.position), godot_v3(p_bounds.size)));
}

void GodotPageHost::set_decode(const PageDecode &p_decode) {
    set_material_param("voxel_size", p_decode.voxel_size);
    set_material_param("chunk_size", p_decode.chunk_size);
    set_material_param("skirt_size", p_decode.skirt_size);
    set_material_param("pos_scale", p_decode.pos_scale);
    set_material_param("origin_chunk_lo", godot_v3i(p_decode.origin_chunk_lo));
}

void GodotPageHost::set_materials(const Material::Library::Image &p_image) {
    PackedByteArray bytes;
    bytes.resize(int64_t(p_image.rgba.size() * sizeof(float)));
    std::memcpy(bytes.ptrw(), p_image.rgba.data(), p_image.rgba.size() * sizeof(float));
    material_texture = ImageTexture::create_from_image(Image::create_from_data(p_image.width, p_image.height, false, Image::FORMAT_RGBAF, bytes));
    set_material_param("material_tex", material_texture->get_rid());
}

void GodotPageHost::set_debug_view(int p_view) {
    set_material_param("debug_view", p_view);
}

ov::gpu::Buffer GodotPageHost::rt_attach_page(int p_page, ov::gpu::Texture p_texture) {
    // On the render thread: these RenderingServer calls execute immediately.
    RenderingServer *rs = RenderingServer::get_singleton();
    PageRT &page = rt_pages[p_page];
    page.rs_texture = rs->texture_rd_create(device->get_rid(p_texture));
    rs->material_set_param(page.material, "quad_tex", page.rs_texture);
    page.command = device->import_buffer(rs->multimesh_get_command_buffer_rd_rid(page.multimesh));
    return page.command;
}

void GodotPageHost::rt_detach_page(int p_page) {
    PageRT &page = rt_pages[p_page];
    if (page.rs_texture.is_valid()) {
        RenderingServer::get_singleton()->free_rid(page.rs_texture);
    }
    device->free(page.command); // the multimesh's (already freed)
    page = PageRT();
}

void GodotPageHost::rt_set_palette_texture(ov::gpu::Texture p_texture) {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rt_palette_texture.is_valid()) {
        rs->free_rid(rt_palette_texture);
        rt_palette_texture = RID();
    }
    if (!p_texture.is_valid()) {
        return;
    }
    rt_palette_texture = rs->texture_rd_create(device->get_rid(p_texture));
    // The pages' materials are set on the main thread.
    const RID texture = rt_palette_texture;
    godot_callable([this, texture]() { set_material_param("palette_tex", texture); }).call_deferred();
}
