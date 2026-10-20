#include "quad_page_pool.hpp"

#include "debug_log.hpp"
#include "material_library.hpp"
#include "world_coord.hpp"

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>

static constexpr const char PAGE_SHADER[] = R"gdshader(
shader_type spatial;
// Corners are chunk-local (page_format.hpp): world positions are rebuilt here from the group
// header's chunk id; the MultiMesh instance transforms are unused (left zero), so transform to
// view space ourselves.
render_mode skip_vertex_transform, cull_back;

// (RenderingServer::shader_set_code does not run the preprocessor: no #define here.)
/*GLSL_PAGE_DECODE*/

uniform usampler2D quad_tex : filter_nearest, repeat_disable;
// The chunks' palettes (palette_pool.hpp) and the materials' parameters (Material::Library::make_image).
uniform usampler2D palette_tex : filter_nearest, repeat_disable;
uniform sampler2D material_tex : filter_nearest, repeat_disable;
// 0 shaded, 1 normals, 2 lod, 3 page, 4 shaded + quad wireframe, 5 chunk hash,
// 6 group (INSTANCE_ID), 7 quad index within the group (VERTEX_ID / 4), 8 faceted, 9 material
uniform int debug_view = 0;
uniform float voxel_size = 0.5;
// Sea level in metres above the world origin: below -1e29, no height colours (VoxelTerrain sets it for
// the continents preset).
uniform float sea_level_m = -1e30;
// World origin in chunk-id units, low 32 bits: positions are drawn relative to it.
uniform ivec3 origin_chunk_lo = ivec3(0);
uniform int chunk_size = 32;
uniform int skirt_size = 2;
uniform float pos_scale = 256.0; // PageFormat::pos_scale(chunk_size), set by the mesher
// Skirt depth per LOD (metres, SDFBase::skirt_depth_m): v_height adds the skirt push back, so the height
// colours see the unpushed surface and the dip along chunk borders keeps the colour around it.
uniform float skirt_depth_m[32];
// An instance's surface at LOD L lies instance_erode_m * (2^L - 1) metres inside its LOD 0 one (the band rule's
// erosion from LOD 1, VoxelTerrain.archetype_erode): solid textures sample that far out, so they don't jump across a
// LOD seam (with --instance_resolve, parts that start eroding later are over-corrected).
uniform float instance_erode_m = 0.0;
instance uniform int page_id = 0;

varying vec3 v_world_normal;
varying vec2 v_quad_uv;
varying flat float v_lod;
varying flat vec3 v_chunk;
varying flat ivec2 v_ids;
varying float v_height; // metres above the world origin (shaded view: contour lines)
// The quad's two materials, from the provoking corner A: x = A's (global id), y = the other one's | its corner mask
// << MATERIAL_ID_BITS (bit k, k = 1..3: corner k is not A's material). The fragment's weight is the mask interpolated
// over v_quad_uv, so the blend adds no interpolant: triangles are smaller than a pixel here, every attribute the
// fragment reads costs setup per triangle.
varying flat ivec2 v_materials;

vec3 hash3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

float hash1(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// Trilinear value noise with its analytic gradient, one lattice cell per unit: (value in [0, 1], d/dp).
vec4 value_noise_d(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 u = f * f * (3.0 - 2.0 * f);
    vec3 du = 6.0 * f * (1.0 - f);
    float a = hash1(i);
    float b = hash1(i + vec3(1, 0, 0));
    float c = hash1(i + vec3(0, 1, 0));
    float d = hash1(i + vec3(1, 1, 0));
    float e = hash1(i + vec3(0, 0, 1));
    float g = hash1(i + vec3(1, 0, 1));
    float h = hash1(i + vec3(0, 1, 1));
    float k = hash1(i + vec3(1, 1, 1));
    float kx = b - a;
    float ky = c - a;
    float kz = e - a;
    float kxy = a - b - c + d;
    float kyz = a - c - e + h;
    float kzx = a - b - e + g;
    float kxyz = -a + b + c - d + e - g - h + k;
    float v = a + kx * u.x + ky * u.y + kz * u.z + kxy * u.x * u.y + kyz * u.y * u.z + kzx * u.z * u.x + kxyz * u.x * u.y * u.z;
    vec3 grad = du * vec3(kx + kxy * u.y + kzx * u.z + kxyz * u.y * u.z, ky + kyz * u.z + kxy * u.x + kxyz * u.z * u.x,
            kz + kzx * u.x + kyz * u.y + kxyz * u.x * u.y);
    return vec4(v, grad);
}

// Bark (PATTERN_BARK) as a solid texture, so it needs no surface frame: noise stretched along world up makes plates
// and V grooves that run up a trunk and wander slowly; a sparse cell field puts knots where a kept cell's point lies
// near the surface, and the grain bends around them. Its axis is world up: a near-horizontal branch gets grooves
// around it (the branch's axis is not known here). p in metres. Returns the plate height in [0, 1] and its gradient
// per metre (the bend's and the wander's ignored); knot: the knot's core, ring: its growth rings, in [0, 1].
vec4 bark_height(vec3 p, out float knot, out float ring) {
    const vec3 KNOT_CELL = vec3(0.7, 1.6, 0.7);
    // The 2 x 2 x 2 cells nearest p: a knot point lies in the middle 70% of its cell, and a knot reaches ~0.4 m.
    vec3 g = floor(p / KNOT_CELL - 0.5);
    vec3 bend = vec3(0.0);
    float nearest = 1e9;
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                vec3 c = g + vec3(float(x), float(y), float(z));
                if (hash1(c + 71.3) > 0.35) {
                    continue; // about a third of the cells hold a knot
                }
                vec3 dk = p - (c + 0.15 + 0.7 * hash3(c + 19.1)) * KNOT_CELL;
                vec3 dn = dk / vec3(0.08, 0.13, 0.08);
                float r2 = dot(dn, dn);
                nearest = min(nearest, r2);
                bend.xz += dk.xz * (1.5 * exp(-0.3 * r2));
            }
        }
    }
    knot = exp(-nearest);
    ring = exp(-0.15 * nearest) * (0.5 + 0.5 * cos(sqrt(nearest) * 7.0));
    // Grooves ~11 cm apart, 6x longer than wide. Coarse on purpose: a solid texture also varies across the surface, and
    // neighbouring chunks' surfaces may differ by ~1 cm (a 4.5 cm groove showed it as a line along chunk seams).
    const vec3 SCALE = vec3(9.0, 9.0 / 6.0, 9.0);
    vec3 wander = vec3(1.6, 0.0, 1.6) * value_noise_d(p * vec3(1.2, 0.25, 1.2)).x;
    vec4 n = value_noise_d((p + bend) * SCALE + wander);
    // Plates with flat tops, V grooves where the noise crosses its middle.
    float s = 2.0 * n.x - 1.0;
    float plate = abs(s) * 1.8;
    vec3 grad = plate < 1.0 ? sign(s) * 3.6 * n.yzw * SCALE : vec3(0.0);
    return vec4(min(plate, 1.0), grad);
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

// A material's shading: the ground's procedural albedo (computed once by the caller), or its parameters from
// material_tex and its pattern. p: the fragment's position (metres, render space), n its normal; footprint: metres
// per pixel there (patterns fade out below a few pixels per feature); lod: the chunk's; bump: added to the world normal.
void material_shading(uint mat_id, vec3 ground, vec3 p, vec3 n, float footprint, float lod, out vec3 albedo,
        out float roughness, out vec3 emission, out vec3 bump) {
    albedo = ground;
    roughness = 1.0;
    emission = vec3(0.0);
    bump = vec3(0.0);
    if (mat_id == MATERIAL_GROUND) {
        return;
    }
    ivec2 mt = ivec2(int(MATERIAL_TEXELS * (mat_id % MATERIALS_PER_ROW)), int(mat_id / MATERIALS_PER_ROW));
    vec4 m0 = texelFetch(material_tex, mt, 0);
    vec4 m1 = texelFetch(material_tex, mt + ivec2(1, 0), 0);
    albedo = m0.rgb;
    roughness = m0.a;
    emission = m1.rgb;
    if (m1.a == MATERIAL_PATTERN_LAVA) {
        // Lava: a warped crust, dark seams over a glow that drifts slowly.
        vec2 q = p.xz * 0.15;
        q += vec2(sin(q.y * 1.7 + TIME * 0.15), sin(q.x * 1.3 - TIME * 0.11));
        float seam = abs(sin(q.x * 2.1) * sin(q.y * 1.9) + 0.5 * sin((q.x - q.y) * 3.3));
        float pulse = 0.75 + 0.25 * sin(q.x * 0.7 + q.y * 0.9 + TIME * 0.3);
        emission = mix(m1.rgb * 0.11, m1.rgb * pulse, smoothstep(0.03, 0.18, seam));
    } else if (m1.a == MATERIAL_PATTERN_BARK) {
        // Bark: dark grooves, lighter plates, knots darker with rings; the grooves rougher and bumped ~1.5 cm deep.
        // Past ~4 pixels per groove (11 cm) the detail fades to the plates' mean.
        float knot, ring;
        vec4 h = bark_height(p + n * (instance_erode_m * (exp2(lod) - 1.0)), knot, ring);
        float detail = 1.0 - smoothstep(0.012, 0.03, footprint);
        float plate = mix(0.65, h.x, detail);
        albedo = mix(m0.rgb * 0.5, m0.rgb * 1.4, plate);
        albedo = mix(albedo, m0.rgb * mix(0.35, 0.7, ring), max(knot, ring * 0.6) * detail);
        roughness = min(m0.a + 0.1 * (1.0 - plate), 1.0);
        bump = -0.015 * detail * (h.yzw - dot(h.yzw, n) * n);
    }
}

void vertex() {
    int corner = VERTEX_ID % 4; // base quad corners A, B, C, D
    uvec4 h0 = texelFetch(quad_tex, ivec2(0, INSTANCE_ID), 0);
    uvec4 h1 = texelFetch(quad_tex, ivec2(1, INSTANCE_ID), 0);
    uvec4 c = texelFetch(quad_tex, ivec2(int(HEADER_TEXELS) + VERTEX_ID, INSTANCE_ID), 0);
    ivec3 chunk = ivec3(int(h0.x | (h0.y << 16u)), int(h0.z | (h0.w << 16u)), int(h1.x | (h1.y << 16u)));

    // ChunkMath::vertices_to_world. Empty / killed rows are all zero: every corner lands on the
    // same point -> zero area.
    int lod = chunk.x == 0 ? 0 : findLSB(chunk.x);
    float voxel_dim = float(chunk_size - 2 * skirt_size);
    float chunk_world_dim = voxel_dim * voxel_size;
    float size = float(1 << lod) * chunk_world_dim;
    // Wrapping int32 difference: exact while the chunk is within 2^31 ids of the origin. The origin is a
    // multiple of the root chunk span, so the low word keeps the LOD bit used above.
    ivec3 rel = chunk - origin_chunk_lo;
    vec3 chunk_min = vec3(rel) * (chunk_world_dim * 0.5) - vec3(size * 0.5);
    vec3 local = page_decode_local(c, pos_scale);
    vec3 world = chunk_min + (local - float(skirt_size)) * size / voxel_dim;
    vec3 n = c != uvec4(0u) ? page_decode_normal(c) : vec3(0.0, 1.0, 0.0);
    // The quad's materials: a = corner A's, b = the first other one in B, C, D (a when uniform), and which corners
    // are not a. Flat values come from the provoking corner, A in both triangles: only A reads the other corners (the
    // quad's texels are consecutive) and goes through the palette, the others' values are discarded. A third
    // material in one quad (three meeting inside one cell) draws as b.
    v_materials = ivec2(0);
    if (corner == 0) {
        uint ia = page_decode_palette(c);
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
    v_ids = ivec2(INSTANCE_ID, VERTEX_ID / 4);
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
    float up = clamp(v_world_normal.y, 0.0, 1.0);
    vec3 rock = vec3(0.36, 0.32, 0.28);
    vec3 albedo = mix(rock, vec3(0.24, 0.36, 0.16), smoothstep(0.8, 0.95, up));
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
    albedo *= mix(1.0 - 0.35 * contour, 1.0, smoothstep(0.0, 1.5, line));
    vec3 ground = albedo;
    vec3 p = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
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
        float jitter = value_noise_d(p / (voxel_size * exp2(v_lod)) * 1.7).x - 0.5;
        mat_id = to_high + 0.6 * jitter > 0.5 ? max(mat_id, mat_b) : min(mat_id, mat_b);
    }
    float roughness;
    vec3 emission;
    float footprint = -VERTEX.z * 2.0 / (PROJECTION_MATRIX[1][1] * VIEWPORT_SIZE.y); // metres per pixel
    vec3 bump;
    material_shading(uint(mat_id), ground, p, normalize(v_world_normal), footprint, v_lod, albedo, roughness, emission, bump);
    NORMAL = normalize(NORMAL + mat3(VIEW_MATRIX) * bump);
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
        emission = vec3(float(v_ids.y) / 63.0, 0.2, 1.0 - float(v_ids.y) / 63.0);
    } else if (debug_view == 8) {
        // Faceted: one clay colour, each triangle lit by its own plane (the vertex normals, shared across a
        // crease, smear the shading of sharp edges).
        albedo = vec3(0.5);
        emission = vec3(0.0);
        NORMAL = normalize(cross(dFdy(VERTEX), dFdx(VERTEX)));
    } else if (debug_view == 9) {
        albedo = vec3(0.0);
        emission = uint(mat_id) == MATERIAL_GROUND ? vec3(0.15) : hash3(vec3(float(mat_id) * 0.61 + 0.5)) * 0.8 + 0.2;
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
}
)gdshader";

void QuadPagePool::initialize(RID p_scenario, uint32_t p_page_capacity, uint32_t p_initial_pages, float p_voxel_size) {
    ERR_FAIL_COND_MSG(p_page_capacity % QUAD_GROUP != 0, "QuadPagePool: page capacity must be a multiple of 64 quads.");
    RenderingServer *rs = RenderingServer::get_singleton();
    scenario = p_scenario;
    page_capacity = p_page_capacity;
    voxel_size = p_voxel_size;

    // Shared base mesh: QUAD_GROUP quads, corners A B C D = vertices 4q+0..3, triangles A,B,D / A,D,C
    // (== the legacy emit_quad positive winding). Positions are placeholders: the shader ignores them.
    PackedVector3Array vertices;
    PackedInt32Array indices;
    for (uint32_t q = 0; q < QUAD_GROUP; q++) {
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
    rs->shader_set_code(shader, String(PAGE_SHADER).replace("/*GLSL_PAGE_DECODE*/", PageFormat::glsl_decode() + Material::glsl_constants()));
    material_params["voxel_size"] = voxel_size;

    for (uint32_t i = 0; i < p_initial_pages; i++) {
        free_pages.push_back(_create_page());
    }
}

int QuadPagePool::_create_page(int p_index) {
    RenderingServer *rs = RenderingServer::get_singleton();
    Page page;
    page.material = rs->material_create();
    rs->material_set_shader(page.material, shader);
    Array keys = material_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        rs->material_set_param(page.material, keys[i], material_params[keys[i]]);
    }
    rs->material_set_param(page.material, "debug_view", debug_view);

    page.multimesh = rs->multimesh_create();
    // Order matters: the indirect command buffer is only created by set_mesh when the data was
    // allocated with use_indirect first.
    rs->multimesh_allocate_data(page.multimesh, page_capacity / QUAD_GROUP, RenderingServer::MULTIMESH_TRANSFORM_2D,
            false, false, true);
    rs->multimesh_set_mesh(page.multimesh, base_mesh);
    // Without CPU data the engine cannot compute bounds; set by the user of the page.
    rs->multimesh_set_custom_aabb(page.multimesh, AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)));

    page.instance = rs->instance_create2(page.multimesh, scenario);
    rs->instance_geometry_set_material_override(page.instance, page.material);
    rs->instance_set_visible(page.instance, false);

    int index = p_index;
    if (index < 0) {
        index = pages.size();
        pages.push_back(page);
    } else {
        pages[index] = page;
    }
    rs->instance_geometry_set_shader_parameter(page.instance, "page_id", index);

    // Texture creation must happen on the render thread; it also zeroes the group count (a fresh
    // page would otherwise draw every group).
    rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_create_page).bind(index, page.multimesh, page.material));
    // A reused index left trimmed_pages before this call.
    pages_alive_peak = MAX(pages_alive_peak, _alive_pages());
    return index;
}

void QuadPagePool::_rt_create_page(int p_page, RID p_multimesh, RID p_material) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    RenderingServer *rs = RenderingServer::get_singleton();

    Ref<RDTextureFormat> format;
    format.instantiate();
    format->set_format(TEX_FORMAT);
    format->set_width(TEX_WIDTH);
    format->set_height(get_page_rows());
    format->set_usage_bits(RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RenderingDevice::TEXTURE_USAGE_CAN_UPDATE_BIT);
    Ref<RDTextureView> view;
    view.instantiate();

    if (!rt_zero_texture.is_valid()) {
        rt_zero_texture = rd->texture_create(format, view);
        rd->texture_clear(rt_zero_texture, Color(0, 0, 0, 0), 0, 1, 0, 1);
    }

    if ((int)rt_pages.size() <= p_page) {
        rt_pages.resize(p_page + 1);
    }
    PageRT &page = rt_pages[p_page];
    page.texture = rd->texture_create(format, view);
    rd->texture_clear(page.texture, Color(0, 0, 0, 0), 0, 1, 0, 1);
    // We are on the render thread: these RenderingServer calls execute immediately.
    page.rs_texture = rs->texture_rd_create(page.texture);
    rs->material_set_param(p_material, "quad_tex", page.rs_texture);
    page.command = rs->multimesh_get_command_buffer_rd_rid(p_multimesh);
    rt_set_group_count(p_page, 0);
}

void QuadPagePool::_rt_reset_page(int p_page) {
    rt_set_group_count(p_page, 0);
}

int QuadPagePool::acquire_page() {
    int page;
    if (free_pages.is_empty()) {
        int reuse = -1;
        if (!trimmed_pages.is_empty()) {
            reuse = trimmed_pages[trimmed_pages.size() - 1];
            trimmed_pages.remove_at(trimmed_pages.size() - 1);
        }
        page = _create_page(reuse);
        growth_events++;
        VLOG(2, "QuadPagePool: grew to ", (int64_t)pages.size(), " pages");
    } else {
        page = free_pages[free_pages.size() - 1];
        free_pages.remove_at(free_pages.size() - 1);
    }
    pages[page].in_use = true;
    pages_in_use++;
    RenderingServer::get_singleton()->instance_set_visible(pages[page].instance, true);
    return page;
}

void QuadPagePool::release_page(int p_page) {
    ERR_FAIL_INDEX(p_page, (int)pages.size());
    ERR_FAIL_COND(!pages[p_page].in_use);
    RenderingServer *rs = RenderingServer::get_singleton();
    pages[p_page].in_use = false;
    pages_in_use--;
    rs->instance_set_visible(pages[p_page].instance, false);
    rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_reset_page).bind(p_page));
    free_pages.push_back(p_page);
}

int QuadPagePool::trim(int p_keep_free, int p_max) {
    RenderingServer *rs = RenderingServer::get_singleton();
    int freed = 0;
    while ((int)free_pages.size() > p_keep_free && freed < p_max) {
        const int index = free_pages[free_pages.size() - 1];
        free_pages.remove_at(free_pages.size() - 1);
        Page &page = pages[index];
        // Render-server frees are queued in order with call_on_render_thread: everything already
        // queued for this page (resets, kills, copies) runs before its texture goes away.
        rs->free_rid(page.instance);
        rs->free_rid(page.multimesh);
        rs->free_rid(page.material);
        page = Page();
        rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_free_page).bind(index));
        trimmed_pages.push_back(index);
        pages_trimmed++;
        freed++;
    }
    return freed;
}

void QuadPagePool::_rt_free_page(int p_page) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    PageRT &page = rt_pages[p_page];
    RenderingServer::get_singleton()->free_rid(page.rs_texture);
    rd->free_rid(page.texture);
    page = PageRT(); // the command buffer belonged to the (already freed) multimesh
}

void QuadPagePool::set_page_aabb(int p_page, const AABB &p_aabb) {
    RenderingServer::get_singleton()->multimesh_set_custom_aabb(pages[p_page].multimesh, p_aabb);
}

void QuadPagePool::set_material_param(const StringName &p_name, const Variant &p_value) {
    material_params[p_name] = p_value;
    for (const Page &page : pages) {
        if (page.material.is_valid()) {
            RenderingServer::get_singleton()->material_set_param(page.material, p_name, p_value);
        }
    }
}

void QuadPagePool::set_debug_view(int p_view) {
    debug_view = p_view;
    set_material_param("debug_view", p_view);
}

QuadPagePool::PageRD QuadPagePool::rt_get_page_rd(int p_page) {
    PageRD out;
    if (p_page < (int)rt_pages.size()) {
        out.texture = rt_pages[p_page].texture;
        out.command = rt_pages[p_page].command;
    }
    return out;
}

void QuadPagePool::rt_set_group_count(int p_page, uint32_t p_groups) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page);
    ERR_FAIL_COND_MSG(!page_rd.command.is_valid(), "QuadPagePool: page has no indirect command buffer.");
    ERR_FAIL_COND_MSG(p_groups > get_page_rows(), vformat("QuadPagePool: group count %d > page rows on page %d", p_groups, p_page));
    PackedByteArray data;
    data.resize(4);
    data.encode_u32(0, p_groups);
    rd->buffer_update(page_rd.command, INDIRECT_INSTANCE_COUNT_OFFSET, 4, data);
}

void QuadPagePool::rt_copy_page_from(int p_page, RID p_src, uint32_t p_src_row) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page);
    ERR_FAIL_COND_MSG(!page_rd.texture.is_valid(), vformat("QuadPagePool: batch copy into a freed page %d", p_page));
    rd->texture_copy(p_src, page_rd.texture, Vector3(0, p_src_row, 0), Vector3(0, 0, 0),
            Vector3(TEX_WIDTH, get_page_rows(), 1), 0, 0, 0, 0);
}

void QuadPagePool::rt_kill_range(int p_page, uint32_t p_offset, uint32_t p_count) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page);
    if (!page_rd.texture.is_valid()) {
        return; // trimmed
    }
    ERR_FAIL_COND_MSG(p_offset % QUAD_GROUP != 0, "QuadPagePool: kill ranges must start on a group.");
    const uint32_t first_row = p_offset / QUAD_GROUP;
    const uint32_t rows = (p_count + QUAD_GROUP - 1) / QUAD_GROUP;
    ERR_FAIL_COND_MSG(first_row + rows > get_page_rows(), vformat("QuadPagePool: kill out of page: page %d rows %d+%d", p_page, first_row, rows));
    rd->texture_copy(rt_zero_texture, page_rd.texture, Vector3(0, 0, 0), Vector3(0, first_row, 0),
            Vector3(TEX_WIDTH, rows, 1), 0, 0, 0, 0);
}

void QuadPagePool::rt_copy_rows(int p_src_page, int p_dst_page, uint32_t p_src_row, uint32_t p_dst_row, uint32_t p_rows) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    ERR_FAIL_INDEX(p_src_page, (int)rt_pages.size());
    ERR_FAIL_INDEX(p_dst_page, (int)rt_pages.size());
    ERR_FAIL_COND(p_src_page == p_dst_page);
    ERR_FAIL_COND_MSG(!rt_pages[p_src_page].texture.is_valid() || !rt_pages[p_dst_page].texture.is_valid(), vformat("QuadPagePool: copy rows on a freed page %d -> %d", p_src_page, p_dst_page));
    ERR_FAIL_COND_MSG(p_src_row + p_rows > get_page_rows() || p_dst_row + p_rows > get_page_rows(), vformat("QuadPagePool: copy rows out of page %d:%d -> %d:%d x %d", p_src_page, p_src_row, p_dst_page, p_dst_row, p_rows));
    rd->texture_copy(rt_pages[p_src_page].texture, rt_pages[p_dst_page].texture, Vector3(0, p_src_row, 0),
            Vector3(0, p_dst_row, 0), Vector3(TEX_WIDTH, p_rows, 1), 0, 0, 0, 0);
}

Dictionary QuadPagePool::get_stats() const {
    Dictionary d;
    const int64_t alive = _alive_pages();
    d["pages_total"] = alive;
    d["pages_in_use"] = (int64_t)pages_in_use;
    d["page_capacity"] = (int64_t)page_capacity;
    d["slots_total"] = alive * page_capacity;
    d["growth_events"] = (int64_t)growth_events;
    d["pages_trimmed"] = (int64_t)pages_trimmed;
    const double page_mb = (double)get_page_rows() * PageFormat::ROW_BYTES / 1048576.0;
    d["vram_mb"] = alive * page_mb;
    d["pages_peak"] = pages_alive_peak;
    d["vram_peak_mb"] = pages_alive_peak * page_mb;
    return d;
}

void QuadPagePool::destroy() {
    RenderingServer *rs = RenderingServer::get_singleton();
    for (Page &page : pages) {
        if (!page.multimesh.is_valid()) {
            continue; // trimmed
        }
        rs->free_rid(page.instance);
        rs->free_rid(page.multimesh);
        rs->free_rid(page.material);
    }
    pages.clear();
    free_pages.clear();
    rs->free_rid(shader);
    rs->free_rid(base_mesh);
    // Render-thread resources are freed there, after everything queued before; then we die.
    rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_destroy));
}

void QuadPagePool::_rt_destroy() {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    RenderingServer *rs = RenderingServer::get_singleton();
    for (PageRT &page : rt_pages) {
        if (page.rs_texture.is_valid()) {
            rs->free_rid(page.rs_texture);
        }
        if (page.texture.is_valid()) {
            rd->free_rid(page.texture);
        }
    }
    rt_pages.clear();
    if (rt_zero_texture.is_valid()) {
        rd->free_rid(rt_zero_texture);
    }
    memdelete(this);
}
